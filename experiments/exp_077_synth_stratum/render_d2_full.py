"""exp_077 D2-FULL STAGE 2 (render) — mass-render the FINAL D2 dataset with PER-SLOT REJECTION.

    SHARD=k NSHARDS=n python render_d2_full.py

Sharded by TARGET PAIR (so "exactly 8 passing operators per target pair" is a per-shard invariant)
and fully resumable: one JSONL row per ACCEPTED tuple is flushed immediately, plus one row per
ATTEMPT, so a preemption loses at most the tuple in flight.

PER-SLOT REJECTION SAMPLING (the thing that makes "exactly 8 per pair" true rather than hoped for):
for each of the 8 slots of a target pair we redraw the operator (params / easing / flip / swap) up
to `attempts_per_shader` times, then swap to another allowed shader NEVER used by this pair (so the
>= 6-distinct-shaders constraint can only ever improve), until BOTH clips of the tuple pass the
frozen gate AND the degenerate-frame gate. Timing is redrawn per attempt alongside the params
(ratified 2026-07-24; see the note at the retry loop).

ACCEPT = frozen gate AND DFG (2026-07-24 ruling)
------------------------------------------------
`dfg.py` is an ADDITIVE accept-criterion evaluated only on clips that already passed the frozen
gate, so the accepted set is a strict SUBSET of the frozen gate's and tau = 0.2543 / the frozen
gate / the 40-shader blacklist are UNTOUCHED (no recalibration). Its thresholds are the ones
DFG_CALIB.json selected against a pre-committed acceptance bar.

`param_clamp.py` is ABANDONED PERMANENTLY (its caps were the wrong sign for the two worst
offenders and collapsed them onto the destructive constant). It is kept on disk as a record and
NEVER runs: `param_filter=None`, i.e. the byte-identical validated sampling path.

Reuses, unmodified: streams_real (real-stream render + gated operator sampling + timing),
d2_metrics (all four metrics + the frozen `verdict`), render_d2 (ClipCache / filmstrips).
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from diffusion.exp_utils import load_config  # noqa: E402

import d2_metrics  # noqa: E402
import dfg  # noqa: E402
import render_d2 as rd2  # noqa: E402  (ClipCache / save_strip / strip_indices / git_commit)
import streams_real as sr  # noqa: E402
from engine import videoio  # noqa: E402
from engine.glrunner import GLRunner  # noqa: E402

CONFIG_PATH = HERE / "config_d2full.yaml"
PLAN = HERE / "d2full_plan.json"
CALIB = HERE / "DFG_CALIB.json"
log = logging.getLogger("exp077.d2full")


class GrayCache:
    """96x72 grayscale of the FULL source clips (3.3 MB each) — M1/M2 recompute them per clip
    otherwise, which is ~2/3 of the metric cost."""

    def __init__(self) -> None:
        self.d: dict[str, np.ndarray] = {}

    def get(self, clip_id: str, arr: np.ndarray) -> np.ndarray:
        g = self.d.get(clip_id)
        if g is None:
            g = d2_metrics.to_small_gray(arr)
            self.d[clip_id] = g
        return g


def score(clip: np.ndarray, a_src, b_src, i0: int, j0: int, ga, gb) -> tuple[dict, np.ndarray]:
    """d2_metrics.score_clip with the source grayscales supplied (identical numbers).

    Also returns the clip's 96x72 grayscale so the DFG reuses it instead of recomputing it."""
    gf = d2_metrics.to_small_gray(clip)
    out = {"assert1": d2_metrics.assert1_pure_phase(clip, a_src, b_src, i0, j0),
           "assert2": d2_metrics.assert2_seam(clip, i0, j0)}
    out.update(d2_metrics.m1_m2(clip, a_src, b_src, i0, j0, gray=(gf, ga, gb)))
    return out, gf


def clip_row(stem: str, role: str, m: dict, v: dict, flag_thr: float) -> dict:
    """Per-clip metric record (metrics + frozen verdict + the NON-GATING m1_min flag)."""
    return {
        "stem": stem, "role": role,
        "assert1": {k: (round(x, 6) if isinstance(x, float) else x) for k, x in m["assert1"].items()},
        "assert2": {"seam_ratio": [round(x, 4) for x in m["assert2"]["seam_ratio"]],
                    "seam_max_ratio": round(m["assert2"]["seam_max_ratio"], 4),
                    "seam_mae": [round(x, 4) for x in m["assert2"]["seam_mae"]],
                    "bucket_delta": [round(x, 4) for x in m["assert2"]["bucket_delta"]]},
        "m1_p10": round(m["m1_p10"], 5), "m1_min": round(m["m1_min"], 5),
        "m1_mean": round(m["m1_mean"], 5), "m1_p10_frame": m["m1_p10_frame"],
        "m2_max_dq": round(m["m2_max_dq"], 5), "m2_max_dq_frame": m["m2_max_dq_frame"],
        "n_ramp": m["n_ramp"],
        "m1_min_flag": bool(m["m1_min"] < flag_thr),
        "verdict": v,
    }


def main() -> None:
    shard = int(os.environ.get("SHARD", "0"))
    nshards = int(os.environ.get("NSHARDS", "1"))
    cfg = load_config(CONFIG_PATH)
    inf, smp, d2, gt = cfg["inference"], cfg["sampling"], cfg["d2"], cfg["gate"]
    H, W, T, K = inf["height"], inf["width"], inf["num_frames"], inf["anchor_frames"]
    seed = cfg["runtime"]["seed"]
    tau, flag_thr = gt["tau"], gt["m1_min_flag_threshold"]

    # OUTSUB / PLANFILE let the first-chunk audit render an oversampled probe set into its OWN
    # directory tree with the identical code path — it can never contaminate the dataset.
    outsub = os.environ.get("OUTSUB", cfg["outputs"]["subdir"])
    planfile = HERE / os.environ.get("PLANFILE", PLAN.name)
    root = REPO_ROOT / cfg["outputs"]["dir"] / outsub
    vid_dir, strip_dir, meta_dir = root / "videos", root / "filmstrips", root / "meta"
    rej_dir, rej_strip = root / "rejects", root / "rejects_filmstrips"
    man_dir = root / "dataset_manifests"
    for d in (vid_dir, strip_dir, meta_dir, rej_dir, rej_strip, man_dir):
        d.mkdir(parents=True, exist_ok=True)
    tup_path = meta_dir / f"tuples_shard{shard:02d}.jsonl"
    att_path = meta_dir / f"attempts_shard{shard:02d}.jsonl"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout, force=True)
    log.info("D2-FULL shard=%d/%d host=%s root=%s plan=%s", shard, nshards, os.uname().nodename,
             root, planfile.name)

    plan = json.loads(planfile.read_text())
    keep_shaders, keep_easings = plan["keep_shaders"], plan["keep_easings"]

    # ---- shader bank: gate-1 validated, then RESTRICTED to keep_shaders (sampling-time blacklist)
    runner = GLRunner(W, H)
    log.info("GL: %s", runner.renderer_name())
    bank_all, bank_info = sr.d2_shader_bank(runner, Path(cfg["model"]["shader_bank"]),
                                            tol=smp["endpoint_tol"], holdout=plan["holdout_shaders"])
    missing = sorted(set(keep_shaders) - set(bank_all))
    assert not missing, f"keep_shaders absent from the gate-1 bank: {missing}"
    bank = {k: v for k, v in bank_all.items() if k in set(keep_shaders)}
    assert len(bank) == 72, len(bank)
    assert not (set(bank) & set(plan["blacklist"])), "blacklisted shader reached the sampling bank"
    assert not (set(bank) & set(plan["holdout_shaders"])), "holdout shader reached the sampling bank"
    assert set(keep_easings) == set(sr.d2_easings()) - set(plan["drop_easings"]), "easing mismatch"
    log.info("bank: gate1=%d -> keep=%d shaders | easings=%d %s",
             len(bank_all), len(bank), len(keep_easings), keep_easings)
    if shard == 0:
        info = {k: v for k, v in bank_info.items() if k != "gate1_report"}
        info.update({"n_sampling_bank": len(bank), "sampling_bank": sorted(bank),
                     "easings": keep_easings, "blacklist_applied_at": "sampling time",
                     "blacklist": plan["blacklist"]})
        (meta_dir / "bank_info.json").write_text(json.dumps(info, indent=2))

    # ---- parameter clamp: ABANDONED PERMANENTLY (2026-07-24 ruling). NEVER runs. ----
    # param_clamp.py stays on disk as a record. param_filter=None is the byte-identical path the
    # 448-tuple audit validated.
    assert not smp.get("param_clamp", False), "param clamping is abandoned; set param_clamp: false"
    pfilter = None
    log.info("param_clamp: ABANDONED (param_filter=None)")

    # ---- degenerate-frame gate: additive, AND-composed DOWNSTREAM of the frozen gate ----
    dcfg = cfg["dfg"]
    dfg_on = bool(dcfg.get("enabled", False))
    dfg_cfg = {k: dcfg.get(k) for k in ("theta_black", "theta_white", "theta_flat", "K",
                                       "theta_sat", "sat_flat_mult")}
    if CALIB.exists():
        cal = json.loads(CALIB.read_text())
        if dfg_on:
            assert cal.get("ships"), "DFG enabled but DFG_CALIB.json says the escape triggered"
            assert cal["chosen"] == dfg_cfg, f"config dfg {dfg_cfg} != calibrated {cal['chosen']}"
        else:
            assert not cal.get("ships"), "DFG_CALIB.json ships a detector but dfg.enabled is false"
    log.info("DFG: %s %s", "ACTIVE" if dfg_on else "DISABLED (escape)", json.dumps(dfg_cfg))
    fdfg = open(meta_dir / f"dfg_shard{shard:02d}.jsonl", "a", buffering=1)
    dfg_seen: Counter = Counter()      # per shader: clips that reached the DFG (gate-passing)
    dfg_rej: Counter = Counter()       # per shader: clips the DFG rejected
    dfg_test: Counter = Counter()      # which test fired on rejected clips
    n_dfg_reject = 0

    cache = rd2.ClipCache(REPO_ROOT / cfg["inputs"]["clips_dir"], d2["clip_cache"])
    gray = GrayCache()
    mine = [t for t in plan["targets"] if t["target_index"] % nshards == shard]
    log.info("shard %d: %d target pairs (%d slots)", shard, len(mine), len(mine) * 8)

    # ---- resume: accepted tuples already on disk ----
    done: dict[int, dict] = {}
    if tup_path.exists():
        for line in tup_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["tuple_id"]] = r
        log.info("resuming: %d accepted tuples already present", len(done))

    commit = rd2.git_commit()
    ftup = open(tup_path, "a", buffering=1)
    fatt = open(att_path, "a", buffering=1)
    idx_strip = rd2.strip_indices(T)
    t_start = time.time()
    n_render = n_accept = n_exhaust = n_timing_redraw = 0
    n_gate2_exhaust = 0
    n_rej_saved = sum(1 for _ in rej_dir.glob("*.mp4"))
    leg_fail: Counter = Counter()
    attempts_hist: Counter = Counter()

    def render_and_score(op, p, i0, j0, pair, role_srcs):
        nonlocal n_render
        a_src, b_src = role_srcs
        t_r = time.time()
        clip = sr.render_real(runner, bank, op, a_src, b_src, p)
        n_render += 1
        ga, gb = gray.get(pair["A"], a_src), gray.get(pair["B"], b_src)
        m, gf = score(clip, a_src, b_src, i0, j0, ga, gb)
        v = d2_metrics.verdict(m, tau, assert1_tol=gt["assert1_tol"],
                               seam_max=gt["seam_max"], m2_max=gt["m2_max_dq"])
        return clip, m, v, (gf, ga, gb), time.time() - t_r

    def dfg_check(clip, grays, i0, j0, op, stem, role, tid, n_att):
        """DFG on a clip that ALREADY passed the frozen gate. Returns (ok, payload)."""
        nonlocal n_dfg_reject
        gf, ga, gb = grays
        dfg_seen[op.shader] += 1
        # sat is computed even when the sat TEST is disabled: it costs ~1 resize/frame and makes
        # every rejection log self-describing (a flat matte's hue is otherwise invisible).
        ft = dfg.features(clip, i0, j0, gray=gf, gray_a=ga, gray_b=gb, need_sat=True)
        res = dfg.evaluate(ft, dfg_cfg)
        if not res["reject"]:
            return True, {"n_flag": res["n_flag"], "n_window": res["n_window"],
                          "by_test": res["by_test"]}
        n_dfg_reject += 1
        dfg_rej[op.shader] += 1
        for k, v in res["by_test"].items():
            if v:
                dfg_test[k] += v
        # every rejection logged with shader, params and the per-frame features of flagged frames
        fdfg.write(json.dumps({
            "tuple_id": tid, "attempt": n_att, "stem": stem, "role": role,
            "shader": op.shader, "easing": op.easing, "flip": op.flip, "swap": op.swap,
            "params": op.params, "dfg": dfg.compact(ft, res)}) + "\n")
        return False, dfg.compact(ft, res)

    for n, tgt in enumerate(mine):
        ti = tgt["target_index"]
        tp = tgt["target_pair"]
        a_t = cache.get(tp["A"], tp["A_mp4"])
        b_t = cache.get(tp["B"], tp["B_mp4"])
        assert a_t.shape == (T, H, W, 3) and b_t.shape == (T, H, W, 3), f"{a_t.shape}/{b_t.shape}"
        # shaders already committed for this target pair (planned + whatever resume accepted)
        pair_shaders = {done[s["tuple_id"]]["shader"] if s["tuple_id"] in done else s["shader"]
                        for s in tgt["slots"]}

        for slot in tgt["slots"]:
            tid = slot["tuple_id"]
            if tid in done:
                continue
            rp = slot["ref_pair"]
            a_r = cache.get(rp["A"], rp["A_mp4"])
            b_r = cache.get(rp["B"], rp["B_mp4"])
            rng = random.Random(f"{seed}-d2full-{tid}")
            stem_t, stem_r = f"d2f_{tid:04d}_tgt", f"d2f_{tid:04d}_ref"
            pair_frames = [(a_t[K - 1], b_t[T - K]), (a_r[K - 1], b_r[T - K])]

            accepted = None
            n_att = 0
            # MEASURED 2026-07-24 (n=228 slots, first 10 min of the mass render): slot difficulty is
            # carried by the TIMING draw, not by the params or the shader. Attempts-per-accepted-slot
            # clustered at {1: 165, ..., 26: 33, 51: 6, 76: 3} — 26 = "5 shaders x 5 param redraws all
            # failed, then ONE timing redraw passed on its first attempt". Only 4 slots were ever
            # rescued by a shader swap. Freezing timing per slot therefore cost 25 wasted renders per
            # hard slot and pushed realized overdraw to 7.35x, breaking the spec's 2.5x ceiling
            # (which assumes independent attempts). Timing is now redrawn per attempt, exactly like
            # the params. Consequence, reported in the audit: the delivered onset/release law is
            # U[0,1]-derived CONDITIONED ON GATE PASS, the same conditioning the params already carry.
            for redraw in range(1):
                # shader ladder: planned shader first, then shaders NEVER used by this target pair
                others = [s for s in keep_shaders if s not in pair_shaders]
                rng.shuffle(others)
                ladder = [slot["shader"]] + others[: d2["max_shader_swaps"]]
                for si, shader in enumerate(ladder):
                    for k in range(d2["attempts_per_shader"]):
                        n_att += 1
                        timing = sr.sample_timing(rng, T, K, frac=smp["timing_frac"])
                        i0, j0 = sr.phase_indices(T, timing["onset"], timing["release"])
                        op, gate_res, gtries = sr.sample_gated_operator(
                            runner, bank, rng, shader, pair_frames,
                            tol=smp["endpoint_tol_operator"], easings=keep_easings,
                            p_flip=smp["p_flip"], p_swap=smp["p_swap"],
                            max_tries=smp["max_gate_tries"], param_filter=pfilter)
                        if op is None:
                            n_gate2_exhaust += 1
                            fatt.write(json.dumps({
                                "tuple_id": tid, "attempt": n_att, "shader": shader,
                                "shader_swap": si, "timing_redraw": redraw,
                                "gate2_exhausted": True, "accepted": False}) + "\n")
                            continue
                        p = sr.progress_ramp(T, K, op.easing, timing["onset"], timing["release"])
                        clip_t, m_t, v_t, g_t, s_t = render_and_score(op, p, i0, j0, tp, (a_t, b_t))
                        row_t = clip_row(stem_t, "target", m_t, v_t, flag_thr)
                        rec = {"tuple_id": tid, "attempt": n_att, "shader": shader,
                               "shader_swap": si, "timing_redraw": redraw,
                               "easing": op.easing, "flip": op.flip, "swap": op.swap,
                               "gate2_tries": gtries, "target": row_t}

                        def save_reject(rstem, cl, why):
                            """Bounded sample of rejected clips for the viewer's audit section."""
                            nonlocal n_rej_saved
                            if n_rej_saved >= d2["reject_mp4_per_shard"]:
                                return
                            videoio.write_clip(rej_dir / f"{rstem}.mp4", cl, fps=inf["fps"])
                            rd2.save_strip(rej_strip / f"{rstem}.jpg", cl, idx_strip,
                                           d2["strip_frame_w"])
                            (rej_dir / f"{rstem}.json").write_text(json.dumps(dict(
                                rec, reject_stem=rstem, reject_reason=why, params=op.params,
                                timing={k2: timing[k2] for k2 in
                                        ("onset", "release", "duration", "u1", "u2")},
                                A=tp["A"], B=tp["B"]), indent=1))
                            n_rej_saved += 1

                        if not v_t["pass"]:
                            for leg in ("assert1", "assert2", "m1", "m2"):
                                if not v_t[leg]:
                                    leg_fail[leg] += 1
                            rec["accepted"] = False
                            fatt.write(json.dumps(rec) + "\n")
                            save_reject(f"d2f_{tid:04d}_a{n_att}_tgt", clip_t, "frozen_gate")
                            continue
                        # ---- DFG on the TARGET (downstream of the frozen gate) ----
                        if dfg_on:
                            ok_t, dp_t = dfg_check(clip_t, g_t, i0, j0, op, stem_t, "target",
                                                   tid, n_att)
                            rec["target"]["dfg"] = dp_t
                            if not ok_t:
                                leg_fail["dfg"] += 1
                                rec["accepted"] = False
                                rec["dfg_rejected"] = "target"
                                fatt.write(json.dumps(rec) + "\n")
                                save_reject(f"d2f_{tid:04d}_a{n_att}_tgt", clip_t, "dfg")
                                continue
                        clip_r, m_r, v_r, g_r, s_r = render_and_score(op, p, i0, j0, rp, (a_r, b_r))
                        row_r = clip_row(stem_r, "reference", m_r, v_r, flag_thr)
                        rec["reference"] = row_r
                        if not v_r["pass"]:
                            for leg in ("assert1", "assert2", "m1", "m2"):
                                if not v_r[leg]:
                                    leg_fail[leg] += 1
                            rec["accepted"] = False
                            fatt.write(json.dumps(rec) + "\n")
                            continue
                        # ---- DFG on the REFERENCE ----
                        if dfg_on:
                            ok_r, dp_r = dfg_check(clip_r, g_r, i0, j0, op, stem_r, "reference",
                                                   tid, n_att)
                            rec["reference"]["dfg"] = dp_r
                            if not ok_r:
                                leg_fail["dfg"] += 1
                                rec["accepted"] = False
                                rec["dfg_rejected"] = "reference"
                                fatt.write(json.dumps(rec) + "\n")
                                save_reject(f"d2f_{tid:04d}_a{n_att}_ref", clip_r, "dfg")
                                continue
                        rec["accepted"] = True
                        fatt.write(json.dumps(rec) + "\n")
                        accepted = {
                            "tuple_id": tid, "target_index": ti, "slot": slot["slot"],
                            "planned_shader": slot["shader"], "shader": op.shader,
                            "shader_swapped": si > 0, "easing": op.easing, "flip": op.flip,
                            "swap": op.swap, "params": op.params, "op_id": op.op_id,
                            "extension": sr.EXTENSION, "aux_kind": None,
                            "target_pair": {"A": tp["A"], "B": tp["B"]},
                            "ref_pair": {"A": rp["A"], "B": rp["B"]},
                            "target_stem": stem_t, "reference_stem": stem_r,
                            "timing": {k2: timing[k2] for k2 in
                                       ("onset", "release", "duration", "u1", "u2")},
                            "phase": {"i0": i0, "j0": j0},
                            "gate2": {"tol": smp["endpoint_tol_operator"], "tries": gtries,
                                      "mae": [[round(x, 4) for x in r] for r in gate_res]},
                            "layer_identity": {"a_stream_is_source": True,
                                               "b_stream_is_source": True},
                            "attempts": n_att, "timing_redraws": redraw,
                            "render_s": round(s_t + s_r, 2),
                            "clips": {"target": row_t, "reference": row_r},
                            "m1_min_flag": bool(row_t["m1_min_flag"] or row_r["m1_min_flag"]),
                            "param_clamp": False, "dfg_enabled": dfg_on, "dfg_config": dfg_cfg,
                            "engine_git_commit": commit, "tau": tau,
                        }
                        for stem, cl in ((stem_t, clip_t), (stem_r, clip_r)):
                            videoio.write_clip(vid_dir / f"{stem}.mp4", cl, fps=inf["fps"])
                            rd2.save_strip(strip_dir / f"{stem}.jpg", cl, idx_strip,
                                           d2["strip_frame_w"])
                        break
                    if accepted:
                        break
                if accepted:
                    break
            if accepted is None:
                n_exhaust += 1
                log.error("slot tid=%d target=%d EXHAUSTED after %d attempts", tid, ti, n_att)
                fatt.write(json.dumps({"tuple_id": tid, "slot_exhausted": True,
                                       "attempts": n_att, "accepted": False}) + "\n")
                continue
            ftup.write(json.dumps(accepted) + "\n")
            done[tid] = accepted
            pair_shaders.add(accepted["shader"])
            attempts_hist[n_att] += 1
            n_accept += 1

        od = n_render / max(2 * n_accept, 1)
        if (n + 1) % 2 == 0 or n == len(mine) - 1:
            el = time.time() - t_start
            log.info("shard %d: %d/%d targets | %d accepted | %d renders (%.2fx) | %d DFG-rej "
                     "| %.1f min (%.2fs/render) | cache h/m=%d/%d", shard, n + 1, len(mine),
                     n_accept, n_render, od, n_dfg_reject, el / 60,
                     el / max(n_render, 1), cache.hits, cache.misses)
        # ---- BUDGET GUARD: overdraw ceiling 2.5x -> STOP AND REPORT (spec). The 64-accepted
        # floor keeps the very first slots (where the ratio is dominated by 1-2 hard draws) from
        # tripping it; 64 accepted clips = 128 denominators, enough for the ratio to be real.
        if n_accept >= 64 and od > d2["overdraw_ceiling"]:
            ftup.close()
            fatt.close()
            fdfg.close()
            (meta_dir / f"OVERDRAW_BREACH_shard{shard:02d}.json").write_text(json.dumps({
                "shard": shard, "n_accept": n_accept, "n_render": n_render,
                "realized_overdraw": round(od, 4), "ceiling": d2["overdraw_ceiling"],
                "n_dfg_reject": n_dfg_reject, "leg_failures": dict(leg_fail),
                "dfg_reject_by_test": dict(dfg_test),
                "dfg_per_shader": {s: {"seen": dfg_seen[s], "rejected": dfg_rej[s]}
                                   for s in sorted(dfg_seen)}}, indent=2))
            sys.exit(f"[render] OVERDRAW CEILING BREACHED on shard {shard}: {od:.3f}x > "
                     f"{d2['overdraw_ceiling']}x after {n_accept} accepted tuples — STOPPING")

    ftup.close()
    fatt.close()
    fdfg.close()

    # ---- per-shard manifest for the encode stage + per-shard stats ----
    rows = [json.loads(line) for line in tup_path.read_text().splitlines() if line.strip()]
    stems = sorted({r[k] for r in rows for k in ("target_stem", "reference_stem")})
    (man_dir / f"clips_shard{shard:02d}.json").write_text(json.dumps(
        [{"stem": s, "video": f"{s}.mp4"} for s in stems], indent=1))
    stats = {
        "shard": shard, "nshards": nshards, "n_targets": len(mine),
        "n_tuples_accepted": len(rows), "n_clips": len(stems),
        "n_renders_this_run": n_render, "n_accepted_this_run": n_accept,
        "n_slots_exhausted": n_exhaust, "n_timing_redraws": n_timing_redraw,
        "n_gate2_exhausted_draws": n_gate2_exhaust,
        "attempts_hist_this_run": dict(sorted(attempts_hist.items())),
        "leg_failures_this_run": dict(leg_fail),
        "param_clamp_active": False,
        "dfg_enabled": dfg_on, "dfg_config": dfg_cfg,
        "dfg_clips_seen_this_run": sum(dfg_seen.values()),
        "dfg_clips_rejected_this_run": n_dfg_reject,
        "dfg_reject_rate_this_run": round(n_dfg_reject / max(sum(dfg_seen.values()), 1), 4),
        "dfg_reject_by_test_this_run": dict(dfg_test.most_common()),
        "dfg_per_shader_this_run": {s: {"seen": dfg_seen[s], "rejected": dfg_rej[s],
                                       "rate": round(dfg_rej[s] / dfg_seen[s], 4)}
                                    for s in sorted(dfg_seen)},
        "dfg_shaders_over_60pct_min20_this_run": sorted(
            s for s in dfg_seen if dfg_seen[s] >= 20 and dfg_rej[s] / dfg_seen[s] > 0.60),
        "overdraw_this_run": round(n_render / max(2 * n_accept, 1), 4),
        "wall_min": round((time.time() - t_start) / 60, 2),
        "cache_hits": cache.hits, "cache_misses": cache.misses,
        "complete": len(rows) == len(mine) * 8,
    }
    (meta_dir / f"stats_shard{shard:02d}.json").write_text(json.dumps(stats, indent=2))
    log.info("shard %d DONE: %s", shard, json.dumps({k: v for k, v in stats.items()
                                                     if k != "attempts_hist_this_run"}))
    if not stats["complete"]:
        sys.exit(f"[render] shard {shard} INCOMPLETE: {len(rows)} of {len(mine)*8} tuples")


if __name__ == "__main__":
    main()
