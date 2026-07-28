"""ctt_v2 S3 PILOT — 63 clips on BANK contents. Advisor ruling 4: a HARD GATE.

Why this exists: exp_080's only validation (run_0001, 31 clips) used CORPUS clips as contents,
which the owner has since ruled illegal — the transition class effect unfolds inside corpus
clips, so manner leaks into what is supposed to be neutral "content". Depth quality and join
behaviour on bank material (openvid / vcbench / davis at 480x640) is therefore genuinely
untested, and the engine has additionally been copied to a new branch. Nothing about the full
1,800-clip render may start until this passes.

PRE-COMMITTED BARS (frozen in config_s3.yaml BEFORE this ran; no bar may be relaxed after
seeing results — that is the whole point of pre-commitment):

    n = 63, stratified 9 per family across ALL 7 camera families, forcing the heavy-optics
    tail (world-space dissolves, rack focus, fog, dolly-zoom, depth wipe) because the slow
    arms are also the visually risky ones.

    pure-phase        byte-exact on EVERY clip, MAX condition (never a mean)
    join ratio        <= 2.0 at BOTH joins, gating
    join distribution median <= 1.3 and p90 <= 1.8   (run_0001 baseline: 0.94 / 1.15 / 1.86)
    parallax          median >= 2.0 px               (run_0001 baseline: 3.31)
    blind visual      <= 3 BAD of 63, and no single family contributing >= 3 of its 9
    timing            project full-render wall time from per-family timings

The visual audit is a separate operator step; this script produces the numbers, the filmstrips
and a shuffled blind-audit sheet, then writes PILOT_RESULT.json with every bar marked
PASS/FAIL against the frozen values.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from diffusion.exp_utils import TeeLogger, load_config  # noqa: E402

from engine3d import cameras, depth, metrics, ops3d, videoio  # noqa: E402
from engine3d.render3d import MeshRenderer  # noqa: E402

log = logging.getLogger("s3pilot")

# the five heavy-optics arms the advisor required in the stratification
FORCED = [
    ("dissolve_fbm",  dict(dissolve="fbm", dissolve_freq=1.6)),
    ("rackfocus",     dict(focus=12.0)),
    ("fog",           dict(fog=1.6)),
    ("dollyzoom",     dict(dolly_zoom=1.0)),          # dolly/spiral only — see fallback below
    ("depthwipe",     dict(blend="depth_wipe", wipe_band=0.2)),
]
DZ_FALLBACK = ("handheld_mblur", dict(handheld=0.7, motion_blur=3))


def op_id_of(op: ops3d.Operator3D, onset: int, release: int) -> str:
    d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in op.__dict__.items()}
    d.pop("seed", None)                     # seed drives handheld noise only; keep it out of ID
    key = json.dumps(d, sort_keys=True) + f"|{onset}|{release}"
    return f"{op.path}_{hashlib.sha1(key.encode()).hexdigest()[:8]}"


def main() -> None:
    cfg = load_config(HERE / "config_s3.yaml")
    inf, s3, tim, gate = cfg["inference"], cfg["s3"], cfg["timing"], cfg["gate"]
    NF, dev = inf["num_frames"], cfg["runtime"]["device"]
    rng = random.Random(cfg["runtime"]["seed"])

    out_dir = REPO_ROOT / cfg["outputs"]["dir"] / "pilot"
    vid_dir, strip_dir = out_dir / "videos", out_dir / "filmstrips"
    for d in (vid_dir, strip_dir):
        d.mkdir(parents=True, exist_ok=True)

    with TeeLogger(out_dir / "pilot.log"):
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)-7s %(message)s",
                            datefmt="%H:%M:%S", stream=sys.stdout, force=True)

        pool = json.loads((REPO_ROOT / cfg["inputs"]["content_pool"]).read_text())
        train_ids = [e["clip_id"] for e in pool["training"]]
        clip_path = {e["clip_id"]: Path(e["mp4"])      # absolute; spans both banks
                     for e in pool["training"] + pool["reserved"]}
        split = {"n_reserved": pool["n_reserved"]}
        cache_dir = REPO_ROOT / cfg["inputs"]["depth_cache"]
        log.info("pilot contents: %d TRAINING bank endpoints (reserved %d held out)",
                 len(train_ids), split["n_reserved"])

        renderer = MeshRenderer(inf["width"], inf["height"], step=inf["mesh_step"])
        log.info("GL context: %s", renderer.renderer_name())

        frames_cache: dict[str, np.ndarray] = {}
        depth_cache: dict[str, np.ndarray] = {}

        def clip_frames(cid: str) -> np.ndarray:
            if cid not in frames_cache:
                f = videoio.read_clip(clip_path[cid])[:NF]
                assert f.shape[0] == NF, f"{cid}: {f.shape[0]} frames < {NF}"
                if len(frames_cache) > 24:
                    frames_cache.pop(next(iter(frames_cache)))
                frames_cache[cid] = f
            return frames_cache[cid]

        def clip_depth(cid: str) -> np.ndarray:
            if cid not in depth_cache:
                npy = cache_dir / f"{cid}.npy"
                if npy.exists():
                    d = np.load(npy).astype(np.float32)
                else:                       # cache miss: compute and persist (should not happen)
                    log.warning("depth cache MISS for %s — computing", cid)
                    d = depth.disparity_stack(clip_frames(cid), device=dev)
                    np.save(npy, d.astype(np.float16))
                if len(depth_cache) > 24:
                    depth_cache.pop(next(iter(depth_cache)))
                depth_cache[cid] = d
            return depth_cache[cid]

        def draw_timing() -> tuple[int, int]:
            w0, w1 = tim["window"]
            span = w1 - w0
            return (int(round(w0 + rng.random() * tim["jitter_frac"] * span)),
                    int(round(w1 - rng.random() * tim["jitter_frac"] * span)))

        def base_op(family: str, **over) -> ops3d.Operator3D:
            op = ops3d.sample_operator(rng)
            op.path = family
            op.easing = rng.choice(ops3d.PATH_EASINGS)
            op.amplitude = rng.uniform(0.55, 1.6) * s3["amplitude_scale"]
            # start from a clean baseline so a "forced" arm isolates ONE optical axis
            op.handheld = op.fog = op.focus = op.dolly_zoom = 0.0
            op.motion_blur, op.dissolve, op.blend = 1, "none", "crossfade"
            for k, v in over.items():
                setattr(op, k, v)
            return op

        # ---- the 63-clip plan: 9 per family x 7 families ---------------------------------
        plan = []
        for family in sorted(cameras.PATHS):
            for slot in range(s3["pilot_per_family"]):
                if slot < 4:
                    tag, over = "random", {}
                else:
                    tag, over = FORCED[slot - 4]
                    if tag == "dollyzoom" and family not in ("dolly", "spiral"):
                        tag, over = DZ_FALLBACK          # dolly-zoom is undefined off-axis
                a, b = rng.sample(train_ids, 2)
                onset, release = draw_timing()
                plan.append({"family": family, "slot": slot, "tag": tag, "over": over,
                             "A": a, "B": b, "onset": onset, "release": release})
        assert len(plan) == s3["pilot_n"], f"pilot plan is {len(plan)}, expected {s3['pilot_n']}"

        rows, t0 = [], time.time()
        for n, e in enumerate(plan, 1):
            op = base_op(e["family"], **e["over"])
            onset, release = e["onset"], e["release"]
            A, B = clip_frames(e["A"]), clip_frames(e["B"])
            da, db = clip_depth(e["A"]), clip_depth(e["B"])
            t = time.time()
            clip = ops3d.render_transition_stream(renderer, op, A, B, da, db, onset, release)
            render_s = time.time() - t

            # ---- pure-phase identity: MAX condition, asserted per clip (dsx failure #2) ----
            max_pure = float(max(np.abs(clip[:onset + 1].astype(np.int16)
                                        - A[:onset + 1].astype(np.int16)).max(),
                                 np.abs(clip[release:].astype(np.int16)
                                        - B[release:].astype(np.int16)).max()))
            byte_exact = bool(np.array_equal(clip[:onset + 1], A[:onset + 1])
                              and np.array_equal(clip[release:], B[release:]))
            r_in = ops3d.join_ratio(clip, onset + 1)
            r_out = ops3d.join_ratio(clip, release)
            za0 = depth.to_view_depth(da[onset + 1], op.depth_near, op.depth_far, op.depth_gamma)
            pi = metrics.parallax_index(clip[onset + 1: onset + 7], za0)

            oid = op_id_of(op, onset, release)
            stem = f"s3p_{n:03d}_{e['family']}_{e['tag']}_{oid[-8:]}"
            videoio.write_clip(vid_dir / f"{stem}.mp4", clip, fps=inf["fps"])
            ramp = np.linspace(onset, release, 7).astype(int).tolist()
            PIL.Image.fromarray(videoio.filmstrip(clip, [0, 8] + ramp + [112, 120])).save(
                strip_dir / f"{stem}.jpg", quality=88)

            rows.append({
                "stem": stem, "op_id": oid, "family": e["family"], "tag": e["tag"],
                "A": e["A"], "B": e["B"], "onset": onset, "release": release,
                "byte_exact": byte_exact, "max_pure": max_pure,
                "join_in": round(r_in, 3), "join_out": round(r_out, 3),
                "join_max": round(max(r_in, r_out), 3),
                "parallax": pi, "render_s": round(render_s, 2),
                "describe": op.describe(),
                "params": {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in op.__dict__.items()},
            })
            log.info("%-58s join=(%.2f,%.2f) PI=%.2f pure=%.1f %.1fs",
                     stem[:58], r_in, r_out, pi["pi"], max_pure, render_s)

        # ---- score against the PRE-COMMITTED bars ----------------------------------------
        joins = [r["join_in"] for r in rows] + [r["join_out"] for r in rows]
        pis = [r["parallax"]["pi"] for r in rows]
        rhos = [r["parallax"]["rho"] for r in rows]
        secs = [r["render_s"] for r in rows]
        by_family: dict[str, dict] = {}
        for fam in sorted({r["family"] for r in rows}):
            fr = [r for r in rows if r["family"] == fam]
            by_family[fam] = {
                "n": len(fr),
                "join_median": round(float(np.median([r["join_max"] for r in fr])), 3),
                "join_max": round(max(r["join_max"] for r in fr), 3),
                "join_over_2": sum(1 for r in fr if r["join_max"] > gate["join_max"]),
                "parallax_median": round(float(np.median([r["parallax"]["pi"] for r in fr])), 2),
                "rho_median": round(float(np.median([r["parallax"]["rho"] for r in fr])), 3),
                "sec_mean": round(float(np.mean([r["render_s"] for r in fr])), 1),
                "sec_max": round(max(r["render_s"] for r in fr), 1),
            }

        bars = {
            "pure_phase_byte_exact_all": {
                "value": sum(r["byte_exact"] for r in rows), "bar": len(rows), "op": "==",
                "pass": all(r["byte_exact"] for r in rows)},
            "max_pure_over_all_clips": {
                "value": max(r["max_pure"] for r in rows), "bar": 0.0, "op": "==",
                "pass": max(r["max_pure"] for r in rows) == 0.0},
            "join_all_under_2": {
                "value": sum(1 for j in joins if j > gate["join_max"]), "bar": 0, "op": "==",
                "pass": all(j <= gate["join_max"] for j in joins)},
            "join_median": {
                "value": round(float(np.median(joins)), 3), "bar": gate["join_median_bar"],
                "op": "<=", "pass": float(np.median(joins)) <= gate["join_median_bar"]},
            "join_p90": {
                "value": round(float(np.percentile(joins, 90)), 3), "bar": gate["join_p90_bar"],
                "op": "<=", "pass": float(np.percentile(joins, 90)) <= gate["join_p90_bar"]},
            "parallax_median": {
                "value": round(float(np.median(pis)), 3), "bar": gate["parallax_median_bar"],
                "op": ">=", "pass": float(np.median(pis)) >= gate["parallax_median_bar"]},
        }
        auto_pass = all(b["pass"] for b in bars.values())
        full_h = sum(secs) / len(secs) * (s3["n_ops"] * s3["contents_per_op"]) / 3600.0

        result = {
            "created": "2026-07-25", "stratum": "S3", "phase": "PILOT",
            "authority": "fable-advisor ruling 4 — MANDATORY gate before the 1,800-clip render",
            "engine_source": "exp_080_depth3d_realstream_121 @ cecf231/fc58617/e47c7f1, "
                             "engine3d/ copied byte-identical",
            "n": len(rows),
            "contents": "207 TRAINING bank endpoints (the 20 reserved are eval-only)",
            "bars_precommitted": bars,
            "automated_verdict": "PASS" if auto_pass else "FAIL",
            "visual_audit": {"status": "PENDING — operator blind review of all 63",
                             "max_bad_allowed": gate["pilot_max_bad"],
                             "max_bad_per_family": gate["pilot_max_bad_per_family"]},
            "join_distribution": {
                "median": round(float(np.median(joins)), 3),
                "p90": round(float(np.percentile(joins, 90)), 3),
                "max": round(float(np.max(joins)), 3)},
            "parallax": {"median_pi": round(float(np.median(pis)), 3),
                         "median_rho": round(float(np.median(rhos)), 3),
                         "min_pi": round(float(np.min(pis)), 3)},
            "timing": {"sec_mean": round(float(np.mean(secs)), 1),
                       "sec_median": round(float(np.median(secs)), 1),
                       "sec_max": round(float(np.max(secs)), 1),
                       "projected_full_render_gpu_hours": round(full_h, 1),
                       "note": "projection = mean s/clip x 1,800 clips; excludes gate overdraw"},
            "by_family": by_family,
            "clips": rows,
        }
        (HERE / "PILOT_RESULT.json").write_text(json.dumps(result, indent=1))

        # blind audit sheet: shuffled, identity withheld until the operator has scored it
        order = list(range(len(rows)))
        random.Random(4242).shuffle(order)
        (out_dir / "BLIND_ORDER.json").write_text(json.dumps(
            {"note": "audit in this order; `stem` is revealed only after scoring",
             "order": [{"blind_id": i, "stem": rows[j]["stem"]} for i, j in enumerate(order)]},
            indent=1))

        log.info("PILOT automated verdict: %s", result["automated_verdict"])
        for k, b in bars.items():
            log.info("  %-28s %-10s %s %-8s  %s", k, b["value"], b["op"], b["bar"],
                     "PASS" if b["pass"] else "FAIL")
        log.info("join median %.2f p90 %.2f max %.2f | parallax median %.2f | "
                 "%.1f s/clip mean -> %.1f GPU-h projected for 1,800",
                 result["join_distribution"]["median"], result["join_distribution"]["p90"],
                 result["join_distribution"]["max"], result["parallax"]["median_pi"],
                 result["timing"]["sec_mean"], full_h)
        print(f"[done] pilot -> {out_dir}")


if __name__ == "__main__":
    main()
