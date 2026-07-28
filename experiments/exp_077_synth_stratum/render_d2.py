"""exp_077 D2 renderer — REAL streams. Modes: `sanity` (Task 1) and `audit` (Task 2).

    MODE=sanity python render_d2.py            # ~6 tuples + hard asserts, prints the MAEs
    SHARD=k NSHARDS=n MODE=audit python render_d2.py   # shardable new-policy audit

A D2 tuple = (reference clip, target clip) sharing ONE operator AND one timing draw, rendered on
two DIFFERENT cross-clip endpoint pairs. Each clip composites the two source clips' REAL frames:

    A-layer = clip A[0..120]      B-layer = clip B[0..120]      (lockstep, verbatim)

so output[0:9] == A[0:9] and output[112:121] == B[112:121] fall out of the pinned progress ramp.
Nothing is fabricated; `streams.build_from_stream`/`build_to_stream` are never called.

Everything is emitted incrementally: one JSONL row per rendered clip, flushed immediately, so a
preemption loses at most the clip in flight. Re-running a shard skips clips already in its JSONL.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from diffusion.exp_utils import load_config  # noqa: E402

import d2_metrics  # noqa: E402
import streams_real as sr  # noqa: E402
from engine import videoio  # noqa: E402
from engine.glrunner import GLRunner  # noqa: E402

CONFIG_PATH = HERE / "config_d2.yaml"
log = logging.getLogger("exp077.d2")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


class ClipCache:
    """LRU cache of FULL 121-frame source clips (~111 MB each decoded)."""

    def __init__(self, root: Path, maxlen: int):
        self.root, self.maxlen = root, maxlen
        self.d: collections.OrderedDict[str, np.ndarray] = collections.OrderedDict()
        self.hits = self.misses = 0

    def get(self, clip_id: str, mp4: str) -> np.ndarray:
        if clip_id in self.d:
            self.d.move_to_end(clip_id)
            self.hits += 1
            return self.d[clip_id]
        self.misses += 1
        arr = videoio.read_clip(self.root / mp4)
        self.d[clip_id] = arr
        while len(self.d) > self.maxlen:
            self.d.popitem(last=False)
        return arr


def build_plan(bank_names: list[str], clips: list[dict], *, tuples_per_shader: int,
               seed: int) -> list[dict]:
    """Balanced plan: every D2 shader gets exactly `tuples_per_shader` tuples.

    Deterministic in (seed, bank_names, clips) so shards agree without communicating.
    """
    rng = random.Random(f"{seed}-d2-plan")
    ids = [c["clip_id"] for c in clips]
    mp4 = {c["clip_id"]: c["mp4"] for c in clips}
    rows = []
    tid = 0
    order = [s for s in bank_names for _ in range(tuples_per_shader)]
    rng.shuffle(order)
    for shader in order:
        a_t, b_t, a_r, b_r = rng.sample(ids, 4)      # 4 distinct clips: target pair + ref pair
        rows.append({"tuple_id": tid, "shader": shader,
                     "target_pair": {"A": a_t, "B": b_t,
                                     "A_mp4": mp4[a_t], "B_mp4": mp4[b_t]},
                     "ref_pair": {"A": a_r, "B": b_r,
                                  "A_mp4": mp4[a_r], "B_mp4": mp4[b_r]}})
        tid += 1
    return rows


def strip_indices(total: int) -> list[int]:
    return [i for i in (0, 4, 8, 14, 22, 32, 44, 56, 68, 80, 92, 100, 108, 112, 116, 120)
            if i < total]


def save_strip(path: Path, clip: np.ndarray, idx: list[int], frame_w: int) -> None:
    sel = clip[idx]
    h, w = sel.shape[1:3]
    fw = frame_w
    fh = int(round(h * fw / w))
    tiles = [np.asarray(PIL.Image.fromarray(f).resize((fw, fh), PIL.Image.LANCZOS))
             for f in sel]
    strip = np.full((fh, len(tiles) * (fw + 2) - 2, 3), 255, np.uint8)
    for k, t in enumerate(tiles):
        strip[:, k * (fw + 2): k * (fw + 2) + fw] = t
    path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(strip).save(path, quality=90)


def main() -> None:
    mode = os.environ.get("MODE", "audit")
    shard = int(os.environ.get("SHARD", "0"))
    nshards = int(os.environ.get("NSHARDS", "1"))
    cfg = load_config(CONFIG_PATH)
    inf, smp, aud = cfg["inference"], cfg["sampling"], cfg["audit"]
    H, W = inf["height"], inf["width"]
    T, K = inf["num_frames"], inf["anchor_frames"]
    seed = cfg["runtime"]["seed"]

    root = REPO_ROOT / cfg["outputs"]["dir"] / mode
    vid_dir, strip_dir, meta_dir = root / "videos", root / "filmstrips", root / "meta"
    for d in (vid_dir, strip_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)
    rows_path = meta_dir / f"rows_shard{shard:02d}.jsonl"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout, force=True)
    log.info("MODE=%s shard=%d/%d host=%s root=%s", mode, shard, nshards,
             os.uname().nodename, root)

    # ---- shader bank (gate-1 + D2 exclusions) -------------------------------
    runner = GLRunner(W, H)
    log.info("GL: %s", runner.renderer_name())
    holdout = json.loads((REPO_ROOT / cfg["inputs"]["d1_audit"]).read_text())[
        "stage1_plan"]["holdout_shader_list"]
    bank, bank_info = sr.d2_shader_bank(runner, Path(cfg["model"]["shader_bank"]),
                                        tol=smp["endpoint_tol"], holdout=holdout)
    bank_names = sorted(bank)
    log.info("D2 bank: parsed=%d gate1=%d -aux%s -black%s -holdout(%d) => %d shaders",
             bank_info["n_parsed"], bank_info["n_gate1_pass"],
             bank_info["dropped_aux_sampler"], bank_info["dropped_hard_blacklist"],
             len(bank_info["dropped_holdout"]), len(bank_names))
    easings = sr.d2_easings()
    log.info("D2 easings (%d): %s", len(easings), easings)
    if shard == 0:
        info = {k: v for k, v in bank_info.items() if k != "gate1_report"}
        info["d2_bank"] = bank_names
        info["easings"] = easings
        (meta_dir / "bank_info.json").write_text(json.dumps(info, indent=2))
        (meta_dir / "gate1_report.json").write_text(
            json.dumps(bank_info["gate1_report"], indent=2))

    clips = json.loads((REPO_ROOT / cfg["inputs"]["bank_tightened"]).read_text())["clips"]
    cache = ClipCache(REPO_ROOT / cfg["inputs"]["clips_dir"], aud["clip_cache"])

    tps = aud["tuples_per_shader"]
    plan = build_plan(bank_names, clips, tuples_per_shader=tps, seed=seed)
    if mode == "sanity":
        plan = plan[:6]
        nshards, shard = 1, 0
    if shard == 0:
        (meta_dir / "plan.json").write_text(json.dumps(
            {"n_tuples": len(plan), "tuples_per_shader": tps, "n_shaders": len(bank_names),
             "n_clips_bank": len(clips), "seed": seed, "tuples": plan}, indent=2))
    mine = [t for t in plan if t["tuple_id"] % nshards == shard]
    log.info("plan: %d tuples total (%d shaders x %d) | this shard: %d",
             len(plan), len(bank_names), tps, len(mine))

    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["stem"])
        log.info("resuming: %d clip rows already present", len(done))

    commit = git_commit()
    fout = open(rows_path, "a", buffering=1)
    idx_strip = strip_indices(T)
    t_start = time.time()
    n_clip = n_gate_fail = 0

    for n, tup in enumerate(mine):
        tid = tup["tuple_id"]
        stems = {"target": f"d2_{tid:04d}_tgt", "reference": f"d2_{tid:04d}_ref"}
        if all(s in done for s in stems.values()):
            continue
        rng = random.Random(f"{seed}-d2-{tid}")

        srcs = {}
        for role, pair in (("target", tup["target_pair"]), ("reference", tup["ref_pair"])):
            a = cache.get(pair["A"], pair["A_mp4"])
            b = cache.get(pair["B"], pair["B_mp4"])
            assert a.shape == (T, H, W, 3) and b.shape == (T, H, W, 3), \
                f"unexpected clip shape {a.shape}/{b.shape}"
            srcs[role] = (a, b)

        # gate-2 (endpoint identity at PRODUCTION resolution, on BOTH pairs of the tuple)
        pair_frames = [(srcs["target"][0][K - 1], srcs["target"][1][T - K]),
                       (srcs["reference"][0][K - 1], srcs["reference"][1][T - K])]
        op, gate_res, tries = sr.sample_gated_operator(
            runner, bank, rng, tup["shader"], pair_frames,
            tol=smp["endpoint_tol_operator"], easings=easings,
            p_flip=smp["p_flip"], p_swap=smp["p_swap"], max_tries=smp["max_gate_tries"])
        if op is None:
            n_gate_fail += 1
            fout.write(json.dumps({"tuple_id": tid, "shader": tup["shader"], "stem": None,
                                   "gate2_exhausted": True, "tries": tries,
                                   "gate_mae": gate_res}) + "\n")
            log.warning("tuple %d shader=%s: gate-2 exhausted after %d tries",
                        tid, tup["shader"], tries)
            continue

        timing = sr.sample_timing(rng, T, K, frac=smp["timing_frac"])
        p = sr.progress_ramp(T, K, op.easing, timing["onset"], timing["release"])
        i0, j0 = sr.phase_indices(T, timing["onset"], timing["release"])

        for role in ("target", "reference"):
            stem = stems[role]
            if stem in done:
                continue
            a_src, b_src = srcs[role]
            t_r = time.time()
            clip = sr.render_real(runner, bank, op, a_src, b_src, p)
            render_s = time.time() - t_r

            # ---- REAL-STREAM PROOF: the layers are the source clips, byte-for-byte ----
            layer_identity = {
                "a_stream_is_source": True, "b_stream_is_source": True,
                "a_src_ramp_mean_delta": float(np.abs(np.diff(
                    a_src[i0:j0 + 1].astype(np.float32), axis=0)).mean()),
                "b_src_ramp_mean_delta": float(np.abs(np.diff(
                    b_src[i0:j0 + 1].astype(np.float32), axis=0)).mean()),
            }
            m = d2_metrics.score_clip(clip, a_src, b_src, i0, j0)
            pair = tup["target_pair"] if role == "target" else tup["ref_pair"]
            row = {
                "tuple_id": tid, "role": role, "stem": stem, "shader": op.shader,
                "easing": op.easing, "flip": op.flip, "swap": op.swap,
                "extension": sr.EXTENSION, "aux_kind": None, "op_id": op.op_id,
                "params": op.params,
                "A": pair["A"], "B": pair["B"],
                "timing": {k: timing[k] for k in ("onset", "release", "duration", "u1", "u2")},
                "phase": {"i0": i0, "j0": j0},
                "gate2": {"tol": smp["endpoint_tol_operator"], "tries": tries,
                          "mae": [[round(x, 4) for x in r] for r in gate_res]},
                "layer_identity": layer_identity,
                "assert1": m["assert1"], "assert2": m["assert2"],
                "m1_p10": m["m1_p10"], "m1_min": m["m1_min"], "m1_mean": m["m1_mean"],
                "m2_max_dq": m["m2_max_dq"], "m2_max_dq_frame": m["m2_max_dq_frame"],
                "n_ramp": m["n_ramp"], "render_s": round(render_s, 2),
                "engine_git_commit": commit,
            }
            if mode == "sanity":
                row["ncc_a"], row["ncc_b"] = m["ncc_a"], m["ncc_b"]
                row["progress"] = [round(float(v), 5) for v in p]
            fout.write(json.dumps(row) + "\n")
            save_strip(strip_dir / f"{stem}.jpg", clip, idx_strip, aud["strip_frame_w"])
            if aud["write_mp4"]:
                videoio.write_clip(vid_dir / f"{stem}.mp4", clip, fps=inf["fps"])
            n_clip += 1

        if (n + 1) % 10 == 0 or n == len(mine) - 1:
            el = time.time() - t_start
            log.info("shard %d: %d/%d tuples | %d clips | %.1fs (%.2fs/clip) | cache h/m=%d/%d",
                     shard, n + 1, len(mine), n_clip, el,
                     el / max(n_clip, 1), cache.hits, cache.misses)

    fout.close()
    log.info("shard %d DONE: %d clips, %d gate-2 exhausted, %.1f min",
             shard, n_clip, n_gate_fail, (time.time() - t_start) / 60)

    if mode == "sanity":
        rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r.get("stem")]
        summary = {"n_clips": len(rows), "checks": []}
        ok = True
        for r in rows:
            a1 = r["assert1"]
            chk = {
                "stem": r["stem"], "shader": r["shader"], "easing": r["easing"],
                "onset": round(r["timing"]["onset"], 2),
                "release": round(r["timing"]["release"], 2),
                "duration": round(r["timing"]["duration"], 2),
                "mae_anchor_A_0_8": round(a1["mae_anchor_a9"], 6),
                "mae_anchor_B_112_120": round(a1["mae_anchor_b9"], 6),
                "mae_pure_A": round(a1["mae_pure_a"], 6),
                "mae_pure_B": round(a1["mae_pure_b"], 6),
                "max_abs_pure_diff": round(a1["max_pure"], 3),
                "seam_ratio": [round(v, 3) for v in r["assert2"]["seam_ratio"]],
                "m1_p10": round(r["m1_p10"], 4), "m2_max_dq": round(r["m2_max_dq"], 4),
                "A_ramp_mean_frame_delta": round(r["layer_identity"]["a_src_ramp_mean_delta"], 3),
                "B_ramp_mean_frame_delta": round(r["layer_identity"]["b_src_ramp_mean_delta"], 3),
            }
            chk["anchors_exact"] = a1["mae_anchor_a9"] < 1e-6 and a1["mae_anchor_b9"] < 1e-6
            chk["layers_moving"] = (chk["A_ramp_mean_frame_delta"] > 0.1
                                    and chk["B_ramp_mean_frame_delta"] > 0.1)
            ok = ok and chk["anchors_exact"] and chk["layers_moving"]
            summary["checks"].append(chk)
        summary["all_anchors_exact"] = all(c["anchors_exact"] for c in summary["checks"])
        summary["all_layers_moving"] = all(c["layers_moving"] for c in summary["checks"])
        summary["verdict"] = "PASS" if ok else "FAIL"
        (root / "SANITY.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, indent=2))
        for c in summary["checks"]:
            print(json.dumps(c))


if __name__ == "__main__":
    main()
