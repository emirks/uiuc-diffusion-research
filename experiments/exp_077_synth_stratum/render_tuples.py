"""exp_077 TASK 0d (steps 1-2) — render ~20 synthetic TUPLES + per-tuple metadata.

A *tuple* = (reference clip, target clip) that share the SAME operator — same shader,
same continuously-sampled uniforms, same easing, same spatial flip/direction, same aux
map, AND the same transition TIMING (onset / duration / curve inside the num_frames
window) — but are rendered on DIFFERENT endpoint pairs. This is the operator-fixed /
content-varying counterfactual axis the IC-LoRA generalist trains on.

Continuous sampling: shader uniforms are drawn continuously (engine.shaders.sample_params,
no grid); the transition timing (onset frame + duration) is drawn continuously inside the
[anchor-1, num_frames-anchor] window and is treated as part of the operator (ref & target
share it). Total clip length is `inference.num_frames` (LENGTH parameter, default 121).

Determinism: seed + the written metadata suffice to re-render a clip bit-for-bit. Verified
by reconstructing tuple 0's operator from its own metadata JSON and re-rendering (raw uint8
tensor equality — before the lossy mp4 encode).

Reuses exp_075's engine (imported through the `engine/` symlink; exp_075 is never touched).
The only new render logic here is the timing-aware progress ramp — the exp_075 ramp fixes
the transition to the whole [8, 112] window, so a timing-parameterised ramp cannot be reused
from it without editing that experiment.
"""

from __future__ import annotations

import json
import logging
import pathlib
import random
import subprocess
import sys
import time

import numpy as np
import PIL.Image
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from diffusion.exp_utils import TeeLogger, load_config, next_run_dir  # noqa: E402

from engine import maps, operators, shaders, streams, videoio  # noqa: E402
from engine.glrunner import GLRunner  # noqa: E402

CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"
log = logging.getLogger("exp077.render")


# --------------------------------------------------------------------------
# Timing-aware progress ramp (the one piece of render logic new to exp_077)
# --------------------------------------------------------------------------
def timed_progress_ramp(total: int, n_start: int, n_end: int, easing: str,
                        onset: float, duration: float) -> np.ndarray:
    """Per-frame progress pinned to 0 across start9 and 1 across end9, with the eased
    ramp confined to [onset, onset+duration] inside the window. Holds 0 before onset and
    1 after the ramp completes, so onset/duration are a genuine timing degree of freedom."""
    t0, t1 = n_start - 1, total - n_end          # 8 and 112 for 9 / 121 / 9
    onset = float(np.clip(onset, t0, t1 - 1))
    end = float(np.clip(onset + duration, onset + 1e-6, t1))
    t = np.arange(total, dtype=np.float64)
    u = np.clip((t - onset) / (end - onset), 0.0, 1.0)
    p = np.asarray(streams.EASINGS[easing](u), dtype=np.float64)
    p[t <= onset] = 0.0
    p[t >= end] = 1.0
    p[: t0 + 1] = 0.0                            # endpoint identity: start block == progress 0
    p[t1:] = 1.0                                 # endpoint identity: end block   == progress 1
    return np.clip(p, 0.0, 1.0)


def render_timed(runner: GLRunner, bank: dict, op: operators.Operator,
                 start9: np.ndarray, end9: np.ndarray, total: int,
                 onset: float, duration: float) -> np.ndarray:
    """engine.operators.render_sample, but with the timing-aware ramp above."""
    prog = runner.program(op.shader, bank[op.shader].source)
    if op.aux_kind:
        runner.set_aux_map(maps.make_map(op.aux_kind, runner.height, runner.width, op.aux_seed))
    aux_uniform = shaders.AUX_SAMPLER_SHADERS.get(op.shader)
    a_stream = streams.build_from_stream(start9, total, op.extension)
    b_stream = streams.build_to_stream(end9, total, op.extension)
    p = timed_progress_ramp(total, len(start9), len(end9), op.easing, onset, duration)
    out = np.empty_like(a_stream)
    for t in range(total):
        fa = operators._apply_flip(a_stream[t], op.flip)
        fb = operators._apply_flip(b_stream[t], op.flip)
        if op.swap:
            frame = runner.render(prog, fb, fa, 1.0 - p[t], op.params, aux_uniform)
        else:
            frame = runner.render(prog, fa, fb, p[t], op.params, aux_uniform)
        out[t] = operators._apply_flip(frame, op.flip)
    return out


def operator_from_meta(m: dict) -> operators.Operator:
    """Reconstruct an Operator from a metadata dict (params lists -> tuples for vec uniforms)."""
    params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in m["params"].items()}
    return operators.Operator(
        op_id=m["op_id"], shader=m["shader"], params=params, easing=m["easing"],
        flip=m["flip"], swap=m["swap"], extension=m["extension"],
        aux_kind=m["aux_kind"], aux_seed=m["aux_seed"])


def discover_pairs(cond_dir: pathlib.Path) -> dict:
    found = {}
    for p in sorted(cond_dir.glob("*_start9.mp4")):
        cid = p.name[: -len("_start9.mp4")]
        end = cond_dir / f"{cid}_end9.mp4"
        if end.exists():
            found[cid] = {"start9": p, "end9": end}
    return found


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def sample_shared_operator(bank, rng, runner, pair_frames, *, tol, easings, p_flip, p_swap,
                           extension, max_tries=80):
    """Rejection-sample ONE operator that passes the endpoint gate on EVERY pair in the tuple."""
    for _ in range(max_tries):
        op = operators.sample_operator(bank, rng, extensions=(extension,), easings=easings,
                                       p_flip=p_flip, p_swap=p_swap)
        res = [operators.check_operator(runner, bank, op, fa, fb) for fa, fb in pair_frames]
        if all(max(d0, d1) <= tol for d0, d1 in res):
            return op, res
    raise RuntimeError(f"no operator passed the endpoint gate on all pairs in {max_tries} tries")


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    out_dir = REPO_ROOT / cfg["outputs"]["dir"]
    run_id, run_dir = next_run_dir(out_dir)

    with TeeLogger(run_dir / "run.log"):
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
                            datefmt="%H:%M:%S", stream=sys.stdout, force=True)
        yaml.safe_dump(cfg, open(run_dir / "config_snapshot.yaml", "w"))

        inf, smp = cfg["inference"], cfg["sampling"]
        H, W = inf["height"], inf["width"]
        T, K = inf["num_frames"], inf["anchor_frames"]        # T = LENGTH parameter
        t0, t1 = K - 1, T - K
        window_len = t1 - t0
        seed = cfg["runtime"]["seed"]
        rng = random.Random(seed)
        commit = git_commit()
        log.info("num_frames=%d (length param) window=[%d,%d] len=%d seed=%d commit=%s",
                 T, t0, t1, window_len, seed, commit[:10])

        bank_dir = pathlib.Path(cfg["model"]["shader_bank"])
        bank = shaders.load_bank(bank_dir)
        runner = GLRunner(W, H)
        log.info("GL: %s | parsed %d shaders", runner.renderer_name(), len(bank))
        bank, _report = operators.validate_bank(runner, bank, tol=smp["endpoint_tol"])
        log.info("usable shaders after gate-1: %d", len(bank))

        cond_dir = REPO_ROOT / cfg["inputs"]["cond_dir"]
        clips = discover_pairs(cond_dir)
        ids = sorted(clips)
        log.info("endpoint pairs available: %d", len(ids))

        cache: dict = {}

        def load(path):
            if path not in cache:
                cache[path] = videoio.read_clip(path)
            return cache[path]

        def pair_endpoints(cid):
            s9 = load(clips[cid]["start9"])[:K]
            e9 = load(clips[cid]["end9"])[-K:]
            return s9, e9

        easings = sorted(streams.EASINGS)
        vid_dir = run_dir / "videos"
        strip_dir = run_dir / "filmstrips"
        meta_dir = run_dir / "meta"
        for d in (vid_dir, strip_dir, meta_dir):
            d.mkdir(parents=True, exist_ok=True)
        strip_idx = [i for i in (0, 8, 20, 35, 50, 60, 70, 85, 100, 112, 120) if i < T]

        tuples_index = []
        clip_rows = []                       # rows for the VAE-encode manifest
        first_tuple_arrays = {}              # kept for the determinism re-render check

        for idx in range(smp["n_tuples"]):
            ref_id, tgt_id = rng.sample(ids, 2)          # two DIFFERENT endpoint pairs
            s9_ref, e9_ref = pair_endpoints(ref_id)
            s9_tgt, e9_tgt = pair_endpoints(tgt_id)
            pair_frames = [(s9_ref[-1], e9_ref[0]), (s9_tgt[-1], e9_tgt[0])]

            op, gate_res = sample_shared_operator(
                bank, rng, runner, pair_frames, tol=smp["endpoint_tol_operator"],
                easings=easings, p_flip=smp["p_flip"], p_swap=smp["p_swap"],
                extension=smp["extension"])

            # continuous transition timing, shared by both clips of the tuple
            onset = t0 + rng.uniform(0.0, smp["onset_frac_max"]) * window_len
            remaining = t1 - onset
            duration = max(smp["dur_min_frames"],
                           rng.uniform(smp["dur_frac_min"], 1.0) * remaining)

            # aux-map internal params are fully determined by (aux_kind, aux_seed)
            aux_info = None
            if op.aux_kind:
                aux_info = {"family": op.aux_kind, "seed": op.aux_seed,
                            "note": "internal params (center/theta/freq/...) are seed-derived in engine.maps.make_map"}

            op_meta = {
                "shader": op.shader, "params": op.params, "easing": op.easing,
                "flip": op.flip, "swap": op.swap, "extension": op.extension,
                "aux_kind": op.aux_kind, "aux_seed": op.aux_seed, "op_id": op.op_id,
            }

            clip_specs = []
            for role, cid, s9, e9 in (("ref", ref_id, s9_ref, e9_ref),
                                      ("tgt", tgt_id, s9_tgt, e9_tgt)):
                stem = f"tup{idx:02d}_{role}"
                t = time.time()
                clip = render_timed(runner, bank, op, s9, e9, T, onset, duration)
                d0, d1 = operators.endpoint_fidelity(clip, s9, e9)
                videoio.write_clip(vid_dir / f"{stem}.mp4", clip, fps=inf["fps"])
                PIL.Image.fromarray(videoio.filmstrip(clip, strip_idx)).save(
                    strip_dir / f"{stem}.jpg", quality=88)
                if idx == 0:
                    first_tuple_arrays[stem] = clip.copy()
                clip_specs.append({
                    "role": role, "stem": stem, "video": f"{stem}.mp4",
                    "endpoint_source_id": cid,
                    "endpoint_frame_ranges": {
                        "start_block": [0, K - 1], "start_source": f"{cid}_start9.mp4[0:{K}]",
                        "end_block": [T - K, T - 1], "end_source": f"{cid}_end9.mp4[-{K}:]"},
                    "rendered_endpoint_mae": {"start": round(d0, 4), "end": round(d1, 4)},
                    "render_s": round(time.time() - t, 2),
                })
                clip_rows.append({"video": f"{stem}.mp4", "stem": stem,
                                  "role": role, "tuple": idx})
                log.info("tup%02d %-3s %-22s %-14s ep=(%.3f,%.3f)",
                         idx, role, cid, op.shader, d0, d1)

            gate = {"tol": smp["endpoint_tol_operator"],
                    "ref_pair_mae": [round(gate_res[0][0], 4), round(gate_res[0][1], 4)],
                    "tgt_pair_mae": [round(gate_res[1][0], 4), round(gate_res[1][1], 4)],
                    "passed": True}

            tup_meta = {
                "tuple_index": idx,
                "operator": op_meta,
                "timing": {"onset": onset, "duration": duration, "curve": op.easing,
                           "window": [t0, t1], "num_frames": T},
                "aux_map": aux_info,
                "reference": clip_specs[0],
                "target": clip_specs[1],
                "endpoint_identity_gate": gate,
                "render_seed": seed,
                "engine_git_commit": commit,
            }
            json.dump(tup_meta, open(meta_dir / f"tuple_{idx:02d}.json", "w"), indent=2)
            tuples_index.append({"tuple_index": idx, "shader": op.shader,
                                 "reference_stem": clip_specs[0]["stem"],
                                 "target_stem": clip_specs[1]["stem"],
                                 "reference_source": ref_id, "target_source": tgt_id})

        # -- VAE-encode manifest (data_root = videos dir; latents named "<stem>.pt") --
        json.dump(clip_rows, open(vid_dir / "clips_manifest.json", "w"), indent=2)
        json.dump({"n_tuples": smp["n_tuples"], "n_clips": len(clip_rows),
                   "num_frames": T, "tuples": tuples_index},
                  open(run_dir / "tuples.json", "w"), indent=2)

        # ---------------------------------------------------------------- determinism check
        log.info("determinism: re-rendering tuple 0 from its metadata JSON ...")
        m = json.load(open(meta_dir / "tuple_00.json"))
        op2 = operator_from_meta(m["operator"])
        onset2, duration2 = m["timing"]["onset"], m["timing"]["duration"]
        det = {"tuple": 0, "clips": {}}
        for role in ("reference", "target"):
            cid = m[role]["endpoint_source_id"]
            s9, e9 = pair_endpoints(cid)
            clip2 = render_timed(runner, bank, op2, s9, e9, T, onset2, duration2)
            stem = m[role]["stem"]
            equal = bool(np.array_equal(clip2, first_tuple_arrays[stem]))
            maxdiff = int(np.abs(clip2.astype(np.int16)
                                 - first_tuple_arrays[stem].astype(np.int16)).max())
            det["clips"][stem] = {"byte_exact": equal, "max_abs_diff": maxdiff}
            log.info("  %-12s byte_exact=%s max_abs_diff=%d", stem, equal, maxdiff)
        det["all_byte_exact"] = all(c["byte_exact"] for c in det["clips"].values())
        json.dump(det, open(run_dir / "determinism_check.json", "w"), indent=2)

        log.info("DONE %s | %d tuples, %d clips | determinism all_byte_exact=%s",
                 run_id, smp["n_tuples"], len(clip_rows), det["all_byte_exact"])
        print(f"[done] {run_id} -> {run_dir}")


if __name__ == "__main__":
    main()
