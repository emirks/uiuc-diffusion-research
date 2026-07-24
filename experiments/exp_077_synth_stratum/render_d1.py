"""exp_077 D1 STAGE 1 (render) — render the D1 tuples from d1_plan.json. JOB-ARRAY SHARDABLE.

For each tuple in this shard (tuple_id % NSHARDS == SHARD):
  1. Load the target pair (start9 from clip A, end9 from clip B) and the ref pair (cross-clip).
  2. REJECTION-SAMPLE a full operator with the plan-FIXED shader + aux family, varying uniforms /
     easing / flip / swap / aux_seed / timing, until the endpoint-identity gate passes on BOTH the
     target pair and the ref pair (MANDATORY per-operator gate). Timing is shared by ref & target
     (part of operator identity).
  3. Render the target clip and the ref clip (extension=flow, 121 frames) via the proven
     `render_tuples.render_timed`; write both mp4s + the full 0d per-tuple metadata.

Idempotent: a tuple whose two mp4s + metadata already exist is skipped (death/preempt safe).
Reuses exp_075's engine (symlink) and the smoke's render logic; adds only the D1 orchestration.
Deterministic: each tuple's operator draw is seeded by (global_seed, tuple_id), so the result is
independent of shard layout and re-renders bit-for-bit.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from diffusion.exp_utils import load_config  # noqa: E402

from engine import operators, shaders, streams, videoio  # noqa: E402
from engine.glrunner import GLRunner  # noqa: E402
from render_tuples import render_timed  # noqa: E402  (proven timing-aware render)

CONFIG_PATH = HERE / "config_d1.yaml"
PLAN = HERE / "d1_plan.json"
log = logging.getLogger("exp077.d1render")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def sample_fixed_operator(bank, rng, runner, shader, aux_kind, pair_frames, *, tol, easings,
                          p_flip, p_swap, extension, max_tries):
    """Rejection-sample an operator with FIXED shader + aux family, varying everything else,
    that passes the endpoint gate on EVERY pair in pair_frames. Returns (op, gate_res, tries)."""
    sh = bank[shader]
    for i in range(1, max_tries + 1):
        params = shaders.sample_params(sh, rng)
        easing = rng.choice(easings)
        flip = rng.choice(operators.FLIPS[1:]) if rng.random() < p_flip else "none"
        swap = rng.random() < p_swap
        aux_seed = rng.randrange(1 << 30) if aux_kind else 0
        op = operators.Operator(
            op_id=f"{shader}_{abs(hash((shader, str(sorted(params.items())), easing, flip, swap, aux_kind, aux_seed))) % (10**8):08d}",
            shader=shader, params=params, easing=easing, flip=flip, swap=swap,
            extension=extension, aux_kind=aux_kind, aux_seed=aux_seed)
        res = [operators.check_operator(runner, bank, op, fa, fb) for fa, fb in pair_frames]
        if all(max(d0, d1) <= tol for d0, d1 in res):
            return op, res, i
    raise RuntimeError(f"[{shader} aux={aux_kind}] no operator passed the gate in {max_tries} tries")


def main() -> None:
    shard = int(os.environ.get("SHARD", "0"))
    nshards = int(os.environ.get("NSHARDS", "1"))
    cfg = load_config(CONFIG_PATH)
    inf, smp = cfg["inference"], cfg["sampling"]
    H, W = inf["height"], inf["width"]
    T, K = inf["num_frames"], inf["anchor_frames"]
    seed = cfg["runtime"]["seed"]
    clips_root = REPO_ROOT / cfg["inputs"]["clips_dir"]
    run_dir = REPO_ROOT / cfg["outputs"]["dir"] / "d1"
    vid_dir = run_dir / "videos"
    meta_dir = run_dir / "meta"
    stats_dir = run_dir / "render_stats"
    man_dir = run_dir / "dataset_manifests"
    for d in (vid_dir, meta_dir, stats_dir, man_dir):
        d.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout, force=True)

    plan = json.loads(PLAN.read_text())
    tuples = [t for t in plan["tuples"] if t["tuple_id"] % nshards == shard]
    log.info("shard %d/%d: %d of %d tuples | run_dir=%s", shard, nshards, len(tuples),
             plan["n_tuples"], run_dir)

    bank_dir = Path(cfg["model"]["shader_bank"])
    bank = shaders.load_bank(bank_dir)
    runner = GLRunner(W, H)
    log.info("GL: %s | parsed %d shaders", runner.renderer_name(), len(bank))
    bank, _ = operators.validate_bank(runner, bank, tol=smp["endpoint_tol"])
    log.info("usable shaders after gate-1: %d", len(bank))
    commit = git_commit()
    easings = sorted(streams.EASINGS)
    t0, t1 = K - 1, T - K
    window_len = t1 - t0

    # cache only the 9-frame endpoint slices per clip (full clips are ~111MB each)
    ep_cache: dict = {}

    def endpoints(clip_id, mp4):
        if clip_id not in ep_cache:
            clip = videoio.read_clip(clips_root / mp4)
            ep_cache[clip_id] = (clip[:K].copy(), clip[-K:].copy())
        return ep_cache[clip_id]

    clip_rows = []
    stats = {"shard": shard, "n_tuples": len(tuples), "tries_total": 0, "rejections": 0,
             "per_shader_tries": {}, "skipped": 0, "rendered": 0}
    ep_mae = []

    for n, tup in enumerate(tuples):
        tid = tup["tuple_id"]
        stem_t = f"tup{tid:04d}_tgt"
        stem_r = f"tup{tid:04d}_ref"
        meta_path = meta_dir / f"tuple_{tid:04d}.json"
        # idempotent skip
        if meta_path.exists() and (vid_dir / f"{stem_t}.mp4").exists() and (vid_dir / f"{stem_r}.mp4").exists():
            m = json.loads(meta_path.read_text())
            clip_rows.append({"video": f"{stem_t}.mp4", "stem": stem_t, "role": "tgt", "tuple": tid})
            clip_rows.append({"video": f"{stem_r}.mp4", "stem": stem_r, "role": "ref", "tuple": tid})
            stats["skipped"] += 1
            continue

        tp, rp = tup["target_pair"], tup["ref_pair"]
        s9_t, e9_t = endpoints(tp["A"], tp["A_mp4"])[0], endpoints(tp["B"], tp["B_mp4"])[1]
        s9_r, e9_r = endpoints(rp["A"], rp["A_mp4"])[0], endpoints(rp["B"], rp["B_mp4"])[1]
        pair_frames = [(s9_t[-1], e9_t[0]), (s9_r[-1], e9_r[0])]

        rng = random.Random(f"{seed}-{tid}")
        op, gate_res, tries = sample_fixed_operator(
            bank, rng, runner, tup["shader"], tup["aux_kind"], pair_frames,
            tol=smp["endpoint_tol_operator"], easings=easings, p_flip=smp["p_flip"],
            p_swap=smp["p_swap"], extension=smp["extension"], max_tries=smp["max_gate_tries"])
        stats["tries_total"] += tries
        stats["rejections"] += tries - 1
        stats["per_shader_tries"].setdefault(tup["shader"], [0, 0])
        stats["per_shader_tries"][tup["shader"]][0] += tries
        stats["per_shader_tries"][tup["shader"]][1] += 1

        # continuous timing, shared by both clips of the tuple (part of operator identity)
        onset = t0 + rng.uniform(0.0, smp["onset_frac_max"]) * window_len
        remaining = t1 - onset
        duration = max(smp["dur_min_frames"], rng.uniform(smp["dur_frac_min"], 1.0) * remaining)

        clip_specs = []
        for role, stem, s9, e9, pair in (("target", stem_t, s9_t, e9_t, tp),
                                         ("reference", stem_r, s9_r, e9_r, rp)):
            t = time.time()
            clip = render_timed(runner, bank, op, s9, e9, T, onset, duration)
            d0, d1 = operators.endpoint_fidelity(clip, s9, e9)
            ep_mae.append(max(d0, d1))
            videoio.write_clip(vid_dir / f"{stem}.mp4", clip, fps=inf["fps"])
            clip_specs.append({
                "role": role, "stem": stem, "video": f"{stem}.mp4",
                "endpoint_A_id": pair["A"], "endpoint_B_id": pair["B"],
                "start_source": f"{pair['A']}[0:{K}]", "end_source": f"{pair['B']}[-{K}:]",
                "start_block": [0, K - 1], "end_block": [T - K, T - 1],
                "rendered_endpoint_mae": {"start": round(d0, 4), "end": round(d1, 4)},
                "render_s": round(time.time() - t, 2),
            })
            clip_rows.append({"video": f"{stem}.mp4", "stem": stem,
                              "role": "tgt" if role == "target" else "ref", "tuple": tid})

        aux_info = None
        if op.aux_kind:
            aux_info = {"family": op.aux_kind, "seed": op.aux_seed,
                        "note": "internal params (center/theta/freq/...) are seed-derived in engine.maps.make_map"}
        tup_meta = {
            "tuple_id": tid, "target_index": tup["target_index"],
            "operator": {"shader": op.shader, "params": op.params, "easing": op.easing,
                         "flip": op.flip, "swap": op.swap, "extension": op.extension,
                         "aux_kind": op.aux_kind, "aux_seed": op.aux_seed, "op_id": op.op_id},
            "timing": {"onset": onset, "duration": duration, "curve": op.easing,
                       "window": [t0, t1], "num_frames": T},
            "aux_map": aux_info,
            "target": clip_specs[0],      # the clip trained to be generated
            "reference": clip_specs[1],    # the in-context demo (same operator, disjoint content)
            "endpoint_identity_gate": {
                "tol": smp["endpoint_tol_operator"], "tries": tries,
                "target_pair_mae": [round(gate_res[0][0], 4), round(gate_res[0][1], 4)],
                "ref_pair_mae": [round(gate_res[1][0], 4), round(gate_res[1][1], 4)], "passed": True},
            "render_seed": f"{seed}-{tid}",
            "engine_git_commit": commit,
        }
        meta_path.write_text(json.dumps(tup_meta, indent=2))
        stats["rendered"] += 1
        if (n + 1) % 25 == 0 or n == len(tuples) - 1:
            log.info("shard %d: %d/%d done (rendered=%d skipped=%d) last=%s tries=%d",
                     shard, n + 1, len(tuples), stats["rendered"], stats["skipped"], op.shader, tries)

    stats["max_endpoint_mae"] = round(max(ep_mae), 5) if ep_mae else None
    # gate rejection rate for this shard's freshly-rendered tuples (rejected draws / total draws)
    stats["gate_rejection_rate"] = round(stats["rejections"] / stats["tries_total"], 5) \
        if stats["tries_total"] else None
    (stats_dir / f"stats_shard{shard}.json").write_text(json.dumps(stats, indent=2))
    (man_dir / f"clips_shard{shard}.json").write_text(json.dumps(clip_rows, indent=2))
    log.info("shard %d DONE: rendered=%d skipped=%d tries_total=%d rejections=%d max_ep_mae=%s",
             shard, stats["rendered"], stats["skipped"], stats["tries_total"],
             stats["rejections"], stats["max_endpoint_mae"])


if __name__ == "__main__":
    main()
