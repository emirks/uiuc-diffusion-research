"""ctt_v2 S3 — precompute the per-frame stabilised depth stack for every content clip.

Advisor ruling 4: "precompute and cache depth stacks (float16, ~71 MB/clip) ONCE, before the
pilot, so pilot timings reflect render cost only."

Depth is a pure function of the clip, so the cache stays valid no matter how the operator grid
or the endpoint split later changes. Resumable: an existing .npy is never recomputed.

    sbatch experiments/exp_082_s3_stratum/job_depth.sbatch
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from diffusion.exp_utils import load_config  # noqa: E402

from engine3d import depth, videoio  # noqa: E402


def main() -> None:
    cfg = load_config(HERE / "config_s3.yaml")
    NF = cfg["inference"]["num_frames"]
    dev = cfg["runtime"]["device"]

    # Cache EVERY clip in the pool — training and reserved alike. The reserved clips are
    # eval-only for the training grid, but held-out-op eval renders will need their depth too,
    # and the marginal cost is ~4 s/clip. Re-running after an expansion computes only the new
    # clips (an existing .npy is never recomputed).
    pool = json.loads((REPO_ROOT / cfg["inputs"]["content_pool"]).read_text())
    clip_path = {e["clip_id"]: Path(e["mp4"]) for e in pool["training"] + pool["reserved"]}
    split = {"training": [e["clip_id"] for e in pool["training"]],
             "reserved_eval_only": [e["clip_id"] for e in pool["reserved"]]}
    ids = sorted(clip_path)

    cache = REPO_ROOT / cfg["inputs"]["depth_cache"]
    cache.mkdir(parents=True, exist_ok=True)

    todo = [c for c in ids if not (cache / f"{c}.npy").exists()]
    print(f"[depth] {len(ids)} clips ({len(split['training'])} train + "
          f"{len(split['reserved_eval_only'])} reserved) | {len(todo)} to compute, "
          f"{len(ids) - len(todo)} cached", flush=True)

    report, t0 = [], time.time()
    for n, cid in enumerate(todo, 1):
        t = time.time()
        frames = videoio.read_clip(clip_path[cid])[:NF]
        assert frames.shape[0] == NF, f"{cid}: {frames.shape[0]} frames < {NF}"
        d = depth.disparity_stack(frames, device=dev)
        fl = float(depth.flicker(d))
        np.save(cache / f"{cid}.npy", d.astype(np.float16))
        report.append({"clip_id": cid, "flicker": round(fl, 5),
                       "seconds": round(time.time() - t, 2)})
        if n % 20 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"[depth] {n}/{len(todo)}  {el/60:.1f} min  "
                  f"({el/n:.1f} s/clip, eta {(len(todo)-n)*el/n/60:.1f} min)", flush=True)

    if report:
        flicks = sorted(r["flicker"] for r in report)
        print(f"[depth] flicker: min {flicks[0]:.4f} median {flicks[len(flicks)//2]:.4f} "
              f"p90 {flicks[int(0.9*len(flicks))]:.4f} max {flicks[-1]:.4f}")
    out = HERE / "DEPTH_CACHE_REPORT.json"
    prev = json.loads(out.read_text())["clips"] if out.exists() else []
    merged = {r["clip_id"]: r for r in prev + report}
    out.write_text(json.dumps(
        {"created": "2026-07-25", "n_cached": len(list(cache.glob("*.npy"))),
         "cache_dir": str(cache), "clips": sorted(merged.values(), key=lambda r: r["clip_id"])},
        indent=1))
    missing = [c for c in ids if not (cache / f"{c}.npy").exists()]
    assert not missing, f"{len(missing)} clips still missing a depth stack: {missing[:5]}"
    print(f"[depth] DONE — {len(list(cache.glob('*.npy')))} stacks cached -> {cache}")


if __name__ == "__main__":
    main()
