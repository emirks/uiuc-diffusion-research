#!/usr/bin/env python
"""CTT v2 / S1 -- measure tau_endpoint, the endpoint-identity tolerance.

A5 RULING 3(ii): the ONLY per-clip hard rejects allowed in S1 selection are mechanical
(decode corruption, frozen/black frames, endpoint identity on the prefix window), and the
endpoint-identity threshold is

    tau = p95 of prefix rel-L2 measured on the specialists' EXISTING inline-validation outputs

i.e. we ask "how faithfully does an LTX-2 specialist actually reproduce its 9-frame prefix
anchor in the generated video?" and set the reject bar at the 95th percentile of that
already-observed distribution.  Nothing here touches DINOv2 or any harness substrate --
this is a raw-pixel measurement (Ruling 3(i): no harness-substrate gate in data selection).

Measurement definition (pinned here, reported in S1_GRID.json):
  * generated video  -> frames 0..8 (PX_PREFIX), RGB uint8 -> float [0,1]
  * conditioning mp4 -> its 9 frames, same decode path
  * rel_l2 = ||gen - cond||_2 / ||cond||_2      (encode_conditioning.rel_l2's formula)

Pixel space, not latent space: Ruling 3 puts this measurement on Day 0 on CPU, and a latent
rel-L2 would need the VAE (GPU) -- and would also be a *learned* substrate, which is exactly
what the stop-list bans from selection.  Both the primary (all checkpoints) and the
cross-check (final checkpoint only) percentiles are reported.

Usage
-----
  PY=/projects/illinois/eng/cs/jrehg/users/emirkisa/envs/diffusion/bin/python
  $PY scripts/ctt_v2/s1/measure_tau_endpoint.py --out outputs/ctt_v2/s1/tau_endpoint.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_ROOT = REPO_ROOT / "outputs/training/ladder2"
CONDS = REPO_ROOT / "eval_ladder/conds"

#: the prefix window, imported in spirit from encode_conditioning.PX_PREFIX (9 pixel frames)
PX_PREFIX = 9
#: std clip geometry (portrait corpus)
H, W = 640, 480


def decode(path: Path, n: int) -> np.ndarray:
    """First `n` frames of an mp4 as float32 [n,H,W,3] in [0,1]. Raises on short/corrupt."""
    # -threads 1: ffmpeg's default multi-threaded decode trips the login node's per-user
    # thread ceiling as soon as a few decodes run concurrently (pthread_create() failed --
    # the same lesson the caption_strips extractor banked). The worker pool is the only
    # concurrency knob.
    cmd = ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
           "-vf", f"select=lt(n\\,{n})", "-vsync", "0",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg rc={proc.returncode}: {proc.stderr.decode()[:200]}")
    raw = proc.stdout
    arr = np.frombuffer(raw, np.uint8)
    if arr.size % (H * W * 3):
        raise ValueError(f"{path}: unexpected frame geometry ({arr.size} bytes)")
    arr = arr.reshape(-1, H, W, 3)
    if len(arr) < n:
        raise ValueError(f"{path}: decoded {len(arr)} frames, wanted {n}")
    return arr[:n].astype(np.float32) / 255.0


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """encode_conditioning.rel_l2, in numpy: ||a-b|| / max(||b||, eps)."""
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-12))


def cond_path_of(sample: dict) -> Path | None:
    """The prefix conditioning mp4 of one validation sample, remapped to this tree.

    The training configs were written when ladder2 lived at experiments/ladder2/; the tree
    was promoted to eval_ladder/ afterwards, so only the basename is trustworthy.
    """
    for cond in sample.get("conditions") or []:
        if cond.get("type") == "prefix":
            return CONDS / Path(cond["video"]).name
    return None


def jobs_for(spec_dir: Path) -> list[dict]:
    cfg = yaml.unsafe_load((spec_dir / "training_config.yaml").read_text())
    samples = cfg["validation"]["samples"]
    out = []
    for idx, sample in enumerate(samples, start=1):        # files are step_NNNNNN_<1-based>.mp4
        cond = cond_path_of(sample)
        if cond is None:
            continue                                        # sample 3 is prompt-only: no anchor
        for gen in sorted(spec_dir.glob(f"samples/step_*_{idx}.mp4")):
            out.append({"arm": spec_dir.name, "sample": idx, "cond": cond, "gen": gen,
                        "step": int(gen.stem.split("_")[1])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/ctt_v2/s1/tau_endpoint.json")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    specs = sorted(TRAIN_ROOT.glob("spec_*"))
    assert len(specs) == 11, f"expected 11 specialists, found {len(specs)}"
    jobs = [j for d in specs for j in jobs_for(d)]
    print(f"[tau] {len(specs)} specialists, {len(jobs)} prefix-conditioned inline-validation clips")

    def measure(j: dict) -> dict:
        try:
            g = decode(j["gen"], PX_PREFIX)
            c = decode(j["cond"], PX_PREFIX)
            return {**j, "cond": str(j["cond"]), "gen": str(j["gen"].relative_to(REPO_ROOT)),
                    "rel_l2": rel_l2(g, c), "error": None}
        except Exception as exc:                            # noqa: BLE001 - recorded, not raised
            return {**j, "cond": str(j["cond"]), "gen": str(j["gen"].relative_to(REPO_ROOT)),
                    "rel_l2": None, "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(measure, jobs))

    bad = [r for r in rows if r["error"]]
    ok = [r for r in rows if r["rel_l2"] is not None]
    assert ok, "no measurements succeeded"

    def stats(vals: list[float]) -> dict:
        a = np.asarray(vals)
        return {"n": int(a.size), "min": float(a.min()), "p50": float(np.percentile(a, 50)),
                "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99)), "max": float(a.max()),
                "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0}

    allv = [r["rel_l2"] for r in ok]
    final = [r["rel_l2"] for r in ok if r["step"] == 2000]
    own = [r["rel_l2"] for r in ok if r["sample"] == 1]
    ood = [r["rel_l2"] for r in ok if r["sample"] == 2]

    result = {
        "measured": "2026-07-28",
        "authority": "A5 SYNTHESIS RULING 3(ii) / A1b Q1(e)(ii)",
        "definition": ("pixel-space rel-L2 between the generated clip's frames 0..8 and the "
                       "9-frame prefix conditioning mp4; RGB uint8 -> float [0,1]; "
                       "rel_l2 = ||gen-cond|| / ||cond||"),
        "space": "pixel (NOT latent, NOT DINOv2 -- Ruling 3(i) bans harness substrate in selection)",
        "source": "outputs/training/ladder2/spec_*/samples/step_*_{1,2}.mp4 (inline validation)",
        "TAU_ENDPOINT": stats(allv)["p95"],
        "tau_basis": "p95 over ALL prefix-conditioned inline-validation clips (all checkpoints)",
        "all_checkpoints": stats(allv),
        "final_checkpoint_step2000": stats(final) if final else None,
        "by_sample_own_class_anchor": stats(own) if own else None,
        "by_sample_ood_anchor": stats(ood) if ood else None,
        "by_arm": {d.name: stats([r["rel_l2"] for r in ok if r["arm"] == d.name]) for d in specs},
        "by_step": {str(s): stats([r["rel_l2"] for r in ok if r["step"] == s])
                    for s in sorted({r["step"] for r in ok})},
        "decode_errors": bad,
        "rows": [{k: r[k] for k in ("arm", "sample", "step", "gen", "rel_l2")} for r in ok],
    }

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    s = result["all_checkpoints"]
    print(f"[tau] n={s['n']}  p50={s['p50']:.4f}  p90={s['p90']:.4f}  "
          f"p95={s['p95']:.4f}  max={s['max']:.4f}")
    if final:
        f = result["final_checkpoint_step2000"]
        print(f"[tau] step-2000 only: n={f['n']} p50={f['p50']:.4f} p95={f['p95']:.4f}")
    print(f"[tau] TAU_ENDPOINT = {result['TAU_ENDPOINT']:.4f}  ->  {out.relative_to(REPO_ROOT)}")
    if bad:
        print(f"[tau] WARNING {len(bad)} decode errors", file=sys.stderr)


if __name__ == "__main__":
    main()
