"""Per-frame texture (RUNBOOK §3.2's low-texture definedness gate).

    "Definedness: inlier fraction < 40% or LOW-TEXTURE FRAME -> frame undefined."

Optical flow is only as trustworthy as the structure it can track. On a
textureless frame — a white flash, a fade to black, a flat colour wash, all of
which a stylized transition corpus is full of — the aperture problem bites and
the estimator returns a smooth, confident-looking, meaningless field. A camera fit
to that noise can easily report a HIGH inlier fraction (noise agrees with itself),
so the inlier gate alone does not catch it. §3.2 names the second gate for exactly
this reason, and it needs an independent measurement of whether there was anything
to track in the first place.

Texture = mean Sobel gradient magnitude of the grayscale frame, normalized to
[0,1] intensity — a standard trackability proxy, computed at the same 432x320 the
flow runs at, so the pixels measured are the pixels RAFT saw.

A FRAME PAIR is low-texture if EITHER of its frames is: flow needs structure at
both ends of the step.

The threshold is a corpus-only calibration (a percentile of the corpus texture
distribution), pinned in gates.yaml before any candidate exam runs. This module
computes and caches the distribution; it does not choose the threshold.
"""

from __future__ import annotations

import json
import time

import numpy as np

from ..video_io import load_frames
from . import flowcache, paths

CACHE_TAG = "texture-v1"


def frame_texture(frames: np.ndarray) -> np.ndarray:
    """uint8 [T,H,W,3] -> mean |Sobel| per frame, on [0,1] grayscale."""
    import cv2

    out = np.empty(len(frames), dtype=np.float64)
    for i, f in enumerate(frames):
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        out[i] = float(np.hypot(gx, gy).mean())
    return out


def pair_texture(tex: np.ndarray) -> np.ndarray:
    """Per-frame texture -> per-PAIR texture: the weaker of the two endpoints.
    Flow between frames t and t+1 needs structure at both."""
    return np.minimum(tex[:-1], tex[1:])


def main() -> int:
    """One CPU pass over the corpus -> $WB_CACHE/texture.npz (223 x 121)."""
    corpus = paths.load_corpus()
    keys = paths.corpus_keys(corpus)
    out_path = paths.WB_CACHE / "texture.npz"
    if out_path.exists():
        print(f"[texture] warm: {out_path}")
        return 0

    paths.WB_CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    for i, k in enumerate(keys):
        frames, _ = load_frames(paths.clip_path(k), short_side=None)
        frames = flowcache.resize_for_flow(frames)      # the pixels RAFT actually saw
        rows.append(frame_texture(frames))
        if (i + 1) % 40 == 0:
            print(f"[texture] {i + 1}/{len(keys)} ({time.time() - t0:.0f}s)", flush=True)

    T = np.stack(rows)                                  # [223, 121]
    P = np.stack([pair_texture(r) for r in rows])       # [223, 120]
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, keys=np.array(keys), frame_texture=T, pair_texture=P,
                        tag=CACHE_TAG)
    tmp.replace(out_path)

    flat = P.ravel()
    pcts = {f"p{p}": float(np.percentile(flat, p)) for p in (1, 2, 5, 10, 25, 50, 75, 95)}
    print(f"[texture] corpus pair-texture distribution over {flat.size} pairs:")
    for k2, v in pcts.items():
        print(f"    {k2:>4s} {v:.5f}")
    print(f"[texture] wrote {out_path} in {time.time() - t0:.0f}s")
    (paths.WB_OUT / "step0").mkdir(parents=True, exist_ok=True)
    (paths.WB_OUT / "step0" / "texture_distribution.json").write_text(
        json.dumps({"percentiles": pcts, "n_pairs": int(flat.size),
                    "note": "corpus-only calibration input for §3.2's low-texture "
                            "gate; the threshold is pinned in gates.yaml"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
