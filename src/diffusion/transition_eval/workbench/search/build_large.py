"""GPU pass — DINOv2-large [121,1024] CLS embeddings for all 223 corpus clips.

The metric-search GPU track (owner-authorized escalation). Same short_side=256
decode, same DINOv2-family AutoImageProcessor, same L2-normalized CLS token — only
the backbone changes (base 768 -> large 1024). Reuses the deployed video_features
decode+extract+cache, so a requeue skips completed clips. Writes ONLY to
$WB_CACHE/search (never the certified shared cache). The base-space core masks and
endpoint indices are reused verbatim downstream; nothing is recomputed here.
"""

from __future__ import annotations

import time

import numpy as np

from .. import paths

LARGE_MODEL = "facebook/dinov2-large"
LARGE_CACHE = paths.WB_CACHE / "search" / "large_cache"
OUT = paths.WB_CACHE / "search" / "large_feats.npz"


def log(m: str) -> None:
    print(f"[large {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    import torch
    from ...features import DinoExtractor, video_features

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("REFUSING: GPU extraction on CPU. Submit through Slurm.")
        return 1

    corpus = paths.load_corpus()
    keys = paths.corpus_keys(corpus)
    LARGE_CACHE.mkdir(parents=True, exist_ok=True)

    dino = DinoExtractor(LARGE_MODEL, device=device)
    log(f"model {LARGE_MODEL} on {device}; extracting {len(keys)} clips")

    feats_all, t0 = [], time.time()
    for i, key in enumerate(keys):
        feats, _ = video_features(paths.clip_path(key), LARGE_CACHE, dino,
                                  short_side=paths.FEATURE_SHORT_SIDE)
        feats_all.append(np.asarray(feats, dtype=np.float32))
        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(keys)} ({time.time() - t0:.0f}s)")

    dino.free()
    Ts = {f.shape[0] for f in feats_all}
    dims = {f.shape[1] for f in feats_all}
    log(f"frame counts {sorted(Ts)}  dims {sorted(dims)}")
    assert len(Ts) == 1 and len(dims) == 1, "non-uniform large feats"
    F = np.stack(feats_all)                              # [223, 121, 1024]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, keys=np.array(keys), feats=F, model=LARGE_MODEL,
             short_side=paths.FEATURE_SHORT_SIDE)
    log(f"WROTE {OUT}  feats {F.shape} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
