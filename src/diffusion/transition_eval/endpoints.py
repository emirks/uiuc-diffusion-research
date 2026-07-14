"""M5 — Endpoint fidelity and boundary seam detection.

Fidelity: LPIPS + DINO cosine between the generated video's conditioned frames
(first n_prefix / last n_suffix) and the condition clips, after resize-to-cover
+ center-crop to the generated frame geometry (mirroring the ValidationRunner).

Seams: the morph curves can't see a 1-frame discontinuity where conditioning
hands off to generation; temporal LPIPS d(t, t+1) can. A seam = a robust
z-score spike of d at the boundary indices relative to the video's own d(t)
distribution.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np
import torch

from .video_io import resize_cover_crop

# Bump on ANY numeric change to LpipsScorer / endpoint_fidelity / temporal_lpips
# preprocessing — the cache stores their outputs keyed by stat-based video
# identity. Certification's warm rerun runs with the cache OFF and its cold
# anchors with an empty cache dir, so a stale-cache bug cannot pass silently.
LPIPS_CACHE_TAG = "alex-v1"


def lpips_cache_path(key: str, cache_dir: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(cache_dir) / f"lpips_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"


def cached_temporal_lpips(frames: np.ndarray | None, key: str,
                          cache_dir: pathlib.Path | None, scorer: "LpipsScorer") -> np.ndarray:
    """Cache-through temporal_lpips keyed by the video's bundle key (stat-based
    identity + geometry). cache_dir=None disables caching entirely (compute
    fresh, write nothing). A cache miss with frames=None is a caller bug —
    callers may only skip decoding after checking the cache is warm."""
    p = (lpips_cache_path(f"{key}:tlpips:{LPIPS_CACHE_TAG}", cache_dir)
         if cache_dir is not None else None)
    if p is not None and p.exists():
        return np.load(p)["d"]
    if frames is None:
        raise RuntimeError(f"temporal-lpips cache miss for {key} but no frames were decoded")
    d = temporal_lpips(frames, scorer)
    if p is not None:
        np.savez_compressed(p, d=d)
    return d


class LpipsScorer:
    def __init__(self, device: str = "cuda", net: str = "alex"):
        import lpips

        self.device = device
        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()

    @torch.no_grad()
    def pairwise(self, frames1: np.ndarray, frames2: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """Per-index LPIPS between two aligned uint8 frame stacks [N,H,W,3]."""
        assert frames1.shape == frames2.shape, (frames1.shape, frames2.shape)
        out = []
        for i in range(0, len(frames1), batch_size):
            t1 = torch.from_numpy(frames1[i:i + batch_size]).permute(0, 3, 1, 2).float().to(self.device)
            t2 = torch.from_numpy(frames2[i:i + batch_size]).permute(0, 3, 1, 2).float().to(self.device)
            d = self.model(t1 / 127.5 - 1.0, t2 / 127.5 - 1.0)
            out.append(d.flatten().cpu().numpy())
        return np.concatenate(out)

    def free(self) -> None:
        del self.model
        torch.cuda.empty_cache()


def endpoint_fidelity(gen_frames: np.ndarray, gen_feats: np.ndarray,
                      cond_frames: np.ndarray, cond_feats_fn,
                      lpips_scorer: LpipsScorer, side: str) -> dict:
    """side='prefix' compares gen[:n] to the condition clip's n frames;
    side='suffix' compares gen[-n:] to the condition clip's LAST n frames
    (the suffix condition keeps trailing content). cond_feats_fn(frames)
    extracts DINO features for the cropped condition frames."""
    h, w = gen_frames.shape[1:3]
    cond = resize_cover_crop(cond_frames, h, w)
    n = len(cond)
    if side == "prefix":
        gen_slice, feat_slice = gen_frames[:n], gen_feats[:n]
    else:
        gen_slice, feat_slice = gen_frames[-n:], gen_feats[-n:]
    lp = lpips_scorer.pairwise(gen_slice, cond)
    cf = cond_feats_fn(cond)
    dino = (feat_slice * cf).sum(axis=1)
    return {f"{side}_lpips": float(lp.mean()), f"{side}_dino": float(dino.mean())}


def temporal_lpips(frames: np.ndarray, lpips_scorer: LpipsScorer) -> np.ndarray:
    """d[t] = LPIPS(frame_t, frame_{t+1}), length T-1."""
    return lpips_scorer.pairwise(frames[:-1], frames[1:])


def seam_scores(d: np.ndarray, n_prefix: int, n_suffix: int) -> dict:
    """Robust z-score of d at the conditioning handoff boundaries
    (prefix->generated at d[n_prefix-1]; generated->suffix at d[T-n_suffix-1])
    against the rest of the video's d(t) distribution (median/MAD)."""
    T = len(d) + 1
    idxs = {"prefix_seam": n_prefix - 1, "suffix_seam": T - n_suffix - 1}
    rest = np.delete(d, list(idxs.values()))
    med = np.median(rest)
    mad = 1.4826 * np.median(np.abs(rest - med)) + 1e-8
    out = {f"{k}_z": float((d[i] - med) / mad) for k, i in idxs.items()}
    out["max_seam_z"] = max(out.values())
    out["d_argmax"] = int(np.argmax(d))
    return out
