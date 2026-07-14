"""Optical-flow extraction + cache (RUNBOOK §3.1, amendment A2).

BACKBONE: SEA-RAFT (the RUNBOOK's primary choice). A2 concretized the selection
procedure as "attempt SEA-RAFT install once, timeboxed (~30 min); on failure fall
back without ceremony to torchvision RAFT". The attempt SUCCEEDED (2026-07-14
11:26-11:27), so the primary is what ships — taking the fallback while the
primary works would itself be a deviation from the pre-registration. The choice
is made ONCE, before any Phase 1 acceptance test, and is never revisited this
cycle.

Everything needed to reproduce the backbone is pinned in gates.yaml and stamped
into every flow artifact: repo commit, HF weights id, config, iters, resolution.

RESOLUTION — 432x320 (short side 320, per §3.1's "~320px"), reached by decoding
each clip at NATIVE 480x640 and downsampling once with INTER_AREA. Both dims are
divisible by 8, which RAFT-family feature pyramids require. The obvious route
(decode at short_side=320) is a trap: it yields 427x320, and 427 is not divisible
by 8. The deployed torchvision wrapper would not have saved us — signals/flow.py
`_resize` rounds to /8 ONLY when it downscales (`scale == 1.0` returns the frame
untouched), so a 427-tall frame passes straight through to the model.

Flow is cached DENSE, at full resolution, float16: the §3.3 energy-gate epsilon is
a corpus calibration that can only be chosen after seeing the residual
distribution, and the residual is flow minus the fitted camera field. Caching
anything less would make an epsilon recalibration — or any descriptor
re-derivation — a GPU re-run, which is exactly what the front-load rule exists to
prevent. ~15 GB for 223 clips; the projects filesystem has terabytes free.

Every one of a clip's 120 adjacent pairs is cached unconditionally. §3.1's
seam-frame exclusion is a DESCRIPTOR-time decision, not a cache-time one — a
cache that pre-excluded frames could not answer a later question about them.

This module never edits src/diffusion/signals/flow.py (a shared, deployed
module). It is new code in workbench/.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

from ..features import file_key
from ..video_io import load_frames
from . import paths

CACHE_TAG = "searaft-v1"
FLOW_H, FLOW_W = 432, 320          # both % 8 == 0; short side 320 (§3.1)
ITERS = 4                          # SEA-RAFT config/eval/spring-M.json default

SEARAFT_REPO = paths.WB / "outputs/ext/SEA-RAFT"
SEARAFT_COMMIT = "9137517ba24e628442aec097d3afe71d03503b75"
SEARAFT_WEIGHTS = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
SEARAFT_CONFIG = "config/eval/spring-M.json"

PINS = {
    "backbone": "SEA-RAFT",
    "repo": "https://github.com/princeton-vl/SEA-RAFT",
    "commit": SEARAFT_COMMIT,
    "weights": SEARAFT_WEIGHTS,
    "config": SEARAFT_CONFIG,
    "iters": ITERS,
    "resolution_hw": [FLOW_H, FLOW_W],
    "resize": "native 480x640 -> cv2.INTER_AREA -> 432x320 (both %8==0)",
    "input": "RGB float32 in [0,255]; the model normalizes internally (2*x/255 - 1)",
    "cache_tag": CACHE_TAG,
    "dtype": "float16",
    "pairs_per_clip": "all adjacent pairs (120); seam exclusion is descriptor-time",
}


def flow_cache_path(key: str, cache_dir: pathlib.Path) -> pathlib.Path:
    """Mirrors features.feature_cache_path — own prefix, own tag in the key."""
    h = hashlib.sha1(f"{key}:{CACHE_TAG}".encode()).hexdigest()[:16]
    return pathlib.Path(cache_dir) / f"flow_{h}.npz"


def clip_flow_key(path: pathlib.Path) -> str:
    """Content-keyed like the deployed caches: abspath|mtime|size|backbone|res."""
    return file_key(path, "searaft", SEARAFT_COMMIT, SEARAFT_WEIGHTS,
                    f"{FLOW_H}x{FLOW_W}", str(ITERS))


def resize_for_flow(frames: np.ndarray) -> np.ndarray:
    """Native uint8 [T,H,W,3] -> [T,432,320,3], INTER_AREA, deterministic."""
    import cv2

    if frames.shape[1:3] == (FLOW_H, FLOW_W):
        return frames
    return np.stack([cv2.resize(f, (FLOW_W, FLOW_H), interpolation=cv2.INTER_AREA)
                     for f in frames])


class SeaRaftExtractor:
    """SEA-RAFT, loaded from the pinned commit + HF weights. New code — the
    deployed signals/flow.py wrapper is for torchvision RAFT and is not touched."""

    def __init__(self, device: str = "cuda"):
        import argparse

        import torch

        if not SEARAFT_REPO.exists():
            raise RuntimeError(
                f"SEA-RAFT checkout missing at {SEARAFT_REPO}. Clone it at commit "
                f"{SEARAFT_COMMIT}: git clone https://github.com/princeton-vl/SEA-RAFT")
        sys.path.insert(0, str(SEARAFT_REPO))
        sys.path.insert(0, str(SEARAFT_REPO / "core"))
        from core.raft import RAFT

        cfg = json.loads((SEARAFT_REPO / SEARAFT_CONFIG).read_text())
        self.args = argparse.Namespace(**cfg)
        self.device = device
        self.torch = torch
        self.model = RAFT.from_pretrained(SEARAFT_WEIGHTS, args=self.args).to(device).eval()

    @property
    def _no_grad(self):
        return self.torch.no_grad

    def flow_pairs(self, frames: np.ndarray, batch_size: int = 8) -> np.ndarray:
        """uint8 RGB [T,432,320,3] -> float16 flow [T-1, 432, 320, 2] (dx, dy px).

        Batched purely for throughput; the model is deterministic in eval mode and
        each pair is independent, so batching changes no number."""
        torch = self.torch
        x = torch.from_numpy(frames.astype(np.float32)).permute(0, 3, 1, 2)  # [T,3,H,W], 0..255
        out = []
        with torch.no_grad():
            for i in range(0, len(x) - 1, batch_size):
                a = x[i:i + batch_size].to(self.device)
                b = x[i + 1:i + 1 + batch_size].to(self.device)
                n = min(len(a), len(b))
                if n == 0:
                    break
                r = self.model(a[:n], b[:n], iters=ITERS, test_mode=True)
                f = r["flow"]
                if isinstance(f, (list, tuple)):
                    f = f[-1]                       # finest-scale prediction
                out.append(f.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16))
        return np.concatenate(out) if out else np.zeros((0, FLOW_H, FLOW_W, 2), np.float16)

    def free(self) -> None:
        del self.model
        self.torch.cuda.empty_cache()


def build_clip_flow(path: pathlib.Path, extractor: SeaRaftExtractor,
                    cache_dir: pathlib.Path) -> pathlib.Path:
    """Cache one clip's dense flow. Idempotent: a warm entry is left alone."""
    key = clip_flow_key(path)
    cache = flow_cache_path(key, cache_dir)
    if cache.exists():
        return cache
    frames, _ = load_frames(path, short_side=None)     # NATIVE, then INTER_AREA
    frames = resize_for_flow(frames)
    flow = extractor.flow_pairs(frames)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, flow=flow, src=str(path), n_frames=len(frames), **{
        "pin_commit": SEARAFT_COMMIT, "pin_weights": SEARAFT_WEIGHTS, "pin_iters": ITERS})
    tmp.replace(cache)                                 # atomic: no half-written entry
    return cache


def load_clip_flow(path: pathlib.Path, cache_dir: pathlib.Path) -> np.ndarray:
    """float16 [T-1, H, W, 2] from cache. Raises on a miss — the workbench never
    silently recomputes flow outside the cache-build job."""
    cache = flow_cache_path(clip_flow_key(path), cache_dir)
    if not cache.exists():
        raise RuntimeError(f"flow cache miss for {path} ({cache.name}) — run the "
                           f"cache-build job (OPERATIONS §6 step 3)")
    return np.load(cache)["flow"]
