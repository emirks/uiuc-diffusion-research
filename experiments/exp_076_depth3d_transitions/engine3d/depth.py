"""Monocular depth for the 2.5D transition renderer (Depth Anything V2-Small).

V2-Small is the Apache-2.0 checkpoint; Base/Large/Giant are CC-BY-NC. The model
predicts *relative inverse depth* (disparity: larger = nearer), which is exactly
what a parallax renderer wants — we only need the ordering plus a controllable
depth range, not metric scale.
"""

from __future__ import annotations

import functools

import numpy as np
import torch

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


@functools.lru_cache(maxsize=1)
def _model(device: str):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    net = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()
    return proc, net


@torch.inference_mode()
def disparity(frames: np.ndarray, device: str = "cpu") -> np.ndarray:
    """(N,H,W,3) uint8 -> (N,H,W) float32 disparity, per-frame normalised to [0,1]."""
    proc, net = _model(device)
    inputs = proc(images=list(frames), return_tensors="pt").to(device)
    pred = net(**inputs).predicted_depth                      # (N, h, w)
    pred = torch.nn.functional.interpolate(
        pred[:, None], size=frames.shape[1:3], mode="bicubic", align_corners=False
    )[:, 0]
    lo = pred.amin(dim=(1, 2), keepdim=True)
    hi = pred.amax(dim=(1, 2), keepdim=True)
    return ((pred - lo) / (hi - lo + 1e-8)).float().cpu().numpy()


def to_view_depth(disp: np.ndarray, near: float = 1.0, far: float = 4.0,
                  gamma: float = 1.0) -> np.ndarray:
    """Map normalised disparity to a view-space z used by the renderer.

    `far/near` sets how much parallax the scene has: near==far is a flat card,
    a large ratio pushes the background away and exaggerates the 3D. `gamma`
    reshapes the distribution — <1 pulls the midground forward, which usually
    reads better than the model's raw (background-heavy) disparity histogram.
    """
    d = np.clip(disp, 0.0, 1.0) ** gamma
    return (near + (1.0 - d) * (far - near)).astype(np.float32)
