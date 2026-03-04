"""Optical flow extractors.

Supported backends
------------------
RAFTExtractor  — RAFT (large) via torchvision — best visual quality for motion

Install
-------
    pip install torchvision   # RAFT is bundled since torchvision>=0.13

Note on FlowSeek
----------------
FlowSeek (depth-conditioned flow) requires its own checkout and weights.
When available, subclass BaseExtractor following the same interface:

    class FlowSeekExtractor(BaseExtractor):
        name = "flow_flowseek"
        input_type = "video"
        ...
"""
from __future__ import annotations

import numpy as np

from .base import BaseExtractor
from .io import flow_to_rgb


class RAFTExtractor(BaseExtractor):
    """RAFT large — dense optical flow from consecutive frame pairs.

    Result keys
    -----------
    flow : list of np.ndarray, each H×W×2 (dx, dy in pixels), len = N-1 frames
    """

    name = "flow_raft"
    input_type = "video"

    def _load_model(self) -> None:
        import torch
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

        print("  Loading RAFT-large …")
        weights = Raft_Large_Weights.DEFAULT
        self._model = raft_large(weights=weights).to(self.device).eval()
        self._transforms = weights.transforms()
        self._torch = torch

    def extract(self, frames: list[np.ndarray]) -> dict:
        import torch
        from torchvision.transforms.functional import resize as tv_resize
        import torchvision.transforms.functional as TF

        if len(frames) < 2:
            raise ValueError("RAFT requires at least 2 frames")

        flows: list[np.ndarray] = []
        for i in range(len(frames) - 1):
            f1 = TF.to_tensor(frames[i]).unsqueeze(0)
            f2 = TF.to_tensor(frames[i + 1]).unsqueeze(0)
            # RAFT expects images in [0, 1] float; transforms() handles normalisation
            f1t, f2t = self._transforms(f1, f2)
            f1t = f1t.to(self.device)
            f2t = f2t.to(self.device)
            with torch.no_grad():
                predicted = self._model(f1t, f2t)
            # last element is the finest-scale prediction; shape (1,2,H,W)
            flow = predicted[-1].squeeze(0).permute(1, 2, 0).cpu().numpy()
            flows.append(flow.astype(np.float32))

        return {"flow": flows}

    def visualize(self, frames: list[np.ndarray], result: dict) -> list[np.ndarray]:
        """Render each flow field as an HSV-encoded RGB image."""
        return [flow_to_rgb(f) for f in result["flow"]]
