"""Depth estimation extractors.

Supported backends
------------------
DepthAnythingExtractor  — Depth-Anything V2 via HuggingFace Transformers
MiDaSExtractor          — MiDaS via torch.hub (DPT_Large by default)

Install
-------
    pip install transformers timm   # for Depth-Anything V2
    # MiDaS is downloaded automatically via torch.hub on first use
"""
from __future__ import annotations

import numpy as np

from .base import BaseExtractor
from .io import colormap_depth


class DepthAnythingExtractor(BaseExtractor):
    """Depth-Anything V2 (Small) — per-frame, transformer-based."""

    name = "depth_dav2"
    input_type = "image"

    def __init__(self, device: str = "cuda", model_size: str = "Small") -> None:
        super().__init__(device)
        self._model_size = model_size

    def _load_model(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch

        hf_id = f"depth-anything/Depth-Anything-V2-{self._model_size}-hf"
        print(f"  Loading Depth-Anything V2 ({self._model_size}) from {hf_id} …")
        self._processor = AutoImageProcessor.from_pretrained(hf_id)
        self._model = (
            AutoModelForDepthEstimation.from_pretrained(hf_id)
            .to(self.device)
            .eval()
        )
        self._torch = torch

    def extract(self, frames: list[np.ndarray]) -> dict:
        import PIL.Image

        depths: list[np.ndarray] = []
        for frame in frames:
            pil = PIL.Image.fromarray(frame)
            inputs = self._processor(images=pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                pred = self._model(**inputs).predicted_depth
            depth = pred.squeeze().cpu().numpy().astype(np.float32)
            depths.append(depth)
        return {"depth": depths}

    def visualize(self, frames: list[np.ndarray], result: dict) -> list[np.ndarray]:
        return [colormap_depth(d) for d in result["depth"]]


class MiDaSExtractor(BaseExtractor):
    """MiDaS DPT_Large — per-frame, very stable baseline."""

    name = "depth_midas"
    input_type = "image"

    def __init__(self, device: str = "cuda", model_type: str = "DPT_Large") -> None:
        super().__init__(device)
        self._model_type = model_type

    def _load_model(self) -> None:
        import torch

        print(f"  Loading MiDaS {self._model_type} via torch.hub …")
        self._model = (
            torch.hub.load("intel-isl/MiDaS", self._model_type, trust_repo=True)
            .to(self.device)
            .eval()
        )
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if self._model_type in ("DPT_Large", "DPT_Hybrid"):
            self._transform = transforms.dpt_transform
        else:
            self._transform = transforms.small_transform
        self._torch = torch

    def extract(self, frames: list[np.ndarray]) -> dict:
        depths: list[np.ndarray] = []
        for frame in frames:
            inp = self._transform(frame).to(self.device)
            with self._torch.no_grad():
                pred = self._model(inp)
                pred = self._torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depths.append(pred.cpu().numpy().astype(np.float32))
        return {"depth": depths}

    def visualize(self, frames: list[np.ndarray], result: dict) -> list[np.ndarray]:
        return [colormap_depth(d) for d in result["depth"]]
