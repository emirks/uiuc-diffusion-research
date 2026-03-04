"""Segmentation mask extractor.

Supported backends
------------------
SAMExtractor  — Segment-Anything Model (automatic mask generation, per frame)

Install
-------
    pip install git+https://github.com/facebookresearch/segment-anything.git

Weights (download once, ~2.4 GB for ViT-H)
-------
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    # or ViT-B (~375 MB, faster):
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Pass the local path via ``checkpoint`` argument.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseExtractor
from .io import overlay_mask


_DEFAULT_SAM_CHECKPOINT = Path.home() / ".cache" / "sam" / "sam_vit_b_01ec64.pth"
_DEFAULT_SAM_MODEL_TYPE = "vit_b"


class SAMExtractor(BaseExtractor):
    """SAM automatic mask generator — applied per frame.

    Result keys
    -----------
    masks : list of np.ndarray, each H×W uint8 (label map: 0 = background, 1…N = mask index)
    """

    name = "seg_sam"
    input_type = "image"

    def __init__(
        self,
        device: str = "cuda",
        checkpoint: str | Path | None = None,
        model_type: str = _DEFAULT_SAM_MODEL_TYPE,
        points_per_side: int = 16,
    ) -> None:
        super().__init__(device)
        self._checkpoint = Path(checkpoint) if checkpoint else _DEFAULT_SAM_CHECKPOINT
        self._model_type = model_type
        self._points_per_side = points_per_side

    def _load_model(self) -> None:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        if not self._checkpoint.exists():
            raise FileNotFoundError(
                f"SAM weights not found at {self._checkpoint}.\n"
                "Download with:\n"
                "  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth "
                f"-P {self._checkpoint.parent}"
            )
        print(f"  Loading SAM ({self._model_type}) from {self._checkpoint} …")
        sam = sam_model_registry[self._model_type](checkpoint=str(self._checkpoint))
        sam.to(self.device)
        self._model = SamAutomaticMaskGenerator(
            sam,
            points_per_side=self._points_per_side,
        )

    def extract(self, frames: list[np.ndarray]) -> dict:
        label_maps: list[np.ndarray] = []
        for frame in frames:
            anns = self._model.generate(frame)
            # Sort by area descending so smaller masks paint over larger ones
            anns_sorted = sorted(anns, key=lambda a: a["area"], reverse=True)
            label_map = np.zeros(frame.shape[:2], dtype=np.uint8)
            for idx, ann in enumerate(anns_sorted, start=1):
                label_map[ann["segmentation"]] = min(idx, 255)
            label_maps.append(label_map)
        return {"masks": label_maps}

    def visualize(self, frames: list[np.ndarray], result: dict) -> list[np.ndarray]:
        return [
            overlay_mask(frame, mask)
            for frame, mask in zip(frames, result["masks"])
        ]
