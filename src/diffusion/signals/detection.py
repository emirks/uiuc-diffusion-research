"""Bounding-box detection extractor.

Supported backends
------------------
YOLODetectExtractor  — YOLOv8 detection (ultralytics)

Install
-------
    pip install ultralytics
"""
from __future__ import annotations

import numpy as np

from .base import BaseExtractor
from .io import draw_boxes


class YOLODetectExtractor(BaseExtractor):
    """YOLOv8 bounding-box detection — per frame.

    Result keys
    -----------
    boxes  : list of np.ndarray, each (N, 4) — (x1, y1, x2, y2) in pixels
    scores : list of np.ndarray, each (N,)   — confidence scores [0, 1]
    labels : list of np.ndarray, each (N,)   — COCO class indices (int)
    """

    name = "det_yolo"
    input_type = "image"

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
    ) -> None:
        super().__init__(device)
        self._model_name = model_name
        self._conf = conf_threshold

    def _load_model(self) -> None:
        from ultralytics import YOLO

        print(f"  Loading {self._model_name} …")
        self._model = YOLO(self._model_name)

    def extract(self, frames: list[np.ndarray]) -> dict:
        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for frame in frames:
            results = self._model(frame, verbose=False, conf=self._conf, device=self.device)
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                boxes  = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                scores = r.boxes.conf.cpu().numpy().astype(np.float32)
                labels = r.boxes.cls.cpu().numpy().astype(np.int32)
            else:
                boxes  = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,),   dtype=np.float32)
                labels = np.zeros((0,),   dtype=np.int32)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return {"boxes": all_boxes, "scores": all_scores, "labels": all_labels}

    def visualize(self, frames: list[np.ndarray], result: dict) -> list[np.ndarray]:
        return [
            draw_boxes(frame, boxes)
            for frame, boxes in zip(frames, result["boxes"])
        ]
