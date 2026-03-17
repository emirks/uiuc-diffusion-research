"""Pose estimation extractors.

Supported backends
------------------
YOLOPoseExtractor      — YOLOv8-pose (ultralytics), GPU-accelerated
MediaPipeExtractor     — MediaPipe BlazePose, CPU-friendly, also yields 3-D joints

Install
-------
    pip install ultralytics mediapipe
"""
from __future__ import annotations

import numpy as np

from .base import BaseExtractor
from .io import draw_skeleton


class YOLOPoseExtractor(BaseExtractor):
    """YOLOv8-pose — 17 COCO keypoints per detected person, per frame.

    Result keys
    -----------
    keypoints : list of np.ndarray, each (N_persons, 17, 3) — (x, y, conf)
    boxes     : list of np.ndarray, each (N_persons, 4) — (x1, y1, x2, y2)
    """

    name = "pose_yolo"
    input_type = "image"

    def __init__(self, device: str = "cuda", model_name: str = "yolov8n-pose.pt") -> None:
        super().__init__(device)
        self._model_name = model_name

    def _load_model(self) -> None:
        from ultralytics import YOLO

        print(f"  Loading {self._model_name} …")
        self._model = YOLO(self._model_name)

    def extract(self, frames: list[np.ndarray]) -> dict:
        all_kps: list[np.ndarray] = []
        all_boxes: list[np.ndarray] = []
        for frame in frames:
            results = self._model(frame, verbose=False, device=self.device)
            r = results[0]
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                kps = r.keypoints.data.cpu().numpy()          # (N, 17, 3)
                boxes = r.boxes.xyxy.cpu().numpy()            # (N, 4)
            else:
                kps = np.zeros((0, 17, 3), dtype=np.float32)
                boxes = np.zeros((0, 4), dtype=np.float32)
            all_kps.append(kps)
            all_boxes.append(boxes)
        return {"keypoints": all_kps, "boxes": all_boxes}

    def visualize(self, frames: list[np.ndarray], result: dict) -> list[np.ndarray]:
        return [
            draw_skeleton(frame, kps)
            for frame, kps in zip(frames, result["keypoints"])
        ]


class MediaPipeExtractor(BaseExtractor):
    """MediaPipe BlazePose — CPU-friendly, provides both 2-D and 3-D landmarks.

    Result keys
    -----------
    keypoints_2d : list of np.ndarray, each (N_persons, 33, 3) — (x_norm, y_norm, visibility)
    keypoints_3d : list of np.ndarray, each (N_persons, 33, 4) — (x, y, z, visibility) in world coords
                   Note: MediaPipe processes one person at a time; N_persons is always 0 or 1.
    """

    name = "pose_mediapipe"
    input_type = "image"

    def _load_model(self) -> None:
        # MediaPipe package layout varies by version/distribution.
        # Try both known import paths before failing with a clear hint.
        try:
            import mediapipe.solutions.pose as _mp_pose
            import mediapipe.solutions.drawing_utils as _mp_drawing
        except Exception:
            try:
                from mediapipe.python.solutions import pose as _mp_pose
                from mediapipe.python.solutions import drawing_utils as _mp_drawing
            except Exception as exc:
                raise ImportError(
                    "MediaPipe Solutions API is unavailable. "
                    "Install a Solutions-compatible version (e.g. mediapipe==0.10.21) "
                    "or skip pose_mediapipe."
                ) from exc

        print("  Loading MediaPipe BlazePose …")
        self._mp_pose = _mp_pose
        self._mp_drawing = _mp_drawing
        self._model = self._mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )

    def extract(self, frames: list[np.ndarray]) -> dict:
        all_kps_2d: list[np.ndarray] = []
        all_kps_3d: list[np.ndarray] = []

        for frame in frames:
            res = self._model.process(frame)
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                kps_2d = np.array([[l.x, l.y, l.visibility] for l in lm], dtype=np.float32)
                kps_2d = kps_2d[np.newaxis]  # (1, 33, 3)
            else:
                kps_2d = np.zeros((0, 33, 3), dtype=np.float32)

            if res.pose_world_landmarks:
                wlm = res.pose_world_landmarks.landmark
                kps_3d = np.array([[l.x, l.y, l.z, l.visibility] for l in wlm], dtype=np.float32)
                kps_3d = kps_3d[np.newaxis]  # (1, 33, 4)
            else:
                kps_3d = np.zeros((0, 33, 4), dtype=np.float32)

            all_kps_2d.append(kps_2d)
            all_kps_3d.append(kps_3d)

        return {"keypoints_2d": all_kps_2d, "keypoints_3d": all_kps_3d}

    def visualize(self, frames: list[np.ndarray], result: dict) -> list[np.ndarray]:
        import cv2

        out_frames: list[np.ndarray] = []
        for frame, kps_2d in zip(frames, result["keypoints_2d"]):
            canvas = frame.copy()
            h, w = canvas.shape[:2]
            if len(kps_2d) > 0:
                for person_kps in kps_2d:
                    # Convert normalised coords → pixel coords for skeleton drawing
                    kps_px = person_kps.copy()
                    kps_px[:, 0] *= w
                    kps_px[:, 1] *= h
                    # Remap 33 MediaPipe landmarks to 17 COCO for consistent draw_skeleton
                    # (only draw joints that have clear COCO equivalents)
                    for i, (x, y, conf) in enumerate(kps_px):
                        if conf > 0.3:
                            cv2.circle(canvas, (int(x), int(y)), 3, (80, 255, 80), -1)
            out_frames.append(canvas)
        return out_frames
