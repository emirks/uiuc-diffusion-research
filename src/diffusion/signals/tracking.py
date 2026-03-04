"""Object tracking extractor.

Supported backends
------------------
SORTExtractor  — SORT (Simple Online and Realtime Tracking) with an inline
                 Kalman filter.  Runs its own YOLOv8 detection internally,
                 so no separate detection step is required.

Install
-------
    pip install ultralytics filterpy scipy

Reference
---------
Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import BaseExtractor
from .io import draw_boxes


# ---------------------------------------------------------------------------
# Minimal SORT implementation (Kalman + IoU Hungarian assignment)
# ---------------------------------------------------------------------------

def _box_to_z(box: np.ndarray) -> np.ndarray:
    """Convert [x1,y1,x2,y2] → [cx,cy,s,r] measurement vector."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = box[0] + w / 2
    cy = box[1] + h / 2
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([[cx], [cy], [s], [r]], dtype=np.float64)


def _z_to_box(z: np.ndarray) -> np.ndarray:
    """Convert [cx,cy,s,r] state → [x1,y1,x2,y2]."""
    cx, cy, s, r = float(z[0]), float(z[1]), float(z[2]), float(z[3])
    w = np.sqrt(max(s * r, 0.0))
    h = s / (w + 1e-6)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
    inter = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


class _KalmanTrack:
    """Single-object track backed by a 7-D Kalman filter.

    State : [cx, cy, s, r, vcx, vcy, vs]
    Obs   : [cx, cy, s, r]
    """

    _count = 0

    def __init__(self, box: np.ndarray) -> None:
        from filterpy.kalman import KalmanFilter

        _KalmanTrack._count += 1
        self.id = _KalmanTrack._count
        self.hits = 1
        self.no_match_count = 0

        kf = KalmanFilter(dim_x=7, dim_z=4)
        # Constant-velocity motion model
        kf.F = np.eye(7, dtype=np.float64)
        for i in range(4):
            kf.F[i, i + 3] = 1.0          # position += velocity
        kf.H = np.eye(4, 7, dtype=np.float64)

        kf.R[2:, 2:] *= 10.0              # measurement noise (area, ratio)
        kf.P[4:, 4:] *= 1000.0           # high uncertainty on initial velocity
        kf.P *= 10.0
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01

        z = _box_to_z(box)
        kf.x[:4] = z
        self._kf = kf

    def predict(self) -> np.ndarray:
        """Advance state by one step; return predicted [x1,y1,x2,y2]."""
        if self._kf.x[6] + self._kf.x[2] <= 0:
            self._kf.x[6] = 0.0
        self._kf.predict()
        self.no_match_count += 1
        return _z_to_box(self._kf.x)

    def update(self, box: np.ndarray) -> None:
        self.no_match_count = 0
        self.hits += 1
        self._kf.update(_box_to_z(box))

    @property
    def state_box(self) -> np.ndarray:
        return _z_to_box(self._kf.x)


class _SORT:
    """SORT multi-object tracker."""

    def __init__(self, max_age: int = 3, min_hits: int = 2, iou_threshold: float = 0.3) -> None:
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._tracks: list[_KalmanTrack] = []
        _KalmanTrack._count = 0  # reset IDs per video

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Update tracker with new detections.

        Parameters
        ----------
        detections : (N, 4) float  [x1,y1,x2,y2]  (or empty)

        Returns
        -------
        (M, 5) float  [x1,y1,x2,y2,track_id]  — confirmed tracks only
        """
        # Predict all existing tracks
        predicted_boxes = np.array(
            [t.predict() for t in self._tracks], dtype=np.float64
        ) if self._tracks else np.empty((0, 4))

        # Hungarian matching
        matched, unmatched_dets, unmatched_trks = self._match(
            detections, predicted_boxes
        )

        # Update matched tracks
        for trk_idx, det_idx in matched:
            self._tracks[trk_idx].update(detections[det_idx])

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._tracks.append(_KalmanTrack(detections[det_idx]))

        # Remove dead tracks
        self._tracks = [
            t for t in self._tracks if t.no_match_count <= self._max_age
        ]

        # Return confirmed active tracks
        results = []
        for t in self._tracks:
            if t.hits >= self._min_hits or t.no_match_count == 0:
                box = t.state_box
                results.append([*box, t.id])

        return np.array(results, dtype=np.float32) if results else np.empty((0, 5), dtype=np.float32)

    def _match(
        self,
        detections: np.ndarray,
        predictions: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if len(predictions) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(predictions)))

        # Build IoU cost matrix
        iou_matrix = np.zeros((len(predictions), len(detections)), dtype=np.float64)
        for p_idx, pred in enumerate(predictions):
            for d_idx, det in enumerate(detections):
                iou_matrix[p_idx, d_idx] = _iou(pred, det)

        # Hungarian assignment on cost = 1 - IoU
        row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)

        matched: list[tuple[int, int]] = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(predictions)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self._iou_threshold:
                matched.append((r, c))
                unmatched_dets.remove(c)
                unmatched_trks.remove(r)

        return matched, unmatched_dets, unmatched_trks


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class SORTExtractor(BaseExtractor):
    """SORT tracker — runs YOLOv8 detection + Kalman-based tracking over video frames.

    Result keys
    -----------
    tracks : list of np.ndarray, each (M, 5) — (x1, y1, x2, y2, track_id) per frame
    """

    name = "track_sort"
    input_type = "video"

    def __init__(
        self,
        device: str = "cuda",
        detector_model: str = "yolov8n.pt",
        conf_threshold: float = 0.3,
        max_age: int = 3,
        min_hits: int = 2,
        iou_threshold: float = 0.3,
    ) -> None:
        super().__init__(device)
        self._detector_model = detector_model
        self._conf = conf_threshold
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold

    def _load_model(self) -> None:
        from ultralytics import YOLO

        print(f"  Loading SORT tracker (detector: {self._detector_model}) …")
        self._detector = YOLO(self._detector_model)

    def extract(self, frames: list[np.ndarray]) -> dict:
        tracker = _SORT(
            max_age=self._max_age,
            min_hits=self._min_hits,
            iou_threshold=self._iou_threshold,
        )
        all_tracks: list[np.ndarray] = []

        for frame in frames:
            results = self._detector(frame, verbose=False, conf=self._conf, device=self.device)
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                dets = r.boxes.xyxy.cpu().numpy().astype(np.float64)
            else:
                dets = np.empty((0, 4), dtype=np.float64)

            frame_tracks = tracker.update(dets)
            all_tracks.append(frame_tracks)

        return {"tracks": all_tracks}

    def visualize(self, frames: list[np.ndarray], result: dict) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for frame, frame_tracks in zip(frames, result["tracks"]):
            if len(frame_tracks) > 0:
                boxes = frame_tracks[:, :4]
                ids   = frame_tracks[:, 4]
                canvas = draw_boxes(frame, boxes, ids)
            else:
                canvas = frame.copy()
            out.append(canvas)
        return out
