"""Signal extraction library for diffusion video research.

Quick usage
-----------
    from diffusion.signals import SIGNAL_REGISTRY

    extractor = SIGNAL_REGISTRY["depth_dav2"](device="cuda")
    result    = extractor(frames)          # list of H×W×3 uint8 RGB
    viz       = extractor.visualize(frames, result)

Available signals
-----------------
    depth_dav2          — Depth-Anything V2 (per frame)
    depth_midas         — MiDaS DPT_Large   (per frame)
    flow_raft           — RAFT optical flow  (consecutive pairs)
    seg_sam             — SAM auto-segmentation (per frame)
    pose_yolo           — YOLOv8-pose         (per frame)
    pose_mediapipe      — MediaPipe BlazePose (per frame, CPU-friendly)
    det_yolo            — YOLOv8 detection    (per frame)
    track_sort          — SORT tracker        (full video sequence)
"""
from __future__ import annotations

from .base import BaseExtractor
from .depth import DepthAnythingExtractor, MiDaSExtractor
from .flow import RAFTExtractor
from .segmentation import SAMExtractor
from .pose import YOLOPoseExtractor, MediaPipeExtractor
from .detection import YOLODetectExtractor
from .tracking import SORTExtractor

SIGNAL_REGISTRY: dict[str, type[BaseExtractor]] = {
    "depth_dav2":       DepthAnythingExtractor,
    "depth_midas":      MiDaSExtractor,
    "flow_raft":        RAFTExtractor,
    "seg_sam":          SAMExtractor,
    "pose_yolo":        YOLOPoseExtractor,
    "pose_mediapipe":   MediaPipeExtractor,
    "det_yolo":         YOLODetectExtractor,
    "track_sort":       SORTExtractor,
}

__all__ = [
    "BaseExtractor",
    "SIGNAL_REGISTRY",
    "DepthAnythingExtractor",
    "MiDaSExtractor",
    "RAFTExtractor",
    "SAMExtractor",
    "YOLOPoseExtractor",
    "MediaPipeExtractor",
    "YOLODetectExtractor",
    "SORTExtractor",
]
