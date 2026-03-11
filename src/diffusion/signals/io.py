"""I/O helpers for signal extraction: video reading, npy persistence, visualisation."""
from __future__ import annotations

from pathlib import Path

import cv2
import imageio
import numpy as np


# ---------------------------------------------------------------------------
# Video / image reading
# ---------------------------------------------------------------------------

def read_video_frames(path: str | Path) -> tuple[list[np.ndarray], float]:
    """Decode an mp4/avi/… file into a list of H×W×3 uint8 RGB arrays via imageio.

    Returns
    -------
    frames : list of np.ndarray
    fps    : native frame-rate (float)
    """
    reader = imageio.get_reader(str(path), format="ffmpeg")
    fps = float(reader.get_meta_data().get("fps", 8.0))
    frames: list[np.ndarray] = [frame for frame in reader]
    reader.close()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames, fps


def read_image(path: str | Path) -> np.ndarray:
    """Load a single image as H×W×3 uint8 RGB."""
    frame = imageio.imread(str(path))
    if frame is None:
        raise FileNotFoundError(path)
    # drop alpha channel if present
    return np.asarray(frame)[..., :3]


# ---------------------------------------------------------------------------
# Raw signal persistence
# ---------------------------------------------------------------------------

def save_signal_arrays(arrays: list[np.ndarray], out_dir: Path, prefix: str = "frame") -> None:
    """Save each array as a separate .npy file (frame_0000.npy, …)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, arr in enumerate(arrays):
        np.save(out_dir / f"{prefix}_{i:04d}.npy", arr)


def load_signal_arrays(directory: Path) -> list[np.ndarray]:
    """Load all .npy files from a directory, sorted by name."""
    paths = sorted(directory.glob("*.npy"))
    return [np.load(p) for p in paths]


# ---------------------------------------------------------------------------
# Visualisation output
# ---------------------------------------------------------------------------

def save_frames_png(frames: list[np.ndarray], out_dir: Path, prefix: str = "frame") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        imageio.imwrite(str(out_dir / f"{prefix}_{i:04d}.png"), frame)


def save_viz_video(frames: list[np.ndarray], path: Path, fps: float = 8.0) -> None:
    """Write a list of RGB uint8 frames to an mp4 file via imageio/ffmpeg."""
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        format="ffmpeg",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    for frame in frames:
        writer.append_data(np.asarray(frame, dtype=np.uint8))
    writer.close()


# ---------------------------------------------------------------------------
# Top-level save helper used by extract_signals.py
# ---------------------------------------------------------------------------

def save_signal_result(
    result: dict,
    extractor_name: str,
    out_root: Path,
    viz_frames: list[np.ndarray],
    fps: float = 8.0,
) -> None:
    """Persist raw arrays and visualisation for one extractor run.

    Layout::

        out_root/
            <extractor_name>/
                raw/<key>/frame_0000.npy …
                viz/frame_0000.png …
                viz.mp4
    """
    signal_dir = out_root / extractor_name
    raw_dir    = signal_dir / "raw"
    viz_dir    = signal_dir / "viz"

    for key, value in result.items():
        if isinstance(value, list) and value:
            save_signal_arrays(value, raw_dir / key)
        elif isinstance(value, np.ndarray):
            d = raw_dir / key
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "data.npy", value)

    if viz_frames:
        save_frames_png(viz_frames, viz_dir)
        save_viz_video(viz_frames, signal_dir / "viz.mp4", fps=fps)


# ---------------------------------------------------------------------------
# Visualisation utilities shared across extractors
# ---------------------------------------------------------------------------

def colormap_depth(depth: np.ndarray) -> np.ndarray:
    """Convert a single-channel float depth map to an H×W×3 uint8 RGB image."""
    d = depth.astype(np.float32)
    lo, hi = d.min(), d.max()
    if hi > lo:
        d = (d - lo) / (hi - lo)
    d_uint8 = (d * 255).astype(np.uint8)
    colored = cv2.applyColorMap(d_uint8, cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """Convert HxWx2 optical flow to an H×W×3 uint8 RGB image (HSV encoding)."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = mag_norm.astype(np.uint8)
    rgb = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)
    return rgb


def overlay_mask(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a binary or label mask onto an RGB frame."""
    out = frame.copy()
    if mask.ndim == 2:
        # label mask — assign a distinct colour per unique label
        labels = np.unique(mask)
        np.random.seed(0)
        colours = {int(lbl): np.random.randint(50, 255, 3).tolist() for lbl in labels if lbl != 0}
        overlay = frame.copy()
        for lbl, col in colours.items():
            overlay[mask == lbl] = col
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def draw_boxes(frame: np.ndarray, boxes: np.ndarray, ids: np.ndarray | None = None) -> np.ndarray:
    """Draw bounding boxes (Nx4 [x1,y1,x2,y2]) on an RGB frame."""
    out = frame.copy()
    np.random.seed(42)
    palette = np.random.randint(80, 230, (256, 3)).tolist()
    for i, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
        colour = tuple(int(c) for c in palette[int(ids[i]) % 256]) if ids is not None else (0, 220, 90)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour[::-1], 2)
        label = str(int(ids[i])) if ids is not None else ""
        if label:
            cv2.putText(out, label, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour[::-1], 1)
    return out


COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # arms
    (5, 11), (6, 12), (11, 13), (13, 15),    # left leg
    (12, 14), (14, 16),                       # right leg
    (11, 12),                                 # hips
]


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """Draw COCO-17 skeleton onto an RGB frame.

    Parameters
    ----------
    keypoints : np.ndarray, shape (N_persons, 17, 3)  — (x, y, conf)
    """
    out = frame.copy()
    for person_kps in keypoints:
        for i, (x, y, conf) in enumerate(person_kps):
            if conf > 0.3:
                cv2.circle(out, (int(x), int(y)), 4, (255, 80, 80), -1)
        for a, b in COCO_SKELETON:
            xa, ya, ca = person_kps[a]
            xb, yb, cb = person_kps[b]
            if ca > 0.3 and cb > 0.3:
                cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), (80, 200, 255), 2)
    return out
