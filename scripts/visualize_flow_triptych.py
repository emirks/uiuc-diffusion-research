"""Create side-by-side optical-flow visualization video.

Output layout per frame:
    [ original frame_t | HSV flow | quiver overlay on frame_t ]

Inputs
------
- signal_dir: path like
    outputs/signals/.../<sample_id>/<start|end>/flow_raft
  containing:
    raw/flow/frame_XXXX.npy
- source video clip:
    auto-resolved from sibling file:
      <sample_id>/<start|end>_clip.mp4
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from diffusion.signals.io import flow_to_rgb, read_video_frames


def _load_flows(flow_dir: Path) -> list[np.ndarray]:
    paths = sorted(flow_dir.glob("frame_*.npy"))
    if not paths:
        raise FileNotFoundError(f"No flow files found in {flow_dir}")
    return [np.load(p) for p in paths]


def _resolve_source_clip(signal_dir: Path, source_clip: Path | None) -> Path:
    if source_clip is not None:
        return source_clip
    role_dir = signal_dir.parent  # .../<sample>/<role>
    role = role_dir.name
    candidate = role_dir.parent / f"{role}_clip.mp4"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not auto-resolve source clip. Expected {candidate}. "
        "Pass --source-clip explicitly."
    )


def _draw_quiver(
    frame: np.ndarray,
    flow: np.ndarray,
    stride: int = 24,
    scale: float = 1.0,
    max_arrows: int = 1200,
) -> np.ndarray:
    """Draw sparse flow arrows on top of frame."""
    out = frame.copy()
    h, w = flow.shape[:2]

    # Keep drawing cost bounded on large frames.
    current = max(1, (h // stride) * (w // stride))
    if current > max_arrows:
        stride = int(math.ceil(math.sqrt((h * w) / max_arrows)))
        stride = max(stride, 8)

    for y in range(stride // 2, h, stride):
        for x in range(stride // 2, w, stride):
            dx, dy = flow[y, x]
            x2 = int(round(x + dx * scale))
            y2 = int(round(y + dy * scale))
            cv2.arrowedLine(
                out,
                (int(x), int(y)),
                (x2, y2),
                (80, 255, 80),
                1,
                line_type=cv2.LINE_AA,
                tipLength=0.25,
            )
    return out


def _make_triptych(frames: list[np.ndarray], flows: list[np.ndarray], stride: int, scale: float) -> list[np.ndarray]:
    n = min(len(flows), len(frames) - 1)
    out: list[np.ndarray] = []
    for i in range(n):
        frame_t = frames[i]
        hsv = flow_to_rgb(flows[i])
        quiver = _draw_quiver(frame_t, flows[i], stride=stride, scale=scale)
        panel = np.concatenate([frame_t, hsv, quiver], axis=1)
        out.append(panel.astype(np.uint8))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize flow as frame|HSV|quiver triptych")
    p.add_argument("--signal-dir", type=Path, required=True, help="Path to .../<role>/flow_raft")
    p.add_argument("--source-clip", type=Path, default=None, help="Optional explicit source clip path")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output mp4 path (default: <signal-dir>/viz_triptych.mp4)",
    )
    p.add_argument("--stride", type=int, default=24, help="Arrow grid stride in pixels")
    p.add_argument("--scale", type=float, default=1.0, help="Arrow scale factor")
    args = p.parse_args()

    signal_dir = args.signal_dir.resolve()
    flow_dir = signal_dir / "raw" / "flow"
    out_path = args.out.resolve() if args.out else (signal_dir / "viz_triptych.mp4")

    source_clip = _resolve_source_clip(signal_dir, args.source_clip)
    frames, fps = read_video_frames(source_clip)
    flows = _load_flows(flow_dir)
    panels = _make_triptych(frames, flows, stride=args.stride, scale=args.scale)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path),
        format="ffmpeg",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    for panel in panels:
        writer.append_data(panel)
    writer.close()

    print(f"Saved: {out_path}")
    print(f"Frames: {len(panels)} @ {fps:.2f} fps")


if __name__ == "__main__":
    main()

