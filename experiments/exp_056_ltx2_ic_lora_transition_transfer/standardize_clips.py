"""exp_056 — Step 1: standardize all transition clips to a fixed spec.

Every clip in data/processed/transitions/ becomes exactly
480x640 (WxH, portrait 3:4) x 121 frames @ 24 fps, H.264 crf14.

Rules (see README):
- frame selection: 121 evenly-spaced indices over the source frames
  (identity-trim for 122f clips, exact 2x decimation for 242f clips,
  even spread for the one 150f clip) -> the FULL transition arc is
  always preserved; long clips play at higher speed.
- spatial: resize to cover 480x640 (INTER_AREA), center crop.
- names normalized: shadow_smoke.mp4 -> shadow_smoke_0.mp4,
  raven_transiton_2.mp4 (typo) -> raven_transition_2.mp4.

Idempotent: skips outputs that already exist.
Run on a login node (few minutes of CPU) inside the diffusion conda env.
"""

import json
import pathlib
import subprocess

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "data/processed/transitions"
DST_ROOT = REPO_ROOT / "data/processed/transitions_std121"
FFMPEG = (
    "/projects/illinois/eng/cs/jrehg/users/emirkisa/LTX-2-official/.venv/lib/"
    "python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)

W, H, F, FPS = 480, 640, 121, 24

RENAMES = {
    "shadow_smoke/shadow_smoke.mp4": "shadow_smoke/shadow_smoke_0.mp4",
    "raven_transition/raven_transiton_2.mp4": "raven_transition/raven_transition_2.mp4",
}


def frame_indices(n: int) -> np.ndarray:
    """121 evenly-spaced, strictly increasing frame indices over [0, n-1]."""
    idx = np.round(np.linspace(0, n - 1, F)).astype(int)
    assert len(np.unique(idx)) == F, f"non-unique indices for n={n}"
    return idx


def resize_crop(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    s = max(W / w, H / h)
    nw, nh = round(w * s), round(h * s)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4
    frame = cv2.resize(frame, (nw, nh), interpolation=interp)
    x0, y0 = (nw - W) // 2, (nh - H) // 2
    return frame[y0 : y0 + H, x0 : x0 + W]


def standardize(src: pathlib.Path, dst: pathlib.Path) -> dict:
    cap = cv2.VideoCapture(str(src))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert n >= F, f"{src} has only {n} frames"
    want = frame_indices(n)
    wanted = set(want.tolist())

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    proc = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", str(tmp)],
        stdin=subprocess.PIPE,
    )
    written = 0
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            out = resize_crop(frame)[:, :, ::-1]  # BGR -> RGB
            proc.stdin.write(np.ascontiguousarray(out).tobytes())
            written += 1
    cap.release()
    proc.stdin.close()
    assert proc.wait() == 0, f"ffmpeg failed for {src}"
    assert written == F, f"{src}: wrote {written}/{F} frames (decoder returned fewer than probed)"
    tmp.rename(dst)
    return {"src": str(src.relative_to(REPO_ROOT)), "src_frames": n, "indices_span": [int(want[0]), int(want[-1])]}


def main():
    manifest = {}
    for src in sorted(SRC_ROOT.glob("*/*.mp4")):
        rel = str(src.relative_to(SRC_ROOT))
        rel = RENAMES.get(rel, rel)
        dst = DST_ROOT / rel
        if dst.exists():
            print(f"[skip] {rel}")
            continue
        info = standardize(src, dst)
        manifest[rel] = info
        print(f"[done] {rel}  (src {info['src_frames']}f)")

    mpath = DST_ROOT / "standardize_manifest.json"
    existing = json.loads(mpath.read_text()) if mpath.exists() else {}
    existing.update(manifest)
    mpath.write_text(json.dumps(existing, indent=2))
    print(f"[done] {len(existing)} clips in manifest -> {mpath}")


if __name__ == "__main__":
    main()
