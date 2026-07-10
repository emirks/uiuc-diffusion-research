"""exp_058 — standardize the NEW training classes to the exp_056 spec.

Extends data/processed/transitions_std121/ with the 14 one-sided classes not
selected in exp_057 (design.md §3). 480x640 (WxH) x 121f @ 24fps H.264 crf14.
Adds a per-clip min-resolution guard (>=480 on both dims) — run_set_on_fire
mixes 320px and high-res sources and only the high-res clips are taken.
Idempotent (skips existing). Login-node safe (CPU, ~10 min).
"""

import json
import pathlib
import subprocess

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "data/processed/transitions"
DST = REPO_ROOT / "data/processed/transitions_std121"
FFMPEG = (
    "/projects/illinois/eng/cs/jrehg/users/emirkisa/LTX-2-official/.venv/lib/"
    "python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
)

W, H, F, FPS = 480, 640, 121, 24
MIN_DIM = 480

SELECTED = {
    "color_rain": "onesided_transitions/onesided_object_color-rain",
    "cotton_cloud": "onesided_transitions/onesided_object_cotton-cloud",
    "earth_element": "onesided_transitions/onesided_style_camera_earth-element",
    "live_concert": "onesided_transitions/onesided_object_camera_live-concert",
    "luminous_gaze": "onesided_transitions/onesided_style_object_luminous-gaze",
    "monstrosity": "onesided_transitions/onesided_object-monstrosity",
    "mystification": "onesided_transitions/onesided_object_mystification",
    "nature_bloom": "onesided_transitions/onesided_object_nature-bloom",
    "polygon": "onesided_transitions/onesided_style_camera_polygon",
    "saint_glow": "onesided_transitions/onesided_object_saint-glow",
    "sakura_petals": "onesided_transitions/onesided_object_sakura-petals",
    "water_element": "onesided_transitions/onesided_style_water-element",
    "wonderland": "onesided_transitions/onesided_style_wonderland",
    "run_set_on_fire": "onesided_transitions/onesided_style_camera_run-set-on-fire",
}

EXCLUDE = {  # near-dup twins (exp_057 dedup_report.md; twin kept)
    "sakura_petals_1", "wonderland_1",
}


def frame_indices(n: int) -> np.ndarray:
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
    wanted = set(frame_indices(n).tolist())

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
            out = resize_crop(frame)[:, :, ::-1]
            proc.stdin.write(np.ascontiguousarray(out).tobytes())
            written += 1
    cap.release()
    proc.stdin.close()
    assert proc.wait() == 0, f"ffmpeg failed for {src}"
    assert written == F, f"{src}: wrote {written}/{F}"
    tmp.rename(dst)
    return {"src": str(src.relative_to(REPO_ROOT)), "src_frames": n}


def main():
    manifest, skipped = {}, []
    for cls, label_dir in SELECTED.items():
        for src in sorted((SRC / label_dir).glob("*.mp4")):
            if src.stem in EXCLUDE:
                skipped.append(f"{src.stem} (near-dup)")
                continue
            cap = cv2.VideoCapture(str(src))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if min(w, h) < MIN_DIM:
                skipped.append(f"{src.stem} ({w}x{h})")
                continue
            if n < F:
                skipped.append(f"{src.stem} ({n}f)")
                continue
            dst = DST / cls / src.name
            rel = f"{cls}/{src.name}"
            if dst.exists():
                print(f"[skip] {rel}")
                continue
            manifest[rel] = standardize(src, dst)
            print(f"[done] {rel} (src {manifest[rel]['src_frames']}f)")

    mpath = DST / "standardize_manifest.json"
    existing = json.loads(mpath.read_text()) if mpath.exists() else {}
    existing.update(manifest)
    mpath.write_text(json.dumps(existing, indent=2))
    print(f"[done] +{len(manifest)} clips (skipped {len(skipped)}: {skipped})")


if __name__ == "__main__":
    main()
