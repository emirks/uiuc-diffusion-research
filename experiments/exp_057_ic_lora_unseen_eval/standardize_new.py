"""exp_057 — standardize the selected unseen-class clips to the exp_056 spec.

Reads the labeled tree data/processed/transitions/{onesided,twosided}_transitions/,
writes 480x640 (WxH) x 121f @ 24fps H.264 crf14 into
data/processed/transitions_std121/<class>/<clip>.mp4 — same root as the
trained-class std clips, extending the harness reference corpus.

Selection + exclusions per experiments/exp_057_ic_lora_unseen_eval/design.md.
Idempotent (skips existing). Login-node safe (CPU, ~15 min).
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

# class -> source label dir (relative to SRC)
SELECTED = {
    "hero_flight": "onesided_transitions/onesided_camera_hero-flight",
    "super_fast_run": "onesided_transitions/onesided_camera_super-fast-run",
    "plasma_explosion": "onesided_transitions/onesided_camera_plasma-explosion",
    "shadow": "onesided_transitions/onesided_style_shadow",
    "fire_element": "onesided_transitions/onesided_style_fire-element",
    "wireframe": "onesided_transitions/onesided_style_wireframe",
    "illustration_scene": "onesided_transitions/onesided_style_camera_illustration-scene",
    "animalization": "onesided_transitions/onesided_object_animalization",
    "gas_transformation": "onesided_transitions/onesided_object_gas-transformation",
    "portal": "onesided_transitions/onesided_object_portal",
    "giant_grab": "onesided_transitions/onesided_object_giant-grab",
    "money_rain": "onesided_transitions/onesided_object_money-rain",
    "hole_transition": "twosided_transitions/twosided_object_camera_hole-transition",
    "seamless_transition": "twosided_transitions/twosided_camera_seamless-transition",
}

EXCLUDE = {  # near-dup / same-take regenerations (design.md corpus filtering)
    "giant_grab_5", "super_fast_run_11", "plasma_explosion_3",
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
                skipped.append(src.stem)
                continue
            n = int(cv2.VideoCapture(str(src)).get(cv2.CAP_PROP_FRAME_COUNT))
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
    print(f"[done] +{len(manifest)} clips (skipped: {skipped}) -> {mpath}")


if __name__ == "__main__":
    main()
