#!/usr/bin/env python
"""exp_057 — per-class montages (rows=clips, cols=6 frames) for taxonomy verification."""
import sys
import cv2
import numpy as np
from pathlib import Path

SRC = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research/data/processed/transitions")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "montages"
OUT.mkdir(exist_ok=True, parents=True)

THUMB_W = 200
N_COLS = 6

def frames_of(path, n=N_COLS):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(0, total - 1, n).round().astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            fr = np.zeros((90, 160, 3), np.uint8)
        h, w = fr.shape[:2]
        fr = cv2.resize(fr, (THUMB_W, int(h * THUMB_W / w)))
        out.append(fr)
    cap.release()
    # pad to common height
    hmax = max(f.shape[0] for f in out)
    out = [cv2.copyMakeBorder(f, 0, hmax - f.shape[0], 0, 0, cv2.BORDER_CONSTANT) for f in out]
    return np.hstack(out)

def montage(class_dir: Path, max_clips=8):
    rows = []
    clips = sorted(class_dir.glob("*.mp4"))[:max_clips]
    for c in clips:
        strip = frames_of(c)
        label = np.zeros((22, strip.shape[1], 3), np.uint8)
        cv2.putText(label, c.name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(np.vstack([label, strip]))
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    return np.vstack(rows)

CLASSES = sys.argv[2:] if len(sys.argv) > 2 else []
for cd in CLASSES:
    d = next(SRC.glob(f"*sided_transitions/*{cd}*"))
    img = montage(d)
    p = OUT / f"{d.name}.jpg"
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(p, img.shape)
