"""Border-blackness (letterbox / pillarbox) detector for endpoint clips.

WHY THIS EXISTS. Chasing black regions in the S3 pilot audit, I measured border blackness over
all 227 round-1 bank clips: **21 of 227 (9.3%) carry baked-in letterbox or pillarbox**, up to
47% of frame width and 34% of frame height. The round-1 funnel never checked for it. The
advisor ruled the 21 out of the delivery universe unconditionally:

    "A 47%-black frame is not an endpoint. Three independent grounds, any one sufficient: it is
     on the reject list I set for new clips, and grandfathering old clips to a lower bar than
     new ones is indefensible; it measurably aggravates the S3 defect (50% vs 33%); and black
     bars leak into S2 targets and pure phases, teaching the model bars-as-content that no
     inference-time user will supply."

The same detector is added to the expansion funnel so new candidates are held to the identical
bar, "threshold set to catch all 21 known bank cases with zero false positives on visually
clean clips".

ROBUSTNESS. Blackness is measured on SEVERAL frames spread across the clip and reduced with a
median, so one dark frame (a night shot, a fade, a cut to black) cannot condemn a clip — only a
border that is black *throughout* counts as a matte.
"""

from __future__ import annotations

import cv2
import numpy as np

LUMA_THR = 16          # a border row/col is "black" if its mean luma is below this
FRAME_IDX = (8, 30, 60, 90, 112)   # spread across the 121-frame contract, avoiding the very ends
REJECT_FRAC = 0.02     # reject if >2% of height OR width is a persistent black matte


def border_black(path: str, *, luma_thr: int = LUMA_THR,
                 frames: tuple[int, ...] = FRAME_IDX) -> dict:
    """Median persistent black-border fraction, vertical and horizontal."""
    cap = cv2.VideoCapture(str(path))
    v, h = [], []
    for idx in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok or f is None:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        H, W = g.shape
        rows = g.mean(axis=1) < luma_thr
        cols = g.mean(axis=0) < luma_thr
        top = int(np.argmax(~rows)) if (~rows).any() else H
        bot = int(np.argmax(~rows[::-1])) if (~rows).any() else H
        left = int(np.argmax(~cols)) if (~cols).any() else W
        right = int(np.argmax(~cols[::-1])) if (~cols).any() else W
        v.append((top + bot) / H)
        h.append((left + right) / W)
    cap.release()
    if not v:
        return {"vertical": 0.0, "horizontal": 0.0, "max": 0.0, "n_frames": 0}
    fv, fh = float(np.median(v)), float(np.median(h))
    return {"vertical": round(fv, 4), "horizontal": round(fh, 4),
            "max": round(max(fv, fh), 4), "n_frames": len(v)}


def is_letterboxed(path: str, *, reject_frac: float = REJECT_FRAC) -> tuple[bool, dict]:
    m = border_black(path)
    return m["max"] > reject_frac, m
