"""exp_077 D2 — filmstrip / contact-sheet helpers (labeling sheets + the owner sheet)."""

from __future__ import annotations

import pathlib

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

_FONT_PATH = None


def font(size: int = 18):
    global _FONT_PATH
    if _FONT_PATH is None:
        import matplotlib
        _FONT_PATH = str(pathlib.Path(matplotlib.get_data_path())
                         / "fonts" / "ttf" / "DejaVuSans.ttf")
    try:
        return PIL.ImageFont.truetype(_FONT_PATH, size)
    except Exception:
        return PIL.ImageFont.load_default()


def pick_frames(i0: int, j0: int, total: int, n_ramp: int = 8) -> list[int]:
    """0, last pure-A, n_ramp frames spread across the ramp, first pure-B, last frame."""
    ramp = [int(round(i0 + (j0 - i0) * k / (n_ramp + 1))) for k in range(1, n_ramp + 1)]
    idx = [0, i0] + ramp + [j0, total - 1]
    out = []
    for i in idx:
        i = max(0, min(total - 1, i))
        if i not in out:
            out.append(i)
    return out


def clip_block(clip: np.ndarray, idx: list[int], *, cols: int, frame_w: int,
               caption, marks: dict[int, str] | None = None,
               cap_h: int | None = None, cap_size: int = 17) -> np.ndarray:
    """A captioned grid block for ONE clip. `marks` labels individual frames (e.g. anchors).

    `caption` may be a string or a list of lines (long captions must not run off the sheet).
    """
    lines = [caption] if isinstance(caption, str) else list(caption)
    if cap_h is None:
        cap_h = 8 + len(lines) * (cap_size + 3)
    h, w = clip.shape[1:3]
    fw = frame_w
    fh = int(round(h * fw / w))
    rows = (len(idx) + cols - 1) // cols
    pad = 3
    bw = cols * fw + (cols - 1) * pad
    bh = cap_h + rows * (fh + 16 + pad)
    img = PIL.Image.new("RGB", (bw, bh), (250, 250, 250))
    dr = PIL.ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        dr.text((2, 4 + i * (cap_size + 3)), line, fill=(10, 10, 10), font=font(cap_size))
    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        x = c * (fw + pad)
        y = cap_h + r * (fh + 16 + pad)
        tile = PIL.Image.fromarray(clip[i]).resize((fw, fh), PIL.Image.LANCZOS)
        img.paste(tile, (x, y))
        tag = f"t={i}"
        col = (10, 10, 10)
        if marks and i in marks:
            tag += f"  {marks[i]}"
            col = (170, 0, 0)
        dr.text((x + 2, y + fh + 1), tag, fill=col, font=font(13))
        if marks and i in marks:
            dr.rectangle([x, y, x + fw - 1, y + fh - 1], outline=(200, 30, 30), width=3)
    return np.asarray(img)


def stack_blocks(blocks: list[np.ndarray], gap: int = 10,
                 bg: int = 235) -> np.ndarray:
    w = max(b.shape[1] for b in blocks)
    h = sum(b.shape[0] for b in blocks) + gap * (len(blocks) - 1)
    out = np.full((h, w, 3), bg, np.uint8)
    y = 0
    for b in blocks:
        out[y:y + b.shape[0], : b.shape[1]] = b
        y += b.shape[0] + gap
    return out


def save(path, arr: np.ndarray) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(arr).save(path)
