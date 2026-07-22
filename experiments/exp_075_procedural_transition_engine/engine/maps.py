"""Procedural greyscale maps for the shaders that take an auxiliary sampler.

`luma.glsl` reveals the target wherever the map is darker than progress, and
`displacement.glsl` pushes UVs along the map's intensity. Both turn one shader
into a whole family: every distinct map is a visually distinct operator, so this
module is a cheap multiplier on the bank size.
"""

from __future__ import annotations

import numpy as np


def _fbm(h: int, w: int, rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    """Value-noise fBm in [0,1] — smooth blobby reveal masks."""
    acc = np.zeros((h, w), np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        res = 2 ** (o + 1)
        base = rng.random((res + 1, res + 1)).astype(np.float32)
        yi = np.linspace(0, res, h, dtype=np.float32)
        xi = np.linspace(0, res, w, dtype=np.float32)
        y0, x0 = np.floor(yi).astype(int), np.floor(xi).astype(int)
        fy, fx = (yi - y0)[:, None], (xi - x0)[None, :]
        sy, sx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        c00 = base[np.ix_(y0, x0)]
        c01 = base[np.ix_(y0, np.minimum(x0 + 1, res))]
        c10 = base[np.ix_(np.minimum(y0 + 1, res), x0)]
        c11 = base[np.ix_(np.minimum(y0 + 1, res), np.minimum(x0 + 1, res))]
        acc += amp * ((c00 * (1 - sx) + c01 * sx) * (1 - sy)
                      + (c10 * (1 - sx) + c11 * sx) * sy)
        norm += amp
        amp *= 0.5
    return acc / norm


def _grid(h, w):
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    return y, x


def make_map(kind: str, h: int, w: int, seed: int) -> np.ndarray:
    """Return an HxWx3 uint8 map. `kind` selects the family."""
    rng = np.random.default_rng(seed)
    y, x = _grid(h, w)

    if kind == "fbm":
        m = _fbm(h, w, rng, octaves=int(rng.integers(3, 7)))
    elif kind == "radial":
        cy, cx = rng.uniform(0.2, 0.8, 2)
        m = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    elif kind == "linear":
        th = rng.uniform(0, 2 * np.pi)
        m = np.cos(th) * x + np.sin(th) * y
    elif kind == "stripes":
        th, f = rng.uniform(0, np.pi), rng.uniform(3, 25)
        m = 0.5 + 0.5 * np.sin(2 * np.pi * f * (np.cos(th) * x + np.sin(th) * y))
    elif kind == "checker":
        n = int(rng.integers(3, 14))
        m = ((np.floor(y * n) + np.floor(x * n)) % 2).astype(np.float32)
        m = 0.15 * _fbm(h, w, rng, 3) + 0.85 * m       # break perfect ties
    elif kind == "spiral":
        cy, cx = rng.uniform(0.3, 0.7, 2)
        ang = np.arctan2(y - cy, x - cx)
        rad = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        m = 0.5 + 0.5 * np.sin(rng.uniform(2, 8) * ang + rng.uniform(6, 25) * rad)
    elif kind == "voronoi":
        n = int(rng.integers(6, 40))
        pts = rng.random((n, 2)).astype(np.float32)
        d = np.stack([np.sqrt((y - py) ** 2 + (x - px) ** 2) for py, px in pts])
        m = d.min(0)
    else:
        raise ValueError(kind)

    m = m - m.min()
    m = m / (m.max() + 1e-8)
    if rng.random() < 0.5:
        m = 1.0 - m
    return np.repeat((m * 255).astype(np.uint8)[..., None], 3, axis=2)


MAP_KINDS = ("fbm", "radial", "linear", "stripes", "checker", "spiral", "voronoi")
