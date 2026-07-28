"""Arrival-time mattes with the structure plain fbm does not have.

The shipped aux-map bank (`exp_075/engine/maps.py`) is seven *stationary,
isotropic* scalar fields — fbm, radial, linear, stripes, checker, spiral,
voronoi. Ink, paint and liquid are neither stationary nor isotropic: they have a
source, they propagate, they branch, and their statistics change with distance
from the seed. A threshold sweeping across an isotropic field can only ever look
like a dissolve.

Everything here emits the same thing the existing plumbing already consumes: a
single static HxW scalar field in [0, 1]. Because the compositor sweeps a
threshold across it, that field *is* an arrival-time map

    T(x) = normalised time at which pixel x flips from `from` to `to`

so an animated matte reduces to one image — no video needed.

Convention: arrays are in IMAGE space (row 0 = top of the picture).
`aux_upload()` applies the vertical flip the `luma` sampler needs (see
`probe_orientation.py`).

Families
--------
eikonal   fast-marching front from seed points through a spatially varying speed
          field F(x). Branching, source-anchored, slows in "dense paper".
invasion  invasion percolation on a correlated resistance field — the standard
          model for ink wicking into paper / viscous fingering / frost. The
          invasion ORDER is natively an arrival time.
brush     CC0 Krita brush alphas stamped along a parametric path; the stamp
          index is written into the arrival time. This is how commercial paint
          -wipe packs are actually authored.
edge      the same stamping, but the path follows a Canny edge extracted from
          the *target* frame, then bleeds outward — a content-aware boundary
          draw, which no stock matte can do.
"""

from __future__ import annotations

import heapq
import math
import pathlib

import cv2
import numpy as np
import scipy.ndimage as ndi

# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def fbm(h: int, w: int, rng: np.random.Generator, octaves: int = 6,
        lacunarity: float = 2.0, gain: float = 0.5) -> np.ndarray:
    """Value-noise fBm in [0,1]; same construction as exp_075 but tunable."""
    acc = np.zeros((h, w), np.float32)
    amp, norm, res = 1.0, 0.0, 2.0
    for _ in range(octaves):
        r = max(2, int(round(res)))
        base = rng.random((r + 1, r + 1)).astype(np.float32)
        yi = np.linspace(0, r, h, dtype=np.float32)
        xi = np.linspace(0, r, w, dtype=np.float32)
        y0, x0 = np.floor(yi).astype(int), np.floor(xi).astype(int)
        fy, fx = (yi - y0)[:, None], (xi - x0)[None, :]
        sy, sx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        c00 = base[np.ix_(y0, x0)]
        c01 = base[np.ix_(y0, np.minimum(x0 + 1, r))]
        c10 = base[np.ix_(np.minimum(y0 + 1, r), x0)]
        c11 = base[np.ix_(np.minimum(y0 + 1, r), np.minimum(x0 + 1, r))]
        acc += amp * ((c00 * (1 - sx) + c01 * sx) * (1 - sy)
                      + (c10 * (1 - sx) + c11 * sx) * sy)
        norm += amp
        amp *= gain
        res *= lacunarity
    return acc / norm


def norm01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def to_rgb_u8(m: np.ndarray) -> np.ndarray:
    """Scalar field in [0,1] -> HxWx3 uint8, the shape `set_aux_map` wants."""
    return np.repeat((np.clip(m, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)


def aux_upload(m_rgb_u8: np.ndarray) -> np.ndarray:
    """Flip into the orientation the `luma` sampler reads (probe-verified)."""
    return np.ascontiguousarray(m_rgb_u8[::-1])


def _resize(m: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC)


def _equalise(m: np.ndarray) -> np.ndarray:
    """Histogram-flatten the arrival time so the front advances at even speed.

    Without this a map whose values pile up in one band makes the reveal stall
    and then jump. Rank-transform is the cleanest fix and preserves the level
    sets exactly — only their *timing* changes, which is the whole point of an
    arrival-time field.
    """
    flat = m.ravel()
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, np.float32)
    ranks[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    return ranks.reshape(m.shape)


# --------------------------------------------------------------------------
# 1. eikonal / fast marching
# --------------------------------------------------------------------------


def fast_march(speed: np.ndarray, seeds: list[tuple[int, int]]) -> np.ndarray:
    """First-order fast marching: solve |grad T| = 1 / F from the seed set.

    Pure numpy/heapq — no scikit-fmm on this cluster. Kept at working
    resolution (~320x240) and upsampled; the front is piecewise smooth so the
    detail lives in `speed`, not in the grid.
    """
    h, w = speed.shape
    T = np.full((h, w), np.inf, np.float64)
    known = np.zeros((h, w), bool)
    inv_f = 1.0 / np.maximum(speed.astype(np.float64), 1e-3)

    heap: list[tuple[float, int, int]] = []
    for (sy, sx) in seeds:
        T[sy, sx] = 0.0
        heapq.heappush(heap, (0.0, sy, sx))

    push = heapq.heappush
    pop = heapq.heappop
    while heap:
        t, i, j = pop(heap)
        if known[i, j] or t > T[i, j]:
            continue
        known[i, j] = True
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            a, b = i + di, j + dj
            if a < 0 or a >= h or b < 0 or b >= w or known[a, b]:
                continue
            # minimal known neighbour along each axis
            ty = min(T[a - 1, b] if a > 0 and known[a - 1, b] else np.inf,
                     T[a + 1, b] if a < h - 1 and known[a + 1, b] else np.inf)
            tx = min(T[a, b - 1] if b > 0 and known[a, b - 1] else np.inf,
                     T[a, b + 1] if b < w - 1 and known[a, b + 1] else np.inf)
            f = inv_f[a, b]
            if math.isinf(ty):
                cand = tx + f
            elif math.isinf(tx):
                cand = ty + f
            else:
                d = ty - tx
                if abs(d) < f:
                    cand = 0.5 * (ty + tx + math.sqrt(2.0 * f * f - d * d))
                else:
                    cand = min(ty, tx) + f
            if cand < T[a, b]:
                T[a, b] = cand
                push(heap, (cand, a, b))
    T[~np.isfinite(T)] = np.nanmax(T[np.isfinite(T)])
    return T.astype(np.float32)


def eikonal_map(h: int, w: int, seed: int, *, n_seeds: int = 3,
                contrast: float = 6.0, aniso: float = 0.0, aniso_axis: int = 1,
                gravity: float = 0.0, octaves: int = 8,
                ridged: float = 1.0, seed_band: tuple[float, float] = (0.02, 0.98),
                grid: int = 288) -> np.ndarray:
    """Arrival time of a front through a channelled speed field.

    contrast  speed ratio between the fast channels and the slow "paper"
    ridged    blend towards a ridged multifractal (1 - |2n-1|). Plain fbm gives
              round blobs because its fast regions are fat; the ridged form has
              thin high ridges, so the front runs *along* them and branches —
              which is the difference between a dissolve and ink.
    aniso     smear the noise along x -> elongated runs / streaks
    gravity   bias the speed downward -> drips
    """
    rng = np.random.default_rng(seed)
    gw = max(8, int(round(grid * w / h)))
    gh = grid

    n = fbm(gh, gw, rng, octaves=octaves, lacunarity=2.3, gain=0.62)
    if ridged > 0:
        n = (1.0 - ridged) * n + ridged * norm01(1.0 - np.abs(2.0 * n - 1.0)) ** 2.2
        n = norm01(n)
    if aniso > 0:
        # smear the ridges along one axis, then re-ridge: smearing alone just
        # blurs the field back into round blobs.
        k = max(2, int(aniso))
        n = norm01(ndi.uniform_filter1d(n, size=k, axis=aniso_axis, mode="reflect"))
        n = norm01(1.0 - np.abs(2.0 * n - 1.0)) ** 1.6
    speed = np.exp(contrast * (n - 0.5)).astype(np.float32)
    if gravity > 0:
        yy = np.linspace(0.0, 1.0, gh, dtype=np.float32)[:, None]
        speed = speed * (1.0 + gravity * yy) ** 2

    lo, hi = seed_band
    y0, y1 = int(lo * (gh - 4)) + 2, max(int(hi * (gh - 4)) + 2, int(lo * (gh - 4)) + 3)
    seeds = [(int(rng.integers(y0, y1)), int(rng.integers(2, gw - 2)))
             for _ in range(n_seeds)]
    T = fast_march(speed, seeds)
    return _equalise(_resize(norm01(T), h, w))


# --------------------------------------------------------------------------
# 2. invasion percolation (dielectric-breakdown / ink-in-paper family)
# --------------------------------------------------------------------------


def invasion_map(h: int, w: int, seed: int, *, n_seeds: int = 2,
                 correlation: float = 3.0, octaves: int = 6,
                 smooth: float = 1.2, grid: int = 288) -> np.ndarray:
    """Invasion-percolation arrival time on a correlated resistance field.

    Repeatedly invade the lowest-resistance site on the cluster boundary; the
    invasion ORDER is the arrival time. Low `correlation` -> dendritic frost /
    lightning; high -> fat compact fingers (viscous fingering, ink in paper).
    """
    rng = np.random.default_rng(seed)
    gw = max(8, int(round(grid * w / h)))
    gh = grid

    corr = fbm(gh, gw, rng, octaves=octaves, lacunarity=2.2, gain=0.58)
    white = rng.random((gh, gw)).astype(np.float32)
    a = correlation / (1.0 + correlation)
    r = a * corr + (1.0 - a) * white

    order = np.full((gh, gw), -1, np.int32)
    heap: list[tuple[float, int, int]] = []
    pushed = np.zeros((gh, gw), bool)
    for _ in range(n_seeds):
        sy, sx = int(rng.integers(2, gh - 2)), int(rng.integers(2, gw - 2))
        if not pushed[sy, sx]:
            pushed[sy, sx] = True
            heapq.heappush(heap, (float(r[sy, sx]), sy, sx))

    push, pop = heapq.heappush, heapq.heappop
    k = 0
    total = gh * gw
    while heap and k < total:
        _, i, j = pop(heap)
        if order[i, j] >= 0:
            continue
        order[i, j] = k
        k += 1
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            b, c = i + di, j + dj
            if 0 <= b < gh and 0 <= c < gw and order[b, c] < 0 and not pushed[b, c]:
                pushed[b, c] = True
                push(heap, (float(r[b, c]), b, c))
    order[order < 0] = k                                   # unreachable (cannot happen on a 4-connected grid)

    T = norm01(order.astype(np.float32))
    # Invasion percolation without trapping leaves late sites stranded *inside*
    # the cluster; unfiltered they read as grain, not ink. Grey opening (local
    # min then local max) deletes small LATE islands while leaving the dark
    # early branches intact — the morphological form of "the cluster swallows
    # the pockets it surrounds".
    T = ndi.grey_opening(T, size=5)
    T = ndi.median_filter(T, size=3)
    if smooth > 0:
        T = ndi.gaussian_filter(T, smooth)
    return _equalise(_resize(norm01(T), h, w))


# --------------------------------------------------------------------------
# 3. brush-stroke path stamping  (CC0 Krita alphas)
# --------------------------------------------------------------------------

BRUSH_DIR_DEFAULT = "data/raw/cc0_brush_alphas"


def load_brush(path: str | pathlib.Path) -> np.ndarray:
    """Krita brush tip PNG -> float alpha in [0,1] (dark pixels = paint)."""
    import PIL.Image

    im = PIL.Image.open(path)
    if im.mode in ("RGBA", "LA"):
        rgba = np.asarray(im.convert("RGBA"), np.float32) / 255.0
        lum = rgba[..., :3].mean(-1)
        a = (1.0 - lum) * rgba[..., 3]
    else:
        a = 1.0 - np.asarray(im.convert("L"), np.float32) / 255.0
    return np.clip(a, 0.0, 1.0)


def _stamp(canvas_t: np.ndarray, brush: np.ndarray, cy: float, cx: float,
           size: float, angle: float, t: float, thresh: float = 0.30) -> None:
    """Write arrival time `t` wherever this stamp lays down paint and the
    canvas has not been painted earlier (canvas_t is filled with +inf)."""
    bh, bw = brush.shape
    s = size / max(bh, bw)
    m = cv2.getRotationMatrix2D((bw / 2.0, bh / 2.0), angle, s)
    out_w = out_h = int(round(size * 1.5)) or 1
    m[0, 2] += out_w / 2.0 - bw / 2.0
    m[1, 2] += out_h / 2.0 - bh / 2.0
    warped = cv2.warpAffine(brush, m, (out_w, out_h), flags=cv2.INTER_LINEAR,
                            borderValue=0.0)

    H, W = canvas_t.shape
    y0, x0 = int(round(cy - out_h / 2)), int(round(cx - out_w / 2))
    ya, yb = max(0, y0), min(H, y0 + out_h)
    xa, xb = max(0, x0), min(W, x0 + out_w)
    if ya >= yb or xa >= xb:
        return
    sub = warped[ya - y0:yb - y0, xa - x0:xb - x0]
    tgt = canvas_t[ya:yb, xa:xb]
    hit = (sub > thresh) & (tgt > t)
    tgt[hit] = t


def _close_gaps(T: np.ndarray, grow: float = 0.25) -> np.ndarray:
    """Assign arrival times to pixels no stamp ever touched.

    A stroke bank never tiles the frame perfectly, and the compositor needs the
    matte to reach 1.0 everywhere or the last pixels never flip. Unpainted
    pixels inherit their nearest painted pixel's time plus a distance term, so
    holes close just behind the stroke that made them.
    """
    miss = ~np.isfinite(T)
    if not miss.any():
        return norm01(T)
    dist, (iy, ix) = ndi.distance_transform_edt(miss, return_indices=True)
    src = T[iy, ix]
    span = float(np.nanmax(T[~miss]) - np.nanmin(T[~miss]) + 1e-6)
    T = T.copy()
    T[miss] = src[miss] + grow * span * (dist[miss] / (dist.max() + 1e-6))
    return norm01(T)


def _path_points(kind: str, h: int, w: int, rng: np.random.Generator,
                 n_strokes: int) -> list[np.ndarray]:
    """Parametric stroke paths, each an (N,2) array of (y,x) in pixels."""
    paths = []
    if kind == "wipe":
        # broad diagonal strokes sweeping across the frame, slightly wavy
        th = rng.uniform(-0.5, 0.5) + (0.0 if rng.random() < 0.5 else math.pi)
        for k in range(n_strokes):
            u = np.linspace(-0.25, 1.25, 90)
            off = (k + 0.5) / n_strokes
            wob = 0.05 * np.sin(u * rng.uniform(3, 7) + rng.uniform(0, 6.28))
            x = u
            y = off + wob
            ry = math.cos(th) * (y - 0.5) - math.sin(th) * (x - 0.5) + 0.5
            rx = math.sin(th) * (y - 0.5) + math.cos(th) * (x - 0.5) + 0.5
            paths.append(np.stack([ry * h, rx * w], 1))
    elif kind == "scribble":
        for _ in range(n_strokes):
            u = np.linspace(0, 1, 80)
            cy, cx = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
            a0, a1 = rng.uniform(0, 6.28), rng.uniform(1.5, 4.5)
            rad = rng.uniform(0.15, 0.45)
            ang = a0 + a1 * u * (1 if rng.random() < 0.5 else -1)
            rr = rad * (0.4 + 0.9 * u)
            paths.append(np.stack([(cy + rr * np.sin(ang)) * h,
                                   (cx + rr * np.cos(ang)) * w], 1))
    else:                                                   # "splat"
        for _ in range(n_strokes):
            u = np.linspace(0, 1, 40)
            cy, cx = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
            ang = rng.uniform(0, 6.28)
            rr = rng.uniform(0.05, 0.55) * u ** 0.6
            paths.append(np.stack([(cy + rr * np.sin(ang)) * h,
                                   (cx + rr * np.cos(ang)) * w], 1))
    return paths


def brush_map(h: int, w: int, seed: int, brushes: list[np.ndarray], *,
              kind: str = "wipe", n_strokes: int = 7, size: float = 0.34,
              jitter: float = 0.35, spacing: float = 0.11) -> np.ndarray:
    """Stamp brush alphas along paths, writing stamp ORDER into arrival time."""
    rng = np.random.default_rng(seed)
    T = np.full((h, w), np.inf, np.float32)
    paths = _path_points(kind, h, w, rng, n_strokes)

    base = size * max(h, w)
    stamps: list[tuple[float, float, float, float, np.ndarray]] = []
    for p in paths:
        br = brushes[int(rng.integers(len(brushes)))]
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        n_st = max(2, int(seg / max(spacing * base, 1.0)))
        u = np.linspace(0, 1, n_st)
        yy = np.interp(u, np.linspace(0, 1, len(p)), p[:, 0])
        xx = np.interp(u, np.linspace(0, 1, len(p)), p[:, 1])
        for q in range(n_st):
            press = 0.55 + 0.45 * math.sin(math.pi * (q + 0.5) / n_st)   # pressure ramp
            sz = base * press * (1.0 + jitter * (rng.random() - 0.5))
            ang = float(rng.uniform(0, 360))
            dy = yy[q] + jitter * base * 0.10 * (rng.random() - 0.5)
            dx = xx[q] + jitter * base * 0.10 * (rng.random() - 0.5)
            stamps.append((dy, dx, sz, ang, br))

    n = len(stamps)
    for k, (cy, cx, sz, ang, br) in enumerate(stamps):
        _stamp(T, br, cy, cx, sz, ang, k / max(n - 1, 1))
    T = _close_gaps(T)
    # Krita tips are textured, so a stamp leaves pinholes that `_close_gaps`
    # dates very late. Opening removes those bright specks; the median tidies
    # what is left. Both keep the bristle silhouette, which is the whole point.
    T = ndi.grey_opening(T, size=5)
    T = ndi.median_filter(T, size=3)
    return _equalise(T)


def edge_draw_map(h: int, w: int, seed: int, brushes: list[np.ndarray],
                  target: np.ndarray, *, size: float = 0.075,
                  spacing: float = 0.30, bleed: float = 1.0) -> np.ndarray:
    """Content-aware boundary draw: stamp along the TARGET frame's own edges.

    Canny edges of the incoming shot are ordered along a sweep direction and
    inked in that order, then the paint bleeds outward from the drawn line. The
    reveal therefore traces the thing it is revealing — the one matte behaviour
    a stock pack physically cannot have, because a stock pack never sees the
    footage.
    """
    rng = np.random.default_rng(seed)
    # the matte lives in map space; the frame may not — resize before Canny so the
    # edge coordinates and the canvas share a coordinate system.
    if target.shape[:2] != (h, w):
        target = cv2.resize(target, (w, h), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.4)
    med = float(np.median(g))
    edges = cv2.Canny(g, int(max(0, 0.62 * med)), int(min(255, 1.35 * med)))
    ys, xs = np.nonzero(edges)
    if ys.size < 400:                                        # flat shot: loosen
        edges = cv2.Canny(g, 20, 60)
        ys, xs = np.nonzero(edges)
    if ys.size < 60:
        return eikonal_map(h, w, seed)

    th = rng.uniform(0, 2 * math.pi)
    key = math.cos(th) * (xs / w) + math.sin(th) * (ys / h)
    idx = np.argsort(key)
    ys, xs = ys[idx], xs[idx]

    base = size * max(h, w)
    # stamp every ~(spacing * base)-th edge pixel: dense enough that the nib
    # actually draws a continuous line rather than dotting it.
    stride = max(1, int(round(spacing * base)))
    T = np.full((h, w), np.inf, np.float32)
    sel = list(range(0, ys.size, stride))
    n = len(sel)
    for k, q in enumerate(sel):
        br = brushes[int(rng.integers(len(brushes)))]
        sz = base * (0.75 + 0.5 * rng.random())
        _stamp(T, br, float(ys[q]), float(xs[q]), sz, float(rng.uniform(0, 360)),
               k / max(n - 1, 1))
    T = _close_gaps(T, grow=bleed)
    return _equalise(ndi.median_filter(T, size=3))


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

# Each entry: (family, human label, builder kwargs). `target` is injected for
# the content-aware kinds.
NEW_MAP_SPECS: dict[str, dict] = {
    "eikonal_ink":     dict(fn="eikonal", label="eikonal front · ink in paper",
                            kw=dict(n_seeds=3, contrast=7.0, octaves=8)),
    "eikonal_burst":   dict(fn="eikonal", label="eikonal front · single-seed burst",
                            kw=dict(n_seeds=1, contrast=8.5, octaves=8)),
    "eikonal_streak":  dict(fn="eikonal", label="eikonal front · horizontal streaks",
                            kw=dict(n_seeds=2, contrast=8.0, aniso=26, aniso_axis=1,
                                    octaves=8)),
    "eikonal_drip":    dict(fn="eikonal", label="eikonal front · gravity drip",
                            kw=dict(n_seeds=5, contrast=7.0, aniso=22, aniso_axis=0,
                                    gravity=2.0, seed_band=(0.0, 0.12), octaves=8)),
    "invasion_ink":    dict(fn="invasion", label="invasion percolation · ink wick",
                            kw=dict(n_seeds=2, correlation=3.0, smooth=1.2)),
    "invasion_frost":  dict(fn="invasion", label="invasion percolation · dendritic frost",
                            kw=dict(n_seeds=3, correlation=1.0, smooth=0.8)),
    "invasion_finger": dict(fn="invasion", label="invasion percolation · viscous fingering",
                            kw=dict(n_seeds=1, correlation=8.0, smooth=1.6)),
    "brush_wipe":      dict(fn="brush", label="brush stamping · paint wipe",
                            kw=dict(kind="wipe", n_strokes=6, size=0.34)),
    "brush_scribble":  dict(fn="brush", label="brush stamping · scribble reveal",
                            kw=dict(kind="scribble", n_strokes=11, size=0.22)),
    "brush_splat":     dict(fn="brush", label="brush stamping · splatter",
                            kw=dict(kind="splat", n_strokes=16, size=0.20)),
    "edge_draw":       dict(fn="edge", label="content-aware boundary draw (Canny of B)",
                            kw=dict(size=0.05, spacing=0.15, bleed=1.0)),
    "edge_draw_fine":  dict(fn="edge", label="content-aware boundary draw · fine nib",
                            kw=dict(size=0.03, spacing=0.12, bleed=1.4)),
}


def build_new_map(name: str, h: int, w: int, seed: int, *,
                  brushes: list[np.ndarray] | None = None,
                  target: np.ndarray | None = None) -> np.ndarray:
    spec = NEW_MAP_SPECS[name]
    fn, kw = spec["fn"], dict(spec["kw"])
    if fn == "eikonal":
        return eikonal_map(h, w, seed, **kw)
    if fn == "invasion":
        return invasion_map(h, w, seed, **kw)
    if fn == "brush":
        assert brushes, "brush maps need CC0 brush alphas"
        return brush_map(h, w, seed, brushes, **kw)
    if fn == "edge":
        assert brushes is not None and target is not None
        return edge_draw_map(h, w, seed, brushes, target, **kw)
    raise ValueError(name)


NEW_MAP_KINDS = tuple(NEW_MAP_SPECS)
CONTENT_AWARE = tuple(k for k, v in NEW_MAP_SPECS.items() if v["fn"] == "edge")
