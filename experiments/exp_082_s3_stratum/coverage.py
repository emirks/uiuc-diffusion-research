"""S3 COVERAGE metric — the M1 analogue the 3D stratum was missing.

WHY. The 63-clip pilot came back 23/63 BAD on blind visual audit while every gate stayed green:
`join_ratio` caught only 2 of the 23, and BAD clips actually had a LOWER mean join_max (1.20)
than GOOD ones (1.34). join_ratio measures temporal continuity across the two handoffs; a black
wedge that opens smoothly mid-ramp and closes smoothly is perfectly continuous. S2 has M1, which
asks of every ramp frame "does this resemble either source stream?"; S3 had no such check.

Advisor ruling: build the analogue, and "prefer the general form (fraction of ramp pixels
resembling neither source stream under local patch matching) over a pure black detector, because
your own BAD criterion includes tearing/melting, and rackfocus at 6/7 smells like a defect class
that isn't black."

WHY LOCAL, WITH SEARCH. A global correlation cannot work here: the defect is spatially local (a
corner wedge leaves 90% of the frame perfect). But a naive position-wise patch comparison cannot
work either, because a 3D camera move legitimately DISPLACES content — every patch moves. So
each render patch is matched against both source frames over a local search window: content that
merely moved is explained, content that exists nowhere in either source is not.

FLAT-PATCH HANDLING. Normalised correlation is undefined on a uniform patch (zero variance), and
both a fabricated black void and a genuinely dark region are uniform. Uniform patches are
therefore compared by MEAN LUMA agreement instead: a black void is unexplained unless the source
had something equally dark in the neighbourhood.

The threshold is calibrated on the 63 hand-labelled pilot clips and frozen BEFORE the re-pilot,
which then serves as its out-of-sample test. Pre-committed instrument bar (advisor):
**>=20/23 BAD caught at <=3/40 GOOD falsely flagged.**
"""

from __future__ import annotations

import cv2
import numpy as np

GRAY_HW = (72, 96)      # (H, W) — same 4:3 reduction S2's M1 uses
PATCH = 8               # -> 9 x 12 = 108 non-overlapping patches
SEARCH = 6              # +/- px at 96x72; ~1/8 of frame width, covers realistic camera excursion
FLAT_STD = 4.0          # a patch with std below this is "uniform" and compared by mean luma
NCC_OK = 0.55           # a textured patch is explained if its best NCC reaches this
LUMA_OK = 12.0          # a uniform patch is explained if some source patch is within this luma
N_RAMP_SAMPLES = 12     # ramp frames scored per clip


def _to_gray(frames: np.ndarray) -> np.ndarray:
    h, w = GRAY_HW
    out = np.empty((len(frames), h, w), np.float32)
    for i, f in enumerate(frames):
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        out[i] = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
    return out


def _blocks(img: np.ndarray) -> np.ndarray:
    """(H,W) -> (nH, nW, PATCH*PATCH) non-overlapping patches."""
    h, w = img.shape
    nh, nw = h // PATCH, w // PATCH
    return (img[: nh * PATCH, : nw * PATCH]
            .reshape(nh, PATCH, nw, PATCH).transpose(0, 2, 1, 3).reshape(nh, nw, -1))


def _explained(render: np.ndarray, srcs: list[np.ndarray]) -> np.ndarray:
    """Per-patch: is this render patch explained by SOME source patch within +/-SEARCH?

    Returns a (nH, nW) boolean array.
    """
    R = _blocks(render)                                   # (nh, nw, P)
    r_mean = R.mean(-1)
    r_std = R.std(-1)
    r_flat = r_std < FLAT_STD
    Rc = R - r_mean[..., None]
    Rn = np.linalg.norm(Rc, axis=-1) + 1e-6

    best_ncc = np.full(r_mean.shape, -1.0, np.float32)
    best_dluma = np.full(r_mean.shape, 1e9, np.float32)

    for src in srcs:
        for dy in range(-SEARCH, SEARCH + 1):
            for dx in range(-SEARCH, SEARCH + 1):
                sh = np.roll(np.roll(src, dy, axis=0), dx, axis=1)
                S = _blocks(sh)
                s_mean = S.mean(-1)
                Sc = S - s_mean[..., None]
                Sn = np.linalg.norm(Sc, axis=-1) + 1e-6
                ncc = (Rc * Sc).sum(-1) / (Rn * Sn)
                np.maximum(best_ncc, ncc, out=best_ncc)
                np.minimum(best_dluma, np.abs(r_mean - s_mean), out=best_dluma)

    textured_ok = (~r_flat) & (best_ncc >= NCC_OK)
    flat_ok = r_flat & (best_dluma <= LUMA_OK)
    return textured_ok | flat_ok


def coverage(clip: np.ndarray, a_src: np.ndarray, b_src: np.ndarray,
             onset: int, release: int) -> dict:
    """Fraction of ramp patches explained by neither source stream.

    `unexplained_p95` is the headline: the 95th percentile over sampled ramp frames of the
    per-frame unexplained-patch fraction. A percentile rather than a mean because the defect is
    concentrated in a few frames near peak camera excursion, and a mean dilutes it away.
    """
    lo, hi = onset + 1, release
    if hi - lo < 2:
        return {"unexplained_p95": 0.0, "unexplained_max": 0.0, "unexplained_mean": 0.0,
                "n_frames": 0, "worst_frame": -1}
    idx = np.unique(np.linspace(lo, hi - 1, min(N_RAMP_SAMPLES, hi - lo)).astype(int))
    g_clip = _to_gray(clip[idx])
    g_a = _to_gray(a_src[idx])
    g_b = _to_gray(b_src[idx])

    frac = np.empty(len(idx), np.float32)
    for k in range(len(idx)):
        ok = _explained(g_clip[k], [g_a[k], g_b[k]])
        frac[k] = 1.0 - ok.mean()
    return {"unexplained_p95": float(np.percentile(frac, 95)),
            "unexplained_max": float(frac.max()),
            "unexplained_mean": float(frac.mean()),
            "n_frames": int(len(idx)),
            "worst_frame": int(idx[int(np.argmax(frac))]),
            "per_frame": [round(float(v), 4) for v in frac]}
