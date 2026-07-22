"""Parameterised virtual-camera trajectories for the 2.5D transition renderer.

The realism trick is that the two layers ride **one continuous camera trajectory**,
not two separate effects. Layer A is rendered from the first part of the path and
layer B from the same path shifted so that it comes to rest exactly at the end.
The eye reads a single camera flying out of one scene and into the next, rather
than two clips being blended.

Every path is the identity at excursion s = 0, which is what keeps the
conditioning buckets bit-accurate.
"""

from __future__ import annotations

import numpy as np


def _I() -> np.ndarray:
    return np.eye(4, dtype="f4")


def _T(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    m = _I()
    m[:3, 3] = (x, y, z)
    return m


def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    m = _I(); m[1, 1] = c; m[1, 2] = -s; m[2, 1] = s; m[2, 2] = c
    return m


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    m = _I(); m[0, 0] = c; m[0, 2] = s; m[2, 0] = -s; m[2, 2] = c
    return m


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    m = _I(); m[0, 0] = c; m[0, 1] = -s; m[1, 0] = s; m[1, 1] = c
    return m


# --------------------------------------------------------------------------
# Paths: excursion s (arbitrary units, 0 = at rest) -> (view matrix, shear)
# --------------------------------------------------------------------------

def dolly(s: float, **_):
    """Push the camera along its own axis — the classic fly-through."""
    return _T(0.0, 0.0, s), (0.0, 0.0)


def truck(s: float, *, axis: int = 0, **_):
    """Slide the camera sideways/vertically. Pure parallax, no rotation."""
    return (_T(-s, 0.0, 0.0) if axis == 0 else _T(0.0, -s, 0.0)), (0.0, 0.0)


def orbit(s: float, *, pivot: float = 2.0, **_):
    """Arc horizontally around a point at depth `pivot`."""
    return _T(0, 0, -pivot) @ _Ry(s) @ _T(0, 0, pivot), (0.0, 0.0)


def crane(s: float, *, pivot: float = 2.0, **_):
    """Arc vertically around a point at depth `pivot`."""
    return _T(0, 0, -pivot) @ _Rx(s) @ _T(0, 0, pivot), (0.0, 0.0)


def roll(s: float, **_):
    """Rotate about the optical axis."""
    return _Rz(s), (0.0, 0.0)


def shear(s: float, *, axis: int = 0, pivot: float = 2.0, **_):
    """Depth-dependent lateral offset — a stereo-like swing with no rotation."""
    return _I(), ((s, 0.0) if axis == 0 else (0.0, s))


def spiral(s: float, *, pivot: float = 2.0, turns: float = 1.0, **_):
    """Dolly forward while orbiting — the standard NeRF-eval 'spiral' trajectory."""
    v = _T(0, 0, -pivot) @ _Ry(s * turns) @ _T(0, 0, pivot) @ _T(0, 0, s)
    return v, (0.0, 0.0)


PATHS = {
    "dolly": dolly, "truck": truck, "orbit": orbit, "crane": crane,
    "roll": roll, "shear": shear, "spiral": spiral,
}

# Sensible excursion scales per path, so an "amplitude" multiplier means the same
# thing everywhere: translations in scene units, rotations in radians.
PATH_SCALE = {
    "dolly": 0.9, "truck": 0.55, "orbit": 0.42, "crane": 0.30,
    "roll": 0.35, "shear": 0.30, "spiral": 0.55,
}


def handheld(seed: int, n: int, amount: float) -> np.ndarray:
    """Smooth 6-DoF jitter — a perfectly rigid virtual camera reads as CGI.

    Low-frequency band-limited noise, zeroed at both ends so it never disturbs
    the conditioning buckets.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    out = np.zeros((n, 6), dtype=np.float64)
    for k in range(6):
        for freq, amp in ((1.3, 1.0), (2.7, 0.45), (5.1, 0.2)):
            out[:, k] += amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    out *= amount * np.sin(np.pi * t)[:, None] ** 2      # taper to zero at both ends
    out[:, :3] *= 0.02                                   # translation units
    out[:, 3:] *= 0.012                                  # radians
    return out


def apply_handheld(view: np.ndarray, j: np.ndarray) -> np.ndarray:
    return (_T(*j[:3]) @ _Rx(j[3]) @ _Ry(j[4]) @ _Rz(j[5]) @ view).astype("f4")
