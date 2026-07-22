"""Turn 9-frame endpoint clips into full-length layer streams, plus easing ramps.

The task contract is: a 121-frame clip whose first 9 frames are the `start9`
endpoint and whose last 9 frames are the `end9` endpoint. A procedural operator
composites two *streams* — the "from" layer and the "to" layer — so both layers
must exist for all 121 frames even though we are only given 9 frames of each.

How the 9 given frames are extended matters. `hold` freezes them, which teaches
the model that everything stops during a transition — exactly the wrong prior.
`boomerang` and `flow` keep both layers alive underneath the effect.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Layer extension policies
# --------------------------------------------------------------------------


def _boomerang_indices(n: int, total: int) -> list[int]:
    """0,1,..,n-1,n-2,..,0,1,.. — ping-pong without repeating the turn frames."""
    if n == 1:
        return [0] * total
    period = list(range(n)) + list(range(n - 2, 0, -1))
    return [period[i % len(period)] for i in range(total)]


def _flow_extend(clip: np.ndarray, total: int, *, decay: float = 0.97,
                 forward: bool = True) -> np.ndarray:
    """Extrapolate motion past the clip by re-applying its mean terminal flow."""
    import cv2

    n = len(clip)
    if forward:
        a, b, seed = clip[-2], clip[-1], clip[-1]
    else:
        a, b, seed = clip[1], clip[0], clip[0]
    g0 = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3, 5, 1.2, 0)

    h, w = seed.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out, cur, amp = [], seed, 1.0
    for _ in range(total - n):
        amp *= decay
        mx = (xx - flow[..., 0] * amp).astype(np.float32)
        my = (yy - flow[..., 1] * amp).astype(np.float32)
        cur = cv2.remap(cur, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out.append(cur)
    return np.stack(out) if out else np.empty((0, h, w, 3), np.uint8)


def build_from_stream(start9: np.ndarray, total: int, policy: str) -> np.ndarray:
    """Stream whose first len(start9) frames are `start9` verbatim."""
    n = len(start9)
    if policy == "hold":
        tail = np.repeat(start9[-1:], total - n, axis=0)
    elif policy == "flow":
        tail = _flow_extend(start9, total, forward=True)
    else:                                              # boomerang (default)
        idx = _boomerang_indices(n, total)[n:]
        tail = start9[idx]
    return np.concatenate([start9, tail], axis=0)[:total]


def build_to_stream(end9: np.ndarray, total: int, policy: str) -> np.ndarray:
    """Stream whose last len(end9) frames are `end9` verbatim."""
    rev = build_from_stream(end9[::-1].copy(), total, policy)
    return rev[::-1].copy()


# --------------------------------------------------------------------------
# Easing
# --------------------------------------------------------------------------

def _clamp01(u):
    return np.clip(u, 0.0, 1.0)


EASINGS = {
    "linear":        lambda u: u,
    "smoothstep":    lambda u: u * u * (3 - 2 * u),
    "smootherstep":  lambda u: u ** 3 * (u * (6 * u - 15) + 10),
    "in_cubic":      lambda u: u ** 3,
    "out_cubic":     lambda u: 1 - (1 - u) ** 3,
    "in_out_cubic":  lambda u: np.where(u < 0.5, 4 * u ** 3,
                                        1 - (-2 * u + 2) ** 3 / 2),
    "in_out_sine":   lambda u: -(np.cos(np.pi * u) - 1) / 2,
    "in_expo":       lambda u: np.where(u <= 0, 0.0, 2 ** (10 * u - 10)),
    "out_expo":      lambda u: np.where(u >= 1, 1.0, 1 - 2 ** (-10 * u)),
    # snap: almost nothing happens, then the effect fires late
    "snap_late":     lambda u: _clamp01((u - 0.45) / 0.55) ** 2,
    # snap_early: the effect fires immediately, then settles
    "snap_early":    lambda u: 1 - _clamp01((0.55 - u) / 0.55) ** 2,
    # mid_hold: fast in, pause at half-effect, fast out — a very common editorial beat
    "mid_hold":      lambda u: np.where(u < 0.35, _clamp01(u / 0.35) * 0.5,
                                np.where(u < 0.65, 0.5,
                                         0.5 + _clamp01((u - 0.65) / 0.35) * 0.5)),
}


def progress_ramp(total: int, n_start: int, n_end: int, easing: str) -> np.ndarray:
    """Per-frame progress, pinned to 0 across the start9 block and 1 across end9.

    Pinning is what guarantees the rendered clip reproduces the conditioning
    frames bit-for-bit (for any shader that honours the p=0/p=1 identities).
    """
    t = np.arange(total, dtype=np.float64)
    t0, t1 = n_start - 1, total - n_end          # 8 and 112 for 9/121/9
    u = _clamp01((t - t0) / (t1 - t0))
    p = np.asarray(EASINGS[easing](u), dtype=np.float64)
    p[: t0 + 1] = 0.0
    p[t1:] = 1.0
    return _clamp01(p)
