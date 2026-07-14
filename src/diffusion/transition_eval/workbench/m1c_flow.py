"""M1c_flow — object metric on the flow residual (RUNBOOK §3.3).

    "Residual: flow - fitted camera field, inside the effect region, core frames
     only. Energy gate first: mean |residual| < epsilon -> frame undefined. Set
     epsilon from the corpus residual-magnitude distribution (report chosen
     percentile). This is the designed kill for the polygon sink: near-static
     residuals must exit as undefined, not collapse to a shared descriptor.
     Descriptor per frame: 8-bin magnitude-weighted orientation histogram on a 3x3
     grid + mean divergence + mean curl + normalized magnitude. Magnitudes
     normalized by image diagonal (resolution must not leak). Clip descriptor:
     per-frame vectors -> resample 32 -> z-score -> L2."

THE ENERGY GATE IS THE WHOLE POINT. The incumbent M1c died of a hub: near-zero
residual descriptors all look alike, so every clip's nearest neighbour became
polygon (58% of the corpus retrieved polygon; one clip absorbed 201 of 2230 k-NN
slots). A near-static residual carries no object motion — it is NOISE, and
normalizing noise to unit length makes it look like a confident descriptor
pointing in an arbitrary direction. The gate makes those frames exit as UNDEFINED
(NaN, which the frozen kernel already treats as "cannot retrieve") instead of
collapsing onto a shared sink. Whether it worked is decided by the §1.4 hubness
gate, not by us.

"INSIDE THE EFFECT REGION" — no spatial mask exists. S is a TEMPORAL mask (it
selects frames; the certified cache holds CLS tokens, no patch tokens, no spatial
effect masks). So the effect region is not handed to us; it is what the camera fit
REJECTS. Operationally the residual is computed over the whole frame and the
descriptor is MAGNITUDE-WEIGHTED, so pixels the camera explains (residual ~ 0)
contribute ~nothing and the effect region dominates by construction. The literal
outlier set (residual > the Huber delta) is recorded as a diagnostic area
fraction, never used to gate.

Everything is normalized by the image diagonal, so resolution cannot leak into a
descriptor and make 432x320 flow incomparable to any other size.
"""

from __future__ import annotations

import numpy as np

from . import curves, m1b_flow

N_BINS = 8
GRID = (3, 3)
N_RESAMPLE = 32
MAX_UNDEFINED_CORE = 0.30    # inherited from §3.2 — the only stated clip-level rule


def camera_field(params: np.ndarray, h: int, w: int) -> np.ndarray:
    """Render the fitted similarity as a dense flow field [H, W, 2].

    The inverse of m1b_flow's parameterization: p -> s*R(theta)*p + t, flow = p' - p."""
    tx, ty, log_s, theta = params
    s = np.exp(log_s)
    ys, xs = np.mgrid[0:h, 0:w]
    x = xs.astype(np.float64) - (w - 1) / 2.0
    y = ys.astype(np.float64) - (h - 1) / 2.0
    a, b = s * np.cos(theta), s * np.sin(theta)
    u = a * x - b * y + tx - x
    v = b * x + a * y + ty - y
    return np.stack([u, v], axis=-1)


def residual_field(flow: np.ndarray, params: np.ndarray) -> np.ndarray:
    """flow - fitted camera field, for one frame pair."""
    h, w = flow.shape[:2]
    return flow.astype(np.float64) - camera_field(params, h, w)


def frame_energy(res: np.ndarray, diag: float) -> float:
    """mean |residual|, normalized by the image diagonal (§3.3)."""
    return float(np.linalg.norm(res, axis=-1).mean() / diag)


def frame_descriptor(res: np.ndarray, diag: float) -> np.ndarray:
    """One frame -> [3*3*8 + 3] = 75-d vector (§3.3).

    Orientation histogram is MAGNITUDE-WEIGHTED, so a pixel the camera model
    already explains contributes nothing: the effect region selects itself."""
    h, w = res.shape[:2]
    u, v = res[..., 0], res[..., 1]
    mag = np.hypot(u, v) / diag                     # resolution cannot leak
    ang = np.arctan2(v, u)                          # [-pi, pi]
    b = np.floor((ang + np.pi) / (2 * np.pi) * N_BINS).astype(int) % N_BINS

    gh, gw = GRID
    hist = np.zeros((gh, gw, N_BINS))
    rows = np.linspace(0, h, gh + 1).astype(int)
    cols = np.linspace(0, w, gw + 1).astype(int)
    for i in range(gh):
        for j in range(gw):
            m = mag[rows[i]:rows[i + 1], cols[j]:cols[j + 1]]
            bb = b[rows[i]:rows[i + 1], cols[j]:cols[j + 1]]
            hist[i, j] = np.bincount(bb.ravel(), weights=m.ravel(), minlength=N_BINS)

    # integral quantities only — no curvature/torsion (§1.3: noise amplifiers)
    dudx = np.gradient(u, axis=1)
    dudy = np.gradient(u, axis=0)
    dvdx = np.gradient(v, axis=1)
    dvdy = np.gradient(v, axis=0)
    div = float((dudx + dvdy).mean() / diag)
    curl = float((dvdx - dudy).mean() / diag)
    return np.concatenate([hist.ravel(), [div, curl, float(mag.mean())]])


def clip_residual_stats(flow: np.ndarray, traj: dict) -> dict:
    """Per-frame residual energy for one clip — the input to the corpus
    calibration of epsilon (§3.3). Undefined camera fits give NaN energy, never 0:
    a frame whose camera could not be fitted has no meaningful residual."""
    T = len(flow)
    diag = float(np.hypot(flow.shape[1], flow.shape[2]))
    energy = np.full(T, np.nan)
    for i in range(T):
        if not traj["defined"][i]:
            continue
        energy[i] = frame_energy(residual_field(flow[i], traj["params"][i]), diag)
    return {"energy": energy, "diag": diag}


def clip_curve(flow: np.ndarray, traj: dict, core_pairs: np.ndarray,
               eps: float) -> dict:
    """§3.3 clip descriptor, with the energy gate applied FIRST.

    A frame is undefined if its camera fit failed OR its residual energy is below
    epsilon. The clip is undefined if too much of its core is undefined — the
    designed exit for near-static clips."""
    T = len(flow)
    diag = float(np.hypot(flow.shape[1], flow.shape[2]))
    idx = np.flatnonzero(core_pairs[:T])
    if idx.size == 0:
        return {"curve": None, "defined": False, "undefined_frac": 1.0,
                "reason": "no core frame pairs", "n_gated": 0}

    vecs, ok, gated = [], [], 0
    for i in idx:
        if not traj["defined"][i]:
            ok.append(False)
            continue
        res = residual_field(flow[i], traj["params"][i])
        if frame_energy(res, diag) < eps:       # ENERGY GATE — first, before any descriptor
            ok.append(False)
            gated += 1
            continue
        vecs.append(frame_descriptor(res, diag))
        ok.append(True)

    ok = np.array(ok)
    undef = float(1.0 - ok.mean())
    if undef > MAX_UNDEFINED_CORE or len(vecs) < 2:
        return {"curve": None, "defined": False, "undefined_frac": undef,
                "reason": (f"{undef:.0%} of core pairs undefined "
                           f"({gated} energy-gated) > {MAX_UNDEFINED_CORE:.0%}"),
                "n_gated": gated}
    return {"curve": curves.resample(np.stack(vecs), N_RESAMPLE), "defined": True,
            "undefined_frac": undef, "reason": None, "n_gated": gated}


def effect_area_fraction(res: np.ndarray, delta: float = m1b_flow.HUBER_DELTA_PX) -> float:
    """Diagnostic only (never gates): the fraction of pixels the camera fit calls
    an outlier — the literal "effect region" §3.3 alludes to."""
    return float(np.mean(np.linalg.norm(res, axis=-1) > delta))


def corpus_descriptors(per_clip: list[dict]) -> list[np.ndarray | None]:
    """Corpus-frozen per-channel z-scoring (§1.3), identical for every clip."""
    defined = [d["curve"] for d in per_clip if d["defined"]]
    if not defined:
        return [None] * len(per_clip)
    scaler = curves.fit_channel_scaler(defined)
    return [curves.zscore(d["curve"], scaler) if d["defined"] else None
            for d in per_clip]
