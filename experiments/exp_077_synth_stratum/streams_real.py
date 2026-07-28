"""exp_077 D2 — REAL layer streams, D2 timing, D2 easings, D2 (plain-only) shader bank.

WHY THIS MODULE EXISTS (the D1 negative result)
-----------------------------------------------
D1 fed the renderer 9-frame endpoint snippets and FABRICATED the other ~112 frames of each
layer with an "extension policy" (`boomerang` = motion runs backward, `hold` = freeze,
`flow` = Farneback flow re-applied with decay 0.97 => error compounds => the frame melts).
D1 used `flow` and was rejected on visual review: "endpoints are nice but the transitions
are NOT — a big proportion completely breaks the scene."

The endpoint bank clips are FULL 121-frame standardised clips. The renderer was inventing
frames we already have. D2 therefore uses the REAL frames:

    A-layer = clip A frames t=0..120        (verbatim)
    B-layer = clip B frames t=0..120        (verbatim)

played in LOCKSTEP. There is NO extension policy in this path — the functions that could
fabricate frames (`streams.build_from_stream` / `build_to_stream`) are never called from
here, and metadata records `extension: "none"`.

Pinned anchors fall out for free: progress is pinned 0 for t <= onset and 1 for t >= release,
with onset >= 8 and release <= 112, so output[0:9] == A[0:9] and output[112:121] == B[112:121]
for any shader that honours the p=0/p=1 identities (which the gate enforces).

Nothing in exp_075's engine is modified; it is imported read-only through the `engine` symlink.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np

from engine import operators, shaders, streams
from engine.glrunner import GLRunner

# --------------------------------------------------------------------------
# D2 policy constants (LOCKED SPEC)
# --------------------------------------------------------------------------
EXTENSION = "none"                 # no fabrication anywhere in the D2 path

# 12 engine easings minus the two that manufacture near-cuts
DROPPED_EASINGS = ("snap_early", "snap_late")

# shaders whose p=0/p=1 identity is broken by construction (notes/dataset/procedural_operators.md)
HARD_BLACKLIST = ("tangentMotionBlur", "AdvancedMosaic", "InvertedPageCurl")

TIMING_FRAC = 0.20                 # onset/release each drawn in the first/last 20% of the window


def d2_easings() -> list[str]:
    return sorted(set(streams.EASINGS) - set(DROPPED_EASINGS))


def d2_shader_bank(runner: GLRunner, bank_dir, *, tol: float, holdout: list[str]
                   ) -> tuple[dict[str, shaders.Shader], dict]:
    """Gate-passing PLAIN shaders only: no aux-sampler shaders, no blacklist, no holdout.

    D2 has ZERO aux maps, so `luma` / `displacement` (the two shaders that need an auxiliary
    image sampler) are dropped along with the hard blacklist and D1's 8 held-out shaders.
    """
    raw = shaders.load_bank(bank_dir)
    gated, report = operators.validate_bank(runner, raw, tol=tol)
    aux = sorted(set(gated) & set(shaders.AUX_SAMPLER_SHADERS))
    black = sorted(set(gated) & set(HARD_BLACKLIST))
    held = sorted(set(gated) & set(holdout))
    drop = set(aux) | set(black) | set(held)
    bank = {k: v for k, v in gated.items() if k not in drop}
    info = {
        "n_parsed": len(raw),
        "n_gate1_pass": len(gated),
        "dropped_aux_sampler": aux,
        "dropped_hard_blacklist": black,
        "dropped_holdout": held,
        "n_d2_bank": len(bank),
        "gate1_tol": tol,
        "gate1_report": report,
    }
    return bank, info


# --------------------------------------------------------------------------
# Timing: bounded continuous manifold
# --------------------------------------------------------------------------
def sample_timing(rng: random.Random, total: int, anchor: int,
                  frac: float = TIMING_FRAC) -> dict[str, float]:
    """onset = t0 + u1*frac*window ; release = t1 - u2*frac*window ; u1,u2 ~ U[0,1] INDEPENDENT.

    With total=121, anchor=9, frac=0.20: window = [8, 112] (len 104), onset in [8, 28.8],
    release in [91.2, 112] => duration >= 62.4 frames and each pure phase <= ~20.8 frames.
    """
    t0, t1 = anchor - 1, total - anchor
    win = t1 - t0
    u1, u2 = rng.random(), rng.random()
    onset = t0 + u1 * frac * win
    release = t1 - u2 * frac * win
    return {"onset": onset, "release": release, "duration": release - onset,
            "u1": u1, "u2": u2, "window": [t0, t1], "frac": frac, "num_frames": total}


def progress_ramp(total: int, anchor: int, easing: str,
                  onset: float, release: float) -> np.ndarray:
    """Eased 0->1 across [onset, release], pinned 0 before onset and 1 after release."""
    t0, t1 = anchor - 1, total - anchor
    t = np.arange(total, dtype=np.float64)
    u = np.clip((t - onset) / max(release - onset, 1e-9), 0.0, 1.0)
    p = np.asarray(streams.EASINGS[easing](u), dtype=np.float64)
    p[t <= onset] = 0.0
    p[t >= release] = 1.0
    p[: t0 + 1] = 0.0            # redundant given onset >= t0, kept as a hard guarantee
    p[t1:] = 1.0                 # redundant given release <= t1
    return np.clip(p, 0.0, 1.0)


def phase_indices(total: int, onset: float, release: float) -> tuple[int, int]:
    """(last pure-A frame index, first pure-B frame index). Ramp = range(i0+1, j0)."""
    i0 = int(np.floor(onset))
    j0 = int(np.ceil(release))
    return i0, j0


# --------------------------------------------------------------------------
# Operator sampling (plain shaders, no aux, extension="none")
# --------------------------------------------------------------------------
def _op_id(shader: str, params: dict, easing: str, flip: str, swap: bool) -> str:
    key = f"{shader}|{sorted(params.items())}|{easing}|{flip}|{swap}"
    return f"{shader}_{hashlib.sha1(key.encode()).hexdigest()[:8]}"


def make_operator(bank, rng: random.Random, shader: str, *, easings, p_flip: float,
                  p_swap: float, p_vary: float = 0.85,
                  param_filter=None) -> operators.Operator:
    """One operator draw on a FIXED shader. aux_kind is always None; extension is "none".

    `param_filter` (D2-FULL addition, 2026-07-24): optional callable
    f(shader_obj, params, rng) -> (params, events) applied to the sampled parameter dict
    before the operator is built — see param_clamp.py. Defaults to None, in which case this
    function is byte-identical to the path the 448-tuple audit validated. Any clamp events are
    attached to the returned operator as `.clamp_events` (a plain attribute, so the frozen
    engine dataclass is untouched)."""
    params = shaders.sample_params(bank[shader], rng, p_vary=p_vary)
    events: list = []
    if param_filter is not None:
        params, events = param_filter(bank[shader], params, rng)
    easing = rng.choice(easings)
    flip = rng.choice(operators.FLIPS[1:]) if rng.random() < p_flip else "none"
    swap = rng.random() < p_swap
    op = operators.Operator(
        op_id=_op_id(shader, params, easing, flip, swap), shader=shader, params=params,
        easing=easing, flip=flip, swap=swap, extension=EXTENSION,
        aux_kind=None, aux_seed=0)
    op.clamp_events = events            # plain (non-frozen, non-slots) dataclass
    return op


def sample_gated_operator(runner, bank, rng, shader, pair_frames, *, tol, easings,
                          p_flip, p_swap, max_tries: int = 20, param_filter=None):
    """Rejection-sample (at PRODUCTION resolution) until the endpoint-identity gate passes on
    EVERY (frame_a, frame_b) pair of the tuple. Returns (op, gate_res, tries) or (None, ..)."""
    last = None
    for i in range(1, max_tries + 1):
        op = make_operator(bank, rng, shader, easings=easings, p_flip=p_flip, p_swap=p_swap,
                           param_filter=param_filter)
        res = [operators.check_operator(runner, bank, op, fa, fb) for fa, fb in pair_frames]
        last = (op, res, i)
        if all(max(d0, d1) <= tol for d0, d1 in res):
            return op, res, i
    return None, last[1] if last else None, max_tries


# --------------------------------------------------------------------------
# Rendering on REAL streams
# --------------------------------------------------------------------------
def render_real(runner: GLRunner, bank, op: operators.Operator, a_stream: np.ndarray,
                b_stream: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Composite REAL A/B streams frame-by-frame under the progress ramp `p`.

    a_stream / b_stream are the source clips' own frames, verbatim, same length as `p`.
    No aux sampler is ever bound: D2 operators carry aux_kind=None.
    """
    total = len(p)
    assert len(a_stream) == len(b_stream) == total, "streams must be lockstep and full length"
    prog = runner.program(op.shader, bank[op.shader].source)
    out = np.empty_like(a_stream)
    for t in range(total):
        fa = operators._apply_flip(a_stream[t], op.flip)
        fb = operators._apply_flip(b_stream[t], op.flip)
        if op.swap:
            frame = runner.render(prog, fb, fa, 1.0 - p[t], op.params, None)
        else:
            frame = runner.render(prog, fa, fb, p[t], op.params, None)
        out[t] = operators._apply_flip(frame, op.flip)
    return out


def operator_meta(op: operators.Operator) -> dict[str, Any]:
    return {"shader": op.shader, "params": op.params, "easing": op.easing, "flip": op.flip,
            "swap": op.swap, "extension": EXTENSION, "aux_kind": None, "op_id": op.op_id}
