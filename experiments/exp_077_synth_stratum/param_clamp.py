"""exp_077 D2-FULL — PARAMETER CLAMP wrapper around `engine.shaders.sample_params`.

WHY (2026-07-24 ruling): the quantitative audit of the rejected D1 build found a SECOND,
substrate-independent root cause that D2 inherits verbatim — UNBOUNDED CONTINUOUS SHADER
PARAMETERS. `shaders.sample_params(p_vary=0.85)` multiplies a uniform's default by
U[0.25, 4.0] (count hints) or U[0.3, 3.0] (generic), which is how measured failures like
`rotate_scale_fade scale=21.0` (60 frames of flat orange), `EdgeTransition
edge_brightness=20.3` (black edge-map), `ColourDistance power=7.46` (psychedelic) and
`fadecolor color=...` (fade through near-black) arise. Most offenders are already in the
40-shader blacklist; EdgeTransition / fadecolor / ColourDistance are D2-KEPT and survive.

This module is a WRAPPER: exp_075's `engine/` is imported read-only and NOT modified. The
frozen gate, tau = 0.2543 and the 40-shader blacklist are UNCHANGED — narrowing the sampler's
input distribution upstream of a frozen gate only makes the gate more conservative, so no
recalibration is performed.

THE RULE (as ruled, applied to the FINAL param dict — note a default is 1.0x itself, so
rule 1 can never move a default; rules 2 and 3 deliberately can):

 1. GLOBAL RELATIVE  — every continuous param: clip to [0.33x, 3.0x] of the shader default.
                       (Kills the x4 count-hint tail; this is what stops scale=21.0.)
 2. POWER/BRIGHTNESS — name contains power|bright|gamma|expo|contrast|bloom|glow: clip to
                       [0.5x, 2.0x] of default AND |v| <= 3.0. (The relative cap alone does
                       NOT stop edge_brightness=20.3 / power=7.46; this does. Where the two
                       constraints are incompatible — default 8.0 => [4,16] vs <=3.0 — the
                       ABSOLUTE cap wins, by design: it is the safety constraint.)
 3. COLOUR           — name contains color|colour, or ANY 3-vector whose components are all in
                       [0,1]: require luma = 0.2126R + 0.7152G + 0.0722B in [0.15, 0.85].
                       Resample up to 8 times; on exhaustion fall back to the shader DEFAULT
                       colour. (Stops fade-through-near-black and flat-bgcolor fills.)
 4. LEFT ALONE       — booleans and enums. Params whose natural domain is a [0,1] position /
                       progress (center|origin|offset|position|point|progress|phase|anchor):
                       additionally clipped to [0,1].

Ints / ivecs (grid counts) get the same [0.33x, 3.0x] relative band as rule 1 — the engine
draws them at U[0.3, 3.5], so this is a marginal narrowing in the same direction; reported
separately in the clamp log as rule "relative_int".

EVERY clamp / resample is logged (shader, param, raw -> clamped, rule) for free diagnostics.
"""

from __future__ import annotations

import random
from typing import Any

# ---- name classes (case-insensitive substring match) ----------------------
POWER_HINTS = ("power", "bright", "gamma", "expo", "contrast", "bloom", "glow")
COLOR_HINTS = ("color", "colour")
POSITION_HINTS = ("center", "centre", "origin", "offset", "position", "point",
                  "progress", "phase", "anchor")

REL_LO, REL_HI = 0.33, 3.0          # rule 1
POW_LO, POW_HI = 0.5, 2.0           # rule 2 relative
POW_ABS = 3.0                       # rule 2 absolute
LUMA_LO, LUMA_HI = 0.15, 0.85       # rule 3
LUMA_TRIES = 8


def _has(name: str, hints) -> bool:
    n = name.lower()
    return any(h in n for h in hints)


def luma(rgb) -> float:
    return 0.2126 * float(rgb[0]) + 0.7152 * float(rgb[1]) + 0.0722 * float(rgb[2])


def _is_color(name: str, val, default) -> bool:
    """Named a colour, or ANY 3-vector with all components in [0,1]."""
    if _has(name, COLOR_HINTS):
        return isinstance(val, (tuple, list)) and len(val) >= 3
    if isinstance(val, (tuple, list)) and len(val) == 3:
        return all(isinstance(x, (int, float)) and 0.0 <= float(x) <= 1.0 for x in val)
    return False


def _band(d: float, lo_mult: float, hi_mult: float) -> tuple[float, float]:
    """Relative band around a default, correct for negative and zero defaults."""
    if d == 0.0:
        return 0.0, 1.0          # engine draws U[0,1] here; that IS its natural domain
    a, b = d * lo_mult, d * hi_mult
    return (a, b) if a <= b else (b, a)


def _clamp_scalar(name: str, v: float, d: float) -> tuple[float, str | None]:
    lo, hi = _band(float(d), REL_LO, REL_HI)
    rule = "relative"
    if _has(name, POWER_HINTS):
        plo, phi = _band(float(d), POW_LO, POW_HI)
        lo, hi = max(lo, plo), min(hi, phi)
        hi = min(hi, POW_ABS)
        lo = max(lo, -POW_ABS)
        if lo > hi:                      # absolute safety cap wins over the relative band
            lo = hi
        rule = "power_abs"
    out = min(max(float(v), lo), hi)
    if _has(name, POSITION_HINTS):
        clipped = min(max(out, 0.0), 1.0)
        if clipped != out:
            out, rule = clipped, "position01"
    return round(out, 4), (rule if abs(out - float(v)) > 1e-9 else None)


def _clamp_int(name: str, v: int, d: int) -> tuple[int, str | None]:
    # RULE 4 GUARD: an int uniform at its default is either an ENUM selector (Box.location = 0,
    # Slides.type = 0) or the canonical count — rule 1 is relative to the default and therefore
    # vacuous there, so never move it. Without this guard the max(1, ...) floor silently rewrote
    # every `location = 0` / `type = 0` draw to 1 and deleted that enum branch from the dataset.
    if int(v) == int(d):
        return int(v), None
    lo, hi = _band(float(d), REL_LO, REL_HI)
    out = max(1, int(round(min(max(float(v), lo), hi))))
    return out, ("relative_int" if out != int(v) else None)


def clamp_params(shader_name: str, params: dict[str, Any], defaults: dict[str, Any],
                 rng: random.Random) -> tuple[dict[str, Any], list[dict]]:
    """Return (clamped_params, events). `defaults` maps uniform name -> parsed default."""
    out: dict[str, Any] = {}
    events: list[dict] = []

    def ev(param, raw, new, rule, **extra):
        events.append({"shader": shader_name, "param": param, "raw": raw, "clamped": new,
                       "rule": rule, "was_default": raw == defaults.get(param), **extra})

    for k, v in params.items():
        d = defaults.get(k)
        if isinstance(v, bool) or d is None:
            out[k] = v                                    # rule 4: booleans / enums untouched
            continue

        # ---- rule 3: colour ----
        if _is_color(k, v, d):
            n = 3
            cur = [float(x) for x in v[:n]]
            tail = tuple(v[n:])
            if LUMA_LO <= luma(cur) <= LUMA_HI:
                out[k] = v
                continue
            new = None
            for i in range(LUMA_TRIES):
                cand = [round(rng.random(), 3) for _ in range(n)]
                if LUMA_LO <= luma(cand) <= LUMA_HI:
                    new = tuple(cand) + tail
                    ev(k, list(v), list(new), "color_luma_resample",
                       luma_raw=round(luma(cur), 4), luma_new=round(luma(cand), 4), tries=i + 1)
                    break
            if new is None:                               # exhausted -> shader DEFAULT colour
                new = tuple(d)
                ev(k, list(v), list(new), "color_luma_default_fallback",
                   luma_raw=round(luma(cur), 4), luma_new=round(luma(list(d)[:3]), 4),
                   tries=LUMA_TRIES)
            out[k] = new
            continue

        # ---- scalars ----
        if isinstance(v, float):
            new, rule = _clamp_scalar(k, v, float(d))
            if rule:
                ev(k, v, new, rule)
            out[k] = new
            continue
        if isinstance(v, int):
            new, rule = _clamp_int(k, v, int(d))
            if rule:
                ev(k, v, new, rule)
            out[k] = new
            continue

        # ---- vectors (non-colour) ----
        if isinstance(v, (tuple, list)):
            comps, changed = [], False
            for i, x in enumerate(v):
                dx = d[i] if isinstance(d, (tuple, list)) and i < len(d) else x
                if isinstance(x, bool):
                    comps.append(x)
                elif isinstance(x, float):
                    nv, r = _clamp_scalar(k, x, float(dx))
                    comps.append(nv)
                    changed = changed or bool(r)
                elif isinstance(x, int):
                    nv, r = _clamp_int(k, x, int(dx))
                    comps.append(nv)
                    changed = changed or bool(r)
                else:
                    comps.append(x)
            new = tuple(comps)
            if changed:
                ev(k, list(v), list(new), "relative_vec")
            out[k] = new
            continue

        out[k] = v
    return out, events


def defaults_of(shader) -> dict[str, Any]:
    """{uniform name -> parsed default} for a shaders.Shader (tunable uniforms only)."""
    return {u.name: u.default for u in shader.tunable}


def make_filter(enabled: bool = True):
    """Build the `param_filter` callable consumed by streams_real.make_operator.

    Signature: f(shader_obj, params, rng) -> (params, events).  `enabled=False` is the
    identity (byte-identical to the pre-clamp audit path)."""
    if not enabled:
        return None

    def _f(shader, params, rng):
        return clamp_params(shader.name, params, defaults_of(shader), rng)

    return _f
