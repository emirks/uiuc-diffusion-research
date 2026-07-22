"""GL-Transitions shader bank: parse the .glsl files and sample their uniforms.

Each gl-transitions shader declares its tunable parameters as GLSL uniforms with
the default encoded in a trailing comment:

    uniform float bounces;      // = 3.0
    uniform ivec2 size;         // = ivec2(4)
    uniform vec4 bgcolor;       // = vec4(0.0, 0.0, 0.0, 1.0)

That comment is the whole reason this bank is a *parameterised operator family*
rather than 125 fixed effects: parsing it gives us a typed parameter space per
shader that we can sample to blow the bank up by 1-2 orders of magnitude.
"""

from __future__ import annotations

import dataclasses
import pathlib
import random
import re
from typing import Any

# `uniform <type> <name> ;`  followed by  `// = <default>`  or  `/* = <default> */`
_UNIFORM_RE = re.compile(
    r"^\s*uniform\s+(?P<type>[a-zA-Z0-9_]+)\s+(?P<name>[a-zA-Z0-9_]+)\s*"
    r"(?:/\*\s*=\s*(?P<cdefault>[^*]*?)\s*\*/)?\s*;"
    r"(?:\s*//\s*=?\s*(?P<default>.*))?$",
    re.MULTILINE,
)

_VEC_RE = re.compile(r"^\s*[iub]?vec[234]\s*\(\s*(?P<args>.*?)\s*\)\s*$")

# Shaders needing an auxiliary image sampler. We support them by feeding a
# procedurally generated map (see maps.py) — each map is a distinct operator.
AUX_SAMPLER_SHADERS = {"luma": "luma", "displacement": "displacementMap"}


@dataclasses.dataclass
class Uniform:
    name: str
    gtype: str          # float | int | bool | vec2 | vec3 | vec4 | ivec2 | sampler2D
    default: Any        # python scalar / tuple, or None when unparseable


@dataclasses.dataclass
class Shader:
    name: str
    source: str
    uniforms: list[Uniform]

    @property
    def tunable(self) -> list[Uniform]:
        return [u for u in self.uniforms
                if u.gtype != "sampler2D" and u.default is not None]


def _parse_default(gtype: str, raw: str | None):
    if raw is None:
        return None
    raw = raw.strip().rstrip(";").strip()
    if not raw:
        return None
    m = _VEC_RE.match(raw)
    if m:
        args = [a.strip() for a in m.group("args").split(",") if a.strip()]
        try:
            vals = [float(a) for a in args]
        except ValueError:
            return None
        n = int(gtype[-1])
        if len(vals) == 1:            # vec4(1.0) broadcasts
            vals = vals * n
        if len(vals) != n:
            return None
        return tuple(int(v) for v in vals) if gtype.startswith("ivec") else tuple(vals)
    if gtype == "bool":
        return raw.lower().startswith("t")
    try:
        return int(raw) if gtype == "int" else float(raw)
    except ValueError:
        return None


def load_bank(transitions_dir: str | pathlib.Path) -> dict[str, Shader]:
    """Parse every .glsl in `transitions_dir` into a Shader."""
    bank: dict[str, Shader] = {}
    for path in sorted(pathlib.Path(transitions_dir).glob("*.glsl")):
        src = path.read_text()
        uniforms = [
            Uniform(
                name=m.group("name"),
                gtype=m.group("type"),
                default=_parse_default(
                    m.group("type"), m.group("default") or m.group("cdefault")
                ),
            )
            for m in _UNIFORM_RE.finditer(src)
        ]
        bank[path.stem] = Shader(name=path.stem, source=src, uniforms=uniforms)
    return bank


# --------------------------------------------------------------------------
# Parameter sampling
# --------------------------------------------------------------------------
# Sampling has to stay inside each shader's *plausible* range or the effect
# degenerates (a wipe with smoothness 40 is a slow crossfade; a grid of
# 200x200 cells is noise). We infer the range from the default's magnitude and
# from naming conventions that gl-transitions authors follow consistently.

_FRACTIONAL_HINTS = ("smooth", "ratio", "amount", "strength", "intensity", "opacity",
                     "width", "height", "size", "scale", "pause", "border", "fade",
                     "random", "seed", "blur", "zoom", "shadow", "persp", "unzoom")
_COUNT_HINTS = ("count", "bounces", "steps", "segments", "squares", "cells", "bars",
                "rows", "cols", "num", "n_", "speed", "freq", "rotations", "spins")


def _jitter_float(name: str, d: float, rng: random.Random) -> float:
    lname = name.lower()
    if d == 0.0:
        return round(rng.uniform(0.0, 1.0), 4)
    if 0.0 < abs(d) <= 1.0 and any(h in lname for h in _FRACTIONAL_HINTS):
        # keep in the open unit interval: these are almost always fractions
        v = abs(d) * rng.uniform(0.15, 3.0)
        return round(min(max(v, 1e-3), 1.0), 4)
    if any(h in lname for h in _COUNT_HINTS):
        return round(abs(d) * rng.uniform(0.25, 4.0), 4)
    v = d * rng.uniform(0.3, 3.0)
    return round(v, 4)


def sample_params(shader: Shader, rng: random.Random, *, p_vary: float = 0.85
                  ) -> dict[str, Any]:
    """Draw one point from the shader's parameter space.

    `p_vary` is the per-uniform probability of deviating from the default, so a
    fraction of operators land on the canonical gl-transitions look.
    """
    out: dict[str, Any] = {}
    for u in shader.tunable:
        if rng.random() > p_vary:
            out[u.name] = u.default
            continue
        d, t = u.default, u.gtype
        if t == "float":
            out[u.name] = _jitter_float(u.name, float(d), rng)
        elif t == "int":
            out[u.name] = max(1, int(round(abs(d) * rng.uniform(0.3, 3.5)))) or 1
        elif t == "bool":
            out[u.name] = rng.random() < 0.5
        elif t.startswith("ivec"):
            out[u.name] = tuple(max(1, int(round(abs(x) * rng.uniform(0.3, 3.5))) or 1)
                                for x in d)
        elif t == "vec4" and "colo" in u.name.lower():
            out[u.name] = (round(rng.random(), 3), round(rng.random(), 3),
                           round(rng.random(), 3), d[3])
        elif t == "vec3" and "colo" in u.name.lower():
            out[u.name] = tuple(round(rng.random(), 3) for _ in range(3))
        elif t.startswith("vec"):
            out[u.name] = tuple(_jitter_float(u.name, float(x), rng) for x in d)
        else:
            out[u.name] = d
    return out


def param_signature(name: str, params: dict[str, Any]) -> str:
    """Stable short id for a (shader, params) pair — used in filenames + manifests."""
    if not params:
        return name
    body = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{name}[{body}]"
