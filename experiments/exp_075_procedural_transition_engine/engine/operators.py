"""Operator sampling, the endpoint-identity gate, and clip rendering.

An *operator* here is not just a shader. It is the tuple

    (shader, sampled uniforms, easing curve, spatial flip, direction swap,
     layer-extension policy, auxiliary map)

which is what turns ~125 GLSL primitives into a bank of thousands of visually
distinguishable transitions while keeping the operator statistically independent
of the endpoint content it is applied to.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any

import numpy as np

from . import maps, shaders, streams
from .glrunner import GLRunner, ShaderCompileError

FLIPS = ("none", "h", "v", "hv")


@dataclasses.dataclass
class Operator:
    op_id: str
    shader: str
    params: dict[str, Any]
    easing: str
    flip: str
    swap: bool
    extension: str
    aux_kind: str | None = None
    aux_seed: int = 0

    def describe(self) -> str:
        bits = [self.shader, f"ease={self.easing}"]
        if self.flip != "none":
            bits.append(f"flip={self.flip}")
        if self.swap:
            bits.append("reversed")
        if self.aux_kind:
            bits.append(f"map={self.aux_kind}")
        bits.append(f"ext={self.extension}")
        if self.params:
            bits.append(" ".join(f"{k}={v}" for k, v in sorted(self.params.items())))
        return "  ".join(bits)


def _apply_flip(img: np.ndarray, flip: str) -> np.ndarray:
    if flip == "h":
        return img[:, ::-1]
    if flip == "v":
        return img[::-1]
    if flip == "hv":
        return img[::-1, ::-1]
    return img


# --------------------------------------------------------------------------
# Bank validation
# --------------------------------------------------------------------------

def validate_bank(runner: GLRunner, bank: dict[str, shaders.Shader],
                  tol: float = 2.0) -> tuple[dict[str, shaders.Shader], list[dict]]:
    """Keep only shaders that compile AND satisfy the p=0/p=1 identities.

    A shader that does not return `from` at progress 0 (or `to` at progress 1)
    silently corrupts the conditioning frames of every sample it generates — the
    model would be trained on endpoints that do not match its own inputs. This
    gate is cheap and non-negotiable.
    """
    rng = np.random.default_rng(0)
    h, w = runner.height, runner.width
    a = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    b = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)

    kept, report = {}, []
    for name, sh in sorted(bank.items()):
        row = {"shader": name, "n_params": len(sh.tunable)}
        try:
            prog = runner.program(name, sh.source)
        except ShaderCompileError as exc:
            row.update(status="compile_error", detail=str(exc)[:180])
            report.append(row)
            continue

        aux_uniform = shaders.AUX_SAMPLER_SHADERS.get(name)
        if aux_uniform:
            runner.set_aux_map(maps.make_map("fbm", h, w, 0))
        defaults = {u.name: u.default for u in sh.tunable}
        try:
            f0 = runner.render(prog, a, b, 0.0, defaults, aux_uniform)
            f1 = runner.render(prog, a, b, 1.0, defaults, aux_uniform)
        except Exception as exc:
            row.update(status="render_error", detail=str(exc)[:180])
            report.append(row)
            continue

        e0 = float(np.abs(f0.astype(np.float32) - a.astype(np.float32)).mean())
        e1 = float(np.abs(f1.astype(np.float32) - b.astype(np.float32)).mean())
        row.update(mae_p0=round(e0, 3), mae_p1=round(e1, 3))
        if e0 <= tol and e1 <= tol:
            row["status"] = "ok"
            kept[name] = sh
        else:
            row["status"] = "endpoint_violation"
        report.append(row)
    return kept, report


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def check_operator(runner: GLRunner, bank: dict[str, shaders.Shader], op: "Operator",
                   frame_a: np.ndarray, frame_b: np.ndarray) -> tuple[float, float]:
    """Endpoint identity for ONE sampled operator, on the real frames it will use.

    `validate_bank` is not sufficient: it tests each shader at its *default*
    parameters, and the p=0/p=1 identities turn out to be parameter-dependent.
    e.g. `undulatingBurnOut` is clean at its defaults but leaves MAE 8.1 on the
    end block at a sampled `smoothness`. Every operator must be checked at the
    parameters it will actually be rendered with.
    """
    prog = runner.program(op.shader, bank[op.shader].source)
    if op.aux_kind:
        runner.set_aux_map(maps.make_map(op.aux_kind, runner.height, runner.width,
                                         op.aux_seed))
    aux_uniform = shaders.AUX_SAMPLER_SHADERS.get(op.shader)
    fa, fb = _apply_flip(frame_a, op.flip), _apply_flip(frame_b, op.flip)
    if op.swap:
        r0 = runner.render(prog, fb, fa, 1.0, op.params, aux_uniform)
        r1 = runner.render(prog, fb, fa, 0.0, op.params, aux_uniform)
    else:
        r0 = runner.render(prog, fa, fb, 0.0, op.params, aux_uniform)
        r1 = runner.render(prog, fa, fb, 1.0, op.params, aux_uniform)
    d0 = float(np.abs(_apply_flip(r0, op.flip).astype(np.float32)
                      - frame_a.astype(np.float32)).mean())
    d1 = float(np.abs(_apply_flip(r1, op.flip).astype(np.float32)
                      - frame_b.astype(np.float32)).mean())
    return d0, d1


def sample_valid_operator(runner: GLRunner, bank: dict[str, shaders.Shader],
                          rng: random.Random, frame_a: np.ndarray,
                          frame_b: np.ndarray, *, tol: float = 0.5,
                          max_tries: int = 20, stats: dict | None = None,
                          **kw) -> "Operator":
    """Rejection-sample an operator that passes the endpoint gate at its own params."""
    for _ in range(max_tries):
        op = sample_operator(bank, rng, **kw)
        d0, d1 = check_operator(runner, bank, op, frame_a, frame_b)
        if stats is not None:
            stats["tried"] = stats.get("tried", 0) + 1
        if max(d0, d1) <= tol:
            return op
        if stats is not None:
            stats["rejected"] = stats.get("rejected", 0) + 1
            stats.setdefault("rejected_shaders", {})
            stats["rejected_shaders"][op.shader] = \
                stats["rejected_shaders"].get(op.shader, 0) + 1
    raise RuntimeError(f"no operator passed the endpoint gate in {max_tries} tries")


def sample_operator(bank: dict[str, shaders.Shader], rng: random.Random,
                    *, extensions=("boomerang",), easings=None,
                    p_flip: float = 0.6, p_swap: float = 0.5) -> Operator:
    name = rng.choice(sorted(bank))
    sh = bank[name]
    params = shaders.sample_params(sh, rng)
    easing = rng.choice(easings or sorted(streams.EASINGS))
    flip = rng.choice(FLIPS[1:]) if rng.random() < p_flip else "none"
    swap = rng.random() < p_swap
    aux_kind = aux_seed = None
    if name in shaders.AUX_SAMPLER_SHADERS:
        aux_kind = rng.choice(maps.MAP_KINDS)
        aux_seed = rng.randrange(1 << 30)
    return Operator(
        op_id=f"{name}_{abs(hash((name, str(sorted(params.items())), easing, flip, swap, aux_kind, aux_seed))) % (10 ** 8):08d}",
        shader=name, params=params, easing=easing, flip=flip, swap=swap,
        extension=rng.choice(extensions), aux_kind=aux_kind, aux_seed=aux_seed or 0,
    )


def bank_capacity(bank: dict[str, shaders.Shader], *, samples_per_param: int = 6,
                  n_easings: int = 12) -> int:
    """Rough count of distinguishable operators the bank can emit."""
    total = 0
    for sh in bank.values():
        n = max(1, samples_per_param ** min(len(sh.tunable), 4))
        if sh.name in shaders.AUX_SAMPLER_SHADERS:
            n *= len(maps.MAP_KINDS) * 8
        total += n
    return total * n_easings * len(FLIPS) * 2      # × flips × swap


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_sample(runner: GLRunner, bank: dict[str, shaders.Shader], op: Operator,
                  start9: np.ndarray, end9: np.ndarray, total: int = 121
                  ) -> np.ndarray:
    """Render the full `total`-frame procedural transition clip."""
    n_start, n_end = len(start9), len(end9)
    prog = runner.program(op.shader, bank[op.shader].source)
    if op.aux_kind:
        runner.set_aux_map(maps.make_map(op.aux_kind, runner.height, runner.width,
                                         op.aux_seed))
    aux_uniform = shaders.AUX_SAMPLER_SHADERS.get(op.shader)

    a_stream = streams.build_from_stream(start9, total, op.extension)
    b_stream = streams.build_to_stream(end9, total, op.extension)
    p = streams.progress_ramp(total, n_start, n_end, op.easing)

    out = np.empty_like(a_stream)
    for t in range(total):
        fa = _apply_flip(a_stream[t], op.flip)
        fb = _apply_flip(b_stream[t], op.flip)
        if op.swap:
            frame = runner.render(prog, fb, fa, 1.0 - p[t], op.params, aux_uniform)
        else:
            frame = runner.render(prog, fa, fb, p[t], op.params, aux_uniform)
        out[t] = _apply_flip(frame, op.flip)
    return out


def endpoint_fidelity(clip: np.ndarray, start9: np.ndarray, end9: np.ndarray
                      ) -> tuple[float, float]:
    """MAE of the rendered clip's conditioning blocks against the given endpoints."""
    n_s, n_e = len(start9), len(end9)
    d0 = np.abs(clip[:n_s].astype(np.float32) - start9.astype(np.float32)).mean()
    d1 = np.abs(clip[-n_e:].astype(np.float32) - end9.astype(np.float32)).mean()
    return float(d0), float(d1)
