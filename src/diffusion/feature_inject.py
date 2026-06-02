"""Feature injection for LTX-2 transformer — the read side of the cache.

Loads tensors written by `diffusion.feature_cache.FeatureCache` and injects
them into a downstream denoising pass via *write* hooks on the transformer's
attention linears (and optionally the block output). This is the mechanism
behind MasaCtrl / Plug-and-Play / DiTCtrl style structure transfer.

Design mirror of `FeatureCache`:
  * The injector loads one cached phase (default "recon") and one substep
    (default "predictor"), keyed by step index.
  * A denoising loop calls `set_step(step_idx)` before each outer step and
    `set_substep(substep)` before each transformer call.
  * Write hooks on `attn1.to_k` / `attn1.to_v` (and optionally `to_q`, and
    the block output) overwrite the configured token positions with the
    cached source tensors, gated on (step ∈ inject_steps, substep matches,
    layer ∈ inject_layers).

Why this works at the linear-output point. The cache stored Q/K/V as the
output of `to_q/to_k/to_v` — pre-RMSNorm, pre-RoPE. The downstream RMSNorm
and RoPE are deterministic functions of the token (and its position), so
overwriting the linear output before norm/RoPE is exactly equivalent to
overwriting the K/V that scaled-dot-product attention consumes, provided
the injected tokens occupy the same positions in both passes. For
self-injection into the same sample (same latent layout) that holds exactly.

Token-scoped injection. `token_ids` restricts the overwrite to a subset of
sequence positions (e.g. the free-middle latent frames of a C2V transition).
Positions outside `token_ids` keep the edit pass's own K/V untouched — so
conditioned/anchor tokens are never disturbed.

Strength. `strength ∈ [0, 1]` blends cached over current:
    out[:, ids] = strength * cached[:, ids] + (1 - strength) * out[:, ids]
strength=1.0 is a hard replace (MasaCtrl default).

Batch handling. The default edit pass here is CFG=1 (batch=1), matching the
cached CFG=1 recon. If you inject into a CFG>1 pass (batch=2 = [uncond, cond]),
set `cfg_batch=True`; the injector then writes the cached (batch-1) tensor
into both rows. If you also pass `cond_only_at_cfg=True`, the cached tensor is
written ONLY into the cond row (row 1) and the uncond row (row 0) is left
untouched — that preserves the CFG mix `v_uncond + s · (v_cond − v_uncond)`
inside the injection region instead of collapsing v_uncond≈v_cond there. The
LTX-2 pipeline lays out batched CFG as [uncond ; cond] (see prepare_sample in
exp_040), so row 0 = uncond and row 1 = cond.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any, Iterable

import torch
import torch.nn as nn
import yaml

log = logging.getLogger(__name__)

# Sites this injector can write. block_out is handled specially (tuple output).
INJECTABLE_SITES = {"attn1_q", "attn1_k", "attn1_v", "block_out"}

_SITE_TO_SUBMODULE = {
    "attn1_q": ("attn1", "to_q"),
    "attn1_k": ("attn1", "to_k"),
    "attn1_v": ("attn1", "to_v"),
}


class FeatureInjector:
    def __init__(
        self,
        cache_dir: pathlib.Path,
        *,
        inject_layers: Iterable[int],
        inject_steps: Iterable[int],
        sites: Iterable[str],
        token_ids: torch.Tensor | None,
        phase: str = "recon",
        substep: str = "predictor",
        strength: float = 1.0,
        cfg_batch: bool = False,
        cond_only_at_cfg: bool = False,
        device: str = "cuda:0",
        log: logging.Logger | None = None,
    ) -> None:
        self.cache_dir   = pathlib.Path(cache_dir)
        self.phase       = phase
        self.read_substep = substep
        self.strength    = float(strength)
        self.cfg_batch   = bool(cfg_batch)
        self.cond_only_at_cfg = bool(cond_only_at_cfg)
        self.device      = device
        self.log         = log or logging.getLogger(__name__)

        self.inject_layers = sorted(set(int(i) for i in inject_layers))
        self.layer_set     = set(self.inject_layers)
        self.inject_steps  = sorted(set(int(i) for i in inject_steps))
        self.step_set      = set(self.inject_steps)
        self.sites         = list(sites)
        for s in self.sites:
            if s not in INJECTABLE_SITES:
                raise ValueError(f"Cannot inject site {s}. Valid: {sorted(INJECTABLE_SITES)}")
        self.site_set = set(self.sites)

        self.token_ids = (
            None if token_ids is None
            else token_ids.to(device=device, dtype=torch.long)
        )

        # Manifest — verify the cache actually has what we want to inject.
        manifest_path = self.cache_dir / "manifest.yaml"
        self.manifest = yaml.safe_load(manifest_path.read_text()) if manifest_path.exists() else {}
        if self.manifest:
            cached_layers = set(self.manifest.get("layer_indices", []))
            missing = self.layer_set - cached_layers
            if missing:
                raise ValueError(f"inject_layers {sorted(missing)} were not cached "
                                 f"(cache has {sorted(cached_layers)}).")
            cached_steps = set(self.manifest.get("step_indices", {}).get(phase, []))
            missing_s = self.step_set - cached_steps
            if missing_s:
                raise ValueError(f"inject_steps {sorted(missing_s)} were not cached "
                                 f"for phase '{phase}' (cache has {sorted(cached_steps)}).")
            cached_sites = set(self.manifest.get("sites", []))
            missing_site = self.site_set - cached_sites
            if missing_site:
                raise ValueError(f"sites {sorted(missing_site)} were not cached "
                                 f"(cache has {sorted(cached_sites)}).")
            # If the cache is token-scoped, the injection token_ids must match
            # the scope exactly (same set + ascending order) — the scoped cache
            # is stored compactly and aligned 1:1 with these ids.
            cache_scope = self.manifest.get("token_scope", None)
            if cache_scope is not None:
                if self.token_ids is None:
                    raise ValueError("Cache is token-scoped but injector token_ids is None "
                                     "(would inject all positions from a partial cache).")
                want = [int(i) for i in self.token_ids.tolist()]
                if want != list(cache_scope):
                    raise ValueError(
                        f"Injection token_ids do not match the cache's token_scope: "
                        f"{len(want)} ids vs {len(cache_scope)} scoped. The scoped "
                        f"cache only stored those positions.")

        # Runtime state.
        self._handles: list[tuple[Any, str]] = []
        self._cur_step: int | None = None
        self._cur_substep: str | None = None
        self._active = False
        # Loaded per-step source tensors: {layer: {site: Tensor on device}}
        self._cur_src: dict[int, dict[str, torch.Tensor]] = {}

        self.log.info(
            "FeatureInjector: phase=%s substep=%s layers=%s steps=%s sites=%s "
            "strength=%.2f tokens=%s cfg_batch=%s cond_only_at_cfg=%s",
            phase, substep, self.inject_layers, self.inject_steps, self.sites,
            self.strength,
            "ALL" if self.token_ids is None else f"{self.token_ids.numel()} ids",
            self.cfg_batch, self.cond_only_at_cfg,
        )

    # ── attach / detach ────────────────────────────────────────────────────────

    def attach(self, transformer: nn.Module) -> None:
        blocks = transformer.transformer_blocks
        for lidx in self.inject_layers:
            block = blocks[lidx]
            for site in self.sites:
                if site == "block_out":
                    h = block.register_forward_hook(self._make_block_hook(lidx))
                    self._handles.append((h, f"block_out@L{lidx}"))
                else:
                    attn_name, lin_name = _SITE_TO_SUBMODULE[site]
                    lin = getattr(getattr(block, attn_name), lin_name)
                    h = lin.register_forward_hook(self._make_linear_hook(lidx, site))
                    self._handles.append((h, f"{site}@L{lidx}"))
        self.log.info("FeatureInjector attached: %d write-hooks.", len(self._handles))

    def detach(self) -> None:
        for h, _ in self._handles:
            h.remove()
        self._handles.clear()
        self.log.info("FeatureInjector detached.")

    # ── per-step / per-substep gating ──────────────────────────────────────────

    def set_step(self, step_idx: int) -> None:
        """Load the cached source tensors for this step (if it's an inject step)."""
        self._cur_step = int(step_idx)
        self._cur_src = {}
        if self._cur_step not in self.step_set:
            return
        step_path = self.cache_dir / self.phase / f"step_{self._cur_step:03d}.pt"
        payload = torch.load(step_path, weights_only=False)
        blocks = payload["blocks"].get(self.read_substep, {})
        for lidx in self.inject_layers:
            if lidx not in blocks:
                continue
            self._cur_src[lidx] = {}
            for site in self.sites:
                if site in blocks[lidx]:
                    self._cur_src[lidx][site] = (
                        blocks[lidx][site].to(self.device, dtype=torch.float32)
                    )

    def set_substep(self, substep: str) -> None:
        """Enable injection only on the substep we cached from."""
        self._cur_substep = substep
        self._active = (
            substep == self.read_substep
            and self._cur_step in self.step_set
            and len(self._cur_src) > 0
        )

    # ── hook factories ─────────────────────────────────────────────────────────

    def _blend(self, out: torch.Tensor, cached: torch.Tensor) -> torch.Tensor:
        """Overwrite token positions of `out` with `cached` (blended by strength).

        out: [B, N, D] (B=1, or B=2 when cfg_batch).
        cached: [1, N, D] (full token axis) OR [1, n_ids, D] (token-scoped cache,
        where the saved tokens align 1:1 — same ascending order — with
        self.token_ids). The scoped form is auto-detected by token count.
        """
        out = out.clone()
        cached = cached.to(out.dtype)
        ids = self.token_ids
        if self.cfg_batch:
            # Layout is [uncond ; cond] (see exp_040 prepare_sample). When
            # cond_only_at_cfg=True, write only into row 1 (cond) so the CFG
            # mix v_uncond + s·(v_cond − v_uncond) does NOT collapse inside
            # the injection region.
            rows = [1] if self.cond_only_at_cfg else range(out.shape[0])
        else:
            rows = [0]
        s = self.strength
        n_ids = None if ids is None else ids.numel()
        for r in rows:
            cr = cached[0] if cached.shape[0] == 1 else cached[r]
            if ids is None:
                out[r] = s * cr + (1.0 - s) * out[r]
            elif cr.shape[0] == n_ids and cr.shape[0] != out.shape[1]:
                # Token-scoped cache: compact [n_ids, D] maps 1:1 onto ids.
                out[r, ids, :] = s * cr + (1.0 - s) * out[r, ids, :]
            else:
                # Full-token cache: index the ids out of the full axis.
                out[r, ids, :] = s * cr[ids, :] + (1.0 - s) * out[r, ids, :]
        return out

    def _make_linear_hook(self, layer_idx: int, site: str):
        inj = self

        def _hook(module: nn.Module, _inp: Any, out: torch.Tensor) -> torch.Tensor | None:
            if not inj._active:
                return None
            src = inj._cur_src.get(layer_idx, {})
            if site not in src:
                return None
            return inj._blend(out, src[site])

        return _hook

    def _make_block_hook(self, layer_idx: int):
        inj = self

        def _hook(module: nn.Module, _inp: Any, out: Any):
            if not inj._active:
                return None
            src = inj._cur_src.get(layer_idx, {})
            if "block_out" not in src:
                return None
            if isinstance(out, tuple):
                video = inj._blend(out[0], src["block_out"])
                return (video,) + tuple(out[1:])
            return inj._blend(out, src["block_out"])

        return _hook
