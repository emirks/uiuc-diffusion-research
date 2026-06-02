"""Comprehensive feature cache for LTX-2 transformer during inversion / generation.

Hook-based capture of intermediate tensors produced by an LTX-2 transformer
(`LTX2VideoTransformer3DModel`) during any phase that calls
``transformer(...)`` — most notably RF-Solver inversion (midpoint, CFG=1),
midpoint reconstruction (CFG=1), and Euler regeneration (CFG=gen).

What gets captured per outer step (cheap, recorded explicitly by the loop):
    * ``z_t_in``       — packed latent into the transformer at this substep
                         (``[B, N, in_channels=128]`` bf16)
    * ``v_pred``       — raw model output (post-CFG mix where applicable),
                         packed (``[B, N, 128]`` bf16)
    * ``sigma_curr``, ``sigma_mid`` (midpoint only), ``sigma_next``, ``dtau``
    * ``t_value``      — per-token timestep scalar (= sigma * 1000)
    * ``phase``, ``step_idx``, ``substep`` metadata.

What gets captured per (step, substep, layer) via forward hooks (heavy):
    * ``block_out``    — output of ``transformer_blocks[l]`` (video stream
                         only — block returns ``(video, audio)`` tuple).
                         Shape ``[B, N_video, D=4096]`` bf16.
    * ``attn1_q``, ``attn1_k``, ``attn1_v``
                       — outputs of ``transformer_blocks[l].attn1.to_q/to_k/to_v``.
                         These are the **self-attention Q,K,V tokens
                         pre-RMSNorm and pre-RoPE** (LTX-2 applies both inside
                         the attention forward, after the linear projection).
                         Shape ``[B, N_video, D=4096]`` bf16.
    * Optional sites (config-toggleable):
        * ``attn2_q``, ``attn2_k``, ``attn2_v`` — text cross-attention.
          ``attn2_q`` is video-side (``[B, N_video, D]``); ``attn2_k``,
          ``attn2_v`` are text-side (``[B, N_text≈128, D]``).
        * ``ff_out``     — output of the block's video FFN.
        * ``audio_attn1_q/k/v`` — audio self-attn (`[B, N_audio, D_audio=2048]`).
        * ``a2v_q``, ``a2v_k``, ``a2v_v`` — audio→video cross-attn (Q=video, K,V=audio).

File layout written into ``<sample_dir>/feature_cache/`` :

    feature_cache/
        manifest.yaml         — config + model dims + index of saved steps
        static.pt             — clean_latents, conditioning_mask, sigmas_inv,
                                sigmas_gen, sample_id, prompt strings.
        invert/step_NN.pt     — one file per outer step that is in the
                                save grid. Contains both the cheap step
                                payload AND the heavy {site, layer} dict.
        recon/step_NN.pt
        regen/step_NN.pt

Each per-step ``.pt`` is a dict::

    {
        "phase":      "invert" | "recon" | "regen",
        "step_idx":   int,
        "sigma_curr": float, "sigma_next": float, "dtau": float,
        "sigma_mid":  float | None,        # midpoint phases only
        "t_value":    float,                # = sigma_curr * 1000
        "step_payload": {
            "predictor": {
                "z_in":    Tensor[B, N, 128]  bf16,
                "v_pred":  Tensor[B, N, 128]  bf16,
                "sigma":   float,
            },
            "corrector": {...}   # midpoint phases only
            "euler":     {...}   # regen only (synonym for the lone Euler pass)
        },
        "blocks": {
            "predictor": {
                layer_idx: {
                    "block_out": Tensor[B, N_video, D]  bf16,
                    "attn1_q":   Tensor[B, N_video, D]  bf16,
                    "attn1_k":   Tensor[B, N_video, D]  bf16,
                    "attn1_v":   Tensor[B, N_video, D]  bf16,
                    # optional sites if enabled
                },
                ...
            },
            "corrector": {...}    # midpoint phases only
            "euler":     {...}    # regen only
        }
    }

Caveats:

1. Q,K,V tensors are **pre-RMSNorm and pre-RoPE**. The attention computation
   inside ``LTX2Attention.forward`` applies ``qk_norm`` (RMSNorm across
   heads) and rotary embeddings to Q and K before the scaled dot-product.
   Pre-norm/pre-RoPE is the most reusable cache point (RoPE positions are
   deterministic and can be re-applied at any later inject time, even with
   shifted layouts). If you want the post-norm-post-RoPE Q,K used inside
   SDP, apply the same RMSNorm + RoPE pipeline yourself, or extend this
   module to hook the attention forward directly.

2. CFG batching is preserved as-is. During Euler regen with CFG > 1 the
   transformer is called with ``[2, N, ...]`` (uncond cat cond). All
   captured tensors then have ``B=2`` — the user must split into
   ``[uncond, cond]`` chunks at inject time. Invert/recon at CFG=1 have B=1.

3. Block output is taken from the **video stream only** (``out[0]``). The
   block returns a ``(video_hidden_states, audio_hidden_states)`` tuple.

4. The cache holds at most ONE (step, substep)'s heavy tensors in memory at
   a time. Each call to ``flush_step`` writes the full per-step dict to
   disk and clears the in-memory bucket. Memory peak is bounded by the
   selected layer count × site count × per-tensor size.

5. Hooks are attached at the class level on the transformer instance and
   persist across samples. Call ``detach()`` once at end of run.

6. The audio-side hooks default to OFF since the audio stream is a fixed
   silent context across all phases in our inversion setup; capturing them
   wastes disk unless the experiment actually varies audio.
"""
from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Iterable

import torch
import torch.nn as nn
import yaml

log = logging.getLogger(__name__)


# Canonical site names. Extending this set requires updating _register_hooks.
VIDEO_SITES = {
    "block_out",
    "attn1_q", "attn1_k", "attn1_v",
    "attn2_q", "attn2_k", "attn2_v",
    "ff_out",
}
AUDIO_SITES = {
    "audio_attn1_q", "audio_attn1_k", "audio_attn1_v",
    "a2v_q", "a2v_k", "a2v_v",
}
ALL_SITES = VIDEO_SITES | AUDIO_SITES


def _dtype_from_str(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


class FeatureCache:
    """Hook-based feature cache for LTX-2 transformer.

    Usage::

        cache = FeatureCache(cfg, log=log)
        cache.attach(pipe.transformer)
        ...
        cache.start_sample(sample_dir)
        cache.save_static({...})
        for i in range(num_steps):
            cache.set_step(phase='invert', step_idx=i,
                           sigma_curr=..., sigma_next=..., sigma_mid=...)
            cache.set_substep('predictor', sigma=sigma_curr)
            v_raw = transformer(...)  # hooks capture
            cache.record_cheap(substep='predictor', z_in=z, v_pred=v_raw)
            cache.set_substep('corrector', sigma=sigma_mid)
            v_mid_raw = transformer(...)
            cache.record_cheap(substep='corrector', z_in=z_mid, v_pred=v_mid_raw)
            cache.flush_step()
        cache.end_sample()
        ...
        cache.detach()
    """

    def __init__(self, cfg: dict, *, log: logging.Logger | None = None) -> None:
        self.cfg               = dict(cfg)
        self.log               = log or logging.getLogger(__name__)
        self.dtype             = _dtype_from_str(cfg.get("dtype", "bfloat16"))
        self.layer_indices     = sorted(set(int(i) for i in cfg["layer_indices"]))
        self.layer_set         = set(self.layer_indices)
        self.step_sets: dict[str, set[int]] = {
            "invert": set(int(i) for i in cfg.get("step_indices_invert", [])),
            "recon":  set(int(i) for i in cfg.get("step_indices_recon",  [])),
            "regen":  set(int(i) for i in cfg.get("step_indices_regen",  [])),
        }
        self.substeps_midpoint = list(cfg.get("substeps_midpoint", ["predictor"]))
        self.sites             = list(cfg.get("sites", ["block_out", "attn1_q", "attn1_k", "attn1_v"]))
        for s in self.sites:
            if s not in ALL_SITES:
                raise ValueError(f"Unknown site: {s}. Valid: {sorted(ALL_SITES)}")
        self.site_set          = set(self.sites)

        # Optional token scoping. When set, only these sequence positions are
        # saved per tensor (token dim = 1), shrinking the cache to the region
        # we actually inject into (e.g. the C2V free middle). Default None =
        # cache the full token axis (backward-compatible). The injector
        # auto-detects a scoped cache by comparing the cached token count to
        # the number of injection ids, so the saved order MUST match the
        # injector's token_ids order (both built ascending from frame slices).
        ts = cfg.get("token_scope", None)
        self.token_scope: torch.Tensor | None = (
            None if ts is None
            else torch.as_tensor(sorted(int(i) for i in ts), dtype=torch.long)
        )

        # Hook registry: list of (handle, label) for detach().
        self._handles: list[tuple[Any, str]] = []
        self._attached_transformer: nn.Module | None = None
        self._num_blocks: int | None = None

        # Per-call capture state (set by set_step / set_substep).
        self._cur_phase:   str | None = None
        self._cur_step:    int | None = None
        self._cur_substep: str | None = None
        self._cur_sigma_curr: float | None = None
        self._cur_sigma_mid:  float | None = None
        self._cur_sigma_next: float | None = None
        self._cur_sigma:      float | None = None    # σ of the active substep

        # Per-step buffer: phase/step/substep → {layer → {site → tensor}}
        self._blocks_bucket: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
        # Cheap per-substep payload at the current step.
        self._cheap_bucket: dict[str, dict[str, Any]] = {}

        # Per-sample sink (set by start_sample).
        self._sample_dir: pathlib.Path | None = None
        self._saved_step_index: dict[str, list[int]] = {"invert": [], "recon": [], "regen": []}

        self.log.info(
            "FeatureCache config: layers=%s  steps[inv]=%d  steps[rec]=%d  steps[reg]=%d  "
            "substeps_mp=%s  sites=%s  dtype=%s",
            self.layer_indices,
            len(self.step_sets["invert"]),
            len(self.step_sets["recon"]),
            len(self.step_sets["regen"]),
            self.substeps_midpoint,
            self.sites,
            cfg.get("dtype", "bfloat16"),
        )

    # ── attach / detach ────────────────────────────────────────────────────────

    def attach(self, transformer: nn.Module) -> None:
        """Register forward hooks on the configured layers and sites.

        Hooks gate on (phase, step, substep) state — a hook called outside a
        configured (step, substep) is a no-op. Layer membership and site
        membership are baked into the hook closure at registration time.
        """
        if self._attached_transformer is not None:
            raise RuntimeError("FeatureCache already attached.")
        self._attached_transformer = transformer

        if not hasattr(transformer, "transformer_blocks"):
            raise AttributeError(
                "transformer.transformer_blocks not found; expected an LTX-2 "
                "video transformer. Cannot attach feature cache."
            )
        blocks = transformer.transformer_blocks
        self._num_blocks = len(blocks)

        for lidx in self.layer_indices:
            if lidx < 0 or lidx >= self._num_blocks:
                raise ValueError(f"layer index {lidx} out of range [0, {self._num_blocks-1}]")
            block = blocks[lidx]

            # block_out — forward hook on the block itself.
            if "block_out" in self.site_set:
                h = block.register_forward_hook(self._make_block_out_hook(lidx))
                self._handles.append((h, f"block_out@L{lidx}"))

            # Video self-attn Q/K/V — linear hooks.
            if "attn1_q" in self.site_set:
                h = block.attn1.to_q.register_forward_hook(self._make_linear_hook(lidx, "attn1_q"))
                self._handles.append((h, f"attn1_q@L{lidx}"))
            if "attn1_k" in self.site_set:
                h = block.attn1.to_k.register_forward_hook(self._make_linear_hook(lidx, "attn1_k"))
                self._handles.append((h, f"attn1_k@L{lidx}"))
            if "attn1_v" in self.site_set:
                h = block.attn1.to_v.register_forward_hook(self._make_linear_hook(lidx, "attn1_v"))
                self._handles.append((h, f"attn1_v@L{lidx}"))

            # Video × text cross-attn Q/K/V (optional).
            if "attn2_q" in self.site_set:
                h = block.attn2.to_q.register_forward_hook(self._make_linear_hook(lidx, "attn2_q"))
                self._handles.append((h, f"attn2_q@L{lidx}"))
            if "attn2_k" in self.site_set:
                h = block.attn2.to_k.register_forward_hook(self._make_linear_hook(lidx, "attn2_k"))
                self._handles.append((h, f"attn2_k@L{lidx}"))
            if "attn2_v" in self.site_set:
                h = block.attn2.to_v.register_forward_hook(self._make_linear_hook(lidx, "attn2_v"))
                self._handles.append((h, f"attn2_v@L{lidx}"))

            # Video FFN output (optional).
            if "ff_out" in self.site_set:
                h = block.ff.register_forward_hook(self._make_linear_hook(lidx, "ff_out"))
                self._handles.append((h, f"ff_out@L{lidx}"))

            # Audio self-attn (optional).
            if "audio_attn1_q" in self.site_set:
                h = block.audio_attn1.to_q.register_forward_hook(self._make_linear_hook(lidx, "audio_attn1_q"))
                self._handles.append((h, f"audio_attn1_q@L{lidx}"))
            if "audio_attn1_k" in self.site_set:
                h = block.audio_attn1.to_k.register_forward_hook(self._make_linear_hook(lidx, "audio_attn1_k"))
                self._handles.append((h, f"audio_attn1_k@L{lidx}"))
            if "audio_attn1_v" in self.site_set:
                h = block.audio_attn1.to_v.register_forward_hook(self._make_linear_hook(lidx, "audio_attn1_v"))
                self._handles.append((h, f"audio_attn1_v@L{lidx}"))

            # Audio → video cross-attn (Q=video, K,V=audio) (optional).
            if "a2v_q" in self.site_set:
                h = block.audio_to_video_attn.to_q.register_forward_hook(self._make_linear_hook(lidx, "a2v_q"))
                self._handles.append((h, f"a2v_q@L{lidx}"))
            if "a2v_k" in self.site_set:
                h = block.audio_to_video_attn.to_k.register_forward_hook(self._make_linear_hook(lidx, "a2v_k"))
                self._handles.append((h, f"a2v_k@L{lidx}"))
            if "a2v_v" in self.site_set:
                h = block.audio_to_video_attn.to_v.register_forward_hook(self._make_linear_hook(lidx, "a2v_v"))
                self._handles.append((h, f"a2v_v@L{lidx}"))

        self.log.info(
            "FeatureCache attached: %d hooks across %d layers (transformer depth=%d).",
            len(self._handles), len(self.layer_indices), self._num_blocks,
        )

    def detach(self) -> None:
        for h, _ in self._handles:
            h.remove()
        self._handles.clear()
        self._attached_transformer = None
        self.log.info("FeatureCache detached.")

    def set_token_scope(self, token_ids: torch.Tensor | list[int] | None) -> None:
        """Set (or clear) the per-sample token scope.

        Token positions depend on the per-sample latent layout (H'·W'), so a
        caller that sweeps samples of differing resolution sets the scope each
        sample before ``start_sample``. Pass None to cache the full token axis.
        """
        if token_ids is None:
            self.token_scope = None
        else:
            t = torch.as_tensor([int(i) for i in token_ids], dtype=torch.long)
            self.token_scope = torch.sort(t).values
        self.log.info(
            "FeatureCache token_scope: %s",
            "ALL" if self.token_scope is None else f"{self.token_scope.numel()} tokens",
        )

    # ── hook factories ─────────────────────────────────────────────────────────

    def _should_save(self) -> bool:
        if self._cur_phase is None or self._cur_step is None or self._cur_substep is None:
            return False
        return self._cur_step in self.step_sets.get(self._cur_phase, set())

    def _save_tensor(self, layer_idx: int, site: str, tensor: torch.Tensor) -> None:
        substep = self._cur_substep
        assert substep is not None
        t = tensor.detach()
        if self.token_scope is not None:
            # Slice the token axis (dim=1) down to the scoped positions.
            t = t.index_select(1, self.token_scope.to(t.device))
        bucket = self._blocks_bucket.setdefault(substep, {}).setdefault(layer_idx, {})
        bucket[site] = t.to("cpu", dtype=self.dtype, non_blocking=False).contiguous()

    def _make_block_out_hook(self, layer_idx: int):
        cache = self

        def _hook(module: nn.Module, inp: Any, out: Any) -> None:
            if not cache._should_save():
                return
            # Block returns (video_hidden_states, audio_hidden_states).
            h = out[0] if isinstance(out, tuple) else out
            cache._save_tensor(layer_idx, "block_out", h)

        return _hook

    def _make_linear_hook(self, layer_idx: int, site: str):
        cache = self

        def _hook(module: nn.Module, inp: Any, out: Any) -> None:
            if not cache._should_save():
                return
            t = out[0] if isinstance(out, tuple) else out
            cache._save_tensor(layer_idx, site, t)

        return _hook

    # ── per-sample lifecycle ───────────────────────────────────────────────────

    def start_sample(self, sample_dir: pathlib.Path) -> None:
        self._sample_dir = pathlib.Path(sample_dir)
        cache_dir = self._sample_dir / "feature_cache"
        for phase in ("invert", "recon", "regen"):
            (cache_dir / phase).mkdir(parents=True, exist_ok=True)
        self._saved_step_index = {"invert": [], "recon": [], "regen": []}
        self.log.info("FeatureCache.start_sample: %s", cache_dir)

    def save_static(self, payload: dict[str, Any]) -> None:
        assert self._sample_dir is not None, "Call start_sample first."
        path = self._sample_dir / "feature_cache" / "static.pt"
        # Cast tensors to storage dtype to keep disk down (caller already passes
        # CPU tensors; we just match dtype where sensible).
        torch.save(payload, path)
        size_mb = path.stat().st_size / 1e6
        self.log.info("FeatureCache.save_static: %s (%.1f MB)", path.name, size_mb)

    def end_sample(self) -> None:
        assert self._sample_dir is not None
        # Write manifest.
        manifest = {
            "schema_version": 1,
            "model_id":       self.cfg.get("model_id", "Lightricks/LTX-2"),
            "num_blocks":     self._num_blocks,
            "layer_indices":  self.layer_indices,
            "step_indices":   {k: sorted(v) for k, v in self.step_sets.items()},
            "substeps_midpoint": self.substeps_midpoint,
            "sites":          self.sites,
            "dtype":          str(self.dtype).replace("torch.", ""),
            "saved_steps":    self._saved_step_index,
            "token_scope":    (None if self.token_scope is None
                               else [int(i) for i in self.token_scope.tolist()]),
        }
        path = self._sample_dir / "feature_cache" / "manifest.yaml"
        with path.open("w") as f:
            yaml.safe_dump(manifest, f, sort_keys=False)
        self.log.info(
            "FeatureCache.end_sample: manifest written.  Saved steps: "
            "inv=%d recon=%d regen=%d.",
            len(self._saved_step_index["invert"]),
            len(self._saved_step_index["recon"]),
            len(self._saved_step_index["regen"]),
        )
        self._sample_dir = None

    # ── per-step / per-substep context ─────────────────────────────────────────

    def set_step(
        self,
        *,
        phase: str,
        step_idx: int,
        sigma_curr: float,
        sigma_next: float,
        sigma_mid: float | None = None,
    ) -> None:
        if phase not in self.step_sets:
            raise ValueError(f"Unknown phase: {phase}")
        self._cur_phase      = phase
        self._cur_step       = int(step_idx)
        self._cur_sigma_curr = float(sigma_curr)
        self._cur_sigma_next = float(sigma_next)
        self._cur_sigma_mid  = None if sigma_mid is None else float(sigma_mid)
        self._cur_substep    = None
        self._cur_sigma      = None
        self._blocks_bucket  = {}
        self._cheap_bucket   = {}

    def set_substep(self, substep: str, *, sigma: float) -> None:
        # Midpoint phases use 'predictor' and 'corrector'; Euler uses 'euler'.
        # Hook saving for midpoint substeps is gated on
        #   substep in self.substeps_midpoint    (for invert/recon)
        # Regen always uses substep == 'euler' and is unconditional within
        # a saved step.
        if substep not in {"predictor", "corrector", "euler"}:
            raise ValueError(f"Unknown substep: {substep}")
        if (self._cur_phase in ("invert", "recon")
                and substep in ("predictor", "corrector")
                and substep not in self.substeps_midpoint):
            # Soft-disable: hooks should not save during this substep.
            self._cur_substep = None
            self._cur_sigma   = None
            return
        self._cur_substep = substep
        self._cur_sigma   = float(sigma)

    def record_cheap(
        self,
        *,
        substep: str,
        z_in: torch.Tensor,
        v_pred: torch.Tensor,
    ) -> None:
        """Record per-substep cheap payload (z_in and v_pred).

        Called by the invert / recon / regen loops once per transformer call.
        Always recorded (regardless of layer/site config) so long as the
        current step is in the save set. v_pred is the **raw post-CFG-mix
        output of the transformer** (matches what the loop integrates with).
        """
        if not self._should_save():
            return
        self._cheap_bucket[substep] = {
            "z_in":    z_in.detach().to("cpu", dtype=self.dtype, non_blocking=False).contiguous(),
            "v_pred":  v_pred.detach().to("cpu", dtype=self.dtype, non_blocking=False).contiguous(),
            "sigma":   self._cur_sigma if self._cur_sigma is not None else float("nan"),
        }

    def flush_step(self) -> pathlib.Path | None:
        """Write the current step's payload to ``<phase>/step_NN.pt`` and clear the bucket."""
        if self._sample_dir is None:
            raise RuntimeError("Call start_sample first.")
        if self._cur_phase is None or self._cur_step is None:
            return None
        if self._cur_step not in self.step_sets[self._cur_phase]:
            # not in save grid; drop silently.
            self._blocks_bucket = {}
            self._cheap_bucket  = {}
            return None

        payload = {
            "phase":      self._cur_phase,
            "step_idx":   self._cur_step,
            "sigma_curr": self._cur_sigma_curr,
            "sigma_next": self._cur_sigma_next,
            "sigma_mid":  self._cur_sigma_mid,
            "dtau":       (self._cur_sigma_next - self._cur_sigma_curr)
                          if self._cur_sigma_curr is not None and self._cur_sigma_next is not None
                          else None,
            "t_value":    (self._cur_sigma_curr * 1000.0)
                          if self._cur_sigma_curr is not None else None,
            "step_payload": self._cheap_bucket,
            "blocks":     self._blocks_bucket,
        }
        path = (self._sample_dir
                / "feature_cache"
                / self._cur_phase
                / f"step_{self._cur_step:03d}.pt")
        torch.save(payload, path)
        self._saved_step_index[self._cur_phase].append(self._cur_step)

        # Logging summary line.
        n_layers = max((len(v) for v in self._blocks_bucket.values()), default=0)
        n_subs   = len(self._blocks_bucket)
        size_mb  = path.stat().st_size / 1e6
        self.log.info(
            "  cache.flush[%s step=%d σ=%.4f]: %d substep(s) × %d layers → %s (%.1f MB)",
            self._cur_phase, self._cur_step, self._cur_sigma_curr or 0.0,
            n_subs, n_layers, path.name, size_mb,
        )

        self._blocks_bucket = {}
        self._cheap_bucket  = {}
        return path
