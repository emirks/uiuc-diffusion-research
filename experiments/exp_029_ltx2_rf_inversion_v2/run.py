"""exp_029 — LTX-2 RF-Solver flow inversion (strict-consistency v2).

Fork of exp_027 that closes six methodology gaps. See README "Why a v2"
for full motivation; short version:

  1. Inversion step count 30 → 40 (match generation grid; same σ samples).
  2. CFG=1 round-trip documented as SOLVER SELF-CONSISTENCY, not
     generation-trajectory recovery — the actual test of interest.
  3. New `regenerate` phase: from z1, run the FULL pipeline forward
     (Euler, CFG=gen_cfg) and compare z0 vs z0_regen. Dual gate.
  4. Audio strategy: default `zeros` passes literal `torch.zeros` to
     invert/recon/regen — silent DAVIS clips have no audio trajectory
     to preserve; zeros isolates the video flow and is reproducible.
     `AudioContextRecorder` still runs during base gen for forensics
     (`audio_record.pt`). Opt-in `capture_and_replay` available for
     ablation. Also fixes exp_027's misnamed `prepare_audio_latents(
     noise_scale=0, latents=None)` which actually returned randn.
     See PHASES_AND_CONTRACTS.md §6 for full discussion.
  5. Per-step CSV diagnostics for every phase (velocity / latent norms
     split by C2V mask, σ, step time, x0_pred norm).
  6. Comment on retrieve_latents sample_mode="argmax" — deterministic;
     a future stochastic switch would silently break C2V conditioning.

Workflow per sample (under run_dir/<sample_id>/):
  (a) Stage-1 generation     → z0  (audio: pipeline default noisy+stepped)
                              + AudioContextRecord saved as forensics
  (b) Inversion (CFG=1)      → z1       (audio: torch.zeros)
  (c) Reconstruction (CFG=1) → z0_recon (audio: torch.zeros, matches inv)
  (d) Regeneration (CFG=gen) → z0_regen (audio: torch.zeros, matches inv+recon)
  (e) MetricSuite over (z0,z0_recon) and (z0,z0_regen); dual gate.

Saved per sample:
  z0.pt, z1.pt, z_t_25/50/75.pt   — packed bfloat16 (1, N, 128)
  source_video.mp4                 — decode(z0)
  recon_video.mp4                  — decode(z0_recon)
  regen_video.mp4                  — decode(z0_regen)
  audio_record.pt                  — captured audio_hidden_states trajectory
  step_diag_<phase>.csv            — per-step diagnostics for each phase
  inv_meta.yaml                    — full metrics, gate status, audio summary
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import logging
import pathlib
import sys
import time
from typing import Any

import lpips as lpips_pkg
import numpy as np
import torch
import torchvision.io as tio
import yaml
from PIL import Image
from skimage.metrics import structural_similarity as _ssim_skimage

from diffusers.utils import export_to_video
from diffusers.pipelines.ltx2 import LTX2ConditionPipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import (
    LTX2VideoCondition,
    calculate_shift,
    retrieve_latents,
    retrieve_timesteps,
)

from diffusion.exp_utils import load_config, next_run_dir, TeeLogger

REPO_ROOT      = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = pathlib.Path(__file__).parent / "config.yaml"

LTX_TEMPORAL_SCALE = 8
DEVICE             = "cuda:0"
NUM_TRAIN_TIMESTEPS = 1000  # FlowMatchEulerDiscreteScheduler default; t = sigma * 1000

log = logging.getLogger(__name__)


# ── Frame / clip helpers (identical to exp_027) ──────────────────────────────

def load_frames_from_mp4(path: str | pathlib.Path, n: int, from_end: bool = False) -> list[Image.Image]:
    video, _, _ = tio.read_video(str(path), pts_unit="sec", output_format="THWC")
    if video.shape[0] == 0:
        raise ValueError(f"No frames decoded from {path}")
    frames = video[-n:] if from_end else video[:n]
    return [Image.fromarray(f.numpy()) for f in frames]


def load_frames_from_dir(path: str | pathlib.Path, n: int, from_end: bool = False) -> list[Image.Image]:
    jpgs = sorted(glob.glob(str(pathlib.Path(path) / "*.jpg")))
    if not jpgs:
        raise FileNotFoundError(f"No .jpg files in {path}")
    jpgs = jpgs[-n:] if from_end else jpgs[:n]
    return [Image.open(p).convert("RGB") for p in jpgs]


def load_clip_frames(sample: dict, repo_root: pathlib.Path, n: int) -> tuple[list[Image.Image], list[Image.Image]]:
    if "start_clip" in sample:
        start = load_frames_from_mp4(repo_root / sample["start_clip"], n, from_end=False)
        end   = load_frames_from_mp4(repo_root / sample["end_clip"],   n, from_end=True)
    else:
        start = load_frames_from_dir(repo_root / sample["start_images"], n, from_end=False)
        end   = load_frames_from_dir(repo_root / sample["end_images"],   n, from_end=True)
    return start, end


def end_clip_index(num_frames: int, num_clip_frames: int) -> int:
    n_lat = (num_frames      - 1) // LTX_TEMPORAL_SCALE + 1
    k_lat = (num_clip_frames - 1) // LTX_TEMPORAL_SCALE + 1
    return n_lat - k_lat


# ── Audio context recorder (Fix #4) ───────────────────────────────────────────

class AudioContextRecorder:
    """Wraps `pipe.transformer.forward` and snapshots `audio_hidden_states`
    at every call. Used during base generation to capture the audio
    trajectory the pipeline silently steps in parallel with the video.

    Why we need this: `LTX2ConditionPipeline` initializes audio latents at
    `noise_scale = sigmas[0] ≈ 1.0` (pure Gaussian noise) and applies
    `audio_scheduler.step(...)` at every iteration. The transformer's
    cross-attention between video and audio tokens therefore sees a
    *noisy then progressively denoised* audio context during generation.
    Inversion that passes zeros sees a different ODE.

    The recorder captures one tensor per transformer call. Under CFG>1
    the pipeline batches cond+uncond as `cat([audio_latents]*2)` (line
    1332 of pipeline_ltx2_condition.py — both halves identical at
    capture-time), so we store only the first half. The companion
    `audio_timestep` is captured the same way.

    Use:
        with AudioContextRecorder(pipe.transformer) as rec:
            pipe(...)
        # rec.captures: list[dict] with keys 'audio_hidden_states',
        # 'audio_timestep', 'video_timestep', step-indexed by call order.
    """

    def __init__(self, transformer):
        self._transformer = transformer
        self._original_forward = None
        self.captures: list[dict[str, Any]] = []

    def __enter__(self) -> "AudioContextRecorder":
        self.captures = []
        self._original_forward = self._transformer.forward
        recorder = self

        def wrapped_forward(*args, **kwargs):
            ah = kwargs.get("audio_hidden_states")
            at = kwargs.get("audio_timestep")
            vt = kwargs.get("timestep")
            # Take first half of CFG-doubled batch (cond and uncond audio
            # are identical at the point of capture; both come from the
            # same audio_latents tensor — see pipeline line 1332).
            def first_half(x):
                if x is None:
                    return None
                if x.dim() >= 1 and x.shape[0] >= 2 and x.shape[0] % 2 == 0:
                    return x[: x.shape[0] // 2].detach().clone().cpu()
                return x.detach().clone().cpu()
            recorder.captures.append({
                "audio_hidden_states": first_half(ah),
                "audio_timestep":      first_half(at),
                "video_timestep":      first_half(vt),
                "audio_hs_shape":      tuple(ah.shape) if ah is not None else None,
                "audio_hs_norm":       float(ah.float().norm().item()) if ah is not None else None,
                "audio_hs_mean":       float(ah.float().mean().item()) if ah is not None else None,
                "audio_hs_std":        float(ah.float().std().item())  if ah is not None else None,
                "audio_hs_dtype":      str(ah.dtype) if ah is not None else None,
            })
            return recorder._original_forward(*args, **kwargs)

        self._transformer.forward = wrapped_forward
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._original_forward is not None:
            self._transformer.forward = self._original_forward

    def summary(self) -> dict:
        return {
            "num_calls":  len(self.captures),
            "first_norm": self.captures[0]["audio_hs_norm"] if self.captures else None,
            "last_norm":  self.captures[-1]["audio_hs_norm"] if self.captures else None,
            "shape":      self.captures[0]["audio_hs_shape"] if self.captures else None,
            "dtype":      self.captures[0]["audio_hs_dtype"] if self.captures else None,
        }


# ── Per-step diagnostics CSV (Fix #5) ─────────────────────────────────────────

class StepDiagnostics:
    """CSV writer for per-step solver diagnostics.

    One row per outer step. For midpoint integrators a row corresponds
    to one (σ_curr → σ_next) advance regardless of how many transformer
    calls it contained internally (we capture both `v` and `v_mid`).
    For Euler one row = one transformer call.

    Norm split by mask:
      z_cond_norm: ‖z[conditioned positions]‖  → should stay ≈ clean_latents norm
      z_free_norm: ‖z[free positions]‖         → drives the solver error
    """

    COLUMNS = [
        "phase", "step_idx", "sigma_curr", "sigma_next", "sigma_mid", "dtau",
        "v_norm_raw", "v_norm_clamped",
        "v_mid_norm_raw", "v_mid_norm_clamped",
        "z_in_norm", "z_mid_norm", "z_next_norm",
        "z_cond_norm", "z_free_norm",
        "x0_pred_norm",
        "dt_s",
    ]

    def __init__(self, path: pathlib.Path):
        self.path = path
        self._fh = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.COLUMNS)
        self._writer.writeheader()

    def write(self, row: dict) -> None:
        # Fill any missing columns with empty string for partial rows (e.g. Euler).
        out = {col: row.get(col, "") for col in self.COLUMNS}
        self._writer.writerow(out)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "StepDiagnostics":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _norms_split_by_mask(z: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """Returns (cond_norm, free_norm). mask is [B, N, 1] in {0,1}."""
    m = mask.squeeze(-1).bool()  # [B, N]
    zf = z.float()                # [B, N, C]
    z_cond = zf[m.unsqueeze(-1).expand_as(zf)]
    z_free = zf[(~m).unsqueeze(-1).expand_as(zf)]
    cn = float(z_cond.norm().item()) if z_cond.numel() else 0.0
    fn = float(z_free.norm().item()) if z_free.numel() else 0.0
    return cn, fn


# ── RF-Solver inverter (refactored) ───────────────────────────────────────────

class RFInverter:
    """RF-Solver midpoint 2nd-order inversion + reconstruction + Euler
    regeneration for LTX-2 Stage 1.

    All phases share:
      - Per-token timestep `t·(1−mask)` (conditioned tokens see ~0 σ).
      - x0-domain clamp + hard re-clamp of conditioned positions.
      - Audio replay (Fix #4) — same `audio_hidden_states` sequence at
        every step, sourced from base-gen capture.

    Phase-specific:
      - invert/reconstruct: midpoint, CFG=1 (positive prompt only).
      - regenerate: Euler, CFG=gen_cfg (positive + negative prompts).
    """

    def __init__(self, pipe: LTX2ConditionPipeline, device: str = DEVICE) -> None:
        self.pipe = pipe
        self.device = device
        self.transformer = pipe.transformer
        self.vae = pipe.vae

        # Per-sample cache — POSITIVE-only views for CFG=1 phases.
        self.conditioning_mask: torch.Tensor | None = None
        self.clean_latents:     torch.Tensor | None = None
        self.prompt_embeds:          torch.Tensor | None = None  # [1, seq, d]
        self.prompt_attn_mask:       torch.Tensor | None = None
        self.audio_prompt_embeds:    torch.Tensor | None = None
        self.audio_prompt_attn_mask: torch.Tensor | None = None
        # COMBINED [neg ; pos] views for CFG>1 phases (matches pipeline line 1208).
        # When no negative_prompt is supplied these stay None and regenerate()
        # will fall back to CFG=1 behavior.
        self.prompt_embeds_cfg:          torch.Tensor | None = None  # [2, seq, d]
        self.prompt_attn_mask_cfg:       torch.Tensor | None = None
        self.audio_prompt_embeds_cfg:    torch.Tensor | None = None
        self.audio_prompt_attn_mask_cfg: torch.Tensor | None = None

        # Replay state (Fix #4) — list of dicts produced by AudioContextRecorder.
        self.audio_replay: list[dict] = []
        # Fallback zeros tensor (used when audio_strategy=zeros).
        self.audio_zeros: torch.Tensor | None = None
        self.audio_num_frames: int = 1
        self.latent_num_frames: int = 0
        self.latent_height:     int = 0
        self.latent_width:      int = 0
        self.frame_rate:        float = 24.0
        self.audio_strategy:    str = "capture_and_replay"

    # ── sample-level setup ────────────────────────────────────────────────────

    def prepare_sample(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        conditioning_mask: torch.Tensor,
        clean_latents: torch.Tensor,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
        frame_rate: float,
        audio_replay: list[dict] | None,
        audio_strategy: str = "capture_and_replay",
        max_sequence_length: int = 256,
    ) -> None:
        """Encode prompts (positive + optional negative), stash conditioning state,
        and load the audio trajectory from base generation.

        Mirrors `LTX2ConditionPipeline.__call__` lines 1190-1214 exactly:
        encode_prompt → (optional cat([neg, pos])) → connectors. Storing both
        the positive-only ([1, …]) and combined ([2, …]) views means CFG=1
        phases (invert, reconstruct) and CFG>1 phases (regenerate) share the
        same connector output, just sliced differently.
        """
        device = self.device
        has_neg = bool(negative_prompt and negative_prompt.strip())

        (pe, pm, ne, nm) = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt if has_neg else None,
            do_classifier_free_guidance=has_neg,
            num_videos_per_prompt=1,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        if has_neg:
            # uncond first — matches pipeline line 1208 exactly
            pe_full = torch.cat([ne, pe], dim=0)
            pm_full = torch.cat([nm, pm], dim=0)
        else:
            pe_full = pe
            pm_full = pm

        additive = (1 - pm_full.to(pe_full.dtype)) * -1_000_000.0
        cp_full, cap_full, cam_full = self.pipe.connectors(pe_full, additive, additive_mask=True)

        if has_neg:
            # Positive view = second half. Connector output is batch-wise.
            self.prompt_embeds          = cp_full[1:2]
            self.audio_prompt_embeds    = cap_full[1:2]
            self.prompt_attn_mask       = cam_full[1:2]
            self.audio_prompt_attn_mask = cam_full[1:2]
            self.prompt_embeds_cfg          = cp_full       # [2, seq, d]
            self.audio_prompt_embeds_cfg    = cap_full
            self.prompt_attn_mask_cfg       = cam_full
            self.audio_prompt_attn_mask_cfg = cam_full
        else:
            self.prompt_embeds          = cp_full
            self.audio_prompt_embeds    = cap_full
            self.prompt_attn_mask       = cam_full
            self.audio_prompt_attn_mask = cam_full
            self.prompt_embeds_cfg          = None
            self.audio_prompt_embeds_cfg    = None
            self.prompt_attn_mask_cfg       = None
            self.audio_prompt_attn_mask_cfg = None

        self.conditioning_mask = conditioning_mask.to(device=device)
        self.clean_latents     = clean_latents.to(device=device)
        self.latent_num_frames = latent_num_frames
        self.latent_height     = latent_height
        self.latent_width      = latent_width
        self.frame_rate        = frame_rate
        self.audio_strategy    = audio_strategy

        # Build the audio replay buffer. With capture_and_replay we use the
        # tensors recorded during base generation. With "zeros" (legacy) we
        # fall back to the exp_027 behavior — a single zeros tensor reused
        # at every step.
        if audio_strategy == "capture_and_replay":
            if not audio_replay:
                raise ValueError(
                    "audio_strategy='capture_and_replay' requires a non-empty "
                    "audio_replay list (typically captured during base generation)."
                )
            self.audio_replay = audio_replay
            self.audio_num_frames = audio_replay[0]["audio_hidden_states"].shape[2]
        elif audio_strategy == "zeros":
            # True zeros — NOT prepare_audio_latents(noise_scale=0, latents=None),
            # which exp_027 used and which actually returns randn
            # (_create_noised_state computes 0·new_randn + 1·initial_randn = randn).
            #
            # Shape: must match what the transformer ACTUALLY receives, which
            # is the 3D POST-_pack_audio_latents form (B, audio_num_frames,
            # audio_channels * mel_bins_compressed) — NOT the 4D pre-pack form
            # the pipeline's `prepare_audio_latents` returns. The pipeline calls
            # `_pack_audio_latents` (line 968/984 of pipeline_ltx2_condition.py)
            # before passing to the transformer.
            #
            # Easiest correct path: build zeros via `torch.zeros_like` against
            # the captured template from base gen (capture always runs, so
            # `audio_replay[0]` is available even for the zeros strategy).
            if not audio_replay:
                raise ValueError(
                    "audio_strategy='zeros' still requires the captured "
                    "audio_replay (for shape/dtype reference). Pass the "
                    "AudioContextRecorder output."
                )
            template = audio_replay[0]["audio_hidden_states"]   # CPU bf16, batch=1
            self.audio_zeros = torch.zeros_like(template).to(
                device=device, dtype=self.transformer.dtype
            )
            # audio_num_frames is the inner-most-but-one dim of the packed
            # tensor: shape (B, audio_num_frames, audio_channels*mel_bins).
            self.audio_num_frames = template.shape[1]
            self.audio_replay = []
        else:
            raise ValueError(f"Unknown audio_strategy: {audio_strategy!r}")

    # ── audio fetch (Fix #4) ──────────────────────────────────────────────────

    def _audio_for(self, phase: str, step_idx: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (audio_hidden_states, audio_timestep) for one transformer call.

        `phase`     : "invert" (replay reversed) or "forward" (gen-order replay).
        `step_idx`  : 0..N-1 of the *current phase loop*.
        `batch_size`: 1 for CFG=1 calls, 2 for CFG>1 batched (cond+uncond duplicate).
        """
        if self.audio_strategy == "zeros":
            ah = self.audio_zeros
            t_dtype = self.transformer.dtype
            at = torch.full((1,), 0.0, device=self.device, dtype=t_dtype)
            if batch_size == 2:
                ah = torch.cat([ah, ah], dim=0)
                at = torch.cat([at, at], dim=0)
            return ah.to(self.device, dtype=t_dtype), at

        # capture_and_replay
        N = len(self.audio_replay)
        if phase == "invert":
            gen_step = N - 1 - step_idx  # inversion step k undoes generation step (N-1-k)
        elif phase == "forward":
            gen_step = step_idx          # reconstruct / regenerate align with gen order
        else:
            raise ValueError(f"Unknown phase: {phase!r}")
        gen_step = max(0, min(N - 1, gen_step))

        rec = self.audio_replay[gen_step]
        ah = rec["audio_hidden_states"].to(self.device, dtype=self.transformer.dtype)
        at = rec["audio_timestep"].to(self.device, dtype=self.transformer.dtype)
        # Drop generation's batch dim if it was already CFG-doubled (we
        # stored only the first half; size matches single batch).
        if ah.shape[0] != 1:
            ah = ah[:1]
        if at.dim() >= 1 and at.shape[0] != 1:
            at = at[:1]
        if batch_size == 2:
            ah = torch.cat([ah, ah], dim=0)
            at = torch.cat([at, at], dim=0)
        return ah, at

    # ── transformer call ──────────────────────────────────────────────────────

    def _call_transformer(
        self,
        z_packed: torch.Tensor,
        sigma_scalar: float,
        *,
        phase: str,
        step_idx: int,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """One transformer call (or two batched if CFG>1). Returns velocity in fp32.

        For guidance_scale==1: single forward, positive prompt only.
        For guidance_scale > 1: batched cond+uncond, mixed per the LTX-2
        pipeline's recipe (line 1363-1372 of pipeline_ltx2_condition.py):

            v_cfg = v_uncond + s · (v_cond − v_uncond)
        """
        t_dtype = self.transformer.dtype
        do_cfg = guidance_scale > 1.0
        bsz_in = z_packed.shape[0]

        if do_cfg:
            if self.prompt_embeds_cfg is None:
                raise ValueError("CFG>1 requested but no negative_prompt was provided to prepare_sample.")
            z_in = torch.cat([z_packed, z_packed], dim=0).to(t_dtype)
            prompt_embeds     = self.prompt_embeds_cfg          # [2, seq, d] (neg ; pos)
            audio_prompt_emb  = self.audio_prompt_embeds_cfg
            prompt_attn_mask  = self.prompt_attn_mask_cfg
            audio_prompt_mask = self.audio_prompt_attn_mask_cfg
            ah, at = self._audio_for(phase, step_idx, batch_size=2)
        else:
            z_in = z_packed.to(t_dtype)
            prompt_embeds     = self.prompt_embeds
            audio_prompt_emb  = self.audio_prompt_embeds
            prompt_attn_mask  = self.prompt_attn_mask
            audio_prompt_mask = self.audio_prompt_attn_mask
            ah, at = self._audio_for(phase, step_idx, batch_size=1)

        # Per-token timestep: conditioned tokens see ~0 diffusion time.
        t_value = float(sigma_scalar) * NUM_TRAIN_TIMESTEPS
        bsz_call = z_in.shape[0]
        t = torch.full((bsz_call,), t_value, device=z_packed.device, dtype=t_dtype)
        cond_mask_t = self.conditioning_mask.squeeze(-1).to(t_dtype)
        if bsz_call == 2:
            cond_mask_t = torch.cat([cond_mask_t, cond_mask_t], dim=0)
        video_timestep = t.unsqueeze(-1) * (1 - cond_mask_t)

        noise_pred_video, _ = self.transformer(
            hidden_states=z_in,
            audio_hidden_states=ah,
            encoder_hidden_states=prompt_embeds,
            audio_encoder_hidden_states=audio_prompt_emb,
            timestep=video_timestep,
            audio_timestep=at,
            encoder_attention_mask=prompt_attn_mask,
            audio_encoder_attention_mask=audio_prompt_mask,
            num_frames=self.latent_num_frames,
            height=self.latent_height,
            width=self.latent_width,
            fps=self.frame_rate,
            audio_num_frames=self.audio_num_frames,
            video_coords=None,
            audio_coords=None,
            attention_kwargs=None,
            return_dict=False,
        )
        v = noise_pred_video.float()
        if do_cfg:
            v_uncond, v_cond = v.chunk(2)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        return v

    # ── x0 clamp ──────────────────────────────────────────────────────────────

    def _x0_clamp_velocity(
        self, z_packed: torch.Tensor, v_packed: torch.Tensor, sigma_scalar: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the LTX-2 pipeline's x0-domain clamp.

        Returns (v_clamped, x0_pred_pre_clamp). The x0_pred is the
        velocity-implied clean prediction at σ; useful for diagnostics
        (Fix #5) and tracked per step.

        At σ < 1e-4 we short-circuit (see CFG_AND_PROMPT.md §9 pitfall 5):
        the divide by σ otherwise squashes ALL velocity components to
        zero. The hard re-clamp in `_midpoint_step` keeps conditioned
        positions pinned.
        """
        sigma = float(sigma_scalar)
        x0_pred_pre = z_packed - v_packed * sigma  # for diagnostics
        if sigma < 1e-4:
            return v_packed, x0_pred_pre
        mask = self.conditioning_mask
        x0_pred_clean = x0_pred_pre * (1 - mask) + self.clean_latents.float() * mask
        v_clamped = (z_packed - x0_pred_clean) / sigma
        return v_clamped.to(v_packed.dtype), x0_pred_pre

    # ── midpoint step (used by invert + reconstruct) ──────────────────────────

    def _midpoint_step(
        self,
        z: torch.Tensor,
        sigma_curr: float,
        sigma_next: float,
        *,
        phase: str,
        step_idx: int,
        diag: StepDiagnostics | None,
        phase_label: str,
    ) -> torch.Tensor:
        dtau = sigma_next - sigma_curr
        sigma_mid = sigma_curr + dtau / 2.0
        mask = self.conditioning_mask
        t0 = time.perf_counter()

        v_raw = self._call_transformer(z, sigma_curr, phase=phase, step_idx=step_idx, guidance_scale=1.0)
        v, x0_pred = self._x0_clamp_velocity(z, v_raw, sigma_curr)

        z_mid = z + (dtau / 2.0) * v

        v_mid_raw = self._call_transformer(z_mid, sigma_mid, phase=phase, step_idx=step_idx, guidance_scale=1.0)
        v_mid, _ = self._x0_clamp_velocity(z_mid, v_mid_raw, sigma_mid)

        z_next = z + dtau * v_mid

        # Hard re-clamp of conditioned positions to clean clip latents.
        z_next = z_next * (1 - mask) + self.clean_latents * mask
        z_next = z_next.to(z.dtype)

        if diag is not None:
            cn, fn = _norms_split_by_mask(z_next, mask)
            diag.write({
                "phase":             phase_label,
                "step_idx":          step_idx,
                "sigma_curr":        f"{sigma_curr:.6f}",
                "sigma_next":        f"{sigma_next:.6f}",
                "sigma_mid":         f"{sigma_mid:.6f}",
                "dtau":              f"{dtau:.6f}",
                "v_norm_raw":        f"{v_raw.float().norm().item():.4f}",
                "v_norm_clamped":    f"{v.float().norm().item():.4f}",
                "v_mid_norm_raw":    f"{v_mid_raw.float().norm().item():.4f}",
                "v_mid_norm_clamped":f"{v_mid.float().norm().item():.4f}",
                "z_in_norm":         f"{z.float().norm().item():.4f}",
                "z_mid_norm":        f"{z_mid.float().norm().item():.4f}",
                "z_next_norm":       f"{z_next.float().norm().item():.4f}",
                "z_cond_norm":       f"{cn:.4f}",
                "z_free_norm":       f"{fn:.4f}",
                "x0_pred_norm":      f"{x0_pred.float().norm().item():.4f}",
                "dt_s":              f"{time.perf_counter() - t0:.3f}",
            })
        return z_next

    # ── Euler step (used by regenerate; mirrors pipeline_ltx2_condition.py) ───

    def _euler_step(
        self,
        z: torch.Tensor,
        sigma_curr: float,
        sigma_next: float,
        *,
        phase: str,
        step_idx: int,
        guidance_scale: float,
        diag: StepDiagnostics | None,
        phase_label: str,
    ) -> torch.Tensor:
        """One Euler step under CFG. Matches `LTX2ConditionPipeline.__call__`
        denoising loop (line 1339-1398): one (possibly batched) transformer
        call, CFG mix, x0 clamp, then z + dtau · v_clamped.
        """
        dtau = sigma_next - sigma_curr
        mask = self.conditioning_mask
        t0 = time.perf_counter()

        v_raw = self._call_transformer(
            z, sigma_curr, phase=phase, step_idx=step_idx, guidance_scale=guidance_scale
        )
        v, x0_pred = self._x0_clamp_velocity(z, v_raw, sigma_curr)
        z_next = z + dtau * v
        z_next = z_next * (1 - mask) + self.clean_latents * mask
        z_next = z_next.to(z.dtype)

        if diag is not None:
            cn, fn = _norms_split_by_mask(z_next, mask)
            diag.write({
                "phase":          phase_label,
                "step_idx":       step_idx,
                "sigma_curr":     f"{sigma_curr:.6f}",
                "sigma_next":     f"{sigma_next:.6f}",
                "sigma_mid":      "",
                "dtau":           f"{dtau:.6f}",
                "v_norm_raw":     f"{v_raw.float().norm().item():.4f}",
                "v_norm_clamped": f"{v.float().norm().item():.4f}",
                "v_mid_norm_raw":     "",
                "v_mid_norm_clamped": "",
                "z_in_norm":      f"{z.float().norm().item():.4f}",
                "z_mid_norm":     "",
                "z_next_norm":    f"{z_next.float().norm().item():.4f}",
                "z_cond_norm":    f"{cn:.4f}",
                "z_free_norm":    f"{fn:.4f}",
                "x0_pred_norm":   f"{x0_pred.float().norm().item():.4f}",
                "dt_s":           f"{time.perf_counter() - t0:.3f}",
            })
        return z_next

    # ── σ-grid builder ────────────────────────────────────────────────────────

    def _build_sigma_grid(self, num_steps: int, scheduler) -> np.ndarray:
        """The dynamic-shifted σ grid (length num_steps + 1, descending from
        σ_max≈1 to 0). Identical to what the Stage-1 pipeline computes for
        `num_inference_steps=num_steps`.
        """
        sigmas_seed = np.linspace(1.0, 1.0 / num_steps, num_steps)
        N = self.conditioning_mask.shape[1]
        mu = calculate_shift(
            N,
            scheduler.config.get("base_image_seq_len", 1024),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.95),
            scheduler.config.get("max_shift", 2.05),
        )
        sched = copy.deepcopy(scheduler)
        retrieve_timesteps(sched, num_steps, self.device, timesteps=None, sigmas=sigmas_seed, mu=mu)
        return sched.sigmas.cpu().numpy().astype(np.float64)

    # ── public solver methods ─────────────────────────────────────────────────

    @torch.inference_mode()
    def invert(
        self,
        z0_packed: torch.Tensor,
        num_steps: int,
        scheduler,
        checkpoint_sigmas: list[float],
        *,
        diag: StepDiagnostics | None = None,
    ) -> tuple[torch.Tensor, dict[float, torch.Tensor], np.ndarray]:
        """z0 (clean) → z1 (noise) via RF-Solver midpoint, σ ascending."""
        sigmas_gen = self._build_sigma_grid(num_steps, scheduler)
        sigmas_inv = sigmas_gen[::-1].copy()  # ascending

        z = z0_packed.clone()
        mask = self.conditioning_mask
        z = z * (1 - mask) + self.clean_latents * mask

        checkpoints: dict[float, torch.Tensor] = {}
        target_to_step: dict[float, int] = {}
        for target in checkpoint_sigmas:
            sigmas_after = sigmas_inv[1:]
            target_to_step[target] = int(np.argmin(np.abs(sigmas_after - target)))

        for i in range(len(sigmas_inv) - 1):
            sigma_curr = float(sigmas_inv[i])
            sigma_next = float(sigmas_inv[i + 1])
            z = self._midpoint_step(
                z, sigma_curr, sigma_next,
                phase="invert", step_idx=i, diag=diag, phase_label="invert",
            )
            log.info(
                "  invert step %2d/%d  σ: %.4f → %.4f   ‖z‖=%.2f",
                i + 1, len(sigmas_inv) - 1, sigma_curr, sigma_next, z.float().norm().item(),
            )
            for target, idx_after in target_to_step.items():
                if idx_after == i:
                    checkpoints[target] = z.detach().cpu().to(torch.bfloat16).clone()

        return z, checkpoints, sigmas_inv

    @torch.inference_mode()
    def reconstruct(
        self,
        z1_packed: torch.Tensor,
        num_steps: int,
        scheduler,
        *,
        diag: StepDiagnostics | None = None,
    ) -> torch.Tensor:
        """z1 (noise) → z0_recon (clean) via the SAME midpoint solver, σ descending.

        Tests SOLVER SELF-CONSISTENCY at CFG=1. Use `regenerate` for
        generation-trajectory recovery at CFG=gen_cfg.
        """
        sigmas_gen = self._build_sigma_grid(num_steps, scheduler)
        z = z1_packed.clone()
        mask = self.conditioning_mask
        z = z * (1 - mask) + self.clean_latents * mask

        for i in range(len(sigmas_gen) - 1):
            sigma_curr = float(sigmas_gen[i])
            sigma_next = float(sigmas_gen[i + 1])
            z = self._midpoint_step(
                z, sigma_curr, sigma_next,
                phase="forward", step_idx=i, diag=diag, phase_label="reconstruct",
            )
            log.info(
                "  recon  step %2d/%d  σ: %.4f → %.4f   ‖z‖=%.2f",
                i + 1, len(sigmas_gen) - 1, sigma_curr, sigma_next, z.float().norm().item(),
            )
        return z

    @torch.inference_mode()
    def regenerate(
        self,
        z1_packed: torch.Tensor,
        num_steps: int,
        scheduler,
        guidance_scale: float,
        *,
        diag: StepDiagnostics | None = None,
    ) -> torch.Tensor:
        """z1 (noise) → z0_regen (clean) via Euler + CFG=guidance_scale.

        Mirrors `LTX2ConditionPipeline.__call__` exactly (line 1339-1398).
        Tests GENERATION-TRAJECTORY RECOVERY: does the inverted z1 round-trip
        back to z0 under the same flow that produced z0?
        """
        sigmas_gen = self._build_sigma_grid(num_steps, scheduler)
        z = z1_packed.clone()
        mask = self.conditioning_mask
        z = z * (1 - mask) + self.clean_latents * mask

        for i in range(len(sigmas_gen) - 1):
            sigma_curr = float(sigmas_gen[i])
            sigma_next = float(sigmas_gen[i + 1])
            z = self._euler_step(
                z, sigma_curr, sigma_next,
                phase="forward", step_idx=i, guidance_scale=guidance_scale,
                diag=diag, phase_label="regenerate",
            )
            log.info(
                "  regen  step %2d/%d  σ: %.4f → %.4f   ‖z‖=%.2f",
                i + 1, len(sigmas_gen) - 1, sigma_curr, sigma_next, z.float().norm().item(),
            )
        return z


# ── Latent ↔ pixel helpers ────────────────────────────────────────────────────

def normalize_and_pack(pipe: LTX2ConditionPipeline, latents_5d_denorm: torch.Tensor) -> torch.Tensor:
    z = pipe._normalize_latents(
        latents_5d_denorm,
        pipe.vae.latents_mean,
        pipe.vae.latents_std,
        pipe.vae.config.scaling_factor,
    )
    z = pipe._pack_latents(z, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size)
    return z


def unpack_and_denormalize(
    pipe: LTX2ConditionPipeline,
    z_packed: torch.Tensor,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    z = pipe._unpack_latents(
        z_packed,
        latent_num_frames,
        latent_height,
        latent_width,
        pipe.transformer_spatial_patch_size,
        pipe.transformer_temporal_patch_size,
    )
    z = pipe._denormalize_latents(
        z, pipe.vae.latents_mean, pipe.vae.latents_std, pipe.vae.config.scaling_factor
    )
    return z


@torch.inference_mode()
def decode_latents_to_video(
    pipe: LTX2ConditionPipeline,
    latents_5d_denorm: torch.Tensor,
) -> np.ndarray:
    latents = latents_5d_denorm.to(pipe.vae.dtype)
    timestep = None
    if pipe.vae.config.timestep_conditioning:
        timestep = torch.tensor([0.0], device=latents.device, dtype=latents.dtype)
    video = pipe.vae.decode(latents, timestep, return_dict=False)[0]
    video = pipe.video_processor.postprocess_video(video, output_type="np")
    return (np.clip(video[0], 0.0, 1.0) * 255).astype(np.uint8)


# ── MetricSuite (unchanged from exp_027) ──────────────────────────────────────

def _stats(arr: np.ndarray, *, key: str) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    worst = int(arr.argmin()) if key in ("psnr", "ssim", "cosine") else int(arr.argmax())
    return {
        "mean":        float(arr.mean()),
        "std":         float(arr.std()),
        "min":         float(arr.min()),
        "max":         float(arr.max()),
        "worst_frame": worst,
        "per_frame":   [float(x) for x in arr.tolist()],
    }


class MetricSuite:
    def __init__(self, device: str = DEVICE) -> None:
        self.device = device
        self.lpips_model = lpips_pkg.LPIPS(net="alex", verbose=False).to(device).eval()

    @staticmethod
    def psnr(a: np.ndarray, b: np.ndarray) -> dict:
        af = a.astype(np.float64) / 255.0
        bf = b.astype(np.float64) / 255.0
        mse = ((af - bf) ** 2).reshape(a.shape[0], -1).mean(axis=1)
        psnr = 10.0 * np.log10(1.0 / np.maximum(mse, 1e-12))
        return _stats(psnr, key="psnr")

    @staticmethod
    def ssim(a: np.ndarray, b: np.ndarray) -> dict:
        ssims = np.array(
            [_ssim_skimage(fa, fb, channel_axis=-1, data_range=255) for fa, fb in zip(a, b)],
            dtype=np.float64,
        )
        return _stats(ssims, key="ssim")

    @staticmethod
    def temporal(a: np.ndarray, b: np.ndarray) -> dict:
        if a.shape[0] < 2:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                    "worst_frame": 0, "per_frame": []}
        af = a.astype(np.float64) / 255.0
        bf = b.astype(np.float64) / 255.0
        d_src = af[1:] - af[:-1]
        d_rec = bf[1:] - bf[:-1]
        per_pair = np.abs(d_src - d_rec).reshape(a.shape[0] - 1, -1).mean(axis=1)
        return _stats(per_pair, key="temporal")

    @torch.inference_mode()
    def lpips(self, a: np.ndarray, b: np.ndarray) -> dict:
        assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"

        def _prep(v: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(v).to(self.device).float() / 127.5 - 1.0
            return t.permute(0, 3, 1, 2).contiguous()

        ta = _prep(a); tb = _prep(b)
        scores = []
        for i in range(0, ta.shape[0], 8):
            d = self.lpips_model(ta[i:i + 8], tb[i:i + 8]).flatten()
            scores.append(d.detach().cpu())
        return _stats(torch.cat(scores).numpy(), key="lpips")

    @staticmethod
    def latent(z_a: torch.Tensor, z_b: torch.Tensor) -> dict:
        a = z_a.float().flatten()
        b = z_b.float().flatten()
        diff = a - b
        l2 = float(diff.norm().item())
        n = int(a.numel())
        norm_a = float(a.norm().item())
        norm_b = float(b.norm().item())
        rel = l2 / max(norm_a, 1e-8)
        cos = float(torch.dot(a, b).item()) / max(norm_a * norm_b, 1e-8)
        return {
            "l2": l2, "l2_per_element": l2 / (n ** 0.5),
            "relative": rel, "cosine": cos,
            "n_elements": n, "norm_a": norm_a, "norm_b": norm_b,
        }

    def evaluate(
        self,
        src_video: np.ndarray,
        recon_video: np.ndarray,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
    ) -> dict:
        return {
            "psnr":     self.psnr(src_video, recon_video),
            "ssim":     self.ssim(src_video, recon_video),
            "lpips":    self.lpips(src_video, recon_video),
            "temporal": self.temporal(src_video, recon_video),
            "latent":   self.latent(z_a, z_b),
        }


# ── Video save ────────────────────────────────────────────────────────────────

def save_video(path: pathlib.Path, video_uint8: np.ndarray, fps: int) -> None:
    frames = [Image.fromarray(f) for f in video_uint8]
    export_to_video(frames, str(path), fps=int(fps))


# ── Main ─────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg     = load_config(args.config)
    out_dir = REPO_ROOT / cfg["outputs"]["dir"]
    run_id, run_dir = next_run_dir(out_dir)

    with TeeLogger(run_dir / "run.log"):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
            force=True,
        )
        print(f"[info] run_dir : {run_dir}")
        print(f"[info] samples : {len(cfg['samples'])}")

        # Parse config
        model_id        = cfg["model"]["model_id"]
        num_frames      = cfg["inference"]["num_frames"]
        frame_rate      = float(cfg["inference"]["frame_rate"])
        height          = cfg["inference"]["height"]
        width           = cfg["inference"]["width"]
        gen_steps       = cfg["inference"]["num_inference_steps"]
        gen_cfg         = cfg["inference"]["guidance_scale"]
        seed            = cfg["runtime"]["seed"]
        num_clip_frames = cfg["inputs"]["num_clip_frames"]
        start_strength  = cfg["inputs"]["start_clip_strength"]
        end_strength    = cfg["inputs"]["end_clip_strength"]
        negative_prompt = cfg["inputs"]["negative_prompt"].strip()

        inv_cfg         = cfg["inversion"]
        inv_steps       = inv_cfg["num_steps"]
        inv_cfg_scale   = inv_cfg["guidance_scale"]
        inv_solver      = inv_cfg["solver"]
        ckpt_sigmas     = list(inv_cfg["cache_sigma_checkpoints"])
        audio_strategy  = inv_cfg.get("audio_strategy", "capture_and_replay")
        # Gate thresholds — informational only; logged per condition, never gate the run.
        gate_cfg        = inv_cfg.get("gate", inv_cfg.get("escalation", {}))
        latent_rel_max          = float(gate_cfg["latent_rel_max"])
        latent_cos_min          = float(gate_cfg["latent_cos_min"])
        latent_rel_max_mult     = float(gate_cfg.get("latent_rel_max_regen_mult", 2.0))
        latent_cos_min_regen    = float(gate_cfg.get("latent_cos_min_regen", latent_cos_min - 0.02))

        regen_cfg       = cfg["regeneration"]
        regen_steps     = int(regen_cfg["num_steps"])
        regen_cfg_scale = float(regen_cfg["guidance_scale"])

        if inv_cfg_scale != 1.0:
            log.warning(
                "inversion.guidance_scale = %.2f != 1.0 — invert/reconstruct ALWAYS "
                "execute at CFG=1 here (RF-Solver invertibility requirement). "
                "Use the regeneration block to test CFG>1 trajectory recovery.",
                inv_cfg_scale,
            )

        end_idx = end_clip_index(num_frames, num_clip_frames)

        log.info("Loading LTX2ConditionPipeline from %s …", model_id)
        t0 = time.perf_counter()
        pipe = LTX2ConditionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload(device=DEVICE)
        pipe.vae.enable_tiling()
        log.info("Pipeline loaded in %.1fs.", time.perf_counter() - t0)
        stage1_scheduler = pipe.scheduler

        metric_suite = MetricSuite(device=DEVICE)

        summary: list[dict] = []

        for idx, sample in enumerate(cfg["samples"]):
            sample_id  = sample["sample_id"]
            prompt     = sample["prompt"].strip()
            sample_dir = run_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            log.info("─── Sample %d/%d  id=%s  (difficulty=%s) ───",
                     idx + 1, len(cfg["samples"]), sample_id, sample.get("difficulty", "?"))

            start_frames, end_frames = load_clip_frames(sample, REPO_ROOT, num_clip_frames)
            conditions = [
                LTX2VideoCondition(frames=start_frames, index=0,       strength=start_strength),
                LTX2VideoCondition(frames=end_frames,   index=end_idx, strength=end_strength),
            ]
            generator = torch.Generator(device=DEVICE).manual_seed(seed)

            # ── (a) Stage-1 generation with audio capture ─────────────────────
            pipe.scheduler = stage1_scheduler
            pipe.disable_lora()
            log.info("Stage 1 generation: %dx%d  %d steps  guidance=%.1f",
                     height, width, gen_steps, gen_cfg)

            t_gen = time.perf_counter()
            with AudioContextRecorder(pipe.transformer) as audio_rec:
                video_latent_5d, _audio_latent = pipe(
                    conditions=conditions,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    num_inference_steps=gen_steps,
                    sigmas=None,
                    guidance_scale=gen_cfg,
                    generator=generator,
                    output_type="latent",
                    return_dict=False,
                )
            audio_summary = audio_rec.summary()
            log.info("Stage 1 done in %.1fs.  latent shape=%s",
                     time.perf_counter() - t_gen, tuple(video_latent_5d.shape))
            log.info(
                "Audio capture: %d transformer calls (=2×%d for CFG); shape=%s, "
                "norm: first=%.3f → last=%.3f, dtype=%s",
                audio_summary["num_calls"], gen_steps, audio_summary["shape"],
                audio_summary["first_norm"], audio_summary["last_norm"], audio_summary["dtype"],
            )

            # Each generation step makes 2 transformer calls (cond+uncond
            # batched). Both halves share audio_hidden_states (see
            # AudioContextRecorder docstring). One captured record per
            # transformer call → take every-other-record to get one per
            # gen step. Since CFG > 1, both halves are identical so
            # taking either works — we use even indices.
            if audio_summary["num_calls"] == 2 * gen_steps:
                audio_trajectory = audio_rec.captures[::2]
            elif audio_summary["num_calls"] == gen_steps:
                audio_trajectory = list(audio_rec.captures)
            else:
                raise RuntimeError(
                    f"Unexpected #transformer calls during base gen: "
                    f"{audio_summary['num_calls']} (expected {gen_steps} or {2*gen_steps})."
                )
            # Save the captured audio trajectory for forensic inspection
            # (heavy — bf16, ~MB-scale; user can delete if not needed).
            torch.save(
                {
                    "audio_hidden_states": [r["audio_hidden_states"].to(torch.bfloat16) for r in audio_trajectory],
                    "audio_timestep":      [r["audio_timestep"]      for r in audio_trajectory],
                    "video_timestep":      [r["video_timestep"]      for r in audio_trajectory],
                    "num_gen_steps":       gen_steps,
                    "shape":               audio_summary["shape"],
                    "dtype":               audio_summary["dtype"],
                },
                sample_dir / "audio_record.pt",
            )

            # ── Pack + normalize Stage-1 output → z0 ─────────────────────────
            z0_packed = normalize_and_pack(pipe, video_latent_5d).to(DEVICE)
            log.info("z0 packed shape: %s  norm=%.2f",
                     tuple(z0_packed.shape), z0_packed.float().norm().item())

            latent_height = height // pipe.vae_spatial_compression_ratio
            latent_width  = width  // pipe.vae_spatial_compression_ratio
            latent_num_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1

            condition_frames, condition_strengths, condition_indices = pipe.preprocess_conditions(
                conditions, height, width, num_frames, device=DEVICE
            )
            cond_latents_list: list[torch.Tensor] = []
            for cond_tensor in condition_frames:
                # Fix #6 — retrieve_latents determinism contract.
                # sample_mode="argmax" returns the posterior MODE (deterministic).
                # `generator` is passed for signature compatibility but is UNUSED
                # in argmax mode. Do not change to sample_mode="sample" without
                # auditing every downstream cache and gate: stochastic conditioning
                # latents would silently break round-trip identity across runs.
                cl = retrieve_latents(pipe.vae.encode(cond_tensor), generator=generator, sample_mode="argmax")
                cl = pipe._normalize_latents(cl, pipe.vae.latents_mean, pipe.vae.latents_std).to(
                    device=DEVICE, dtype=z0_packed.dtype
                )
                cl = pipe._pack_latents(cl, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size)
                cond_latents_list.append(cl)

            zeros_5d = torch.zeros(
                (1, 1, latent_num_frames, latent_height, latent_width),
                device=DEVICE, dtype=z0_packed.dtype,
            )
            cmask_packed = pipe._pack_latents(
                zeros_5d, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size
            )
            throwaway = torch.zeros_like(z0_packed)
            _, cmask_packed, clean_latents = pipe.apply_visual_conditioning(
                throwaway, cmask_packed, cond_latents_list,
                condition_strengths, condition_indices,
                latent_height=latent_height, latent_width=latent_width,
            )
            log.info("conditioning_mask shape=%s   active tokens=%d / %d",
                     tuple(cmask_packed.shape),
                     int((cmask_packed > 0).sum().item()),
                     cmask_packed.shape[1])

            # ── Prepare inverter ─────────────────────────────────────────────
            inverter = RFInverter(pipe, device=DEVICE)
            inverter.prepare_sample(
                prompt=prompt,
                negative_prompt=negative_prompt,
                conditioning_mask=cmask_packed,
                clean_latents=clean_latents,
                latent_num_frames=latent_num_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                frame_rate=frame_rate,
                audio_replay=audio_trajectory,
                audio_strategy=audio_strategy,
            )

            # ── (b) + (c) + (d): canonical-step attempt ───────────────────────
            def _attempt(n_steps: int) -> dict:
                log.info(
                    "─ Attempt: invert+recon=%d midpoint steps (CFG=%.1f), regen=%d Euler steps (CFG=%.1f) ─",
                    n_steps, inv_cfg_scale, regen_steps, regen_cfg_scale,
                )

                diag_invert     = StepDiagnostics(sample_dir / f"step_diag_invert_n{n_steps}.csv")
                diag_reconstruct = StepDiagnostics(sample_dir / f"step_diag_reconstruct_n{n_steps}.csv")
                diag_regenerate  = StepDiagnostics(sample_dir / f"step_diag_regenerate_n{regen_steps}_n{n_steps}.csv")

                try:
                    t_inv = time.perf_counter()
                    z1, checkpoints, sigmas_inv = inverter.invert(
                        z0_packed, n_steps, stage1_scheduler, ckpt_sigmas, diag=diag_invert,
                    )
                    t_inv_done = time.perf_counter() - t_inv
                    log.info("Inversion done in %.1fs.  z1 norm=%.2f", t_inv_done, z1.float().norm().item())

                    t_rec = time.perf_counter()
                    z0_recon = inverter.reconstruct(z1, n_steps, stage1_scheduler, diag=diag_reconstruct)
                    t_rec_done = time.perf_counter() - t_rec
                    log.info("Reconstruction done in %.1fs.  z0_recon norm=%.2f", t_rec_done, z0_recon.float().norm().item())

                    t_reg = time.perf_counter()
                    z0_regen = inverter.regenerate(
                        z1, regen_steps, stage1_scheduler, regen_cfg_scale, diag=diag_regenerate,
                    )
                    t_reg_done = time.perf_counter() - t_reg
                    log.info("Regeneration done in %.1fs.  z0_regen norm=%.2f", t_reg_done, z0_regen.float().norm().item())
                finally:
                    diag_invert.close()
                    diag_reconstruct.close()
                    diag_regenerate.close()

                src_5d   = unpack_and_denormalize(pipe, z0_packed, latent_num_frames, latent_height, latent_width)
                recon_5d = unpack_and_denormalize(pipe, z0_recon,  latent_num_frames, latent_height, latent_width)
                regen_5d = unpack_and_denormalize(pipe, z0_regen,  latent_num_frames, latent_height, latent_width)
                src_video   = decode_latents_to_video(pipe, src_5d)
                recon_video = decode_latents_to_video(pipe, recon_5d)
                regen_video = decode_latents_to_video(pipe, regen_5d)

                m_recon = metric_suite.evaluate(src_video, recon_video, z0_packed, z0_recon)
                m_regen = metric_suite.evaluate(src_video, regen_video, z0_packed, z0_regen)

                log.info(
                    "[inv_recon] latent rel=%.4f cos=%.5f  |  PSNR=%.2f SSIM=%.4f LPIPS=%.4f",
                    m_recon["latent"]["relative"], m_recon["latent"]["cosine"],
                    m_recon["psnr"]["mean"], m_recon["ssim"]["mean"], m_recon["lpips"]["mean"],
                )
                log.info(
                    "[inv_regen] latent rel=%.4f cos=%.5f  |  PSNR=%.2f SSIM=%.4f LPIPS=%.4f",
                    m_regen["latent"]["relative"], m_regen["latent"]["cosine"],
                    m_regen["psnr"]["mean"], m_regen["ssim"]["mean"], m_regen["lpips"]["mean"],
                )

                return {
                    "num_steps":  n_steps,
                    "t_inv_s":    round(t_inv_done, 1),
                    "t_rec_s":    round(t_rec_done, 1),
                    "t_reg_s":    round(t_reg_done, 1),
                    "z1":         z1.detach().cpu().to(torch.bfloat16),
                    "z0_recon":   z0_recon.detach().cpu().to(torch.bfloat16),
                    "z0_regen":   z0_regen.detach().cpu().to(torch.bfloat16),
                    "checkpoints": checkpoints,
                    "sigmas_inv": sigmas_inv.tolist(),
                    "metrics_recon": m_recon,
                    "metrics_regen": m_regen,
                    "src_video":   src_video,
                    "recon_video": recon_video,
                    "regen_video": regen_video,
                }

            def _gate(a: dict) -> tuple[bool, dict]:
                lr = a["metrics_recon"]["latent"]
                rr = a["metrics_regen"]["latent"]
                rel_ok_recon = lr["relative"] < latent_rel_max
                cos_ok_recon = lr["cosine"]   > latent_cos_min
                rel_ok_regen = rr["relative"] < latent_rel_max * latent_rel_max_mult
                cos_ok_regen = rr["cosine"]   > latent_cos_min_regen
                all_ok = rel_ok_recon and cos_ok_recon and rel_ok_regen and cos_ok_regen
                return all_ok, {
                    "recon_relative": lr["relative"],
                    "recon_cosine":   lr["cosine"],
                    "regen_relative": rr["relative"],
                    "regen_cosine":   rr["cosine"],
                    "recon_rel_ok":   rel_ok_recon,
                    "recon_cos_ok":   cos_ok_recon,
                    "regen_rel_ok":   rel_ok_regen,
                    "regen_cos_ok":   cos_ok_regen,
                }

            attempts: list[dict] = []
            attempt = _attempt(inv_steps)
            attempts.append(attempt)

            # Gate is INFORMATIONAL only — no retry. The dominant gate failure
            # mode (regen_rel ≫ ceiling) reflects the CFG=1↔CFG=gen flow mismatch
            # by design, not a fixable solver-quality problem. Log per-condition
            # pass/fail so the operator can read it at a glance.
            gate_passed, gate_info = _gate(attempt)

            def _mark(ok: bool) -> str:
                return "PASS" if ok else "FAIL"

            log.info("[gate]  (informational; no retry)")
            log.info(
                "  recon  rel=%.4f  (< %.3f)  → %s   |   cos=%.5f  (> %.4f)  → %s",
                gate_info["recon_relative"], latent_rel_max, _mark(gate_info["recon_rel_ok"]),
                gate_info["recon_cosine"],   latent_cos_min, _mark(gate_info["recon_cos_ok"]),
            )
            log.info(
                "  regen  rel=%.4f  (< %.3f)  → %s   |   cos=%.5f  (> %.4f)  → %s",
                gate_info["regen_relative"], latent_rel_max * latent_rel_max_mult, _mark(gate_info["regen_rel_ok"]),
                gate_info["regen_cosine"],   latent_cos_min_regen,                 _mark(gate_info["regen_cos_ok"]),
            )
            log.info("  verdict: %s", _mark(gate_passed))

            # ── Persist artefacts ────────────────────────────────────────────
            torch.save(z0_packed.detach().cpu().to(torch.bfloat16), sample_dir / "z0.pt")
            torch.save(attempt["z1"], sample_dir / "z1.pt")
            for target, tensor in attempt["checkpoints"].items():
                tag = f"z_t_{int(round(target * 100)):02d}.pt"
                torch.save(tensor, sample_dir / tag)
            torch.save(attempt["z0_recon"], sample_dir / "z0_recon.pt")
            torch.save(attempt["z0_regen"], sample_dir / "z0_regen.pt")

            save_video(sample_dir / "source_video.mp4", attempt["src_video"],   fps=int(frame_rate))
            save_video(sample_dir / "recon_video.mp4",  attempt["recon_video"], fps=int(frame_rate))
            save_video(sample_dir / "regen_video.mp4",  attempt["regen_video"], fps=int(frame_rate))

            meta = {
                "sample_id":   sample_id,
                "prompt":      prompt,
                "seed":        seed,
                "clip_conditioning": {
                    "num_clip_frames": num_clip_frames,
                    "start_index":     0,
                    "end_index":       end_idx,
                    "start_strength":  start_strength,
                    "end_strength":    end_strength,
                },
                "scheduler_config": dict(stage1_scheduler.config),
                "solver":      inv_solver,
                "inversion_cfg":   inv_cfg_scale,
                "regeneration_cfg": regen_cfg_scale,
                "audio_strategy":  audio_strategy,
                "audio_capture":   audio_summary,
                "generation": {
                    "num_steps":      gen_steps,
                    "guidance_scale": gen_cfg,
                },
                "regeneration": {
                    "num_steps":      regen_steps,
                    "guidance_scale": regen_cfg_scale,
                    "solver":         regen_cfg.get("solver", "euler"),
                },
                "attempts": [
                    {
                        "num_steps":     a["num_steps"],
                        "nfe_invert":    a["num_steps"] * 2,
                        "nfe_recon":     a["num_steps"] * 2,
                        "nfe_regen":     regen_steps * 2,  # Euler+CFG=4 → 2 calls/step
                        "t_inv_s":       a["t_inv_s"],
                        "t_rec_s":       a["t_rec_s"],
                        "t_reg_s":       a["t_reg_s"],
                        "metrics_recon": a["metrics_recon"],
                        "metrics_regen": a["metrics_regen"],
                        "sigmas_inv":    a["sigmas_inv"],
                    }
                    for a in attempts
                ],
                "gate": {
                    "passed":                gate_passed,
                    "latent_rel_max":        latent_rel_max,
                    "latent_cos_min":        latent_cos_min,
                    "latent_rel_max_regen":  latent_rel_max * latent_rel_max_mult,
                    "latent_cos_min_regen":  latent_cos_min_regen,
                    **gate_info,
                },
                "checkpoint_sigmas_targets":  ckpt_sigmas,
                "checkpoint_sigmas_actual":   list(attempt["checkpoints"].keys()),
            }
            with (sample_dir / "inv_meta.yaml").open("w") as f:
                yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

            mr = attempt["metrics_recon"]
            mg = attempt["metrics_regen"]
            summary.append({
                "sample_id":      sample_id,
                "difficulty":     sample.get("difficulty"),
                "gate": {
                    "verdict":              gate_passed,
                    "recon_rel_ok":         gate_info["recon_rel_ok"],
                    "recon_cos_ok":         gate_info["recon_cos_ok"],
                    "regen_rel_ok":         gate_info["regen_rel_ok"],
                    "regen_cos_ok":         gate_info["regen_cos_ok"],
                    "thresholds": {
                        "recon_rel_max":    latent_rel_max,
                        "recon_cos_min":    latent_cos_min,
                        "regen_rel_max":    latent_rel_max * latent_rel_max_mult,
                        "regen_cos_min":    latent_cos_min_regen,
                    },
                },
                "num_steps_used": attempt["num_steps"],
                # inv_recon (solver self-consistency)
                "recon_latent_rel": mr["latent"]["relative"],
                "recon_latent_cos": mr["latent"]["cosine"],
                "recon_psnr_mean":  mr["psnr"]["mean"],
                "recon_ssim_mean":  mr["ssim"]["mean"],
                "recon_lpips_mean": mr["lpips"]["mean"],
                # inv_regen (generation-trajectory recovery)
                "regen_latent_rel": mg["latent"]["relative"],
                "regen_latent_cos": mg["latent"]["cosine"],
                "regen_psnr_mean":  mg["psnr"]["mean"],
                "regen_ssim_mean":  mg["ssim"]["mean"],
                "regen_lpips_mean": mg["lpips"]["mean"],
            })

            del attempts, attempt
            torch.cuda.empty_cache()

        with (run_dir / "config_snapshot.yaml").open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        with (run_dir / "summary.yaml").open("w") as f:
            yaml.safe_dump({"run_id": run_id, "samples": summary}, f, sort_keys=False, allow_unicode=True)

        passed = sum(1 for s in summary if s["gate"]["verdict"])

        def _row(*cells: str, widths: list[int]) -> str:
            return "│ " + " │ ".join(c.ljust(w) for c, w in zip(cells, widths)) + " │"

        def _sep(widths: list[int], left: str, mid: str, right: str) -> str:
            return left + mid.join("─" * (w + 2) for w in widths) + right

        widths = [42, 6, 4, 7, 8, 6, 6, 7, 8, 6, 6]
        headers_top = ["sample_id", "diff", "gate",
                       "rec_rel", "rec_cos", "PSNR", "SSIM",
                       "reg_rel", "reg_cos", "PSNR", "SSIM"]

        log.info("")
        log.info("RF inversion v2 — metrics summary  (gate is informational; no retry)")
        log.info(
            "Thresholds: recon rel<%.3f & cos>%.4f  |  regen rel<%.3f & cos>%.4f",
            latent_rel_max, latent_cos_min,
            latent_rel_max * latent_rel_max_mult, latent_cos_min_regen,
        )
        log.info(_sep(widths, "┌", "┬", "┐"))
        log.info(_row(*headers_top, widths=widths))
        log.info(_sep(widths, "├", "┼", "┤"))
        for s in summary:
            verdict = "PASS" if s["gate"]["verdict"] else "FAIL"
            log.info(_row(
                s["sample_id"][:42],
                (s["difficulty"] or "?")[:6],
                verdict,
                f"{s['recon_latent_rel']:.4f}",
                f"{s['recon_latent_cos']:.5f}",
                f"{s['recon_psnr_mean']:.2f}",
                f"{s['recon_ssim_mean']:.4f}",
                f"{s['regen_latent_rel']:.4f}",
                f"{s['regen_latent_cos']:.5f}",
                f"{s['regen_psnr_mean']:.2f}",
                f"{s['regen_ssim_mean']:.4f}",
                widths=widths,
            ))
        log.info(_sep(widths, "└", "┴", "┘"))
        log.info("Gate verdict: %d/%d samples PASS.", passed, len(summary))


if __name__ == "__main__":
    main()
