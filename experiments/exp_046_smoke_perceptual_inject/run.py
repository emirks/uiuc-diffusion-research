"""exp_033 — LTX-2 RF-Solver inversion with end-clip first-latent-frame dropout.

Iteration It-2 of the RF-inversion research loop (notes/rf_inversion_loop.md).

Motivation. exp_032's TRUE self-conditioning (clean_latents = exact z0 slices)
cleared the perceptual bar (8/10 pass) but is NOT deployable for real editing:
in production C2V you only have the two endpoint sub-clips, never z0 from a
full-video encode, so anchors must be sub-clip-isolated. exp_032's anchors
also leak middle-frame info into the end-clip position (causal VAE), which
defeats the editable-boundary property.

The discontinuity is concentrated at ONE position. LTX-2's causal video VAE
encodes the first latent frame of any video as a fresh-start anchor (no
temporal context). For the start sub-clip this matches the full-video encode
(both start there). For the END sub-clip it does NOT: in the standalone
sub-clip encode it is again a fresh start, but in the full-video encode the
corresponding latent position carries causal info from the entire middle of
the clip. That single position is maximally discontinuous; deeper positions
into the end sub-clip carry causal context from within the sub-clip and are
closer to the full-encode values.

Fork of exp_032 with EXACTLY ONE change: restore production sub-clip anchors
(as in exp_030) but ZERO OUT the conditioning mask at the first latent frame
of the end sub-clip — i.e. drop the worst-discontinuity position from the
hard-pin. The solver fills that one position freely; the remaining boundary
data is what production would supply.

Hypothesis. If the end-clip first-latent-frame mismatch is the dominant cost
of production-anchored inversion, exp_033's recon should approach exp_032's
numbers (PSNR median ≈ 40, 8+/10 pass) while using deployable anchors. If the
mismatch is distributed across the whole sub-clip, exp_033 stays closer to
exp_030's collapse — and we move to richer fixes (deeper dropout, soft pin,
anchor warmup).

Everything else is identical to exp_030/032: per-clip `max_area` resolution,
encoded-silent audio context, invert+recon+regen, informational dual gate.

Workflow per sample (under run_dir/<sample_id>/):
  (a) Load source clip → VAE encode → z0  (audio: encoded silent mel)
  (b) Inversion (CFG=1)      → z1       — solver self-consistency
  (c) Reconstruction (CFG=1) → z0_recon
  (d) Regeneration (CFG=gen) → z0_regen — production-trajectory recovery
  (e) MetricSuite over (z0,z0_recon) and (z0,z0_regen); dual gate (informational).

Saved per sample:
  z0.pt, z1.pt, z_t_25/50/75.pt   — packed bfloat16 (1, N, 128)
  source_video.mp4                 — decode(z0) for visual reference
  recon_video.mp4                  — decode(z0_recon)
  regen_video.mp4                  — decode(z0_regen)
  step_diag_<phase>.csv            — per-step diagnostics for each phase
  inv_meta.yaml                    — full metrics, gate status
"""
from __future__ import annotations

import argparse
import copy
import csv
import logging
import pathlib
import sys
import time

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

from diffusion.exp_utils import load_config, next_run_dir, resolve_resolution, TeeLogger

REPO_ROOT      = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = pathlib.Path(__file__).parent / "config.yaml"

LTX_TEMPORAL_SCALE = 8
DEVICE             = "cuda:0"
NUM_TRAIN_TIMESTEPS = 1000  # FlowMatchEulerDiscreteScheduler default; t = sigma * 1000

log = logging.getLogger(__name__)


# ── Frame / clip helpers ─────────────────────────────────────────────────────

def load_source_clip(
    path: pathlib.Path, num_frames: int
) -> tuple[list[Image.Image], tuple[int, int]]:
    """Load the first `num_frames` PIL frames from a single source mp4.

    Returns (frames, (H, W)) where H, W are the *source* dimensions before
    any resize. Used both to feed VAE encoding and to derive a target
    resolution via resolve_resolution(max_area, ref_image).
    """
    video, _, _ = tio.read_video(str(path), pts_unit="sec", output_format="THWC")
    if video.shape[0] == 0:
        raise ValueError(f"No frames decoded from {path}")
    if video.shape[0] < num_frames:
        raise ValueError(
            f"{path.name} only has {video.shape[0]} frames; need ≥ {num_frames}."
        )
    src = video[:num_frames]
    src_hw = (int(src.shape[1]), int(src.shape[2]))
    return [Image.fromarray(f.numpy()) for f in src], src_hw


def build_silent_audio_context(
    pipe: "LTX2ConditionPipeline", num_frames: int, frame_rate: float,
    device: str, dtype: torch.dtype,
) -> torch.Tensor:
    """Encode a literal silent mel spectrogram through `pipe.audio_vae` and
    return a fixed, in-distribution audio context to use at every step of
    invert/recon/regen.

    Rationale: shoving torch.zeros into the transformer's audio cross-
    attention is out-of-distribution — the audio VAE never outputs zero
    latents and the transformer never saw "zero audio" at training. Encoding
    a literal silent mel (zero-valued input to the audio encoder) keeps the
    latent inside the audio VAE's output manifold. We then hold this latent
    fixed across all video diffusion steps, so invert and recon (and regen)
    see exactly the same audio context — the audio axis becomes a constant
    rather than a free variable, isolating the video flow.

    Shape derivation mirrors `LTX2ConditionPipeline.__call__`:
        audio_latents_per_second = sample_rate / hop_length / vae_temporal_compression
        audio_num_frames         = round(duration_s * audio_latents_per_second)
        T_mel (pre-VAE)          = audio_num_frames * vae_temporal_compression
    Audio encoder input shape:  (1, in_channels, T_mel, mel_bins)
    Audio encoder output shape: (1, latent_channels, audio_num_frames, latent_mel_bins)
    Returned packed shape:      (1, audio_num_frames, latent_channels * latent_mel_bins)
    """
    duration_s = num_frames / float(frame_rate)
    audio_latents_per_second = (
        pipe.audio_sampling_rate / pipe.audio_hop_length
        / float(pipe.audio_vae_temporal_compression_ratio)
    )
    audio_num_frames = round(duration_s * audio_latents_per_second)
    T_mel = audio_num_frames * pipe.audio_vae_temporal_compression_ratio
    num_mel_bins = pipe.audio_vae.config.mel_bins
    in_channels  = pipe.audio_vae.config.in_channels

    silent_mel = torch.zeros(
        (1, in_channels, T_mel, num_mel_bins),
        dtype=pipe.audio_vae.dtype, device=device,
    )
    posterior = pipe.audio_vae.encode(silent_mel).latent_dist
    audio_latents_4d = posterior.mode()  # deterministic (no sampling)

    # Use the pipeline's own ingestion path: pack + normalize, no noise.
    packed = pipe.prepare_audio_latents(
        latents=audio_latents_4d,
        noise_scale=0.0,
        device=device,
        dtype=dtype,
    )
    return packed


def end_clip_index(num_frames: int, num_clip_frames: int) -> int:
    n_lat = (num_frames      - 1) // LTX_TEMPORAL_SCALE + 1
    k_lat = (num_clip_frames - 1) // LTX_TEMPORAL_SCALE + 1
    return n_lat - k_lat


# ── Per-step diagnostics CSV ──────────────────────────────────────────────────

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
      - A FIXED audio context (encoded silent mel) used at every step.

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

        # Fixed audio context — packed, normalized, encoded silent mel.
        self.audio_context: torch.Tensor | None = None
        self.audio_num_frames: int = 1
        self.latent_num_frames: int = 0
        self.latent_height:     int = 0
        self.latent_width:      int = 0
        self.frame_rate:        float = 24.0

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
        audio_context: torch.Tensor,
        max_sequence_length: int = 256,
    ) -> None:
        """Encode prompts (positive + optional negative), stash conditioning state,
        and register the fixed audio context.

        Prompt path mirrors `LTX2ConditionPipeline.__call__` lines 1190-1214
        exactly: encode_prompt → (optional cat([neg, pos])) → connectors.
        Storing both the positive-only ([1, …]) and combined ([2, …]) views
        means CFG=1 phases (invert, reconstruct) and CFG>1 phases (regenerate)
        share the same connector output, just sliced differently.
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

        # Register the fixed audio context (encoded silent mel — packed,
        # normalized). Used unchanged for every transformer call across
        # invert / reconstruct / regenerate.
        self.audio_context = audio_context.to(
            device=device, dtype=self.transformer.dtype
        )
        self.audio_num_frames = self.audio_context.shape[1]

    # ── audio fetch ───────────────────────────────────────────────────────────

    def _audio_for(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (audio_hidden_states, audio_timestep) for one transformer call.

        The audio context is held fixed across all phases and steps (encoded
        silent mel). audio_timestep is always 0 (the "clean audio" σ).
        `batch_size`: 1 for CFG=1 calls, 2 for CFG>1 batched (cond+uncond duplicate).
        """
        t_dtype = self.transformer.dtype
        ah = self.audio_context
        at = torch.full((1,), 0.0, device=self.device, dtype=t_dtype)
        if batch_size == 2:
            ah = torch.cat([ah, ah], dim=0)
            at = torch.cat([at, at], dim=0)
        return ah.to(self.device, dtype=t_dtype), at

    # ── transformer call ──────────────────────────────────────────────────────

    def _call_transformer(
        self,
        z_packed: torch.Tensor,
        sigma_scalar: float,
        *,
        guidance_scale: float = 1.0,
        guidance_per_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One transformer call (or two batched if CFG>1). Returns velocity in fp32.

        For guidance_scale==1 (and no per-token spec): single forward,
        positive prompt only.
        For guidance_scale > 1: batched cond+uncond, mixed per the LTX-2
        pipeline's recipe (line 1363-1372 of pipeline_ltx2_condition.py):

            v_cfg = v_uncond + s · (v_cond − v_uncond)

        `guidance_per_token` (exp_044): a [1, N, 1] float tensor of per-token
        guidance scales. When supplied, ALWAYS does the batched cond+uncond
        forward and mixes `v = v_uncond + s_tok ⊙ (v_cond − v_uncond)` with
        s_tok broadcast over channels. This lets the free-middle tokens run at
        a different (e.g. lower) CFG than the conditioned anchor tokens, so the
        z1-encoded source smoke in the free middle isn't washed out by the
        production guidance while the anchors stay sharply guided.
        """
        t_dtype = self.transformer.dtype
        do_cfg = (guidance_per_token is not None) or (guidance_scale > 1.0)
        bsz_in = z_packed.shape[0]

        if do_cfg:
            if self.prompt_embeds_cfg is None:
                raise ValueError("CFG>1 requested but no negative_prompt was provided to prepare_sample.")
            z_in = torch.cat([z_packed, z_packed], dim=0).to(t_dtype)
            prompt_embeds     = self.prompt_embeds_cfg          # [2, seq, d] (neg ; pos)
            audio_prompt_emb  = self.audio_prompt_embeds_cfg
            prompt_attn_mask  = self.prompt_attn_mask_cfg
            audio_prompt_mask = self.audio_prompt_attn_mask_cfg
            ah, at = self._audio_for(batch_size=2)
        else:
            z_in = z_packed.to(t_dtype)
            prompt_embeds     = self.prompt_embeds
            audio_prompt_emb  = self.audio_prompt_embeds
            prompt_attn_mask  = self.prompt_attn_mask
            audio_prompt_mask = self.audio_prompt_attn_mask
            ah, at = self._audio_for(batch_size=1)

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
            if guidance_per_token is not None:
                s_tok = guidance_per_token.to(v_uncond.dtype)   # [1, N, 1] broadcast over channels
                v = v_uncond + s_tok * (v_cond - v_uncond)
            else:
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
        step_idx: int,
        diag: StepDiagnostics | None,
        phase_label: str,
    ) -> torch.Tensor:
        dtau = sigma_next - sigma_curr
        sigma_mid = sigma_curr + dtau / 2.0
        mask = self.conditioning_mask
        t0 = time.perf_counter()

        v_raw = self._call_transformer(z, sigma_curr, guidance_scale=1.0)
        v, x0_pred = self._x0_clamp_velocity(z, v_raw, sigma_curr)

        z_mid = z + (dtau / 2.0) * v

        v_mid_raw = self._call_transformer(z_mid, sigma_mid, guidance_scale=1.0)
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
        step_idx: int,
        guidance_scale: float,
        guidance_per_token: torch.Tensor | None = None,
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
            z, sigma_curr, guidance_scale=guidance_scale,
            guidance_per_token=guidance_per_token,
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
                step_idx=i, diag=diag, phase_label="invert",
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
                step_idx=i, diag=diag, phase_label="reconstruct",
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
        guidance_per_token: torch.Tensor | None = None,
        diag: StepDiagnostics | None = None,
    ) -> torch.Tensor:
        """z1 (noise) → z0_regen (clean) via Euler + CFG=guidance_scale.

        Mirrors `LTX2ConditionPipeline.__call__` exactly (line 1339-1398).
        Tests GENERATION-TRAJECTORY RECOVERY: does the inverted z1 round-trip
        back to z0 under the same flow that produced z0?

        `guidance_per_token` (exp_044): optional [1, N, 1] per-token CFG scales.
        When given, overrides `guidance_scale` and applies localized guidance
        (e.g. CFG=1 on free-middle tokens, CFG=gen on anchor tokens).
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
                step_idx=i, guidance_scale=guidance_scale,
                guidance_per_token=guidance_per_token,
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


def free_mid_pixel_range(free_latent_frames: list[int]) -> tuple[int, int]:
    """Pixel-frame [lo:hi) covering the given free latent frames.

    Causal VAE: latent frame f decodes to pixel frames [(f-1)*8+1 : f*8].
    For the contiguous free zone (e.g. [4..12]) this returns [(4-1)*8+1 :
    12*8+1) = [25:97).
    """
    a, b = min(free_latent_frames), max(free_latent_frames)
    return (a - 1) * 8 + 1, b * 8 + 1


def build_guidance_per_token(mask: torch.Tensor, anchor_cfg: float, free_cfg: float) -> torch.Tensor:
    """[1, N, 1] per-token guidance: anchor_cfg where mask>0, free_cfg where mask==0.

    Lets the unclamped free-middle tokens regen at a different (lower) CFG than
    the conditioned anchor tokens, so z1's encoded source smoke in the free
    middle isn't washed out by the production text guidance.
    """
    anchor = torch.as_tensor(float(anchor_cfg), device=mask.device, dtype=mask.dtype)
    free   = torch.as_tensor(float(free_cfg),   device=mask.device, dtype=mask.dtype)
    return torch.where(mask > 0, anchor, free)


# ── Main ─────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def _psig(video_uint8, lo, hi):
    """Perceptual smoke signals on the free-middle pixel frames [lo:hi).
    Returns means of: luminance (darkening), texture (Laplacian energy),
    saturation, temporal-diff (billowing dynamics). All on [0,1] scale."""
    v = video_uint8[lo:hi].astype(np.float32) / 255.0          # [F,H,W,3]
    L = 0.299 * v[..., 0] + 0.587 * v[..., 1] + 0.114 * v[..., 2]
    lap = np.abs(4 * L[:, 1:-1, 1:-1] - L[:, :-2, 1:-1] - L[:, 2:, 1:-1]
                 - L[:, 1:-1, :-2] - L[:, 1:-1, 2:])
    mx = v.max(-1); mn = v.min(-1); sat = (mx - mn) / (mx + 1e-6)
    tdiff = np.abs(L[1:] - L[:-1]).mean() if len(L) > 1 else 0.0
    return {"lum": round(float(L.mean()), 4), "tex": round(float(lap.mean()), 4),
            "sat": round(float(sat.mean()), 4), "tdiff": round(float(tdiff), 4)}


def _load_z0_and_dims(z0_src_run, sample_id, meta, spatial_ratio):
    """Load cached z0.pt + (latent_num_frames, lh, lw) from inv_meta render_HxW.

    NB: LTX-2's effective packed-latent spatial compression is `spatial_ratio`
    (= pipe.vae_spatial_compression_ratio, =32 for the 19B), NOT 8. Temporal is
    8 (16 latent frames ↔ 121 pixel frames) but only matters for pixel mapping.
    """
    z0 = torch.load(z0_src_run / sample_id / "z0.pt", map_location="cpu", weights_only=False).float()
    H, W = meta["render_HxW"]
    lh, lw = H // spatial_ratio, W // spatial_ratio
    N = z0.shape[1]
    lnf = N // (lh * lw)
    assert lnf * lh * lw == N, f"{sample_id}: z0 N={N} != lnf*{lh}*{lw} (ratio {spatial_ratio})"
    return z0, lnf, lh, lw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = REPO_ROOT / cfg["outputs"]["dir"]
    run_id, run_dir = next_run_dir(out_dir)

    with TeeLogger(run_dir / "run.log"):
        logging.basicConfig(level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
            datefmt="%H:%M:%S", stream=sys.stdout, force=True)
        print(f"[info] run_dir : {run_dir}")

        model_id   = cfg["model"]["model_id"]
        frame_rate = float(cfg["inference"]["frame_rate"])
        z0_src_run = REPO_ROOT / cfg["z0_source"]["run_dir"]
        free_frames = list(cfg["free_frames"])               # e.g. [4..12]
        priors      = list(cfg["priors"])                     # list of names
        seed = int(cfg["runtime"]["seed"])
        ps = 1  # LTX-2 19B patch sizes

        log.info("Loading LTX2ConditionPipeline from %s …", model_id)
        t0 = time.perf_counter()
        pipe = LTX2ConditionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload(device=DEVICE)
        pipe.vae.enable_tiling()
        log.info("Pipeline loaded in %.1fs.", time.perf_counter() - t0)
        metric_suite = MetricSuite(device=DEVICE)

        import yaml as _yaml
        mean = pipe.vae.latents_mean; std = pipe.vae.latents_std
        scaling = pipe.vae.config.scaling_factor

        # ── Precompute all clips' normalized-5d middles + spatial-pooled middles ──
        clip_meta = {}
        mids5d = {}          # sample_id -> normalized 5d middle [1,128,Fmid,lh,lw]
        mids_pool = {}       # sample_id -> [1,128,Fmid]  (spatial mean)
        endp5d = {}          # sample_id -> (frame3 [1,128,1,lh,lw], frame13 ...)
        dims = {}
        # Donor pool = all clips we might extract smoke from (targets + donors).
        target_ids = [s["sample_id"] for s in cfg["samples"]]
        pool_ids = list(cfg.get("donor_pool", target_ids))
        for t in target_ids:
            if t not in pool_ids: pool_ids.append(t)
        for sid in pool_ids:
            meta = _yaml.safe_load(open(z0_src_run / sid / "inv_meta.yaml"))
            clip_meta[sid] = meta
            z0, lnf, lh, lw = _load_z0_and_dims(z0_src_run, sid, meta, pipe.vae_spatial_compression_ratio)
            dims[sid] = (lnf, lh, lw)
            zn = pipe._unpack_latents(z0.to(DEVICE), lnf, lh, lw, ps, ps).float().cpu()  # [1,128,16,lh,lw]
            mid = zn[:, :, free_frames, :, :].clone()                 # [1,128,Fmid,lh,lw]
            mids5d[sid] = mid
            mids_pool[sid] = mid.mean(dim=(3, 4))                      # [1,128,Fmid]
            endp5d[sid] = (zn[:, :, 3:4, :, :].clone(), zn[:, :, 13:14, :, :].clone())  # frame 3, 13
            del z0, zn
        log.info("precomputed middles for %d clips", len(mids5d))

        rng = torch.Generator().manual_seed(seed)
        summary = []

        for idx, s in enumerate(cfg["samples"]):
            sid = s["sample_id"]
            sample_dir = run_dir / sid; sample_dir.mkdir(parents=True, exist_ok=True)
            lnf, lh, lw = dims[sid]
            meta = clip_meta[sid]
            log.info("─── %d/%d  %s  (latent %dx%d) ───", idx + 1, len(cfg["samples"]), sid, lh, lw)

            z0 = torch.load(z0_src_run / sid / "z0.pt", map_location="cpu", weights_only=False).float().to(DEVICE)
            zn = pipe._unpack_latents(z0, lnf, lh, lw, ps, ps).float()   # [1,128,16,lh,lw] normalized

            # reference video = decode(z0)
            zd_ref = pipe._denormalize_latents(zn.clone(), mean, std, scaling)
            ref_video = decode_latents_to_video(pipe, zd_ref)
            save_video(sample_dir / "source_video.mp4", ref_video, fps=int(frame_rate))

            a, b = min(free_frames), max(free_frames)
            px_lo, px_hi = (a - 1) * 8 + 1, b * 8 + 1
            log.info("  free frames %s → pixels [%d:%d)", free_frames, px_lo, px_hi)

            # perceptual reference signatures: REAL (source) and BASELINE (exp_033 prompt-only regen)
            real_sig = _psig(ref_video, px_lo, px_hi)
            log.info("  [REAL source            ] lum=%.3f tex=%.3f sat=%.3f tdiff=%.4f",
                     real_sig["lum"], real_sig["tex"], real_sig["sat"], real_sig["tdiff"])
            base_sig = None
            base_mp4 = z0_src_run / sid / "regen_video.mp4"
            if base_mp4.exists():
                bv, _, _ = tio.read_video(str(base_mp4), pts_unit="sec", output_format="THWC")
                bv = bv.numpy().astype(np.uint8)
                base_sig = _psig(bv, px_lo, px_hi)
                base_lpips = metric_suite.lpips(ref_video[px_lo:px_hi], bv[px_lo:px_hi])["mean"]
                log.info("  [BASELINE prompt-regen  ] PSNR=  --   lpips=%.3f | lum=%.3f tex=%.3f sat=%.3f tdiff=%.4f",
                         base_lpips, base_sig["lum"], base_sig["tex"], base_sig["sat"], base_sig["tdiff"])
                del bv

            mid_self = zn[:, :, free_frames, :, :].clone()   # [1,128,Fmid,lh,lw]
            Fmid = mid_self.shape[2]

            # ── prior builders ────────────────────────────────────────────────
            def donors(exclude_self, same_grid):
                out = []
                for osid in dims.keys():
                    if exclude_self and osid == sid: continue
                    if same_grid and dims[osid][1:] != (lh, lw): continue
                    out.append(osid)
                return out

            def build_prior(name):
                if name == "src":
                    return mid_self.clone()
                if name == "gauss":
                    g = torch.randn(mid_self.shape, generator=rng).to(DEVICE)
                    # rms-match to z0 middle
                    g = g * (mid_self.float().std() / g.std())
                    return g + mid_self.mean()
                if name == "endpoint_hold":
                    # hold last start-anchor frame (frame 3) across the middle
                    f3 = zn[:, :, 3:4, :, :]
                    return f3.repeat(1, 1, Fmid, 1, 1).clone()
                if name == "endpoint_interp":
                    f_start = zn[:, :, 3, :, :]; f_end = zn[:, :, 13, :, :]
                    ts = torch.linspace(0, 1, Fmid, device=zn.device).view(1, 1, Fmid, 1, 1)
                    return (f_start.unsqueeze(2) * (1 - ts) + f_end.unsqueeze(2) * ts).clone()
                if name in ("smoke_bcast_loo", "smoke_bcast_all"):
                    excl = (name == "smoke_bcast_loo")
                    ds = donors(exclude_self=excl, same_grid=False)
                    pool = torch.stack([mids_pool[d] for d in ds]).mean(0)   # [1,128,Fmid]
                    return pool.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, lh, lw).to(DEVICE)
                if name in ("smoke_spatial_loo", "smoke_spatial_all"):
                    excl = (name == "smoke_spatial_loo")
                    ds = donors(exclude_self=excl, same_grid=True)
                    if not ds: return None
                    return torch.stack([mids5d[d] for d in ds]).mean(0).to(DEVICE)  # [1,128,Fmid,lh,lw]
                if name == "smoke_bcast_loo_keepspatial":
                    # deployable-ish: shared channel-state (LOO bcast) carrying the
                    # recipient's z1-derived spatial structure is NOT available here;
                    # approximate spatial structure with endpoint_interp, shift channel-mean
                    # toward shared smoke state.
                    ds = donors(exclude_self=True, same_grid=False)
                    shared = torch.stack([mids_pool[d] for d in ds]).mean(0).to(DEVICE)  # [1,128,Fmid]
                    f_start = zn[:, :, 3, :, :]; f_end = zn[:, :, 13, :, :]
                    tsv = torch.linspace(0, 1, Fmid, device=zn.device).view(1, 1, Fmid, 1, 1)
                    base = f_start.unsqueeze(2) * (1 - tsv) + f_end.unsqueeze(2) * tsv  # [1,128,Fmid,lh,lw]
                    base_mean = base.mean(dim=(3, 4), keepdim=True)                     # [1,128,Fmid,1,1]
                    return (base - base_mean + shared.unsqueeze(-1).unsqueeze(-1)).clone()
                if name.startswith("donor:"):
                    # PERCEPTUAL injection: pin a SINGLE donor's REAL free-middle latent
                    # (real smoke texture + billowing dynamics) into the target. Requires
                    # same grid. This is "extract donor sample's smoke, inject at gen".
                    dsid = name.split(":", 1)[1]
                    if dsid == sid or dims[dsid][1:] != (lh, lw):
                        return None
                    return mids5d[dsid].to(DEVICE).clone()                 # [1,128,Fmid,lh,lw]
                if name.startswith("donorblend:"):
                    # soft blend toward a donor's middle: alpha*donor + (1-alpha)*endpoint_interp
                    _, dsid, a = name.split(":")
                    alpha = float(a)
                    if dsid == sid or dims[dsid][1:] != (lh, lw):
                        return None
                    f_start = zn[:, :, 3, :, :]; f_end = zn[:, :, 13, :, :]
                    tsv = torch.linspace(0, 1, Fmid, device=zn.device).view(1, 1, Fmid, 1, 1)
                    base = f_start.unsqueeze(2) * (1 - tsv) + f_end.unsqueeze(2) * tsv
                    don = mids5d[dsid].to(DEVICE)
                    return (alpha * don + (1 - alpha) * base).clone()
                if name.startswith("smokedelta:"):
                    # DISENTANGLED smoke transfer: extract the donor's smoke as a
                    # DELTA from its OWN scene-baseline (endpoint interp), add onto
                    # the TARGET's scene-baseline. Removes donor scene colors
                    # (fixes saturation) and builds on the target's endpoints
                    # (fixes onset/offset continuity). Optional trailing :alpha
                    # scales the delta. Same grid required.
                    parts = name.split(":")
                    dsid = parts[1]; dscale = float(parts[2]) if len(parts) > 2 else 1.0
                    if dsid == sid or dims[dsid][1:] != (lh, lw):
                        return None
                    def _interp(f3, f13):
                        tsv = torch.linspace(0, 1, Fmid, device=DEVICE).view(1, 1, Fmid, 1, 1)
                        return f3.to(DEVICE) * (1 - tsv) + f13.to(DEVICE) * tsv
                    d3, d13 = endp5d[dsid]
                    donor_interp = _interp(d3, d13)                 # donor scene morph (no smoke)
                    smoke_delta = mids5d[dsid].to(DEVICE) - donor_interp
                    t3, t13 = endp5d[sid]
                    target_interp = _interp(t3, t13)                # target scene morph
                    return (target_interp + dscale * smoke_delta).clone()
                if name.startswith("tempblend:") or name.startswith("tempdelta:"):
                    # TEMPORALLY-WINDOWED injection: target scene at smoke onset/
                    # offset (continuity), donor's full real smoke at the occlusion
                    # PEAK (where the scene is occluded so donor identity is hidden).
                    # w(i) is a Gaussian bump over the Fmid free frames peaked at the
                    # darkest latent frame (~8 → Fmid index 4). Optional trailing
                    # :sigma controls window width.
                    parts = name.split(":")
                    kind = parts[0]; dsid = parts[1]
                    sigma = float(parts[2]) if len(parts) > 2 else 2.0
                    if dsid == sid or dims[dsid][1:] != (lh, lw):
                        return None
                    peak = 4.0  # Fmid index for latent frame 8 (free frames start at 4)
                    ii = torch.arange(Fmid, device=DEVICE, dtype=torch.float32)
                    w = torch.exp(-0.5 * ((ii - peak) / sigma) ** 2)
                    w = (w / w.max()).view(1, 1, Fmid, 1, 1)        # peak=1, edges→0
                    def _interp2(f3, f13):
                        tsv = torch.linspace(0, 1, Fmid, device=DEVICE).view(1, 1, Fmid, 1, 1)
                        return f3.to(DEVICE) * (1 - tsv) + f13.to(DEVICE) * tsv
                    t3, t13 = endp5d[sid]
                    target_interp = _interp2(t3, t13)
                    if kind == "tempblend":
                        donor_mid = mids5d[dsid].to(DEVICE)
                        return ((1 - w) * target_interp + w * donor_mid).clone()
                    else:  # tempdelta: add windowed scene-free smoke delta
                        d3, d13 = endp5d[dsid]
                        smoke_delta = mids5d[dsid].to(DEVICE) - _interp2(d3, d13)
                        return (target_interp + w * smoke_delta).clone()
                raise ValueError(name)

            results = []
            for name in priors:
                prior = build_prior(name)
                if prior is None:
                    log.info("  prior '%s' skipped (no same-grid donors)", name); continue
                zn_mod = zn.clone()
                zn_mod[:, :, free_frames, :, :] = prior.to(zn.dtype)
                zd = pipe._denormalize_latents(zn_mod, mean, std, scaling)
                vid = decode_latents_to_video(pipe, zd)
                save_video(sample_dir / f"prior_{name}.mp4", vid, fps=int(frame_rate))
                fm = metric_suite.psnr(ref_video[px_lo:px_hi], vid[px_lo:px_hi])["mean"]
                fm_lpips = metric_suite.lpips(ref_video[px_lo:px_hi], vid[px_lo:px_hi])["mean"]
                psig = _psig(vid, px_lo, px_hi)
                log.info("  [%-22s] PSNR=%6.2f lpips=%.3f | lum=%.3f tex=%.3f sat=%.3f tdiff=%.4f",
                         name, fm, fm_lpips, psig["lum"], psig["tex"], psig["sat"], psig["tdiff"])
                results.append({"name": name, "free_mid_psnr": round(float(fm), 2),
                                "free_mid_lpips": round(float(fm_lpips), 3), "psig": psig})
                del zn_mod, zd, vid, prior
                torch.cuda.empty_cache()

            summary.append({"sample_id": sid, "latent_HxW": [lh, lw],
                            "free_mid_pixel_range": [px_lo, px_hi],
                            "real_sig": real_sig, "baseline_sig": base_sig, "priors": results})
            del z0, zn; torch.cuda.empty_cache()

        with (run_dir / "config_snapshot.yaml").open("w") as f:
            _yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        with (run_dir / "summary.yaml").open("w") as f:
            _yaml.safe_dump({"run_id": run_id, "samples": summary}, f, sort_keys=False, allow_unicode=True)

        # table
        import statistics as st
        log.info("")
        log.info("exp_045 middle-prior DECODE feasibility — FREE-MID PSNR (ceiling for pin/inject)")
        hdr = "clip".ljust(18) + "".join(p[:14].rjust(15) for p in priors)
        log.info(hdr); log.info("-" * len(hdr))
        cols = {p: [] for p in priors}
        for s in summary:
            by = {r["name"]: r for r in s["priors"]}
            row = s["sample_id"].ljust(18)
            for p in priors:
                if p in by:
                    v = by[p]["free_mid_psnr"]; cols[p].append(v); row += f"{v:15.2f}"
                else:
                    row += "          skip "
            log.info(row)
        log.info("-" * len(hdr))
        medrow = "MEDIAN".ljust(18) + "".join((f"{st.median(cols[p]):15.2f}" if cols[p] else "          n/a  ") for p in priors)
        log.info(medrow)
        log.info("[done] %s → %s", run_id, run_dir)


if __name__ == "__main__":
    main()
