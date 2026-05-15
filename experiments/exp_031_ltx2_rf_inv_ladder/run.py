"""exp_031 — LTX-2 RF-Solver inversion: the R0→R5 localization ladder.

Iteration It-0 of the RF-inversion research loop (notes/rf_inversion_loop.md).
Localizes WHERE degradation enters between exp_029 (works on generated latents,
inv_recon_rel ~0.01) and exp_030 (collapsed on real clips, ~0.68) by moving
exactly ONE variable per rung.

Per rung (R0..R5) the harness builds z0 and the conditioning differently — see
config.yaml `ladder:` — then runs the SAME invert + reconstruct (40 midpoint
steps each, CFG=1). No regeneration, no gate/escalation: recon is the clean
solver self-consistency signal and the only thing this iteration measures.

z0_source:
  gen_latent     — load exp_029's z0.pt directly (transformer-output latent)
  decode_encode  — decode(z0.pt) → pixels → VAE-encode (encoder-output latent,
                   same content) — the "decode-as-fake-real-clip" trick
  encode_clip    — VAE-encode a real .mp4 (edge-padded to num_frames)

conditioning:
  external — two separate clips (exp_029-style C2V)
  none     — vanilla inversion (all-zero mask)
  self     — clean_latents = z0's own endpoint positions (the cond invariant
             holds exactly, by construction)

Primary metric: FREE-positions-only latent rel/cos. Conditioned positions are
hard-pinned every solver step in both directions → their round-trip error is
trivially ~0; including them (exp_030's all-positions metric) only adds a
constant offset. all/cond latent + perceptual PSNR/SSIM/LPIPS also logged.

Saved per (rung, sample) under run_dir/<rung>/<sample_id>/:
  z0.pt, z1.pt, z0_recon.pt, z_t_25/50/75.pt   — packed bfloat16
  source_video.mp4   — decode(z0)
  recon_video.mp4    — decode(z0_recon)
  step_diag_invert_n40.csv, step_diag_reconstruct_n40.csv
  meta.yaml          — full metrics for this rung/sample
"""
from __future__ import annotations

import argparse
import copy
import csv
import logging
import pathlib
import statistics
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

from diffusion.exp_utils import load_config, next_run_dir, TeeLogger

REPO_ROOT      = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = pathlib.Path(__file__).parent / "config.yaml"

LTX_TEMPORAL_SCALE  = 8
DEVICE              = "cuda:0"
NUM_TRAIN_TIMESTEPS = 1000  # FlowMatchEulerDiscreteScheduler default; t = sigma * 1000

log = logging.getLogger(__name__)


# ── Frame / clip helpers ─────────────────────────────────────────────────────

def load_frames_from_mp4(path: pathlib.Path, n: int | None = None,
                         from_end: bool = False) -> list[Image.Image]:
    """Load PIL frames from an mp4. n=None → all frames; else first/last n."""
    video, _, _ = tio.read_video(str(path), pts_unit="sec", output_format="THWC")
    if video.shape[0] == 0:
        raise ValueError(f"No frames decoded from {path}")
    if n is not None:
        video = video[-n:] if from_end else video[:n]
    return [Image.fromarray(f.numpy()) for f in video]


def pad_or_trim_frames(frames: list[Image.Image], n: int) -> list[Image.Image]:
    """Make `frames` exactly length n. Longer → take first n. Shorter → repeat
    the last frame (static tail; no motion discontinuity, unlike loop-tiling).
    """
    if len(frames) >= n:
        return frames[:n]
    return frames + [frames[-1]] * (n - len(frames))


def end_clip_index(num_frames: int, num_clip_frames: int) -> int:
    n_lat = (num_frames      - 1) // LTX_TEMPORAL_SCALE + 1
    k_lat = (num_clip_frames - 1) // LTX_TEMPORAL_SCALE + 1
    return n_lat - k_lat


def build_zeros_audio_context(
    pipe: LTX2ConditionPipeline, num_frames: int, frame_rate: float,
    device: str, dtype: torch.dtype,
) -> torch.Tensor:
    """Return a true-zeros audio context of the exact packed shape the
    transformer expects (exp_029's "zeros" strategy).

    The packed shape (1, audio_num_frames, audio_channels*latent_mel_bins) is
    derived by running the real ingestion path once on a silent mel, then
    zeroing the result — so the shape is always correct without hardcoding.
    Held fixed across every rung; identical in invert and recon, so audio is a
    constant, never a variable in the round-trip.
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
    audio_latents_4d = pipe.audio_vae.encode(silent_mel).latent_dist.mode()
    packed = pipe.prepare_audio_latents(
        latents=audio_latents_4d, noise_scale=0.0, device=device, dtype=dtype,
    )
    return torch.zeros_like(packed)


# ── Per-step diagnostics CSV ──────────────────────────────────────────────────

class StepDiagnostics:
    """CSV writer for per-step solver diagnostics. One row per outer midpoint
    step. Norm split by mask: z_cond_norm should stay ≈ clean_latents norm;
    z_free_norm drives the solver error.
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
        self._writer.writerow({col: row.get(col, "") for col in self.COLUMNS})

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
    m = mask.squeeze(-1).bool()
    zf = z.float()
    z_cond = zf[m.unsqueeze(-1).expand_as(zf)]
    z_free = zf[(~m).unsqueeze(-1).expand_as(zf)]
    cn = float(z_cond.norm().item()) if z_cond.numel() else 0.0
    fn = float(z_free.norm().item()) if z_free.numel() else 0.0
    return cn, fn


# ── RF-Solver inverter (invert + reconstruct only; CFG=1) ─────────────────────

class RFInverter:
    """RF-Solver midpoint 2nd-order inversion + reconstruction for LTX-2 Stage 1.

    Both phases: midpoint integrator, CFG=1 (positive prompt only), per-token
    timestep `t·(1−mask)`, x0-domain clamp + hard re-clamp of conditioned
    positions, a FIXED audio context. No regeneration here — exp_031 measures
    solver self-consistency only.
    """

    def __init__(self, pipe: LTX2ConditionPipeline, device: str = DEVICE) -> None:
        self.pipe = pipe
        self.device = device
        self.transformer = pipe.transformer
        self.vae = pipe.vae

        self.conditioning_mask: torch.Tensor | None = None
        self.clean_latents:     torch.Tensor | None = None
        self.prompt_embeds:          torch.Tensor | None = None
        self.prompt_attn_mask:       torch.Tensor | None = None
        self.audio_prompt_embeds:    torch.Tensor | None = None
        self.audio_prompt_attn_mask: torch.Tensor | None = None

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
        conditioning_mask: torch.Tensor,
        clean_latents: torch.Tensor,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
        frame_rate: float,
        audio_context: torch.Tensor,
        max_sequence_length: int = 256,
    ) -> None:
        """Encode the positive prompt, stash conditioning state, register the
        fixed audio context. CFG=1 only — no negative prompt path.

        Prompt path mirrors `LTX2ConditionPipeline.__call__` (encode_prompt →
        connectors), positive branch only.
        """
        device = self.device
        (pe, pm, _ne, _nm) = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            device=device,
            max_sequence_length=max_sequence_length,
        )
        additive = (1 - pm.to(pe.dtype)) * -1_000_000.0
        cp, cap, cam = self.pipe.connectors(pe, additive, additive_mask=True)
        self.prompt_embeds          = cp
        self.audio_prompt_embeds    = cap
        self.prompt_attn_mask       = cam
        self.audio_prompt_attn_mask = cam

        self.conditioning_mask = conditioning_mask.to(device=device)
        self.clean_latents     = clean_latents.to(device=device)
        self.latent_num_frames = latent_num_frames
        self.latent_height     = latent_height
        self.latent_width      = latent_width
        self.frame_rate        = frame_rate

        self.audio_context = audio_context.to(device=device, dtype=self.transformer.dtype)
        self.audio_num_frames = self.audio_context.shape[1]

    # ── transformer call (CFG=1) ──────────────────────────────────────────────

    def _call_transformer(self, z_packed: torch.Tensor, sigma_scalar: float) -> torch.Tensor:
        """One transformer call, positive prompt only. Returns velocity in fp32."""
        t_dtype = self.transformer.dtype
        z_in = z_packed.to(t_dtype)

        # Per-token timestep: conditioned tokens see ~0 diffusion time.
        t_value = float(sigma_scalar) * NUM_TRAIN_TIMESTEPS
        t = torch.full((z_in.shape[0],), t_value, device=z_packed.device, dtype=t_dtype)
        cond_mask_t = self.conditioning_mask.squeeze(-1).to(t_dtype)
        video_timestep = t.unsqueeze(-1) * (1 - cond_mask_t)

        at = torch.zeros((1,), device=self.device, dtype=t_dtype)  # clean-audio σ
        noise_pred_video, _ = self.transformer(
            hidden_states=z_in,
            audio_hidden_states=self.audio_context,
            encoder_hidden_states=self.prompt_embeds,
            audio_encoder_hidden_states=self.audio_prompt_embeds,
            timestep=video_timestep,
            audio_timestep=at,
            encoder_attention_mask=self.prompt_attn_mask,
            audio_encoder_attention_mask=self.audio_prompt_attn_mask,
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
        return noise_pred_video.float()

    # ── x0 clamp ──────────────────────────────────────────────────────────────

    def _x0_clamp_velocity(
        self, z_packed: torch.Tensor, v_packed: torch.Tensor, sigma_scalar: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LTX-2 x0-domain clamp. Returns (v_clamped, x0_pred_pre_clamp).

        At σ < 1e-4 we short-circuit: the divide by σ otherwise squashes all
        velocity components to zero. The hard re-clamp in `_midpoint_step`
        keeps conditioned positions pinned.
        """
        sigma = float(sigma_scalar)
        x0_pred_pre = z_packed - v_packed * sigma
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

        v_raw = self._call_transformer(z, sigma_curr)
        v, x0_pred = self._x0_clamp_velocity(z, v_raw, sigma_curr)

        z_mid = z + (dtau / 2.0) * v

        v_mid_raw = self._call_transformer(z_mid, sigma_mid)
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

    # ── σ-grid builder ────────────────────────────────────────────────────────

    def _build_sigma_grid(self, num_steps: int, scheduler) -> np.ndarray:
        """The dynamic-shifted σ grid (length num_steps + 1, descending from
        σ_max≈1 to 0) — identical to what Stage-1 computes for this step count.
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

        target_to_step: dict[float, int] = {}
        for target in checkpoint_sigmas:
            sigmas_after = sigmas_inv[1:]
            target_to_step[target] = int(np.argmin(np.abs(sigmas_after - target)))

        checkpoints: dict[float, torch.Tensor] = {}
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
        Tests SOLVER SELF-CONSISTENCY at CFG=1.
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


@torch.inference_mode()
def encode_frames_to_z0(
    pipe: LTX2ConditionPipeline,
    frames: list[Image.Image],
    height: int,
    width: int,
    device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """PIL frames → preprocess → VAE-encode → normalize + pack → z0 (packed)."""
    src_tensor = pipe.video_processor.preprocess_video(frames, height=height, width=width)
    src_tensor = src_tensor.to(device=device, dtype=pipe.vae.dtype)
    video_latent_5d = retrieve_latents(
        pipe.vae.encode(src_tensor), generator=generator, sample_mode="argmax"
    )
    return normalize_and_pack(pipe, video_latent_5d).to(device)


# ── MetricSuite ──────────────────────────────────────────────────────────────

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


def _latent_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Relative L2 + cosine between two flattened latent selections."""
    a = a.float().flatten()
    b = b.float().flatten()
    n = int(a.numel())
    if n == 0:
        return {"l2": 0.0, "l2_per_element": 0.0, "relative": 0.0, "cosine": 1.0,
                "n_elements": 0, "norm_a": 0.0, "norm_b": 0.0}
    diff = a - b
    l2 = float(diff.norm().item())
    norm_a = float(a.norm().item())
    norm_b = float(b.norm().item())
    rel = l2 / max(norm_a, 1e-8)
    cos = float(torch.dot(a, b).item()) / max(norm_a * norm_b, 1e-8)
    return {
        "l2": l2, "l2_per_element": l2 / (n ** 0.5),
        "relative": rel, "cosine": cos,
        "n_elements": n, "norm_a": norm_a, "norm_b": norm_b,
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
    def latent_masked(z_a: torch.Tensor, z_b: torch.Tensor, mask: torch.Tensor) -> dict:
        """Latent rel/cos split into free / cond / all positions.

        free is the PRIMARY signal: conditioned positions are hard-pinned every
        solver step, so their round-trip error is trivially ~0 (or a constant
        offset) and only dilutes the all-positions number.
        mask is [B, N, 1] in {0,1}.
        """
        m = mask.squeeze(-1).bool()              # [B, N]
        a = z_a.float(); b = z_b.float()         # [B, N, C]
        free_sel = (~m).unsqueeze(-1).expand_as(a)
        cond_sel = m.unsqueeze(-1).expand_as(a)
        out = {
            "all":  _latent_stats(a.flatten(), b.flatten()),
            "free": _latent_stats(a[free_sel], b[free_sel]),
        }
        out["cond"] = _latent_stats(a[cond_sel], b[cond_sel]) if bool(m.any()) else None
        return out

    def evaluate(
        self,
        src_video: np.ndarray,
        recon_video: np.ndarray,
        z0_packed: torch.Tensor,
        z0_recon: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        return {
            "psnr":   self.psnr(src_video, recon_video),
            "ssim":   self.ssim(src_video, recon_video),
            "lpips":  self.lpips(src_video, recon_video),
            "latent": self.latent_masked(z0_packed, z0_recon, mask),
        }


def save_video(path: pathlib.Path, video_uint8: np.ndarray, fps: int) -> None:
    export_to_video([Image.fromarray(f) for f in video_uint8], str(path), fps=int(fps))


# ── z0 + conditioning builders ───────────────────────────────────────────────

def build_z0(
    pipe: LTX2ConditionPipeline,
    *,
    z0_source: str,
    z0_pt_path: pathlib.Path | None,
    clip_path: pathlib.Path | None,
    num_frames: int,
    height: int,
    width: int,
    latent_dims: tuple[int, int, int],
    device: str,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> tuple[torch.Tensor, list[Image.Image] | None]:
    """Build z0 (packed) per the rung's z0_source.

    Returns (z0_packed, source_frames). source_frames is the PIL frame list the
    z0 is derived from (decoded or loaded) — used to drive `self`-conditioning
    geometry — or None for gen_latent (no pixels involved).
    """
    lnf, lh, lw = latent_dims
    if z0_source == "gen_latent":
        z0 = torch.load(z0_pt_path, map_location="cpu").to(device=device, dtype=dtype)
        expected_n = lnf * lh * lw
        assert z0.dim() == 3 and z0.shape[0] == 1 and z0.shape[1] == expected_n, (
            f"{z0_pt_path}: shape {tuple(z0.shape)} != expected (1, {expected_n}, C)"
        )
        return z0, None

    if z0_source == "decode_encode":
        z0_src = torch.load(z0_pt_path, map_location="cpu").to(device=device, dtype=dtype)
        src_5d = unpack_and_denormalize(pipe, z0_src, lnf, lh, lw)
        pixels = decode_latents_to_video(pipe, src_5d)            # (F, H, W, 3) uint8
        frames = [Image.fromarray(f) for f in pixels]
        z0 = encode_frames_to_z0(pipe, frames, height, width, device, generator)
        return z0, frames

    if z0_source == "encode_clip":
        frames = pad_or_trim_frames(load_frames_from_mp4(clip_path), num_frames)
        z0 = encode_frames_to_z0(pipe, frames, height, width, device, generator)
        return z0, frames

    raise ValueError(f"unknown z0_source: {z0_source}")


def build_conditioning(
    pipe: LTX2ConditionPipeline,
    *,
    mode: str,
    z0_packed: torch.Tensor,
    geom_start_frames: list[Image.Image] | None,
    geom_end_frames: list[Image.Image] | None,
    height: int,
    width: int,
    num_frames: int,
    end_idx: int,
    start_strength: float,
    end_strength: float,
    latent_dims: tuple[int, int, int],
    device: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (conditioning_mask, clean_latents), both packed.

    external — clean_latents = VAE-encoded external clips (exp_029-style C2V).
    none     — all-zero mask, vanilla inversion.
    self     — same mask geometry as external, but clean_latents = z0's own
               endpoint positions (the cond invariant holds exactly). The
               geom_* frames only drive mask geometry; their content is
               discarded for `self`.
    """
    lnf, lh, lw = latent_dims
    zeros_5d = torch.zeros((1, 1, lnf, lh, lw), device=device, dtype=z0_packed.dtype)
    cmask_zeros = pipe._pack_latents(
        zeros_5d, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size
    )

    if mode == "none":
        return cmask_zeros, torch.zeros_like(z0_packed)

    if mode not in ("external", "self"):
        raise ValueError(f"unknown conditioning mode: {mode}")

    conditions = [
        LTX2VideoCondition(frames=geom_start_frames, index=0,       strength=start_strength),
        LTX2VideoCondition(frames=geom_end_frames,   index=end_idx, strength=end_strength),
    ]
    condition_frames, condition_strengths, condition_indices = pipe.preprocess_conditions(
        conditions, height, width, num_frames, device=device
    )
    cond_latents_list: list[torch.Tensor] = []
    for cond_tensor in condition_frames:
        cl = retrieve_latents(pipe.vae.encode(cond_tensor), generator=generator, sample_mode="argmax")
        cl = pipe._normalize_latents(cl, pipe.vae.latents_mean, pipe.vae.latents_std).to(
            device=device, dtype=z0_packed.dtype
        )
        cl = pipe._pack_latents(cl, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size)
        cond_latents_list.append(cl)

    throwaway = torch.zeros_like(z0_packed)
    _, cmask_packed, clean_latents_ext = pipe.apply_visual_conditioning(
        throwaway, cmask_zeros, cond_latents_list,
        condition_strengths, condition_indices,
        latent_height=lh, latent_width=lw,
    )
    if mode == "external":
        return cmask_packed, clean_latents_ext
    # self: pin to z0's own positions
    return cmask_packed, z0_packed.clone()


# ── Main ─────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rungs", type=str, default=None,
                        help="comma-separated rung names to run (default: all in config)")
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

        # ── config ────────────────────────────────────────────────────────────
        model_id        = cfg["model"]["model_id"]
        height          = int(cfg["inference"]["height"])
        width           = int(cfg["inference"]["width"])
        num_frames      = int(cfg["inference"]["num_frames"])
        frame_rate      = float(cfg["inference"]["frame_rate"])
        seed            = int(cfg["runtime"]["seed"])
        num_clip_frames = int(cfg["inputs"]["num_clip_frames"])
        start_strength  = float(cfg["inputs"]["start_clip_strength"])
        end_strength    = float(cfg["inputs"]["end_clip_strength"])

        inv_cfg       = cfg["inversion"]
        inv_steps     = int(inv_cfg["num_steps"])
        inv_cfg_scale = float(inv_cfg["guidance_scale"])
        inv_solver    = str(inv_cfg["solver"])
        ckpt_sigmas   = list(inv_cfg["cache_sigma_checkpoints"])
        r0_halt_max   = float(inv_cfg.get("r0_halt_free_rel_max", 0.05))

        ladder = cfg["ladder"]
        if args.rungs:
            wanted = {r.strip() for r in args.rungs.split(",")}
            ladder = [r for r in ladder if r["rung"] in wanted]

        davis_z0_run_dir = REPO_ROOT / cfg["davis"]["z0_run_dir"]
        davis_samples    = cfg["davis"]["samples"]
        ss_samples       = cfg["shadow_smoke"]["samples"]
        sample_sets = {"davis": davis_samples, "shadow_smoke": ss_samples}

        end_idx           = end_clip_index(num_frames, num_clip_frames)
        latent_num_frames = (num_frames - 1) // LTX_TEMPORAL_SCALE + 1

        print(f"[info] run_dir   : {run_dir}")
        print(f"[info] rungs     : {[r['rung'] for r in ladder]}")
        print(f"[info] resolution: {height}x{width}  num_frames={num_frames}  steps={inv_steps}")

        if inv_cfg_scale != 1.0:
            log.warning("inversion.guidance_scale=%.2f ignored — invert/recon always CFG=1.", inv_cfg_scale)

        # ── pipeline ──────────────────────────────────────────────────────────
        log.info("Loading LTX2ConditionPipeline from %s …", model_id)
        t0 = time.perf_counter()
        pipe = LTX2ConditionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload(device=DEVICE)
        pipe.vae.enable_tiling()
        pipe.disable_lora()
        log.info("Pipeline loaded in %.1fs.", time.perf_counter() - t0)
        stage1_scheduler = pipe.scheduler
        metric_suite = MetricSuite(device=DEVICE)

        mod_value = int(pipe.vae_spatial_compression_ratio * pipe.transformer_spatial_patch_size)
        assert height % mod_value == 0 and width % mod_value == 0, (
            f"resolution {height}x{width} not divisible by mod_value {mod_value}"
        )
        latent_height = height // pipe.vae_spatial_compression_ratio
        latent_width  = width  // pipe.vae_spatial_compression_ratio
        latent_dims   = (latent_num_frames, latent_height, latent_width)

        z0_dtype = torch.bfloat16
        summary: list[dict] = []

        # ── ladder ────────────────────────────────────────────────────────────
        for rung_spec in ladder:
            rung        = rung_spec["rung"]
            z0_source   = rung_spec["z0_source"]
            cond_mode   = rung_spec["conditioning"]
            sample_set  = rung_spec["sample_set"]
            clip_field   = rung_spec.get("clip_field")
            prompt_field = rung_spec.get("prompt_field")
            samples = sample_sets[sample_set]

            log.info("")
            log.info("══════════ RUNG %s  ║  z0=%s  cond=%s  set=%s ══════════",
                     rung, z0_source, cond_mode, sample_set)
            log.info("note: %s", rung_spec.get("note", ""))

            for s_idx, sample in enumerate(samples):
                sample_id = sample["sample_id"]
                sample_dir = run_dir / rung / sample_id
                sample_dir.mkdir(parents=True, exist_ok=True)

                # prompt + source paths per rung/sample
                if sample_set == "davis":
                    prompt = sample[prompt_field if prompt_field else "prompt"].strip()
                    z0_pt_path = davis_z0_run_dir / sample_id / "z0.pt"
                    clip_path  = (REPO_ROOT / sample[clip_field]) if clip_field else None
                else:
                    prompt = sample[prompt_field if prompt_field else "prompt"].strip()
                    z0_pt_path = None
                    clip_path  = REPO_ROOT / sample[clip_field if clip_field else "clip"]

                log.info("─── %s / %s  (%d/%d) ───", rung, sample_id, s_idx + 1, len(samples))
                generator = torch.Generator(device=DEVICE).manual_seed(seed)

                # ── build z0 ──────────────────────────────────────────────────
                t_z0 = time.perf_counter()
                z0_packed, source_frames = build_z0(
                    pipe,
                    z0_source=z0_source,
                    z0_pt_path=z0_pt_path,
                    clip_path=clip_path,
                    num_frames=num_frames,
                    height=height, width=width,
                    latent_dims=latent_dims,
                    device=DEVICE, dtype=z0_dtype, generator=generator,
                )
                log.info("z0 built (%s) in %.1fs  shape=%s norm=%.2f",
                         z0_source, time.perf_counter() - t_z0,
                         tuple(z0_packed.shape), z0_packed.float().norm().item())

                # ── conditioning geometry frames ──────────────────────────────
                # external: separate DAVIS start/end clips (exp_029-style).
                # self:     z0's own source frames drive mask geometry only.
                # none:     unused.
                if cond_mode == "external":
                    geom_start = load_frames_from_mp4(
                        REPO_ROOT / sample["start_clip"], num_clip_frames, from_end=False)
                    geom_end = load_frames_from_mp4(
                        REPO_ROOT / sample["end_clip"], num_clip_frames, from_end=True)
                elif cond_mode == "self":
                    if source_frames is None:
                        raise RuntimeError(f"{rung}: 'self' conditioning needs source frames "
                                           f"but z0_source={z0_source} produced none")
                    geom_start = source_frames[:num_clip_frames]
                    geom_end   = source_frames[-num_clip_frames:]
                else:  # none
                    geom_start = geom_end = None

                cmask_packed, clean_latents = build_conditioning(
                    pipe,
                    mode=cond_mode,
                    z0_packed=z0_packed,
                    geom_start_frames=geom_start,
                    geom_end_frames=geom_end,
                    height=height, width=width, num_frames=num_frames,
                    end_idx=end_idx,
                    start_strength=start_strength, end_strength=end_strength,
                    latent_dims=latent_dims,
                    device=DEVICE, generator=generator,
                )
                n_cond = int((cmask_packed > 0).sum().item())
                log.info("conditioning=%s  active tokens=%d / %d",
                         cond_mode, n_cond, cmask_packed.shape[1])

                # ── audio context (zeros, fixed) ──────────────────────────────
                audio_context = build_zeros_audio_context(
                    pipe, num_frames, frame_rate, device=DEVICE, dtype=pipe.transformer.dtype
                )

                # ── prepare inverter ──────────────────────────────────────────
                inverter = RFInverter(pipe, device=DEVICE)
                inverter.prepare_sample(
                    prompt=prompt,
                    conditioning_mask=cmask_packed,
                    clean_latents=clean_latents,
                    latent_num_frames=latent_num_frames,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    frame_rate=frame_rate,
                    audio_context=audio_context,
                )

                # ── invert + reconstruct ──────────────────────────────────────
                diag_inv = StepDiagnostics(sample_dir / f"step_diag_invert_n{inv_steps}.csv")
                diag_rec = StepDiagnostics(sample_dir / f"step_diag_reconstruct_n{inv_steps}.csv")
                try:
                    t_inv = time.perf_counter()
                    z1, checkpoints, sigmas_inv = inverter.invert(
                        z0_packed, inv_steps, stage1_scheduler, ckpt_sigmas, diag=diag_inv)
                    t_inv_s = time.perf_counter() - t_inv
                    log.info("invert done in %.1fs  z1 norm=%.2f", t_inv_s, z1.float().norm().item())

                    t_rec = time.perf_counter()
                    z0_recon = inverter.reconstruct(z1, inv_steps, stage1_scheduler, diag=diag_rec)
                    t_rec_s = time.perf_counter() - t_rec
                    log.info("recon  done in %.1fs  z0_recon norm=%.2f", t_rec_s, z0_recon.float().norm().item())
                finally:
                    diag_inv.close()
                    diag_rec.close()

                # ── metrics ───────────────────────────────────────────────────
                src_5d   = unpack_and_denormalize(pipe, z0_packed, *latent_dims)
                recon_5d = unpack_and_denormalize(pipe, z0_recon,  *latent_dims)
                src_video   = decode_latents_to_video(pipe, src_5d)
                recon_video = decode_latents_to_video(pipe, recon_5d)
                m = metric_suite.evaluate(src_video, recon_video, z0_packed, z0_recon, cmask_packed)

                lat = m["latent"]
                free, allp, cond = lat["free"], lat["all"], lat["cond"]
                log.info(
                    "[%s/%s]  free rel=%.4f cos=%.5f  |  all rel=%.4f  |  PSNR=%.2f SSIM=%.4f LPIPS=%.4f",
                    rung, sample_id, free["relative"], free["cosine"], allp["relative"],
                    m["psnr"]["mean"], m["ssim"]["mean"], m["lpips"]["mean"],
                )

                # ── persist ───────────────────────────────────────────────────
                torch.save(z0_packed.detach().cpu().to(torch.bfloat16), sample_dir / "z0.pt")
                torch.save(z1.detach().cpu().to(torch.bfloat16), sample_dir / "z1.pt")
                torch.save(z0_recon.detach().cpu().to(torch.bfloat16), sample_dir / "z0_recon.pt")
                for target, tensor in checkpoints.items():
                    torch.save(tensor, sample_dir / f"z_t_{int(round(target * 100)):02d}.pt")
                save_video(sample_dir / "source_video.mp4", src_video,   fps=int(frame_rate))
                save_video(sample_dir / "recon_video.mp4",  recon_video, fps=int(frame_rate))

                meta = {
                    "rung": rung, "sample_id": sample_id,
                    "z0_source": z0_source, "conditioning": cond_mode,
                    "sample_set": sample_set,
                    "prompt": prompt,
                    "z0_pt_path": str(z0_pt_path) if z0_pt_path else None,
                    "clip_path": str(clip_path) if clip_path else None,
                    "render_HxW": [height, width], "num_frames": num_frames,
                    "num_steps": inv_steps, "solver": inv_solver,
                    "seed": seed,
                    "active_cond_tokens": n_cond,
                    "total_tokens": int(cmask_packed.shape[1]),
                    "t_inv_s": round(t_inv_s, 1), "t_rec_s": round(t_rec_s, 1),
                    "metrics": m,
                    "sigmas_inv": sigmas_inv.tolist(),
                    "checkpoint_sigmas": list(checkpoints.keys()),
                }
                with (sample_dir / "meta.yaml").open("w") as f:
                    yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

                summary.append({
                    "rung": rung, "sample_id": sample_id,
                    "z0_source": z0_source, "conditioning": cond_mode,
                    "free_rel":  free["relative"], "free_cos":  free["cosine"],
                    "all_rel":   allp["relative"], "all_cos":   allp["cosine"],
                    "cond_rel":  cond["relative"] if cond else None,
                    "psnr_mean": m["psnr"]["mean"],
                    "ssim_mean": m["ssim"]["mean"],
                    "lpips_mean": m["lpips"]["mean"],
                    "t_inv_s": round(t_inv_s, 1), "t_rec_s": round(t_rec_s, 1),
                })

                del z0_packed, z1, z0_recon, src_5d, recon_5d, src_video, recon_video
                torch.cuda.empty_cache()

        # ── rollup per rung ───────────────────────────────────────────────────
        def _agg(rows: list[dict], key: str) -> dict:
            vals = [r[key] for r in rows if r[key] is not None]
            if not vals:
                return {"mean": None, "median": None, "min": None, "max": None}
            return {
                "mean":   float(statistics.fmean(vals)),
                "median": float(statistics.median(vals)),
                "min":    float(min(vals)),
                "max":    float(max(vals)),
            }

        ladder_rollup: list[dict] = []
        for rung_spec in ladder:
            rung = rung_spec["rung"]
            rows = [r for r in summary if r["rung"] == rung]
            if not rows:
                continue
            ladder_rollup.append({
                "rung": rung,
                "z0_source": rung_spec["z0_source"],
                "conditioning": rung_spec["conditioning"],
                "n_samples": len(rows),
                "free_rel":  _agg(rows, "free_rel"),
                "free_cos":  _agg(rows, "free_cos"),
                "all_rel":   _agg(rows, "all_rel"),
                "psnr_mean": _agg(rows, "psnr_mean"),
                "ssim_mean": _agg(rows, "ssim_mean"),
                "lpips_mean": _agg(rows, "lpips_mean"),
                "note": rung_spec.get("note", ""),
            })

        # ── R0 HALT gate ──────────────────────────────────────────────────────
        r0 = next((r for r in ladder_rollup if r["rung"] == "R0"), None)
        r0_halt = False
        if r0 is not None and r0["free_rel"]["mean"] is not None:
            r0_halt = r0["free_rel"]["mean"] > r0_halt_max
            if r0_halt:
                log.warning("")
                log.warning("████ R0 HALT GATE TRIPPED ████  R0 free_rel mean=%.4f > %.4f",
                            r0["free_rel"]["mean"], r0_halt_max)
                log.warning("The unified harness does NOT reproduce exp_029. R1-R5 are "
                            "UNTRUSTWORTHY until the harness is debugged.")
            else:
                log.info("[R0 halt gate] OK — R0 free_rel mean=%.4f ≤ %.4f (harness reproduces exp_029)",
                         r0["free_rel"]["mean"], r0_halt_max)

        # ── write summary + snapshot ──────────────────────────────────────────
        with (run_dir / "config_snapshot.yaml").open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        with (run_dir / "summary.yaml").open("w") as f:
            yaml.safe_dump({
                "run_id": run_id,
                "r0_halt_triggered": r0_halt,
                "ladder": ladder_rollup,
                "samples": summary,
            }, f, sort_keys=False, allow_unicode=True)

        # ── ladder table ──────────────────────────────────────────────────────
        widths = [4, 14, 9, 9, 9, 9, 7, 7, 7]
        headers = ["rung", "z0_source", "cond", "free_rel", "free_cos", "all_rel",
                   "PSNR", "SSIM", "LPIPS"]

        def _row(cells: list[str]) -> str:
            return "│ " + " │ ".join(c.ljust(w) for c, w in zip(cells, widths)) + " │"

        def _sep(l: str, m: str, r: str) -> str:
            return l + m.join("─" * (w + 2) for w in widths) + r

        log.info("")
        log.info("exp_031 — RF-inversion localization ladder  (FREE-positions metric is primary)")
        log.info(_sep("┌", "┬", "┐"))
        log.info(_row(headers))
        log.info(_sep("├", "┼", "┤"))
        prev_free = None
        for rr in ladder_rollup:
            fr = rr["free_rel"]["mean"]
            cliff = ""
            if prev_free is not None and prev_free > 1e-9 and fr is not None:
                ratio = fr / prev_free
                if ratio >= 3.0:
                    cliff = f"  ◄── CLIFF ×{ratio:.1f}"
            log.info(_row([
                rr["rung"],
                rr["z0_source"],
                rr["conditioning"],
                f"{fr:.4f}" if fr is not None else "—",
                f"{rr['free_cos']['mean']:.5f}" if rr["free_cos"]["mean"] is not None else "—",
                f"{rr['all_rel']['mean']:.4f}" if rr["all_rel"]["mean"] is not None else "—",
                f"{rr['psnr_mean']['mean']:.2f}",
                f"{rr['ssim_mean']['mean']:.4f}",
                f"{rr['lpips_mean']['mean']:.4f}",
            ]) + cliff)
            prev_free = fr
        log.info(_sep("└", "┴", "┘"))
        log.info("Cliff = first rung whose free_rel mean is ≥3× its predecessor's.")
        log.info("[done] %s → %s", run_id, run_dir)


if __name__ == "__main__":
    main()
