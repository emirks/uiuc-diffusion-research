"""LTX-2 stage-1 latent initialisation strategies for C2V experiments.

All techniques answer one question: what should the stage-1 output latent look
like at initialisation (before the denoising loop)?

Each strategy implements the ``LatentInitStrategy`` protocol and returns
``(initial_video_latent, noise_scale)``.  The pipeline passes these to
``denoise_audio_video()``, which feeds them into the GaussianNoiser:

    latent = noise × (mask × noise_scale) + initial_latent × (1 - mask × noise_scale)

For output tokens (mask=1):
    latent = noise_scale × noise  +  (1 - noise_scale) × initial_latent

``noise_scale=1.0`` → pure noise (current exp_016 baseline).
``noise_scale<1.0`` → the initial_latent bleeds through as a structural bias.

Adding a new technique:
  1. Add a class here that implements ``compute()``.
  2. Import it in the experiment's run.py.
  3. Pass an instance as ``latent_init_fn=`` to the pipeline.
  No other files need changing (beyond the one-time CHANGES.md entry).
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import torch

from ltx_core.model.video_vae import VideoEncoder
from ltx_core.types import VideoPixelShape
from ltx_pipelines.utils.args import ClipConditioningInput
from ltx_pipelines.utils.media_io import load_video_conditioning

log = logging.getLogger(__name__)

# LTX-2 VAE temporal downscale factor: (F_pix - 1) // 8 + 1 = F_lat
LTX_TEMPORAL_SCALE = 8


# ── Core math ──────────────────────────────────────────────────────────────────

def slerp(z0: torch.Tensor, z1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Spherical linear interpolation between two latent tensors.

    Treats each tensor as a single high-dimensional vector.  Falls back to
    linear interpolation when the vectors are nearly parallel (avoids
    division by zero).

    Args:
        z0:    Start tensor, arbitrary shape.
        z1:    End tensor, same shape as z0.
        alpha: Interpolation factor.  0.0 → z0, 1.0 → z1.

    Returns:
        Interpolated tensor with same shape and dtype as z0.
    """
    orig_dtype = z0.dtype
    orig_shape = z0.shape

    # Use float64 for the angle computation to avoid numerical drift with
    # high-dimensional vectors (e.g. 128×8×12 ≈ 12k elements per frame).
    a = z0.reshape(-1).to(torch.float64)
    b = z1.reshape(-1).to(torch.float64)

    a_norm = a / (a.norm() + 1e-12)
    b_norm = b / (b.norm() + 1e-12)

    dot   = (a_norm * b_norm).sum().clamp(-1.0, 1.0)
    omega = torch.acos(dot)

    if omega.abs().item() < 1e-6:
        # Vectors nearly parallel → linear interpolation is numerically safe.
        return ((1.0 - alpha) * z0 + alpha * z1).to(orig_dtype)

    sin_omega = torch.sin(omega)
    result = (
        torch.sin((1.0 - alpha) * omega) / sin_omega * a
        + torch.sin(alpha * omega)        / sin_omega * b
    )
    return result.reshape(orig_shape).to(orig_dtype)


# ── Protocol ───────────────────────────────────────────────────────────────────

@runtime_checkable
class LatentInitStrategy(Protocol):
    """Protocol for stage-1 video latent initialisation strategies.

    Passed as ``latent_init_fn`` to ``KeyframeInterpolationPipeline.__call__()``.
    The pipeline calls ``compute()`` after building clip conditionings but
    before ``denoise_audio_video()``, using the stage-1 ``VideoPixelShape``
    (half resolution) and the already-loaded ``VideoEncoder``.
    """

    def compute(
        self,
        clips: list[ClipConditioningInput],
        output_shape: VideoPixelShape,
        video_encoder: VideoEncoder,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, float]:
        """Return (initial_video_latent, noise_scale).

        ``initial_video_latent`` shape: ``[B, C, F_lat, H_lat, W_lat]``
        matching the stage-1 latent volume, or ``None`` to skip.

        ``noise_scale`` in [0, 1]:  1.0 → pure noise (no bias),
        values < 1 blend the guide into the initial latent.
        """
        ...


# ── Strategies ─────────────────────────────────────────────────────────────────

class PureNoiseInit:
    """Baseline: pure Gaussian noise initialisation — identical to exp_016."""

    def compute(
        self,
        clips: list[ClipConditioningInput],
        output_shape: VideoPixelShape,
        video_encoder: VideoEncoder,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, float]:
        return None, 1.0


class SlerpInit:
    """SLERP-guided initialisation of the middle temporal frames (strategy 2).

    Before denoising, the middle latent frames (indices F_lat_clip .. F_lat_out
    - F_lat_clip) are initialised as a SLERP interpolation between:
      - the *last* VAE latent frame of the start clip
      - the *first* VAE latent frame of the end clip

    alpha values are evenly spaced in (0, 1) so neither boundary is copied.

    The GaussianNoiser then mixes this guide with random noise:
        latent_mid = blend_alpha × noise + (1 - blend_alpha) × slerp_guide

    blend_alpha=1.0 → pure noise  (same as exp_016 baseline)
    blend_alpha=0.95 → 5 % guide  (subtle, recommended starting point)
    blend_alpha=0.5  → 50 % guide (strong structural bias)
    blend_alpha=0.0  → clean guide (no noise; very rigid, usually over-fit)
    """

    def __init__(self, blend_alpha: float = 0.95) -> None:
        if not 0.0 <= blend_alpha <= 1.0:
            raise ValueError(f"blend_alpha must be in [0, 1], got {blend_alpha}")
        self.blend_alpha = blend_alpha

    def compute(
        self,
        clips: list[ClipConditioningInput],
        output_shape: VideoPixelShape,
        video_encoder: VideoEncoder,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, float]:
        if len(clips) != 2:
            log.warning("SlerpInit requires exactly 2 clips; falling back to pure noise.")
            return None, 1.0

        start_clip, end_clip = clips

        # Encode both clips at the stage-1 output resolution.
        # This duplicates the VAE encode that also happens inside
        # clip_conditionings_by_adding_guiding_latent, but that cost is
        # negligible compared to 40 transformer denoising steps.
        start_video = load_video_conditioning(
            start_clip.path,
            output_shape.height,
            output_shape.width,
            start_clip.num_clip_frames,
            dtype,
            device,
        )
        end_video = load_video_conditioning(
            end_clip.path,
            output_shape.height,
            output_shape.width,
            end_clip.num_clip_frames,
            dtype,
            device,
        )
        encoded_start = video_encoder(start_video)   # [B, C, F_lat_clip, H_lat, W_lat]
        encoded_end   = video_encoder(end_video)

        B, C, F_lat_clip, H_lat, W_lat = encoded_start.shape
        F_lat_out = (output_shape.frames - 1) // LTX_TEMPORAL_SCALE + 1

        mid_start = F_lat_clip
        mid_end   = F_lat_out - F_lat_clip
        mid_len   = mid_end - mid_start

        if mid_len <= 0:
            log.warning(
                "SlerpInit: no middle frames available "
                "(F_lat_out=%d, F_lat_clip=%d); falling back to pure noise.",
                F_lat_out, F_lat_clip,
            )
            return None, 1.0

        log.info(
            "SlerpInit: F_lat_out=%d  F_lat_clip=%d  mid=[%d:%d]  blend_alpha=%.3f",
            F_lat_out, F_lat_clip, mid_start, mid_end, self.blend_alpha,
        )

        # Boundary latent frames: last of start, first of end.
        z_a = encoded_start[:, :, -1, :, :].float()   # [B, C, H_lat, W_lat]
        z_b = encoded_end[:,   :,  0, :, :].float()

        # SLERP guide: alpha in (1/(mid+1), 2/(mid+1), ..., mid/(mid+1))
        z_guide_frames: list[torch.Tensor] = []
        for k in range(mid_len):
            alpha_k = (k + 1) / (mid_len + 1)
            z_guide_frames.append(slerp(z_a, z_b, alpha_k).unsqueeze(2))  # [B,C,1,H,W]
        z_guide = torch.cat(z_guide_frames, dim=2)   # [B, C, mid_len, H_lat, W_lat]

        # Full output latent: zeros outside the middle → after noiser those
        # positions become blend_alpha × noise + 0 ≈ noise (fine, since the
        # frozen conditioning tokens guide those boundary positions anyway).
        z_init = torch.zeros(B, C, F_lat_out, H_lat, W_lat, device=device, dtype=encoded_start.dtype)
        z_init[:, :, mid_start:mid_end, :, :] = z_guide.to(encoded_start.dtype)

        return z_init, self.blend_alpha
