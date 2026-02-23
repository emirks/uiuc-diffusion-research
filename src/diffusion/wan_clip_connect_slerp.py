"""Wan 2.1 clip-to-clip video connecting pipeline — SLERP-guided variant.

Extends the hard-constraint denoising baseline with SLERP (Spherical Linear
Interpolation) guidance for the middle latent frames.

Baseline approach
-----------------
The middle frames are initialised as pure Gaussian noise and denoised freely
between the hard-constrained anchor regions.  The model must invent the entire
transition from scratch.

SLERP guidance approach (this file)
-------------------------------------
Before the denoising loop, a "ghost" transition tensor z_guide is computed by
spherically interpolating between the boundary frames of the two anchor clips:

    z_guide[k] = slerp(z_start[-1], z_end[0], alpha_k)

    alpha_k = (k + 1) / (mid_len + 1),  k = 0 … mid_len-1

The canvas middle section is then initialised as a noisy version of z_guide
at the first noise level sigma_0, instead of pure Gaussian noise:

    latents_mid_init = (1 - sigma_0) * z_guide + sigma_0 * noise

This gives the diffusion model a coherent starting point that already morphs
from the start clip's latent features to the end clip's latent features.  The
model still denoises freely (no per-step override for the middle), so it can
add realistic motion and appearance detail on top of the SLERP skeleton.

Architecture (Wan 2.1 VAE F8T4C16 at 480p):
  Temporal compression  : 4x  →  (F-1)//4 + 1 latent frames
  Spatial compression   : 8x  →  H//8, W//8 latent pixels
  Latent channels       : 16
"""
from __future__ import annotations

import html
from dataclasses import dataclass

import regex as re
import torch
from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_ftfy_available, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from transformers import AutoTokenizer, UMT5EncoderModel

if is_ftfy_available():
    import ftfy

logger = logging.get_logger(__name__)


# ── Public tensor helpers ─────────────────────────────────────────────────────

def normalize_clip_to_neg_one_to_one(clip: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] → [-1, 1]. Input: (B, C, F, H, W) float."""
    return clip * 2.0 - 1.0


def denormalize_clip_to_zero_to_one(clip: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] → [0, 1]. Input: (B, C, F, H, W) float."""
    return (clip + 1.0) * 0.5


# ── Latent layout ─────────────────────────────────────────────────────────────

@dataclass
class WanLatentLayout:
    """Latent-space dimensions for a single clip-to-clip run."""
    latent_frames: int  # total frames on the latent canvas
    latent_h: int       # latent spatial height
    latent_w: int       # latent spatial width
    anchor_len: int     # latent frames occupied by each anchor clip
    mid_start: int      # first latent frame of the generated middle section
    mid_end: int        # one-past-last latent frame of the generated middle


def derive_latent_layout(
    *,
    num_frames: int,
    height: int,
    width: int,
    anchor_frames: int,
    temporal_compression: int = 4,
    spatial_compression: int = 8,
) -> WanLatentLayout:
    """Compute latent canvas layout from pixel-space parameters.

    Uses the Wan VAE causal formula: latent_T = (F - 1) // temporal_compression + 1.
    Raises ValueError if the anchor clips leave no room for a middle section.
    """
    latent_frames = (num_frames - 1) // temporal_compression + 1
    anchor_len    = (anchor_frames - 1) // temporal_compression + 1
    latent_h      = height // spatial_compression
    latent_w      = width  // spatial_compression
    mid_start     = anchor_len
    mid_end       = latent_frames - anchor_len

    if mid_end <= mid_start:
        raise ValueError(
            f"No middle frames available: total_latent={latent_frames}, "
            f"anchor_len={anchor_len} per side. "
            "Increase num_frames or decrease anchor_frames."
        )

    return WanLatentLayout(
        latent_frames=latent_frames,
        latent_h=latent_h,
        latent_w=latent_w,
        anchor_len=anchor_len,
        mid_start=mid_start,
        mid_end=mid_end,
    )


# ── Clip validation ───────────────────────────────────────────────────────────

def validate_clip_tensor(tensor: torch.Tensor, name: str = "clip") -> None:
    """Assert shape (B, C, F, H, W) and values in [-1, 1].

    Raises ValueError on any violation.
    """
    if tensor.ndim != 5:
        raise ValueError(
            f"{name}: expected 5-D tensor (B, C, F, H, W), got shape {tuple(tensor.shape)}"
        )
    lo = tensor.min().item()
    hi = tensor.max().item()
    if lo < -1.05 or hi > 1.05:
        raise ValueError(
            f"{name}: values outside [-1, 1] range "
            f"(min={lo:.3f}, max={hi:.3f}). "
            "Normalize clips to [-1, 1] before passing them to the pipeline."
        )


# ── Hard constraint ───────────────────────────────────────────────────────────

def overwrite_anchor_regions(
    latents: torch.Tensor,
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    layout: WanLatentLayout,
) -> torch.Tensor:
    """Return a new tensor with start/end sections replaced by z_start and z_end.

    The middle section (layout.mid_start : layout.mid_end) is left unchanged.
    """
    out = latents.clone()
    out[:, :, :layout.anchor_len, :, :] = z_start
    out[:, :, layout.mid_end:,    :, :] = z_end
    return out


# ── SLERP helpers ─────────────────────────────────────────────────────────────

def slerp(z0: torch.Tensor, z1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Spherical linear interpolation between two latent tensors.

    Both tensors are treated as a single high-dimensional vector (flattened).
    The angle Omega between them is computed in float64 for numerical
    stability, then the weighted sum of the two directions is returned in the
    original dtype and shape.

    Falls back to linear interpolation when the vectors are nearly parallel
    (|Omega| < 1e-6 rad) to avoid division by zero.

    Args:
        z0: Start tensor, arbitrary shape.
        z1: End tensor, same shape as z0.
        alpha: Interpolation factor.  0.0 → z0, 1.0 → z1.

    Returns:
        Interpolated tensor with same shape and dtype as z0.
    """
    orig_dtype = z0.dtype
    orig_shape = z0.shape

    # Work in float64 for the angle computation to avoid numerical drift with
    # very high-dimensional vectors (e.g. 16×60×106 ≈ 100 k elements).
    a = z0.reshape(-1).to(torch.float64)
    b = z1.reshape(-1).to(torch.float64)

    a_norm = a / (a.norm() + 1e-12)
    b_norm = b / (b.norm() + 1e-12)

    dot   = (a_norm * b_norm).sum().clamp(-1.0, 1.0)
    omega = torch.acos(dot)

    if omega.abs().item() < 1e-6:
        # Vectors are (nearly) parallel → linear interpolation is safe.
        return ((1.0 - alpha) * z0 + alpha * z1).to(orig_dtype)

    sin_omega = torch.sin(omega)
    result = (
        torch.sin((1.0 - alpha) * omega) / sin_omega * a
        + torch.sin(alpha * omega) / sin_omega * b
    )
    return result.reshape(orig_shape).to(orig_dtype)


def compute_slerp_guidance(
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    layout: WanLatentLayout,
) -> torch.Tensor:
    """Build a SLERP-interpolated guidance tensor for the middle latent frames.

    Interpolates between the boundary frames of the two anchor clips:
        z_start[:, :, -1, :, :]  — last latent frame of the start anchor
        z_end[:, :,   0, :, :]  — first latent frame of the end anchor

    For each of the mid_len middle slots, alpha is spaced evenly in (0, 1):
        alpha_k = (k + 1) / (mid_len + 1)

    This ensures alpha never reaches exactly 0 or 1, keeping the guide as a
    true interpolation rather than a copy of either boundary.

    Args:
        z_start: Encoded start anchor, shape (1, C, anchor_len, H, W).
        z_end:   Encoded end anchor,   shape (1, C, anchor_len, H, W).
        layout:  Latent layout from derive_latent_layout().

    Returns:
        z_guide of shape (1, C, mid_len, H, W) in the same dtype as z_start.
    """
    mid_len = layout.mid_end - layout.mid_start

    # Boundary frames: (1, C, H, W) each.
    z_a = z_start[:, :, -1, :, :]
    z_b = z_end[:,   :,  0, :, :]

    frames: list[torch.Tensor] = []
    for k in range(mid_len):
        alpha = (k + 1) / (mid_len + 1)
        frames.append(slerp(z_a, z_b, alpha).unsqueeze(2))  # (1, C, 1, H, W)

    return torch.cat(frames, dim=2)  # (1, C, mid_len, H, W)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _clean_prompt(text: str) -> str:
    if is_ftfy_available():
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


class WanVideoConnectingSlerpPipeline(DiffusionPipeline):
    """Clip-to-clip video connecting pipeline with SLERP-guided initialisation.

    Extends the hard-constraint baseline by initialising the middle latent
    frames with a spherically-interpolated "ghost" transition between the two
    anchor clips, rather than pure Gaussian noise.  The denoising loop then
    refines this structured initialisation while keeping the anchor regions
    hard-constrained at every step.

    Component layout matches Wan-AI/Wan2.1-T2V-14B-Diffusers (and 1.3B).

    Typical usage::

        pipe = WanVideoConnectingSlerpPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
        )
        result = pipe(
            start_clip=start_tensor,   # (1, 3, 24, 480, 848) in [-1, 1]
            end_clip=end_tensor,
            prompt="...",
            num_frames=72,
        )
        frames = result.frames[0]
    """

    # Controls model offload order when enable_model_cpu_offload() is called.
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: WanTransformer3DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLWan,
    ) -> None:
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
        )
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal
        self.vae_scale_factor_spatial  = self.vae.config.scale_factor_spatial
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    # ── Text encoding ─────────────────────────────────────────────────────────

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str],
        max_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        prompt = [prompt] if isinstance(prompt, str) else list(prompt)
        prompt = [_clean_prompt(p) for p in prompt]

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        mask      = text_inputs.attention_mask.to(device)
        seq_lens  = mask.gt(0).sum(dim=1).long()

        embeds = self.text_encoder(input_ids, mask).last_hidden_state
        embeds = embeds.to(dtype=dtype, device=device)

        # Trim each sequence to its true length, then re-pad to a uniform shape.
        # This matches the official WanPipeline behaviour exactly.
        embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
        embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in embeds],
            dim=0,
        )
        return embeds

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None,
        do_cfg: bool,
        max_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode prompt and optional negative prompt into T5 embeddings."""
        prompt_embeds = self._get_t5_prompt_embeds(
            prompt, max_sequence_length, device, dtype
        )
        negative_embeds = None
        if do_cfg:
            neg = negative_prompt if negative_prompt is not None else ""
            negative_embeds = self._get_t5_prompt_embeds(
                neg, max_sequence_length, device, dtype
            )
        return prompt_embeds, negative_embeds

    # ── VAE helpers ───────────────────────────────────────────────────────────

    def _vae_mean_and_std_inv(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std_inv) for the VAE latent normalisation transform.

        The official pipeline stores the normalisation as:
            std_inv = 1 / config.latents_std
        and converts denoised latents → VAE input via:
            vae_input = latents / std_inv + mean
                      = latents * config.latents_std + config.latents_mean

        The inverse (VAE output → denoiser latents) is therefore:
            latents = (z_vae - mean) * std_inv
        """
        z_dim = self.vae.config.z_dim
        mean    = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, z_dim, 1, 1, 1)
            .to(device=device, dtype=dtype)
        )
        std_inv = (
            (1.0 / torch.tensor(self.vae.config.latents_std))
            .view(1, z_dim, 1, 1, 1)
            .to(device=device, dtype=dtype)
        )
        return mean, std_inv

    def _encode_clip(
        self,
        clip: torch.Tensor,   # (B, C, F, H, W) in [-1, 1]
        device: torch.device,
    ) -> torch.Tensor:
        """Encode a pixel-space clip to normalised denoiser latents (float32).

        Uses the posterior mode (deterministic) rather than a sample to give
        stable, noise-free anchor representations.
        """
        clip = clip.to(device=device, dtype=self.vae.dtype)
        z    = self.vae.encode(clip).latent_dist.mode()
        mean, std_inv = self._vae_mean_and_std_inv(device=z.device, dtype=torch.float32)
        return (z.to(torch.float32) - mean) * std_inv

    # ── Main call ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def __call__(
        self,
        start_clip: torch.Tensor,           # (1, 3, anchor_frames, H, W) in [-1, 1]
        end_clip: torch.Tensor,             # (1, 3, anchor_frames, H, W) in [-1, 1]
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        num_frames: int = 72,
        height: int = 480,
        width: int = 848,
        anchor_frames: int = 24,
        num_inference_steps: int = 15,
        guidance_scale: float = 5.0,
        generator: torch.Generator | None = None,
        max_sequence_length: int = 226,
        output_type: str = "np",
        return_dict: bool = True,
        enable_vae_tiling: bool = True,
        enable_slerp_guidance: bool = True,
    ) -> WanPipelineOutput | tuple:
        """Run the SLERP-guided clip-to-clip connecting pipeline.

        Args:
            start_clip: First anchor clip, shape (1, 3, anchor_frames, H, W),
                        float32 in [-1, 1].
            end_clip: Last anchor clip, same shape and range.
            prompt: Text describing the desired transition.
            negative_prompt: Optional negative guidance text.
            num_frames: Total pixel frames in the output (must satisfy
                        ``(F-1) % temporal_compression == 0``).
            height: Output height in pixels (must be divisible by 16).
            width: Output width in pixels (must be divisible by 16).
            anchor_frames: Number of pixel frames in each anchor clip.
            num_inference_steps: Denoising steps.
            guidance_scale: CFG scale; set to 1.0 to disable.
            generator: Optional torch.Generator for deterministic output.
            max_sequence_length: Maximum T5 token length.
            output_type: ``"np"`` (numpy arrays), ``"pil"``, or ``"latent"``.
            return_dict: Whether to return a WanPipelineOutput or a raw tuple.
            enable_vae_tiling: If True, use the VAE's tiled decode to reduce
                peak VRAM (recommended for 480p+ or long videos).
            enable_slerp_guidance: If True, initialise the middle latent frames
                as a SLERP interpolation between anchor boundary frames (noised
                to sigma_0), rather than as pure Gaussian noise.  The denoising
                loop still runs freely over the middle section; SLERP only sets
                the structured starting point.
        """
        device = self._execution_device
        do_cfg = guidance_scale > 1.0
        transformer_dtype = self.transformer.dtype

        # 1. Validate clips ───────────────────────────────────────────────────
        validate_clip_tensor(start_clip, "start_clip")
        validate_clip_tensor(end_clip,   "end_clip")

        # 2. Latent layout ────────────────────────────────────────────────────
        layout = derive_latent_layout(
            num_frames=num_frames,
            height=height,
            width=width,
            anchor_frames=anchor_frames,
            temporal_compression=self.vae_scale_factor_temporal,
            spatial_compression=self.vae_scale_factor_spatial,
        )
        logger.info(
            "Latent layout: total=%d  anchor=%d  middle=[%d:%d]  h=%d  w=%d",
            layout.latent_frames, layout.anchor_len,
            layout.mid_start, layout.mid_end,
            layout.latent_h, layout.latent_w,
        )

        # 3. Text embeddings ──────────────────────────────────────────────────
        prompt_embeds, negative_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_cfg=do_cfg,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=transformer_dtype,
        )

        # 4. Encode anchor clips to normalised latents (float32) ──────────────
        z_start = self._encode_clip(start_clip, device=device)
        z_end   = self._encode_clip(end_clip,   device=device)

        # 5. Initialise canvas and scheduler ──────────────────────────────────
        canvas_shape = (
            1,
            self.transformer.config.in_channels,
            layout.latent_frames,
            layout.latent_h,
            layout.latent_w,
        )

        # Scheduler must be set before reading sigma_0 for SLERP init.
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # descending (high noise → low noise)

        # Canvas: start with pure noise everywhere.
        latents = randn_tensor(
            canvas_shape, generator=generator, device=device, dtype=torch.float32
        )

        if enable_slerp_guidance:
            # Compute SLERP "ghost" transition for the middle frames.
            # Spherically interpolates from the last frame of z_start to the
            # first frame of z_end, producing mid_len intermediate latents.
            z_guide = compute_slerp_guidance(z_start, z_end, layout)

            # Flow-matching forward process: x_sigma = (1-sigma)*x_0 + sigma*eps
            # Apply this to z_guide at sigma_0 so the middle canvas starts from
            # a noisy version of the SLERP skeleton rather than pure noise.
            sigma_0 = self.scheduler.sigmas[0].item()
            noise_guide = randn_tensor(
                z_guide.shape, generator=generator, device=device, dtype=torch.float32
            )
            z_guide_noisy = (1.0 - sigma_0) * z_guide + sigma_0 * noise_guide
            latents[:, :, layout.mid_start:layout.mid_end, :, :] = z_guide_noisy

            logger.info(
                "SLERP guidance: sigma_0=%.4f  mid_frames=%d",
                sigma_0, layout.mid_end - layout.mid_start,
            )

        # 7. Denoising loop ───────────────────────────────────────────────────
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # Expand timestep to batch dimension (required by Wan transformer).
                timestep = t.expand(latents.shape[0])

                # Forward pass — conditional.
                # Wan uses separate cond/uncond passes with a key-value cache,
                # NOT a concatenated batch (unlike older DDPM-style pipelines).
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents.to(transformer_dtype),
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]

                # Forward pass — unconditional (classifier-free guidance).
                if do_cfg:
                    with self.transformer.cache_context("uncond"):
                        noise_uncond = self.transformer(
                            hidden_states=latents.to(transformer_dtype),
                            timestep=timestep,
                            encoder_hidden_states=negative_embeds,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # Scheduler step: x_t → x_{t-1}.
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # Hard constraint: overwrite anchor regions.
                # Re-noise the clean anchor latents to the noise level of the
                # NEXT timestep (sigma_{i+1}) so they blend seamlessly with the
                # generated middle section.
                #
                # Flow-matching forward process: x_sigma = (1-sigma)*x_0 + sigma*eps
                # scheduler.sigmas has len(timesteps)+1 entries; sigmas[i+1] is
                # the noise level that latents should be at after step i.
                if i < len(timesteps) - 1:
                    sigma_next = self.scheduler.sigmas[i + 1].item()
                    noise_s = randn_tensor(
                        z_start.shape, generator=generator, device=device, dtype=z_start.dtype
                    )
                    noise_e = randn_tensor(
                        z_end.shape, generator=generator, device=device, dtype=z_end.dtype
                    )
                    z_start_noisy = (1.0 - sigma_next) * z_start + sigma_next * noise_s
                    z_end_noisy   = (1.0 - sigma_next) * z_end   + sigma_next * noise_e
                    latents = overwrite_anchor_regions(latents, z_start_noisy, z_end_noisy, layout)
                else:
                    # Final step: paste the clean (zero-noise) anchor latents directly.
                    latents = overwrite_anchor_regions(latents, z_start, z_end, layout)

                progress_bar.update()

        # 8. Decode latents → video ───────────────────────────────────────────
        if output_type == "latent":
            video = latents
        else:
            latents = latents.to(dtype=self.vae.dtype)
            # Invert the normalisation applied during encoding to recover the
            # raw VAE latent space: vae_input = latents / std_inv + mean
            #                                 = latents * config.latents_std + config.latents_mean
            mean, std_inv = self._vae_mean_and_std_inv(
                device=latents.device, dtype=self.vae.dtype
            )
            vae_input = latents / std_inv + mean
            # Tiled decode reduces peak VRAM by decoding in spatial tiles instead
            # of the full frame at once (important for 480p+ or long videos).
            if enable_vae_tiling:
                self.vae.enable_tiling()
            video = self.vae.decode(vae_input, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)
        return WanPipelineOutput(frames=video)
