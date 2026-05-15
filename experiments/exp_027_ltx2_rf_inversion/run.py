    """exp_027 — LTX-2 RF-Solver flow inversion.

    Workflow per sample
    -------------------
    1. Run LTX-2 Stage-1 generation (40 steps, CFG=4, C2V conditioning) — same recipe
    as exp_020. Output is the source-of-truth packed latent ``z0``.
    2. Invert ``z0 → z1`` with RF-Solver midpoint 2nd-order (30 steps, CFG=1),
    replicating LTX2ConditionPipeline's per-token timestep + x0-domain clamp.
    3. Reconstruct ``z1 → z0_recon`` with the **same** solver and grid in reverse.
    4. Decode both through the VAE; per-frame LPIPS gate (target mean < 0.05).
    5. If the gate fails, retry inversion+reconstruction at 50 steps.

    Outputs cached per sample (under run_dir/<sample_id>/):
    - z0.pt, z1.pt, z_t_25.pt, z_t_50.pt, z_t_75.pt   (packed bfloat16)
    - source_video.mp4, recon_video.mp4
    - inv_meta.yaml — LPIPS stats, σ grid, NFE, gate pass/fail
    """
    from __future__ import annotations

    import argparse
    import copy
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
    NUM_TRAIN_TIMESTEPS = 1000  # FlowMatchEulerDiscreteScheduler default; t = sigma * this

    log = logging.getLogger(__name__)


    # ── Frame / clip helpers (identical to exp_020) ──────────────────────────────

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


    # ── RF-Solver inverter ────────────────────────────────────────────────────────

    class RFInverter:
        """RF-Solver midpoint 2nd-order inversion + reconstruction for LTX-2 Stage 1.

        Operates entirely on packed normalized latents ``[B, N, C]`` and replicates
        LTX2ConditionPipeline's two conditioning mechanisms:

        (i)  per-token timestep ``t·(1−mask)`` — conditioned tokens see ~0 diffusion time
        (ii) x0-domain clamp: predict x0 from velocity, clamp conditioned positions to
            the clean clip latents, convert back to velocity. Same as the pipeline
            does at every step (pipeline_ltx2_condition.py:1387–1395).

        No CFG (uncond only). No audio update (audio stream is zeros throughout).
        """

        def __init__(self, pipe: LTX2ConditionPipeline, device: str = DEVICE) -> None:
            self.pipe = pipe
            self.device = device
            # The transformer is the only module that runs during inversion.
            self.transformer = pipe.transformer
            self.vae = pipe.vae

            # Cached per sample
            self.conditioning_mask: torch.Tensor | None = None    # [1, N, 1]
            self.clean_latents:     torch.Tensor | None = None    # [1, N, C]
            self.prompt_embeds:     torch.Tensor | None = None    # text embeds (uncond pathway, see note)
            self.prompt_attn_mask:  torch.Tensor | None = None
            self.audio_prompt_embeds:    torch.Tensor | None = None
            self.audio_prompt_attn_mask: torch.Tensor | None = None
            self.audio_zeros:       torch.Tensor | None = None    # [1, N_audio, C_audio]
            self.audio_num_frames:  int = 1
            self.latent_num_frames: int = 0
            self.latent_height:     int = 0
            self.latent_width:      int = 0
            self.frame_rate:        float = 24.0
            # σ grid for the current solver run (in [0, 1])
            self.sigmas: np.ndarray | None = None

        # ── set up for a sample ──────────────────────────────────────────────────

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
            max_sequence_length: int = 256,
        ) -> None:
            """Encode the prompt (positive only — CFG=1) and stash conditioning state.

            Mirrors LTX2ConditionPipeline lines 1190–1214: encode_prompt → additive
            attention mask → connectors that produce *both* video and audio text
            embeddings from the same prompt encoding.
            """
            device = self.device

            # 1) Gemma prompt embeddings (positive only — CFG=1).
            (
                prompt_embeds,
                prompt_attention_mask,
                _neg_embeds,
                _neg_mask,
            ) = self.pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=1,
                device=device,
                max_sequence_length=max_sequence_length,
            )

            # 2) AV connectors split text embedding into video / audio streams and
            #    produce a combined (already-additive) attention mask.
            additive_attention_mask = (1 - prompt_attention_mask.to(prompt_embeds.dtype)) * -1_000_000.0
            connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = self.pipe.connectors(
                prompt_embeds, additive_attention_mask, additive_mask=True
            )
            self.prompt_embeds = connector_prompt_embeds
            self.prompt_attn_mask = connector_attention_mask
            self.audio_prompt_embeds = connector_audio_prompt_embeds
            self.audio_prompt_attn_mask = connector_attention_mask

            self.conditioning_mask = conditioning_mask.to(device=device)
            self.clean_latents = clean_latents.to(device=device)
            self.latent_num_frames = latent_num_frames
            self.latent_height = latent_height
            self.latent_width = latent_width
            self.frame_rate = frame_rate

            # 3) Dummy audio latents. The transformer requires `audio_hidden_states`
            # to be present even when we don't care about audio output. We construct
            # the minimum-valid packed audio tensor at noise_scale=0 → all zeros
            # post-normalization. The audio stream is never stepped/updated.
            num_mel_bins = (
                self.pipe.audio_vae.config.mel_bins
                if getattr(self.pipe, "audio_vae", None) is not None
                else 64
            )
            audio_channels = (
                self.pipe.audio_vae.config.latent_channels
                if getattr(self.pipe, "audio_vae", None) is not None
                else 8
            )
            self.audio_zeros = self.pipe.prepare_audio_latents(
                batch_size=1,
                num_channels_latents=audio_channels,
                audio_latent_length=1,
                num_mel_bins=num_mel_bins,
                noise_scale=0.0,
                dtype=prompt_embeds.dtype,
                device=device,
                generator=None,
                latents=None,
            )
            self.audio_num_frames = 1

        # ── transformer call (uncond, masked timestep) ───────────────────────────

        def _call_transformer(self, z_packed: torch.Tensor, sigma_scalar: float) -> torch.Tensor:
            """One uncond transformer call. Returns velocity prediction in packed shape (fp32)."""
            bsz = z_packed.shape[0]
            t_dtype = self.transformer.dtype  # bf16 — solver state may be fp32, so cast at the boundary
            z_in = z_packed.to(t_dtype)
            # Train timestep convention: t = σ × num_train_timesteps (in [0, 1000]).
            t_value = float(sigma_scalar) * NUM_TRAIN_TIMESTEPS
            t = torch.full((bsz,), t_value, device=z_packed.device, dtype=t_dtype)
            # Per-token timestep: conditioned tokens see ~0 diffusion time.
            # conditioning_mask: [B, N, 1] → squeeze last dim → [B, N]; broadcast with t.
            cond_mask_t = self.conditioning_mask.squeeze(-1).to(t_dtype)
            video_timestep = t.unsqueeze(-1) * (1 - cond_mask_t)  # [B, N]
            audio_timestep = t

            noise_pred_video, _noise_pred_audio = self.transformer(
                hidden_states=z_in,
                audio_hidden_states=self.audio_zeros,
                encoder_hidden_states=self.prompt_embeds,
                audio_encoder_hidden_states=self.audio_prompt_embeds,
                timestep=video_timestep,
                audio_timestep=audio_timestep,
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

        # ── x0-domain clamp on the packed velocity ───────────────────────────────

        def _x0_clamp_velocity(
            self,
            z_packed: torch.Tensor,
            v_packed: torch.Tensor,
            sigma_scalar: float,
        ) -> torch.Tensor:
            """Apply the same conditioning enforcement LTX2ConditionPipeline does.

            x0_pred       = z − v·σ
            x0_pred_clean = x0_pred·(1−m) + clean·m
            v_clamped     = (z − x0_pred_clean) / σ

            At σ → 0 the divide by σ degenerates: the formula squashes *all* velocity
            components (including uncond positions) toward zero. We sidestep that by
            returning ``v`` unchanged when σ is below the LTX-2 ``shift_terminal``;
            the hard re-clamp of conditioned positions inside ``_midpoint_step``
            keeps the C2V tokens pinned in that regime.
            """
            sigma = float(sigma_scalar)
            if sigma < 1e-4:
                return v_packed
            mask = self.conditioning_mask  # [B, N, 1] broadcasts over channels
            x0_pred = z_packed - v_packed * sigma
            x0_pred_clean = x0_pred * (1 - mask) + self.clean_latents.float() * mask
            v_clamped = (z_packed - x0_pred_clean) / sigma
            return v_clamped.to(v_packed.dtype)

        # ── midpoint step (one outer step = two transformer calls) ───────────────

        def _midpoint_step(
            self,
            z: torch.Tensor,
            sigma_curr: float,
            sigma_next: float,
        ) -> torch.Tensor:
            """Advance z from σ_curr to σ_next via the RF-Solver midpoint update.

            Same code path whether σ_next > σ_curr (inversion) or σ_next < σ_curr
            (reconstruction); the algebraic sign of dτ falls out of (σ_next − σ_curr).
            """
            dtau = sigma_next - sigma_curr
            sigma_mid = sigma_curr + dtau / 2.0

            # First call at (z, σ_curr)
            v = self._call_transformer(z, sigma_curr)
            v = self._x0_clamp_velocity(z, v, sigma_curr)

            # Half-step probe
            z_mid = z + (dtau / 2.0) * v

            # Second call at (z_mid, σ_mid)
            v_mid = self._call_transformer(z_mid, sigma_mid)
            v_mid = self._x0_clamp_velocity(z_mid, v_mid, sigma_mid)

            # Final update uses the midpoint velocity
            z_next = z + dtau * v_mid

            # Hard re-clamp of conditioned positions to the clean clip latents.
            # The x0-clamp on the velocity already does this implicitly per step,
            # but accumulated floating-point drift is silenced here.
            mask = self.conditioning_mask
            z_next = z_next * (1 - mask) + self.clean_latents * mask
            return z_next.to(z.dtype)

        # ── inversion and reconstruction ─────────────────────────────────────────

        def _build_sigma_grid(self, num_steps: int, scheduler) -> np.ndarray:
            """30-step dynamic-shifted σ grid that matches Stage-1's exact recipe.

            Uses a fresh copy of the Stage-1 scheduler so this call does not
            perturb any other scheduler state held by the pipeline.
            """
            # Sigma seed for set_timesteps: linspace(1, 1/N, N) — same as pipeline line 1273.
            sigmas_seed = np.linspace(1.0, 1.0 / num_steps, num_steps)

            # Video sequence length (packed) for mu — depends on patchified token count.
            N = self.conditioning_mask.shape[1]
            mu = calculate_shift(
                N,
                scheduler.config.get("base_image_seq_len", 1024),
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_shift", 0.95),
                scheduler.config.get("max_shift", 2.05),
            )

            sched = copy.deepcopy(scheduler)
            retrieve_timesteps(
                sched,
                num_steps,
                self.device,
                timesteps=None,
                sigmas=sigmas_seed,
                mu=mu,
            )
            # After set_timesteps, sched.sigmas is a (N+1,) tensor descending from σ_max≈1
            # to (typically) shift_terminal ≈ 0.1. We need the σ values at each "step",
            # which means the (N+1,) endpoint array.
            sigmas_full = sched.sigmas.cpu().numpy().astype(np.float64)
            return sigmas_full

        @torch.inference_mode()
        def invert(
            self,
            z0_packed: torch.Tensor,
            num_steps: int,
            scheduler,
            checkpoint_sigmas: list[float],
        ) -> tuple[torch.Tensor, dict[float, torch.Tensor], np.ndarray]:
            """z0 (clean) → z1 (noise) via RF-Solver midpoint.

            Returns (z1, {σ: z_at_σ} checkpoints, σ_grid used).
            """
            sigmas_gen = self._build_sigma_grid(num_steps, scheduler)
            # Inversion grid: reverse order (ascending σ), drop duplicate endpoints.
            sigmas_inv = sigmas_gen[::-1].copy()  # ascending

            z = z0_packed.clone()
            # Hard clamp at σ=0 too — conditioned positions are already z0's clean positions.
            mask = self.conditioning_mask
            z = z * (1 - mask) + self.clean_latents * mask

            # Pre-select the checkpoint step indices (closest to each target σ).
            # We log after the step lands on σ_{i+1}, i.e. after stepping from sigmas_inv[i].
            checkpoints: dict[float, torch.Tensor] = {}
            target_to_step: dict[float, int] = {}
            for target in checkpoint_sigmas:
                # Use σ_after = sigmas_inv[i+1] for matching; pick the i whose σ_after is closest.
                sigmas_after = sigmas_inv[1:]
                idx_after = int(np.argmin(np.abs(sigmas_after - target)))
                target_to_step[target] = idx_after  # 0-indexed across the step loop

            for i in range(len(sigmas_inv) - 1):
                sigma_curr = float(sigmas_inv[i])
                sigma_next = float(sigmas_inv[i + 1])
                z = self._midpoint_step(z, sigma_curr, sigma_next)
                log.info(
                    "  invert step %2d/%d  σ: %.4f → %.4f   ‖z‖=%.2f",
                    i + 1, len(sigmas_inv) - 1, sigma_curr, sigma_next, z.float().norm().item(),
                )
                # Record any checkpoint that resolves at this step.
                for target, idx_after in target_to_step.items():
                    if idx_after == i:
                        checkpoints[target] = z.detach().cpu().to(torch.bfloat16).clone()

            z1 = z
            return z1, checkpoints, sigmas_inv

        @torch.inference_mode()
        def reconstruct(
            self,
            z1_packed: torch.Tensor,
            num_steps: int,
            scheduler,
        ) -> torch.Tensor:
            """z1 (noise) → z0_recon (clean) via the SAME midpoint solver, σ descending."""
            sigmas_gen = self._build_sigma_grid(num_steps, scheduler)  # descending

            z = z1_packed.clone()
            mask = self.conditioning_mask
            # Conditioned positions are clean throughout — re-affirm at noise endpoint too.
            z = z * (1 - mask) + self.clean_latents * mask

            for i in range(len(sigmas_gen) - 1):
                sigma_curr = float(sigmas_gen[i])
                sigma_next = float(sigmas_gen[i + 1])
                z = self._midpoint_step(z, sigma_curr, sigma_next)
                log.info(
                    "  recon  step %2d/%d  σ: %.4f → %.4f   ‖z‖=%.2f",
                    i + 1, len(sigmas_gen) - 1, sigma_curr, sigma_next, z.float().norm().item(),
                )
            return z


    # ── Latent ↔ pixel helpers ────────────────────────────────────────────────────

    def normalize_and_pack(pipe: LTX2ConditionPipeline, latents_5d_denorm: torch.Tensor) -> torch.Tensor:
        """Take pipe's [B, C, F', H', W'] denormalized output → packed normalized [B, N, C]."""
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
        """Inverse of normalize_and_pack."""
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
        """[B, C, F, H, W] (denormalized) → np.uint8 [F, H_px, W_px, 3].

        Matches the pipeline's defaults (decode_timestep=0.0, decode_noise_scale=0.0)
        so there is no random noise injection — both source and reconstruction decodes
        are bit-identical given the same latent. That keeps the LPIPS gate meaningful.
        """
        latents = latents_5d_denorm.to(pipe.vae.dtype)
        timestep = None
        if pipe.vae.config.timestep_conditioning:
            timestep = torch.tensor([0.0], device=latents.device, dtype=latents.dtype)
        video = pipe.vae.decode(latents, timestep, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(video, output_type="np")  # [B, F, H, W, 3] in [0,1]
        return (np.clip(video[0], 0.0, 1.0) * 255).astype(np.uint8)


    # ── Metric suite ──────────────────────────────────────────────────────────────

    def _stats(arr: np.ndarray, *, key: str) -> dict:
        """Mean / std / min / max with worst-frame index, plus full per-frame list."""
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
        """Unified inversion-validation metrics.

        Decoded-space (source video vs reconstruction):
        - PSNR (pixel L2 fidelity, dB)
        - SSIM (luminance / contrast / structure)
        - LPIPS (perceptual; AlexNet backbone — same as the prior gate)
        - Temporal consistency: mean |Δsrc(t,t+1) − Δrec(t,t+1)|₁

        Latent-space (z0 vs z0_recon, decode-free):
        - L2, L2-per-element, relative (‖Δ‖ / ‖z0‖)
        - Cosine similarity ⟨z0, z0_recon⟩ / (‖z0‖‖z0_recon‖)

        Every per-frame array is returned alongside summary stats (worst-frame index
        included) so a single bad frame is debuggable.
        """

        def __init__(self, device: str = DEVICE) -> None:
            self.device = device
            self.lpips_model = lpips_pkg.LPIPS(net="alex", verbose=False).to(device).eval()

        # ── decoded-space ────────────────────────────────────────────────────────

        @staticmethod
        def psnr(a: np.ndarray, b: np.ndarray) -> dict:
            af = a.astype(np.float64) / 255.0
            bf = b.astype(np.float64) / 255.0
            mse = ((af - bf) ** 2).reshape(a.shape[0], -1).mean(axis=1)
            # data_range = 1.0 in float space; clamp mse to avoid log(0)
            psnr = 10.0 * np.log10(1.0 / np.maximum(mse, 1e-12))
            return _stats(psnr, key="psnr")

        @staticmethod
        def ssim(a: np.ndarray, b: np.ndarray) -> dict:
            ssims = np.array(
                [
                    _ssim_skimage(fa, fb, channel_axis=-1, data_range=255)
                    for fa, fb in zip(a, b)
                ],
                dtype=np.float64,
            )
            return _stats(ssims, key="ssim")

        @staticmethod
        def temporal(a: np.ndarray, b: np.ndarray) -> dict:
            """Mean-L1 of (frame-pair motion in src minus frame-pair motion in recon)."""
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

            ta = _prep(a)
            tb = _prep(b)
            scores = []
            for i in range(0, ta.shape[0], 8):
                d = self.lpips_model(ta[i:i + 8], tb[i:i + 8]).flatten()
                scores.append(d.detach().cpu())
            return _stats(torch.cat(scores).numpy(), key="lpips")

        # ── latent-space (decode-free) ───────────────────────────────────────────

        @staticmethod
        def latent(z0: torch.Tensor, z0_recon: torch.Tensor) -> dict:
            """Inputs: packed normalized latents [B, N, C]."""
            a = z0.float().flatten()
            b = z0_recon.float().flatten()
            diff = a - b
            l2 = float(diff.norm().item())
            n = int(a.numel())
            norm_a = float(a.norm().item())
            norm_b = float(b.norm().item())
            rel = l2 / max(norm_a, 1e-8)
            cos = float(torch.dot(a, b).item()) / max(norm_a * norm_b, 1e-8)
            return {
                "l2":             l2,
                "l2_per_element": l2 / (n ** 0.5),
                "relative":       rel,
                "cosine":         cos,
                "n_elements":     n,
                "norm_z0":        norm_a,
                "norm_z0_recon":  norm_b,
            }

        # ── one-shot evaluate ────────────────────────────────────────────────────

        def evaluate(
            self,
            src_video: np.ndarray,
            recon_video: np.ndarray,
            z0_packed: torch.Tensor,
            z0_recon_packed: torch.Tensor,
        ) -> dict:
            return {
                "psnr":     self.psnr(src_video, recon_video),
                "ssim":     self.ssim(src_video, recon_video),
                "lpips":    self.lpips(src_video, recon_video),
                "temporal": self.temporal(src_video, recon_video),
                "latent":   self.latent(z0_packed, z0_recon_packed),
            }


    # ── Video save ────────────────────────────────────────────────────────────────

    def save_video(path: pathlib.Path, video_uint8: np.ndarray, fps: int) -> None:
        """[F, H, W, 3] uint8 → mp4. Uses diffusers' export_to_video to dodge the
        torchvision.io / PyAV >=10 incompatibility (write_video sets frame.pict_type
        to the string "NONE" which newer PyAV rejects)."""
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

            # ── Parse config ─────────────────────────────────────────────────────
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
            esc_cfg         = inv_cfg["escalation"]
            latent_rel_max  = float(esc_cfg["latent_rel_max"])
            latent_cos_min  = float(esc_cfg["latent_cos_min"])
            lpips_threshold = float(esc_cfg.get("lpips_threshold", 0.05))  # reported only
            retry_steps     = int(esc_cfg["retry_num_steps"])

            if inv_cfg_scale != 1.0:
                log.warning(
                    "inversion.guidance_scale = %.2f != 1.0. This run will still execute "
                    "with CFG=1 (uncond only) — the higher-CFG path is not implemented "
                    "in exp_027 to keep the NFE budget at 60.", inv_cfg_scale,
                )

            end_idx = end_clip_index(num_frames, num_clip_frames)

            # ── Load pipeline (no LoRA — Stage 1 only; no upsampler) ────────────
            log.info("Loading LTX2ConditionPipeline from %s …", model_id)
            t0 = time.perf_counter()
            pipe = LTX2ConditionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
            pipe.enable_model_cpu_offload(device=DEVICE)
            pipe.vae.enable_tiling()
            log.info("Pipeline loaded in %.1fs.", time.perf_counter() - t0)
            # Keep the Stage-1 scheduler reference — that's the only one we'll use.
            stage1_scheduler = pipe.scheduler

            # Unified metric suite (initialized once for all samples — LPIPS net is heavy).
            metric_suite = MetricSuite(device=DEVICE)

            # ── Per-sample loop ───────────────────────────────────────────────────
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

                # ── (a) Stage-1 generation — source of truth ──────────────────────
                pipe.scheduler = stage1_scheduler
                pipe.disable_lora()
                log.info("Stage 1 generation: %dx%d  %d steps  guidance=%.1f",
                        height, width, gen_steps, gen_cfg)

                t_gen = time.perf_counter()
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
                log.info("Stage 1 done in %.1fs.  latent shape=%s",
                        time.perf_counter() - t_gen, tuple(video_latent_5d.shape))

                # ── Pack + normalize Stage-1 output → z0 ──────────────────────────
                z0_packed = normalize_and_pack(pipe, video_latent_5d).to(DEVICE)
                log.info("z0 packed shape: %s  norm=%.2f",
                        tuple(z0_packed.shape), z0_packed.float().norm().item())

                # ── Build conditioning mask + clean latents (same way prepare_latents does) ─
                latent_height = height // pipe.vae_spatial_compression_ratio
                latent_width  = width  // pipe.vae_spatial_compression_ratio
                latent_num_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1

                condition_frames, condition_strengths, condition_indices = pipe.preprocess_conditions(
                    conditions, height, width, num_frames, device=DEVICE
                )
                cond_latents_list: list[torch.Tensor] = []
                for cond_tensor in condition_frames:
                    cl = retrieve_latents(pipe.vae.encode(cond_tensor), generator=generator, sample_mode="argmax")
                    cl = pipe._normalize_latents(cl, pipe.vae.latents_mean, pipe.vae.latents_std).to(
                        device=DEVICE, dtype=z0_packed.dtype
                    )
                    cl = pipe._pack_latents(cl, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size)
                    cond_latents_list.append(cl)

                # Empty packed [B, N, 1] mask, fill via apply_visual_conditioning.
                zeros_5d = torch.zeros(
                    (1, 1, latent_num_frames, latent_height, latent_width),
                    device=DEVICE, dtype=z0_packed.dtype,
                )
                cmask_packed = pipe._pack_latents(
                    zeros_5d, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size
                )
                # apply_visual_conditioning will overwrite latents at conditioned slots; we
                # only want the mask + clean_latents, not the rewritten latents, so we pass
                # a throwaway latents tensor of the right shape.
                throwaway = torch.zeros_like(z0_packed)
                _, cmask_packed, clean_latents = pipe.apply_visual_conditioning(
                    throwaway,
                    cmask_packed,
                    cond_latents_list,
                    condition_strengths,
                    condition_indices,
                    latent_height=latent_height,
                    latent_width=latent_width,
                )
                log.info("conditioning_mask shape=%s   active tokens=%d / %d",
                        tuple(cmask_packed.shape),
                        int((cmask_packed > 0).sum().item()),
                        cmask_packed.shape[1])

                # ── Prepare RF inverter for this sample ─────────────────────────
                inverter = RFInverter(pipe, device=DEVICE)
                inverter.prepare_sample(
                    prompt=prompt,
                    conditioning_mask=cmask_packed,
                    clean_latents=clean_latents,
                    latent_num_frames=latent_num_frames,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    frame_rate=frame_rate,
                )

                # ── (b) + (c) + (d): one attempt at the canonical step count ───
                def _attempt(n_steps: int) -> dict:
                    log.info("─ Inversion attempt: %d steps  solver=%s  CFG=%.1f ─",
                            n_steps, inv_solver, inv_cfg_scale)
                    t_inv = time.perf_counter()
                    z1, checkpoints, sigmas_inv = inverter.invert(
                        z0_packed, n_steps, stage1_scheduler, ckpt_sigmas
                    )
                    t_inv_done = time.perf_counter() - t_inv
                    log.info("Inversion done in %.1fs.  z1 norm=%.2f",
                            t_inv_done, z1.float().norm().item())

                    t_rec = time.perf_counter()
                    z0_recon = inverter.reconstruct(z1, n_steps, stage1_scheduler)
                    t_rec_done = time.perf_counter() - t_rec
                    log.info("Reconstruction done in %.1fs.  z0_recon norm=%.2f",
                            t_rec_done, z0_recon.float().norm().item())

                    # Decode source + recon for pixel-space metrics.
                    src_5d   = unpack_and_denormalize(pipe, z0_packed, latent_num_frames, latent_height, latent_width)
                    recon_5d = unpack_and_denormalize(pipe, z0_recon,  latent_num_frames, latent_height, latent_width)
                    src_video   = decode_latents_to_video(pipe, src_5d)
                    recon_video = decode_latents_to_video(pipe, recon_5d)

                    # Unified metric evaluation.
                    metrics = metric_suite.evaluate(src_video, recon_video, z0_packed, z0_recon)
                    log.info(
                        "Latent: rel=%.4f  cos=%.5f  l2/elem=%.5f  (gate: rel<%.3f & cos>%.4f)",
                        metrics["latent"]["relative"], metrics["latent"]["cosine"],
                        metrics["latent"]["l2_per_element"], latent_rel_max, latent_cos_min,
                    )
                    log.info(
                        "PSNR: mean=%.2f dB  (min=%.2f @ frame %d, max=%.2f)",
                        metrics["psnr"]["mean"], metrics["psnr"]["min"],
                        metrics["psnr"]["worst_frame"], metrics["psnr"]["max"],
                    )
                    log.info(
                        "SSIM: mean=%.4f  (min=%.4f @ frame %d, max=%.4f)",
                        metrics["ssim"]["mean"], metrics["ssim"]["min"],
                        metrics["ssim"]["worst_frame"], metrics["ssim"]["max"],
                    )
                    log.info(
                        "LPIPS: mean=%.4f  (max=%.4f @ frame %d)",
                        metrics["lpips"]["mean"], metrics["lpips"]["max"],
                        metrics["lpips"]["worst_frame"],
                    )
                    log.info(
                        "Temporal: mean=%.5f  (max=%.5f @ pair %d)",
                        metrics["temporal"]["mean"], metrics["temporal"]["max"],
                        metrics["temporal"]["worst_frame"],
                    )

                    return {
                        "num_steps":   n_steps,
                        "t_inv_s":     round(t_inv_done, 1),
                        "t_rec_s":     round(t_rec_done, 1),
                        "z1":          z1.detach().cpu().to(torch.bfloat16),
                        "z0_recon":    z0_recon.detach().cpu().to(torch.bfloat16),
                        "checkpoints": checkpoints,
                        "sigmas_inv":  sigmas_inv.tolist(),
                        "metrics":     metrics,
                        "src_video":   src_video,
                        "recon_video": recon_video,
                    }

                def _gate_passes(a: dict) -> tuple[bool, dict]:
                    lat = a["metrics"]["latent"]
                    rel_ok = lat["relative"] < latent_rel_max
                    cos_ok = lat["cosine"]   > latent_cos_min
                    return (rel_ok and cos_ok), {
                        "relative":  lat["relative"],  "rel_ok":  rel_ok,
                        "cosine":    lat["cosine"],    "cos_ok":  cos_ok,
                    }

                attempts: list[dict] = []
                attempt = _attempt(inv_steps)
                attempts.append(attempt)

                gate_passed, gate_info = _gate_passes(attempt)
                retry_executed = False
                if not gate_passed:
                    log.warning(
                        "Latent gate FAILED at %d steps (rel=%.4f%s, cos=%.5f%s). "
                        "Auto-retry at %d steps …",
                        inv_steps,
                        gate_info["relative"], "" if gate_info["rel_ok"] else " ✗",
                        gate_info["cosine"],   "" if gate_info["cos_ok"] else " ✗",
                        retry_steps,
                    )
                    retry = _attempt(retry_steps)
                    attempts.append(retry)
                    retry_executed = True
                    gate_passed, gate_info = _gate_passes(retry)
                    attempt = retry  # use the retry's tensors for caching

                # ── Save artefacts (from the *last* attempt = canonical or retry) ─
                torch.save(z0_packed.detach().cpu().to(torch.bfloat16), sample_dir / "z0.pt")
                torch.save(attempt["z1"], sample_dir / "z1.pt")
                for target, tensor in attempt["checkpoints"].items():
                    tag = f"z_t_{int(round(target * 100)):02d}.pt"
                    torch.save(tensor, sample_dir / tag)

                save_video(sample_dir / "source_video.mp4", attempt["src_video"],   fps=int(frame_rate))
                save_video(sample_dir / "recon_video.mp4",  attempt["recon_video"], fps=int(frame_rate))

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
                    "inversion_cfg": inv_cfg_scale,
                    "generation": {
                        "num_steps":      gen_steps,
                        "guidance_scale": gen_cfg,
                    },
                    "attempts": [
                        {
                            "num_steps":  a["num_steps"],
                            "nfe":        a["num_steps"] * 2,
                            "t_inv_s":    a["t_inv_s"],
                            "t_rec_s":    a["t_rec_s"],
                            "metrics":    a["metrics"],
                            "sigmas_inv": a["sigmas_inv"],
                        }
                        for a in attempts
                    ],
                    "gate":           {
                        "passed":          gate_passed,
                        "latent_rel_max":  latent_rel_max,
                        "latent_cos_min":  latent_cos_min,
                        "achieved_rel":    attempt["metrics"]["latent"]["relative"],
                        "achieved_cos":    attempt["metrics"]["latent"]["cosine"],
                        "rel_ok":          gate_info["rel_ok"],
                        "cos_ok":          gate_info["cos_ok"],
                    },
                    "retry_executed": retry_executed,
                    "lpips_threshold_reported": lpips_threshold,
                    "checkpoint_sigmas_targets": ckpt_sigmas,
                    "checkpoint_sigmas_actual":  list(attempt["checkpoints"].keys()),
                }
                with (sample_dir / "inv_meta.yaml").open("w") as f:
                    yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

                m = attempt["metrics"]
                summary.append({
                    "sample_id":      sample_id,
                    "difficulty":     sample.get("difficulty"),
                    "gate_passed":    gate_passed,
                    "num_steps_used": attempt["num_steps"],
                    "retry_executed": retry_executed,
                    # Latent (primary gate)
                    "latent_rel":     m["latent"]["relative"],
                    "latent_cos":     m["latent"]["cosine"],
                    "latent_l2_per_element": m["latent"]["l2_per_element"],
                    # Decoded-space (reported)
                    "psnr_mean":      m["psnr"]["mean"],
                    "psnr_min":       m["psnr"]["min"],
                    "ssim_mean":      m["ssim"]["mean"],
                    "ssim_min":       m["ssim"]["min"],
                    "lpips_mean":     m["lpips"]["mean"],
                    "lpips_max":      m["lpips"]["max"],
                    "temporal_mean":  m["temporal"]["mean"],
                    "temporal_max":   m["temporal"]["max"],
                })

                # Free large per-attempt tensors before the next sample.
                del attempts, attempt
                torch.cuda.empty_cache()

            # ── Run-level artefacts ──────────────────────────────────────────────
            with (run_dir / "config_snapshot.yaml").open("w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            with (run_dir / "summary.yaml").open("w") as f:
                yaml.safe_dump({"run_id": run_id, "samples": summary}, f, sort_keys=False, allow_unicode=True)

            passed = sum(1 for s in summary if s["gate_passed"])
            log.info("─" * 70)
            log.info(
                "RF inversion summary: %d/%d samples passed gate (rel<%.3f & cos>%.4f)",
                passed, len(summary), latent_rel_max, latent_cos_min,
            )
            log.info(
                "%-4s  %-45s  %-7s  %-8s  %-8s  %-8s  %-8s  %s",
                "tag", "sample_id", "rel", "cos", "psnr", "ssim", "lpips", "steps",
            )
            for s in summary:
                tag = "PASS" if s["gate_passed"] else "FAIL"
                log.info(
                    "  [%s] %-45s  %.4f  %.5f  %5.2fdB  %.4f  %.4f  %d%s",
                    tag, s["sample_id"],
                    s["latent_rel"], s["latent_cos"],
                    s["psnr_mean"], s["ssim_mean"], s["lpips_mean"],
                    s["num_steps_used"], "  (retry)" if s["retry_executed"] else "",
                )
            log.info("─" * 70)


    if __name__ == "__main__":
        main()
