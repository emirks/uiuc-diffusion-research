"""exp_041 — Feature-injection experiment (MasaCtrl-style self-attn K,V).

Reads a recon-pass feature cache produced by exp_040 and injects the source
self-attention K,V into the free-middle tokens of a re-noised reconstruction.

Hypothesis (one direction):
    Source self-attn K,V (mid layers, descending recon steps), injected into
    the free middle of a reconstruction whose free-middle noise was reseeded,
    transport the source transition — the output moves toward the source
    recon despite the different free-middle noise.

Why this design (see README.md for the extended reasoning):
  * MasaCtrl keeps the edit pass's Q (spatial addressing) and replaces K,V
    (content per position). We inject attn1_k / attn1_v only — not block_out
    (which sweeps the whole residual) and not Q (which forces the attention
    pattern). RF-Edit injects V only; MasaCtrl K,V; consensus is K,V.
  * Spatial scope = FREE MIDDLE only (latent frames 4..11). The C2V anchors
    (start frames 0..3, end frames 12..15 minus the drop1 token) are
    hard-pinned by the conditioning mask + clamp every step.
  * Edit-step k ↔ cached-step k (both descending recon, no σ flip). Inject on
    the predictor substep only (the σ_curr call we cached); corrector runs free.

CFG convention (RF-Edit): the source cache is at CFG=1; the edit pass may run
at production CFG (>1). At CFG>1 every transformer call is batched
[uncond, cond] and the injector (cfg_batch=True) writes the cached tensor into
both rows. At CFG>1 the reference (a CFG=1 recon) is itself off the edit
trajectory, so the headline stays C - B on the free middle (injection is the
only difference between B and C).

Config (single-sample, single-variant — legacy):
    source.cache_run_dir  → a sample dir holding z1.pt, z0_recon.pt, feature_cache/
    source.sample_id
    inference.guidance_scale

Config (multi-sample, multi-variant — dense runs):
    source.cache_run_dir  → the exp_040 RUN dir (parent of per-sample dirs)
    source.samples        → [sample_id, ...]
    inference.variants    → [{name, guidance_scale, prompt?}, ...]

Passes per (sample, variant):
    reference  = z0_recon.pt        (source recon)
    B  perturbed baseline           reconstruct(z1_pert)            no injection
    C  perturbed + injection        reconstruct(z1_pert) + src K,V injected
    D  self-injection (null)        reconstruct(z1)      + src K,V injected

Outputs (run_dir/{sample}/{variant}/): reference_recon.mp4,
    perturbed_baseline.mp4, perturbed_inject.mp4, self_inject.mp4, metrics.yaml.
    Plus run_dir/summary.yaml, config_snapshot.yaml, run.log.

How to run:
    source /workspace/cache/pod_init.sh
    conda activate /workspace/envs/diff
    cd /workspace/diffusion-research
    python experiments/exp_041_ltx2_feature_inject/run.py --config <config>.yaml
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import pathlib
import sys
import time

import numpy as np
import torch
import yaml
from PIL import Image

from diffusers.pipelines.ltx2 import LTX2ConditionPipeline

from diffusion.exp_utils import load_config, next_run_dir, TeeLogger
from diffusion.feature_inject import FeatureInjector

REPO_ROOT      = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = pathlib.Path(__file__).parent / "config.yaml"
DEVICE         = "cuda:0"

log = logging.getLogger(__name__)


# ── Borrow the frozen RFInverter + helpers from exp_040 (same machinery) ──────
def _load_exp040_module():
    path = REPO_ROOT / "experiments/exp_040_ltx2_feature_cache/run.py"
    spec = importlib.util.spec_from_file_location("exp040_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Midpoint reconstruction with optional injection ───────────────────────────
@torch.inference_mode()
def reconstruct_with_injection(
    inverter,
    z_start: torch.Tensor,
    num_steps: int,
    scheduler,
    *,
    injector: FeatureInjector | None = None,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Midpoint 2nd-order reconstruction, mirroring exp_040's `_midpoint_step`,
    with optional K,V injection on the predictor substep.

    `guidance_scale` follows the RF-Edit convention: the source cache is taken
    at CFG=1, but the edit pass may run at production CFG (>1). At CFG>1 every
    transformer call is batched [uncond, cond]; the injector (cfg_batch=True)
    writes the single cached tensor into both rows. The corrector substep
    always runs free (we only cached/inject the predictor).
    """
    sigmas_gen = inverter._build_sigma_grid(num_steps, scheduler)
    mask = inverter.conditioning_mask
    z = z_start.clone()
    z = z * (1 - mask) + inverter.clean_latents * mask

    for i in range(len(sigmas_gen) - 1):
        sigma_curr = float(sigmas_gen[i])
        sigma_next = float(sigmas_gen[i + 1])
        dtau = sigma_next - sigma_curr
        sigma_mid = sigma_curr + dtau / 2.0

        if injector is not None:
            injector.set_step(i)

        # Predictor (σ_curr) — injection fires here if this step was cached.
        if injector is not None:
            injector.set_substep("predictor")
        v_raw = inverter._call_transformer(z, sigma_curr, guidance_scale=guidance_scale)
        v, _ = inverter._x0_clamp_velocity(z, v_raw, sigma_curr)
        z_mid = z + (dtau / 2.0) * v

        # Corrector (σ_mid) — runs free (no cached features here).
        if injector is not None:
            injector.set_substep("corrector")
        v_mid_raw = inverter._call_transformer(z_mid, sigma_mid, guidance_scale=guidance_scale)
        v_mid, _ = inverter._x0_clamp_velocity(z_mid, v_mid_raw, sigma_mid)

        z_next = z + dtau * v_mid
        z_next = z_next * (1 - mask) + inverter.clean_latents * mask
        z = z_next.to(z.dtype)
    return z


def _frame_slice_for_latent_frames(latent_frames: list[int]) -> tuple[int, int]:
    """Pixel-frame [start, end) covered by a contiguous run of latent frames.

    Causal VAE: latent f>=1 covers pixels (f-1)*8+1 .. f*8.
    For latent frames [a..b] (a>=1): pixels [(a-1)*8+1 .. b*8 + 1).
    """
    a, b = min(latent_frames), max(latent_frames)
    px_start = (a - 1) * 8 + 1
    px_end   = b * 8 + 1
    return px_start, px_end


# ── Sample / variant resolution ───────────────────────────────────────────────
def _resolve_samples(cfg) -> list[tuple[str, pathlib.Path]]:
    """Return [(sample_id, cache_sample_dir), ...].

    Multi-sample: source.cache_run_dir is the exp_040 RUN dir; source.samples
    lists sample_ids (each a subdir). Legacy single-sample: source.cache_run_dir
    is itself the sample dir and source.sample_id names it.
    """
    src = cfg["source"]
    if src.get("samples"):
        base = REPO_ROOT / src["cache_run_dir"]
        return [(sid, base / sid) for sid in src["samples"]]
    cdir = REPO_ROOT / src["cache_run_dir"]
    return [(src["sample_id"], cdir)]


def _resolve_variants(cfg) -> list[dict]:
    """Return [{name, guidance_scale, prompt?}, ...]. Legacy: a single variant
    built from inference.guidance_scale."""
    inf = cfg["inference"]
    if inf.get("variants"):
        return list(inf["variants"])
    gs = float(inf["guidance_scale"])
    return [{"name": f"cfg{gs:g}".replace(".", "p"), "guidance_scale": gs}]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = REPO_ROOT / cfg["outputs"]["dir"]
    run_id, run_dir = next_run_dir(out_dir)

    with TeeLogger(run_dir / "run.log"):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
            datefmt="%H:%M:%S", stream=sys.stdout, force=True,
        )
        print(f"[info] run_dir : {run_dir}")

        exp040 = _load_exp040_module()
        RFInverter             = exp040.RFInverter
        unpack_and_denormalize = exp040.unpack_and_denormalize
        decode_latents_to_video = exp040.decode_latents_to_video
        MetricSuite            = exp040.MetricSuite

        sample_specs = _resolve_samples(cfg)
        variants     = _resolve_variants(cfg)
        n_steps      = int(cfg["inference"]["num_inference_steps"])
        inj_cfg      = cfg["injection"]
        pseed        = int(cfg["perturb"]["seed"])
        fm_frames    = list(cfg["perturb"]["free_middle_latent_frames"])
        log.info("Samples: %s", [s for s, _ in sample_specs])
        log.info("Variants: %s", [v["name"] for v in variants])

        # ── Load pipeline once ────────────────────────────────────────────────
        log.info("Loading LTX2ConditionPipeline …")
        t0 = time.perf_counter()
        pipe = LTX2ConditionPipeline.from_pretrained(cfg["model"]["model_id"], torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload(device=DEVICE)
        pipe.vae.enable_tiling()
        stage1_scheduler = pipe.scheduler
        log.info("Pipeline loaded in %.1fs.", time.perf_counter() - t0)

        metric_suite = MetricSuite(device=DEVICE)
        inverter = RFInverter(pipe, device=DEVICE, feature_cache=None)
        summary: list[dict] = []

        from diffusers.utils import export_to_video

        for sample_id, cache_sample_dir in sample_specs:
            cache_dir = cache_sample_dir / "feature_cache"
            static = torch.load(cache_dir / "static.pt", weights_only=False)
            z1       = torch.load(cache_sample_dir / "z1.pt",       weights_only=False).to(DEVICE)
            z0_recon = torch.load(cache_sample_dir / "z0_recon.pt", weights_only=False).to(DEVICE)

            latent_num_frames = int(static["latent_num_frames"])
            latent_height     = int(static["latent_height"])
            latent_width      = int(static["latent_width"])
            frame_rate        = float(static["frame_rate"])
            tpf               = latent_height * latent_width
            log.info("═══ sample %s  F'=%d H'=%d W'=%d  tpf=%d  N=%d ═══",
                     sample_id, latent_num_frames, latent_height, latent_width,
                     tpf, z1.shape[1])

            # Free-middle token ids for this sample's layout.
            fm_token_ids = torch.cat([
                torch.arange(f * tpf, (f + 1) * tpf) for f in fm_frames
            ]).to(DEVICE)

            # Perturb z1 free middle once per sample (seed fixed across variants).
            z1_pert = z1.clone()
            block = z1[:, fm_token_ids, :]
            g = torch.Generator(device=DEVICE).manual_seed(pseed)
            fresh = torch.randn(block.shape, generator=g, device=DEVICE, dtype=z1.dtype)
            if cfg["perturb"].get("match_rms", True):
                cur_rms = block.float().pow(2).mean().sqrt()
                new_rms = fresh.float().pow(2).mean().sqrt().clamp(min=1e-8)
                fresh = (fresh.float() * (cur_rms / new_rms)).to(z1.dtype)
            z1_pert[:, fm_token_ids, :] = fresh
            log.info("Perturbed free middle (seed=%d): RMS %.4f → %.4f", pseed,
                     block.float().pow(2).mean().sqrt().item(),
                     z1_pert[:, fm_token_ids, :].float().pow(2).mean().sqrt().item())

            def _decode(z_packed: torch.Tensor) -> np.ndarray:
                z5d = unpack_and_denormalize(pipe, z_packed, latent_num_frames,
                                             latent_height, latent_width)
                return decode_latents_to_video(pipe, z5d)

            px_lo, px_hi = _frame_slice_for_latent_frames(fm_frames)

            # Decode reference ONCE per sample (was previously in variant loop).
            ref_video = _decode(z0_recon)
            hi = min(px_hi, ref_video.shape[0])

            # Noise-floor oracle diagnostic: decode (z0_recon + (z1_pert - z1)),
            # i.e. the source recon PLUS the noise difference. This is the
            # theoretical upper bound on C-vs-ref similarity that a velocity-
            # equality injection (block_out hard inject at strength=1) can reach.
            # If C-pass approaches oracle PSNR, injection IS doing its job and
            # the remaining gap is just preserved noise.
            try:
                z_oracle = (z0_recon + (z1_pert - z1)).to(z1.dtype)
                oracle_video = _decode(z_oracle)
                oracle_psnr_fm = metric_suite.psnr(ref_video[px_lo:hi], oracle_video[px_lo:hi])["mean"]
                oracle_ssim_fm = metric_suite.ssim(ref_video[px_lo:hi], oracle_video[px_lo:hi])["mean"]
                oracle_lpips_fm = metric_suite.lpips(ref_video[px_lo:hi], oracle_video[px_lo:hi])["mean"]
                log.info("[ORACLE noise-floor ceiling, %s] free-mid PSNR=%.2f SSIM=%.4f LPIPS=%.4f",
                         sample_id, oracle_psnr_fm, oracle_ssim_fm, oracle_lpips_fm)
                del oracle_video
            except Exception as e:
                log.warning("oracle decode failed: %s", e)

            # B-pass cache: B = reconstruct(z1_pert) without injector. It is
            # variant-independent on (sites, cond_only) and only depends on
            # (gscale, prompt, neg_prompt). Variants sharing this triple
            # reuse the cached zB / videoB tensors.
            b_cache: dict[tuple, tuple[torch.Tensor, np.ndarray]] = {}

            for variant in variants:
                vname  = variant["name"]
                gscale = float(variant["guidance_scale"])
                vprompt = variant.get("prompt", static["prompt"])
                if vprompt is None:
                    vprompt = static["prompt"]
                vneg = variant.get("negative_prompt", static["negative_prompt"])
                if vneg is None:
                    vneg = static["negative_prompt"]
                do_cfg = gscale > 1.0
                # Per-variant overrides (default to the top-level injection block).
                v_sites = list(variant.get("sites", inj_cfg["sites"]))
                v_steps = list(variant.get("inject_steps", inj_cfg["steps"]))
                v_layers = list(variant.get("inject_layers", inj_cfg["layers"]))
                v_cond_only = bool(variant.get("cond_only_at_cfg", False))
                vdir = run_dir / sample_id / vname
                vdir.mkdir(parents=True, exist_ok=True)
                log.info("── %s / %s  CFG=%.2f  sites=%s  layers=%d  steps=%d  cond_only=%s  prompt=%r neg=%r ──",
                         sample_id, vname, gscale, v_sites, len(v_layers), len(v_steps), v_cond_only,
                         (vprompt[:48] + "…") if vprompt else vprompt,
                         (vneg[:48] + "…") if vneg else vneg)

                # Prepare inverter for this sample + variant prompt.
                inverter.prepare_sample(
                    prompt=vprompt,
                    negative_prompt=vneg,
                    conditioning_mask=static["conditioning_mask"].to(DEVICE),
                    clean_latents=static["clean_latents"].to(DEVICE),
                    latent_num_frames=latent_num_frames,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    frame_rate=frame_rate,
                    audio_context=static["audio_context"].to(DEVICE),
                )

                injector = FeatureInjector(
                    cache_dir,
                    inject_layers=v_layers,
                    inject_steps=v_steps,
                    sites=v_sites,
                    token_ids=fm_token_ids,
                    phase=inj_cfg["phase"],
                    substep=inj_cfg["substep"],
                    strength=float(inj_cfg["strength"]),
                    cfg_batch=do_cfg,
                    cond_only_at_cfg=v_cond_only,
                    device=DEVICE,
                    log=log,
                )

                b_key = (gscale, vprompt, vneg)
                if b_key in b_cache:
                    log.info("[%s/%s] B: cache hit (reusing zB from prior variant)", sample_id, vname)
                    zB, videoB = b_cache[b_key]
                else:
                    log.info("[%s/%s] B: perturbed baseline (no inject)", sample_id, vname)
                    zB = reconstruct_with_injection(inverter, z1_pert, n_steps, stage1_scheduler,
                                                    injector=None, guidance_scale=gscale)
                    videoB = _decode(zB)
                    b_cache[b_key] = (zB, videoB)

                log.info("[%s/%s] C: perturbed + inject", sample_id, vname)
                injector.attach(pipe.transformer)
                zC = reconstruct_with_injection(inverter, z1_pert, n_steps, stage1_scheduler,
                                                injector=injector, guidance_scale=gscale)
                videoC = _decode(zC)

                log.info("[%s/%s] D: self-inject null", sample_id, vname)
                zD = reconstruct_with_injection(inverter, z1, n_steps, stage1_scheduler,
                                                injector=injector, guidance_scale=gscale)
                videoD = _decode(zD)
                injector.detach()

                def _metrics(name: str, vid: np.ndarray) -> dict:
                    full = {
                        "psnr":  metric_suite.psnr(ref_video, vid)["mean"],
                        "ssim":  metric_suite.ssim(ref_video, vid)["mean"],
                        "lpips": metric_suite.lpips(ref_video, vid)["mean"],
                    }
                    fm = {
                        "psnr":  metric_suite.psnr(ref_video[px_lo:hi], vid[px_lo:hi])["mean"],
                        "ssim":  metric_suite.ssim(ref_video[px_lo:hi], vid[px_lo:hi])["mean"],
                        "lpips": metric_suite.lpips(ref_video[px_lo:hi], vid[px_lo:hi])["mean"],
                    }
                    log.info("  [%s] full PSNR=%.2f SSIM=%.4f LPIPS=%.4f | free-mid PSNR=%.2f SSIM=%.4f LPIPS=%.4f",
                             name, full["psnr"], full["ssim"], full["lpips"],
                             fm["psnr"], fm["ssim"], fm["lpips"])
                    return {"full_clip": full, "free_middle": fm}

                metrics = {
                    "B_perturbed_baseline": _metrics("B", videoB),
                    "C_perturbed_inject":   _metrics("C", videoC),
                    "D_self_inject_null":   _metrics("D", videoD),
                }
                b_fm = metrics["B_perturbed_baseline"]["free_middle"]
                c_fm = metrics["C_perturbed_inject"]["free_middle"]
                delta = {
                    "psnr":  c_fm["psnr"]  - b_fm["psnr"],
                    "ssim":  c_fm["ssim"]  - b_fm["ssim"],
                    "lpips": c_fm["lpips"] - b_fm["lpips"],
                }
                log.info("  HEADLINE (free-mid C-B): ΔPSNR=%+.2f ΔSSIM=%+.4f ΔLPIPS=%+.4f",
                         delta["psnr"], delta["ssim"], delta["lpips"])

                # NB: export_to_video's np.ndarray branch does (frame * 255), so
                # passing uint8 numpy frames double-multiplies → mod-256 wrap that
                # looks like color inversion. Route through PIL like exp_040.
                for nm, vid in (("reference_recon.mp4", ref_video),
                                ("perturbed_baseline.mp4", videoB),
                                ("perturbed_inject.mp4", videoC),
                                ("self_inject.mp4", videoD)):
                    frames = [Image.fromarray(f) for f in vid]
                    export_to_video(frames, str(vdir / nm), fps=int(frame_rate))

                with (vdir / "metrics.yaml").open("w") as f:
                    yaml.safe_dump({
                        "run_id": run_id,
                        "sample_id": sample_id,
                        "variant": vname,
                        "guidance_scale": gscale,
                        "prompt": vprompt,
                        "injection": dict(inj_cfg),
                        "variant_overrides": {
                            "sites": v_sites,
                            "inject_layers": v_layers,
                            "inject_steps": v_steps,
                            "cond_only_at_cfg": v_cond_only,
                        },
                        "perturb": dict(cfg["perturb"]),
                        "free_middle_pixel_frames": [px_lo, hi],
                        "metrics_vs_reference": metrics,
                        "headline_free_middle_C_minus_B": delta,
                    }, f, sort_keys=False)

                summary.append({
                    "sample_id": sample_id, "variant": vname, "guidance_scale": gscale,
                    "delta_free_middle": delta,
                    "B_free_mid_psnr": b_fm["psnr"], "C_free_mid_psnr": c_fm["psnr"],
                    "D_full_psnr": metrics["D_self_inject_null"]["full_clip"]["psnr"],
                })

        with (run_dir / "summary.yaml").open("w") as f:
            yaml.safe_dump({"run_id": run_id, "results": summary}, f, sort_keys=False)
        with (run_dir / "config_snapshot.yaml").open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        log.info("[done] %s → %s  (%d sample×variant runs)", run_id, run_dir, len(summary))
        for r in summary:
            log.info("  %s/%s  C-B ΔPSNR=%+.2f", r["sample_id"], r["variant"],
                     r["delta_free_middle"]["psnr"])


if __name__ == "__main__":
    main()
