"""exp_021 — LTX-2 C2V — trajectory logging on top of exp_020's setup.

Records the complete Stage-1 denoising trajectory alongside the standard VC
generation (video + audio output is identical to exp_020).

What is logged per sample (Stage 1 only, 40 steps by default):

    z_t      [S, C, F', H', W']  bfloat16  VAE latent before each step
    v_pred   [S, C, F', H', W']  bfloat16  model velocity/noise prediction
    z_final  [C, F', H', W']     bfloat16  final clean latent z_0
    timesteps  List[float]                  denoising t value for each step
    hidden_states  Dict                     optional h^l at selected layers

    S = num_inference_steps (Stage 1)

Output layout::

    run_dir/
      <sample_id>/
        s{seed}_K{K}_steps{N}.mp4       final video (same as exp_020)
        trajectory_stage1.pt            trajectory dict (torch.save format)
        config_snapshot.yaml
      config_snapshot.yaml
      summary.yaml
      run.log

Loading the trajectory::

    data = torch.load("trajectory_stage1.pt", weights_only=False)
    z_t     = data["z_t"]       # [S, C, F', H', W']
    v_pred  = data["v_pred"]    # [S, C, F', H', W']
    z_final = data["z_final"]   # [C, F', H', W']  (= z_t[S-1] after the last step)
    ts      = data["timesteps"] # list of float, length S

    # Frame-level geometric quantities at denoising step i:
    #   delta_p_z = z_t[i, :, p+1] - z_t[i, :, p]  (video-time difference)
    #   curvature  = ||delta_p_z[p+1] - delta_p_z[p]||

Tip — accessing optional hidden states::

    hs = data["hidden_states"]   # dict {step_idx: {layer_idx: Tensor[N, D]}}
    # step_idx and layer_idx are integers; N = packed token count; D = hidden dim

See: Experiment E1 in research notes.

How to run::

    source /workspace/miniforge3/etc/profile.d/conda.sh
    conda activate /workspace/envs/diff
    cd /workspace/diffusion-research
    python experiments/exp_021_trajectory_logging/run.py
"""
from __future__ import annotations

import argparse
import glob
import logging
import pathlib
import sys
import time
from typing import Any

import torch
import torchvision.io as tio
import yaml
from PIL import Image

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2ConditionPipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

from diffusion.exp_utils import load_config, next_run_dir, TeeLogger

REPO_ROOT      = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = pathlib.Path(__file__).parent / "config.yaml"

LTX_TEMPORAL_SCALE = 8   # VAE causal temporal downscale factor
DEVICE             = "cuda:0"

log = logging.getLogger(__name__)


# ── Frame / clip helpers ──────────────────────────────────────────────────────

def load_frames_from_mp4(path: str | pathlib.Path, n: int, from_end: bool = False) -> list[Image.Image]:
    """Decode up to *n* RGB frames from an MP4 (first or last *n*)."""
    video, _, _ = tio.read_video(str(path), pts_unit="sec", output_format="THWC")
    if video.shape[0] == 0:
        raise ValueError(f"No frames decoded from {path}")
    frames = video[-n:] if from_end else video[:n]
    return [Image.fromarray(f.numpy()) for f in frames]


def load_frames_from_dir(path: str | pathlib.Path, n: int, from_end: bool = False) -> list[Image.Image]:
    """Load up to *n* RGB frames from a JPEG directory (first or last *n*)."""
    jpgs = sorted(glob.glob(str(pathlib.Path(path) / "*.jpg")))
    if not jpgs:
        raise FileNotFoundError(f"No .jpg files in {path}")
    jpgs = jpgs[-n:] if from_end else jpgs[:n]
    return [Image.open(p).convert("RGB") for p in jpgs]


def load_clip_frames(sample: dict, repo_root: pathlib.Path, n: int) -> tuple[list[Image.Image], list[Image.Image]]:
    """Dispatch frame loading by input format (MP4 or JPEG dir)."""
    if "start_clip" in sample:
        start = load_frames_from_mp4(repo_root / sample["start_clip"], n, from_end=False)
        end   = load_frames_from_mp4(repo_root / sample["end_clip"],   n, from_end=True)
    else:
        start = load_frames_from_dir(repo_root / sample["start_images"], n, from_end=False)
        end   = load_frames_from_dir(repo_root / sample["end_images"],   n, from_end=True)
    return start, end


def end_clip_index(num_frames: int, num_clip_frames: int) -> int:
    """Latent index where the end clip's first latent token should be placed.

    Aligns the clip's last latent with the output's last latent frame so
    the end conditioning covers the final temporal region of the video.
    index=-1 must NOT be used for multi-frame clips (trims to 1 latent).
    """
    n_lat = (num_frames      - 1) // LTX_TEMPORAL_SCALE + 1
    k_lat = (num_clip_frames - 1) // LTX_TEMPORAL_SCALE + 1
    return n_lat - k_lat


# ── Trajectory Logger ─────────────────────────────────────────────────────────

class TrajectoryLogger:
    """Patches the Stage-1 scheduler to record the full denoising trajectory.

    Attach once after model loading (call ``attach``).  Before each sample
    call ``reset``, after each sample call ``flush``.  Remove the patch when
    done by calling ``detach``.

    Implementation note
    -------------------
    The scheduler's ``step`` method is patched at the **class level**
    (``FlowMatchEulerDiscreteScheduler.step``) with an identity guard so that
    only calls made on the *exact* Stage-1 scheduler instance are intercepted.

    Why class-level and not instance-level
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``LTX2ConditionPipeline`` calls ``copy.deepcopy(self.scheduler)`` internally
    to create a separate audio scheduler.  If ``step`` is replaced as an
    *instance attribute*, the deepcopy inherits the patched function, and every
    ``audio_scheduler.step(...)`` call would invoke ``_original_step`` **bound
    to the original Stage-1 scheduler**, double-advancing its ``_step_index``.
    After 20 outer steps × 2 scheduler calls = 40 advances the scheduler runs
    out of sigmas and raises ``IndexError: index 41 is out of bounds``.

    A class-level patch receives ``sched_self`` explicitly; the deepcopied
    audio scheduler is a *different object* so ``sched_self is _target`` is
    ``False`` and it falls straight through to the original class method without
    any logging side-effects.

    The patch intercepts three tensors each step::

        model_output  — v_θ / ε_θ prediction (post-CFG, pre-scheduler update)
        sample        — z_t (noisy latent BEFORE the scheduler update)
        result        — z_{t-1} (noisy latent AFTER the update, stored only
                        for the last step as z_final = z_0)

    All tensors are moved to CPU and cast to bfloat16 immediately to bound
    GPU memory usage.
    """

    def __init__(self, traj_cfg: dict) -> None:
        self.enabled         = traj_cfg.get("enabled", True)
        self.save_hidden     = traj_cfg.get("save_hidden_states", False)
        self.layer_fracs: list[float] = traj_cfg.get(
            "hidden_state_config", {}
        ).get("layer_fractions", [0.25, 0.5, 0.75, 1.0])
        self.step_fracs: list[float] = traj_cfg.get(
            "hidden_state_config", {}
        ).get("step_fractions", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        self._original_step:     Any = None   # saved class-level step function
        self._target_scheduler:  Any = None   # Stage-1 scheduler identity guard
        self._hooks: list = []

        # per-sample state
        self._total_steps: int = 0
        self._step_count:  int = 0
        self._timesteps:   list[float]        = []
        self._z_t_list:    list[torch.Tensor] = []
        self._v_pred_list: list[torch.Tensor] = []
        self._z_final:     torch.Tensor | None = None
        self._hidden:      dict[int, dict[int, torch.Tensor]] = {}

    # ── setup / teardown ─────────────────────────────────────────────────────

    def attach(self, scheduler: Any, transformer: Any | None = None) -> None:
        """Patch ``FlowMatchEulerDiscreteScheduler.step`` at the class level.

        An identity guard (``sched_self is self._target_scheduler``) ensures
        that only calls on the *exact* Stage-1 scheduler instance are logged.
        Deepcopied audio schedulers and the Stage-2 scheduler are unaffected.

        Call once after model loading.  ``transformer`` is only needed when
        ``save_hidden_states`` is True.
        """
        if not self.enabled:
            return

        self._target_scheduler = scheduler
        # Save the current class-level step function (unbound in Python 3).
        self._original_step    = FlowMatchEulerDiscreteScheduler.step

        logger = self   # closure reference

        def _class_patched_step(
            sched_self:   Any,
            model_output: torch.Tensor,
            timestep:     torch.Tensor | float,
            sample:       torch.Tensor,
            **kwargs: Any,
        ) -> Any:
            should_log = sched_self is logger._target_scheduler
            if should_log:
                t_val = timestep.item() if torch.is_tensor(timestep) else float(timestep)
                logger._timesteps.append(t_val)
                logger._z_t_list.append(sample.detach().cpu().to(torch.bfloat16))
                logger._v_pred_list.append(model_output.detach().cpu().to(torch.bfloat16))

            # Call the original unbound function with the correct scheduler instance.
            result = logger._original_step(sched_self, model_output, timestep, sample, **kwargs)

            if should_log:
                # Extract z_{t-1} (handles both tuple and dataclass returns).
                if isinstance(result, tuple):
                    z_next = result[0]
                elif hasattr(result, "prev_sample"):
                    z_next = result.prev_sample
                else:
                    z_next = result
                logger._z_final = z_next.detach().cpu().to(torch.bfloat16)
                logger._step_count += 1

            return result

        FlowMatchEulerDiscreteScheduler.step = _class_patched_step
        log.info(
            "TrajectoryLogger: patched FlowMatchEulerDiscreteScheduler.step "
            "(class-level, target id=%d)",
            id(scheduler),
        )

        if self.save_hidden and transformer is not None:
            self._register_hidden_hooks(transformer)

    def _register_hidden_hooks(self, transformer: Any) -> None:
        """Register forward hooks on selected transformer blocks."""
        try:
            blocks = transformer.transformer_blocks
        except AttributeError:
            log.warning(
                "TrajectoryLogger: transformer.transformer_blocks not found; "
                "hidden-state logging disabled."
            )
            return

        n = len(blocks)
        layer_indices = sorted({
            max(0, min(n - 1, round(f * (n - 1))))
            for f in self.layer_fracs
        })
        log.info(
            "TrajectoryLogger: registering hidden-state hooks at layers %s "
            "(transformer depth=%d)",
            layer_indices, n,
        )

        for lidx in layer_indices:
            hook = blocks[lidx].register_forward_hook(
                self._make_hidden_hook(lidx)
            )
            self._hooks.append(hook)

    def _make_hidden_hook(self, layer_idx: int):
        logger = self

        def _hook(module: Any, inp: Any, out: Any) -> None:
            if logger._total_steps == 0:
                return
            target_steps = {
                round(f * max(logger._total_steps - 1, 1))
                for f in logger.step_fracs
            }
            if logger._step_count not in target_steps:
                return
            h = out[0] if isinstance(out, tuple) else out
            step_bucket = logger._hidden.setdefault(logger._step_count, {})
            step_bucket[layer_idx] = h.detach().cpu().to(torch.bfloat16)

        return _hook

    def detach(self) -> None:
        """Restore the original class-level step and remove transformer hooks."""
        if not self.enabled:
            return
        if self._original_step is not None:
            FlowMatchEulerDiscreteScheduler.step = self._original_step
            self._original_step    = None
            self._target_scheduler = None
            log.info("TrajectoryLogger: FlowMatchEulerDiscreteScheduler.step restored.")
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── per-sample lifecycle ──────────────────────────────────────────────────

    def reset(self, total_steps: int) -> None:
        """Clear per-sample state.  Call before each sample's Stage-1 call."""
        self._total_steps  = total_steps
        self._step_count   = 0
        self._timesteps    = []
        self._z_t_list     = []
        self._v_pred_list  = []
        self._z_final      = None
        self._hidden       = {}

    def flush(self, sample_id: str, out_dir: pathlib.Path) -> pathlib.Path | None:
        """Stack collected tensors and save to *out_dir/<sample_id>_trajectory_stage1.pt*.

        Returns the path written, or None if logging is disabled.
        """
        if not self.enabled:
            return None

        if not self._z_t_list:
            log.warning("TrajectoryLogger.flush: no steps recorded for %s", sample_id)
            return None

        z_t    = torch.stack(self._z_t_list)    # [S, C, F', H', W']
        v_pred = torch.stack(self._v_pred_list)  # [S, C, F', H', W']

        data = {
            "sample_id":    sample_id,
            "timesteps":    self._timesteps,        # List[float], len S
            "z_t":          z_t,                    # [S, C, F', H', W'] bfloat16
            "v_pred":       v_pred,                 # [S, C, F', H', W'] bfloat16
            "z_final":      self._z_final,          # [C, F', H', W'] bfloat16
            "hidden_states": self._hidden,          # {step_idx: {layer_idx: [N,D]}}
            "meta": {
                "num_steps":    self._step_count,
                "latent_shape": tuple(z_t.shape[1:]),   # (C, F', H', W')
                "has_hidden":   bool(self._hidden),
                "hidden_steps": sorted(self._hidden.keys()),
                "hidden_layers": (
                    sorted(next(iter(self._hidden.values())).keys())
                    if self._hidden else []
                ),
            },
        }

        out_path = out_dir / f"{sample_id}_trajectory_stage1.pt"
        torch.save(data, out_path)

        size_mb = out_path.stat().st_size / 1e6
        log.info(
            "Trajectory saved: %s  (%.1f MB, %d steps, shape=%s%s)",
            out_path.name, size_mb, self._step_count,
            tuple(z_t.shape),
            f", hidden@{sorted(self._hidden.keys())}" if self._hidden else "",
        )
        return out_path


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

        model_id        = cfg["model"]["model_id"]
        lora_strength   = cfg["model"]["distilled_lora_strength"]
        num_frames      = cfg["inference"]["num_frames"]
        frame_rate      = float(cfg["inference"]["frame_rate"])
        height          = cfg["inference"]["height"]
        width           = cfg["inference"]["width"]
        num_steps       = cfg["inference"]["num_inference_steps"]
        guidance_scale  = cfg["inference"]["guidance_scale"]
        seed            = cfg["runtime"]["seed"]
        num_clip_frames = cfg["inputs"]["num_clip_frames"]
        start_strength  = cfg["inputs"]["start_clip_strength"]
        end_strength    = cfg["inputs"]["end_clip_strength"]
        negative_prompt = cfg["inputs"]["negative_prompt"].strip()
        cfg_traj        = cfg.get("trajectory", {"enabled": True})

        end_idx = end_clip_index(num_frames, num_clip_frames)
        log.info(
            "Clip index: start=0  end=%d  (num_frames=%d → %d lat, clip=%d px → %d lat)",
            end_idx, num_frames,
            (num_frames - 1) // LTX_TEMPORAL_SCALE + 1,
            num_clip_frames,
            (num_clip_frames - 1) // LTX_TEMPORAL_SCALE + 1,
        )

        # ── Condition Pipeline Generation (ltx2.md): load → offload → tiling ──
        # https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/ltx2.md#condition-pipeline-generation
        # Doc uses enable_sequential_cpu_offload; we use enable_model_cpu_offload (whole components).
        log.info("Loading LTX2ConditionPipeline from %s …", model_id)
        t0   = time.perf_counter()
        pipe = LTX2ConditionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        # Whole-component offload (see DiffusionPipeline.enable_model_cpu_offload docstring).
        pipe.enable_model_cpu_offload(device=DEVICE)
        pipe.vae.enable_tiling()
        log.info("Pipeline loaded in %.1fs.", time.perf_counter() - t0)

        # Keep the default stage-1 scheduler so it can be restored each sample.
        stage1_scheduler = pipe.scheduler

        # ── Two-stages Generation (same doc page): LoRA + Stage-2 scheduler ──
        # https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/ltx2.md#two-stages-generation
        # FLF2V Condition example uses a fully-distilled checkpoint without extra LoRA; for
        # Lightricks/LTX-2 we attach Stage-2 distilled LoRA here (T2V snippet weight_name).
        log.info("Loading distilled LoRA (Stage 2) …")
        pipe.load_lora_weights(
            model_id,
            adapter_name="stage_2_distilled",
            weight_name="ltx-2-19b-distilled-lora-384.safetensors",
        )
        pipe.set_adapters("stage_2_distilled", lora_strength)

        stage2_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
        )

        # Spatial upsampler (×2 each spatial dim, latent → latent) — matches T2V two-stage snippet.
        log.info("Loading spatial upsampler …")
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            model_id, subfolder="latent_upsampler", torch_dtype=torch.bfloat16
        )
        upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
        upsample_pipe.enable_model_cpu_offload(device=DEVICE)

        # ── Attach trajectory logger to Stage-1 scheduler ────────────────────
        # stage1_scheduler is the only scheduler that runs all 40 steps.
        # stage2_scheduler is a separate object; its .step is never patched.
        logger = TrajectoryLogger(cfg_traj)
        logger.attach(
            scheduler=stage1_scheduler,
            transformer=pipe.transformer if cfg_traj.get("save_hidden_states", False) else None,
        )

        # ── Per-sample loop ───────────────────────────────────────────────────
        summary: list[dict] = []

        for idx, sample in enumerate(cfg["samples"]):
            sample_id  = sample["sample_id"]
            prompt     = sample["prompt"].strip()
            sample_dir = run_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            start_src = sample.get("start_clip") or sample.get("start_images", "")
            end_src   = sample.get("end_clip")   or sample.get("end_images",   "")
            log.info("─── Sample %d/%d  id=%s ───", idx + 1, len(cfg["samples"]), sample_id)
            print(f"[info] start  : {start_src}")
            print(f"[info] end    : {end_src}")
            print(f"[info] prompt : {prompt[:80]}…")

            start_frames, end_frames = load_clip_frames(sample, REPO_ROOT, num_clip_frames)

            # LTX2VideoCondition — the core diffusers API for visual conditioning.
            # frames: list[PIL.Image] → encoded by the pipeline VAE internally.
            # index:  latent frame index where this clip's first token is placed.
            # strength=1.0: fully clean anchor (no denoising applied to this region).
            conditions = [
                LTX2VideoCondition(frames=start_frames, index=0,       strength=start_strength),
                LTX2VideoCondition(frames=end_frames,   index=end_idx, strength=end_strength),
            ]

            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            t_infer   = time.perf_counter()

            # ── Stage 1 ───────────────────────────────────────────────────────
            pipe.scheduler = stage1_scheduler
            pipe.disable_lora()
            log.info("Stage 1: %dx%d  %d steps  guidance=%.1f", height, width, num_steps, guidance_scale)

            # Reset logger BEFORE the call so step counter and lists are clean.
            logger.reset(total_steps=num_steps)

            video_latent, audio_latent = pipe(
                conditions=conditions,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_steps,
                sigmas=None,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="latent",
                return_dict=False,
            )
            log.info("Stage 1 done in %.1fs.", time.perf_counter() - t_infer)

            # ── Save trajectory immediately after Stage 1 ─────────────────────
            # Flush before Stage 2 (different scheduler, different tensor shapes).
            traj_path = logger.flush(sample_id=sample_id, out_dir=sample_dir)

            # ── Upsample ×2 spatial ───────────────────────────────────────────
            t_up = time.perf_counter()
            upscaled_latent = upsample_pipe(
                latents=video_latent, output_type="latent", return_dict=False
            )[0]
            log.info("Upsample done in %.1fs.", time.perf_counter() - t_up)

            # ── Stage 2: distilled LoRA, 3 steps ─────────────────────────────
            # Condition Pipeline FLF2V + Two-stages T2V: noise_scale from T2V (renoise); no negative_prompt
            # at guidance_scale=1.0 (CFG disabled — see pipeline do_classifier_free_guidance).
            pipe.scheduler = stage2_scheduler
            pipe.enable_lora()
            log.info("Stage 2: 3 steps  guidance=1.0  %dx%d (distilled LoRA)", width * 2, height * 2)
            t_s2 = time.perf_counter()

            video, audio = pipe(
                latents=upscaled_latent,
                audio_latents=audio_latent,
                prompt=prompt,
                width=width * 2,
                height=height * 2,
                num_frames=num_frames,  # doc examples omit (default 121); explicit if config changes
                num_inference_steps=3,
                noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                generator=generator,
                guidance_scale=1.0,
                output_type="np",
                return_dict=False,
            )
            elapsed = time.perf_counter() - t_infer
            log.info("Stage 2 done in %.1fs.  Total: %.1fs.", time.perf_counter() - t_s2, elapsed)

            # ── Save video ────────────────────────────────────────────────────
            video_path = sample_dir / f"s{seed}_K{num_clip_frames}_steps{num_steps}.mp4"
            encode_video(
                video[0],
                fps=int(frame_rate),
                audio=audio[0].float().cpu(),
                audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
                output_path=str(video_path),
            )
            log.info("Saved %s", video_path)

            # ── Config snapshot ───────────────────────────────────────────────
            with (sample_dir / "config_snapshot.yaml").open("w") as f:
                yaml.safe_dump({
                    "sample_id":  sample_id,
                    "prompt":     prompt,
                    "start_src":  start_src,
                    "end_src":    end_src,
                    "clip_conditioning": {
                        "num_clip_frames": num_clip_frames,
                        "start_index":     0,
                        "end_index":       end_idx,
                        "start_strength":  start_strength,
                        "end_strength":    end_strength,
                    },
                    "inference":   cfg["inference"],
                    "runtime":     cfg["runtime"],
                    "output":      str(video_path),
                    "trajectory":  str(traj_path) if traj_path else None,
                    "elapsed_s":   round(elapsed, 1),
                }, f, sort_keys=False, allow_unicode=True)

            summary.append({
                "sample_id":  sample_id,
                "video":      str(video_path),
                "trajectory": str(traj_path) if traj_path else None,
                "elapsed_s":  round(elapsed, 1),
            })

        # ── Cleanup ───────────────────────────────────────────────────────────
        logger.detach()

        # ── Run-level artefacts ───────────────────────────────────────────────
        with (run_dir / "config_snapshot.yaml").open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        with (run_dir / "summary.yaml").open("w") as f:
            yaml.safe_dump({"run_id": run_id, "samples": summary}, f, sort_keys=False, allow_unicode=True)

        total = sum(s["elapsed_s"] for s in summary)
        log.info("All %d samples done.  Total inference: %.1fs", len(summary), total)
        for s in summary:
            traj_note = f"  traj→ {pathlib.Path(s['trajectory']).name}" if s.get("trajectory") else ""
            print(f"[done] {s['sample_id']}  →  {s['video']}{traj_note}")


if __name__ == "__main__":
    main()
