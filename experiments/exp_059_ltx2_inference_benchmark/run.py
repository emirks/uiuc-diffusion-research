"""exp_059 — LTX-2 inference benchmark: granular wall-clock timing on one GPU.

One process = one arm (fresh CUDA context). Call 1 is COLD (first read of the
checkpoints from the project FS); calls 2..N are WARM (OS page cache serves the
files; loads reduce to host-read + H2D + build). NOTE: StateDictRegistry was
tried and rejected -- it caches state dicts ON THE GPU (loader passes the CUDA
device), and LoRA fusion then runs out-of-place to protect the cached copy:
39 GB cached + 39 GB fused = OOM on 80 GB at the dev pipeline's stage-2 build.
DummyRegistry (stock CLI behavior) fuses in-place and fits.

Granular sections per call, in execution order:
  prompt_encode            Gemma build + encode + free (both prompts)
  image_conditioner        conditioning encoder scope (no images here -> ~0)
  transformer_build        state-dict load + module build + H2D   (per stage)
  denoise                  the actual diffusion loop              (per stage)
  transformer_free         weights -> meta device                 (per stage)
  upsample                 x2 spatial upsampler (build + forward + free)
  audio_decode             audio VAE + vocoder
  vae_decode               video VAE decode (accumulated inside the iterator)
  mux                      encode_video total minus vae_decode (H.264 + I/O)

Timing uses time.perf_counter() with torch.cuda.synchronize() fences.
Results: <run_dir>/<arm>/timings.json + one mp4 per call + run.log.
"""

import argparse
import json
import logging
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.loader.registry import StateDictRegistry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
from ltx_pipelines.utils.media_io import encode_video

log = logging.getLogger("exp_059")

# ---------------------------------------------------------------------------
# Section recorder — a global ordered list, reset per generation call.
# Stage identity is recoverable from order + the recorded step count/res.
# ---------------------------------------------------------------------------
SECTIONS: list[dict] = []


def _sync() -> None:
    torch.cuda.synchronize()


def _add(name: str, seconds: float, **extra) -> None:
    SECTIONS.append({"name": name, "seconds": round(seconds, 3), **extra})


def _wrap_simple(cls, method_name: str, section: str):
    orig = getattr(cls, method_name)

    def timed(self, *args, **kwargs):
        _sync()
        t0 = time.perf_counter()
        out = orig(self, *args, **kwargs)
        _sync()
        _add(section, time.perf_counter() - t0)
        return out

    setattr(cls, method_name, timed)


def install_patches() -> None:
    _wrap_simple(PromptEncoder, "__call__", "prompt_encode")
    _wrap_simple(ImageConditioner, "__call__", "image_conditioner")
    _wrap_simple(VideoUpsampler, "__call__", "upsample")
    _wrap_simple(AudioDecoder, "__call__", "audio_decode")
    _wrap_simple(VideoDecoder, "__call__", "video_decode_init")

    # DiffusionStage: split transformer build / denoise loop / free.
    orig_ctx = DiffusionStage._transformer_ctx
    orig_run = DiffusionStage.run

    @contextmanager
    def timed_ctx(self, **kwargs):
        _sync()
        t0 = time.perf_counter()
        with orig_ctx(self, **kwargs) as transformer:
            _sync()
            _add("transformer_build", time.perf_counter() - t0)
            yield transformer
            _sync()
            t1 = time.perf_counter()
        _sync()
        _add("transformer_free", time.perf_counter() - t1)

    def timed_run(self, transformer, denoiser, sigmas, noiser, width, height, frames, fps, *args, **kwargs):
        _sync()
        t0 = time.perf_counter()
        out = orig_run(self, transformer, denoiser, sigmas, noiser, width, height, frames, fps, *args, **kwargs)
        _sync()
        _add("denoise", time.perf_counter() - t0, steps=len(sigmas) - 1, width=width, height=height)
        return out

    DiffusionStage._transformer_ctx = timed_ctx
    DiffusionStage.run = timed_run


def timed_video_iter(it):
    """Attribute per-chunk VAE decode time (inside next()) to 'vae_decode'."""
    total = 0.0
    while True:
        t0 = time.perf_counter()
        try:
            chunk = next(it)
        except StopIteration:
            break
        _sync()
        total += time.perf_counter() - t0
        yield chunk
    _add("vae_decode", total)


# ---------------------------------------------------------------------------


def gpu_meta() -> dict:
    smi = ""
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,pcie.link.gen.current",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return {
        "node": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "nvidia_smi": smi,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
    }


def build_pipeline(arm_cfg: dict, cfg: dict):
    m = cfg["model"]
    compile_cfg = None
    if arm_cfg.get("compile", "none") != "none":
        compile_cfg = CompilationConfig(mode=arm_cfg["compile"])
    # GPU-resident weights (state dicts cached on the CUDA device). Distilled-only:
    # the dev pipeline's stage-2 LoRA fusion goes out-of-place and doubles 39 GB.
    registry = StateDictRegistry() if arm_cfg.get("registry") else None
    if arm_cfg["pipeline"] == "dev":
        return TI2VidTwoStagesPipeline(
            checkpoint_path=m["dev_checkpoint"],
            distilled_lora=[
                LoraPathStrengthAndSDOps(
                    m["distilled_lora"], m["distilled_lora_strength"], LTXV_LORA_COMFY_RENAMING_MAP
                )
            ],
            spatial_upsampler_path=m["spatial_upsampler"],
            gemma_root=m["gemma_root"],
            loras=[],
            compilation_config=compile_cfg,
        )
    return DistilledPipeline(
        distilled_checkpoint_path=m["distilled_checkpoint"],
        gemma_root=m["gemma_root"],
        spatial_upsampler_path=m["spatial_upsampler"],
        loras=[],
        registry=registry,
        compilation_config=compile_cfg,
    )


def generate_once(pipeline, arm_cfg: dict, cfg: dict, seed: int, out_path: Path) -> dict:
    g = cfg["generation"]
    num_frames = arm_cfg.get("num_frames", g["num_frames"])
    frame_rate = g["frame_rate"]
    tiling = TilingConfig.default()
    chunks = get_video_chunks_number(num_frames, tiling)

    SECTIONS.clear()
    torch.cuda.reset_peak_memory_stats()
    _sync()
    t0 = time.perf_counter()

    if arm_cfg["pipeline"] == "dev":
        video, audio = pipeline(
            prompt=g["prompt"],
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            seed=seed,
            height=arm_cfg["height"],
            width=arm_cfg["width"],
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=arm_cfg.get("num_inference_steps", g["num_inference_steps"]),
            video_guider_params=MultiModalGuiderParams(**g["video_guider"]),
            audio_guider_params=MultiModalGuiderParams(**g["audio_guider"]),
            images=[],
            tiling_config=tiling,
            max_batch_size=g["max_batch_size"],
        )
    else:
        video, audio = pipeline(
            prompt=g["prompt"],
            seed=seed,
            height=arm_cfg["height"],
            width=arm_cfg["width"],
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[],
            tiling_config=tiling,
        )

    t_encode0 = time.perf_counter()
    encode_video(
        video=timed_video_iter(video),
        fps=frame_rate,
        audio=audio,
        output_path=str(out_path),
        video_chunks_number=chunks,
    )
    _sync()
    t_end = time.perf_counter()

    encode_total = t_end - t_encode0
    vae_decode = next((s["seconds"] for s in SECTIONS if s["name"] == "vae_decode"), 0.0)
    _add("mux", encode_total - vae_decode)

    return {
        "seed": seed,
        "total_s": round(t_end - t0, 3),
        "sections": list(SECTIONS),
        "peak_vram_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "output": out_path.name,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--run-dir", required=True, help="parent run dir; arm writes to <run-dir>/<arm>/")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    arm_cfg = cfg["arms"][args.arm]

    arm_dir = Path(args.run_dir) / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    result_path = arm_dir / "timings.json"
    if result_path.exists():
        print(f"[skip] {result_path} already exists")
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(arm_dir / "run.log")],
    )
    log.info("arm=%s cfg=%s", args.arm, arm_cfg)

    install_patches()

    result = {
        "arm": args.arm,
        "arm_cfg": arm_cfg,
        "meta": gpu_meta(),
        "pipeline_init_s": None,
        "calls": [],
    }
    log.info("meta: %s", result["meta"])

    with torch.inference_mode():
        t0 = time.perf_counter()
        pipeline = build_pipeline(arm_cfg, cfg)
        result["pipeline_init_s"] = round(time.perf_counter() - t0, 3)

        g = cfg["generation"]
        warm_seeds = g["warm_seeds"][: arm_cfg.get("warm_repeats", len(g["warm_seeds"]))]
        plan = [("cold", g["cold_seed"])] + [("warm", s) for s in warm_seeds]
        for i, (kind, seed) in enumerate(plan):
            name = f"{kind}{i}_s{seed}_{arm_cfg['width']}x{arm_cfg['height']}.mp4"
            log.info("=== call %d/%d (%s, seed %d) ===", i + 1, len(plan), kind, seed)
            rec = generate_once(pipeline, arm_cfg, cfg, seed, arm_dir / name)
            rec["kind"] = kind
            result["calls"].append(rec)
            log.info("call done: total=%.1fs sections=%s", rec["total_s"],
                     [(s["name"], s["seconds"]) for s in rec["sections"]])
            # persist incrementally so a preempted job keeps partial data
            (arm_dir / "timings.partial.json").write_text(json.dumps(result, indent=2))

    result_path.write_text(json.dumps(result, indent=2))
    (arm_dir / "timings.partial.json").unlink(missing_ok=True)
    log.info("wrote %s", result_path)


if __name__ == "__main__":
    main()
