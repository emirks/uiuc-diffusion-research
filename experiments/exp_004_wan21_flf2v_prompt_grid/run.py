#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Invalid config: {path}")
    return cfg


def resolve_path(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (root / q)


def pick_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {name}")
    return mapping[name]


def center_crop_to_aspect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    src_ar = src_w / src_h
    tgt_ar = target_w / target_h

    if src_ar > tgt_ar:
        new_w = int(round(src_h * tgt_ar))
        left = max(0, (src_w - new_w) // 2)
        right = left + new_w
        box = (left, 0, right, src_h)
    else:
        new_h = int(round(src_w / tgt_ar))
        top = max(0, (src_h - new_h) // 2)
        bottom = top + new_h
        box = (0, top, src_w, bottom)
    return img.crop(box)


def preprocess_image_for_target(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    cropped = center_crop_to_aspect(img, target_w=target_w, target_h=target_h)
    return cropped.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)


def validate_resolution(target_w: int, target_h: int) -> None:
    if target_w % 16 != 0 or target_h % 16 != 0:
        down_w = target_w - (target_w % 16)
        up_w = down_w + 16
        down_h = target_h - (target_h % 16)
        up_h = down_h + 16
        raise SystemExit(
            f"Invalid resolution {target_w}x{target_h}: Wan requires width/height divisible by 16.\n"
            f"Suggested width: {down_w} or {up_w}\n"
            f"Suggested height: {down_h} or {up_h}"
        )


def next_run_dir(out_dir: Path) -> tuple[str, Path]:
    existing = []
    for p in out_dir.glob("run_*"):
        if not p.is_dir():
            continue
        try:
            existing.append(int(p.name.split("_", 1)[1]))
        except Exception:
            continue
    nxt = (max(existing) + 1) if existing else 1
    run_id = f"run_{nxt:04d}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def missing_modules(names: list[str]) -> list[str]:
    return [n for n in names if importlib.util.find_spec(n) is None]


def preflight_dependency_check() -> None:
    required = ["ftfy", "imageio", "imageio_ffmpeg"]
    missing = missing_modules(required)
    if missing:
        raise SystemExit(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + "\nInstall in your env with:\n"
            + "pip install "
            + " ".join(missing)
        )


def preflight_function_check() -> None:
    print("[preflight] checking preprocess + export_to_video...")

    img = Image.new("RGB", (1080, 1080), color=(10, 20, 30))
    out = preprocess_image_for_target(img, target_w=848, target_h=480)
    if out.size != (848, 480):
        raise SystemExit(f"Preprocess check failed: got {out.size}, expected (848, 480)")

    frames = []
    for i in range(4):
        arr = np.zeros((64, 96, 3), dtype=np.uint8)
        arr[..., 0] = 40 * i
        arr[..., 1] = 80
        arr[..., 2] = 160
        frames.append(arr)

    with tempfile.TemporaryDirectory(prefix="exp004_preflight_") as td:
        test_video = Path(td) / "preflight.mp4"
        export_to_video(frames, str(test_video), fps=8)
        if not test_video.exists() or test_video.stat().st_size == 0:
            raise SystemExit("export_to_video preflight failed: output missing or empty")

    print("[preflight] ok")


def discover_frame_pairs(frame_pairs_root: Path, num_pairs: int, explicit_ids: list[str]) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []

    if explicit_ids:
        selected_dirs = [frame_pairs_root / x for x in explicit_ids]
    else:
        selected_dirs = sorted([p for p in frame_pairs_root.iterdir() if p.is_dir()])[:num_pairs]

    for d in selected_dirs:
        first = d / "first.png"
        last = d / "last.png"
        if first.exists() and last.exists():
            pairs.append((d.name, first, last))

    if len(pairs) < num_pairs and not explicit_ids:
        raise SystemExit(
            f"Found only {len(pairs)} valid frame pairs under {frame_pairs_root}, expected at least {num_pairs}."
        )
    if explicit_ids and len(pairs) != len(explicit_ids):
        raise SystemExit("Some explicit frame_pair_ids are invalid (missing first.png/last.png).")

    return pairs


def main() -> None:
    preflight_dependency_check()
    preflight_function_check()

    root = repo_root()
    cfg = load_cfg(Path(__file__).with_name("config.yaml"))

    out_dir = resolve_path(root, cfg["outputs"]["root_dir"]) / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    target_h = int(cfg["inference"]["height"])
    target_w = int(cfg["inference"]["width"])
    validate_resolution(target_w=target_w, target_h=target_h)

    frame_pairs_root = resolve_path(root, cfg["inputs"]["frame_pairs_root"])
    num_pairs = int(cfg["inputs"].get("num_frame_pairs", 3))
    explicit_ids = list(cfg["inputs"].get("frame_pair_ids", []) or [])
    prompts = list(cfg["inputs"].get("prompts", []))
    if len(prompts) < 1:
        raise SystemExit("config.inputs.prompts must contain at least one prompt.")

    frame_pairs = discover_frame_pairs(frame_pairs_root, num_pairs=num_pairs, explicit_ids=explicit_ids)
    print(f"[info] frame_pairs_selected: {[x[0] for x in frame_pairs]}")
    print(f"[info] prompts_count: {len(prompts)}")
    print(f"[info] total_runs: {len(frame_pairs) * len(prompts)}")

    if bool(cfg["runtime"].get("dry_run", True)):
        print("[dry-run] config/data look valid.")
        return

    dtype = pick_dtype(cfg["model"]["torch_dtype"])
    requested_device = str(cfg["runtime"].get("device", "auto")).lower()
    cuda_available = torch.cuda.is_available()
    use_cuda = cuda_available and requested_device != "cpu"

    print("[runtime] environment")
    print(f"[runtime] torch: {torch.__version__}")
    print(f"[runtime] requested_device: {requested_device}")
    print(f"[runtime] cuda_available: {cuda_available}")
    print(f"[runtime] selected_device: {'cuda' if use_cuda else 'cpu'}")
    print(f"[runtime] dtype: {dtype}")
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[runtime] cuda_device_0: {gpu_name} ({total_gb:.1f} GiB)")

    print(f"[info] loading: {cfg['model']['repo_id']}")
    pipe = WanImageToVideoPipeline.from_pretrained(
        cfg["model"]["repo_id"],
        torch_dtype=dtype,
        use_safetensors=bool(cfg["model"].get("use_safetensors", True)),
    )

    if use_cuda:
        if bool(cfg["runtime"].get("enable_model_cpu_offload", True)):
            print("[runtime] pipeline mode: enable_model_cpu_offload=True")
            pipe.enable_model_cpu_offload()
        else:
            print("[runtime] pipeline mode: full model to cuda")
            pipe = pipe.to("cuda")
    else:
        print("[runtime] pipeline mode: cpu")
        pipe = pipe.to("cpu")

    negative_prompt = cfg["inputs"].get("negative_prompt")
    base_seed = int(cfg["runtime"]["seed"])
    fps = int(cfg["outputs"].get("fps", 16))

    run_counter = 0
    for pair_id, start_path, end_path in frame_pairs:
        start_img = Image.open(start_path).convert("RGB")
        end_img = Image.open(end_path).convert("RGB")

        print(f"[preprocess] pair={pair_id} target_size: {target_w}x{target_h}")
        print(f"[preprocess] start_original: {start_img.size[0]}x{start_img.size[1]}")
        print(f"[preprocess] end_original: {end_img.size[0]}x{end_img.size[1]}")
        start_img = preprocess_image_for_target(start_img, target_w=target_w, target_h=target_h)
        end_img = preprocess_image_for_target(end_img, target_w=target_w, target_h=target_h)

        for prompt_idx, prompt in enumerate(prompts):
            run_counter += 1
            run_id, run_dir = next_run_dir(out_dir)

            gen_device = "cuda" if use_cuda else "cpu"
            run_seed = base_seed + run_counter - 1
            generator = torch.Generator(device=gen_device).manual_seed(run_seed)

            print(f"[run] ({run_counter}/{len(frame_pairs) * len(prompts)}) {run_id} pair={pair_id} prompt_idx={prompt_idx} seed={run_seed}")
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=start_img,
                last_image=end_img,
                height=target_h,
                width=target_w,
                num_frames=int(cfg["inference"]["num_frames"]),
                guidance_scale=float(cfg["inference"]["guidance_scale"]),
                num_inference_steps=int(cfg["inference"]["num_inference_steps"]),
                generator=generator,
            )

            frames = output.frames[0]
            video_path = run_dir / "sample.mp4"
            export_to_video(frames, str(video_path), fps=fps)

            run_cfg = dict(cfg)
            run_cfg["_resolved"] = {
                "pair_id": pair_id,
                "start_frame": str(start_path),
                "end_frame": str(end_path),
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "seed": run_seed,
            }
            with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(run_cfg, f, sort_keys=False, allow_unicode=False)

            record = {
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "video_path": str(video_path),
                "pair_id": pair_id,
                "start_frame": str(start_path),
                "end_frame": str(end_path),
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": run_seed,
                "device": "cuda" if use_cuda else "cpu",
                "dtype": str(dtype),
                "config": cfg,
            }
            with (out_dir / "runs_metadata.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"[saved] {video_path}")
            print(f"[saved] {run_dir / 'config_snapshot.yaml'}")

    print(f"[saved] {out_dir / 'runs_metadata.jsonl'}")
    print("[done]")


if __name__ == "__main__":
    main()
