"""exp_060 — σ_seed generation via the trainer's ValidationRunner (exp_057 fork).

The decision-generating arm = the exp_056 IC-LoRA (step 3000), the exact
adapter scored in the archived exp_057 generations. Recipe identical to
exp_057 (480x640x121@24, 30 steps, CFG 4, STG 1 stg_v [29], prefix 9f /
suffix 8f, ICTRANS + type-blind endpoint caption). Per-sample seeds let all
12 items x 5 seeds = 60 videos ride ONE model load in a single 1-GPU job
(ValidationRunner caches prompt embeddings + conditioning media on CPU, then
generates each sample with its own torch.Generator seed).

Per item: conditions = clip A's start9 (prefix, 9f) + end9 (suffix, 8f) +
clip B as reference (include_in_output=False). Output filename encodes item +
seed. Skip-if-exists -> requeue / preemption-safe (a restart rebuilds only
the samples whose output mp4 is still missing).

    cd $LAB/LTX-2-official/packages/ltx-trainer
    uv run --frozen python <this file> --lora <ckpt> [--seeds 42,43,44,45,46]
"""

import argparse
import json
import re
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

from ltx_trainer.config import (
    PrefixConditionConfig,
    ReferenceConditionConfig,
    SuffixConditionConfig,
    ValidationConfig,
    ValidationSample,
)
from ltx_trainer.model_loader import load_transformer
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.validation_runner import ValidationRunner

LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
MODEL = LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"
GEMMA = LAB / "cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized"
EXP = LAB / "diffusion-research/experiments/exp_060_sigma_seed"
STD = LAB / "diffusion-research/data/processed/transitions_std121"
OUT_DIR = LAB / "diffusion-research/outputs/videos/exp_060_sigma_seed/adapter"
COND = EXP / "dataset/cond"

TARGET_MODULES = [
    "attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0",
    "ff.net.0.proj", "ff.net.2",
]


def clip_dir(stem: str) -> str:
    cls = "_".join(stem.split("_")[:-1])
    return str(STD / cls / f"{stem}.mp4")


def build_sample(it: dict, seed: int) -> ValidationSample:
    a, b = it["clip_a"], it["clip_b"]
    return ValidationSample(
        prompt="ICTRANS " + it["caption_a"],
        seed=seed,
        conditions=[
            PrefixConditionConfig(video=str(COND / f"{a}_start9.mp4"), num_frames=9),
            SuffixConditionConfig(video=str(COND / f"{a}_end9.mp4"), num_frames=8),
            ReferenceConditionConfig(
                video=clip_dir(b),
                downscale_factor=1, temporal_scale_factor=1, include_in_output=False),
        ],
    )


def item_id(cls: str, seed: int) -> str:
    return f"sigseed__{cls}__s{seed}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    sel = json.loads((EXP / "dataset/selection.json").read_text())
    items = sel["items"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # one (item, seed) task per missing output -> resumable
    todo = [(it, seed) for seed in seeds for it in items
            if not (OUT_DIR / (item_id(it["class"], seed) + ".mp4")).exists()]
    print(f"[info] {len(todo)}/{len(items) * len(seeds)} (item,seed) videos to generate")
    if not todo:
        print("[done] nothing to do")
        return

    m = re.search(r"step_(\d+)", Path(args.lora).name)
    step = int(m.group(1)) if m else 0

    val_cfg = ValidationConfig(
        samples=[build_sample(it, seed) for it, seed in todo],
        negative_prompt="worst quality, inconsistent motion, distorted, jittery",
        video_dims=(480, 640, 121),
        frame_rate=24.0,
        seed=42,                         # per-sample seed overrides this
        inference_steps=30,
        interval=1,
        guidance_scale=4.0,
        stg_scale=1.0,
        stg_blocks=[29],
        stg_mode="stg_v",
        generate_audio=False,
    )

    device = torch.device("cuda")
    print("[info] building ValidationRunner (Gemma embed + condition encode, then freed)")
    runner = ValidationRunner(config=val_cfg, model_path=MODEL, text_encoder_path=GEMMA)

    print("[info] loading transformer")
    transformer = load_transformer(MODEL, device="cpu", dtype=torch.bfloat16)
    print(f"[info] applying LoRA (r={args.rank}, a={args.alpha}, attn+FFN): {args.lora}")
    transformer = get_peft_model(
        transformer,
        LoraConfig(r=args.rank, lora_alpha=args.alpha, target_modules=TARGET_MODULES,
                   lora_dropout=0.0, init_lora_weights=True),
    )
    sd = load_file(args.lora)
    sd = {k.replace("diffusion_model.", "", 1): v for k, v in sd.items()}
    set_peft_model_state_dict(transformer.get_base_model(), sd)
    print("[info] LoRA checkpoint loaded")
    transformer = transformer.to(device).eval()

    progress = TrainingProgress(enabled=True, total_steps=1)
    saved = runner.run(transformer=transformer, step=step, output_dir=OUT_DIR / "_runner",
                       device=device, progress=progress)
    for idx, path in saved:
        it, seed = todo[idx]
        dst = OUT_DIR / (item_id(it["class"], seed) + ".mp4")
        Path(path).rename(dst)
        print(f"[done] {dst.name} -> {dst}")


if __name__ == "__main__":
    main()
