"""exp_058 — v2 eval inference via the trainer's ValidationRunner (exp_057 fork).

Identical recipe (seed 42, 480x640x121@24, 30 steps, CFG 4, STG 1 stg_v [29]).
Reads dataset/quads_v2.json: per-item cond_dir + prefix_only flag (prefix-only
items get NO suffix condition). Runs in the ltx-trainer uv env:

    cd $LAB/LTX-2-official/packages/ltx-trainer
    uv run --frozen python <this file> --label ic2 --lora <ABSOLUTE ckpt> [--chunk i --num-chunks n]
    uv run --frozen python <this file> --label base            [--chunk ...]

Per-chunk runner dirs; skip-if-exists on quads/<id>.mp4 (requeue-safe).
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
REPO = LAB / "diffusion-research"
EXP = REPO / "experiments/exp_058_ic_lora_diverse_retrain"
STD = REPO / "data/processed/transitions_std121"
OUT_ROOT = REPO / "outputs/videos/exp_058_ic_lora_diverse_retrain"

TARGET_MODULES = [
    "attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0",
    "ff.net.0.proj", "ff.net.2",
]


def build_sample(q: dict) -> ValidationSample:
    e, cond_dir = q["endpoints"], REPO / q["cond_dir"]
    conds = [PrefixConditionConfig(video=str(cond_dir / f"{e}_start9.mp4"), num_frames=9)]
    if not q["prefix_only"]:
        conds.append(SuffixConditionConfig(video=str(cond_dir / f"{e}_end9.mp4"), num_frames=8))
    conds.append(ReferenceConditionConfig(
        video=str(STD / q["reference_class"] / (q["reference"] + ".mp4")),
        downscale_factor=1, temporal_scale_factor=1, include_in_output=False))
    return ValidationSample(prompt=q["prompt"], conditions=conds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=["ic2", "base"])
    ap.add_argument("--lora", default=None)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--num-chunks", type=int, default=1)
    args = ap.parse_args()
    assert (args.label == "ic2") == bool(args.lora), "ic2 needs --lora; base must not"
    if args.lora:
        assert Path(args.lora).is_absolute() and Path(args.lora).exists(), \
            f"LoRA path must be absolute + existing (ops-gotcha #5): {args.lora}"

    quads = json.loads((EXP / "dataset/quads_v2.json").read_text())
    quads = [q for q in quads if q["label"] == args.label]
    quads = quads[args.chunk::args.num_chunks]

    out_dir = OUT_ROOT / args.label / f"chunk{args.chunk}of{args.num_chunks}"
    quad_dir = OUT_ROOT / args.label / "quads"
    quad_dir.mkdir(parents=True, exist_ok=True)

    todo = [q for q in quads if not (quad_dir / (q["id"] + ".mp4")).exists()]
    print(f"[info] label={args.label} chunk={args.chunk}/{args.num_chunks}: "
          f"{len(todo)}/{len(quads)} to generate")
    if not todo:
        print("[done] nothing to do")
        return

    step = 0
    if args.lora:
        m = re.search(r"step_(\d+)", Path(args.lora).name)
        step = int(m.group(1)) if m else 0

    val_cfg = ValidationConfig(
        samples=[build_sample(q) for q in todo],
        negative_prompt="worst quality, inconsistent motion, distorted, jittery",
        video_dims=(480, 640, 121),
        frame_rate=24.0,
        seed=42,
        inference_steps=30,
        interval=1,
        guidance_scale=4.0,
        stg_scale=1.0,
        stg_blocks=[29],
        stg_mode="stg_v",
        generate_audio=False,
    )

    device = torch.device("cuda")
    print("[info] building ValidationRunner")
    runner = ValidationRunner(config=val_cfg, model_path=MODEL, text_encoder_path=GEMMA)

    print("[info] loading transformer")
    transformer = load_transformer(MODEL, device="cpu", dtype=torch.bfloat16)
    if args.lora:
        print(f"[info] applying LoRA (r={args.rank}, a={args.alpha}): {args.lora}")
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
    saved = runner.run(transformer=transformer, step=step, output_dir=out_dir,
                       device=device, progress=progress)
    for idx, path in saved:
        qid = todo[idx]["id"]
        dst = quad_dir / (qid + ".mp4")
        Path(path).rename(dst)
        print(f"[done] {qid} -> {dst}")


if __name__ == "__main__":
    main()
