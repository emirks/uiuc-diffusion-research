"""exp_056 — standalone IC-LoRA quadruple inference via the trainer's ValidationRunner.

Generates the post-training test quadruples from dataset/quads.json:
each sample = prefix (first 9 px frames of the endpoints clip) + suffix (last
9 -> final latent only) + FULL in-context reference clip, type-blind prompt.
Same code path as in-training validation; LoRA loaded exactly as the trainer
does (PEFT wrap incl. FFN targets + diffusion_model.-prefix strip).

Runs in the ltx-trainer uv env:
    cd $LAB/LTX-2-official/packages/ltx-trainer
    uv run --frozen python <this file> --label ic_lora --lora <ckpt.safetensors> [--chunk 0 --num-chunks 4]
    uv run --frozen python <this file> --label base                              [--chunk ...]

--label ic_lora runs the 38 ic_* quads; --label base runs the 8 base_* twins.
Skip-if-exists on quads/<id>.mp4 -> requeue/preemption-safe; outputs renamed
from the runner's step_XXXXXX_N.mp4 to stable quadruple ids.
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
EXP = LAB / "diffusion-research/experiments/exp_056_ltx2_ic_lora_transition_transfer"
STD = LAB / "diffusion-research/data/processed/transitions_std121"
OUT_ROOT = LAB / "diffusion-research/outputs/videos/exp_056_ltx2_ic_lora_transition_transfer"

# must match config_ic.yaml (rank 32/alpha 32, attention + FFN)
TARGET_MODULES = [
    "attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0",
    "ff.net.0.proj", "ff.net.2",
]


def build_sample(q: dict) -> ValidationSample:
    e = q["endpoints"]
    return ValidationSample(
        prompt=q["prompt"],
        conditions=[
            PrefixConditionConfig(video=str(EXP / f"dataset/cond_q/{e}_start9.mp4"), num_frames=9),
            SuffixConditionConfig(video=str(EXP / f"dataset/cond_q/{e}_end9.mp4"), num_frames=8),
            ReferenceConditionConfig(
                video=str(STD / q["reference_class"] / (q["reference"] + ".mp4")),
                downscale_factor=1, temporal_scale_factor=1, include_in_output=False),
        ],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=["ic_lora", "base"])
    ap.add_argument("--lora", default=None, help="lora_weights_step_*.safetensors (required for ic_lora)")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--num-chunks", type=int, default=1)
    args = ap.parse_args()
    assert (args.label == "ic_lora") == bool(args.lora), "ic_lora needs --lora; base must not have one"

    quads = json.loads((EXP / "dataset/quads.json").read_text())
    want_base = args.label == "base"
    quads = [q for q in quads if q["arm"].startswith("base_") == want_base]
    quads = quads[args.chunk::args.num_chunks]

    # Per-chunk runner dir: concurrent chunks previously shared samples/ and
    # raced on step_XXXXXX_N.mp4 names (renames grabbed other chunks' files).
    out_dir = OUT_ROOT / args.label / f"chunk{args.chunk}of{args.num_chunks}"
    quad_dir = OUT_ROOT / args.label / "quads"
    quad_dir.mkdir(parents=True, exist_ok=True)

    todo = [q for q in quads if not (quad_dir / (q["id"] + ".mp4")).exists()]
    print(f"[info] label={args.label} chunk={args.chunk}/{args.num_chunks}: "
          f"{len(todo)}/{len(quads)} quads to generate")
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
    print("[info] building ValidationRunner (Gemma embed + condition encode, then freed)")
    runner = ValidationRunner(config=val_cfg, model_path=MODEL, text_encoder_path=GEMMA)

    print("[info] loading transformer")
    transformer = load_transformer(MODEL, device="cpu", dtype=torch.bfloat16)
    if args.lora:
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
    saved = runner.run(transformer=transformer, step=step, output_dir=out_dir,
                       device=device, progress=progress)
    for idx, path in saved:
        qid = todo[idx]["id"]
        dst = quad_dir / (qid + ".mp4")
        Path(path).rename(dst)
        print(f"[done] {qid} -> {dst}")


if __name__ == "__main__":
    main()
