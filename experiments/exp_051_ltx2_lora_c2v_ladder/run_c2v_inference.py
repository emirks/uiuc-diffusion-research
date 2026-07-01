"""exp_051 — standalone C2V inference via the official trainer's ValidationRunner.

Generates the capability-ladder samples: LTX-2 19B (optionally + a LoRA checkpoint)
conditioned on the first 9 + last 8 pixel frames (= first 2 + last 1 latent frames)
of the earth_wave_0 test clip, prompted for a SHDWSMK shadow-smoke transition.

Runs in the ltx-trainer uv env:
    cd $LAB/LTX-2-official/packages/ltx-trainer
    uv run --frozen python <this file> --label base
    uv run --frozen python <this file> --label t2v  --lora <lora_weights_step_02000.safetensors>

Three test transitions (earth_wave_0/1/2 endpoints; ew2 runs at portrait dims),
two prompts each, identical seed/settings -> 6 samples per arm:
    sample 1: ew0 trigger prompt      sample 2: ew0 prompt WITHOUT trigger
    sample 3: ew1 trigger prompt      sample 4: ew1 prompt WITHOUT trigger
    sample 5: ew2 trigger prompt      sample 6: ew2 prompt WITHOUT trigger
Condition clips are 9-frame cuts (first 9 / last 9 source frames); the suffix
keeps only the final latent frame of its 2-latent encode (num_frames=8), so no
foreign-transition content can bleed into the end anchor via the causal VAE.

Uses the exact same code path as in-training validation (ValidationRunner), and the
exact same LoRA loading as the trainer (PEFT wrap + diffusion_model.-prefix strip +
set_peft_model_state_dict on the base model).
"""

import argparse
import re
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

from ltx_trainer.config import (
    PrefixConditionConfig,
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
EXP = LAB / "diffusion-research/experiments/exp_051_ltx2_lora_c2v_ladder"
OUT_ROOT = LAB / "diffusion-research/outputs/videos/exp_051_ltx2_lora_c2v_ladder"

# All exp_050/exp_051 rank-32 arms share these video-attention target modules.
TARGET_MODULES = [
    "attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0",
]

# (clip key, per-sample video_dims (W, H, F), caption-style prompt body)
TEST_CLIPS = [
    (
        "ew0",
        (640, 480, 121),
        "A young woman with slicked-back dark hair, white futuristic wraparound "
        "sunglasses and a black leather jacket with white piping stands in a narrow "
        "bookstore aisle between tall shelves packed with colorful books, seen from "
        "a high angle. A dense mass of black smoke sweeps across the frame and "
        "engulfs her, and the scene transforms into a woman with a voluminous afro "
        "and dark sunglasses, wearing a black crop top and a silver choker, holding "
        "the metal railings of a sunlit concrete staircase beneath tall bright "
        "windows, looking up at the camera.",
    ),
    (
        "ew1",
        (640, 480, 121),
        "A man in a blue baseball cap and a red sweater tied over a blue striped "
        "shirt stands with his arms crossed on an old European street in front of "
        "a stone building. A dense mass of black smoke sweeps across the frame and "
        "engulfs him, and the scene transforms into a woman in a white headscarf "
        "and a red tracksuit crouching to tie her shoe beside a vegetable market "
        "stall with green crates, while a vendor stands behind the produce.",
    ),
    (
        "ew2",
        (480, 640, 121),
        "A blonde woman in a fluffy red sweater and brown trousers leans forward "
        "on a hilltop overlooking a distant city under a clear blue sky, smiling "
        "at the camera. A dense mass of black smoke sweeps across the frame and "
        "engulfs her, and the scene transforms into a close-up of a smiling woman "
        "with long braids, holding her hands to her cheeks with light blue nails, "
        "wearing a colorful top in a bright desert landscape.",
    ),
]


def build_conditions(clip_key: str):
    return [
        PrefixConditionConfig(video=str(EXP / f"dataset/cond_{clip_key}_start9.mp4"), num_frames=9),
        SuffixConditionConfig(video=str(EXP / f"dataset/cond_{clip_key}_end9.mp4"), num_frames=8),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="arm name -> output subdir (base/t2v/i2v_ff05/c2v)")
    ap.add_argument("--lora", default=None, help="path to lora_weights_step_*.safetensors (omit for base model)")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--wandb", action="store_true", help="log samples to W&B project creative-transition-transfer")
    args = ap.parse_args()

    out_dir = OUT_ROOT / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    step = 0
    if args.lora:
        m = re.search(r"step_(\d+)", Path(args.lora).name)
        step = int(m.group(1)) if m else 0

    samples = []
    for clip_key, dims, body in TEST_CLIPS:
        samples.append(ValidationSample(
            prompt="SHDWSMK " + body, conditions=build_conditions(clip_key), video_dims=dims))
        samples.append(ValidationSample(
            prompt=body, conditions=build_conditions(clip_key), video_dims=dims))

    val_cfg = ValidationConfig(
        samples=samples,
        negative_prompt="worst quality, inconsistent motion, distorted, jittery",
        video_dims=(640, 480, 121),
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

    # Build the runner FIRST: it loads Gemma, caches prompt embeddings, encodes the
    # conditioning clips with the VAE encoder, then frees both - so the 19B
    # transformer is never co-resident with the text encoder.
    print(f"[info] building ValidationRunner (label={args.label})")
    runner = ValidationRunner(config=val_cfg, model_path=MODEL, text_encoder_path=GEMMA)

    print("[info] loading transformer")
    transformer = load_transformer(MODEL, device="cpu", dtype=torch.bfloat16)

    if args.lora:
        print(f"[info] applying LoRA (r={args.rank}, alpha={args.alpha}): {args.lora}")
        transformer = get_peft_model(
            transformer,
            LoraConfig(
                r=args.rank,
                lora_alpha=args.alpha,
                target_modules=TARGET_MODULES,
                lora_dropout=0.0,
                init_lora_weights=True,
            ),
        )
        sd = load_file(args.lora)
        sd = {k.replace("diffusion_model.", "", 1): v for k, v in sd.items()}
        set_peft_model_state_dict(transformer.get_base_model(), sd)
        print("[info] LoRA checkpoint loaded")

    transformer = transformer.to(device).eval()

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project="creative-transition-transfer",
            name=f"exp051_infer_{args.label}",
            tags=["ltx2", "c2v-inference", "exp_051", args.label],
            config={"lora": args.lora, "rank": args.rank, "alpha": args.alpha, "step": step},
        )

    progress = TrainingProgress(enabled=True, total_steps=1)
    saved = runner.run(
        transformer=transformer,
        step=step,
        output_dir=out_dir,
        device=device,
        progress=progress,
        wandb_run=wandb_run,
    )
    for idx, path in saved:
        print(f"[done] sample {idx + 1} -> {path}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
