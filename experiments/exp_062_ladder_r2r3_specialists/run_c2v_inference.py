"""exp_062 — R2/R3 generation from the per-class specialist LoRAs, via the trainer's
ValidationRunner (exp_051 fork). Per docs/eval_ladder/PLAN.md §5.

For a given --class and --seed, generates that class's:
  R2 (held-in endpoints) = the grid's r2_items (train clips)
  R3 (unseen endpoints)  = the grid's test_items (test clips)
at BOTH specialist checkpoints (step_00250 and step_02000 — the pre-registered
checkpoint-sensitivity robustness check). Conditioning is sidedness-BLIND
(prefix 9f + suffix 8f) exactly as trained. Prompt = ICTRANS + type-blind caption
(test-clip prompts reuse exp_061's for cross-rung parity).

Co-residence-safe order (exp_051): build the runner FIRST (loads+frees Gemma,
encodes conds), THEN load the 19B; swap LoRA state-dict per checkpoint. Run in the
ltx-trainer uv env:
    cd $LAB/LTX-2-official/packages/ltx-trainer
    uv run --frozen python <this file> --class shadow --seed 42
Skip-if-exists per output mp4 (requeue-safe). Submit AFTER exp_062 training (PLAN §C1).
"""
import argparse, json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

from ltx_trainer.config import (
    PrefixConditionConfig, SuffixConditionConfig, ValidationConfig, ValidationSample,
)
from ltx_trainer.model_loader import load_transformer
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.validation_runner import ValidationRunner

LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
MODEL = LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"
GEMMA = LAB / "cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized"
REPO = LAB / "diffusion-research"
EXP = REPO / "experiments/exp_062_ladder_r2r3_specialists"
COND = EXP / "dataset/cond"
CKPT_ROOT = REPO / "outputs/training/exp_062_ladder_r2r3_specialists"
OUT_ROOT = REPO / "outputs/videos/exp_062_ladder_r2r3_specialists"
STEPS = [250, 2000]

# specialist = video-attention targets (exp_051)
TARGET_MODULES = [
    "attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0",
]


def load_captions() -> dict:
    caps = json.loads((REPO / "experiments/exp_058_ic_lora_diverse_retrain/dataset/captions.json").read_text())
    caps.update(json.loads((EXP / "dataset/captions_r2.json").read_text()))
    sel = json.loads((REPO / "experiments/exp_061_ladder_r0_r1/dataset/selection.json").read_text())
    for it in (sel["items"] if isinstance(sel, dict) else sel):   # exp_061 wins (cross-rung parity)
        if it.get("caption"):
            caps[it["clip"]] = it["caption"]
    return caps


def build_sample(clip: str, caps: dict) -> ValidationSample:
    conds = [
        PrefixConditionConfig(video=str(COND / f"{clip}_start9.mp4"), num_frames=9),
        SuffixConditionConfig(video=str(COND / f"{clip}_end9.mp4"), num_frames=8),
    ]
    return ValidationSample(prompt=f"ICTRANS {caps[clip]}", conditions=conds, video_dims=(480, 640, 121))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    args = ap.parse_args()

    grid = json.loads((REPO / "docs/eval_ladder/ladder_items_v1.json").read_text())["classes"]
    g = grid[args.cls]
    caps = load_captions()
    # (rung, clip) targets in a fixed order
    targets = [("R2", c) for c in g["r2_items"]] + [("R3", c) for c in g["test_items"]]

    def out_path(rung, clip, step):
        return OUT_ROOT / rung / f"{rung}__{args.cls}__{clip}__s{args.seed}__ckpt{step}.mp4"

    # what's left to do across both checkpoints
    pending = [(r, c, s) for (r, c) in targets for s in STEPS if not out_path(r, c, s).exists()]
    if not pending:
        print(f"[done] {args.cls} s{args.seed}: nothing to do")
        return
    for r, c, s in pending:
        out_path(r, c, s).parent.mkdir(parents=True, exist_ok=True)
    print(f"[info] {args.cls} s{args.seed}: {len(pending)} videos pending across steps {STEPS}")

    samples = [build_sample(c, caps) for (_, c) in targets]
    val_cfg = ValidationConfig(
        samples=samples, negative_prompt="worst quality, inconsistent motion, distorted, jittery",
        video_dims=(480, 640, 121), frame_rate=24.0, seed=args.seed, inference_steps=30,
        interval=1, guidance_scale=4.0, stg_scale=1.0, stg_blocks=[29], stg_mode="stg_v",
        generate_audio=False,
    )

    device = torch.device("cuda")
    print(f"[info] building ValidationRunner (class={args.cls}, seed={args.seed})")
    runner = ValidationRunner(config=val_cfg, model_path=MODEL, text_encoder_path=GEMMA)
    print("[info] loading transformer + PEFT wrap")
    transformer = load_transformer(MODEL, device="cpu", dtype=torch.bfloat16)
    transformer = get_peft_model(
        transformer,
        LoraConfig(r=args.rank, lora_alpha=args.alpha, target_modules=TARGET_MODULES,
                   lora_dropout=0.0, init_lora_weights=True),
    )
    transformer = transformer.to(device).eval()

    for step in STEPS:
        needed = [(r, c) for (r, c) in targets if not out_path(r, c, step).exists()]
        if not needed:
            continue
        ckpt = CKPT_ROOT / args.cls / "checkpoints" / f"lora_weights_step_{step:05d}.safetensors"
        if not ckpt.exists():
            print(f"[warn] missing checkpoint (train not done?): {ckpt} — skipping step {step}")
            continue
        sd = load_file(str(ckpt))
        sd = {k.replace("diffusion_model.", "", 1): v for k, v in sd.items()}
        set_peft_model_state_dict(transformer.get_base_model(), sd)
        print(f"[info] {args.cls} step {step}: generating {len(targets)} samples")
        tmp = OUT_ROOT / "_runner" / f"{args.cls}_s{args.seed}_ckpt{step}"
        progress = TrainingProgress(enabled=True, total_steps=1)
        saved = runner.run(transformer=transformer, step=step, output_dir=tmp,
                           device=device, progress=progress)
        for idx, path in saved:
            rung, clip = targets[idx]
            dst = out_path(rung, clip, step)
            Path(path).rename(dst)
            print(f"[done] {rung}__{args.cls}__{clip}__s{args.seed}__ckpt{step} -> {dst}")


if __name__ == "__main__":
    main()
