"""exp_067 Stage-2 generation — residual-reference IC-LoRA grid.

Fork of exp_065 run_grid_inference.py with ONE semantic change: the ValidationRunner
is constructed with residual_reference=True (gamma=1.0922), so in-context references are
residualized (endpoint-content-subtracted) at inference EXACTLY as in training — the third
call site of ltx_trainer.residual_reference. Feeding raw references to a residual-trained
adapter would be a train/inference mismatch.

Run in the ltx-trainer uv env:
    cd $LAB/LTX-2-official/packages/ltx-trainer
    uv run --frozen python <this file> --manifest <abs manifest.json> --seed 42 \
        [--chunk i --num-chunks n]
Skip-if-exists on OUT_ROOT/<rung>/<id>__s<seed>.mp4 (requeue-safe).
"""
import argparse, json, re
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

from ltx_trainer.config import (
    PrefixConditionConfig, ReferenceConditionConfig, SuffixConditionConfig,
    ValidationConfig, ValidationSample,
)
from ltx_trainer.model_loader import load_transformer
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.validation_runner import ValidationRunner

LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
MODEL = LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"
GEMMA = LAB / "cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized"
REPO = LAB / "diffusion-research"
STD = REPO / "data/processed/transitions_std121"
OUT_ROOT = REPO / "outputs/videos/exp_067_s2_residual_generalist"
GAMMA = 1.0922

TARGET_MODULES = [
    "attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0",
    "ff.net.0.proj", "ff.net.2",
]


def build_sample(r: dict) -> ValidationSample:
    e, cond_dir = r["endpoints"], REPO / r["cond_dir"]
    conds = [PrefixConditionConfig(video=str(cond_dir / f"{e}_start9.mp4"), num_frames=9)]
    if not r["prefix_only"]:
        conds.append(SuffixConditionConfig(video=str(cond_dir / f"{e}_end9.mp4"), num_frames=8))
    if r.get("reference"):
        conds.append(ReferenceConditionConfig(
            video=str(STD / r["reference_class"] / (r["reference"] + ".mp4")),
            downscale_factor=1, temporal_scale_factor=1, include_in_output=False))
    return ValidationSample(prompt=r["prompt"], conditions=conds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="absolute path to manifest_s2.json")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--num-chunks", type=int, default=1)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    args = ap.parse_args()

    doc = json.loads(Path(args.manifest).read_text())
    adapter = (REPO / doc["adapter"]) if doc["adapter"] else None
    if adapter is not None:
        assert adapter.exists(), f"adapter missing: {adapter}"

    rows = [r for r in doc["rows"] if not r.get("deferred")][args.chunk::args.num_chunks]

    def out_path(r):
        return OUT_ROOT / r["rung"] / f"{r['id']}__s{args.seed}.mp4"

    todo = [r for r in rows if not out_path(r).exists()]
    print(f"[info] {Path(args.manifest).name} seed={args.seed} chunk={args.chunk}/{args.num_chunks}: "
          f"{len(todo)}/{len(rows)} to generate (residual_reference=ON, gamma={GAMMA})")
    if not todo:
        print("[done] nothing to do")
        return
    for r in todo:
        out_path(r).parent.mkdir(parents=True, exist_ok=True)

    m = re.search(r"step_(\d+)", adapter.name) if adapter else None
    step = int(m.group(1)) if m else 0

    val_cfg = ValidationConfig(
        samples=[build_sample(r) for r in todo],
        negative_prompt="worst quality, inconsistent motion, distorted, jittery",
        video_dims=(480, 640, 121), frame_rate=24.0, seed=args.seed,
        inference_steps=30, interval=1, guidance_scale=4.0,
        stg_scale=1.0, stg_blocks=[29], stg_mode="stg_v", generate_audio=False,
    )

    device = torch.device("cuda")
    runner = ValidationRunner(config=val_cfg, model_path=MODEL, text_encoder_path=GEMMA,
                              residual_reference=True, residual_gamma=GAMMA)
    transformer = load_transformer(MODEL, device="cpu", dtype=torch.bfloat16)
    if adapter is not None:
        print(f"[info] loading adapter {adapter.name}")
        transformer = get_peft_model(
            transformer,
            LoraConfig(r=args.rank, lora_alpha=args.alpha, target_modules=TARGET_MODULES,
                       lora_dropout=0.0, init_lora_weights=True))
        sd = load_file(str(adapter))
        sd = {k.replace("diffusion_model.", "", 1): v for k, v in sd.items()}
        set_peft_model_state_dict(transformer.get_base_model(), sd)
    transformer = transformer.to(device).eval()

    tag = Path(args.manifest).stem
    tmp_dir = OUT_ROOT / "_runner" / f"{tag}_s{args.seed}_c{args.chunk}"
    saved = runner.run(transformer=transformer, step=step, output_dir=tmp_dir,
                       device=device, progress=TrainingProgress(enabled=True, total_steps=1))
    for idx, path in saved:
        dst = out_path(todo[idx])
        Path(path).rename(dst)
        print(f"[done] {todo[idx]['id']} s{args.seed} -> {dst}")


if __name__ == "__main__":
    main()
