"""exp_061 — R0/R1 ladder-baseline generation via the trainer's ValidationRunner
(exp_060 fork; NO LoRA — the plain LTX-2 19B dev transformer).

R0 = base model, prompt only (conditions=[]): does the base prior contain the
     effect at all?
R1 = base model + endpoint conditioning (prefix 9f + suffix 8f, exp_051
     recipe), prompt only, NO reference, NO adapter: conditioning suppression.

Recipe otherwise identical to the exp_057/060 decision-generating contract:
480x640x121@24, 30 steps, CFG 4, STG 1 stg_v [29], negative prompt, prompt =
"ICTRANS " + type-blind endpoint caption of the item clip (trigger kept for
prompt parity with the exp_056/057 base twins and future adapter rungs;
exp_051 showed the token is inert without the adapter). Same items and seeds
in both arms (paired design).

Chunking: the deterministic full task list (items x seeds, selection order) is
split into --num-chunks contiguous chunks; a task belongs to chunk
(index * num_chunks) // n_tasks. Skip-if-exists per output mp4 -> resumable.

    cd $LAB/LTX-2-official/packages/ltx-trainer
    uv run --frozen python <this file> --arm r0 --chunk 0 --num-chunks 2
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

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
EXP = LAB / "diffusion-research/experiments/exp_061_ladder_r0_r1"
REPO = LAB / "diffusion-research"
OUT_ROOT = REPO / "outputs/videos/exp_061_ladder_r0_r1"
COND = EXP / "dataset/cond"


def build_sample(it: dict, seed: int, arm: str) -> ValidationSample:
    conditions = []
    if arm == "r1":
        stem = it["clip"]
        conditions = [
            PrefixConditionConfig(video=str(COND / f"{stem}_start9.mp4"), num_frames=9),
            SuffixConditionConfig(video=str(COND / f"{stem}_end9.mp4"), num_frames=8),
        ]
    return ValidationSample(
        prompt="ICTRANS " + it["caption"],
        seed=seed,
        conditions=conditions,
    )


def item_id(arm: str, it: dict, seed: int) -> str:
    return f"{arm}__{it['class']}__{it['clip']}__s{seed}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["r0", "r1"])
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--num-chunks", type=int, default=1)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    sel = json.loads((EXP / "dataset/selection.json").read_text())
    items = sel["items"]
    assert all(it["caption"] for it in items), "missing captions — run caption_missing.py"

    out_dir = OUT_ROOT / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = OUT_ROOT / "config_snapshot.yaml"
    if not snap.exists():
        shutil.copyfile(EXP / "config.yaml", snap)

    # deterministic full task list -> contiguous chunks -> skip-if-exists
    tasks = [(it, seed) for it in items for seed in seeds]
    mine = [t for i, t in enumerate(tasks)
            if (i * args.num_chunks) // len(tasks) == args.chunk]
    todo = [(it, seed) for it, seed in mine
            if not (out_dir / (item_id(args.arm, it, seed) + ".mp4")).exists()]
    print(f"[info] arm={args.arm} chunk {args.chunk}/{args.num_chunks}: "
          f"{len(todo)}/{len(mine)} videos to generate ({len(tasks)} arm total)")
    if not todo:
        print("[done] nothing to do")
        return

    val_cfg = ValidationConfig(
        samples=[build_sample(it, seed, args.arm) for it, seed in todo],
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

    print("[info] loading transformer (base model, NO LoRA)")
    transformer = load_transformer(MODEL, device="cpu", dtype=torch.bfloat16)
    transformer = transformer.to(device).eval()

    progress = TrainingProgress(enabled=True, total_steps=1)
    saved = runner.run(transformer=transformer, step=0,
                       output_dir=out_dir / f"_runner_{args.chunk}",
                       device=device, progress=progress)
    for idx, path in saved:
        it, seed = todo[idx]
        dst = out_dir / (item_id(args.arm, it, seed) + ".mp4")
        Path(path).rename(dst)
        print(f"[done] {dst.name} -> {dst}")


if __name__ == "__main__":
    main()
