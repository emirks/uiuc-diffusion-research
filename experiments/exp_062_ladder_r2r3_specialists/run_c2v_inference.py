"""exp_062 — R2/R3/R1K/R3X generation via the trainer's ValidationRunner (exp_051
fork). AMENDED 2026-07-16 to ladder_items_v2.json (PLAN Amendment 1): conditioning is
SIDE-KEYED (suffix only iff the class is two_sided) and specialist checkpoints come
from the per-class `<cls>_keyed/` dir (one_sided = prefix-only retrains; two_sided =
symlink to the blind==keyed run).

Modes (per --class, --seed):
  default : R2 (grid r2_items, held-in) + R3 (grid test_items, unseen), ckpt 250 & 2000
  --no-adapter : R1K = base 19B, NO adapter (zero-init PEFT = base), prefix-only,
                 test_items only. Only valid for one_sided (r1k) classes.
  --r3x   : R3X = donor endpoints from grid r3x.recipients[cls].donors, prefix-only
                 (recipients are one_sided), ckpt 2000 only.

Conditioning: prefix(9f) always; suffix(8f) iff grid class suffix_conditioning
(default/R2/R3) — R1K and R3X are prefix-only by construction. Prompt = ICTRANS +
type-blind caption (exp_061 wins for cross-rung parity).

Co-residence-safe (exp_051): build the runner FIRST (loads+frees Gemma, encodes
conds), THEN load the 19B; swap LoRA state per checkpoint (base = zero-init, no load).
Run in the ltx-trainer uv env:
    cd $LAB/LTX-2-official/packages/ltx-trainer
    uv run --frozen python <this file> --class shadow --seed 42            # R2/R3
    uv run --frozen python <this file> --class shadow --seed 42 --no-adapter  # R1K
    uv run --frozen python <this file> --class shadow --seed 42 --r3x         # R3X
Skip-if-exists per output mp4 (requeue-safe).
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
GRID = json.loads((REPO / "docs/eval_ladder/ladder_items_v2.json").read_text())

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


def build_sample(clip: str, caps: dict, use_suffix: bool) -> ValidationSample:
    conds = [PrefixConditionConfig(video=str(COND / f"{clip}_start9.mp4"), num_frames=9)]
    if use_suffix:
        conds.append(SuffixConditionConfig(video=str(COND / f"{clip}_end9.mp4"), num_frames=8))
    return ValidationSample(prompt=f"ICTRANS {caps[clip]}", conditions=conds, video_dims=(480, 640, 121))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--no-adapter", dest="no_adapter", action="store_true", help="R1K: base model, prefix-only")
    ap.add_argument("--r3x", action="store_true", help="R3X: cross-class donor endpoints")
    args = ap.parse_args()

    classes = GRID["classes"]
    g = classes[args.cls]
    caps = load_captions()

    # ---- mode -> (rung, targets [(rung,clip)], steps, use_suffix) ----
    if args.no_adapter:
        assert g["r1k"], f"{args.cls} is not an R1K (one_sided) class"
        rung, steps, use_suffix = "R1K", [None], False
        targets = [(rung, c) for c in g["test_items"]]
    elif args.r3x:
        assert args.cls in GRID["r3x"]["recipients"], f"{args.cls} not an R3X recipient (B8)"
        rung, steps, use_suffix = "R3X", [2000], False   # recipients one_sided -> prefix-only
        targets = [(rung, d["donor_clip"]) for d in GRID["r3x"]["recipients"][args.cls]["donors"]]
    else:
        steps, use_suffix = [250, 2000], bool(g["suffix_conditioning"])
        targets = [("R2", c) for c in g["r2_items"]] + [("R3", c) for c in g["test_items"]]

    def out_path(rung, clip, step):
        stem = f"{rung}__{args.cls}__{clip}__s{args.seed}"
        if step is not None:
            stem += f"__ckpt{step}"
        return OUT_ROOT / rung / f"{stem}.mp4"

    pending = [(r, c, s) for (r, c) in targets for s in steps if not out_path(r, c, s).exists()]
    if not pending:
        print(f"[done] {args.cls} s{args.seed} ({targets[0][0]}): nothing to do")
        return
    for r, c, s in pending:
        out_path(r, c, s).parent.mkdir(parents=True, exist_ok=True)
    cond_mode = "prefix+suffix" if use_suffix else "prefix-only"
    print(f"[info] {args.cls} s{args.seed} {targets[0][0]}: {len(pending)} pending, {cond_mode}, steps {steps}")

    samples = [build_sample(c, caps, use_suffix) for (_, c) in targets]
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
                   lora_dropout=0.0, init_lora_weights=True),   # B=0 -> identity == base model until a ckpt is loaded
    )
    transformer = transformer.to(device).eval()
    ckpt_dir = CKPT_ROOT / f"{args.cls}_keyed" / "checkpoints"

    for step in steps:
        needed = [(r, c) for (r, c) in targets if not out_path(r, c, step).exists()]
        if not needed:
            continue
        if step is not None:   # load the specialist checkpoint; R1K leaves the zero-init (base) adapter
            ckpt = ckpt_dir / f"lora_weights_step_{step:05d}.safetensors"
            if not ckpt.exists():
                print(f"[warn] missing checkpoint (train not done?): {ckpt} — skipping step {step}")
                continue
            sd = load_file(str(ckpt))
            sd = {k.replace("diffusion_model.", "", 1): v for k, v in sd.items()}
            set_peft_model_state_dict(transformer.get_base_model(), sd)
        label = "base" if step is None else f"step {step}"
        print(f"[info] {args.cls} {rung if (args.no_adapter or args.r3x) else 'R2/R3'} {label}: generating {len(targets)} samples")
        tmp = OUT_ROOT / "_runner" / f"{args.cls}_s{args.seed}_{targets[0][0]}_{step}"
        progress = TrainingProgress(enabled=True, total_steps=1)
        saved = runner.run(transformer=transformer, step=(step or 0), output_dir=tmp,
                           device=device, progress=progress)
        for idx, path in saved:
            r, clip = targets[idx]
            dst = out_path(r, clip, step)
            Path(path).rename(dst)
            print(f"[done] {dst.name}")


if __name__ == "__main__":
    main()
