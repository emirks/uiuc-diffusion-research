"""Emit the training configs for the bottleneck campaign — derived, never hand-edited.

The recipe is FROZEN to ic_gen's (rank 32 / alpha 32, attn+FFN, lr 2e-4, batch 1, bf16,
shifted_logit_normal, seed 42) so that the ONLY variable between B1 and the incumbent is the width
of the reference channel. Every config below starts from one dict and differs in named fields only.

Configs emitted:

  equiv_mine     150 steps, plain recipe, run on the PRIVATE trainer worktree ($LAB/LTX-2-bneck)
  equiv_lineage  150 steps, plain recipe, run on the LINEAGE trainer ($LAB/LTX-2-cond-bleed-fix)
                 -> the advisor's trainer-equivalence check. B1-vs-ic_gen crosses trainer versions,
                    so we must first show the diff is a no-op for the full-ref path. Identical seed
                    and identical config: the two loss sequences should agree within noise.
  b1_smoke        20 steps, bottleneck ON -> 0e sanity: forward+backward runs, encoder gradients are
                    nonzero, and we get a MEASURED steps/sec for the real B1 budget.
  b1            5000 steps, bottleneck ON, checkpoints at 1500/3000/5000 (the advisor's read points)

Validation is disabled in all of them. These are mechanical checks and a single-variable comparison;
a real claim-bearing run restores the ID/OOD/control triad per the `lora-train` skill.

    python experiments/exp_078_operator_token_bottleneck/make_configs.py
"""

import copy
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parent
REPO_ROOT = EXP.parents[1]
LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
OUT_DIR = EXP / "configs"

ROOT = EXP / "dataset" / "roots" / "b1"
TRAIN_OUT = REPO_ROOT / "outputs" / "training" / "bneck"

# Frozen ic_gen recipe. Read off outputs/training/ladder2/ic_gen/training_config.yaml.
BASE = {
    "model": {
        "model_path": str(LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"),
        "text_encoder_path": str(LAB / "cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized"),
        "training_mode": "lora",
        "load_checkpoint": None,
    },
    "lora": {
        "rank": 32,
        "alpha": 32,
        "dropout": 0.0,
        "target_modules": [
            "attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
            "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0",
            "ff.net.0.proj", "ff.net.2",
        ],
    },
    "optimization": {
        "learning_rate": 2e-4,
        "steps": 5000,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "optimizer_type": "adamw",
        "scheduler_type": "linear",
        "scheduler_params": {},
        "enable_gradient_checkpointing": True,
    },
    "acceleration": {
        "mixed_precision_mode": "bf16",
        "load_text_encoder_in_8bit": False,
        "quantization": None,
        "offload_optimizer_during_validation": False,
    },
    "data": {"preprocessed_data_root": str(ROOT), "num_dataloader_workers": 2},
    "flow_matching": {"timestep_sampling_mode": "shifted_logit_normal", "timestep_sampling_params": {}},
    # interval=None disables validation (the schema rejects 0; the trainer treats None as falsy).
    "validation": {"prompts": [], "samples": [], "interval": None, "generate_video": False,
                   "generate_audio": False},
    "checkpoints": {"interval": 500, "keep_last_n": -1, "precision": "bfloat16", "no_resume": False,
                    "save_training_state": "minimal"},
    "hub": {"push_to_hub": False, "hub_model_id": None},
    "seed": 42,
    "training_strategy": {
        "name": "flexible",
        "audio": None,
        "video": {
            "is_generated": True,
            "latents_dir": "latents",
            "cond_clean_latents_dir": "cond_clean_latents",
            "conditions": [
                {"type": "reference", "latents_dir": "reference_latents", "probability": 1.0},
                {"type": "mask", "mask_dir": "masks", "probability": 1.0},
            ],
        },
    },
}

BOTTLENECK = {
    "token_shape": [6, 4, 3],   # K = 72. Legal: 20//4 == 15//3 == 5, and (16-1) % (6-1) == 0.
    "width": 512,
    "depth": 2,
    "num_heads": 8,
    "prefix_latent_frames": 2,
    "suffix_latent_frames": 1,
}

# Retry b1r (advisor amendment): three deltas vs B1, everything else frozen-identical.
#  - skip_scale 1.68 (measured raw/pooled RMS) => tokens demo-dependent by construction.
#  - encoder_lr 5e-4 separate optimizer group (LoRA stays 2e-4).
#  - reference dropout 10% (probability 0.9) — the torch.rand draw happens regardless, so B1's and
#    b1r's noise/sigma RNG streams stay aligned; only the threshold differs.
BOTTLENECK_R = {**BOTTLENECK, "skip_scale": 1.68, "encoder_lr": 5e-4}


def build(name: str, steps: int, bottleneck, ckpt_interval: int) -> dict:
    """bottleneck: False (off) | True/"b1" (B1 encoder) | "r" (b1r pooled-skip retry)."""
    cfg = copy.deepcopy(BASE)
    cfg["optimization"]["steps"] = steps
    cfg["checkpoints"]["interval"] = ckpt_interval
    cfg["output_dir"] = str(TRAIN_OUT / name)
    if bottleneck:
        bn = copy.deepcopy(BOTTLENECK_R if bottleneck == "r" else BOTTLENECK)
        cfg["training_strategy"]["video"]["conditions"][0]["bottleneck"] = bn
        if bottleneck == "r":
            cfg["training_strategy"]["video"]["conditions"][0]["probability"] = 0.9  # 10% ref-dropout
    return cfg


CONFIGS = {
    # (steps, bottleneck, checkpoint interval)
    "b1r": (5000, "r", 500),   # retry: pooled-demo skip + enc lr 5e-4 + 10% ref-dropout
    "equiv_mine": (150, False, 1000),
    "equiv_lineage": (150, False, 1000),
    # lineage-vs-lineage self-consistency control: identical config and seed, unmodified
    # trainer, separate output_dir. Establishes the run-to-run nondeterminism floor, without
    # which an equiv_mine-vs-equiv_lineage divergence cannot be attributed to the diff.
    "equiv_lineage2": (150, False, 1000),
    # second same-code floor (mine-vs-mine), advisor ruling: use max(D_lineage-lineage2,
    # D_mine-mine2) as the denominator, guarding against one unluckily-quiet floor sample.
    "equiv_mine2": (150, False, 1000),
    "b1_smoke": (20, True, 1000),
    "b1": (5000, True, 500),
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (steps, bottleneck, interval) in CONFIGS.items():
        cfg = build(name, steps, bottleneck, interval)
        path = OUT_DIR / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=True))
        k = ""
        if bottleneck:
            f, h, w = BOTTLENECK["token_shape"]
            k = f"  bottleneck K={f * h * w} shape=({f},{h},{w})"
        print(f"[cfg] {path.relative_to(REPO_ROOT)}  steps={steps}{k or '  bottleneck=OFF (plain recipe)'}")


if __name__ == "__main__":
    main()
