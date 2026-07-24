"""exp_077 TASK 0d (step 6 config) — emit the 50-step SMOKE training config.

Frozen ic_gen (generalist IC-LoRA) recipe, byte-for-byte from make_configs.py::generalist_config:
rank32/alpha32 attn+FFN (arms.yaml targets.attn_ffn), lr 2e-4, flexible strategy, reference-latents
+ mask conditioning, cond_clean_latents. Only three things differ, all forced by "this is a
pipeline SMOKE, not a real run":

  * steps = 50 (config.train.steps)
  * validation DISABLED (interval=None, samples=[], skip_initial_validation=true) — nothing emerges
    in 50 steps and the real ic_gen validation samples reference real-corpus clips absent from the
    synthetic root. A real run MUST restore the inline ID+OOD+control triad (lora-train skill).
  * wandb disabled (self-contained smoke).

    python make_train_config.py --run <render_run_dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
MODEL = LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"
GEMMA = LAB / "cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized"
ARMS = REPO_ROOT / "eval_ladder/arms.yaml"
NEG = "worst quality, inconsistent motion, distorted, jittery"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="render_tuples run_dir (holds dataset/roots/synth_smoke)")
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    arms = yaml.safe_load(ARMS.read_text())
    run = Path(args.run).resolve()
    root = run / "dataset" / "roots" / "synth_smoke"
    out_train = REPO_ROOT / cfg["outputs"]["train_dir"] / "synth_smoke"
    tr = cfg["train"]

    video = {"is_generated": True, "latents_dir": "latents",
             "conditions": [
                 {"type": "reference", "latents_dir": "reference_latents", "probability": 1.0},
                 {"type": "mask", "mask_dir": "masks", "probability": 1.0}],
             "cond_clean_latents_dir": "cond_clean_latents"}

    config = {
        "model": {"model_path": str(MODEL), "text_encoder_path": str(GEMMA),
                  "training_mode": "lora",
                  "load_checkpoint": str(out_train / "checkpoints")},
        "lora": {"rank": tr["rank"], "alpha": tr["alpha"], "dropout": 0.0,
                 "target_modules": arms["targets"]["attn_ffn"]},
        "training_strategy": {"name": "flexible", "video": video},
        "optimization": {"learning_rate": float(tr["lr"]), "steps": int(tr["steps"]),
                         "batch_size": 1, "gradient_accumulation_steps": 1, "max_grad_norm": 1.0,
                         "optimizer_type": "adamw", "scheduler_type": "linear",
                         "scheduler_params": {}, "enable_gradient_checkpointing": True},
        "acceleration": {"mixed_precision_mode": "bf16", "quantization": None,
                         "load_text_encoder_in_8bit": False,
                         "offload_optimizer_during_validation": False},
        "data": {"preprocessed_data_root": str(root), "num_dataloader_workers": 2},
        # SMOKE: validation disabled (interval=None) — see module docstring.
        "validation": {"samples": [], "negative_prompt": NEG,
                       "video_dims": [cfg["inference"]["width"], cfg["inference"]["height"],
                                      cfg["inference"]["num_frames"]],
                       "frame_rate": 24.0, "seed": 42, "inference_steps": 30, "interval": None,
                       "guidance_scale": 4.0, "stg_scale": 1.0, "stg_blocks": [29],
                       "stg_mode": "stg_v", "generate_video": False, "generate_audio": False,
                       "skip_initial_validation": True},
        "checkpoints": {"interval": int(tr["steps"]), "keep_last_n": -1,
                        "precision": "bfloat16", "no_resume": False},
        "flow_matching": {"timestep_sampling_mode": "shifted_logit_normal",
                          "timestep_sampling_params": {}},
        "hub": {"push_to_hub": False, "hub_model_id": None},
        "wandb": {"enabled": False, "project": "exp077_synth_stratum", "entity": None,
                  "tags": ["ltx2", "exp077", "smoke", "ic_gen"], "log_validation_videos": False},
        "seed": 42,
        "output_dir": str(out_train),
    }

    cfg_dir = HERE / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "synth_smoke.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, width=10**6))
    print(f"[config] wrote {path}")
    print(f"         root={root}")
    print(f"         steps={config['optimization']['steps']} lr={config['optimization']['learning_rate']} "
          f"targets={len(config['lora']['target_modules'])} (attn_ffn) validation=DISABLED")


if __name__ == "__main__":
    main()
