"""exp_062 — generate per-class training datasets + configs for the 11 R2/R3
specialist LoRAs (eval-ladder rungs R2/R3), per docs/eval_ladder/PLAN.md §2.

For each roster class:
  - dataset/manifests/<class>.json  = [{"caption": <type-blind>, "video": <abs clip>}]
    over the class's split-v1 TRAIN clips (captions: exp_058 captions.json + captions_r2.json)
  - configs/<class>.yaml             = exp_051 c2v recipe verbatim (rank32/a32, lr 1e-4,
    2000 steps, ckpt/250, conditions prefix tb=2 p=1.0 + suffix tb=1 p=1.0 — sidedness-BLIND),
    with data.preprocessed_data_root -> the per-class .precomputed dir this manifest feeds.

Trigger ICTRANS is prepended at PREPROCESS time (process_dataset --lora-trigger), so
manifest captions stay raw type-blind. Precompute -> per-class .precomputed/{latents,conditions}.
Usage: python experiments/exp_062_ladder_r2r3_specialists/build_datasets.py
"""
import json, pathlib
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
STD = REPO_ROOT / "data/processed/transitions_std121"
CAP_058 = REPO_ROOT / "experiments/exp_058_ic_lora_diverse_retrain/dataset/captions.json"
CAP_R2 = EXP / "dataset/captions_r2.json"
MODEL = "/projects/illinois/eng/cs/jrehg/users/emirkisa/cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"
GEMMA = "/projects/illinois/eng/cs/jrehg/users/emirkisa/cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized"

ROSTER = ["shadow", "portal", "super_fast_run", "shadow_smoke", "polygon",
          "wireframe", "animalization", "color_rain",
          "gas_transformation", "hero_flight", "illustration_scene"]


def stem(x):
    return x.split("/")[-1].replace(".mp4", "")


def config_for(cls: str) -> dict:
    pre_root = str(EXP / "dataset/.precomputed" / cls)
    out_dir = str(REPO_ROOT / "outputs/training/exp_062_ladder_r2r3_specialists" / cls)
    return {
        "model": {
            "model_path": MODEL,
            "text_encoder_path": GEMMA,
            "training_mode": "lora",
            "load_checkpoint": None,
        },
        "lora": {
            "rank": 32, "alpha": 32, "dropout": 0.0,
            "target_modules": [
                "attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
                "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0",
            ],
        },
        "training_strategy": {
            "name": "flexible",
            "video": {
                "is_generated": True,
                "latents_dir": "latents",
                "conditions": [
                    {"type": "prefix", "temporal_boundary": 2, "probability": 1.0},
                    {"type": "suffix", "temporal_boundary": 1, "probability": 1.0},
                ],
            },
        },
        "optimization": {
            "learning_rate": 1e-4, "steps": 2000, "batch_size": 1,
            "gradient_accumulation_steps": 1, "max_grad_norm": 1.0,
            "optimizer_type": "adamw", "scheduler_type": "linear",
            "scheduler_params": {}, "enable_gradient_checkpointing": True,
        },
        "acceleration": {
            "mixed_precision_mode": "bf16", "quantization": None,
            "load_text_encoder_in_8bit": False,
            "offload_optimizer_during_validation": False,
        },
        "data": {"preprocessed_data_root": pre_root, "num_dataloader_workers": 2},
        "validation": {
            # text-only final validation (no per-class cond videos needed); the
            # scored R2/R3 generations are produced separately by run_c2v_inference.
            "samples": [{"prompt": "ICTRANS A person stands in a room. The scene transforms into the same person outdoors in daylight."}],
            "negative_prompt": "worst quality, inconsistent motion, distorted, jittery",
            "video_dims": [480, 640, 121], "frame_rate": 24.0, "seed": 42,
            "inference_steps": 30, "interval": 2000, "guidance_scale": 4.0,
            "stg_scale": 1.0, "stg_blocks": [29], "stg_mode": "stg_v",
            "generate_audio": False, "skip_initial_validation": True,
        },
        "checkpoints": {"interval": 250, "keep_last_n": -1, "precision": "bfloat16"},
        "flow_matching": {"timestep_sampling_mode": "shifted_logit_normal", "timestep_sampling_params": {}},
        "hub": {"push_to_hub": False, "hub_model_id": None},
        "wandb": {"enabled": True, "project": "eval-ladder-r2r3", "entity": None,
                  "tags": ["ltx2", "lora", "c2v", "exp_062", cls], "log_validation_videos": True},
        "seed": 42,
        "output_dir": out_dir,
    }


def main() -> None:
    captions = json.loads(CAP_058.read_text())
    captions.update(json.loads(CAP_R2.read_text()))
    split = json.loads((STD / "split_v1.json").read_text())
    cls_split = split.get("classes", split)

    (EXP / "dataset/manifests").mkdir(parents=True, exist_ok=True)
    (EXP / "configs").mkdir(parents=True, exist_ok=True)

    index = {}
    for cls in ROSTER:
        train = [stem(t) for t in cls_split[cls].get("train", [])]
        rows, missing = [], []
        for s in train:
            if s not in captions:
                missing.append(s); continue
            rows.append({"caption": captions[s], "video": str(STD / cls / f"{s}.mp4")})
        if missing:
            raise SystemExit(f"[error] {cls}: captions missing for {missing}")
        mpath = EXP / "dataset/manifests" / f"{cls}.json"
        mpath.write_text(json.dumps(rows, indent=2))

        cpath = EXP / "configs" / f"{cls}.yaml"
        cpath.write_text(yaml.safe_dump(config_for(cls), sort_keys=False))
        index[cls] = {"n_train": len(rows), "manifest": str(mpath.relative_to(REPO_ROOT)),
                      "config": str(cpath.relative_to(REPO_ROOT))}
        print(f"  {cls:20s} {len(rows):2d} train clips -> manifest + config")

    (EXP / "dataset/index.json").write_text(json.dumps(index, indent=2))
    print(f"[done] {len(index)} classes; total train clips = {sum(v['n_train'] for v in index.values())}")


if __name__ == "__main__":
    main()
