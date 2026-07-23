"""ladder2 — emit the 12 training configs (11 specialists + 1 generalist).

One recipe per model kind, no hand-editing. Every config carries the MANDATORY inline
validation triad (`lora-train` skill): ID + OOD + control, every 250 steps, with conditioning
matched to the trained task — so the concept is watched as it emerges instead of being judged
after 2000 steps.

  ID       train-band clip of the trained class -> "is it learning?"
  OOD      train-band clip of a DIFFERENT class -> "does it generalize the effect, or has it
           memorised the training clips?"  (train-band, never test-band: eval endpoints stay
           untouched by anything the training loop renders)
  control  same prompt WITHOUT the token and WITHOUT conditioning -> "does the LoRA leave
           unrelated generation alone?"  This is also the token's own placebo test.

Prompts come from `prompts.render_prompt()` — the same renderer that produced the training
text embeddings and will produce every registry row, so what validation shows is exactly what
generation will get.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

import encode_conditioning as ec  # noqa: E402
import prompts  # noqa: E402

LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
MODEL = LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"
GEMMA = LAB / "cache/huggingface/gemma/gemma-3-12b-it-qat-q4_0-unquantized"
STD = REPO_ROOT / "data/processed/transitions_std121"
DATASET = HERE.parent / "dataset"
CONFIGS = HERE / "configs"
OUT_TRAIN = REPO_ROOT / "outputs/training/ladder2"
INVENTORY = HERE / "inventory.json"
ARMS = HERE.parent / "arms.yaml"

NEG = "worst quality, inconsistent motion, distorted, jittery"
#: OOD donor content — train-band clips of classes that are NOT specialists, so the same
#: two work for every specialist (one-sided / two-sided).
OOD_ONE = "mystification_0"
OOD_TWO = "air_bending_0"
#: generalist OOD: a held-out class (never trained) — a live zero-shot preview each interval
GEN_OOD_REF = ("saint_glow", "saint_glow_0", "saint_glow_1")  # class, endpoint clip, reference


def sample(prompt: str, endpoint: str | None, sided: str, reference: str | None = None) -> dict:
    """A ValidationSample dict: prompt + conditioning matched to the trained task."""
    conds = []
    if endpoint is not None:
        paths = ec.cond_paths(endpoint, sided)
        conds.append({"type": "prefix", "video": str(paths["prefix"]), "num_frames": ec.PX_PREFIX})
        if sided == "two":
            conds.append({"type": "suffix", "video": str(paths["suffix"]),
                          "num_frames": ec.SUFFIX_GEN_FRAMES})
    if reference is not None:
        cls = prompts.clip_class(reference)
        conds.append({"type": "reference", "video": str(STD / cls / f"{reference}.mp4"),
                      "downscale_factor": 1, "temporal_scale_factor": 1, "include_in_output": False})
    return {"prompt": prompt, "conditions": conds} if conds else {"prompt": prompt}


def base_config(name: str, targets: list[str], lr: float, steps: int, ckpt_every: int,
                root: Path, samples: list[dict], tags: list[str], cond_clean: bool,
                conditions: list[dict]) -> dict:
    video = {"is_generated": True, "latents_dir": "latents", "conditions": conditions}
    if cond_clean:
        video["cond_clean_latents_dir"] = "cond_clean_latents"
    return {
        # RESUME WIRING (load-bearing): `checkpoints.no_resume: false` alone does NOT resume —
        # trainer._load_checkpoint() returns immediately when model.load_checkpoint is null, so a
        # requeued job silently restarts at step 0. Pointing it at this run's own checkpoint dir
        # makes _find_checkpoint() pick the latest step_*.safetensors and _resolve_resume_state()
        # restore optimizer/scheduler/RNG/step from the training_state_step_*.pt beside it. On a
        # first run the directory is empty, _find_checkpoint returns None, and training starts at 0.
        "model": {"model_path": str(MODEL), "text_encoder_path": str(GEMMA),
                  "training_mode": "lora",
                  "load_checkpoint": str(OUT_TRAIN / name / "checkpoints")},
        "lora": {"rank": 32, "alpha": 32, "dropout": 0.0, "target_modules": targets},
        "training_strategy": {"name": "flexible", "video": video},
        "optimization": {"learning_rate": lr, "steps": steps, "batch_size": 1,
                         "gradient_accumulation_steps": 1, "max_grad_norm": 1.0,
                         "optimizer_type": "adamw", "scheduler_type": "linear",
                         "scheduler_params": {}, "enable_gradient_checkpointing": True},
        "acceleration": {"mixed_precision_mode": "bf16", "quantization": None,
                         "load_text_encoder_in_8bit": False,
                         "offload_optimizer_during_validation": False},
        "data": {"preprocessed_data_root": str(root), "num_dataloader_workers": 2},
        "validation": {"samples": samples, "negative_prompt": NEG,
                       "video_dims": [480, 640, 121], "frame_rate": 24.0, "seed": 42,
                       "inference_steps": 30, "interval": 250, "guidance_scale": 4.0,
                       "stg_scale": 1.0, "stg_blocks": [29], "stg_mode": "stg_v",
                       "generate_audio": False, "skip_initial_validation": False},
        # resume ENABLED: these run on preemptible `secondary`; a requeue must continue,
        # not restart. ckpt every 250 steps caps preemption loss at ~12 GPU-minutes.
        "checkpoints": {"interval": ckpt_every, "keep_last_n": -1, "precision": "bfloat16",
                        "no_resume": False},
        "flow_matching": {"timestep_sampling_mode": "shifted_logit_normal",
                          "timestep_sampling_params": {}},
        "hub": {"push_to_hub": False, "hub_model_id": None},
        "wandb": {"enabled": True, "project": "ladder2", "entity": None, "tags": tags,
                  "log_validation_videos": True},
        "seed": 42,
        "output_dir": str(OUT_TRAIN / name),
    }


def specialist_config(name: str, cls: str, clips: list[str], sided: str, token: str) -> dict:
    conditions = [{"type": "prefix", "temporal_boundary": 2, "probability": 1.0}]
    if sided == "two":
        conditions.append({"type": "suffix", "temporal_boundary": 1, "probability": 1.0})
    id_clip = clips[0]
    ood_clip = OOD_TWO if sided == "two" else OOD_ONE
    s1, _ = prompts.split_caption(id_clip)
    samples = [
        sample(prompts.render_prompt(id_clip, sided, token), id_clip, sided),      # ID
        sample(prompts.render_prompt(ood_clip, sided, token), ood_clip, sided),    # OOD
        sample(f"{s1}.", None, sided),                                             # control
    ]
    return base_config(
        name=name, targets=yaml.safe_load(ARMS.read_text())["targets"]["attn"],
        lr=1e-4, steps=2000, ckpt_every=250, root=DATASET / "roots" / name,
        samples=samples, tags=["ltx2", "ladder2", "specialist", cls, sided + "sided"],
        cond_clean=(sided == "two"), conditions=conditions)


def generalist_config(name: str, token: str) -> dict:
    conditions = [{"type": "reference", "latents_dir": "reference_latents", "probability": 1.0},
                  {"type": "mask", "mask_dir": "masks", "probability": 1.0}]
    arms = yaml.safe_load(ARMS.read_text())
    ood_cls, ood_clip, ood_ref = GEN_OOD_REF
    samples = [
        # ID: a trained pair (two-sided, exercises the corrected suffix anchor)
        sample(prompts.render_prompt("shadow_smoke_1", "two", token), "shadow_smoke_1", "two",
               reference="shadow_smoke_0"),
        # ID: a trained pair (one-sided)
        sample(prompts.render_prompt("color_rain_1", "one", token), "color_rain_1", "one",
               reference="color_rain_2"),
        # OOD: held-out class, never trained — a live zero-shot preview every interval
        sample(prompts.render_prompt(ood_clip, prompts.sidedness()[ood_cls], token), ood_clip,
               prompts.sidedness()[ood_cls], reference=ood_ref),
        # control: no token, no conditioning, no reference
        sample(prompts.split_caption("color_rain_1")[0] + ".", None, "one"),
    ]
    return base_config(
        name=name, targets=arms["targets"]["attn_ffn"], lr=2e-4, steps=5000, ckpt_every=500,
        root=DATASET / "roots" / name, samples=samples,
        tags=["ltx2", "ladder2", "generalist", "ic_lora"], cond_clean=True, conditions=conditions)


def main() -> None:
    inv = json.loads(INVENTORY.read_text())
    token = yaml.safe_load(ARMS.read_text())["token"]
    sided_of = prompts.sidedness()
    CONFIGS.mkdir(parents=True, exist_ok=True)
    written = []
    for name, meta in inv["models"].items():
        if meta["kind"] == "specialist":
            cls = meta["classes"][0]
            cfg = specialist_config(name, cls, inv["clips"][name], sided_of[cls], token)
        else:
            cfg = generalist_config(name, token)
        path = CONFIGS / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, width=10**6))
        written.append(name)
        n_val = len(cfg["validation"]["samples"])
        print(f"  {name:26s} steps={cfg['optimization']['steps']:5d} "
              f"lr={cfg['optimization']['learning_rate']:g} val={n_val} "
              f"cond_clean={'cond_clean_latents_dir' in cfg['training_strategy']['video']}")
    (CONFIGS / "index.json").write_text(json.dumps(sorted(written), indent=2))
    print(f"[configs] {len(written)} configs -> {CONFIGS.relative_to(REPO_ROOT)} (token={token!r})")


if __name__ == "__main__":
    main()
