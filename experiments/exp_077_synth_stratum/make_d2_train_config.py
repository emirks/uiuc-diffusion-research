"""exp_077 D2-FULL STAGE 5 (config only) — emit the D2 training config = the FROZEN recipe.

The DATA is the only variable. The recipe is taken from the AS-RUN training config of the baseline
generalist that produced the 72.9/72.8 cross baselines:
    outputs/training/ladder2/ic_gen/training_config.yaml
(NOT re-derived from make_configs.generalist_config(), which has drifted since the baseline ran.)

Overrides ONLY: data.preprocessed_data_root (-> the D2 combined root), output_dir /
model.load_checkpoint (-> exp_077/d2_gen), wandb tags/project (provenance). Everything else —
LoRA rank32/alpha32/dropout0 attn+FFN, lr 2e-4, 5000 steps, ckpt/500, flexible strategy with
reference + mask conditioning + cond_clean_latents, bf16, grad-ckpt, batch 1, seed 42,
shifted_logit_normal, and the inline ID/OOD/control validation triad — is kept EXACTLY as run.

THIS SCRIPT ONLY WRITES A CONFIG. The 5000-step train is HELD pending owner approval.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASELINE = REPO_ROOT / "outputs/training/ladder2/ic_gen/training_config.yaml"


def to_lists(x):
    if isinstance(x, (tuple, list)):
        return [to_lists(i) for i in x]
    if isinstance(x, dict):
        return {k: to_lists(v) for k, v in x.items()}
    return x


def main() -> None:
    cfg = yaml.safe_load((HERE / "config_d2full.yaml").read_text())
    tc = to_lists(yaml.unsafe_load(BASELINE.read_text()))

    combined_root = (REPO_ROOT / cfg["outputs"]["dir"] / cfg["outputs"]["subdir"]
                     / "dataset" / "roots" / "d2_combined")
    out_train = REPO_ROOT / cfg["outputs"]["train_dir"] / "d2_gen"

    tc["data"]["preprocessed_data_root"] = str(combined_root)
    tc["output_dir"] = str(out_train)
    tc["model"]["load_checkpoint"] = str(out_train / "checkpoints")
    tc.setdefault("wandb", {})
    tc["wandb"]["enabled"] = True
    tc["wandb"]["project"] = "exp077_synth_stratum"
    tc["wandb"]["tags"] = ["ltx2", "exp077", "d2", "ic_gen", "synth_stratum", "param_clamp"]

    # PATH FIX (not a recipe change): the baseline config predates eval_ladder's promotion out of
    # experiments/ladder2/, so its validation conditioning paths point at a moved directory.
    for s in tc["validation"]["samples"]:
        for cd in s.get("conditions", []):
            v = cd.get("video")
            if v and not Path(v).exists():
                cd["video"] = v.replace("/experiments/ladder2/", "/eval_ladder/")
            if cd.get("video"):
                assert Path(cd["video"]).exists(), f"validation conditioning missing: {cd['video']}"

    # frozen-knob seatbelts
    assert tc["lora"]["rank"] == 32 and tc["lora"]["alpha"] == 32 and tc["lora"]["dropout"] == 0.0
    assert len(tc["lora"]["target_modules"]) == 10, tc["lora"]["target_modules"]
    assert float(tc["optimization"]["learning_rate"]) == 2e-4
    assert tc["optimization"]["steps"] == cfg["train"]["steps"] == 5000
    assert tc["checkpoints"]["interval"] == cfg["train"]["ckpt_every"] == 500
    assert tc["seed"] == 42
    assert tc["flow_matching"]["timestep_sampling_mode"] == "shifted_logit_normal"
    assert tc["acceleration"]["mixed_precision_mode"] == "bf16"
    assert tc["optimization"]["enable_gradient_checkpointing"] is True
    assert tc["optimization"]["batch_size"] == 1
    assert "cond_clean_latents_dir" in tc["training_strategy"]["video"]
    assert [c["type"] for c in tc["training_strategy"]["video"]["conditions"]] == ["reference", "mask"]
    assert len(tc["validation"]["samples"]) == 4     # ID two-sided, ID one-sided, OOD, control

    cfg_dir = HERE / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "d2_gen.yaml"
    path.write_text(yaml.safe_dump(tc, sort_keys=False, width=10 ** 6))
    print(f"[config] wrote {path}")
    print(f"         root={combined_root}")
    print(f"         output_dir={out_train}")
    print(f"         steps={tc['optimization']['steps']} lr={float(tc['optimization']['learning_rate']):g} "
          f"rank={tc['lora']['rank']}/{tc['lora']['alpha']} targets={len(tc['lora']['target_modules'])} "
          f"ckpt/{tc['checkpoints']['interval']} val={len(tc['validation']['samples'])}"
          f"@{tc['validation']['interval']}")
    print("[config] TRAIN IS HELD — submit only on owner approval:")
    print(f"    sbatch {HERE / 'job_train_d2full.sbatch'}")


if __name__ == "__main__":
    main()
