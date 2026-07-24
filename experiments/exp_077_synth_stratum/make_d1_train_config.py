"""exp_077 D1 STAGE 3 (config) — emit the D1 training config = the FROZEN generalist recipe.

The DATA is the only variable. The frozen recipe is taken from the AS-RUN training config of the
baseline generalist that produced the 72.9/72.8 cross baselines:
    outputs/training/ladder2/ic_gen/training_config.yaml
(NOT re-derived from make_configs.generalist_config() — that source has DRIFTED since the baseline
ran: its validation interval changed 1000->250 and its captions file path moved so it no longer
runs. Using the as-run config guarantees a byte-identical recipe, which is exactly what
"data is the only variable" requires.)

Overrides ONLY: data.preprocessed_data_root (-> D1 combined root), output_dir /
model.load_checkpoint (-> exp_077/d1_gen), wandb tags/project (provenance). Everything else —
LoRA rank/alpha/targets, lr, 5000 steps, ckpt/500, reference+mask+cond_clean conditioning,
bf16, grad-ckpt, batch 1, seed 42, shifted_logit_normal, and the inline ID/OOD/control
validation triad (interval 1000, skip_initial True) with its real-corpus conditioning paths —
is kept EXACTLY as the baseline ran it.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASELINE = REPO_ROOT / "outputs/training/ladder2/ic_gen/training_config.yaml"


def to_lists(x):
    if isinstance(x, tuple):
        return [to_lists(i) for i in x]
    if isinstance(x, list):
        return [to_lists(i) for i in x]
    if isinstance(x, dict):
        return {k: to_lists(v) for k, v in x.items()}
    return x


def main() -> None:
    cfg = yaml.safe_load((HERE / "config_d1.yaml").read_text())
    tc = to_lists(yaml.unsafe_load(BASELINE.read_text()))  # as-run frozen recipe

    combined_root = REPO_ROOT / cfg["outputs"]["dir"] / "d1" / "dataset" / "roots" / "d1_combined"
    out_train = REPO_ROOT / cfg["outputs"]["train_dir"] / "d1_gen"

    # override ONLY data + output paths + provenance (recipe otherwise byte-identical to baseline)
    tc["data"]["preprocessed_data_root"] = str(combined_root)
    tc["output_dir"] = str(out_train)
    tc["model"]["load_checkpoint"] = str(out_train / "checkpoints")
    tc.setdefault("wandb", {})
    tc["wandb"]["enabled"] = True
    tc["wandb"]["project"] = "exp077_synth_stratum"
    tc["wandb"]["tags"] = ["ltx2", "exp077", "d1", "ic_gen", "synth_stratum"]

    # PATH FIX (not a recipe change): the baseline config predates eval_ladder's promotion from
    # experiments/ladder2/ to the repo top level, so its validation prefix/suffix conditioning
    # paths point at the now-missing experiments/ladder2/conds/. Remap to the current location so
    # the inline validation actually runs; the SAMPLES (clips/prompts/conditioning) are unchanged.
    for s in tc["validation"]["samples"]:
        for cd in s.get("conditions", []):
            v = cd.get("video")
            if v and not Path(v).exists():
                fixed = v.replace("/experiments/ladder2/", "/eval_ladder/")
                cd["video"] = fixed
            if cd.get("video"):
                assert Path(cd["video"]).exists(), f"validation conditioning missing: {cd['video']}"

    # sanity: the frozen knobs the dossier pins (fail loudly if the baseline template is wrong)
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
    ctypes = [c["type"] for c in tc["training_strategy"]["video"]["conditions"]]
    assert ctypes == ["reference", "mask"], ctypes
    assert len(tc["validation"]["samples"]) == 4  # ID two-sided, ID one-sided, OOD zero-shot, control

    cfg_dir = HERE / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "d1_gen.yaml"
    path.write_text(yaml.safe_dump(tc, sort_keys=False, width=10 ** 6))
    print(f"[config] wrote {path}")
    print(f"         root={combined_root}")
    print(f"         output_dir={out_train}")
    print(f"         steps={tc['optimization']['steps']} lr={float(tc['optimization']['learning_rate']):g} "
          f"rank={tc['lora']['rank']} targets={len(tc['lora']['target_modules'])} "
          f"ckpt/{tc['checkpoints']['interval']} "
          f"val={len(tc['validation']['samples'])}@{tc['validation']['interval']} "
          f"skip_initial={tc['validation']['skip_initial_validation']}")


if __name__ == "__main__":
    main()
