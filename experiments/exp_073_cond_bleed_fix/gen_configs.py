#!/usr/bin/env python
"""Generate the 9 exp_073 training configs by patching the verbatim originals.

Recipe-identical except the intended diffs:
  - data.preprocessed_data_root -> exp_073 shared root (fix & null share it)
  - fix arm ONLY: training_strategy.video.cond_clean_latents_dir = "cond_clean_latents"
  - output_dir -> exp_073 per-arm dir
  - wandb tags/project bookkeeping
Everything else (seed 42, lora, optimization, validation, flow_matching, ...) is untouched,
so fix vs nullA isolates the anchor change and nullA vs nullB (same seed) is the pure
GPU-nondeterminism floor.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP = REPO_ROOT / "experiments/exp_073_cond_bleed_fix"
OUT = EXP / "configs"
ROOTS = EXP / "dataset/roots"
TRAIN_OUT = REPO_ROOT / "outputs/training/exp_073_cond_bleed_fix"

SPEC_ORIG = REPO_ROOT / "experiments/exp_062_ladder_r2r3_specialists/configs"
IC3_ORIG = REPO_ROOT / "experiments/exp_064_ic3_aligned_retrain/config_ic3.yaml"

ARMS = ["fix", "nullA", "nullB"]  # nullA & nullB identical except output_dir


def _patch(cfg: dict, root: Path, name: str, arm: str, extra_tags: list[str]) -> dict:
    c = copy.deepcopy(cfg)
    c["data"]["preprocessed_data_root"] = str(root)
    if arm == "fix":
        c["training_strategy"]["video"]["cond_clean_latents_dir"] = "cond_clean_latents"
    else:
        c["training_strategy"]["video"].pop("cond_clean_latents_dir", None)
    c["output_dir"] = str(TRAIN_OUT / f"{name}_{arm}")
    c["model"]["load_checkpoint"] = None
    # Force a clean single-block run (no auto-resume) so preemption never introduces
    # resume-point RNG artifacts. fix & null get this identically -> mutual comparison clean.
    c.setdefault("checkpoints", {})
    c["checkpoints"]["no_resume"] = True
    c.setdefault("wandb", {})
    c["wandb"]["project"] = "cond-bleed-fix"
    c["wandb"]["tags"] = ["ltx2", "exp_073", "cond_bleed_fix", name, arm, *extra_tags]
    return c


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []

    # specialists (both two-sided): fix + nullA + nullB
    for cls in ("shadow_smoke", "hero_flight"):
        base = yaml.safe_load((SPEC_ORIG / f"{cls}.yaml").read_text())
        for arm in ARMS:
            c = _patch(base, ROOTS / cls, cls, arm, extra_tags=["c2v", "specialist"])
            p = OUT / f"{cls}_{arm}.yaml"
            p.write_text(yaml.safe_dump(c, sort_keys=False, default_flow_style=False, width=10_000))
            written.append(p.name)

    # ic3: fix + nullA + nullB
    base = yaml.safe_load(IC3_ORIG.read_text())
    for arm in ARMS:
        c = _patch(base, ROOTS / "ic3", "ic3", arm, extra_tags=["ic-lora", "v2v"])
        p = OUT / f"ic3_{arm}.yaml"
        p.write_text(yaml.safe_dump(c, sort_keys=False, default_flow_style=False, width=10_000))
        written.append(p.name)

    print(f"[gen_configs] wrote {len(written)} configs -> {OUT}")
    for w in written:
        print("  ", w)


if __name__ == "__main__":
    main()
