"""exp_062 — Amendment 1: generate SIDE-KEYED per-class training configs.

Reads docs/eval_ladder/ladder_items_v2.json. For every class whose
`specialist_conditioning == "prefix_only"` (the 9 one_sided roster classes), emits
configs_keyed/<cls>.yaml identical to the blind config_for() EXCEPT:
  - conditions = [prefix tb=2 p=1.0] ONLY (suffix dropped)
  - output_dir -> outputs/training/exp_062_ladder_r2r3_specialists/<cls>_keyed
  - reuses the SAME manifest + the SAME .precomputed/<cls> latents (conditioning is
    applied at train time from config; latents are conditioning-agnostic —
    ltx2-conditioning-mechanics).
two_sided classes (prefix_suffix) are NOT re-emitted: their blind run already equals
the keyed recipe; a <cls>_keyed symlink to the blind dir is created by the sbatch/setup.

Usage: python experiments/exp_062_ladder_r2r3_specialists/build_datasets_keyed.py
"""
import json, pathlib
import yaml

from build_datasets import config_for, REPO_ROOT, EXP  # reuse the blind builder verbatim

GRID = REPO_ROOT / "docs/eval_ladder/ladder_items_v2.json"


def main() -> None:
    grid = json.loads(GRID.read_text())["classes"]
    (EXP / "configs_keyed").mkdir(parents=True, exist_ok=True)
    keyed = []
    for cls, g in grid.items():
        if g["specialist_conditioning"] != "prefix_only":
            continue  # two_sided: blind == keyed, no retrain
        cfg = config_for(cls)
        # drop the suffix condition -> prefix-only
        conds = cfg["training_strategy"]["video"]["conditions"]
        cfg["training_strategy"]["video"]["conditions"] = [
            c for c in conds if c["type"] == "prefix"
        ]
        assert len(cfg["training_strategy"]["video"]["conditions"]) == 1, cls
        # separate keyed output dir; keep precompute root (reuse) untouched
        cfg["output_dir"] = str(REPO_ROOT / "outputs/training/exp_062_ladder_r2r3_specialists" / f"{cls}_keyed")
        cfg["wandb"]["tags"] = cfg["wandb"]["tags"] + ["keyed", "amendment1"]
        cpath = EXP / "configs_keyed" / f"{cls}.yaml"
        cpath.write_text(yaml.safe_dump(cfg, sort_keys=False))
        keyed.append(cls)
        print(f"  {cls:22s} prefix-only -> {cpath.relative_to(REPO_ROOT)}")
    (EXP / "configs_keyed/index.json").write_text(json.dumps(sorted(keyed), indent=2))
    print(f"[done] {len(keyed)} keyed (prefix-only) configs: {sorted(keyed)}")


if __name__ == "__main__":
    main()
