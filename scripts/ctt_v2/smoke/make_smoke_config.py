"""CTT v2 — derive `mixed_smoke.yaml` from the certified `ic_gen.yaml` (never hand-written).

Only five things change, exactly as `REF_mixed_length.md` "The smallest test that settles it"
prescribes: the data root, the step count, validation off, checkpoints fresh, wandb off, and
`model.load_checkpoint` REMOVED so the run starts from the base model.  Everything that
defines the recipe — strategy block, LoRA targets, flow-matching sampler, optimizer, bf16 —
is copied through byte-for-byte so the gate tests the certified path and not a variant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
IC_GEN = LAB / "diffusion-research/eval_ladder/train/configs/ic_gen.yaml"


def main() -> int:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()

    cfg = yaml.safe_load(IC_GEN.read_text())
    cfg["model"].pop("load_checkpoint", None)           # start from base, no resume target
    cfg["data"]["preprocessed_data_root"] = a.root
    cfg["data"]["num_dataloader_workers"] = 0           # keep the index log in this process
    cfg["optimization"]["steps"] = a.steps
    cfg["optimization"]["batch_size"] = 1               # MANDATORY for mixed shapes
    cfg["validation"] = {"samples": [], "interval": None, "skip_initial_validation": True}
    cfg["checkpoints"] = {"interval": a.steps, "keep_last_n": 1, "precision": "bfloat16",
                          "no_resume": True}
    cfg["wandb"] = {"enabled": False}
    cfg["output_dir"] = a.output_dir

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[smoke-cfg] {out}  root={a.root} steps={a.steps} bs=1 wandb=off validation=off")
    return 0


if __name__ == "__main__":
    sys.exit(main())
