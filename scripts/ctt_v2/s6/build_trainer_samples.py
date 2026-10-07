#!/usr/bin/env python3
"""Convert a physical sampler-mix root (assemble_root --sampler-mix output) into the
trainer-facing form: `samples.jsonl` (lowercase, one row per base pair with an `id` and a
resolved `paths` object) + `mix.json` (the per-stratum weights the StratifiedEpochSampler
realizes). No data is moved — the physical root's symlink trees already reference the sources.

Why this exists instead of build_dataset.py: build_dataset MOVES `outputs/ctt_v2/encodes` into
the dataset dir, but 002 already moved it (it is a compat symlink now), so build_dataset cannot
build an ADDITIVE dataset that shares S0-S4 with 002. The sampler-mix physical root references
every source via a symlink (no move), so it is the safe additive form; this script just emits the
list-based files the trainer reads on top of it.

    python scripts/ctt_v2/s6/build_trainer_samples.py --root outputs/ctt_v2/roots/ctt_v2plus_mix \
        --contract 003_ctt_v2plus
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/ctt_v2"))
import root_common as rc  # noqa: E402

TREES = ["latents", "reference_latents", "cond_clean_latents", "conditions", "masks"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--contract", default="003_ctt_v2plus")
    args = ap.parse_args()
    R = Path(args.root)

    man = json.loads((R / "ROOT_MANIFEST.json").read_text())
    # samples.jsonl (paths derived from each SAMPLES.jsonl row's `rel`)
    n = 0
    with (R / "samples.jsonl").open("w") as out:
        for ln in (R / "SAMPLES.jsonl").read_text().splitlines():
            r = json.loads(ln)
            rel = r["rel"]
            row = {
                "id": f"{r['stratum']}/{r['group_slug']}/{r['target']}__ref_{r['reference']}",
                "stratum": r["stratum"], "group": r["group"], "group_slug": r["group_slug"],
                "target": r["target"], "reference": r["reference"], "sided": r["sided"],
                "caption_key": r["caption_key"], "shape": r["shape"],
                "paths": {t: f"{t}/{rel}" for t in TREES},
                "endpoints": r["endpoints"], "caption_sources": r["caption_sources"],
            }
            out.write(json.dumps(row) + "\n")
            n += 1

    w = man["weights"]
    mix = {
        "schema": "ctt_v2plus_mix/v1",
        "contract": args.contract,
        "stratum_weights_pct": w["intended_pct"],           # sampler per-stratum targets (S6=20)
        "mix_contract_aggregate": rc.mix_contract(args.contract)["weights"],
        "weight_note": w.get("note"),
        "prorata_split": w.get("prorata_split"),
        "realized_row_pct": w.get("realized_pct"),
        "strata_present": man["strata_present"], "strata_absent": man["strata_absent"],
        "authority": "root_common.MIX_CONTRACTS; sampler realizes intended_pct via "
                     "StratifiedEpochSampler (the mix is a config knob, not baked into the data).",
    }
    (R / "mix.json").write_text(json.dumps(mix, indent=1))
    (R / "VERSION").write_text("3.0.0-ctt_v2plus\n")
    if not (R / "captions.json").exists():
        (R / "captions.json").symlink_to("CAPTIONS.json")
    print(f"[ok] samples.jsonl {n} rows | mix.json {json.dumps(w['intended_pct'])}")


if __name__ == "__main__":
    main()
