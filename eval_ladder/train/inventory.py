"""ladder2 — training-data inventory: what each model trains on, and what is missing.

Runs before any GPU work. Two jobs:

1. Derive the 12 training rosters from the FROZEN split (never from a hand-kept list):
   11 specialists (train band of each roster class) + 1 generalist (train band of the 29
   held-in classes, expanded into reference/target pairs).
2. Diff those rosters against the reusable video-latent precompute and hard-fail on gaps,
   emitting `missing_videos.json` for `process_videos.py`.

Text embeddings are deliberately NOT inventoried: ladder2 renders new leak-free prompts, so
EVERY conditions/ file is regenerated. Reusing an old one would silently train on the leaky
caption (advisor defect #3).

    python eval_ladder/train/inventory.py [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

import prompts  # noqa: E402

STD = REPO_ROOT / "data/processed/transitions_std121"
SPLIT_PATH = STD / "split_v1.2.json"
SPLIT_SHA = "c694659d6d2e264528ccb546b43b9974bfecd2770ab674ba7b514981d026e6ce"

#: reusable video-latent sources, searched in order (latents are prompt-agnostic)
LATENT_SOURCES = (
    REPO_ROOT / "experiments/exp_064_ic3_aligned_retrain/dataset/.precomputed_clips/latents",
    REPO_ROOT / "experiments/exp_058_ic_lora_diverse_retrain/dataset/.precomputed_clips/latents",
    REPO_ROOT / "experiments/exp_062_ladder_r2r3_specialists/dataset/.precomputed",
)
MAX_REFS_PER_TARGET = 3
OUT = HERE / "inventory.json"
MISSING = HERE / "missing_videos.json"


def load_split() -> dict:
    split = json.loads(SPLIT_PATH.read_text())
    assert split["split"] == "v1.2", split.get("split")
    assert split["sha256"] == SPLIT_SHA, "split_v1.2.json changed — the ladder is not reproducible"
    return split


def find_latents(cls: str, clip: str) -> Path | None:
    """Locate an existing video-latent .pt for a clip, whatever tree it lives in."""
    for src in LATENT_SOURCES:
        direct = src / cls / f"{clip}.pt"
        if direct.exists():
            return direct
        nested = list((src / cls / "latents").rglob(f"{clip}.pt")) if (src / cls).exists() else []
        if nested:
            return nested[0]
    return None


def make_pairs(rosters: dict[str, list[str]], sided: dict[str, str]) -> list[dict]:
    """Ring-offset reference/target pairs, exp_058/064 semantics (validated)."""
    pairs = []
    for family, stems in sorted(rosters.items()):
        n = len(stems)
        if n < 2:
            print(f"[pairs] {family}: n={n} -> excluded (needs >=2 clips)")
            continue
        k = min(MAX_REFS_PER_TARGET, n - 1)
        for i, target in enumerate(stems):
            for j in range(1, k + 1):
                ref = stems[(i + j) % n]
                pairs.append({
                    "id": f"{family}/{target}__ref_{ref}",
                    "class": family,
                    "sidedness": "twosided" if sided[family] == "two" else "onesided",
                    "target": target,
                    "reference": ref,
                })
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="print the inventory as JSON")
    args = ap.parse_args()

    split = load_split()
    sided = prompts.sidedness()
    quarantined = set(split["quarantined"])
    held_out = set(split["generalist_holdout"])
    roster = split["specialist_roster"]

    def train_band(cls: str) -> list[str]:
        clips = [c for c in sorted(split["classes"][cls]["train"]) if c not in quarantined]
        # a clip with no caption cannot be rendered into a prompt -> cannot be trained on
        return [c for c in clips if c in prompts.captions()]

    models: dict[str, dict] = {}
    for cls in roster:
        clips = train_band(cls)
        assert clips, f"specialist {cls} has no trainable clips"
        models[f"spec_{cls}"] = {
            "kind": "specialist", "classes": [cls], "sided": sided[cls], "clips": clips,
        }

    held_in = [c for c in sorted(split["classes"]) if c not in held_out]
    gen_rosters = {c: train_band(c) for c in held_in}
    dropped = {c: v for c, v in gen_rosters.items() if len(v) < 2}
    gen_rosters = {c: v for c, v in gen_rosters.items() if len(v) >= 2}
    pairs = make_pairs(gen_rosters, sided)
    models["ic_gen"] = {
        "kind": "generalist",
        "classes": sorted(gen_rosters),
        "sided": "mixed",
        "clips": sorted({c for v in gen_rosters.values() for c in v}),
        "n_pairs": len(pairs),
    }

    # --- coverage diff -----------------------------------------------------------------
    all_clips = sorted({c for m in models.values() for c in m["clips"]})
    have, missing = {}, []
    for clip in all_clips:
        cls = prompts.clip_class(clip)
        found = find_latents(cls, clip)
        (have.__setitem__(clip, str(found.relative_to(REPO_ROOT))) if found
         else missing.append((cls, clip)))

    rows = [{"caption": prompts.captions()[c], "video": f"{cls}/{c}.mp4"} for cls, c in missing]
    MISSING.write_text(json.dumps(rows, indent=2))

    inv = {
        "split": "v1.2", "split_sha256": SPLIT_SHA,
        "n_models": len(models), "n_clips": len(all_clips),
        "n_have_latents": len(have), "n_missing_latents": len(missing),
        "held_in": held_in, "held_out": sorted(held_out),
        "generalist_classes_dropped": dropped,
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "clips"} | {"n_clips": len(v["clips"])}
                   for k, v in models.items()},
        "clips": {k: v["clips"] for k, v in models.items()},
        "pairs": pairs,
        "latents": have,
        "missing": [f"{c}/{k}" for c, k in missing],
    }
    OUT.write_text(json.dumps(inv, indent=2))

    if args.json:
        print(json.dumps(inv["models"], indent=2))
    print(f"[inventory] {len(models)} models, {len(all_clips)} distinct training clips")
    for name, m in models.items():
        extra = f", {m['n_pairs']} pairs" if "n_pairs" in m else ""
        print(f"  {name:26s} {m['kind']:11s} {m['sided']:5s} {len(m['clips']):3d} clips{extra}")
    if dropped:
        print(f"[inventory] generalist classes dropped (<2 trainable clips): {dropped}")
    print(f"[inventory] video latents: {len(have)} reusable, {len(missing)} MISSING")
    if missing:
        print(f"            -> {MISSING.relative_to(REPO_ROOT)}: {[f'{c}/{k}' for c, k in missing]}")
    print(f"[inventory] wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
