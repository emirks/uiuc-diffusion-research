#!/usr/bin/env python3
"""Generate docs/eval_ladder/ladder_items_v1.json — the FROZEN item grid for the
eval ladder (rungs R2-R5), per PLAN.md §4. Deterministic; re-derivable against the
recorded split sha256.

Rules (PLAN.md §4, verbatim intent):
  - seeds 42/43/44 for every item at every rung (recorded here, applied at gen time)
  - R3/R4/R5 targets = the class's 2 split-v1 TEST clips (== exp_061 items)
  - selection RNG: random.Random(f"ladder_v1:{class}:{role}").sample(sorted(pool), k)
  - reference (k=1): R4 pool = train clips that WERE ic2-trained; R5 pool = all train
  - r2_items (k=2): pool = train clips minus the chosen reference
  - reference FIXED per class across both test items and all seeds
  - sidedness key: R4 from exp_058 training label; R5 from validated taxonomy label
Usage: python docs/eval_ladder/build_ladder_items.py
"""
import json, random, hashlib, pathlib, sys

try:
    import yaml
except ImportError:
    sys.exit("needs pyyaml (research env)")

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPLIT = ROOT / "data/processed/transitions_std121/split_v1.json"
PAIRS = ROOT / "experiments/exp_058_ic_lora_diverse_retrain/dataset/pairs.json"
AXES = ROOT / "outputs/taxonomy/class_axes.yaml"
OUT = ROOT / "docs/eval_ladder/ladder_items_v1.json"

SEEDS = [42, 43, 44]
R4_CLASSES = ["shadow", "portal", "super_fast_run", "shadow_smoke",
              "polygon", "wireframe", "animalization", "color_rain"]
R5_CLASSES = ["gas_transformation", "hero_flight", "illustration_scene"]
ROSTER = R4_CLASSES + R5_CLASSES
# exp_058 training sidedness (build_dataset.py TWO_SIDED set), restricted to R4 roster
TRAIN_TWO_SIDED = {"shadow_smoke"}


def stem(x: str) -> str:
    return x.split("/")[-1].replace(".mp4", "")


def main() -> None:
    split = json.loads(SPLIT.read_text())
    split_sha = hashlib.sha256(SPLIT.read_bytes()).hexdigest()
    cls = split.get("classes", split)

    # ic2-trained clip set (target + reference roles), from exp_058 pairs.json
    seen = set()
    for p in json.loads(PAIRS.read_text()):
        seen.add(f"{p['class']}/{p['target']}")
        seen.add(f"{p['class']}/{p['reference']}")

    axes = yaml.safe_load(AXES.read_text())["classes"]

    grid = {}
    for c in ROSTER:
        v = cls[c]
        train = sorted(stem(t) for t in v.get("train", []))
        test = [stem(t) for t in v.get("test", [])]
        assert len(test) == 2, f"{c}: expected 2 test clips, got {test}"

        if c in R4_CLASSES:
            pool_ref = sorted(t for t in train if f"{c}/{t}" in seen)
            assert pool_ref, f"{c}: no ic2-trained train clip for reference role"
        else:
            pool_ref = train[:]
        reference = random.Random(f"ladder_v1:{c}:reference").sample(pool_ref, 1)[0]

        pool_r2 = sorted(t for t in train if t != reference)
        r2_items = sorted(random.Random(f"ladder_v1:{c}:r2_items").sample(pool_r2, 2))

        if c in R4_CLASSES:
            key = "two_sided" if c in TRAIN_TWO_SIDED else "one_sided"
            key_source = "exp_058_training"
            gen_rung = "R4"
        else:
            sd = axes[c]["sidedness"]
            key = "two_sided" if sd == "two_sided" else "one_sided"
            key_source = "taxonomy_v1"
            gen_rung = "R5"

        grid[c] = {
            "rungs": ["R2", "R3", gen_rung],
            "test_items": test,                       # R3 + (R4|R5) targets
            "endpoint_seen_by_ic2": {t: (f"{c}/{t}" in seen) for t in test},
            "r2_items": r2_items,                     # R2 held-in targets
            "reference": reference,                   # fixed for R4/R5, all items+seeds
            "reference_ic2_trained": (f"{c}/{reference}" in seen),
            "sidedness_key": key,
            "sidedness_key_source": key_source,
            "suffix_conditioning": (key == "two_sided"),
            "generalist_rung": gen_rung,
            "n_train": len(train),
            "waits_on_sidedness_validation": (c == "hero_flight"),
        }

    doc = {
        "version": "ladder_items_v1",
        "frozen": True,
        "split_file": str(SPLIT.relative_to(ROOT)),
        "split_sha256": split_sha,
        "generalist_adapter": "outputs/training/exp_058_ic_lora_diverse_retrain/ic2/checkpoints/lora_weights_step_05000.safetensors",
        "seeds": SEEDS,
        "specialist_recipe": "exp_051 c2v verbatim (sidedness-BLIND: prefix tb=2 p=1.0 + suffix tb=1 p=1.0), type-blind ICTRANS captions",
        "rule": "random.Random(f'ladder_v1:{class}:{role}').sample(sorted(pool), k); reference k=1, r2_items k=2; ref pool = ic2-trained train clips (R4) or all train (R5); r2 pool = train minus reference",
        "n_clean_r4_items": sum(
            1 for c in R4_CLASSES for t, s in grid[c]["endpoint_seen_by_ic2"].items() if not s),
        "classes": grid,
    }
    OUT.write_text(json.dumps(doc, indent=2))
    # summary
    print(f"[done] {OUT}  (split_sha256={split_sha[:12]}…)")
    print(f"clean R4 items (endpoint unseen by ic2): {doc['n_clean_r4_items']}")
    for c in ROSTER:
        g = grid[c]
        seen_flags = ",".join(f"{t}:{'S' if s else 'U'}" for t, s in g["endpoint_seen_by_ic2"].items())
        wait = " [WAITS sidedness]" if g["waits_on_sidedness_validation"] else ""
        print(f"  {c:20s} {g['generalist_rung']} test=[{seen_flags}] r2={g['r2_items']} "
              f"ref={g['reference']}({'S' if g['reference_ic2_trained'] else 'U'}) "
              f"key={g['sidedness_key']}({g['sidedness_key_source']}){wait}")


if __name__ == "__main__":
    main()
