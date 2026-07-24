"""Deterministic pre-registered split for exp_079 SupCon-T. Writes split.json.

Per the advisor's LOCKED ruling (PRIMARY split, contamination-safe):

  * TRAIN            = the 26 held-in classes' 139 clips MINUS 1 held-out instance per class
                       (lexicographically-last clip of each class) -> 113 clips. Trained on the
                       4 TRAIN manipulations, composite (class, manipulation) SupCon labels.
  * HELDOUT_INSTANCE = the 26 held-out clips (1/class). Never trained; feed the class-separation
                       and instance-ID probes.
  * HELDOUT_CLASS    = the 45 zero-shot-class clips (10 zs classes). NEVER trained — encoded only
                       for the load-bearing temporal-generalization probe. Contamination-safe:
                       they pass through the frozen VAE for probe features only.

Held-in clips are taken from exp_078's encoder_manifest.json (the 139 reference clips, 0 held-out
among them). Zero-shot clips are globbed from the std121 corpus for the 10 zs classes named in the
same manifest. Everything is sorted; no randomness.

    python experiments/exp_079_standalone_operator_encoder/build_split.py
"""

import json
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent
REPO_ROOT = EXP.parents[1]
MANIFEST = REPO_ROOT / "experiments/exp_078_operator_token_bottleneck/encoder_manifest.json"
STD = REPO_ROOT / "data/processed/transitions_std121"
OUT = EXP / "split.json"


def mp4_for(clip: str) -> str:
    hits = sorted(STD.glob(f"*/{clip}.mp4"))
    if not hits:
        raise FileNotFoundError(f"no std121 mp4 for clip {clip}")
    return str(hits[0].relative_to(REPO_ROOT))


def main() -> None:
    man = json.loads(MANIFEST.read_text())
    zs_classes = sorted(man["heldout_classes"])          # the 10 zero-shot classes
    heldin = man["clips"]                                  # 139 held-in reference clips

    # group held-in clips by class, sorted
    by_cls: dict[str, list[str]] = defaultdict(list)
    for c in heldin:
        assert c["cls"] not in zs_classes, f"CONTAMINATION: held-in clip {c['clip']} is a zs class"
        by_cls[c["cls"]].append(c["clip"])
    for cls in by_cls:
        by_cls[cls] = sorted(by_cls[cls])

    train, heldout_instance = [], []
    for cls in sorted(by_cls):
        clips = by_cls[cls]
        # hold out the lexicographically-last clip of each class as the instance-level probe clip
        *tr, held = clips
        for clip in tr:
            train.append({"clip": clip, "cls": cls, "mp4": mp4_for(clip)})
        heldout_instance.append({"clip": held, "cls": cls, "mp4": mp4_for(held)})

    # zero-shot-class clips (held-out CLASS probe set) — enumerate from the corpus
    heldout_class = []
    for cls in zs_classes:
        for mp4 in sorted(STD.glob(f"{cls}/*.mp4")):
            heldout_class.append({"clip": mp4.stem, "cls": cls, "mp4": str(mp4.relative_to(REPO_ROOT))})

    split = {
        "note": "exp_079 SupCon-T PRIMARY split (advisor-locked). zs clips are probe-only, never trained.",
        "train_manips": ["identity", "reverse", "ease_in_g2", "ease_out_g05"],
        "heldout_manips": ["warp_g3", "warp_g033", "warp_g15", "warp_g067"],
        "n_train_clips": len(train),
        "n_heldout_instance_clips": len(heldout_instance),
        "n_heldout_class_clips": len(heldout_class),
        "n_train_classes": len(by_cls),
        "n_heldout_classes": len(zs_classes),
        "heldout_classes": zs_classes,
        "train": train,
        "heldout_instance": heldout_instance,
        "heldout_class": heldout_class,
    }
    OUT.write_text(json.dumps(split, indent=1))

    # seatbelt: clip-disjointness across the three sets
    s_tr = {c["clip"] for c in train}
    s_hi = {c["clip"] for c in heldout_instance}
    s_hc = {c["clip"] for c in heldout_class}
    assert not (s_tr & s_hi) and not (s_tr & s_hc) and not (s_hi & s_hc), "split sets overlap!"
    print(f"[split] train={len(train)} clips ({len(by_cls)} classes)  "
          f"heldout_instance={len(heldout_instance)}  heldout_class={len(heldout_class)} "
          f"({len(zs_classes)} zs classes)")
    print(f"[split] clip-disjoint PASS; wrote {OUT.relative_to(REPO_ROOT)}")
    # per-train-class train-clip counts (SupCon positive availability)
    tc = defaultdict(int)
    for c in train:
        tc[c["cls"]] += 1
    singletons = sorted(k for k, v in tc.items() if v == 1)
    print(f"[split] train classes with only 1 clip (rely on augmented-view positives): "
          f"{len(singletons)} -> {singletons}")


if __name__ == "__main__":
    main()
