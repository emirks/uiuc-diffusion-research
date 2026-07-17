"""exp_064 — ic3: split-aligned IC-LoRA generalist retrain dataset.

Fork of exp_058 build_dataset.py per PLAN Amendment 2 SSA2.4. Three changes:

1. CLIP SOURCE = split_v1.1 TRAIN BAND ONLY (the alignment that ic2 lacked —
   no quad-exclusion heuristics; the split IS the exclusion rule now).
2. SIDEDNESS = owner-final taxonomy (corpus_manifest.json, validated
   2026-07-16). vs exp_058 this flips exactly one trained class:
   giant_grab onesided -> twosided.
3. CLIPS_PRE reuse: clip latents/conditions already precomputed by exp_058
   are symlinked; only genuinely new clips go into the missing-precompute
   manifest (dataset_exp064_missing.json, written to the std root so
   process_dataset.py resolves videos identically).

Same 32 training classes as exp_058 (holdout verbatim: hero_flight,
illustration_scene, gas_transformation, raven_transition, hole_transition,
seamless_transition, jump_transition). Same pair rule (MAX_REFS_PER_TARGET=3,
ring offsets), same mask semantics ([0,1]=1 always; [-1]=1 iff twosided).

Modes:
  python build_dataset.py          # manifests + pairs.json + missing list
  python build_dataset.py --link   # symlink clip tree + pair tree + masks
"""

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
STD_ROOT = REPO_ROOT / "data/processed/transitions_std121"
SPLIT = STD_ROOT / "split_v1.1.json"
MANIFEST_CORPUS = STD_ROOT / "corpus_manifest.json"
PAIRS = EXP / "dataset/pairs.json"
MISSING_MANIFEST = STD_ROOT / "dataset_exp064_missing.json"
CLIPS_PRE = EXP / "dataset/.precomputed_clips"
PAIRS_PRE = EXP / "dataset/.precomputed"
PRE_058 = REPO_ROOT / "experiments/exp_058_ic_lora_diverse_retrain/dataset/.precomputed_clips"

MAX_REFS_PER_TARGET = 3
HOLDOUT = {
    "hero_flight", "illustration_scene", "gas_transformation",
    "raven_transition", "hole_transition", "seamless_transition",
    "jump_transition",
}
CAPTION_SOURCES = [
    REPO_ROOT / "experiments/exp_058_ic_lora_diverse_retrain/dataset/captions.json",
    REPO_ROOT / "experiments/exp_060_sigma_seed/dataset/captions_extra.json",
    REPO_ROOT / "experiments/exp_061_ladder_r0_r1/dataset/captions_extra.json",
]


def load_captions() -> dict:
    caps = {}
    for src in CAPTION_SOURCES:
        if src.exists():
            caps.update(json.loads(src.read_text()))
    return caps


def load_classes() -> tuple[dict[str, list[str]], set[str]]:
    split = json.loads(SPLIT.read_text())
    assert split["split"] == "v1.1", split.get("split")
    corpus = json.loads(MANIFEST_CORPUS.read_text())["classes"]
    classes = {}
    for cls, rec in sorted(split["classes"].items()):
        if cls in HOLDOUT:
            continue
        classes[cls] = sorted(rec["train"])          # TRAIN BAND ONLY
    assert len(classes) == 32, len(classes)
    two_sided = {c for c in classes if corpus[c]["sidedness"] == "twosided"}
    # owner-final keying: exp_058's 9 + giant_grab
    assert two_sided == {
        "air_bending", "display_transition", "earth_wave", "firelava", "flame",
        "flying_cam_transition", "giant_grab", "melt_transition",
        "shadow_smoke", "water_bending",
    }, two_sided
    return classes, two_sided


def make_pairs(classes, two_sided) -> list[dict]:
    pairs = []
    for family, stems in classes.items():
        n = len(stems)
        if n < 2:
            print(f"[pairs] {family}: n={n} -> excluded")
            continue
        k = min(MAX_REFS_PER_TARGET, n - 1)
        sided = "twosided" if family in two_sided else "onesided"
        for i, target in enumerate(stems):
            for j in range(1, k + 1):
                ref = stems[(i + j) % n]
                pairs.append({
                    "id": f"{family}/{target}__ref_{ref}",
                    "class": family, "sidedness": sided,
                    "target": target, "reference": ref,
                })
        print(f"[pairs] {family} ({sided}): n={n} -> {n * k} pairs")
    return pairs


def build_manifests() -> None:
    captions = load_captions()
    classes, two_sided = load_classes()

    missing_caps, missing_pre, rows_missing = [], [], []
    for family, stems in classes.items():
        for stem in stems:
            if stem not in captions:
                missing_caps.append(stem)
                continue
            if not (PRE_058 / "latents" / family / f"{stem}.pt").exists():
                missing_pre.append(f"{family}/{stem}")
                rows_missing.append({"caption": captions[stem],
                                     "video": f"{family}/{stem}.mp4"})
    if missing_caps:
        sys.exit(f"[error] captions missing for {len(missing_caps)}: {missing_caps}")

    MISSING_MANIFEST.write_text(json.dumps(rows_missing, indent=2))
    print(f"[done] {len(missing_pre)} clips need fresh precompute -> "
          f"{MISSING_MANIFEST}: {missing_pre}")

    pairs = make_pairs(classes, two_sided)
    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    PAIRS.write_text(json.dumps(pairs, indent=2))
    n_clips = sum(len(s) for s in classes.values())
    n_two = sum(1 for p in pairs if p["sidedness"] == "twosided")
    print(f"[done] {n_clips} clips / 32 classes -> {len(pairs)} pairs "
          f"({n_two} twosided / {len(pairs) - n_two} onesided) -> {PAIRS}")


def link_pairs() -> None:
    import torch

    classes, _ = load_classes()
    # 1. clip tree: symlink from exp_058 precompute or exp_064's own
    for family, stems in classes.items():
        for stem in stems:
            for sub in ("latents", "conditions"):
                own = CLIPS_PRE / sub / family / f"{stem}.pt"
                if own.exists() and not own.is_symlink():
                    continue                     # freshly precomputed here
                src = PRE_058 / sub / family / f"{stem}.pt"
                if not src.exists():
                    sys.exit(f"[error] no precompute anywhere for {family}/{stem} ({sub})")
                own.parent.mkdir(parents=True, exist_ok=True)
                if own.is_symlink():
                    own.unlink()
                own.symlink_to(src.resolve())

    # 2. pair tree + masks (identical semantics to exp_058)
    pairs = json.loads(PAIRS.read_text())
    made = 0
    mask_cache: dict[tuple, torch.Tensor] = {}
    for p in pairs:
        fam, tgt, ref = p["class"], p["target"], p["reference"]
        srcs = {
            "latents": CLIPS_PRE / "latents" / fam / f"{tgt}.pt",
            "conditions": CLIPS_PRE / "conditions" / fam / f"{tgt}.pt",
            "reference_latents": CLIPS_PRE / "latents" / fam / f"{ref}.pt",
        }
        for sub, src in srcs.items():
            dst = PAIRS_PRE / sub / fam / f"{tgt}__ref_{ref}.pt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
        tdata = torch.load(srcs["latents"], map_location="cpu", weights_only=True)
        f, h, w = int(tdata["num_frames"]), int(tdata["height"]), int(tdata["width"])
        key = (f, h, w, p["sidedness"])
        if key not in mask_cache:
            m = torch.zeros(f, h, w)
            m[:2] = 1.0
            if p["sidedness"] == "twosided":
                m[-1] = 1.0
            mask_cache[key] = m
        mdst = PAIRS_PRE / "masks" / fam / f"{tgt}__ref_{ref}.pt"
        mdst.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mask": mask_cache[key].clone()}, mdst)
        made += 1

    counts = {sub: sum(1 for _ in (PAIRS_PRE / sub).rglob("*.pt")) for sub in
              ("latents", "conditions", "reference_latents", "masks")}
    assert len(set(counts.values())) == 1, f"mismatched counts: {counts}"
    print(f"[done] {made} pairs linked -> {PAIRS_PRE} (counts: {counts})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", action="store_true")
    args = ap.parse_args()
    link_pairs() if args.link else build_manifests()
