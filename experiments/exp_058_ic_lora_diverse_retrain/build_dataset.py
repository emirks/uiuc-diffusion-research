"""exp_058 — build the mixed-conditioning IC-LoRA pair dataset.

Fork of exp_056 build_dataset.py with three changes (design.md §2-3):

1. WHITELIST of training classes (transitions_std121 now also contains the
   HELD-OUT eval classes — discovery over the whole tree would leak them).
2. Clips used by exp_057 quads (endpoints or references) are excluded from
   training for classes with >=7 available clips, so the reused eval suite
   keeps unseen endpoints/demos.
3. Per-pair conditioning MASKS: masks/<K>.pt = {"mask": float32 [F,H,W]} over
   the latent grid — frames {0,1} = 1 always (prefix, 2 latent frames);
   frame {F-1} = 1 iff the pair's class is two-sided (suffix, 1 latent
   frame). Proven bit-exact vs exp_056 prefix(2)+suffix(1) by
   test_mask_conditioning.py. Mask geometry is derived from the target's
   precomputed latent shape at --link time (no hardcoded grid).

Modes:
  python build_dataset.py          # dataset_exp058.json + pairs.json
  python build_dataset.py --link   # symlink pair tree + real mask files
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
STD_ROOT = REPO_ROOT / "data/processed/transitions_std121"
CAPTIONS = EXP / "dataset/captions.json"
PAIRS = EXP / "dataset/pairs.json"
CLIPS_PRE = EXP / "dataset/.precomputed_clips"
PAIRS_PRE = EXP / "dataset/.precomputed"
QUADS_057 = REPO_ROOT / "experiments/exp_057_ic_lora_unseen_eval/dataset/quads.json"

MAX_REFS_PER_TARGET = 3
QUAD_EXCL_MIN_N = 7  # exclude quad-used clips only when the class can afford it

TWO_SIDED = {
    "air_bending", "display_transition", "earth_wave", "firelava", "flame",
    "flying_cam_transition", "melt_transition", "shadow_smoke", "water_bending",
}
ONE_SIDED = {
    "animalization", "fire_element", "giant_grab", "money_rain",
    "plasma_explosion", "portal", "shadow", "super_fast_run", "wireframe",
    "color_rain", "cotton_cloud", "earth_element", "live_concert",
    "luminous_gaze", "monstrosity", "mystification", "nature_bloom",
    "polygon", "saint_glow", "sakura_petals", "water_element", "wonderland",
    "run_set_on_fire",
}
TRAIN_CLASSES = TWO_SIDED | ONE_SIDED


def quad_used_clips() -> dict[str, set[str]]:
    used = defaultdict(set)
    for q in json.loads(QUADS_057.read_text()):
        used[q["endpoints_class"]].add(q["endpoints"])
        used[q["reference_class"]].add(q["reference"])
    return used


def discover_classes() -> dict[str, list[str]]:
    used = quad_used_clips()
    classes, dropped = {}, []
    for family in sorted(TRAIN_CLASSES):
        fam_dir = STD_ROOT / family
        stems = sorted(p.stem for p in fam_dir.glob("*.mp4"))
        if not stems:
            sys.exit(f"[error] no std clips for training class {family}")
        if len(stems) >= QUAD_EXCL_MIN_N and family in used:
            keep = [s for s in stems if s not in used[family]]
            dropped += [s for s in stems if s in used[family]]
            stems = keep
        classes[family] = stems
    print(f"[info] quad-clip training exclusions ({len(dropped)}): {dropped}")
    return classes


def make_pairs(classes: dict[str, list[str]]) -> list[dict]:
    pairs = []
    for family, stems in classes.items():
        n = len(stems)
        if n < 2:
            print(f"[pairs] {family}: n={n} -> excluded")
            continue
        k = min(MAX_REFS_PER_TARGET, n - 1)
        sided = "twosided" if family in TWO_SIDED else "onesided"
        for i, target in enumerate(stems):
            for j in range(1, k + 1):
                ref = stems[(i + j) % n]
                pairs.append({
                    "id": f"{family}/{target}__ref_{ref}",
                    "class": family,
                    "sidedness": sided,
                    "target": target,
                    "reference": ref,
                })
        print(f"[pairs] {family} ({sided}): n={n} -> {n * k} ordered pairs")
    return pairs


def build_manifests() -> None:
    captions = json.loads(CAPTIONS.read_text())
    classes = discover_classes()

    rows, missing = [], []
    for family, stems in classes.items():
        for stem in stems:
            if stem not in captions:
                missing.append(stem)
                continue
            rows.append({"caption": captions[stem], "video": f"{family}/{stem}.mp4"})
    if missing:
        sys.exit(f"[error] captions missing for: {missing}")
    out = STD_ROOT / "dataset_exp058.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"[done] {len(rows)} clip rows -> {out}")

    pairs = make_pairs(classes)
    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    PAIRS.write_text(json.dumps(pairs, indent=2))
    n_two = sum(1 for p in pairs if p["sidedness"] == "twosided")
    print(f"[done] {len(pairs)} pairs ({n_two} twosided / {len(pairs) - n_two} "
          f"onesided) -> {PAIRS}")


def link_pairs() -> None:
    import torch

    pairs = json.loads(PAIRS.read_text())
    lat_src = CLIPS_PRE / "latents"
    cond_src = CLIPS_PRE / "conditions"

    made, missing = 0, []
    mask_cache: dict[tuple, torch.Tensor] = {}
    for p in pairs:
        fam, tgt, ref = p["class"], p["target"], p["reference"]
        srcs = {
            "latents": lat_src / fam / f"{tgt}.pt",
            "conditions": cond_src / fam / f"{tgt}.pt",
            "reference_latents": lat_src / fam / f"{ref}.pt",
        }
        if any(not s.exists() for s in srcs.values()):
            missing.append(p["id"])
            continue
        for sub, src in srcs.items():
            dst = PAIRS_PRE / sub / fam / f"{tgt}__ref_{ref}.pt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())

        # mask: geometry from the target latent [C, F, H, W]
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

    if missing:
        sys.exit(f"[error] {len(missing)} pairs missing precomputed clips, e.g. {missing[:5]}")
    counts = {sub: sum(1 for _ in (PAIRS_PRE / sub).rglob("*.pt")) for sub in
              ("latents", "conditions", "reference_latents", "masks")}
    assert len(set(counts.values())) == 1, f"mismatched counts: {counts}"
    print(f"[done] {made} pairs linked -> {PAIRS_PRE} (counts: {counts}, "
          f"mask geoms: {sorted({k[:3] for k in mask_cache})})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", action="store_true")
    args = ap.parse_args()
    link_pairs() if args.link else build_manifests()
