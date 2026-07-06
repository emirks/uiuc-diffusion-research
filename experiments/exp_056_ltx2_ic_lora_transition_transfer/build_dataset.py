"""exp_056 — Step 2/3: build the IC-LoRA pair dataset.

Two modes:

  python build_dataset.py            # (login node, after captions.json exists)
      - writes data/processed/transitions_std121/dataset.json
        (clip-level: 47 rows {caption, video}, paths relative to that dir ->
        clean latent tree names <family>/<stem>.pt)
      - writes dataset/pairs.json (131 ordered same-class pairs, circulant
        scheme: target i takes the next min(3, n-1) clips of its class as
        references; every clip is target AND reference equally often;
        jump_transition (n=1) is excluded -> reserved as unseen-class
        validation reference)

  python build_dataset.py --link     # (after process_dataset.py has filled
                                     #  dataset/.precomputed_clips/)
      - assembles dataset/.precomputed/{latents,conditions,reference_latents}
        as SYMLINK trees over .precomputed_clips: each clip is VAE-encoded
        exactly once, each pair row is 3 symlinks:
            latents/<K>.pt            -> clips latents/<target>.pt
            conditions/<K>.pt         -> clips conditions/<target>.pt   (caption of the TARGET)
            reference_latents/<K>.pt  -> clips latents/<reference>.pt
        K = <family>/<target>__ref_<reference>
        (PrecomputedDataset matches sources by identical relative path and
        torch.load follows symlinks — verified against the official trainer.)

Rationale: process_dataset.py names every output after the row's TARGET video
path, so distinct pairs sharing a target would collide; symlink assembly
sidesteps that and avoids re-encoding clips O(pairs) times.
"""

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
STD_ROOT = REPO_ROOT / "data/processed/transitions_std121"
CAPTIONS = EXP / "dataset/captions.json"
PAIRS = EXP / "dataset/pairs.json"
CLIPS_PRE = EXP / "dataset/.precomputed_clips"
PAIRS_PRE = EXP / "dataset/.precomputed"

MAX_REFS_PER_TARGET = 3


def clip_key(family: str, stem: str) -> str:
    return f"{family}/{stem}"


def discover_classes() -> dict[str, list[str]]:
    classes = {}
    for fam_dir in sorted(p for p in STD_ROOT.iterdir() if p.is_dir()):
        stems = sorted(p.stem for p in fam_dir.glob("*.mp4"))
        if stems:
            classes[fam_dir.name] = stems
    return classes


def make_pairs(classes: dict[str, list[str]]) -> list[dict]:
    pairs = []
    for family, stems in classes.items():
        n = len(stems)
        if n < 2:
            print(f"[pairs] {family}: n={n} -> excluded (reserved as unseen reference class)")
            continue
        k = min(MAX_REFS_PER_TARGET, n - 1)
        for i, target in enumerate(stems):
            for j in range(1, k + 1):
                ref = stems[(i + j) % n]
                pairs.append({
                    "id": f"{family}/{target}__ref_{ref}",
                    "class": family,
                    "target": target,
                    "reference": ref,
                })
        print(f"[pairs] {family}: n={n} -> {n * k} ordered pairs")
    return pairs


def build_manifests() -> None:
    captions = json.loads(CAPTIONS.read_text())
    classes = discover_classes()

    # clip-level dataset.json for process_dataset.py
    rows, missing = [], []
    for family, stems in classes.items():
        for stem in stems:
            if stem not in captions:
                missing.append(stem)
                continue
            rows.append({"caption": captions[stem], "video": f"{family}/{stem}.mp4"})
    if missing:
        sys.exit(f"[error] captions missing for: {missing}")
    out = STD_ROOT / "dataset.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"[done] {len(rows)} clip rows -> {out}")

    pairs = make_pairs(classes)
    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    PAIRS.write_text(json.dumps(pairs, indent=2))
    print(f"[done] {len(pairs)} pairs -> {PAIRS}")


def link_pairs() -> None:
    pairs = json.loads(PAIRS.read_text())
    lat_src = CLIPS_PRE / "latents"
    cond_src = CLIPS_PRE / "conditions"

    made, missing = 0, []
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
        made += 1

    if missing:
        sys.exit(f"[error] {len(missing)} pairs missing precomputed clips, e.g. {missing[:5]}")
    counts = {sub: sum(1 for _ in (PAIRS_PRE / sub).rglob("*.pt")) for sub in
              ("latents", "conditions", "reference_latents")}
    assert len(set(counts.values())) == 1, f"mismatched counts: {counts}"
    print(f"[done] {made} pairs linked -> {PAIRS_PRE} (counts: {counts})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", action="store_true", help="assemble .precomputed pair symlink tree")
    args = ap.parse_args()
    link_pairs() if args.link else build_manifests()
