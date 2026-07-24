"""Build a private, fully-resolving IC-LoRA training root for the bottleneck campaign.

Why this exists: the shared root `eval_ladder/dataset/roots/ic_gen/` is **partially broken**. Its
`conditions/` and `cond_clean_latents/` trees are 100 % dangling symlinks — they point at
`experiments/ladder2/dataset/...`, a path that ceased to exist when ladder2 was promoted to
`eval_ladder/`. The real files are still on disk under `eval_ladder/dataset/{conditions,cond_clean}/`.

We rebuild rather than repair, for two reasons: the shared tree belongs to a parallel campaign that
is actively running (non-interference), and a private root lets us assert the invariants ourselves.

The pairing convention, read off the existing root:

    <source>/<class>/<target_stem>__ref_<reference_stem>.pt

    latents            -> exp_058 precomputed latents of the TARGET clip
    reference_latents  -> exp_058 precomputed latents of the REFERENCE clip
    masks              -> the real per-pair mask (conditioning window), reused as-is
    conditions         -> leak-free text embedding of the TARGET clip
    cond_clean_latents -> isolation-encoded clean anchors of the TARGET clip (the bleed fix)

THE GUARD THAT MATTERS: the trainer pairs data sources by relative path and **silently skips**
samples missing from any source. A silent drop is invisible in the loss curve, so this script
asserts all five sources carry an identical relative-path set before writing anything.

    python experiments/exp_078_operator_token_bottleneck/build_root.py [--check]
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_ROOT = REPO_ROOT / "eval_ladder" / "dataset" / "roots" / "ic_gen"
DATASET = REPO_ROOT / "eval_ladder" / "dataset"
OUT_ROOT = Path(__file__).resolve().parent / "dataset" / "roots" / "b1"

# source dir -> where its real files live, keyed by which clip stem the file belongs to.
# "target" = the pair's target clip, "reference" = the pair's demo clip, None = copy the existing link.
SOURCES = {
    "latents": ("passthrough", None),
    "reference_latents": ("passthrough", None),
    "masks": ("passthrough", None),
    "conditions": ("repair", DATASET / "conditions"),
    "cond_clean_latents": ("repair", DATASET / "cond_clean"),
}


def pair_stems(name: str) -> tuple[str, str]:
    """`air_bending_0__ref_air_bending_2.pt` -> ('air_bending_0', 'air_bending_2')."""
    stem = name[: -len(".pt")] if name.endswith(".pt") else name
    target, _, reference = stem.partition("__ref_")
    return target, reference


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only; write nothing")
    args = ap.parse_args()

    if not SHARED_ROOT.is_dir():
        print(f"[fatal] shared root not found: {SHARED_ROOT}", file=sys.stderr)
        return 1

    # The pair inventory comes from the source that is known-good (masks are real files).
    pairs = sorted(p.relative_to(SHARED_ROOT / "masks") for p in (SHARED_ROOT / "masks").rglob("*.pt"))
    print(f"[root] {len(pairs)} pairs in the shared inventory")

    plan: dict[str, dict[Path, Path]] = {}
    missing: list[str] = []

    for source, (mode, real_dir) in SOURCES.items():
        plan[source] = {}
        for rel in pairs:
            if mode == "passthrough":
                link = SHARED_ROOT / source / rel
                target = Path(os.readlink(link)) if link.is_symlink() else link.resolve()
            else:
                target_stem, _ = pair_stems(rel.name)
                target = real_dir / rel.parent / f"{target_stem}.pt"
            if not target.exists():
                missing.append(f"{source}/{rel} -> {target}")
            plan[source][rel] = target

    if missing:
        print(f"[fatal] {len(missing)} unresolvable sources, e.g.:", file=sys.stderr)
        for m in missing[:5]:
            print(f"   {m}", file=sys.stderr)
        return 1

    # SEATBELT: identical relative-path sets across all five sources, or the trainer drops silently.
    key_sets = {s: set(m) for s, m in plan.items()}
    reference_set = key_sets["latents"]
    for source, keys in key_sets.items():
        if keys != reference_set:
            only_here = sorted(keys - reference_set)[:3]
            only_there = sorted(reference_set - keys)[:3]
            print(
                f"[fatal] source '{source}' has a different relative-path set "
                f"({len(keys)} vs {len(reference_set)}); only-here={only_here} only-in-latents={only_there}",
                file=sys.stderr,
            )
            return 1
    print(f"[root] seatbelt PASS — all {len(SOURCES)} sources carry the identical {len(reference_set)}-path set")

    if args.check:
        print("[root] --check: nothing written")
        return 0

    written = 0
    for source, mapping in plan.items():
        for rel, target in mapping.items():
            link = OUT_ROOT / source / rel
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
            written += 1

    # Post-write verification: every link must resolve to a real file.
    unresolved = [p for p in OUT_ROOT.rglob("*.pt") if not p.resolve().is_file()]
    if unresolved:
        print(f"[fatal] {len(unresolved)} links do not resolve after write, e.g. {unresolved[:3]}", file=sys.stderr)
        return 1

    print(f"[root] wrote {written} links -> {OUT_ROOT}")
    for source in SOURCES:
        n = len(list((OUT_ROOT / source).rglob("*.pt")))
        print(f"   {source:20s} {n}")
    print("[root] all links resolve to real files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
