#!/usr/bin/env python3
"""Materialize the dataset's clip mp4s — the SSOT holds files, not pointers.

The encode trees arrived in `outputs/ctt_v2/dataset/` carrying `clips/` directories whose
mp4s were SYMLINKS into the render staging dirs (`outputs/videos/ctt_v2_s1*`, `ctt_v2_s2*`).
That leaves the dataset non-self-contained: wipe the staging dirs and the dataset loses its
inspectable media. This script inverts each link: the target file is RENAMED into the
dataset (same filesystem — instant, no duplicate bytes) and a compat symlink is left at the
old staging path, the same pattern the store moves used, so viewers and recorded paths keep
resolving. Training files (latents/cond_clean/conditions/masks) are already real; this only
touches clips.

    python3 scripts/ctt_v2/materialize_clips.py            # dry run
    python3 scripts/ctt_v2/materialize_clips.py --execute
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "datasets/ctt_v2"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    dataset_real = os.path.realpath(DATASET)
    links = sorted(p for p in DATASET.rglob("*") if p.is_symlink())
    print(f"[clips] {len(links)} symlinks inside {DATASET}")

    plan, missing, inside = [], [], 0
    for link in links:
        target = os.path.realpath(link)
        if target.startswith(dataset_real + os.sep):
            inside += 1  # already points into the dataset; nothing to pull in
            continue
        if not os.path.isfile(target):
            missing.append((str(link), target))
            continue
        plan.append((link, target))

    if missing:
        for l, t in missing[:5]:
            print(f"[clips] BROKEN: {l} -> {t}", file=sys.stderr)
        raise SystemExit(f"[clips] {len(missing)} broken symlinks — fix before materializing")

    targets = [t for _, t in plan]
    dupes = len(targets) - len(set(targets))
    if dupes:
        raise SystemExit(f"[clips] {dupes} symlinks share a target — rename cannot satisfy both")

    print(f"[clips] plan: pull in {len(plan)} files by rename (+compat symlink at source); "
          f"{inside} already resolve inside the dataset")
    if not args.execute:
        print("[clips] DRY RUN — nothing changed")
        return

    for link, target in plan:
        link.unlink()                      # drop the dataset-side symlink
        os.rename(target, link)            # pull the real file in (same fs, instant)
        os.symlink(os.path.realpath(link), target)  # staging path keeps resolving

    leftover = [p for p in DATASET.rglob("*") if p.is_symlink()
                and not os.path.realpath(p).startswith(dataset_real + os.sep)]
    if leftover:
        raise SystemExit(f"[clips] {len(leftover)} outward links remain: {leftover[:5]}")
    print(f"[clips] done: {len(plan)} files materialized; dataset holds no outward symlinks")


if __name__ == "__main__":
    main()
