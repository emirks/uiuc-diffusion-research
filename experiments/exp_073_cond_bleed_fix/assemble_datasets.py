#!/usr/bin/env python
"""Assemble the exp_073 training data roots (pure filesystem; run AFTER --mode build).

Builds ONE preprocessed_data_root per FIX training arm, containing per-file symlinks so
PrecomputedDataset globs a file set identical to the original PLUS a cond_clean_latents
source. NULL arms point directly at the ORIGINAL exp_062/exp_064 precompute (unchanged) —
no new root needed. Per-file symlinks (not directory symlinks) so globbing is robust
regardless of pathlib symlink-follow behavior.

Both the fix and null arms of a model share ONE root (fix sets cond_clean_latents_dir,
null omits it) so the ONLY difference is the flag — identical data plumbing and file
globbing. cond_clean_latents is present but simply not globbed by the null.

Specialist root: exp_073/dataset/roots/<cls>/{conditions,latents,audio_latents,cond_clean_latents}
ic3 root:        exp_073/dataset/roots/ic3/{latents,conditions,reference_latents,masks,cond_clean_latents}
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP = REPO_ROOT / "experiments/exp_073_cond_bleed_fix"
ROOTS = EXP / "dataset/roots"

SP_062 = REPO_ROOT / "experiments/exp_062_ladder_r2r3_specialists/dataset/.precomputed"
IC_064_PAIRS_PRE = REPO_ROOT / "experiments/exp_064_ic3_aligned_retrain/dataset/.precomputed"
IC_064_PAIRS_JSON = REPO_ROOT / "experiments/exp_064_ic3_aligned_retrain/dataset/pairs.json"

SPECIALIST_TWOSIDED = ["shadow_smoke", "hero_flight"]


def _link_tree(src_dir: Path, dst_dir: Path) -> int:
    """Per-file symlink every *.pt under src_dir into dst_dir, preserving relative structure."""
    n = 0
    for src in sorted(src_dir.rglob("*.pt")):
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
        n += 1
    return n


def assemble_specialists() -> None:
    for cls in SPECIALIST_TWOSIDED:
        root = ROOTS / cls
        # original sources (symlinked file-by-file, identical to the null/original data)
        for sub in ("conditions", "latents", "audio_latents"):
            src = SP_062 / cls / sub
            if src.exists():
                k = _link_tree(src, root / sub)
                print(f"[spec] {cls}/{sub}: {k} files")
        # cond_clean produced by --mode build (mirrors latents' nested structure)
        cc_src = EXP / "dataset/specialist" / cls / "cond_clean_latents"
        k = _link_tree(cc_src, root / "cond_clean_latents")
        print(f"[spec] {cls}/cond_clean_latents: {k} files -> {root}")


def assemble_ic3() -> None:
    root = ROOTS / "ic3"
    # original pair-tree sources (symlink file-by-file)
    for sub in ("latents", "conditions", "reference_latents", "masks"):
        k = _link_tree(IC_064_PAIRS_PRE / sub, root / sub)
        print(f"[ic3] {sub}: {k} files")
    # cond_clean pair tree: per pair, cond_clean_latents/<fam>/<tgt>__ref_<ref>.pt -> clip cond_clean
    pairs = json.loads(IC_064_PAIRS_JSON.read_text())
    cc_clip = EXP / "dataset/ic3_clips/cond_clean"
    made = 0
    for p in pairs:
        fam, tgt, ref = p["class"], p["target"], p["reference"]
        src = cc_clip / fam / f"{tgt}.pt"
        if not src.exists():
            raise SystemExit(f"[ic3] missing cond_clean clip latent: {src} (run --mode build first)")
        dst = root / "cond_clean_latents" / fam / f"{tgt}__ref_{ref}.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
        made += 1
    print(f"[ic3] cond_clean_latents: {made} pair symlinks -> {root}")

    # sanity: matching counts across all fix sources
    counts = {sub: sum(1 for _ in (root / sub).rglob("*.pt"))
              for sub in ("latents", "conditions", "reference_latents", "masks", "cond_clean_latents")}
    print(f"[ic3] counts: {counts}")
    assert len(set(counts.values())) == 1, f"mismatched counts: {counts}"


if __name__ == "__main__":
    assemble_specialists()
    assemble_ic3()
    print("[assemble] done")
