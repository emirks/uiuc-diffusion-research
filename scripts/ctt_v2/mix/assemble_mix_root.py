"""ctt_v2 — assemble the S0+S2+S3 training root for the exp_081 masked retrain.

WHAT THIS SOLVES
----------------
The advisor ratified sampling-stream weights: **S0 = 15% of the stream**, S2/S3 filling the
remainder at their natural ratio. The trainer has no weighted sampler -- it builds a plain
``DataLoader(dataset, shuffle=True, drop_last=True)`` (trainer.py:712), i.e. uniform over the
sample list. So a stream weight can only be realised **on disk, by replication**: S0 samples
are linked in under several distinct relative paths so uniform sampling draws them more often.

The replication factor is COMPUTED FROM THE DELIVERED COUNTS, not hardcoded. S2/S3 pass through
quality gates, so their final counts are not knowable in advance; solving for the factor here
means the 15% holds whatever lands.

    want:  N0_slots / (N0_slots + N_synth) = w
    =>     N0_slots = w * N_synth / (1 - w)

With the planned 385 / 9,800 that gives 1,729 slots = 4 full copies of S0 + 189 extra, which is
why replication is expressed as "R full copies plus a deterministic remainder subset" rather
than a single integer factor (4x alone lands at 13.6%, 5x at 16.4%).

THE SILENT-DROP GUARD
---------------------
The trainer pairs its five data sources by IDENTICAL RELATIVE PATH and SILENTLY SKIPS any
sample missing from one of them (``datasets.py::_discover_samples`` only debug-logs it). So
every replica must be mirrored into all five trees, and the counts are hard-asserted equal at
the end. That assert is the whole safety story for this file.

Per-file symlinks, never directory symlinks, so globbing stays robust.

Usage:
    python scripts/ctt_v2/mix/assemble_mix_root.py --manifest <mix_manifest.json> \\
        --out eval_ladder/dataset/roots/ctt_v2_mix [--s0-weight 0.15] [--dry-run]

The manifest is ``{"strata": {"S0": [...], "S2": [...], "S3": [...]}}`` where each entry is
``{"sid": str, "src_root": str}`` and ``<src_root>/<source>/<sid>.pt`` exists for all five
sources. Encoding is a separate, earlier stage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

#: the five parallel trees the trainer joins by relative path
SOURCES = ("latents", "reference_latents", "cond_clean_latents", "masks", "conditions")

#: strata that get replicated to hit a stream weight (only the real corpus, by ruling)
WEIGHTED = "S0"


def link(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def solve_slots(n_synth: int, weight: float) -> int:
    """Slots the weighted stratum needs so it is `weight` of the uniform sampling stream."""
    if not 0.0 < weight < 1.0:
        raise ValueError(f"s0-weight must be in (0,1), got {weight}")
    return int(round(weight * n_synth / (1.0 - weight)))


def replica_names(sids: list[str], slots: int) -> list[tuple[str, str]]:
    """Expand `sids` to exactly `slots` (replica_dir, sid) pairs.

    R full copies, then a deterministic prefix of the sorted list for the remainder, so the
    layout is reproducible and reviewable rather than randomly sampled.
    """
    n = len(sids)
    if n == 0:
        raise ValueError("weighted stratum is empty")
    full, rem = divmod(slots, n)
    out = [(f"rep{r}", sid) for r in range(full) for sid in sids]
    out += [(f"rep{full}", sid) for sid in sids[:rem]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--s0-weight", type=float, default=0.15)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text())
    strata: dict[str, list[dict]] = man["strata"]
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root

    missing_stratum = [s for s in ("S0", "S2", "S3") if s not in strata or not strata[s]]
    if missing_stratum:
        sys.exit(f"[assemble] manifest is missing or empty for: {missing_stratum}")

    n_synth = sum(len(strata[s]) for s in strata if s != WEIGHTED)
    slots = solve_slots(n_synth, args.s0_weight)
    s0_sids = sorted(e["sid"] for e in strata[WEIGHTED])
    reps = replica_names(s0_sids, slots)
    realised = slots / (slots + n_synth)

    print(f"[assemble] delivered: " + "  ".join(f"{s}={len(strata[s])}" for s in sorted(strata)))
    print(f"[assemble] synthetic total  = {n_synth}")
    print(f"[assemble] {WEIGHTED} slots = {slots} "
          f"({math.floor(slots / len(s0_sids))} full copies + {slots % len(s0_sids)} extra)")
    print(f"[assemble] realised {WEIGHTED} stream share = {realised:.4f} (target {args.s0_weight})")
    for s in sorted(strata):
        if s != WEIGHTED:
            print(f"[assemble] realised {s} stream share = {len(strata[s]) / (slots + n_synth):.4f}")
    total = slots + n_synth
    print(f"[assemble] epoch size = {total} samples")

    if args.dry_run:
        print("[assemble] dry run — nothing written")
        return

    src_of = {e["sid"]: Path(e["src_root"]) for st in strata.values() for e in st}
    written = 0

    # unweighted strata: one link each, relative path = <stratum>/<sid>.pt
    for stratum, entries in sorted(strata.items()):
        if stratum == WEIGHTED:
            continue
        for e in entries:
            for source in SOURCES:
                link(Path(e["src_root"]) / source / f"{e['sid']}.pt",
                     out_root / source / stratum / f"{e['sid']}.pt")
            written += 1

    # weighted stratum: one link per replica, relative path = <stratum>_<repN>/<sid>.pt
    for rep, sid in reps:
        for source in SOURCES:
            link(src_of[sid] / source / f"{sid}.pt",
                 out_root / source / f"{WEIGHTED}_{rep}" / f"{sid}.pt")
        written += 1

    counts = {s: sum(1 for _ in (out_root / s).rglob("*.pt")) for s in SOURCES}
    if len(set(counts.values())) != 1:
        sys.exit(f"[assemble] FATAL: source counts disagree {counts} — the trainer would "
                 f"SILENTLY DROP the mismatched samples")
    if next(iter(counts.values())) != total:
        sys.exit(f"[assemble] FATAL: wrote {next(iter(counts.values()))} but planned {total}")

    dangling = [str(p) for s in SOURCES for p in (out_root / s).rglob("*.pt")
                if not p.resolve().is_file()]
    if dangling:
        sys.exit(f"[assemble] FATAL: {len(dangling)} dangling links, e.g. {dangling[:3]}")

    print(f"[assemble] OK — {written} samples x {len(SOURCES)} sources, counts {counts}")
    print(f"[assemble] root: {out_root}")
    (out_root / "MIX.json").write_text(json.dumps({
        "s0_weight_target": args.s0_weight,
        "s0_weight_realised": realised,
        "slots": {WEIGHTED: slots, **{s: len(strata[s]) for s in strata if s != WEIGHTED}},
        "epoch_size": total,
        "sources": list(SOURCES),
    }, indent=1))


if __name__ == "__main__":
    main()
