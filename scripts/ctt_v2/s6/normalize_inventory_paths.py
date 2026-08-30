#!/usr/bin/env python3
"""Normalize the pre-existing S0/S2a/S2b/S4 inventory source paths to /taiga-resolvable real
files, so the 003_ctt_v2plus root can assemble on DeltaAI (where /projects is not mounted).

Every cttv2 source tensor physically lives on the shared Taiga storage; the inventories built on
Campus Cluster just record the `/projects` mount prefix (Campus-Cluster-only), and S0's latents
add one symlink layer (exp_064 -> /projects/.../exp_058, whose target is a REAL file at /taiga).

Resolver, per path:
  1. taiga = path.replace('/projects/', '/taiga/')
  2. if taiga is a real file            -> taiga            (S2a/S2b/S4, S6 conditions)
  3. elif taiga is a symlink            -> readlink, swap its target's /projects->/taiga, and use
                                           that if it is a real file (S0's exp_064->exp_058 chain)
  4. else                               -> hard error (unresolvable)

In place, with a .bak backup. S6.json is already /taiga (skipped). Idempotent: a path already a
/taiga real file is left as-is. Every result is asserted to be an existing real file.

    python scripts/ctt_v2/s6/normalize_inventory_paths.py            # report
    python scripts/ctt_v2/s6/normalize_inventory_paths.py --write
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
INV = REPO / "outputs/ctt_v2/inventories"
STRATA = ["S0", "S1", "S2a", "S2b", "S4"]    # S6 is already /taiga-native; S1 added 2026-08-29 (was omitted from the plus build)
FIELDS = ["latents", "cond_clean", "conditions"]


def resolve(path: str) -> tuple[str | None, str]:
    """Return (resolved_taiga_real_file_or_None, how)."""
    if not path:
        return None, "empty"
    if path.startswith("/taiga/") and os.path.isfile(path):
        return path, "already"
    taiga = path.replace("/projects/", "/taiga/", 1)
    if os.path.isfile(taiga) and not os.path.islink(taiga):
        return taiga, "prefix"
    if os.path.islink(taiga):
        tgt = os.readlink(taiga)
        real = tgt.replace("/projects/", "/taiga/", 1) if tgt.startswith("/projects/") else tgt
        if not os.path.isabs(real):
            real = os.path.normpath(os.path.join(os.path.dirname(taiga), real))
        if os.path.isfile(real):
            return real, "symlink1"
    # last resort: fully resolve via /taiga
    if os.path.exists(taiga):
        real = os.path.realpath(taiga)
        if os.path.isfile(real):
            return real, "realpath"
    return None, "UNRESOLVED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    total_bad = 0
    for st in STRATA:
        p = INV / f"{st}.json"
        inv = json.loads(p.read_text())
        clips = inv["clips"]
        stats = {"already": 0, "prefix": 0, "symlink1": 0, "realpath": 0, "UNRESOLVED": 0, "none": 0}
        bad = []
        for stem, c in clips.items():
            for f in FIELDS:
                v = c.get(f)
                if not v:
                    stats["none"] += 1
                    continue
                r, how = resolve(v)
                stats[how] = stats.get(how, 0) + 1
                if r is None:
                    bad.append((stem, f, v))
                elif args.write:
                    c[f] = r
        total_bad += len(bad)
        print(f"[{st}] {len(clips)} clips | " + " ".join(f"{k}={v}" for k, v in stats.items() if v)
              + (f"  BAD={len(bad)} e.g. {bad[:2]}" if bad else "  all resolvable"))
        if args.write and not bad:
            shutil.copy2(p, p.with_suffix(".json.bak"))
            p.write_text(json.dumps(inv, indent=1))
            # post-write assert: every non-empty path is a real file
            inv2 = json.loads(p.read_text())
            miss = [(s, f) for s, c in inv2["clips"].items() for f in FIELDS
                    if c.get(f) and not os.path.isfile(c[f])]
            if miss:
                raise SystemExit(f"[{st}] POST-WRITE {len(miss)} paths are not real files: {miss[:5]}")
            print(f"[{st}] WROTE (backup {p.with_suffix('.json.bak').name}); all paths verified real files")
    if total_bad:
        print(f"\n[normalize] {total_bad} UNRESOLVED paths — not written. Investigate before assembling.")
        return 1
    print("\n[normalize] all source paths resolvable to /taiga real files"
          + ("" if args.write else " (report only — pass --write)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
