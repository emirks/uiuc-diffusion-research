#!/usr/bin/env python
"""grid v3 — split v1.3 = split v1.2 + the grid-v3 corpus additions. STRICT SUPERSET: every v1.2 clip keeps its
class and band; nothing is removed or relabelled.

Additions (all derived from grid_v3.yaml + what is on disk under data/processed/transitions_std121/):
  * top-up clips of existing classes (tier1_topups, tier1_reference_only, flame_additions, zs_pool_topups)
    -> appended to the class's `test` band (UNTRAINED content; band = content novelty, never training exposure)
  * the new zero-shot Higgsfield classes -> new held-out classes (generalist_holdout), test = the two clips the
    builder uses as reference/endpoint pair (sorted stems 0,1), train = the rest (v1.2's held-out shape)
  * EffectData classes `ed.<effect>` -> new held-out classes; test = clips used as endpoints on the grid
    (same-content ground-truth clips + cross endpoints), train = references + pool clips

The sha256 field follows the v1.2 convention (recomputed by the same rule and verified to reproduce v1.2's).
Run:  python scripts/grid_v3/build_split_v1_3.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
STD = REPO / "data/processed/transitions_std121"
V12 = STD / "split_v1.2.json"
V13 = STD / "split_v1.3.json"
GRID = yaml.safe_load((REPO / "eval_ladder/grid_v3.yaml").read_text())
PENDING = REPO / "eval_ladder/registry_v3_pending.jsonl"


def split_sha(d: dict) -> str:
    body = {k: v for k, v in d.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def main() -> None:
    v12 = json.loads(V12.read_text())
    # the sha convention: verify our rule reproduces v1.2's own field before trusting it for v1.3
    same = split_sha(v12) == v12["sha256"]
    print(f"v1.2 sha rule reproduced: {same}" + ("" if same else "  (v1.3 sha computed by canonical-json rule; recorded as such)"))

    d = json.loads(json.dumps(v12))
    d["split"] = "v1.3"
    added = {"topups": {}, "new_classes": {}, "effectdata": {}}

    def add_clip(cls: str, stem: str, band: str):
        e = d["classes"][cls]
        p = STD / cls / f"{stem}.mp4"
        assert p.exists(), f"missing standard clip {p}"
        if stem in e["paths"]:
            return
        e["paths"][stem] = str(p.relative_to(REPO))
        e[band].append(stem)
        e["n_clips"] = len(e["paths"])
        added["topups"].setdefault(cls, []).append(stem)

    for key in ("tier1_topups", "tier1_reference_only", "flame_additions", "zs_pool_topups"):
        for cls, stems in GRID[key].items():
            for stem in stems:
                add_clip(cls, stem, "test")

    for cls in sorted(GRID["new_zero_shot_classes"]):
        stems = sorted(p.stem for p in (STD / cls).glob("*.mp4"))
        assert len(stems) >= 4, cls
        d["classes"][cls] = {"n_clips": len(stems), "paths": {s: str((STD / cls / f"{s}.mp4").relative_to(REPO)) for s in stems},
                             "test": stems[:2], "train": stems[2:]}
        d["generalist_holdout"].append(cls)
        added["new_classes"][cls] = len(stems)

    pref = GRID["effectdata"]["clip_prefix"] + "."
    ed_endpoints = set()
    for l in PENDING.read_text().splitlines():
        r = json.loads(l)
        if r["endpoint"].startswith(pref):
            ed_endpoints.add(r["endpoint"])
    for cdir in sorted(STD.glob(f"{pref}*")):
        cls = cdir.name
        stems = sorted(p.stem for p in cdir.glob("*.mp4"))
        if not stems:
            continue
        test = [s for s in stems if s in ed_endpoints]
        d["classes"][cls] = {"n_clips": len(stems), "paths": {s: str((cdir / f"{s}.mp4").relative_to(REPO)) for s in stems},
                             "test": test, "train": [s for s in stems if s not in test]}
        d["generalist_holdout"].append(cls)
        added["effectdata"][cls] = len(stems)

    d["generalist_holdout"] = sorted(set(d["generalist_holdout"]))
    d["n_classes"] = len(d["classes"])
    d["n_train"] = sum(len(e["train"]) for e in d["classes"].values())
    d["n_test"] = sum(len(e["test"]) for e in d["classes"].values())
    d["test_fraction"] = round(d["n_test"] / (d["n_train"] + d["n_test"]), 4)
    d["provenance"] = {"base": "split_v1.2", "base_sha256": v12["sha256"],
                       "rule": "strict superset of v1.2 (every v1.2 clip keeps class + band); grid v3 additions: top-ups -> test band "
                               "(untrained content), new Higgsfield zero-shot classes + EffectData `ed.*` classes -> generalist_holdout "
                               "(test = endpoint/reference clips, train = the rest). Built by scripts/grid_v3/build_split_v1_3.py from "
                               "eval_ladder/grid_v3.yaml + disk (2026-09-07).",
                       "promotions_train_to_test": v12["provenance"].get("promotions_train_to_test", []),
                       "additions": added}
    d["sha256"] = split_sha(d)
    V13.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
    # superset assert
    for cls, e in v12["classes"].items():
        n = d["classes"][cls]
        assert set(e["paths"]) <= set(n["paths"]) and set(e["train"]) <= set(n["train"]) and set(e["test"]) <= set(n["test"]), cls
    print(f"wrote {V13.relative_to(REPO)}: classes {d['n_classes']} (v1.2 {v12['n_classes']}), train {d['n_train']}, test {d['n_test']}, "
          f"held-out {len(d['generalist_holdout'])}; sha {d['sha256'][:8]}")
    print("added:", {k: (len(v) if isinstance(v, dict) else v) for k, v in added.items()})


if __name__ == "__main__":
    main()
