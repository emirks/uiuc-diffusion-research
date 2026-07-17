#!/usr/bin/env python3
"""split v1.1 — PLAN Amendment 2 §A2.3: re-draw live_concert ONLY.

Owner ruling 2026-07-16: live_concert_0 == live_concert_2 (true duplicate;
live_concert_2 quarantined to _removed/), remaining 7 clips ruled distinct,
downgrading the automated all-mutual-dup finding that had forced all-train.
This script copies split_v1 verbatim for the other 38 classes (byte-identity
asserted) and re-draws live_concert under the FROZEN v1 rule: n=7 in [4,8)
-> 1 test clip, drawn by the same seeded stream random.Random("split_v1:live_concert")
over the sorted surviving clip ids.

Usage: python scripts/build_split_v1_1.py
"""
import hashlib, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data/processed/transitions_std121/split_v1.json"
OUT = ROOT / "data/processed/transitions_std121/split_v1.1.json"
DOC = ROOT / "data/processed/transitions_std121/SPLIT_V1.1.md"

v1 = json.loads(V1.read_text())
v11 = json.loads(V1.read_text())  # independent copy

lc = v1["classes"]["live_concert"]
assert lc["test"] == [] and lc["n_clips"] == 8, "unexpected v1 live_concert state"
survivors = sorted(c for c in lc["train"] if c != "live_concert_2")
assert len(survivors) == 7
test = random.Random("split_v1:live_concert").sample(survivors, 1)
train = [c for c in survivors if c not in test]
paths = {c: p for c, p in lc["paths"].items() if c != "live_concert_2"}
v11["classes"]["live_concert"] = {
    "n_clips": 7, "train": train, "test": test, "paths": paths,
    "redraw": "v1.1 (Amendment 2 A2.3): live_concert_2 removed as owner-ruled "
              "duplicate of _0; 1 test clip drawn from frozen stream "
              "split_v1:live_concert over 7 sorted survivors",
}

# byte-identity for every other class
for cls in v1["classes"]:
    if cls == "live_concert":
        continue
    assert json.dumps(v1["classes"][cls], sort_keys=True) == \
           json.dumps(v11["classes"][cls], sort_keys=True), cls

v11["split"] = "v1.1"
v11["n_train"] = sum(len(r["train"]) for r in v11["classes"].values())
v11["n_test"] = sum(len(r["test"]) for r in v11["classes"].values())
v11["test_fraction"] = round(v11["n_test"] / (v11["n_train"] + v11["n_test"]), 4)
v11["provenance"]["amendment_2"] = (
    "2026-07-16: live_concert re-drawn only (owner duplicate ruling 0==2); "
    "all other 38 classes byte-identical to split_v1; rule unchanged."
)

OUT.write_text(json.dumps(v11, indent=1, sort_keys=True))
sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
DOC.write_text(f"""# split v1.1 — FINAL

sha256: `{sha}`
Derived from split_v1 (sha recorded in SPLIT_V1_FINAL.md) by PLAN Amendment 2 SSA2.3.
Change: live_concert only — live_concert_2 removed (owner-ruled duplicate of _0,
2026-07-16); band re-drawn under the frozen v1 rule (n=7 -> 1 test clip, stream
`split_v1:live_concert` over 7 sorted survivors). Every other class byte-identical
to split_v1 (asserted at build). Totals: {v11['n_train']} train / {v11['n_test']} test.

live_concert v1.1: test = {test}, train = {train}
""")
print(f"live_concert test draw: {test}")
print(f"totals: {v11['n_train']} train / {v11['n_test']} test  sha256 {sha[:16]}...")
