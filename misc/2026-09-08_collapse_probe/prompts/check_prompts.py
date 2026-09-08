#!/usr/bin/env python3
"""Verify prompts.jsonl for the collapse-probe experiment. Exit non-zero on any violation."""
import json, os, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
# repo root = .../diffusion-research (this file is misc/2026-09-08_collapse_probe/prompts/)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GRID = os.path.join(ROOT, "store/gens/005_base_cond/04_neutral_v3__dai/grid.jsonl")
PROMPTS = os.path.join(HERE, "prompts.jsonl")

BANNED = ["dissolve", "crossfade", "fade", "cut", "lerp"]
errs = []

# load grid captions by endpoint (first occurrence)
grid = {}
for line in open(GRID):
    r = json.loads(line)
    grid.setdefault(r["endpoint"], r)

rows = [json.loads(l) for l in open(PROMPTS)]

def wc(s):
    return len(s.split())

def banned_hits(s):
    low = s.lower()
    return [b for b in BANNED if re.search(r"\b" + re.escape(b) + r"\b", low)]

tiers = Counter()
per_class = Counter()
ids = set()
for row in rows:
    pid = row["prompt_id"]
    ep = row["endpoint"]
    if pid in ids:
        errs.append(f"{pid}: duplicate prompt_id")
    ids.add(pid)
    tiers[row["tier"]] += 1

    # clip exists
    cp = os.path.join(ROOT, row["clip_path"])
    if not os.path.exists(cp):
        errs.append(f"{pid}: clip missing {row['clip_path']}")

    # start_caption verbatim from grid
    if ep not in grid:
        errs.append(f"{pid}: endpoint {ep} not in grid")
    elif row["start_caption"] != grid[ep]["prompt"]:
        errs.append(f"{pid}: start_caption not verbatim from grid")

    # endpoint_class / source consistent with grid
    if ep in grid:
        if row["endpoint_class"] != grid[ep]["endpoint_class"]:
            errs.append(f"{pid}: endpoint_class mismatch")
        if row["endpoint_source"] != grid[ep]["endpoint_source"]:
            errs.append(f"{pid}: endpoint_source mismatch")
        if grid[ep]["sided"] != "one":
            errs.append(f"{pid}: endpoint {ep} is not sided==one")
        if grid[ep]["endpoint_source"] not in {"heldin_test","heldout","heldin_train"}:
            errs.append(f"{pid}: endpoint_source not in allowed set")

    # banned words anywhere in the crafted text
    for field in ("change_clause", "end_caption", "full_prompt", "neutral_prompt"):
        hits = banned_hits(row[field])
        if hits:
            errs.append(f"{pid}: banned word(s) {hits} in {field}")

    # length ranges
    n_cc = wc(row["change_clause"])
    if not (15 <= n_cc <= 35):
        errs.append(f"{pid}: change_clause {n_cc} words (need 15-35)")
    n_ec = wc(row["end_caption"])
    if not (25 <= n_ec <= 40):
        errs.append(f"{pid}: end_caption {n_ec} words (need 25-40)")

    # single-sentence check for change_clause and end_caption (exactly one terminal period, no interior . ! ?)
    for field in ("change_clause", "end_caption"):
        s = row[field].strip()
        if not s.endswith("."):
            errs.append(f"{pid}: {field} does not end with a period")
        interior = s[:-1]
        if any(c in interior for c in ".!?"):
            errs.append(f"{pid}: {field} appears to be more than one sentence")

    # full_prompt / neutral_prompt composition
    exp_full = f"{row['start_caption']} {row['change_clause']} {row['end_caption']}"
    if row["full_prompt"] != exp_full:
        errs.append(f"{pid}: full_prompt != start + change + end")
    exp_neu = f"{row['start_caption']} {row['end_caption']}"
    if row["neutral_prompt"] != exp_neu:
        errs.append(f"{pid}: neutral_prompt != start + end")

    # neutral prompt must not contain the change clause
    if row["change_clause"] in row["neutral_prompt"]:
        errs.append(f"{pid}: neutral_prompt contains change_clause")

    if row["tier"] == "high":
        per_class[row["endpoint_class"]] += 1

# counts
if tiers["high"] != 30:
    errs.append(f"high count = {tiers['high']} (need 30)")
if tiers["low"] != 10:
    errs.append(f"low count = {tiers['low']} (need 10)")
if len(rows) != 40:
    errs.append(f"total rows = {len(rows)} (need 40)")

# <=2 per class (count distinct endpoints per class among the 30 high selections)
high_eps_by_class = Counter(grid[r["endpoint"]]["endpoint_class"] for r in rows if r["tier"] == "high")
for cl, n in high_eps_by_class.items():
    if n > 2:
        errs.append(f"class {cl}: {n} high clips (>2)")

# unique mechanism families among HIGH
mechs = [r["mechanism_family"] for r in rows if r["tier"] == "high"]
dup_mech = [m for m, c in Counter(mechs).items() if c > 1]
if dup_mech:
    errs.append(f"duplicate HIGH mechanism families: {dup_mech}")

# each LOW endpoint must also have a HIGH row
high_eps = {r["endpoint"] for r in rows if r["tier"] == "high"}
for r in rows:
    if r["tier"] == "low" and r["endpoint"] not in high_eps:
        errs.append(f"{r['prompt_id']}: LOW endpoint {r['endpoint']} has no HIGH counterpart")

# --- summary ---
print("=== check_prompts summary ===")
print(f"rows: {len(rows)}  high: {tiers['high']}  low: {tiers['low']}")
print(f"distinct high classes: {len(high_eps_by_class)}  max per class: {max(high_eps_by_class.values())}")
print("class distribution (high):")
for cl, n in sorted(high_eps_by_class.items(), key=lambda x: (-x[1], x[0])):
    print(f"  {cl}: {n}")
print("LOW endpoints:", [r["endpoint"] for r in rows if r["tier"] == "low"])
print(f"unique HIGH mechanism families: {len(set(mechs))}")
cc_lens = [wc(r["change_clause"]) for r in rows]
ec_lens = [wc(r["end_caption"]) for r in rows]
print(f"change_clause words: min {min(cc_lens)} max {max(cc_lens)}")
print(f"end_caption   words: min {min(ec_lens)} max {max(ec_lens)}")

if errs:
    print("\n=== VIOLATIONS ===")
    for e in errs:
        print(" -", e)
    sys.exit(1)
print("\nALL CHECKS PASSED")
sys.exit(0)
