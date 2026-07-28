#!/usr/bin/env python
"""ctt_v2 endpoint EXPANSION — round-2 candidate list (vc-bench-hf only).

Advisor ruling (2026-07-25, reversal): expand the endpoint pool inside ONE 12-hour window,
from the vc-bench-hf material already on local disk. No OpenVid fetching, DAVIS exhausted.
The existing bank is READ-ONLY: this pipeline writes to a parallel v2 tree and the deliverable
is an ADDITIVE `bank_tightened_v2` with new IDs; no existing row is ever mutated.

MEASURED CORRECTION to the premise (operator, before running anything): the "1,065 unscreened
clips" figure was wrong. Of 1,261 local mp4s only 1,097 have a VC-Bench.csv row, and round 1
already took EVERY clip meeting its collection bar (aesthetic >= 0.6 AND num_scenes == 1) —
that bar yields exactly the 196 already screened, so **zero** clips are unscreened at the round-1
thresholds. Real headroom therefore requires relaxing the AESTHETIC floor (num_scenes == 1 is
kept absolutely — a multi-scene source has cuts in it by definition and would reintroduce the
fabricated-discontinuity failure class):

    aesthetic >= 0.6, single-scene :  196 total ->   0 unscreened   (round 1 took all)
    aesthetic >= 0.5, single-scene :  579 total -> 383 unscreened
    aesthetic >= 0.4, single-scene :  762 total -> 566 unscreened
    aesthetic >= 0.0, single-scene :  797 total -> 601 unscreened

This script collects the WIDEST pool (601, any aesthetic, single-scene) on purpose. Detection
is the expensive pass and running it over a superset costs the same wall-clock as running it
over a subset, while preserving the option to set the aesthetic floor LATER from real yield
data instead of guessing now. The floor is applied downstream in tighten_v2.py, where the
advisor's ruling on it is recorded.
"""
import ast
import csv
import json
import os

REPO = "/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research"
WORK = os.path.join(REPO, "data/processed/ctt_v2_strata/endpoints_v2/_work")
OLD_CAND = os.path.join(REPO, "data/processed/synth_endpoints/_work/candidates.jsonl")
VC = os.path.join(REPO, "data/raw/vc-bench-hf")
VC_CSV = os.path.join(VC, "VC-Bench.csv")
OUT = os.path.join(WORK, "candidates_v2.jsonl")

AES_MIN = 0.0          # widest pool; the real floor is decided downstream (see docstring)
REQUIRE_SINGLE_SCENE = True


def main() -> None:
    already = set()
    for line in open(OLD_CAND):
        d = json.loads(line)
        if d["source"] == "vcbench":
            already.add(d["orig_id"])

    files = [f for f in os.listdir(VC) if f.lower().endswith(".mp4")]
    base2path = {}
    for f in files:
        base2path.setdefault(f.split("\\")[-1], f)

    rows = list(csv.DictReader(open(VC_CSV)))
    k0 = list(rows[0].keys())[0]           # filename column carries a BOM
    cands, skipped = [], {"no_local_file": 0, "multi_scene": 0, "below_aes": 0, "round1": 0}
    for r in rows:
        fn = r[k0]
        if fn not in base2path:
            skipped["no_local_file"] += 1
            continue
        if fn[:-4] in already:
            skipped["round1"] += 1
            continue
        try:
            aes = float(r["aesthetic_score"])
            ns = int(r["num_scenes"])
        except Exception:
            skipped["multi_scene"] += 1
            continue
        if REQUIRE_SINGLE_SCENE and ns != 1:
            skipped["multi_scene"] += 1
            continue
        if aes < AES_MIN:
            skipped["below_aes"] += 1
            continue
        try:
            res = ast.literal_eval(r["resolution"])
        except Exception:
            res = None
        cands.append({
            "source": "vcbench",
            "orig_id": fn[:-4],
            "orig_ref": base2path[fn],
            "path": os.path.join(VC, base2path[fn]),
            "license": "Pexels (free-to-use) via VC-Bench",
            "meta": {"category": r.get("category"), "aesthetic": aes, "num_scenes": ns,
                     "resolution": res, "length": r.get("length"), "fps": r.get("fps")},
        })

    os.makedirs(WORK, exist_ok=True)
    with open(OUT, "w") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")

    bins = {}
    for c in cands:
        b = round(c["meta"]["aesthetic"] // 0.1 * 0.1, 1)
        bins[b] = bins.get(b, 0) + 1
    print(f"[collect_v2] candidates: {len(cands)}   skipped: {json.dumps(skipped)}")
    print(f"[collect_v2] aesthetic histogram: "
          f"{json.dumps({str(k): bins[k] for k in sorted(bins)})}")
    print(f"[collect_v2] -> {OUT}")


if __name__ == "__main__":
    main()
