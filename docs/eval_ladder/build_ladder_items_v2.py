#!/usr/bin/env python3
"""Generate docs/eval_ladder/ladder_items_v2.json — the eval-ladder item grid as
AMENDED 2026-07-16 (see PLAN.md "Amendment 1"). v2 is derived from the FROZEN
ladder_items_v1.json (every v1 per-class selection is copied verbatim — same split
sha, same test_items/r2_items/reference/endpoint_seen_by_ic2) and LAYERS the
amendment additions on top. It never re-draws or edits any v1 item.

Additions (advisor ruling, 2026-07-16):
  - specialist_conditioning per class: side-KEYED (one_sided -> prefix-only;
    two_sided -> prefix+suffix), superseding v1 D2's sidedness-BLIND specialists.
  - r1k: prefix-only base rung (no adapter) for the one_sided classes' test items —
    re-baselines C4/C6/C7/C8 which otherwise confound adapter-value with suffix
    removal. two_sided R1K == R1 (reuse existing R1 rows).
  - r3x / r4x: cross-class donor-endpoint rungs (contrast C9). For each recipient in
    the eligibility block B8 (one_sided AND scene_swap==false), feed the recipient's
    specialist (R3X) and the generalist w/ recipient reference (R4X twin) endpoints
    borrowed from 4 other B8 classes. No ground truth (foreign endpoints); scored as
    class-effect transfer, never item reconstruction.

Deterministic. Re-derivable. Usage: python docs/eval_ladder/build_ladder_items_v2.py
"""
import json, random, hashlib, pathlib, sys

try:
    import yaml
except ImportError:
    sys.exit("needs pyyaml (research env)")

ROOT = pathlib.Path(__file__).resolve().parents[2]
V1 = ROOT / "docs/eval_ladder/ladder_items_v1.json"
AXES = ROOT / "outputs/taxonomy/class_axes.yaml"
COND = ROOT / "experiments/exp_062_ladder_r2r3_specialists/dataset/cond"
OUT = ROOT / "docs/eval_ladder/ladder_items_v2.json"

N_DONORS = 4  # donor classes per recipient


def main() -> None:
    v1 = json.loads(V1.read_text())
    v1sha = hashlib.sha256(V1.read_bytes()).hexdigest()
    classes = {c: dict(g) for c, g in v1["classes"].items()}  # deep-ish copy of per-class dicts
    axes = yaml.safe_load(AXES.read_text())["classes"]

    # ---- 1. specialist conditioning + r1k membership (per class) ----
    for c, g in classes.items():
        two_sided = bool(g["suffix_conditioning"])
        g["specialist_conditioning"] = "prefix_suffix" if two_sided else "prefix_only"
        # R1K (prefix-only base) is only needed where the specialist dropped the suffix
        g["r1k"] = (not two_sided)

    # ---- 2. R3X eligibility block B8 = one_sided AND scene_swap==false ----
    def scene_swap(c):
        v = axes[c]
        return bool(v.get("scene_swap", False))

    b8 = sorted(c for c, g in classes.items()
                if g["sidedness_key"] == "one_sided" and not scene_swap(c))

    # ---- 3. donor clip fixed per B8 class (ic2-unseen test clip if exactly one) ----
    def donor_clip(d):
        test = sorted(classes[d]["test_items"])
        seen = classes[d]["endpoint_seen_by_ic2"]
        unseen = [t for t in test if not seen[t]]
        if len(unseen) == 1:
            return unseen[0], "ic2_unseen"
        pick = random.Random(f"ladder_v2:{d}:r3x_donor_clip").sample(test, 1)[0]
        return pick, "rng"

    donor_of = {d: donor_clip(d) for d in b8}

    # ---- 4. per-recipient donor list (4 classes) + realized rows ----
    r3x = {}
    for c in b8:
        pool = sorted(x for x in b8 if x != c)
        donors = sorted(random.Random(f"ladder_v2:{c}:r3x_donors").sample(pool, N_DONORS))
        rows = []
        for d in donors:
            clip, how = donor_of[d]
            rows.append({
                "donor_class": d,
                "donor_clip": clip,
                "donor_clip_source": how,
                "endpoint_seen_by_ic2": bool(classes[d]["endpoint_seen_by_ic2"][clip]),
                # recipient is one_sided by construction of B8 -> prefix-only, donor start9 only
                "conditioning": "prefix_only",
                "cond_start9": f"{clip}_start9.mp4",
            })
        r3x[c] = {
            "recipient_conditioning": "prefix_only",
            "recipient_reference": classes[c]["reference"],  # for the R4X twin (generalist)
            "donors": rows,
        }

    # verify every donor cond cut exists (donors are test items -> already cut by make_conds)
    missing = sorted({r["cond_start9"] for c in r3x for r in r3x[c]["donors"]
                      if not (COND / r["cond_start9"]).exists()})
    if missing:
        sys.exit(f"[error] donor cond cuts missing (run make_conds.py): {missing}")

    doc = {
        "version": "ladder_items_v2",
        "supersedes": "ladder_items_v1",
        "amendment": "PLAN.md Amendment 1 (2026-07-16): side-keyed specialists, R1K base rung, R3X/R4X cross-class rungs (C9)",
        "frozen": True,
        "derived_from_v1_sha256": v1sha,
        "split_file": v1["split_file"],
        "split_sha256": v1["split_sha256"],  # UNCHANGED — no split edit
        "generalist_adapter": v1["generalist_adapter"],
        "seeds": v1["seeds"],
        "specialist_recipe": ("exp_051 c2v verbatim, SIDE-KEYED per class "
                              "specialist_conditioning (one_sided: prefix tb=2 p=1.0 only; "
                              "two_sided: + suffix tb=1 p=1.0), type-blind ICTRANS captions. "
                              "Supersedes v1 D2 (sidedness-BLIND)."),
        "r1k": {
            "desc": ("prefix-only base rung: 19B dev, NO adapter, prefix 9f only (no suffix), "
                     "for the one_sided classes' 2 test items x 3 seeds. Re-baselines C4/C6/C7/C8. "
                     "two_sided classes: R1K == R1 (reuse exp_061 R1 rows)."),
            "classes": sorted(c for c, g in classes.items() if g["r1k"]),
            "seeds": v1["seeds"],
        },
        "r3x": {
            "desc": ("cross-class donor-endpoint transfer. Recipient specialist (R3X) and "
                     "generalist-with-recipient-reference (R4X twin) receive endpoints from "
                     "N=4 other B8 classes. step_02000 only. No GT; class-effect transfer."),
            "eligibility_block_B8": b8,
            "excluded": {"super_fast_run": "scene_swap=true (no matched donors)",
                         "two_sided": "pool would be shadow_smoke<->hero_flight only; hero_flight conflict-pending"},
            "n_donors_per_recipient": N_DONORS,
            "donor_rule": "random.Random(f'ladder_v2:{recipient}:r3x_donors').sample(sorted(B8-{recipient}), 4)",
            "donor_clip_rule": "ic2-unseen test clip if exactly one unseen, else random.Random(f'ladder_v2:{donor}:r3x_donor_clip').sample(sorted(test_items),1)[0]; fixed per donor class",
            "donor_clip_of_class": {d: {"clip": donor_of[d][0], "source": donor_of[d][1]} for d in b8},
            "step": 2000,
            "seeds": v1["seeds"],
            "twin": "R4X",
            "scoring_contract": {
                "reference_video": "recipient's grid reference clip (identical in R3X and R4X -> delta is apples-to-apples)",
                "style": "recipient class",
                "sidedness_mask": "recipient sidedness (== donor's, both one_sided by construction)",
                "gt_available": False,
                "note": "M2c (copy) runs vs recipient's TRAINING manifest — flags memorized-content paste-on onto foreign endpoints.",
            },
            "recipients": r3x,
        },
        "contrasts_added": {
            "C9": {
                "contrast": "R3X - R4X",
                "pairing": "item(donor endpoint)+seed, 8 B8 recipients",
                "isolates": "specialist vs generalist on foreign (off-distribution) endpoints",
                "direction_test": "R3X > R4X; one-sided sign test across 8 recipients, alpha=0.05; SECONDARY",
            }
        },
        "rebaselined_contrasts": {
            "note": ("C4/C6/C7/C8 re-baseline onto R1K (prefix-only base) for one_sided classes so "
                     "'adapter value' is not confounded with suffix removal. C1 (R1-R0) stays on the "
                     "original blind R1 per exp_061. C5 (R3-R4) is now conditioning-matched (both keyed)."),
        },
        "classes": classes,
    }
    OUT.write_text(json.dumps(doc, indent=2))
    outsha = hashlib.sha256(OUT.read_bytes()).hexdigest()

    print(f"[done] {OUT}")
    print(f"  derived_from_v1_sha256 = {v1sha}")
    print(f"  ladder_items_v2 sha256 = {outsha}")
    print(f"  split_sha256 (unchanged) = {doc['split_sha256'][:16]}…")
    print(f"  B8 (R3X eligible, {len(b8)}): {b8}")
    print(f"  R1K classes ({len(doc['r1k']['classes'])}): {doc['r1k']['classes']}")
    print("  donor clip per B8 class:")
    for d in b8:
        print(f"    {d:22s} -> {donor_of[d][0]:24s} ({donor_of[d][1]})")
    print("  recipient donor lists:")
    for c in b8:
        print(f"    {c:22s} <- {[r['donor_class'] for r in r3x[c]['donors']]}")
    n_r3x = sum(len(r3x[c]['donors']) for c in b8) * len(doc['seeds'])
    print(f"  counts: R3X = R4X = {n_r3x} videos; R1K = {len(doc['r1k']['classes'])}*2*{len(doc['seeds'])} = "
          f"{len(doc['r1k']['classes'])*2*len(doc['seeds'])}")


if __name__ == "__main__":
    main()
