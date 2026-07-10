"""exp_058 — v1-vs-v2 analysis (login-node, numpy only).

Joins the exp_058 scoring run (v2 items) with exp_057 run_0001 (v1 + base
scores on the SAME 40 suite ids) and writes analysis_v1v2.md:

  1. per-arm v1 vs v2 means (raw appearance, leak, seam, endpoint DINO) with
     per-item deltas on identical inputs (paired comparison — the load-bearing
     table; normalized numbers only where exp_057's gap audit deemed them
     readable, i.e. two-sided styles).
  2. held-out classes (hero_flight / illustration_scene / gas_transformation /
     raven + hole/seamless): per-item v1 vs v2 vs base rows.
  3. in-distribution split: suite items whose class moved INTO v2 training —
     endpoints unseen (quad clips excluded) vs small classes with trained
     clips (fire_element, plasma_explosion, giant_grab flagged trained).
  4. new arms: ic2_prefixonly (+ base_prefixonly twins), ic2_ts_heldout raven
     (+ base twin) — raw metrics; suffix metrics reported only for two-sided.

Usage:
  python analyze_v1v2.py --items outputs/eval/exp_058/quads/run_NNNN/items.jsonl \
      [--v1-items outputs/eval/exp_057/quads/run_0001/items.jsonl]
"""

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).parent

# classes that moved into v2 training with quad clips EXCLUDED from training
TRAINED_UNSEEN_EP = {"shadow", "super_fast_run", "portal", "wireframe",
                     "animalization", "money_rain", "shadow_smoke"}
# small classes trained WITH their quad clips (in-distribution-trained items)
TRAINED_SEEN_EP = {"fire_element", "plasma_explosion", "giant_grab", "flame",
                   "earth_wave", "melt_transition", "water_bending",
                   "air_bending", "display_transition", "flying_cam_transition"}
HELD_OUT = {"hero_flight", "illustration_scene", "gas_transformation",
            "raven_transition", "hole_transition", "seamless_transition",
            "jump_transition"}

COLS = [("appearance_best", "raw app"), ("leak_max_sim_target", "leak"),
        ("max_seam_z", "seam z"), ("prefix_dino", "pfx DINO"),
        ("suffix_dino", "sfx DINO")]


def pick(r, key):
    v = r.get(key)
    return v if v is not None and np.isfinite(v) else None


def ms(rows, key):
    vs = [pick(r, key) for r in rows]
    vs = [v for v in vs if v is not None]
    return f"{np.mean(vs):.3f}±{np.std(vs):.2f}" if vs else "—"


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) and np.isfinite(v) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--v1-items",
                    default="outputs/eval/exp_057/quads/run_0001/items.jsonl")
    args = ap.parse_args()

    p2 = REPO_ROOT / args.items
    p1 = REPO_ROOT / args.v1_items
    v2 = [json.loads(l) for l in p2.read_text().splitlines() if l.strip()]
    v1 = {r["item_id"]: r for l in [p1.read_text().splitlines()]
          for r in (json.loads(x) for x in l if x.strip())}
    quads = {q["id"]: q for q in json.loads(
        (EXP / "dataset/quads_v2.json").read_text())}

    md = [f"# exp_058 v1-vs-v2 analysis — {p2.parent.name} ({len(v2)} v2 items)\n"]

    # --- 1. paired suite comparison -------------------------------------
    suite = [r for r in v2 if r["item_id"] in v1
             and not quads.get(r["item_id"], {}).get("prefix_only")
             and quads.get(r["item_id"], {}).get("label") == "ic2"]
    md.append(f"## 1. Suite rerun: paired v1 vs v2 on {len(suite)} identical items\n")
    md.append("| metric | v1 mean | v2 mean | mean Δ(v2−v1) | v2 better (n) |")
    md.append("|---|---|---|---|---|")
    better_dir = {"appearance_best": +1, "leak_max_sim_target": -1,
                  "max_seam_z": -1, "prefix_dino": +1, "suffix_dino": +1}
    for key, name in COLS:
        deltas, wins, n = [], 0, 0
        for r in suite:
            a, b = pick(v1[r["item_id"]], key), pick(r, key)
            if a is None or b is None:
                continue
            deltas.append(b - a)
            n += 1
            if (b - a) * better_dir[key] > 0:
                wins += 1
        if deltas:
            md.append(f"| {name} | {ms([v1[r['item_id']] for r in suite], key)} | "
                      f"{ms(suite, key)} | {np.mean(deltas):+.3f} | {wins}/{n} |")

    # --- 2. held-out classes: per-item table -----------------------------
    md.append("\n## 2. Held-out classes (never in v2 training)\n")
    md.append("| item | class | v1 raw app | v2 raw app | v1 leak | v2 leak | v1 seam | v2 seam |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(v2, key=lambda x: x["item_id"]):
        q = quads.get(r["item_id"], {})
        cls = q.get("reference_class", "")
        if cls not in HELD_OUT or q.get("label") != "ic2" or q.get("prefix_only"):
            continue
        o = v1.get(r["item_id"], {})
        md.append(f"| {r['item_id'][:52]} | {cls} | {fmt(o.get('appearance_best'))} | "
                  f"{fmt(r.get('appearance_best'))} | {fmt(o.get('leak_max_sim_target'))} | "
                  f"{fmt(r.get('leak_max_sim_target'))} | {fmt(o.get('max_seam_z'))} | "
                  f"{fmt(r.get('max_seam_z'))} |")

    # --- 3. in-distribution split ----------------------------------------
    md.append("\n## 3. Classes that moved INTO v2 training\n")
    for grp, name in [(TRAINED_UNSEEN_EP, "trained class, UNSEEN endpoints/demo (quad clips excluded)"),
                      (TRAINED_SEEN_EP, "trained class, endpoints/demo IN training (small classes)")]:
        rows2 = [r for r in suite if quads[r["item_id"]]["reference_class"] in grp]
        rows1 = [v1[r["item_id"]] for r in rows2]
        md.append(f"\n### {name} (n={len(rows2)})\n")
        md.append("| metric | v1 | v2 |")
        md.append("|---|---|---|")
        for key, cname in COLS:
            md.append(f"| {cname} | {ms(rows1, key)} | {ms(rows2, key)} |")

    # --- 4. new arms ------------------------------------------------------
    md.append("\n## 4. New arms (no v1 counterpart)\n")
    groups = defaultdict(list)
    for r in v2:
        q = quads.get(r["item_id"], {})
        arm = q.get("arm", r.get("arm", "?"))
        if arm in ("ic2_prefixonly", "base_prefixonly", "ic2_ts_heldout",
                   "base_ts_heldout"):
            groups[arm].append(r)
    md.append("| arm | n | raw app | leak | seam z | pfx DINO | sfx DINO |")
    md.append("|---|---|---|---|---|---|---|")
    for arm in ("ic2_prefixonly", "base_prefixonly", "ic2_ts_heldout", "base_ts_heldout"):
        rs = groups.get(arm, [])
        if rs:
            md.append(f"| {arm} | {len(rs)} | {ms(rs, 'appearance_best')} | "
                      f"{ms(rs, 'leak_max_sim_target')} | {ms(rs, 'max_seam_z')} | "
                      f"{ms(rs, 'prefix_dino')} | {ms(rs, 'suffix_dino')} |")
    md.append("\nPer-item, prefix-only (suffix metrics N/A by construction; "
              "sfx DINO here measures landing-near-true-ending, advisory):\n")
    md.append("| item | raw app | leak | pfx DINO |")
    md.append("|---|---|---|---|")
    for r in sorted(groups.get("ic2_prefixonly", []) + groups.get("base_prefixonly", []),
                    key=lambda x: x["item_id"]):
        md.append(f"| {r['item_id'][:60]} | {fmt(r.get('appearance_best'))} | "
                  f"{fmt(r.get('leak_max_sim_target'))} | {fmt(r.get('prefix_dino'))} |")

    out = p2.parent / "analysis_v1v2.md"
    out.write_text("\n".join(md) + "\n")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
