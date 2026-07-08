"""exp_057 — taxonomy-stratified analysis of the harness items (login-node, numpy only).

Joins items.jsonl with quads.json (ref_taxonomy / ref_sidedness / ref_texture)
and writes analysis.md next to items.jsonl:

  1. arm x taxonomy tables (appearance raw+norm, leak, seam, endpoints, timing)
  2. per-style floor-ceiling gap -> normalization-reliability flags
     (gap < GAP_MIN means lerp floor ~ real ceiling: normalized scores there
     are noise amplifiers, read raw metrics instead)
  3. texture strata: cousin (shadow/fire_element) vs novel vs vanish
  4. base-vs-IC twin deltas on identical inputs

Usage: python analyze_taxonomy.py --items outputs/eval/exp_057/quads/run_NNNN/items.jsonl
"""

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
GAP_MIN = 0.05  # appearance floor-ceiling gap below which normalization is unreliable


def ms(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return "—"
    s = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():.2f}±{s:.2f} (n={len(v)})"


def table(rows_by_group, cols, colnames):
    keys = sorted(rows_by_group)
    out = ["| group | " + " | ".join(colnames) + " |",
           "|" + "---|" * (len(cols) + 1)]
    for k in keys:
        rs = rows_by_group[k]
        out.append(f"| {k} (n={len(rs)}) | " +
                   " | ".join(ms([r.get(c) for r in rs]) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    args = ap.parse_args()
    items_path = pathlib.Path(args.items)
    if not items_path.is_absolute():
        items_path = REPO_ROOT / items_path
    rows = [json.loads(l) for l in items_path.read_text().splitlines() if l.strip()]
    quads = {q["id"]: q for q in json.loads((EXP / "dataset/quads.json").read_text())}
    for r in rows:
        q = quads[r["item_id"]]
        r["tax"] = q["ref_taxonomy"]
        r["sided"] = q["ref_sidedness"]
        r["texture"] = ("cousin" if q["ref_texture"].startswith("cousin")
                        else "vanish" if "vanish" in q["ref_texture"]
                        else "trained" if q["ref_texture"] == "trained" else "novel")
        r["ep_dino_mean"] = np.nanmean([r.get("prefix_dino", np.nan) or np.nan,
                                        r.get("suffix_dino", np.nan) or np.nan])

    COLS = ["appearance_best", "norm_appearance_best", "leak_max_sim_target",
            "max_seam_z", "ep_dino_mean", "scalar_depart", "scalar_arrive"]
    NAMES = ["app raw", "app norm", "leak", "seam z", "endpoint DINO", "depart", "arrive"]

    md = [f"# exp_057 taxonomy analysis — {items_path.parent.name} ({len(rows)} items)\n"]

    ic = [r for r in rows if not r["arm"].startswith("base_")]
    base = [r for r in rows if r["arm"].startswith("base_")]

    md.append("## Arms (all items)\n")
    g = defaultdict(list)
    for r in rows:
        g[r["arm"]].append(r)
    md.append(table(g, COLS, NAMES) + "\n")

    md.append("## IC arms x reference taxonomy\n")
    g = defaultdict(list)
    for r in ic:
        g[f"{r['arm']} / {r['tax']}"].append(r)
    md.append(table(g, COLS, NAMES) + "\n")

    md.append("## IC arms x texture stratum (cousin vs novel vs vanish)\n")
    g = defaultdict(list)
    for r in ic:
        if r["arm"] in ("ic_os_inclass", "ic_os_to2s"):
            g[f"{r['arm']} / {r['texture']}"].append(r)
    md.append(table(g, COLS, NAMES) + "\n")

    md.append("## Normalization reliability (per style, appearance floor vs ceiling)\n")
    md.append("| style | n items | floor app (mean) | ceiling app | gap | verdict |")
    md.append("|---|---|---|---|---|---|")
    by_style = defaultdict(list)
    for r in rows:
        by_style[r["style"]].append(r)
    unreliable = []
    for s, rs in sorted(by_style.items()):
        fl = np.mean([r["floor_appearance_best"] for r in rs
                      if r.get("floor_appearance_best") is not None])
        ce = next((r.get("ceil_appearance_best") for r in rs
                   if r.get("ceil_appearance_best") is not None), None)
        if ce is None:
            md.append(f"| {s} | {len(rs)} | {fl:.3f} | — (singleton) | — | RAW ONLY |")
            continue
        gap = ce - fl
        verdict = "ok" if gap >= GAP_MIN else "UNRELIABLE (floor≈ceiling)"
        if gap < GAP_MIN:
            unreliable.append(s)
        md.append(f"| {s} | {len(rs)} | {fl:.3f} | {ce:.3f} | {gap:+.3f} | {verdict} |")
    md.append("")
    if unreliable:
        md.append(f"Norm-unreliable styles (gap<{GAP_MIN}): **{', '.join(unreliable)}** — "
                  "read RAW appearance + leak + judge there; normalized values are noise.\n")

    md.append("## Base-vs-IC twins (identical inputs)\n")
    md.append("| twin | metric | base | ic | delta |")
    md.append("|---|---|---|---|---|")
    ic_by_id = {r["item_id"]: r for r in ic}
    for b in base:
        iid = b["item_id"].replace("base_", "ic_", 1)
        i = ic_by_id.get(iid)
        if not i:
            continue
        for c, n in [("leak_max_sim_target", "leak"), ("max_seam_z", "seam z"),
                     ("appearance_best", "app raw"), ("ep_dino_mean", "endpoint DINO")]:
            bv, iv = b.get(c), i.get(c)
            if bv is None or iv is None or not np.isfinite(bv) or not np.isfinite(iv):
                continue
            md.append(f"| {iid.removeprefix('ic_')} | {n} | {bv:.3f} | {iv:.3f} | {iv-bv:+.3f} |")
    md.append("")

    out = items_path.parent / "analysis.md"
    out.write_text("\n".join(md))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
