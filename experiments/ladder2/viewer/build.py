"""ladder2 — build the results viewer.

    python experiments/ladder2/viewer/build.py [--out outputs/reports/ladder2/index.html]

One self-contained HTML file: the ontology matrix, live statistics for whatever is selected,
every metric table, and the per-input presentation cards. Video is referenced by relative path,
so serve from the REPO ROOT:

    cd $LAB/diffusion-research && python3 -m http.server 8890
    -> http://localhost:8890/outputs/reports/ladder2/index.html

The data model has exactly two levels and no joins at view time:

    card  = one INPUT (input_key: endpoint + prompt + sidedness + reference)
    gen   = one arm's answer to that input, metrics already averaged over pool refs and seeds

Every treatment gen carries its base twin's numbers inline (`b_*`), so the viewer never has to
re-derive the keyed join that `run_eval.py` already enforces. Statistics are recomputed in the
browser from the selected gens — the same formulas as `run_eval.report()`, kept in one JS
function so there is one place to read them.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
REPO_ROOT = LADDER.parents[1]
sys.path.insert(0, str(LADDER))

import encode_conditioning as ec  # noqa: E402
import prompts  # noqa: E402
import report_full as rf  # noqa: E402
import run_eval  # noqa: E402

STD = REPO_ROOT / "data/processed/transitions_std121"
GENS = REPO_ROOT / "outputs/videos/ladder2"
SEEDS = (42, 43)

#: metric -> (label, direction, decimals, group). Groups become the collapsible tables.
METRICS = [
    ("app_ref", "M1a app_ref", "up", 3, "ref"),
    ("cam_zpr", "M1b cam_zpr", "down", 3, "ref"),
    ("obj_csls", "M1c obj_csls", "down", 3, "ref"),
    ("copy_max", "M2a copy_max", "down", 3, "ref"),
    ("cam_dtw", "cam_dtw", "up", 3, "ref"),
    ("cam_corr", "cam_corr", "up", 3, "ref"),
    ("obj_match", "obj_match", "up", 3, "ref"),
    ("app_ref_v3", "app_ref_v3", "up", 3, "ref"),
    ("cross", "cross", "info", 3, "ref"),
    ("margin", "M2b margin", "up", 3, "gen"),
    ("app_target", "app_target", "up", 3, "gen"),
    ("prefix_dino", "M3a prefix_dino", "up", 3, "gen"),
    ("prefix_lpips", "M3a prefix_lpips", "down", 4, "gen"),
    ("max_seam_z", "M3b max_seam_z", "down", 2, "gen"),
    ("scalar_depth", "depth", "info", 3, "gen"),
    ("scalar_depart", "depart", "info", 3, "gen"),
    ("scalar_arrive", "arrive", "info", 3, "gen"),
    ("scalar_core_frac", "core_frac", "info", 3, "gen"),
]
FLAGS = [("near_copy", "near_copy"), ("cross_high", "cross_high"),
         ("app_saturated", "app_sat"), ("core_degenerate", "core_degen"),
         ("intruder", "intruder")]

NOVELTY_ORDER = ["none", "seen", "unseen", "zero_shot"]
CONTENT_ORDER = ["same", "cross", "foreign"]
NOVELTY_LABEL = {"none": "no reference<br><span>specialist</span>",
                 "seen": "seen<br><span>trained demo</span>",
                 "unseen": "unseen<br><span>new demo, trained class</span>",
                 "zero_shot": "zero-shot<br><span>held-out class</span>"}
CONTENT_LABEL = {"same": "same<br><span>endpoint = donor class</span>",
                 "cross": "cross<br><span>endpoint from another class</span>",
                 "foreign": "foreign<br><span>DAVIS footage</span>"}


def rel(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT))


def clip_video(clip: str) -> str | None:
    """Full corpus clip. DAVIS pseudo-clips have no corpus mp4 — only cut windows exist."""
    if prompts.is_davis(clip):
        return None
    p = STD / prompts.clip_class(clip) / f"{clip}.mp4"
    return rel(p) if p.exists() else None


def per_item_metrics() -> dict[str, dict]:
    """item_id -> every metric, averaged pool-refs -> seeds. Reuses report_full's loaders so the
    viewer and the report can never disagree about a number."""
    registry = {r["item_id"]: r for r in run_eval.load_registry()}
    ceil = run_eval.ceilings()
    scored = rf.load_scored()

    acc: dict[str, dict[str, list]] = {}
    for (item, _seed), rows in scored.items():
        if item not in registry:
            continue
        d = acc.setdefault(item, collections.defaultdict(list))
        c = rf.collapse(rows)
        for k, v in c.items():
            d[k].append(v)
        cls = registry[item]["gt_pool_class"]
        if cls in ceil and "app_ref" in c:
            d["pct"].append(c["app_ref"] / ceil[cls])
    return {k: {m: rf.mean_or_nan(v) for m, v in d.items()} for k, d in acc.items()}


def build() -> dict:
    rows = run_eval.load_registry()
    by_id = {r["item_id"]: r for r in rows}
    ceil = run_eval.ceilings()
    metrics = per_item_metrics()
    base_by_key = {r["input_key"]: r["item_id"] for r in rows if r["arm"] == "base"}

    def gen_entry(r: dict) -> dict | None:
        m = metrics.get(r["item_id"])
        if m is None:
            return None
        vids = {}
        for s in SEEDS:
            p = GENS / r["arm"] / f"{r['item_id']}__s{s}.mp4"
            if p.exists():
                vids[str(s)] = rel(p)
        cond = ("none" if r.get("conditioning") == "none" or not r["endpoint"]
                else "prefix+suffix" if r["sided"] == "two" else "prefix")
        e = {
            "id": r["item_id"], "arm": r["arm"], "cell": r["cell"], "videos": vids,
            "novelty": r["ref_novelty"], "content": r["content"], "donor": r["donor_class"],
            "pct_type": r["pct_type"], "cond": cond, "ref": r.get("reference"),
            "mismatched_ref": bool(r.get("mismatched_reference")),
            "ceil": ceil.get(r["gt_pool_class"]),
            "tier": ("base" if r["arm"] == "base" else "floor" if r["arm"] == "text_floor"
                     else "generalist" if r["arm"] == "ic_gen" else "specialist"),
        }
        e["m"] = {k: (None if m.get(k) is None or m.get(k) != m.get(k) else round(m[k], 6))
                  for k, _l, _d, _dp, _g in METRICS}
        e["f"] = {k: (None if m.get(k) is None or m.get(k) != m.get(k) else round(m[k], 4))
                  for k, _l in FLAGS}
        e["pct"] = None if m.get("pct") != m.get("pct") else round(m.get("pct", float("nan")), 6)
        # the keyed join, resolved once at build time
        twin = base_by_key.get(r["input_key"])
        if r["arm"] not in ("base", "text_floor") and twin and twin in metrics:
            bp = metrics[twin].get("pct")
            e["b_pct"] = None if bp != bp else round(bp, 6)
            e["b_id"] = twin
        return e

    cards: dict[str, dict] = {}
    for r in rows:
        g = gen_entry(r)
        if g is None:
            continue
        key = r["input_key"]
        if key not in cards:
            ep, sided = r["endpoint"], r["sided"]
            pre = suf = None
            if ep:
                paths = ec.cond_paths(ep, sided)
                pre = rel(paths["prefix"])
                suf = rel(paths["suffix"]) if sided == "two" else None
            cards[key] = {
                "key": key, "prompt": r["prompt"], "endpoint": ep, "sided": sided,
                "endpoint_class": r.get("endpoint_class"),
                "endpoint_split": r.get("endpoint_split"),
                "prefix_video": pre, "suffix_video": suf,
                "endpoint_video": clip_video(ep) if ep else None,
                "reference": r.get("reference"),
                "reference_class": prompts.clip_class(r["reference"]) if r.get("reference") else None,
                "reference_video": clip_video(r["reference"]) if r.get("reference") else None,
                "gens": [],
            }
        cards[key]["gens"].append(g)

    tier_rank = {"floor": 0, "base": 1, "specialist": 2, "generalist": 3}
    ordered = []
    for c in cards.values():
        c["gens"].sort(key=lambda g: (tier_rank[g["tier"]], g["arm"]))
        ordered.append(c)
    # hardest first: zero-shot before unseen before seen; foreign before cross before same
    nrank = {n: i for i, n in enumerate(reversed(NOVELTY_ORDER))}
    crank = {c: i for i, c in enumerate(reversed(CONTENT_ORDER))}
    def card_sort(c):
        t = [g for g in c["gens"] if g["tier"] != "base"]
        t = t or c["gens"]
        return (min(nrank.get(g["novelty"], 9) for g in t),
                min(crank.get(g["content"], 9) for g in t), c["key"])
    ordered.sort(key=card_sort)

    treatments = [g for c in ordered for g in c["gens"] if g["tier"] not in ("base", "floor")]
    matrix = collections.Counter((g["novelty"], g["content"]) for g in treatments)

    return {
        "meta": {
            "ladder": "ladder2",
            "instrument": "transition_eval 4.0.0-draft.1 (m1a_S3)",
            "registry_rows": len(rows), "generations": sum(
                len(g["videos"]) for c in ordered for g in c["gens"]),
            "cards": len(ordered), "seeds": list(SEEDS),
            "px_prefix": ec.PX_PREFIX, "suffix_gen_frames": ec.SUFFIX_GEN_FRAMES,
            "frames": 121,
        },
        "metrics": [{"k": k, "label": l, "dir": d, "dp": dp, "group": g}
                    for k, l, d, dp, g in METRICS],
        "flags": [{"k": k, "label": l} for k, l in FLAGS],
        "novelty_order": NOVELTY_ORDER, "content_order": CONTENT_ORDER,
        "novelty_label": NOVELTY_LABEL, "content_label": CONTENT_LABEL,
        "matrix": {f"{n}|{c}": matrix.get((n, c), 0) for n in NOVELTY_ORDER for c in CONTENT_ORDER},
        "ceilings": {k: round(v, 6) for k, v in ceil.items()},
        "cards": ordered,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/reports/ladder2/index.html")
    args = ap.parse_args()
    data = build()
    tpl = (HERE / "template.html").read_text()
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    m = data["meta"]
    print(f"[viewer] {m['cards']} cards, {m['generations']} videos -> {args.out} "
          f"({out.stat().st_size / 1e6:.1f} MB)")
    print(f"[viewer] serve from the repo root:  python3 -m http.server 8890")


if __name__ == "__main__":
    main()
