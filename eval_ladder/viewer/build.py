"""ladder2 — build the results viewer.

    python eval_ladder/viewer/build.py [--out outputs/reports/ladder2/index.html]

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
REPO_ROOT = LADDER.parents[0]
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
    def tier_of(r: dict) -> str:
        """The five tiers the viewer aligns into fixed columns. The two clean BASELINES are
        prompt-only and prompt+endpoint (sidedness-aware, NO reference). base WITH a reference is
        NOT a baseline — it is a copier (it reproduces the demo), shown separately and labelled."""
        arm = r["arm"]
        if arm in ("text_floor", "base_prompt"):
            return "prompt_only"
        if arm == "base_cond" or (arm == "base" and not r.get("reference")):
            return "prompt_endpoint"
        if arm == "base":                       # base WITH reference
            return "copier"
        if arm == "ic_gen":
            return "generalist"
        if arm.startswith("spec_"):
            return "specialist"
        return "other"

    def gen_entry(r: dict) -> dict | None:
        m = metrics.get(r["item_id"])
        if m is None:
            return None
        # a baseline video is shared per (endpoint, sided) via video_key; treatments use item_id
        vkey = r.get("video_key", r["item_id"])
        vids = {}
        for s in SEEDS:
            p = GENS / r["arm"] / f"{vkey}__s{s}.mp4"
            if p.exists():
                vids[str(s)] = rel(p)
        cond = ("none" if r.get("conditioning") == "none" or not r["endpoint"]
                else "prefix+suffix" if r["sided"] == "two" else "prefix")
        e = {
            "id": r["item_id"], "arm": r["arm"], "cell": r["cell"], "videos": vids,
            "novelty": r["ref_novelty"], "content": r["content"], "donor": r["donor_class"],
            "pct_type": r["pct_type"], "cond": cond, "ref": r.get("reference"),
            "mismatched_ref": bool(r.get("mismatched_reference")),
            "ceil": ceil.get(r["gt_pool_class"]), "tier": tier_of(r),
        }
        e["m"] = {k: (None if m.get(k) is None or m.get(k) != m.get(k) else round(m[k], 6))
                  for k, _l, _d, _dp, _g in METRICS}
        e["f"] = {k: (None if m.get(k) is None or m.get(k) != m.get(k) else round(m[k], 4))
                  for k, _l in FLAGS}
        e["pct"] = None if m.get("pct") != m.get("pct") else round(m.get("pct", float("nan")), 6)
        return e

    # ---- index the clean baselines by (donor, endpoint, sided): the video is content-identical
    # across donors, but its pool-% is scored against THIS card's donor pool, so key by donor. ----
    prompt_only_idx: dict[tuple, dict] = {}
    prompt_end_idx: dict[tuple, dict] = {}
    for r in rows:
        t = tier_of(r)
        if t not in ("prompt_only", "prompt_endpoint") or not r["endpoint"]:
            continue
        g = gen_entry(r)
        if g is None:
            continue
        key = (r["donor_class"], r["endpoint"], r["sided"])
        idx = prompt_only_idx if t == "prompt_only" else prompt_end_idx
        # prefer the dedicated v2.1.0 arms; only fall back to an old base-no-ref twin
        if key not in idx or r["arm"] in ("base_prompt", "base_cond"):
            idx[key] = g

    # ---- a CARD is one transition applied to one endpoint: (donor, endpoint, sided) ----------
    # slots hold the four aligned tiers + the copier; each is a list (usually one gen).
    def new_card(r: dict) -> dict:
        ep, sided = r["endpoint"], r["sided"]
        paths = ec.cond_paths(ep, sided)
        return {
            "key": f"{r['donor_class']}|{ep}|{sided}",
            "donor": r["donor_class"], "endpoint": ep, "sided": sided,
            "prompt": r["prompt"],
            "endpoint_class": r.get("endpoint_class"), "endpoint_split": r.get("endpoint_split"),
            "prefix_video": rel(paths["prefix"]),
            "suffix_video": rel(paths["suffix"]) if sided == "two" else None,
            "endpoint_video": clip_video(ep),
            "slots": {"prompt_only": [], "prompt_endpoint": [],
                      "specialist": [], "generalist": [], "copier": []},
        }

    cards: dict[str, dict] = {}
    for r in rows:
        if tier_of(r) not in ("specialist", "generalist", "copier"):
            continue                                    # baselines are attached, not carded
        g = gen_entry(r)
        if g is None:
            continue
        key = f"{r['donor_class']}|{r['endpoint']}|{r['sided']}"
        card = cards.get(key) or cards.setdefault(key, new_card(r))
        # carry the generalist's / copier's reference demo for its conditioning ribbon
        if r.get("reference"):
            g["ref_class"] = prompts.clip_class(r["reference"])
            g["ref_video"] = clip_video(r["reference"])
        card["slots"][g["tier"]].append(g)

    # attach the clean baselines to every card by (donor, endpoint, sided)
    for card in cards.values():
        k = (card["donor"], card["endpoint"], card["sided"])
        if k in prompt_only_idx:
            card["slots"]["prompt_only"] = [prompt_only_idx[k]]
        if k in prompt_end_idx:
            card["slots"]["prompt_endpoint"] = [prompt_end_idx[k]]

    # order cards hardest-first by the treatments they carry
    nrank = {n: i for i, n in enumerate(reversed(NOVELTY_ORDER))}
    crank = {c: i for i, c in enumerate(reversed(CONTENT_ORDER))}
    def card_sort(c):
        t = c["slots"]["generalist"] + c["slots"]["specialist"]
        return (min((nrank.get(g["novelty"], 9) for g in t), default=9),
                min((crank.get(g["content"], 9) for g in t), default=9), c["key"])
    ordered = sorted(cards.values(), key=card_sort)

    treatments = [g for c in ordered for s in ("specialist", "generalist")
                  for g in c["slots"][s]]
    matrix = collections.Counter((g["novelty"], g["content"]) for g in treatments)

    n_vid = sum(len(g["videos"]) for c in ordered for s in c["slots"].values() for g in s)
    return {
        "meta": {
            "ladder": "ladder2",
            "instrument": "transition_eval 4.0.0 (m1a_S3)",
            "registry_rows": len(rows), "generations": n_vid,
            "cards": len(ordered), "seeds": list(SEEDS),
            "px_prefix": ec.PX_PREFIX, "suffix_gen_frames": ec.SUFFIX_GEN_FRAMES,
            "frames": 121,
            "tiers": ["prompt_only", "prompt_endpoint", "specialist", "generalist", "copier"],
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
    out = REPO_ROOT / args.out
    # Video src is repo-root-relative, but the browser resolves it against the HTML's own URL.
    # Prepend the relative hop from the viewer back to the repo root so paths resolve whether the
    # page is served over http from the repo root OR opened directly as a file://.
    depth = len(out.parent.relative_to(REPO_ROOT).parts)
    data["meta"]["rel"] = "../" * depth
    tpl = (HERE / "template.html").read_text()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    m = data["meta"]
    print(f"[viewer] {m['cards']} cards, {m['generations']} videos -> {args.out} "
          f"({out.stat().st_size / 1e6:.1f} MB)")
    print(f"[viewer] serve from the repo root:  python3 -m http.server 8890")


if __name__ == "__main__":
    main()
