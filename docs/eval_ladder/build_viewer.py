"""Build the ladder results viewer (single self-contained HTML + embedded data).

v4-instrument edition (owner directive 2026-07-21: v4 default for all lanes).
Joins ladder_v4h items.jsonl rows with the exp_066 eval manifests (video paths,
endpoint conditioning clips, in-context reference demos) and the exp_072 pool
yardstick index (pool-mean app_ref + per-class GT ceilings -> achieved-%),
groups rows into FAMILIES — (style, endpoint clip, seed) — and emits
outputs/reports/ladder_viewer/index.html.

Reference semantics carried per row (this is what 'which reference' means):
  - row app_ref:   base/specialists -> the item's OWN GT clip (content-matched,
                   inflated, copy_max saturated); IC arms -> the DEMO clip;
                   foreign arms -> a DONOR-class clip.
  - pool/pct:      exp_072 pool lane — mean over the same-class reference POOL
                   (own clip + original demo excluded), divided by the class GT
                   ceiling (same-class off-diagonal mean of the certified v4
                   pairwise matrix). One uniform setting for every arm.

Serve from the repo root so root-relative video paths resolve:
    cd $LAB/diffusion-research && python3 -m http.server 8890
    -> http://localhost:8890/outputs/reports/ladder_viewer/index.html

Usage: python3 docs/eval_ladder/build_viewer.py
"""
import collections
import glob
import html
import json
import math
import os
import re
import sys

REPO = "/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research"
EVAL_ROOT = os.path.join(REPO, "outputs/eval/ladder_v4h")
MANIFEST_DIR = os.path.join(REPO, "experiments/exp_066_ladder_v3_scoring/dataset")
TRUST_MAP = os.path.join(
    REPO, ".claude/worktrees/eval-v4-cert/outputs/eval/certification/4.0.0-draft.1/exam/trust_map.json")
POOL_INDEX = os.path.join(REPO, "outputs/eval/exp_072_pool_v4/local_fill/pool_index.json")
OUT_DIR = os.path.join(REPO, "outputs/reports/ladder_viewer")
TAU_COPY = 0.858  # amendment-1 threshold (v3-calibrated; v4 rows reflagged the same way)

METRICS = ["margin", "app_ref", "app_ref_v3", "app_target", "cam_zpr", "cam_dtw",
           "obj_csls", "copy_max", "max_seam_z"]
# indicative delta thresholds for chip coloring: margin/copy/seam are v3-measured
# sigma_seed MDEs; app_ref/pct/cam_zpr are provisional (v4 sigma_seed not yet measured)
MDE = {"margin": 0.037, "app_ref": 0.03, "pct": 0.03, "cam_zpr": 0.05,
       "cam_dtw": 0.076, "copy_max": 0.022, "max_seam_z": 0.27}
LOWER_BETTER = ["cam_dtw", "max_seam_z", "copy_max"]
TRUST_KEY = {"app_ref": "m1a", "pct": "m1a", "cam_zpr": "m1b", "cam_dtw": "m1b",
             "obj_csls": "m1c", "margin": "m2b"}

ARM_META = {  # arm -> [display name, tier group, keyed?]
    "r0":           ["Base · prompt only", "base"],
    "r1":           ["Base · endpoints (P+S always)", "base"],
    "r1k":          ["Base · endpoints KEYED (prefix-only)", "base"],
    "r1k_ext":      ["Base · endpoints KEYED (prefix-only, ext)", "base"],
    "r2_ckpt250":   ["Specialist SEEN @250", "spec"],
    "r2_ckpt2000":  ["Specialist SEEN @2000", "spec"],
    "r3_ckpt250":   ["Specialist UNSEEN @250", "spec"],
    "r3_ckpt2000":  ["Specialist UNSEEN @2000", "spec"],
    "r3x":          ["Specialist FOREIGN", "spec"],
    "ic3_a":        ["IC-LoRA · held-in (A)", "ic"],
    "ic3_b":        ["IC-LoRA · unseen (B)", "ic"],
    "ic3_c":        ["IC-LoRA · zero-shot (C)", "ic"],
    "ic3_x":        ["IC-LoRA · foreign (X)", "ic"],
}
ARM_ORDER = {a: i for i, a in enumerate(ARM_META)}
FOREIGN_ARMS = {"r3x", "ic3_x"}
# arms whose pool-lane rows are still in the queue (empty since fills 9609271-72 landed)
POOL_QUEUED = set()

PRESETS = [
    {"id": "all", "name": "All arms (browse)", "arms": [], "anchor": None, "cross": False,
     "note": "Every family; anchor defaults to the keyed conditioned base when present."},
    {"id": "c1", "name": "C1 · What endpoint conditioning buys", "arms": ["r0", "r1", "r1k", "r1k_ext"],
     "anchor": "r0", "cross": False,
     "note": "Base prompt-only vs base + endpoint anchors, identical items. On one-sided items the conditioned base shown is the KEYED (prefix-only) one."},
    {"id": "c4", "name": "C4 · Specialist value (own unseen items)",
     "arms": ["r1", "r1k", "r1k_ext", "r3_ckpt250", "r3_ckpt2000"], "anchor": "r1", "cross": False,
     "note": "Per-class weights vs conditioned base on the class's own test clips (keyed base on one-sided items)."},
    {"id": "c5", "name": "C5 · Generalist vs specialist (PRIMARY)",
     "arms": ["r3_ckpt2000", "ic3_b"], "anchor": "r3_ckpt2000", "cross": False,
     "note": "Same unseen items: one in-context generalist vs the class's own specialist."},
    {"id": "c8", "name": "C8 · Generalist value over base",
     "arms": ["r1", "r1k", "r1k_ext", "ic3_b"], "anchor": "r1", "cross": False,
     "note": "IC-LoRA unseen-tier vs conditioned base, identical items (keyed base on one-sided)."},
    {"id": "c67", "name": "C6/C7 · Zero-shot (holdout classes)",
     "arms": ["r1", "r1k", "r1k_ext", "ic3_c"], "anchor": "r1", "cross": False,
     "note": "Holdout classes never trained. Margin partially anti-rewards reference-following here (Amendment 1) — trust your eyes + cam classes."},
    {"id": "c9", "name": "C9 · Foreign-endpoint effect transfer",
     "arms": ["r3x", "ic3_x"], "anchor": "r3x", "cross": False,
     "note": "Donor transition on recipient-class endpoints, prefix-only. Ceiling here is a donor-class PROXY — ranking only."},
    {"id": "kb", "name": "K · Keyed vs cracked base (one-sided)",
     "arms": ["r1", "r1k", "r1k_ext"], "anchor": "r1k", "cross": False, "showCracked": True,
     "note": "One-sided items only place r1 is shown: r1 saw the END anchor too (cracked — the outcome was given). r1k is the honest prefix-only base."},
    {"id": "c3", "name": "C3 · Overfit gap (SEEN vs UNSEEN)",
     "arms": ["r2_ckpt2000", "r3_ckpt2000"], "anchor": "r3_ckpt2000", "cross": True,
     "note": "Items DIFFER (train vs test clips) — class-level comparison only, no per-item deltas."},
    {"id": "c11", "name": "C11 · IC-LoRA held-in vs unseen",
     "arms": ["ic3_a", "ic3_b"], "anchor": "ic3_b", "cross": True,
     "note": "Items DIFFER by design (train-band vs test-band endpoints) — class-level only."},
    {"id": "ckpt", "name": "Checkpoint ladder (250 vs 2000)",
     "arms": ["r2_ckpt250", "r2_ckpt2000", "r3_ckpt250", "r3_ckpt2000"],
     "anchor": "r2_ckpt2000", "cross": False, "note": "Specialist training-time effect."},
]


def num(v):
    """NaN/Inf are valid Python-json but invalid JSON — browsers refuse to parse them."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def rel(p):
    if not p:
        return None
    if p.startswith(REPO):
        p = p[len(REPO):]
    return "/" + p.lstrip("/")


def parse_item_id(iid):
    """RUNG__style__clip__sNN[__ckptNNN|__recheck] -> (style, clip, seed) or None."""
    parts = iid.split("__")
    parts = [p for p in parts if not re.fullmatch(r"ckpt\d+|recheck", p)]
    if len(parts) < 4:
        return None
    m = re.fullmatch(r"s(\d+)", parts[3])
    if not m:
        return None
    return parts[1], parts[2], int(m.group(1))


def main():
    trust_raw = json.load(open(TRUST_MAP))
    trust = {cls: {k: bool(v.get(k, False)) for k in ("m1a", "m1b", "m1c", "m2b")}
             for cls, v in trust_raw.items()}

    pool = {"ceilings": {}, "items": {}}
    if os.path.exists(POOL_INDEX):
        pool = json.load(open(POOL_INDEX))
    ceilings = pool.get("ceilings", {})
    pool_items = pool.get("items", {})

    manifests = {}  # (label, item_id) -> manifest item
    for f in sorted(glob.glob(os.path.join(MANIFEST_DIR, "eval_*.json"))):
        label = os.path.basename(f)[len("eval_"):-len(".json")]
        for it in json.load(open(f)):
            manifests[(label, it["item_id"])] = it

    fams = {}
    n_rows = n_ctrl = n_skip = n_pool = 0
    ctrl_seen = set()
    for f in sorted(glob.glob(os.path.join(EVAL_ROOT, "*", "items.jsonl"))):
        label = os.path.basename(os.path.dirname(f))
        for line in open(f):
            r = json.loads(line)
            arm, iid = r["arm"], r["item_id"]
            is_ctrl = arm.startswith("control")
            key_id = iid.split("__", 1)[1] if is_ctrl else iid
            parsed = parse_item_id(key_id)
            if parsed is None or (not is_ctrl and arm not in ARM_META):
                n_skip += 1  # sigma_hero_recheck, ic2 legacy (owner: viewer shows ic3 only)
                continue
            style, clip, seed = parsed
            fk = f"{style}|{clip}|{seed}"
            fam = fams.setdefault(fk, {
                "key": fk, "style": style, "clip": clip, "seed": seed,
                "tags": r.get("tags") or [], "sidedness": r.get("sidedness"),
                "ceil": num(ceilings.get(style)),
                "cond": {}, "refDemo": None, "refEx": None, "arms": [], "controls": []})
            near = (r.get("copy_max") is not None and r["copy_max"] >= TAU_COPY)
            top1 = (r.get("apps_top3") or [[None, None]])[0]
            row = {"a": arm,
                   "m": {k: num(r.get(k)) for k in METRICS},
                   "near": near,
                   "deg": bool(r.get("core_degenerate")),
                   "xh": bool(r.get("cross_high")),
                   "camv": bool(r.get("cam_valid")),
                   "top1": [top1[0], num(top1[1])]}
            pi = pool_items.get(iid)
            if pi is not None:
                row["pool"] = num(pi["pool"])
                row["poolN"] = pi["n"]
                row["poolSrc"] = "h" if pi["src"] == "harness" else "l"
                c = ceilings.get(style)
                row["pct"] = num(pi["pool"] / c) if c else None
                n_pool += 1
            if is_ctrl:
                ck = (fk, arm)
                if ck in ctrl_seen:
                    continue
                ctrl_seen.add(ck)
                fam["controls"].append(row)
                n_ctrl += 1
                continue
            man = manifests.get((label, iid))
            if man:
                rv_name = os.path.basename(man.get("reference_video") or "")
                own = rv_name == f"{clip}.mp4"
                row["refKind"] = ("own" if own else
                                  ("donor" if arm in FOREIGN_ARMS else "demo"))
                row["v"] = rel(man["generated_video"])
                cp, cs = man.get("condition_prefix"), man.get("condition_suffix")
                row["cnd"] = ("PS" if (cp and cs) else "P" if cp else "0")
                if cp and not fam["cond"].get("p"):
                    fam["cond"]["p"] = rel(cp["video"])
                if cs and not fam["cond"].get("s"):
                    fam["cond"]["s"] = rel(cs["video"])
                rv = rel(man.get("reference_video"))
                if arm.startswith(("ic2", "ic3")):
                    fam["refDemo"] = fam["refDemo"] or rv
                else:
                    fam["refEx"] = fam["refEx"] or rv
            if r.get("tags"):
                fam["tags"] = r["tags"]
            fam["arms"].append(row)
            n_rows += 1

    for fam in fams.values():
        fam["arms"].sort(key=lambda x: ARM_ORDER.get(x["a"], 99))
        have = {a["a"] for a in fam["arms"]}
        if "ic3_c" in have:
            fam["band"] = "holdout"
        elif "r3x" in have or "ic3_x" in have:
            fam["band"] = "foreign"
        elif "r2_ckpt2000" in have or "r2_ckpt250" in have or "ic3_a" in have:
            fam["band"] = "train"
        elif "r3_ckpt2000" in have or "r3_ckpt250" in have or "ic3_b" in have:
            fam["band"] = "test"
        else:
            fam["band"] = "canonical"

    families = sorted((f for f in fams.values() if f["arms"]),
                      key=lambda f: (f["style"], f["clip"], f["seed"]))
    data = {"families": families, "trust": trust, "trustKey": TRUST_KEY, "mde": MDE,
            "lowerBetter": LOWER_BETTER, "armMeta": ARM_META, "presets": PRESETS,
            "tauCopy": TAU_COPY, "poolQueued": sorted(POOL_QUEUED),
            "hasPool": bool(pool_items)}

    def pre(path):
        p = os.path.join(REPO, path)
        return html.escape(open(p).read()) if os.path.exists(p) else "(missing)"

    tpl = open(os.path.join(os.path.dirname(__file__), "viewer_template.html")).read()
    out = (tpl.replace("__DATA_JSON__", json.dumps(data, allow_nan=False))
              .replace("__PRE_TIER__", pre("outputs/eval/ladder_v4h/_contrasts/tier_table_v4.md"))
              .replace("__PRE_CONTRASTS__", pre("outputs/eval/ladder_v4h/_contrasts/contrasts_v4.md")))
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "index.html")
    open(out_path, "w").write(out)

    bands = collections.Counter(f["band"] for f in families)
    print(f"[done] {out_path}")
    print(f"  families={len(families)}  gen-rows={n_rows}  control-rows={n_ctrl}  skipped={n_skip}")
    print(f"  bands: {dict(bands)}  rows-with-pool%={n_pool}  ceilings={len(ceilings)}")
    missing = sum(1 for f in families for a in f["arms"]
                  if a.get("v") and not os.path.exists(os.path.join(REPO, a["v"].lstrip("/"))))
    novid = sum(1 for f in families for a in f["arms"] if not a.get("v"))
    print(f"  videos missing on disk: {missing}   rows without manifest video: {novid}")


if __name__ == "__main__":
    sys.exit(main())
