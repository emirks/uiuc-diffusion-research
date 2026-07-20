"""Build the ladder results viewer (single self-contained HTML + embedded data).

Joins all certified ladder_v3 items.jsonl rows with the exp_066 eval manifests
(video paths, endpoint conditioning clips, in-context reference demos), groups
rows into FAMILIES — (style, endpoint clip, seed) — the paired-twin comparison
unit, and emits outputs/reports/ladder_viewer/index.html.

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
EVAL_ROOT = os.path.join(REPO, "outputs/eval/ladder_v3")
MANIFEST_DIR = os.path.join(REPO, "experiments/exp_066_ladder_v3_scoring/dataset")
TRUST_MAP = os.path.join(REPO, "outputs/eval/certification/3.0.0-draft.8/exam/trust_map.json")
OUT_DIR = os.path.join(REPO, "outputs/reports/ladder_viewer")
TAU_COPY = 0.858  # amendment-1; embedded near_copy flags used draft 0.88 -> reflag

METRICS = ["margin", "app_ref", "app_target", "cam_dtw", "obj_match", "copy_max", "max_seam_z"]
MDE = {"margin": 0.037, "app_ref": 0.024, "cam_dtw": 0.076,
       "obj_match": 0.008, "copy_max": 0.022, "max_seam_z": 0.27}
LOWER_BETTER = ["cam_dtw", "max_seam_z", "copy_max"]
TRUST_KEY = {"app_ref": "m1a", "cam_dtw": "m1b", "obj_match": "m1c", "margin": "m2b"}

ARM_META = {  # canonical order, display name, group
    "r0":           ["Base · prompt only (P)", "base"],
    "r1":           ["Base · +endpoints (PE)", "base"],
    "r1k":          ["Base · +keyword", "base"],
    "r1k_ext":      ["Base · +keyword (ext)", "base"],
    "r2_ckpt250":   ["Specialist SEEN @250", "spec"],
    "r2_ckpt2000":  ["Specialist SEEN @2000", "spec"],
    "r3_ckpt250":   ["Specialist UNSEEN @250", "spec"],
    "r3_ckpt2000":  ["Specialist UNSEEN @2000", "spec"],
    "r3x":          ["Specialist FOREIGN", "spec"],
    "ic2_r4":       ["ic2 legacy · unseen", "ic"],
    "ic2_r5":       ["ic2 legacy · zero-shot", "ic"],
    "ic3_a":        ["IC-LoRA · held-in (A)", "ic"],
    "ic3_b":        ["IC-LoRA · unseen (B)", "ic"],
    "ic3_c":        ["IC-LoRA · zero-shot (C)", "ic"],
    "ic3_x":        ["IC-LoRA · foreign (X)", "ic"],
}
ARM_ORDER = {a: i for i, a in enumerate(ARM_META)}

PRESETS = [
    {"id": "all", "name": "All arms (browse)", "arms": [], "anchor": None, "cross": False,
     "note": "Every family; anchor defaults to Base+endpoints when present."},
    {"id": "c1", "name": "C1 · What endpoint conditioning buys", "arms": ["r0", "r1"],
     "anchor": "r0", "cross": False,
     "note": "Base prompt-only vs base + endpoint anchors, identical items."},
    {"id": "c4", "name": "C4 · Specialist value (own unseen items)",
     "arms": ["r1", "r3_ckpt250", "r3_ckpt2000"], "anchor": "r1", "cross": False,
     "note": "Per-class weights vs keyed base on the class's own test clips."},
    {"id": "c5", "name": "C5 · Generalist vs specialist (PRIMARY)",
     "arms": ["r3_ckpt2000", "ic3_b"], "anchor": "r3_ckpt2000", "cross": False,
     "note": "Same unseen items: one in-context generalist vs the class's own specialist."},
    {"id": "c8", "name": "C8 · Generalist value over base",
     "arms": ["r1", "ic3_b"], "anchor": "r1", "cross": False,
     "note": "IC-LoRA unseen-tier vs keyed base, identical items."},
    {"id": "c67", "name": "C6/C7 · Zero-shot (holdout classes)",
     "arms": ["r1", "ic3_c", "ic2_r5"], "anchor": "r1", "cross": False,
     "note": "Holdout classes never trained. Margin partially anti-rewards reference-following here (Amendment 1) — trust your eyes + cam classes."},
    {"id": "c9", "name": "C9 · Foreign-endpoint effect transfer",
     "arms": ["r3x", "ic3_x"], "anchor": "r3x", "cross": False,
     "note": "Donor transition on recipient-class endpoints, prefix-only. Specialist collapses; ic3 reads the reference."},
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
    trust = {cls: {met: bool(v.get(k, False)) for met, k in TRUST_KEY.items()}
             for cls, v in trust_raw.items()}

    manifests = {}  # (label, item_id) -> manifest item
    for f in sorted(glob.glob(os.path.join(MANIFEST_DIR, "eval_*.json"))):
        label = os.path.basename(f)[len("eval_"):-len(".json")]
        for it in json.load(open(f)):
            manifests[(label, it["item_id"])] = it

    fams = {}
    n_rows = n_ctrl = n_skip = 0
    ctrl_seen = set()
    for f in sorted(glob.glob(os.path.join(EVAL_ROOT, "*", "items.jsonl"))):
        label = os.path.basename(os.path.dirname(f))
        for line in open(f):
            r = json.loads(line)
            arm, iid = r["arm"], r["item_id"]
            is_ctrl = arm.startswith("control")
            key_id = iid.split("__", 1)[1] if is_ctrl else iid
            parsed = parse_item_id(key_id)
            if parsed is None or arm == "sigma_hero_recheck":
                n_skip += 1
                continue
            style, clip, seed = parsed
            fk = f"{style}|{clip}|{seed}"
            fam = fams.setdefault(fk, {
                "key": fk, "style": style, "clip": clip, "seed": seed,
                "tags": r.get("tags") or [], "sidedness": r.get("sidedness"),
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
                row["refOwn"] = rv_name == f"{clip}.mp4"
                row["v"] = rel(man["generated_video"])
                cp, cs = man.get("condition_prefix"), man.get("condition_suffix")
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
        if "ic3_c" in have or "ic2_r5" in have:
            fam["band"] = "holdout"
        elif "r3x" in have or "ic3_x" in have:
            fam["band"] = "foreign"
        elif "r2_ckpt2000" in have or "r2_ckpt250" in have or "ic3_a" in have:
            fam["band"] = "train"
        elif "r3_ckpt2000" in have or "r3_ckpt250" in have or "ic3_b" in have or "ic2_r4" in have:
            fam["band"] = "test"
        else:
            fam["band"] = "canonical"

    families = sorted(fams.values(), key=lambda f: (f["style"], f["clip"], f["seed"]))
    data = {"families": families, "trust": trust, "mde": MDE,
            "lowerBetter": LOWER_BETTER, "armMeta": ARM_META, "presets": PRESETS,
            "tauCopy": TAU_COPY}

    def pre(path):
        p = os.path.join(REPO, path)
        return html.escape(open(p).read()) if os.path.exists(p) else "(missing)"

    tpl = open(os.path.join(os.path.dirname(__file__), "viewer_template.html")).read()
    out = (tpl.replace("__DATA_JSON__", json.dumps(data, allow_nan=False))
              .replace("__PRE_TIER__", pre("outputs/eval/ladder_v3/_contrasts/tier_table.md"))
              .replace("__PRE_CONTRASTS__", pre("outputs/eval/ladder_v3/_contrasts/contrasts.md")))
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "index.html")
    open(out_path, "w").write(out)

    bands = collections.Counter(f["band"] for f in families)
    print(f"[done] {out_path}")
    print(f"  families={len(families)}  gen-rows={n_rows}  control-rows={n_ctrl}  skipped={n_skip}")
    print(f"  bands: {dict(bands)}")
    missing = sum(1 for f in families for a in f["arms"]
                  if a.get("v") and not os.path.exists(os.path.join(REPO, a["v"].lstrip("/"))))
    novid = sum(1 for f in families for a in f["arms"] if not a.get("v"))
    print(f"  videos missing on disk: {missing}   rows without manifest video: {novid}")


if __name__ == "__main__":
    sys.exit(main())
