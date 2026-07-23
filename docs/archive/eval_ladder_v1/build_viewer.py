"""Build the ladder side-by-side viewer (flat single-page UI copied from the
well-liked outputs/eval/ladder_v3/_viewer, updated to the v4 instrument +
pool yardstick).

Data per card (cls, endpoints clip, seed):
  - cells per arm: v4 metrics, conditioning (P/PS/0), row-reference kind
    (own GT / demo / donor), pool-mean app_ref + % of class GT ceiling
    (exp_072 index; 'l' = local exact-kernel fill, harness-validated)
  - floors from the synthesized controls (crossfade/hold)
  - start/end anchor cells honour sidedness (one-sided -> start only)

Serve from the repo root:
    cd $LAB/diffusion-research && python3 -m http.server 8890
    -> http://localhost:8890/outputs/reports/ladder_viewer/index.html

Usage: python3 docs/eval_ladder/build_viewer.py
"""
import collections
import glob
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
CORPUS = os.path.join(REPO, "data/processed/transitions_std121/corpus_manifest.json")
OUT_DIR = os.path.join(REPO, "outputs/reports/ladder_viewer")
TAU_COPY = 0.858

ARM_LABEL = {
    "r0": "Base · prompt only",
    "r1": "Base · endpoints (P+S)",
    "r1k": "Base · endpoints KEYED (prefix-only)",
    "r1k_ext": "Base · endpoints KEYED (ext)",
    "r2_ckpt250": "Specialist SEEN @250",
    "r2_ckpt2000": "Specialist SEEN @2000",
    "r3_ckpt250": "Specialist UNSEEN @250",
    "r3_ckpt2000": "Specialist UNSEEN @2000",
    "r3x": "Specialist FOREIGN",
    "ic3_a": "IC-LoRA held-in (A)",
    "ic3_b": "IC-LoRA unseen (B)",
    "ic3_c": "IC-LoRA zero-shot (C)",
    "ic3_x": "IC-LoRA foreign (X)",
}
ARM_TIER = {a: ("base" if a.startswith(("r0", "r1")) else
                "spec" if a.startswith(("r2", "r3")) else "ic") for a in ARM_LABEL}
ARM_ORDER = list(ARM_LABEL)
FOREIGN_ARMS = {"r3x", "ic3_x"}
TRUST_REMAP = {"app_ref": "m1a", "cam_zpr": "m1b", "obj": "m1c", "margin": "m2b"}


def num(v):
    return None if isinstance(v, float) and not math.isfinite(v) else v


def rel(p):
    if not p:
        return None
    if p.startswith(REPO):
        p = p[len(REPO):]
    return "/" + p.lstrip("/")


def parse_item_id(iid):
    parts = [p for p in iid.split("__") if not re.fullmatch(r"ckpt\d+|recheck", p)]
    if len(parts) < 4:
        return None
    m = re.fullmatch(r"s(\d+)", parts[3])
    return (parts[1], parts[2], m.group(1)) if m else None


def load_captions():
    """clip stem -> type-blind caption. Every arm's prompt was 'ICTRANS ' + this
    caption of the ENDPOINTS clip (verified zero mismatches across all gen
    manifests) — prompt parity across rungs is by design (exp_061 README)."""
    caps = {}
    sel = json.load(open(os.path.join(REPO, "experiments/exp_061_ladder_r0_r1/dataset/selection.json")))
    for r in sel["items"]:
        caps[r["clip"]] = r["caption"]
    caps.update(json.load(open(os.path.join(REPO, "experiments/exp_061_ladder_r0_r1/dataset/captions_extra.json"))))
    caps.update(json.load(open(os.path.join(REPO, "experiments/exp_062_ladder_r2r3_specialists/dataset/captions_r2.json"))))
    for f in glob.glob(os.path.join(REPO, "experiments/exp_062_ladder_r2r3_specialists/dataset/manifests/*.json")):
        for r in json.load(open(f)):
            if isinstance(r, dict) and r.get("caption") and r.get("video"):
                caps.setdefault(os.path.splitext(os.path.basename(r["video"]))[0], r["caption"])
    for fn in ("manifest_ic3.json", "manifest_base_ext.json", "manifest_ic3_x.json"):
        for r in json.load(open(os.path.join(REPO, "experiments/exp_065_ladder_v3_grid/dataset", fn)))["rows"]:
            p = r.get("prompt") or ""
            caps.setdefault(r["clip"], p[8:] if p.startswith("ICTRANS ") else p)
    return caps


def main():
    trust_raw = json.load(open(TRUST_MAP))
    trust = {cls: {k: bool(v.get(m, False)) for k, m in TRUST_REMAP.items()}
             for cls, v in trust_raw.items()}
    corpus = json.load(open(CORPUS))
    clip_cls = {os.path.splitext(os.path.basename(k))[0]: m["class"]
                for k, m in corpus["clips"].items()}

    captions = load_captions()
    pool = json.load(open(POOL_INDEX)) if os.path.exists(POOL_INDEX) else {}
    ceilings = pool.get("ceilings", {})
    pool_items = pool.get("items", {})

    manifests = {}
    for f in sorted(glob.glob(os.path.join(MANIFEST_DIR, "eval_*.json"))):
        label = os.path.basename(f)[len("eval_"):-len(".json")]
        for it in json.load(open(f)):
            manifests[(label, it["item_id"])] = it

    cards = {}
    n_rows = n_ctrl = 0
    for f in sorted(glob.glob(os.path.join(EVAL_ROOT, "*", "items.jsonl"))):
        label = os.path.basename(os.path.dirname(f))
        for line in open(f):
            r = json.loads(line)
            arm, iid = r["arm"], r["item_id"]
            is_ctrl = arm.startswith("control")
            key_id = iid.split("__", 1)[1] if is_ctrl else iid
            parsed = parse_item_id(key_id)
            if parsed is None or (not is_ctrl and arm not in ARM_LABEL):
                continue
            cls, clip, seed = parsed
            ck = (cls, clip, seed)
            card = cards.setdefault(ck, {
                "cls": cls, "clip": clip, "seed": seed,
                "sidedness": r.get("sidedness"), "tags": r.get("tags") or [],
                "ceil": num(ceilings.get(cls)),
                "foreign": False, "ep_video": None, "ep_cls": clip_cls.get(clip, cls),
                "cells": {}, "floors": {}, "refs": {}})
            if is_ctrl:
                kind = "lerp" if "lerp" in arm else "hold"
                if kind not in card["floors"]:
                    card["floors"][kind] = {"margin": num(r.get("margin")),
                                            "app_ref": num(r.get("app_ref"))}
                    n_ctrl += 1
                continue
            if r.get("tags"):
                card["tags"] = r["tags"]
            if arm in FOREIGN_ARMS:
                card["foreign"] = True
            cell = {"arm": arm, "item_id": iid,
                    "margin": num(r.get("margin")), "app_ref": num(r.get("app_ref")),
                    "cam_zpr": num(r.get("cam_zpr")), "obj": num(r.get("obj_csls")),
                    "copy_max": num(r.get("copy_max")), "seam": num(r.get("max_seam_z")),
                    "near": (r.get("copy_max") is not None and r["copy_max"] >= TAU_COPY),
                    "deg": bool(r.get("core_degenerate")), "xh": bool(r.get("cross_high"))}
            man = manifests.get((label, iid))
            if man:
                cell["video"] = rel(man["generated_video"])
                cp, cs = man.get("condition_prefix"), man.get("condition_suffix")
                cell["cnd"] = "PS" if (cp and cs) else "P" if cp else "0"
                if cp and not card["ep_video"]:
                    card["ep_video"] = rel(cp["video"])
                rv = man.get("reference_video")
                rv_name = os.path.basename(rv or "")
                own = rv_name == f"{clip}.mp4"
                cell["ref"] = ("own" if own else
                               "donor" if arm in FOREIGN_ARMS else "demo")
                if rv and not own:
                    card["refs"].setdefault(rel(rv), []).append(arm)
            pi = pool_items.get(iid)
            if pi is not None:
                cell["pool"] = num(pi["pool"])
                cell["pooln"] = pi["n"]
                cell["psrc"] = "h" if pi["src"] == "harness" else "l"
                c = ceilings.get(cls)
                cell["pct"] = num(pi["pool"] / c) if c else None
            card["cells"][arm] = cell
            n_rows += 1

    out_cards = []
    for card in cards.values():
        if not card["cells"]:
            continue
        # r0-only cards carry no conditioning video; use the endpoints clip itself
        if not card["ep_video"]:
            card["ep_video"] = f"/data/processed/transitions_std121/{card['ep_cls']}/{card['clip']}.mp4"
        cap = captions.get(card["clip"])
        card["prompt"] = ("ICTRANS " + cap) if cap else None
        card["refs"] = [{"path": p, "arms": sorted(set(a))} for p, a in card["refs"].items()]
        out_cards.append(card)
    out_cards.sort(key=lambda c: (c["cls"], c["clip"], c["seed"]))

    data = {"cards": out_cards, "trust": trust, "mde": {"margin": 0.037},
            "arm_label": ARM_LABEL, "arm_order": ARM_ORDER, "arm_tier": ARM_TIER,
            "tau": TAU_COPY}

    tpl = open(os.path.join(os.path.dirname(__file__), "viewer_template.html")).read()
    out = tpl.replace("__DATA_JSON__", json.dumps(data, allow_nan=False))
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "index.html")
    open(out_path, "w").write(out)

    n_pct = sum(1 for c in out_cards for x in c["cells"].values() if x.get("pct") is not None)
    missing = sum(1 for c in out_cards for x in c["cells"].values()
                  if x.get("video") and not os.path.exists(os.path.join(REPO, x["video"].lstrip("/"))))
    print(f"[done] {out_path}")
    print(f"  cards={len(out_cards)}  cells={n_rows}  ctrl-floors={n_ctrl}  cells-with-pct={n_pct}")
    print(f"  videos missing on disk: {missing}")


if __name__ == "__main__":
    sys.exit(main())
