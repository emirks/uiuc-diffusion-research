#!/usr/bin/env python
"""EffectData gapper screen — build everything the prompt-only screen needs from selection.json.

  media     standardise every needed clip (reference, 3 endpoints, pool) to native-81f 480x640 under
            data/processed/transitions_std121/ed.<Effect>/ + the frame-0 conditioning window (prepare_media.do_effectdata,
            one worker per EFFECT so a zip is never read concurrently); writes media_manifest.json
  clauses   gemini-3.6-flash effect clause for every reference std clip (build_ref_effects_v3.one; append-only into
            misc/refvfx_baseline/reference_effects.json — an existing clause is never re-rolled)
  rows      the stamped registry for arm base_cond_effect_edscreen (3 same-content rows per effect, prompt "{S1}. {clause}.",
            token-free like every base_cond effect row) + the prompt family file + the arms.yaml entry
  manifest  the screen SUPERSET corpus manifest (src/diffusion/transition_eval/build_corpus_manifest.py --out ...)
Run in that order (media before clauses: gemini watches the STANDARD clip).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/grid_v3")); sys.path.insert(0, str(REPO / "eval_ladder"))
CAMP = REPO / "misc/2026-09-08_ed_gapper_screen"
SEL = CAMP / "selection.json"
ARM = "base_cond_effect_edscreen"
REG = REPO / f"eval_ladder/registry_{ARM}.jsonl"
PY = "/taiga/illinois/eng/cs/jrehg/users/emirkisa/envs-aarch64/ltx2/bin/python"


def selection():
    return json.loads(SEL.read_text())["effects"]


def needed_stems(effects):
    for d in effects:
        yield d["reference"]
        for e in d["endpoints"]:
            yield e["std"]
        yield from d["pool"]


def media(workers: int):
    import prepare_media as PM
    cfg = PM.GRID["effectdata"]; pref = cfg["clip_prefix"]
    ann = json.loads((REPO / cfg["annotations"]).read_text()); recs = list(ann.values()) if isinstance(ann, dict) else ann
    member = {}
    for r in recs:
        fn = r["video_path"].rsplit("/", 1)[-1][:-4]; parts = fn.split(",")
        if len(parts) == 3:
            member[f"{pref}.{parts[0].replace('-', '_')}.{parts[1]}.{parts[2]}"] = (parts[0], r["video_path"])
    effects = selection()
    per_effect = {}
    for d in effects:
        stems = sorted({d["reference"], *[e["std"] for e in d["endpoints"]], *d["pool"]})
        per_effect[d["effect"]] = [{"cls": d["cls"], "stem": s, "effect": member[s][0], "member": member[s][1],
                                    "raw_dir": PM.RAW_TREE / "onesided_transitions" / f"onesided_object_{d['cls']}", "sided": "onesided", "n_out": None} for s in stems]
    todo = {e: [j for j in js if not (PM.STD / j["cls"] / f"{j['stem']}.mp4").exists() or not (PM.ec.CONDS / f"{j['stem']}_start9.mp4").exists()] for e, js in per_effect.items()}
    n_all = sum(len(js) for js in per_effect.values()); n_todo = sum(len(js) for js in todo.values())
    print(f"[media] {len(per_effect)} effects, {n_all} clips, {n_todo} to standardise", flush=True)

    def one_effect(e):
        zips = {}; out = []
        try:
            for j in todo[e]:
                try:
                    out.append(PM.do_effectdata(j, zips))
                except Exception as ex:  # noqa: BLE001
                    out.append({"stem": j["stem"], "error": f"{type(ex).__name__}: {ex}"})
        finally:
            for z in zips.values():
                z.close()
        return e, out
    results = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (e, out) in enumerate(ex.map(one_effect, [e for e in todo if todo[e]])):
            results[e] = out
            if i % 20 == 0:
                print(f"  ..{i} effects done", flush=True)
    errors = [o for outs in results.values() for o in outs if "error" in o]
    (CAMP / "media_manifest.json").write_text(json.dumps({"effects": {e: [j["stem"] for j in js] for e, js in per_effect.items()}, "results": results, "errors": errors}, indent=1))
    missing = [s for d in effects for s in needed_stems([d]) if not (PM.STD / d["cls"] / f"{s}.mp4").exists()]
    print(f"[media] done; errors {len(errors)}; still missing {len(missing)} std clips")


def clauses(workers: int):
    import build_ref_effects_v3 as BR
    out = json.loads(BR.DST.read_text()) if BR.DST.exists() else {}
    refs = sorted({d["reference"] for d in selection()})
    todo = [c for c in refs if c not in out]; ready = [c for c in todo if BR.std_path(c) is not None]
    print(f"[clauses] {len(refs)} references; {len(out)} clauses exist; {len(todo)} to do, {len(ready)} with a std clip -> {BR.MODEL}", flush=True)
    fails = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for clip, effect, status in ex.map(BR.one, ready):
            if effect is None:
                fails.append((clip, status)); print(f"  FAIL {clip}: {status}")
            else:
                out[clip] = effect
    BR.DST.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"[clauses] wrote {len(out)} clauses; failed {len(fails)}; without std clip {len(todo) - len(ready)}")


def lower_initial(s: str) -> str:
    return s[:1].lower() + s[1:] if s else s


def rows():
    import prompts as P  # noqa: F401  (eval_ladder) — kept for the token constant if ever needed
    cfg = yaml.safe_load((REPO / "eval_ladder/grid_v3.yaml").read_text())["effectdata"]
    caps = json.loads((REPO / cfg["captions"]).read_text())["descriptions"]
    cl = json.loads((REPO / "misc/refvfx_baseline/reference_effects.json").read_text())
    out, fam, skipped = [], [], []
    for d in selection():
        clause = cl.get(d["reference"])
        if not clause:
            skipped.append(d["effect"]); continue
        for ep in d["endpoints"]:
            s1 = caps[f"{ep['subject']}|A"].strip().rstrip(".")
            prompt = f"{s1}. {lower_initial(clause.strip().rstrip('.'))}."
            key = hashlib.sha1(f"{ep['std']}|{d['reference']}|one".encode()).hexdigest()[:16]
            out.append({"arm": ARM, "cell": "G-zs-same", "conditioning": "prefix", "content": "same", "donor_class": d["cls"],
                        "endpoint": ep["std"], "endpoint_class": d["cls"], "endpoint_source": "effectdata", "endpoint_split": "test",
                        "gt_pool_class": d["cls"], "input_key": key, "item_id": f"G-zs-same__{ARM}__{ep['std']}__ref_{d['reference']}",
                        "mismatched_reference": False, "no_twin": True, "pct_type": "same", "priority": "P1", "prompt": prompt,
                        "ref_novelty": "zero_shot", "reference": d["reference"], "reference_split": "train", "sided": "one", "use_reference": False,
                        "screen_role": d["role"]})
            fam.append({"item_id": out[-1]["item_id"], "prompt": prompt, "S1": s1, "clause": clause})
    REG.write_text("".join(json.dumps(r) + "\n" for r in out))
    (CAMP / "prompts_effect_edscreen.jsonl").write_text("".join(json.dumps(r) + "\n" for r in fam))
    sha = hashlib.sha256("".join(r["prompt"] + "\n" for r in out).encode()).hexdigest()[:12]
    print(f"[rows] {len(out)} rows -> {REG.relative_to(REPO)} (prompt corpus sha {sha}); effects without clause skipped: {len(skipped)}")
    arms_p = REPO / "eval_ladder/arms.yaml"; txt = arms_p.read_text()
    if f"\n  {ARM}:" not in txt:
        block = (f"  {ARM}:\n    kind: base\n    adapter: null\n    targets: null\n"
                 f'    note: "EffectData gapper screen (misc/2026-09-08_ed_gapper_screen): prompt-only effect arm, 300 effects x 3 same rows, seed 42; native 81 f + frame-0 anchor: GEN_FRAMES=81 GEN_PREFIX_FRAMES=1; prompts_effect_edscreen sha {sha}"\n')
        anchor = "  base_cond_effect_v3ed81:"
        i = txt.index(anchor); j = txt.index("\n  ", i + len(anchor)) + 1   # insert after that entry
        arms_p.write_text(txt[:j] + block + txt[j:]); print("[rows] arms.yaml entry added")
    else:
        print("[rows] arms.yaml entry exists")


def manifest():
    out = CAMP / "corpus_manifest_screen.json"
    cmd = [PY, str(REPO / "src/diffusion/transition_eval/build_corpus_manifest.py"), "--repo", str(REPO), "--out", str(out), "--allow-partial", "--no-probe"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-800:])
    m = json.loads(out.read_text()); scr = {d["cls"] for d in selection()}
    print(f"[manifest] {m['n_clips']} clips / {m['n_classes']} classes -> {out.relative_to(REPO)}; screen classes present: {len(scr & set(m['classes']))}/{len(scr)}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["media", "clauses", "rows", "manifest"]); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    {"media": lambda: media(a.workers), "clauses": lambda: clauses(a.workers), "rows": rows, "manifest": manifest}[a.cmd]()


if __name__ == "__main__":
    main()
