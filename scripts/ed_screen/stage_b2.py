#!/usr/bin/env python
"""EffectData gapper screen — stage B2 (owner 2026-09-12: "generate the remaining rows with DCG too; max parallelism; only effect").

DCG w=6 EFFECT on the screen's other two rows (endpoints 1 and 2) of the 40 stage-B effects: 80 clips, seed 42, native 81 f +
frame-0 anchor, same adapter/config/prompt rule as stage B (scripts/ed_screen/stage_b.py). Rows 1-2 were never used for the DCG
criterion of the two-criterion shortlist (HARDNESS.md), so they are an out-of-selection check of the row-0 DCG win.

Store: same harness arm `dualforce_dcg_w6_effect_edscreen`; new immutable subentry gens/032_dualforce_dcg_w6/08_effect_edscreen_r12__dai
registered over the 80-row registry (eval_ladder/registry_dualforce_dcg_w6_effect_edscreen_r12.jsonl) + prompts/016; the
COMBINED 120-row registry eval_ladder/registry_dualforce_dcg_w6_effect_edscreen.jsonl drives scoring (evals/029, label r12c*)
through a per-clip gens dir (eval/gens_b2/<arm>/ -> clips of subentries 07 + 08).
  rows -> gen -> register -> plan -> submit -> readout (eval/STAGE_B2.md)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics as st
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval_ladder")); sys.path.insert(0, str(REPO / "scripts/ed_screen"))
import stage_b as B  # noqa: E402  (constants + ledger; argparse only under __main__)

CAMP, ARM_B, ARM_A, REG_A, REG_B, EVAL_ENTRY, PY, ACCOUNT, TOKEN = B.CAMP, B.ARM_B, B.ARM_A, B.REG_A, B.REG_B, B.EVAL_ENTRY, B.PY, B.ACCOUNT, B.TOKEN
REG_B2 = REPO / f"eval_ladder/registry_{ARM_B}_r12.jsonl"
SUB_B, SUB_B2 = B.SUB_B, REPO / "store/gens/032_dualforce_dcg_w6/08_effect_edscreen_r12__dai"
PROMPTS = REPO / "store/prompts/016_edscreen_effect_token_r12"
GENS_DIR = CAMP / "eval/gens_b2"
ROWS = (1, 2)


def corpus_sha(rows):  # = scripts/store_register.py corpus_sha (not imported: keep this lane free of its CLI)
    key = lambda r: (r["cell"], r["endpoint"], r.get("reference") or "", r["sided"])
    uniq = {key(r): r for r in rows}
    return hashlib.sha256("".join(r["prompt"] for r in sorted(uniq.values(), key=key)).encode()).hexdigest()[:12]


def rows():
    lv = sorted(csv.DictReader(open(CAMP / "eval/screen_levels.csv")), key=lambda r: float(r["prompt_only_level"]))[:B.N_EFFECTS]
    sel = {d["cls"]: d for d in json.loads((CAMP / "selection.json").read_text())["effects"]}
    base = {r["item_id"]: r for r in map(json.loads, filter(str.strip, REG_A.read_text().splitlines()))}
    out, pairs = [], []
    for r in lv:
        d = sel[r["cls"]]
        for i in ROWS:
            ep = d["endpoints"][i]["std"]
            a = base[f"G-zs-same__{ARM_A}__{ep}__ref_{d['reference']}"]
            s1, clause = a["prompt"].split(". ", 1)
            b = dict(a, arm=ARM_B, item_id=f"G-zs-same__{ARM_B}__{ep}__ref_{d['reference']}", prompt=f"{s1}. {TOKEN}. {clause}", use_reference=True)
            out.append(b); pairs.append({"cls": d["cls"], "effect": d["effect"], "row": i, "base_item": a["item_id"], "dcg_item": b["item_id"], "screen_level_3rows": float(r["prompt_only_level"])})
    REG_B2.write_text("".join(json.dumps(r) + "\n" for r in out))
    have = {json.loads(l)["item_id"] for l in REG_B.read_text().splitlines() if l.strip()}
    add = [r for r in out if r["item_id"] not in have]
    with open(REG_B, "a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in add))
    (CAMP / "stage_b2_pairs.json").write_text(json.dumps(pairs, indent=1))
    print(f"[rows] {len(out)} DCG rows -> {REG_B2.relative_to(REPO)}; appended {len(add)} to {REG_B.relative_to(REPO)} (now {len(have) + len(add)} rows)")
    # prompts family 016 (store_register needs a family whose prompt_corpus_sha matches the 80-row registry)
    p15 = [json.loads(l) for l in (REPO / "store/prompts/015_edscreen_effect_token/grid.jsonl").read_text().splitlines() if l.strip()]
    keep = set(p15[0].keys()); sha = corpus_sha(out)
    PROMPTS.mkdir(parents=True, exist_ok=True)
    (PROMPTS / "grid.jsonl").write_text("".join(json.dumps({k: r[k] for k in r if k in keep}) + "\n" for r in out))
    (PROMPTS / "meta.yaml").write_text(f"""id: prompts/016_edscreen_effect_token_r12
seq: 16
shelf: prompts
created: {date.today().isoformat()}
family: B
grammar: "{{S1}}. {TOKEN}. {{EFFECT}}."
role: "EFFECT prompts WITH the transition token for adapter arms on the EffectData gapper screen stage B2: the 40 stage-B effects x the screen's OTHER two same-content rows (endpoints 1 and 2); native 81 f, frame-0 anchor"
rows: {len(out)}
prompt_corpus_sha: {sha}   # sha256[:12] over prompts concatenated in (cell,endpoint,reference,sided) order, unique items
renderer: "scripts/ed_screen/stage_b2.py rows (the prompts/014 base prompt with '{TOKEN}.' inserted after S1 — same S1 and clause byte-for-byte)"
derived_from: "store/prompts/014_edscreen_effect (subset: 80 of 900 rows) + misc/2026-09-08_ed_gapper_screen/eval/screen_levels.csv (lowest 40, rows 1-2)"
source: scripts/ed_screen/stage_b2.py
notes: "side lane of grid v3 (misc/2026-09-08_ed_gapper_screen/PLAN.md). Rows are ARM-FREE. Sibling of prompts/015 (row 0 of the same effects); paired by row with prompts/014."
""")
    print(f"[rows] prompts/016 written: {len(out)} rows, sha {sha}")


def gen():
    if "gen_b2" in B.ledger():
        print("already submitted:", B.ledger()["gen_b2"]); return
    (SUB_B2 / "videos").mkdir(parents=True, exist_ok=True); (CAMP / "gen/logs").mkdir(parents=True, exist_ok=True)
    n = sum(1 for l in REG_B2.read_text().splitlines() if l.strip())
    r = subprocess.run(["sbatch", "--parsable", f"--account={ACCOUNT}", f"--array=0-{n - 1}", "--time=00:40:00", "--job-name=eds_dcg2",
                        f"--output={CAMP}/gen/logs/%x-%A_%a.out",
                        "--export=ALL," + ",".join([f"ARM={ARM_B}", f"REG={REG_B2.relative_to(REPO)}", f"OUT={SUB_B2.relative_to(REPO)}", f"NCHUNKS={n}", "W=6.0",
                                                    "GEN_FRAMES=81", "GEN_PREFIX_FRAMES=1", "LADDER_SPLIT_FILE=split_screen.json"]),
                        str(REPO / "misc/2026-09-07_eval_grid_v2/gen/job_dcg_v3.sbatch")], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr)
    B.ledger({"gen_b2": r.stdout.strip()}); print(f"gen_b2 -> {r.stdout.strip()} ({n} single-clip tasks, account {ACCOUNT})")


def register():
    n = len(list((SUB_B2 / "videos").glob("*.mp4"))); need = sum(1 for l in REG_B2.read_text().splitlines() if l.strip())
    if n < need:
        raise SystemExit(f"{n}/{need} clips present")
    if (SUB_B2 / "meta.yaml").exists():
        print("already registered"); return
    out = subprocess.run([PY, str(REPO / "scripts/store_register.py"), "gen", str(SUB_B2.relative_to(REPO)), "--registry", str(REG_B2.relative_to(REPO)),
                          "--run", "runs/012_dualforce_control", "--step", "1000", "--code", "src/LTX-2-ctt-v2-train packages + scripts/grid_v3/gen_dcg_v3.py (crossfade null, gs4/stg1, w=6) @ ed gapper screen",
                          "--notes", "\"EffectData gapper screen stage B2: DCG w=6 effect on the screen's other two same-content rows (endpoints 1 and 2) of the 40 stage-B effects, seed 42, native 81 f + frame-0 anchor; same harness arm as gens/032_dualforce_dcg_w6/07_effect_edscreen__dai (row 0); paired with gens/005_base_cond/08_effect_edscreen__dai\""],
                         capture_output=True, text=True)
    print(out.stdout[-1200:], out.stderr[-400:])


def plan():
    d = GENS_DIR / ARM_B; d.mkdir(parents=True, exist_ok=True)
    for src in (SUB_B / "videos", SUB_B2 / "videos"):
        for p in sorted(src.glob("*.mp4")):
            q = d / p.name
            if not q.exists():
                q.symlink_to(p)
    print(f"[plan] per-clip gens dir {d.relative_to(REPO)}: {len(list(d.glob('*.mp4')))} clips")
    mdir = CAMP / "eval/manifests_dcg2"; mdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GEN_PREFIX_FRAMES="1", LADDER_SPLIT_FILE="split_screen.json")
    r = subprocess.run([PY, str(REPO / "eval_ladder/run_eval.py"), "--mode", "plan", "--arms", ARM_B, "--extra-registry", str(REG_B), "--gens", str(GENS_DIR),
                        "--scores", str(EVAL_ENTRY / ARM_B), "--eval-dir", str(mdir), "--chunks", "8", "--seeds", "42"], env=env, capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith("[plan]")) or r.stderr[-800:])
    if r.returncode:
        raise SystemExit("PLAN FAILED")


def submit():
    if "score_b2" in B.ledger():
        print("already submitted:", B.ledger()["score_b2"]); return
    mdir = CAMP / "eval/manifests_dcg2"; n = len(list(mdir.glob("eval_c*.json"))); (CAMP / "eval/logs").mkdir(parents=True, exist_ok=True)
    if not n:
        raise SystemExit("no manifests")
    r = subprocess.run(["sbatch", "--parsable", f"--account={ACCOUNT}", f"--array=0-{n - 1}", "--time=00:40:00", "--job-name=eds_dcg2_score", f"--output={CAMP}/eval/logs/%x-%A_%a.out",
                        "--export=ALL," + ",".join([f"MDIR={mdir}", f"EVAL={EVAL_ENTRY / ARM_B}", "LABEL=r12c", "GEN_PREFIX_FRAMES=1", f"CORPUS={CAMP / 'corpus_manifest_screen.json'}"]),
                        str(REPO / "misc/2026-09-07_eval_grid_v2/score_v3.sbatch")], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr)
    B.ledger({"score_b2": r.stdout.strip()}); print(f"score_b2 -> {r.stdout.strip()} ({n} shards)")


def readout():
    os.environ["LADDER_CEILINGS_EXTRA"] = str(CAMP / "ceilings_screen.json"); os.environ["LADDER_SPLIT_FILE"] = "split_screen.json"
    import run_eval
    ceil = run_eval.ceilings()
    ra = {r["item_id"]: r for r in map(json.loads, filter(str.strip, REG_A.read_text().splitlines()))}
    rb = {r["item_id"]: r for r in map(json.loads, filter(str.strip, REG_B.read_text().splitlines()))}
    pa = run_eval.item_pct(run_eval.pool_means(EVAL_ENTRY / ARM_A), ra, ceil); pb = run_eval.item_pct(run_eval.pool_means(EVAL_ENTRY / ARM_B), rb, ceil)
    p0 = {p["cls"]: p for p in json.loads((CAMP / "stage_b_pairs.json").read_text())}
    p12 = json.loads((CAMP / "stage_b2_pairs.json").read_text())
    by = {}
    for p in p12:
        by.setdefault(p["cls"], {"effect": p["effect"], "lv": p["screen_level_3rows"], "rows": []})["rows"].append(p)
    f = lambda v: "" if v is None else f"{v*100:.1f}"
    lines = ["# Stage B2 — DCG w=6 vs prompt-only on the screen's OTHER two rows (endpoints 1-2) of the 40 stage-B effects (seed 42; pool-% of m1a, ceilings_screen)", "",
             "Row 0 = stage B (the row the two-criterion shortlist was selected on). Rows 1-2 were not used for the DCG criterion → out-of-selection check. 3-row = mean over all three.", "",
             "| effect | screen A (3 rows) | base r0 | DCG r0 | Δ r0 | base r1-2 | DCG r1-2 | Δ r1-2 | base 3-row | DCG 3-row | Δ 3-row | ceiling | rule r0 → r1-2 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    agg = {"r0": [], "r12": [], "r3": []}; keep = {"core": [0, 0], "ext": [0, 0]}
    rule = lambda b, d: None if b is None or d is None else "core" if b <= .60 and d >= .80 else "ext" if b <= .80 and d >= .80 else "—"
    for c, e in sorted(by.items(), key=lambda kv: kv[1]["lv"]):
        a0, d0 = pa.get(p0[c]["base_item"]), pb.get(p0[c]["dcg_item"])
        a12 = [pa[p["base_item"]] for p in e["rows"] if p["base_item"] in pa]; d12 = [pb[p["dcg_item"]] for p in e["rows"] if p["dcg_item"] in pb]
        ma, md = (st.mean(a12) if a12 else None), (st.mean(d12) if d12 else None)
        a3 = st.mean(([a0] if a0 is not None else []) + a12) if (a0 is not None or a12) else None
        d3 = st.mean(([d0] if d0 is not None else []) + d12) if (d0 is not None or d12) else None
        if None not in (a0, d0): agg["r0"].append((d0 - a0) * 100)
        if None not in (ma, md): agg["r12"].append((md - ma) * 100)
        if None not in (a3, d3): agg["r3"].append((d3 - a3) * 100)
        r0, r12 = rule(a0, d0), rule(ma, md)
        if r0 in keep:
            keep[r0][0] += 1; keep[r0][1] += (r12 == r0) or (r0 == "ext" and r12 == "core")
        dd = lambda x, y: "" if x is None or y is None else f"{(y-x)*100:+.1f}"
        lines.append(f"| {e['effect']} | {e['lv']:.1f} | {f(a0)} | {f(d0)} | {dd(a0,d0)} | {f(ma)} | {f(md)} | {dd(ma,md)} | {f(a3)} | {f(d3)} | {dd(a3,d3)} | {ceil.get(c, float('nan')):.2f} | {r0} → {r12} |")
    if agg["r0"] and agg["r12"]:
        lines += ["", f"paired effects {len(agg['r12'])}: mean Δ row 0 (selected) {st.mean(agg['r0']):+.1f} pp · rows 1-2 (unselected) {st.mean(agg['r12']):+.1f} pp (median {st.median(agg['r12']):+.1f}, DCG higher on {sum(d > 0 for d in agg['r12'])}/{len(agg['r12'])}) · 3-row {st.mean(agg['r3']):+.1f} pp",
                  f"rule retention on rows 1-2: core (base ≤ 60 & DCG ≥ 80) {keep['core'][1]}/{keep['core'][0]} · ext (base ≤ 80 & DCG ≥ 80) {keep['ext'][1]}/{keep['ext'][0]}"]
    (CAMP / "eval/STAGE_B2.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["rows", "gen", "register", "plan", "submit", "readout"])
    {"rows": rows, "gen": gen, "register": register, "plan": plan, "submit": submit, "readout": readout}[ap.parse_args().cmd]()
