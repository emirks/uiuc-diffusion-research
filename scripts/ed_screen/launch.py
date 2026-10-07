#!/usr/bin/env python
"""EffectData gapper screen — run the prompt-only arm end to end (DeltaAI).

  gen       sbatch the base_cond effect generation (job_gen_v3.sbatch; STACK=official, GEN_FRAMES=81 GEN_PREFIX_FRAMES=1,
            seed 42 only: NCHUNKS=16, array 0-15) into gens/005_base_cond/08_effect_edscreen__dai
  register  store_register.py gen for the subentry (all clips present) + INDEX row to paste
  plan      run_eval --mode plan (rows are no_twin; GEN_PREFIX_FRAMES=1) into eval/manifests; scores go to
            store/evals/029_ed_gapper_screen__dai__2026-09-08/base_cond_effect_edscreen
  submit    sbatch score_v3.sbatch per chunk with CORPUS=<screen superset manifest>
  status    clips / scored rows / error rows
  summary   per-effect prompt-only level (pool-%, LADDER_CEILINGS_EXTRA=ceilings_screen.json) -> eval/screen_levels.csv + REPORT.md
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval_ladder"))
CAMP = REPO / "misc/2026-09-08_ed_gapper_screen"
ARM = "base_cond_effect_edscreen"
REG = REPO / f"eval_ladder/registry_{ARM}.jsonl"
SUB = REPO / "store/gens/005_base_cond/08_effect_edscreen__dai"
EVAL_ENTRY = REPO / "store/evals/029_ed_gapper_screen__dai__2026-09-08"
PY = "/taiga/illinois/eng/cs/jrehg/users/emirkisa/envs-aarch64/ltx2/bin/python"
ACCOUNT = "bhwp-dtai-gh"
LEDGER = CAMP / "ledger.json"
CHUNKS = 16


def ledger(update=None):
    d = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    if update:
        d.update(update); LEDGER.write_text(json.dumps(d, indent=1))
    return d


def rows():
    return [json.loads(l) for l in REG.read_text().splitlines() if l.strip()]


def sbatch(args):
    r = subprocess.run(["sbatch", "--parsable"] + args, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr)
    return r.stdout.strip()


def gen():
    if "gen" in ledger():
        print("gen already submitted:", ledger()["gen"]); return
    (SUB / "videos").mkdir(parents=True, exist_ok=True)
    jid = sbatch([f"--account={ACCOUNT}", f"--array=0-{CHUNKS - 1}%16", "--time=01:15:00", "--job-name=eds_gen",
                  f"--output={CAMP}/gen/logs/%x-%A_%a.out",
                  "--export=ALL," + ",".join([f"ARM={ARM}", f"REG={REG.relative_to(REPO)}", f"OUT={SUB.relative_to(REPO)}", f"NCHUNKS={CHUNKS}",
                                              "STACK=official", "GEN_FRAMES=81", "GEN_PREFIX_FRAMES=1"]),
                  str(REPO / "misc/2026-09-07_eval_grid_v2/gen/job_gen_v3.sbatch")])
    ledger({"gen": jid}); print(f"gen -> {jid} ({len(rows())} rows x 1 seed)")


def register():
    n = len(list((SUB / "videos").glob("*.mp4"))); need = len(rows())
    if n < need:
        raise SystemExit(f"{n}/{need} clips present — not registering yet")
    if (SUB / "meta.yaml").exists():
        print("already registered"); return
    out = subprocess.run([PY, str(REPO / "scripts/store_register.py"), "gen", str(SUB.relative_to(REPO)), "--registry", str(REG.relative_to(REPO)),
                          "--run", "base weights (no adapter)", "--code", "src/LTX-2-official (venv editable) @ ed gapper screen",
                          "--notes", "\"EffectData gapper screen stage A (misc/2026-09-08_ed_gapper_screen): prompt-only effect arm, 300 effects x 3 same-content rows, seed 42, native 81 f + frame-0 anchor (GEN_FRAMES=81 GEN_PREFIX_FRAMES=1); DeltaAI GH200\""],
                         capture_output=True, text=True)
    print(out.stdout[-1500:], out.stderr[-500:])


def plan():
    import shutil
    gens = CAMP / "eval/gens"; gens.mkdir(parents=True, exist_ok=True)
    link = gens / ARM
    if not link.exists():
        link.symlink_to(SUB / "videos")
    mdir = CAMP / "eval/manifests"; mdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GEN_PREFIX_FRAMES="1", LADDER_SPLIT_FILE="split_screen.json")
    r = subprocess.run([PY, str(REPO / "eval_ladder/run_eval.py"), "--mode", "plan", "--arms", ARM, "--extra-registry", str(REG), "--gens", str(gens),
                        "--scores", str(EVAL_ENTRY / ARM), "--eval-dir", str(mdir), "--chunks", str(CHUNKS), "--seeds", "42"], env=env, capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith("[plan]")) or r.stderr[-800:])


def submit():
    if "score" in ledger():
        print("score already submitted:", ledger()["score"]); return
    mdir = CAMP / "eval/manifests"
    n = len(list(mdir.glob("eval_c*.json")))
    jid = sbatch([f"--account={ACCOUNT}", f"--array=0-{n - 1}%16", "--time=01:00:00", "--job-name=eds_score", f"--output={CAMP}/eval/logs/%x-%A_%a.out",
                  "--export=ALL," + ",".join([f"MDIR={mdir}", f"EVAL={EVAL_ENTRY / ARM}", "LABEL=c", "GEN_PREFIX_FRAMES=1", f"CORPUS={CAMP / 'corpus_manifest_screen.json'}"]),
                  str(REPO / "misc/2026-09-07_eval_grid_v2/score_v3.sbatch")])
    ledger({"score": jid}); print(f"score -> {jid} ({n} shards)")


def status():
    rs = rows(); n = len(list((SUB / "videos").glob("*.mp4")))
    scored = err = 0
    for f in (EVAL_ENTRY / ARM).glob("*/items.jsonl"):
        for l in f.read_text().splitlines():
            if l.strip():
                scored += 1; err += "error" in json.loads(l)
    print(f"rows {len(rs)} | clips {n}/{len(rs)} | scored rows {scored} (error rows {err}) | ledger {ledger()}")


def summary():
    os.environ["LADDER_CEILINGS_EXTRA"] = str(CAMP / "ceilings_screen.json"); os.environ["LADDER_SPLIT_FILE"] = "split_screen.json"
    import run_eval
    ceil = run_eval.ceilings(); rs = {r["item_id"]: r for r in rows()}
    pool = run_eval.pool_means(EVAL_ENTRY / ARM); pct = run_eval.item_pct(pool, rs, ceil)
    sel = {d["cls"]: d for d in json.loads((CAMP / "selection.json").read_text())["effects"]}
    per = collections.defaultdict(list)
    for i, v in pct.items():
        per[rs[i]["gt_pool_class"]].append(v)
    out = []
    for cls, vs in per.items():
        d = sel[cls]
        out.append({"cls": cls, "effect": d["effect"], "role": d["role"], "category": d["category"], "n_rows": len(vs), "prompt_only_level": round(st.mean(vs) * 100, 1),
                    "ceiling": round(ceil.get(cls, float("nan")), 3), "pred_gap": d.get("pred_gap"), "measured_gap_eff": d.get("measured_gap_eff"), "measured_base_eff": d.get("measured_base_eff")})
    out.sort(key=lambda r: r["prompt_only_level"])
    with open(CAMP / "eval/screen_levels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    lines = ["# EffectData gapper screen — stage A readout (prompt-only effect level, pool-% of m1a, seed 42, 3 same-content rows per effect)", "",
             f"effects scored {len(out)} / {len(sel)}; rows {len(pct)}", ""]
    for T in (60, 70, 80, 90):
        k = [r for r in out if r["prompt_only_level"] < T]
        lines.append(f"- level < {T}: {len(k)} effects ({sum(r['role']=='exploit' for r in k)} exploit / {sum(r['role']=='explore' for r in k)} explore / {sum(r['role']=='anchor' for r in k)} anchor)")
    for role in ("exploit", "explore", "anchor"):
        k = [r["prompt_only_level"] for r in out if r["role"] == role]
        if k:
            lines.append(f"- {role}: n={len(k)} median {st.median(k):.1f}, share < 80: {sum(v < 80 for v in k) / len(k):.0%}")
    anc = [r for r in out if r["role"] == "anchor"]
    if anc:
        lines += ["", "anchors (one-seed screen level vs grid-v3 two-seed base effect level):", "", "| effect | screen level | grid-v3 base eff | grid-v3 gap |", "|---|---|---|---|"]
        lines += [f"| {r['effect']} | {r['prompt_only_level']} | {r['measured_base_eff']:.1f} | {r['measured_gap_eff']:+.1f} |" for r in anc]
    lines += ["", "## lowest 40 (candidates for stage B)", "", "| effect | role | level | ceiling | pred_gap | category |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['effect']} | {r['role']} | {r['prompt_only_level']} | {r['ceiling']} | {'' if r['pred_gap'] is None else round(r['pred_gap'],1)} | {r['category']} |" for r in out[:40]]
    (CAMP / "eval/REPORT.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines[:16])); print(f"-> {CAMP.relative_to(REPO)}/eval/{{screen_levels.csv,REPORT.md}}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["gen", "register", "plan", "submit", "status", "summary"])
    {"gen": gen, "register": register, "plan": plan, "submit": submit, "status": status, "summary": summary}[ap.parse_args().cmd]()
