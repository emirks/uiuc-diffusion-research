#!/usr/bin/env python
"""EffectData gapper screen — stage B: DCG w=6 effect generation on the SAME rows as the prompt-only screen for the 40
lowest-scoring effects (owner 2026-09-09: "run generation for those 40 same rows with dcg effect too; only 40 generations").

One row per effect (the screen's first endpoint, a fixed rule — not the lowest-scoring row, to avoid picking noise), seed 42,
native 81 f + frame-0 anchor, adapter runs/012@1000 one_way + crossfade-null DCG w=6 (scripts/grid_v3/gen_dcg_v3.py), prompt
"{S1}. sksz. {clause}." (adapter arms carry the token; the base arm's prompt is the same text without it).
  rows      registry eval_ladder/registry_dualforce_dcg_w6_effect_edscreen.jsonl (+ arms.yaml entry) + stage_b_pairs.json
  gen       sbatch job_dcg_v3.sbatch, NCHUNKS=40 -> 40 single-clip tasks (max parallelism), into gens/032_dualforce_dcg_w6/07_effect_edscreen__dai
  register / plan / submit / readout   as stage A (launch.py) for the DCG arm; readout = paired base-vs-DCG per effect on the SAME row
"""
from __future__ import annotations

import argparse
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
ARM_B, ARM_A = "dualforce_dcg_w6_effect_edscreen", "base_cond_effect_edscreen"
REG_B, REG_A = REPO / f"eval_ladder/registry_{ARM_B}.jsonl", REPO / f"eval_ladder/registry_{ARM_A}.jsonl"
SUB_B = REPO / "store/gens/032_dualforce_dcg_w6/07_effect_edscreen__dai"
EVAL_ENTRY = REPO / "store/evals/029_ed_gapper_screen__dai__2026-09-08"
PY = "/taiga/illinois/eng/cs/jrehg/users/emirkisa/envs-aarch64/ltx2/bin/python"
ACCOUNT = "bgjg-dtai-gh"
N_EFFECTS = 40
TOKEN = "sksz"


def ledger(update=None):
    p = CAMP / "ledger.json"; d = json.loads(p.read_text()) if p.exists() else {}
    if update:
        d.update(update); p.write_text(json.dumps(d, indent=1))
    return d


def rows():
    lv = sorted(csv.DictReader(open(CAMP / "eval/screen_levels.csv")), key=lambda r: float(r["prompt_only_level"]))[:N_EFFECTS]
    sel = {d["cls"]: d for d in json.loads((CAMP / "selection.json").read_text())["effects"]}
    base = {r["item_id"]: r for r in map(json.loads, filter(str.strip, REG_A.read_text().splitlines()))}
    out, pairs = [], []
    for r in lv:
        d = sel[r["cls"]]; ep = d["endpoints"][0]["std"]
        a = base[f"G-zs-same__{ARM_A}__{ep}__ref_{d['reference']}"]
        s1, clause = a["prompt"].split(". ", 1)          # base prompt = "{S1}. {clause}." (S1 has no inner ". ")
        b = dict(a, arm=ARM_B, item_id=f"G-zs-same__{ARM_B}__{ep}__ref_{d['reference']}", prompt=f"{s1}. {TOKEN}. {clause}", use_reference=True)
        out.append(b); pairs.append({"cls": d["cls"], "effect": d["effect"], "base_item": a["item_id"], "dcg_item": b["item_id"], "screen_level_3rows": float(r["prompt_only_level"])})
    REG_B.write_text("".join(json.dumps(r) + "\n" for r in out))
    (CAMP / "stage_b_pairs.json").write_text(json.dumps(pairs, indent=1))
    print(f"[rows] {len(out)} DCG rows -> {REG_B.relative_to(REPO)}; sample prompt: {out[0]['prompt'][:160]}")
    arms_p = REPO / "eval_ladder/arms.yaml"; txt = arms_p.read_text()
    if f"\n  {ARM_B}:" not in txt:
        anchor = "  dualforce_dcg_w6_effect_v3ed81:"; i = txt.index(anchor); j = txt.index("\n  ", i + len(anchor)) + 1
        block = (f"  {ARM_B}:\n    kind: generalist\n    targets: attn_ffn\n    step: 1000\n    attention: one_way\n"
                 f"    adapter: store/runs/012_dualforce_control/checkpoints/lora_weights_step_01000.safetensors\n"
                 f'    note: "EffectData gapper screen stage B: DCG w=6 (crossfade null, gs4/stg1) on the 40 lowest prompt-only effects, one row each (screen endpoint 0), seed 42, native 81 f + frame-0 anchor; prompt = base prompt + token"\n')
        arms_p.write_text(txt[:j] + block + txt[j:]); print("[rows] arms.yaml entry added")


def gen():
    if "gen_b" in ledger():
        print("already submitted:", ledger()["gen_b"]); return
    (SUB_B / "videos").mkdir(parents=True, exist_ok=True)
    n = sum(1 for l in REG_B.read_text().splitlines() if l.strip())
    r = subprocess.run(["sbatch", "--parsable", f"--account={ACCOUNT}", f"--array=0-{n - 1}", "--time=00:40:00", "--job-name=eds_dcg",
                        f"--output={CAMP}/gen/logs/%x-%A_%a.out",
                        "--export=ALL," + ",".join([f"ARM={ARM_B}", f"REG={REG_B.relative_to(REPO)}", f"OUT={SUB_B.relative_to(REPO)}", f"NCHUNKS={n}", "W=6.0",
                                                    "GEN_FRAMES=81", "GEN_PREFIX_FRAMES=1", "LADDER_SPLIT_FILE=split_screen.json"]),
                        str(REPO / "misc/2026-09-07_eval_grid_v2/gen/job_dcg_v3.sbatch")], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr)
    ledger({"gen_b": r.stdout.strip()}); print(f"gen_b -> {r.stdout.strip()} ({n} single-clip tasks)")


def register():
    n = len(list((SUB_B / "videos").glob("*.mp4"))); need = sum(1 for l in REG_B.read_text().splitlines() if l.strip())
    if n < need:
        raise SystemExit(f"{n}/{need} clips present")
    if (SUB_B / "meta.yaml").exists():
        print("already registered"); return
    out = subprocess.run([PY, str(REPO / "scripts/store_register.py"), "gen", str(SUB_B.relative_to(REPO)), "--registry", str(REG_B.relative_to(REPO)),
                          "--run", "runs/012_dualforce_control", "--step", "1000", "--code", "src/LTX-2-ctt-v2-train packages + scripts/grid_v3/gen_dcg_v3.py (crossfade null, gs4/stg1, w=6) @ ed gapper screen",
                          "--notes", "\"EffectData gapper screen stage B: DCG w=6 effect on the 40 lowest prompt-only effects, one same-content row each (screen endpoint 0), seed 42, native 81 f + frame-0 anchor; paired with gens/005_base_cond/08_effect_edscreen__dai\""],
                         capture_output=True, text=True)
    print(out.stdout[-1200:], out.stderr[-400:])


def plan():
    gens = CAMP / "eval/gens"; gens.mkdir(parents=True, exist_ok=True); link = gens / ARM_B
    if not link.exists():
        link.symlink_to(SUB_B / "videos")
    mdir = CAMP / "eval/manifests_dcg"; mdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GEN_PREFIX_FRAMES="1", LADDER_SPLIT_FILE="split_screen.json")
    r = subprocess.run([PY, str(REPO / "eval_ladder/run_eval.py"), "--mode", "plan", "--arms", ARM_B, "--extra-registry", str(REG_B), "--gens", str(gens),
                        "--scores", str(EVAL_ENTRY / ARM_B), "--eval-dir", str(mdir), "--chunks", "4", "--seeds", "42"], env=env, capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith("[plan]")) or r.stderr[-800:])


def submit():
    if "score_b" in ledger():
        print("already submitted:", ledger()["score_b"]); return
    mdir = CAMP / "eval/manifests_dcg"; n = len(list(mdir.glob("eval_c*.json")))
    r = subprocess.run(["sbatch", "--parsable", f"--account={ACCOUNT}", f"--array=0-{n - 1}", "--time=00:40:00", "--job-name=eds_dcg_score", f"--output={CAMP}/eval/logs/%x-%A_%a.out",
                        "--export=ALL," + ",".join([f"MDIR={mdir}", f"EVAL={EVAL_ENTRY / ARM_B}", "LABEL=c", "GEN_PREFIX_FRAMES=1", f"CORPUS={CAMP / 'corpus_manifest_screen.json'}"]),
                        str(REPO / "misc/2026-09-07_eval_grid_v2/score_v3.sbatch")], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr)
    ledger({"score_b": r.stdout.strip()}); print(f"score_b -> {r.stdout.strip()} ({n} shards)")


def readout():
    os.environ["LADDER_CEILINGS_EXTRA"] = str(CAMP / "ceilings_screen.json"); os.environ["LADDER_SPLIT_FILE"] = "split_screen.json"
    import run_eval
    ceil = run_eval.ceilings()
    ra = {r["item_id"]: r for r in map(json.loads, filter(str.strip, REG_A.read_text().splitlines()))}
    rb = {r["item_id"]: r for r in map(json.loads, filter(str.strip, REG_B.read_text().splitlines()))}
    pa = run_eval.item_pct(run_eval.pool_means(EVAL_ENTRY / ARM_A), ra, ceil); pb = run_eval.item_pct(run_eval.pool_means(EVAL_ENTRY / ARM_B), rb, ceil)
    pairs = json.loads((CAMP / "stage_b_pairs.json").read_text())
    lines = ["# Stage B — DCG w=6 vs prompt-only on the SAME row (40 lowest screen effects; seed 42; pool-% of m1a, ceilings_screen)", "",
             "| effect | base_cond effect (this row) | DCG w6 effect (this row) | Δ DCG − base | ceiling | screen level (3 rows) |", "|---|---|---|---|---|---|"]
    ds = []
    for p in sorted(pairs, key=lambda p: p["screen_level_3rows"]):
        a, b = pa.get(p["base_item"]), pb.get(p["dcg_item"])
        if a is not None and b is not None:
            ds.append((b - a) * 100)
        lines.append(f"| {p['effect']} | {'' if a is None else f'{a*100:.1f}'} | {'' if b is None else f'{b*100:.1f}'} | {'' if a is None or b is None else f'{(b-a)*100:+.1f}'} | {ceil.get(p['cls'], float('nan')):.2f} | {p['screen_level_3rows']:.1f} |")
    if ds:
        lines += ["", f"paired rows {len(ds)}: mean Δ {st.mean(ds):+.1f} pp, median {st.median(ds):+.1f}, DCG higher on {sum(d > 0 for d in ds)}/{len(ds)}; mean base {st.mean(pa[p['base_item']] for p in pairs if p['base_item'] in pa)*100:.1f} -> mean DCG {st.mean(pb[p['dcg_item']] for p in pairs if p['dcg_item'] in pb)*100:.1f}"]
    (CAMP / "eval/STAGE_B.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["rows", "gen", "register", "plan", "submit", "readout"])
    {"rows": rows, "gen": gen, "register": register, "plan": plan, "submit": submit, "readout": readout}[ap.parse_args().cmd]()
