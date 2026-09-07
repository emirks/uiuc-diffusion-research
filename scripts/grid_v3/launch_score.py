#!/usr/bin/env python
"""grid v3 — close out the generations into the store, plan the v4 scoring, submit it, register the eval.

  register  store_register.py gen for every subentry with all videos present (+ prints INDEX rows) then store_fsck
  plan      run_eval.py --mode plan per harness arm: gens dir of per-arm symlinks into the store subentries,
            manifests under misc/2026-09-07_eval_grid_v2/eval/manifests/<arm>/, scores into the eval entry
            store/evals/028_grid_v3_paper_arms__dai__<date>/<arm>/ ; EffectData arms plan with GEN_PREFIX_FRAMES=1
  submit    sbatch score_v3.sbatch per arm (16 shards; --corpus 677 + --reference-corpus 222, the amendment)
  status    scored rows per arm vs planned
Run in that order once launch_gen.py status shows every subentry complete.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/grid_v3"))
from launch_gen import ARMS, FAMILIES, FAMKEY, TIERS, harness_arm, registry, subentry  # noqa: E402

PY = "/taiga/illinois/eng/cs/jrehg/users/emirkisa/envs-aarch64/ltx2/bin/python"
EVALDIR = REPO / "misc/2026-09-07_eval_grid_v2/eval"
EVAL_ENTRY = f"store/evals/028_grid_v3_paper_arms__dai__{date.today().isoformat()}"
LEDGER = EVALDIR / "score_ledger.json"
ACCOUNTS = {"base_cond": "bhwp-dtai-gh", "ic_gen": "bhwp-dtai-gh", "dualforce_control": "bgjg-dtai-gh", "dualforce_dcg_w6": "bgjg-dtai-gh"}


def all_arms():
    for arm in ARMS:
        for tier in TIERS:
            for fam in ("hf", "ed"):
                yield arm, tier, fam, harness_arm(arm, tier, fam)


def sh(cmd, env=None, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if check and r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"failed: {' '.join(map(str, cmd))}")
    return r.stdout


def expected_rows(ha, arm, tier, fam):
    return [json.loads(l) for l in registry(arm, tier, fam).read_text().splitlines() if l.strip() and json.loads(l)["arm"] == ha]


def register():
    rows_out = []
    for arm, tier, fam, ha in all_arms():
        sub = REPO / subentry(arm, tier, fam)
        rows = expected_rows(ha, arm, tier, fam)
        have = len(list((sub / "videos").glob("*.mp4")))
        if have < 2 * len(rows):
            print(f"SKIP {ha}: {have}/{2 * len(rows)} videos")
            continue
        if (sub / "meta.yaml").exists():
            print(f"already registered: {ha}")
            continue
        spec = ARMS[arm]
        cmd = [PY, str(REPO / "scripts/store_register.py"), "gen", str(sub.relative_to(REPO)), "--registry", str(registry(arm, tier, fam).relative_to(REPO)),
               "--run", spec["run"], "--code", spec["code"], "--notes",
               f"grid v3 (design 3.0.0) {tier} tier, family prompts/{FAMILIES[tier][FAMKEY[fam]]}; "
               + ("EffectData tier: native 81 f, frame-0 anchor (GEN_FRAMES=81 GEN_PREFIX_FRAMES=1); " if fam == "ed" else
                  f"139 kept rows hardlinked from gens/{spec['old'][tier]} (byte-identical inputs); ")
               + "seeds 42/43; DeltaAI GH200"]
        if spec.get("step"):
            cmd += ["--step", str(spec["step"])]
        out = sh(cmd)
        rows_out.append(out.strip().splitlines()[-1] if out.strip() else ha)
        print(out.strip().splitlines()[0][:160])
    print("\n".join(rows_out))
    print(sh([PY, str(REPO / "scripts/store_fsck.py")], check=False)[-600:])


def plan(chunks: int):
    gens = EVALDIR / "gens"
    gens.mkdir(parents=True, exist_ok=True)
    for arm, tier, fam, ha in all_arms():
        sub = REPO / subentry(arm, tier, fam)
        link = gens / ha
        if not link.exists():
            link.symlink_to(sub / "videos")
        mdir = EVALDIR / "manifests" / ha
        mdir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        if fam == "ed":
            env["GEN_PREFIX_FRAMES"] = "1"
        out = sh([PY, str(REPO / "eval_ladder/run_eval.py"), "--mode", "plan", "--arms", ha, "--extra-registry", str(registry(arm, tier, fam)),
                  "--gens", str(gens), "--scores", str(REPO / EVAL_ENTRY / ha), "--eval-dir", str(mdir), "--chunks", str(chunks), "--seeds", "42,43"], env=env, check=False)
        print(f"{ha:40s} " + " | ".join(l for l in out.strip().splitlines() if l.startswith("[plan]"))[:300])


def submit(chunks: int, dry: bool):
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    for arm, tier, fam, ha in all_arms():
        if ha in ledger and not dry:
            print(f"skip {ha}: job {ledger[ha]}")
            continue
        mdir = EVALDIR / "manifests" / ha
        if not (mdir / "eval_c0.json").exists():
            print(f"SKIP {ha}: no manifest")
            continue
        cmd = ["sbatch", "--parsable", f"--account={ACCOUNTS[arm]}", f"--array=0-{chunks - 1}%16", f"--job-name=v3s_{arm[:8]}_{tier[0]}{fam}",
               "--export=ALL," + ",".join([f"MDIR={mdir}", f"EVAL={REPO / EVAL_ENTRY / ha}"] + (["GEN_PREFIX_FRAMES=1"] if fam == "ed" else [])),
               str(REPO / "misc/2026-09-07_eval_grid_v2/score_v3.sbatch")]
        if dry:
            print(" ".join(cmd))
            continue
        jid = sh(cmd).strip()
        ledger[ha] = jid
        LEDGER.write_text(json.dumps(ledger, indent=1))
        print(f"{ha:40s} -> {jid}")


def status():
    for arm, tier, fam, ha in all_arms():
        d = REPO / EVAL_ENTRY / ha
        n = 0
        for f in d.glob("*/items.jsonl"):
            n += sum(1 for l in f.read_text().splitlines() if l.strip())
        planned = 0
        for f in (EVALDIR / "manifests" / ha).glob("eval_c*.json"):
            planned += len(json.loads(f.read_text()))
        print(f"{ha:40s} scored rows {n:6d} / planned {planned:6d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["register", "plan", "submit", "status"])
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"register": register, "plan": lambda: plan(a.chunks), "submit": lambda: submit(a.chunks, a.dry_run), "status": status}[a.cmd]()


if __name__ == "__main__":
    main()
