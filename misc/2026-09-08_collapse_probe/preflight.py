#!/usr/bin/env python3
"""preflight.py — CPU checks that must pass before any GPU launch.

Default (pre-R1) mode:
  * reg/r1.jsonl exists and has one row per prompt
  * every R1 row's start window conds/<endpoint>_start9.mp4 exists
  * the GENERATOR'S OWN build_sample renders each R1 row's prompt == its full_prompt (printed)
  * banned tokens (trained triggers) absent; prompts non-empty and period-terminated
  * arm / seeds / paths consistent; both sbatch files reference existing paths

--post-r1 mode (after R1, before R2/R3):
  * for each seed with a reg/r2_s<seed>.jsonl: R2/R3 registries exist
  * every probe clip's start9 + end9 windows exist
  * R1 outputs decode to 121 frames; manifest start_sha_match all true
  * build_sample renders R2 == full_prompt and R3 == neutral_prompt

Non-zero exit on any failure.

    python preflight.py [PROMPTS_JSONL]
    python preflight.py [PROMPTS_JSONL] --post-r1 [--seeds 42,43,44]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import _probe_common as C

BANNED = ("sksz", "qvtr")  # trained transition-slot triggers must never appear in a base-model prompt


class Check:
    def __init__(self):
        self.fails = []
        self.n = 0

    def ok(self, cond, msg):
        self.n += 1
        if not cond:
            self.fails.append(msg)
        return cond


def _ffprobe_frames(path: Path) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return -1


def check_sbatch(chk: Check) -> None:
    for name in ("job_r1.sbatch", "job_r2r3.sbatch"):
        f = C.HERE / name
        if not chk.ok(f.exists(), f"sbatch missing: {name}"):
            continue
        txt = f.read_text()
        # every absolute path token that looks like a repo path must exist
        for tok in txt.split():
            if tok.startswith("/taiga/") and ("/" in tok) and not any(
                    c in tok for c in ("$", "%", "*", "{")):
                p = Path(tok)
                # only check things that look like files/dirs we own (skip log patterns)
                if p.suffix in (".py", ".sbatch", ".jsonl") or p.name == "activate":
                    chk.ok(p.exists(), f"{name}: referenced path does not exist: {tok}")
        chk.ok("run_gen.py" in txt, f"{name}: does not call run_gen.py")
        chk.ok(f"--arm {C.ARM}" in txt, f"{name}: does not use arm {C.ARM}")


def pre_r1(prompts, chk: Check) -> None:
    reg = C.REG / "r1.jsonl"
    if not chk.ok(reg.exists(), f"missing {reg} (run build_r1.py first)"):
        return
    rows = [json.loads(l) for l in reg.read_text().splitlines() if l.strip()]
    by_pid = {r["prompt_id"]: r for r in rows}
    chk.ok(len(rows) == len(prompts),
           f"reg/r1.jsonl has {len(rows)} rows, prompts has {len(prompts)}")

    for p in prompts:
        r = by_pid.get(p["prompt_id"])
        if not chk.ok(r is not None, f"{p['prompt_id']}: no R1 row"):
            continue
        chk.ok(r["arm"] == C.ARM, f"{p['prompt_id']}: arm != {C.ARM}")
        chk.ok(r["sided"] == "one", f"{p['prompt_id']}: sided != one")
        chk.ok(r.get("conditioning", "none") != "none", f"{p['prompt_id']}: conditioning is none")
        chk.ok(r.get("use_reference") is False, f"{p['prompt_id']}: use_reference != false")
        chk.ok(r["prompt"] == p["full_prompt"], f"{p['prompt_id']}: row prompt != full_prompt")
        w = C.CONDS / f"{r['endpoint']}_start9.mp4"
        chk.ok(w.exists(), f"{p['prompt_id']}: start window missing {w}")
        fp = r["prompt"].lower()
        for b in BANNED:
            chk.ok(b not in fp, f"{p['prompt_id']}: banned token {b!r} in prompt")
        chk.ok(bool(r["prompt"].strip()), f"{p['prompt_id']}: empty prompt")
        chk.ok(r["prompt"].rstrip().endswith("."), f"{p['prompt_id']}: prompt not period-terminated")

    # generator's own prompt rendering
    print("[preflight] rendering every R1 prompt through the generator's build_sample ...")
    rendered = C.render_via_generator(rows)
    for p in prompts:
        r = by_pid.get(p["prompt_id"])
        if r is None:
            continue
        got = rendered.get(r["item_id"])
        match = got == p["full_prompt"]
        chk.ok(match, f"{p['prompt_id']}: RENDERED prompt != full_prompt")
        flag = "OK " if match else "!! "
        print(f"  {flag}{p['prompt_id']} [{p['tier']}]  {got}")

    check_sbatch(chk)


def post_r1(prompts, seeds, chk: Check) -> None:
    man_path = C.REG / "splice_manifest.csv"
    man = {}
    if man_path.exists():
        for row in csv.DictReader(open(man_path)):
            man[(row["prompt_id"], row["seed"])] = row

    for seed in seeds:
        r2p = C.REG / f"r2_s{seed}.jsonl"
        r3p = C.REG / f"r3_s{seed}.jsonl"
        if not chk.ok(r2p.exists() and r3p.exists(),
                      f"seed {seed}: missing r2/r3 registry (run splice_r1.py --seed {seed})"):
            continue
        r2 = [json.loads(l) for l in r2p.read_text().splitlines() if l.strip()]
        r3 = [json.loads(l) for l in r3p.read_text().splitlines() if l.strip()]
        r2_by = {r["prompt_id"]: r for r in r2}
        r3_by = {r["prompt_id"]: r for r in r3}

        for p in prompts:
            pid = p["prompt_id"]
            clip = C.probe_clip(pid, seed)
            for suf in ("start9", "end9"):
                w = C.CONDS / f"{clip}_{suf}.mp4"
                chk.ok(w.exists(), f"seed {seed} {pid}: probe window missing {w}")
            m = man.get((pid, str(seed)))
            if chk.ok(m is not None, f"seed {seed} {pid}: no manifest entry"):
                chk.ok(int(m["r1_frames"]) == 121, f"seed {seed} {pid}: R1 frames {m['r1_frames']} != 121")
                chk.ok(str(m["start_sha_match"]).lower() in ("true", "1"),
                       f"seed {seed} {pid}: start_sha_match not true")
            r2r, r3r = r2_by.get(pid), r3_by.get(pid)
            if chk.ok(r2r is not None, f"seed {seed} {pid}: no R2 row"):
                chk.ok(r2r["sided"] == "two", f"seed {seed} {pid}: R2 sided != two")
                chk.ok(r2r["prompt"] == p["full_prompt"], f"seed {seed} {pid}: R2 prompt != full")
            if chk.ok(r3r is not None, f"seed {seed} {pid}: no R3 row"):
                chk.ok(r3r["sided"] == "two", f"seed {seed} {pid}: R3 sided != two")
                chk.ok(r3r["prompt"] == p["neutral_prompt"], f"seed {seed} {pid}: R3 prompt != neutral")

        # generator render proof for this seed's R2/R3
        rendered = C.render_via_generator(r2 + r3)
        for p in prompts:
            for run, by, want in (("R2", r2_by, p["full_prompt"]), ("R3", r3_by, p["neutral_prompt"])):
                r = by.get(p["prompt_id"])
                if r is None:
                    continue
                chk.ok(rendered.get(r["item_id"]) == want,
                       f"seed {seed} {p['prompt_id']} {run}: RENDERED != {'full' if run=='R2' else 'neutral'}")
        print(f"[preflight:post] seed {seed}: {len(r2)} R2 + {len(r3)} R3 rendered & window-checked")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", nargs="?", default=None)
    ap.add_argument("--post-r1", action="store_true")
    ap.add_argument("--seeds", default=",".join(str(s) for s in C.SEEDS))
    args = ap.parse_args()

    ppath = Path(args.prompts) if args.prompts else C.default_prompts()
    prompts = C.load_prompts(ppath)
    chk = Check()
    print(f"[preflight] prompts={ppath}  ({len(prompts)} rows)  mode={'post-r1' if args.post_r1 else 'pre-r1'}")

    if args.post_r1:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        post_r1(prompts, seeds, chk)
    else:
        pre_r1(prompts, chk)

    print(f"\n[preflight] {chk.n - len(chk.fails)}/{chk.n} checks passed")
    if chk.fails:
        print(f"[preflight] {len(chk.fails)} FAILURE(S):")
        for m in chk.fails:
            print("   -", m)
        raise SystemExit(1)
    print("[preflight] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
