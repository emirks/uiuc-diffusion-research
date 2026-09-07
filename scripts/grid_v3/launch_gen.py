#!/usr/bin/env python
"""grid v3 — generation launcher for the four paper arms x two prompt tiers x two families (contract v2).

  stamp    per-arm registries (stamp_rows.py over prompts/010-013; base twins appended for neutral, no_twin for effect)
           + arms.yaml entries for the harness arms that do not exist yet
  prepare  create the store gen subentries and HARDLINK the existing generations of the 139 kept rows
           (same cell / endpoint / reference / seed => byte-identical input => the same video)
  submit   sbatch arrays per (arm, tier, family): wave 1 = neutral tiers, wave 2 = effect tiers after wave 1
  status   videos present vs expected per subentry + queue summary

Arms (canonical -> old subentries reused):
  base_cond            gens/005_base_cond/{02_neutral,01_effect}__dai         no adapter, prefix(+suffix), no reference
  ic_gen               gens/001_ic_gen/{01_neutral__cc,02_effect__dai}        runs/001@5000 r32 bidirectional
  dualforce_control    gens/013_dualforce_control/{01_neutral,02_effect}__dai runs/012@1000 r128 one_way
  dualforce_dcg_w6     gens/032_dualforce_dcg_w6/{01_neutral,02_effect}__dai  runs/012@1000 + DCG w=6 (seed 42 only)
Families: 010/011 = Higgsfield + reserve rows (121 f, 9-frame prefix) · 012/013 = EffectData tier (81 f, frame-0 anchor).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
STORE = REPO / "store"
EVAL = REPO / "eval_ladder"
PY = "/taiga/illinois/eng/cs/jrehg/users/emirkisa/envs-aarch64/ltx2/bin/python"
GENDIR = REPO / "misc/2026-09-07_eval_grid_v2/gen"
LEDGER = GENDIR / "ledger.json"
ADAPTER_012 = "store/runs/012_dualforce_control/checkpoints/lora_weights_step_01000.safetensors"
ADAPTER_001 = "store/runs/001_ic_gen/checkpoints/lora_weights_step_05000.safetensors"

FAMILIES = {  # tier -> (higgsfield family, effectdata family)
    "neutral": ("010_ctt_v3_neutral", "012_ctt_v3ed_neutral"),
    "effect": ("011_ctt_v3_effect", "013_ctt_v3ed_effect"),
}
ARMS = {
    # canonical: dict(gen_dir, stack, rank, alpha, run, code, old={tier: old subentry}, kind, next_kk)
    "base_cond": dict(gen_dir="005_base_cond", stack="official", rank=32, alpha=32, run="base weights (no adapter)",
                      code="src/LTX-2-official (venv editable) @ grid v3", kind="base", account="bhwp-dtai-gh",
                      old={"neutral": "005_base_cond/02_neutral__dai", "effect": "005_base_cond/01_effect__dai"}, next_kk=4),
    "ic_gen": dict(gen_dir="001_ic_gen", stack="official", rank=32, alpha=32, run="runs/001_ic_gen", step=5000,
                   code="src/LTX-2-official (venv editable) @ grid v3", kind="generalist", account="bhwp-dtai-gh",
                   old={"neutral": "001_ic_gen/01_neutral__cc", "effect": "001_ic_gen/02_effect__dai"}, next_kk=3),
    "dualforce_control": dict(gen_dir="013_dualforce_control", stack="ctt", rank=128, alpha=128, run="runs/012_dualforce_control", step=1000,
                              code="src/LTX-2-ctt-v2-train packages @ grid v3 (one_way IC-LoRA stack)", kind="generalist", account="bgjg-dtai-gh",
                              old={"neutral": "013_dualforce_control/01_neutral__dai", "effect": "013_dualforce_control/02_effect__dai"}, next_kk=3),
    "dualforce_dcg_w6": dict(gen_dir="032_dualforce_dcg_w6", stack="dcg", rank=128, alpha=128, run="runs/012_dualforce_control", step=1000,
                             code="src/LTX-2-ctt-v2-train packages + scripts/grid_v3/gen_dcg_v3.py (crossfade null, gs4/stg1, w=6)", kind="generalist", account="bgjg-dtai-gh",
                             old={"neutral": "032_dualforce_dcg_w6/01_neutral__dai", "effect": "032_dualforce_dcg_w6/02_effect__dai"}, next_kk=3),
}
TIERS = ["neutral", "effect"]
FAMKEY = {"hf": 0, "ed": 1}


def harness_arm(arm: str, tier: str, fam: str) -> str:
    return f"{arm}_{tier}_v3" + ("ed81" if fam == "ed" else "")


def subentry(arm: str, tier: str, fam: str) -> str:
    kk = ARMS[arm]["next_kk"] + TIERS.index(tier) * 2 + FAMKEY[fam]
    return f"store/gens/{ARMS[arm]['gen_dir']}/{kk:02d}_{tier}_v3{'ed81' if fam == 'ed' else ''}__dai"


def registry(arm: str, tier: str, fam: str) -> Path:
    return EVAL / f"registry_{harness_arm(arm, tier, fam)}.jsonl"


def key_of(item_id: str) -> tuple:
    """(cell, endpoint, reference) from an item_id of either grammar; the arm token and a '__dfw6' tag are dropped."""
    parts = item_id.split("__")
    cell, ep = parts[0], parts[2]
    ref = next((p[4:] for p in parts[3:] if p.startswith("ref_")), "")
    return cell, ep, ref


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


# --------------------------------------------------------------------------- stamp
def stamp() -> None:
    base_rows = [l for l in (EVAL / "registry_v3.jsonl").read_text().splitlines() if l.strip() and json.loads(l)["arm"] == "base"]
    arms_cfg = yaml.safe_load((EVAL / "arms.yaml").read_text())
    new_entries = []
    for arm, spec in ARMS.items():
        for tier in TIERS:
            for fam in ("hf", "ed"):
                ha = harness_arm(arm, tier, fam)
                family = FAMILIES[tier][FAMKEY[fam]]
                out = registry(arm, tier, fam)
                if not out.exists():
                    cmd = [PY, str(EVAL / "stamp_rows.py"), "--family", family, "--arm", ha, "--out", str(out)]
                    if spec["kind"] == "base":
                        cmd += ["--strip-token", "--set", "conditioning=prefix", "--set", "use_reference=false", "--set", "no_twin=true"]
                    elif tier == "effect":
                        cmd += ["--set", "no_twin=true"]
                    print(sh(cmd).splitlines()[-1])
                    if spec["kind"] != "base" and tier == "neutral":
                        with out.open("a") as f:
                            for l in base_rows:
                                f.write(l + "\n")
                if ha not in arms_cfg["arms"]:
                    fam_id = family
                    note = f"grid v3 {tier}, prompts/{fam_id}" + (" (strip_sksz)" if spec["kind"] == "base" else "") + \
                        ("; EffectData tier NATIVE 81 f + frame-0 anchor: GEN_FRAMES=81 GEN_PREFIX_FRAMES=1" if fam == "ed" else "")
                    if spec["kind"] == "base":
                        entry = {"kind": "base", "adapter": None, "targets": None, "note": note}
                    elif arm == "ic_gen":
                        entry = {"kind": "generalist", "targets": "attn_ffn", "step": 5000, "adapter": ADAPTER_001,
                                 "note": note + " (runs/001@5000, r32/a32, bidirectional)"}
                    elif arm == "dualforce_control":
                        entry = {"kind": "generalist", "targets": "attn_ffn", "step": 1000, "attention": "one_way", "adapter": ADAPTER_012,
                                 "note": note + " (runs/012@1000, r128/a128, one_way)"}
                    else:
                        entry = {"kind": "generalist", "targets": "attn_ffn", "step": 1000, "attention": "one_way", "adapter": ADAPTER_012,
                                 "note": note + " (runs/012@1000 + test-time DCG w=6, crossfade null, gs4/stg1; generated by scripts/grid_v3/gen_dcg_v3.py)"}
                    new_entries.append((ha, entry))
    if new_entries:
        txt = (EVAL / "arms.yaml").read_text().rstrip("\n") + "\n\n  # ---- GRID v3 paper arms (2026-09-07, scripts/grid_v3/launch_gen.py stamp): the four arms x neutral/effect x families\n"
        for ha, e in new_entries:
            txt += f"  {ha}:\n" + "".join(f"    {k}: {json.dumps(v) if not isinstance(v, str) else v}\n" for k, v in e.items() if k != "note") + f"    note: \"{e['note']}\"\n"
        (EVAL / "arms.yaml").write_text(txt + "\n")
        yaml.safe_load((EVAL / "arms.yaml").read_text())
        print(f"arms.yaml: +{len(new_entries)} arms")


# --------------------------------------------------------------------------- prepare
def prepare() -> None:
    report = {}
    for arm, spec in ARMS.items():
        for tier in TIERS:
            for fam in ("hf", "ed"):
                sub = REPO / subentry(arm, tier, fam)
                vids = sub / "videos"
                vids.mkdir(parents=True, exist_ok=True)
                rows = [json.loads(l) for l in registry(arm, tier, fam).read_text().splitlines() if l.strip()]
                rows = [r for r in rows if r["arm"] == harness_arm(arm, tier, fam)]
                linked = 0
                if fam == "hf":
                    old = STORE / "gens" / spec["old"][tier] / "videos"
                    by_key = {key_of(r["item_id"]): r["item_id"] for r in rows}
                    for p in sorted(old.glob("*.mp4")):
                        m = re.match(r"^(.*)__s(\d+)\.mp4$", p.name)
                        if not m:
                            continue
                        k = key_of(m.group(1))
                        new_id = by_key.get(k)
                        if new_id is None:
                            continue
                        dst = vids / f"{new_id}__s{m.group(2)}.mp4"
                        if not dst.exists():
                            try:
                                os.link(p, dst)
                            except OSError:
                                import shutil
                                shutil.copy2(p, dst)
                            linked += 1
                expected = len(rows) * 2
                have = len(list(vids.glob("*.mp4")))
                report[subentry(arm, tier, fam)] = {"rows": len(rows), "expected_videos": expected, "linked_from_old": linked, "present": have}
                print(f"{subentry(arm, tier, fam):52s} rows {len(rows):3d} expected {expected:3d} linked {linked:3d} present {have:3d}")
    (GENDIR / "prepare_report.json").write_text(json.dumps(report, indent=1))


# --------------------------------------------------------------------------- submit
SIZING = {  # (stack, fam) -> (nchunks, walltime)
    ("official", "hf"): (8, "01:15:00"), ("official", "ed"): (3, "00:50:00"),
    ("ctt", "hf"): (8, "01:15:00"), ("ctt", "ed"): (3, "00:50:00"),
    ("dcg", "hf"): (14, "01:15:00"), ("dcg", "ed"): (5, "00:50:00"),
}


def submit(tiers: list[str], dependency: str | None, dry: bool) -> None:
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    for tier in tiers:
        for arm, spec in ARMS.items():
            for fam in ("hf", "ed"):
                ha = harness_arm(arm, tier, fam)
                sub = subentry(arm, tier, fam)
                if ha in ledger and not dry:
                    print(f"skip {ha}: already submitted as {ledger[ha]['job']}")
                    continue
                nchunks, wall = SIZING[(spec["stack"], fam)]
                ntasks = nchunks * 2
                env = {"ARM": ha, "REG": str(registry(arm, tier, fam).relative_to(REPO)), "OUT": sub, "NCHUNKS": str(nchunks)}
                if fam == "ed":
                    env.update({"GEN_FRAMES": "81", "GEN_PREFIX_FRAMES": "1"})
                if spec["stack"] == "dcg":
                    script = GENDIR / "job_dcg_v3.sbatch"
                    env["W"] = "6.0"
                else:
                    script = GENDIR / "job_gen_v3.sbatch"
                    env.update({"STACK": spec["stack"], "RANK": str(spec["rank"]), "ALPHA": str(spec["alpha"])})
                cmd = ["sbatch", "--parsable", f"--account={spec['account']}", f"--array=0-{ntasks - 1}%16", f"--time={wall}",
                       f"--job-name=v3_{arm[:8]}_{tier[0]}{fam}", "--export=ALL," + ",".join(f"{k}={v}" for k, v in env.items())]
                if dependency:
                    cmd.append(f"--dependency={dependency}")
                cmd.append(str(script))
                if dry:
                    print(" ".join(cmd))
                    continue
                jid = sh(cmd)
                ledger[ha] = {"job": jid, "subentry": sub, "tasks": ntasks, "account": spec["account"], "tier": tier, "family": FAMILIES[tier][FAMKEY[fam]],
                              "submitted": date.today().isoformat(), "dependency": dependency}
                LEDGER.write_text(json.dumps(ledger, indent=1))
                print(f"{ha:40s} -> job {jid}  ({ntasks} tasks, {spec['account']}, {wall})")


# --------------------------------------------------------------------------- status
def status() -> None:
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    tot_have = tot_exp = 0
    for arm in ARMS:
        for tier in TIERS:
            for fam in ("hf", "ed"):
                ha = harness_arm(arm, tier, fam)
                sub = REPO / subentry(arm, tier, fam)
                rows = sum(1 for l in registry(arm, tier, fam).read_text().splitlines() if l.strip() and json.loads(l)["arm"] == ha)
                have = len(list((sub / "videos").glob("*.mp4"))) if (sub / "videos").exists() else 0
                tot_have += have
                tot_exp += rows * 2
                j = ledger.get(ha, {}).get("job", "-")
                print(f"{ha:40s} job {j:>10s}  videos {have:4d}/{rows * 2:4d}")
    print(f"TOTAL videos {tot_have}/{tot_exp}")
    try:
        print(subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-o", "%.14i %.16j %.9T %.12r %.8M %.6D"], capture_output=True, text=True).stdout[:3000])
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stamp", "prepare", "submit", "status"])
    ap.add_argument("--tiers", default="neutral,effect")
    ap.add_argument("--dependency", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "stamp":
        stamp()
    elif a.cmd == "prepare":
        prepare()
    elif a.cmd == "submit":
        submit(a.tiers.split(","), a.dependency, a.dry_run)
    else:
        status()


if __name__ == "__main__":
    main()
