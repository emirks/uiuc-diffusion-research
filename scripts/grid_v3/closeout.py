#!/usr/bin/env python
"""grid v3 — close the scoring pass into the store: per-arm headline table, eval meta.yaml, scores links, INDEX rows.

  summary   per arm: pool-% of m1a (item_pct, the deployed yardstick) by reference-novelty x endpoint-content cell,
            %_same cells headline-eligible, (%_proxy) ranking-only, + copy_max mean; -> <eval entry>/summary.json and
            misc/2026-09-07_eval_grid_v2/eval/REPORT.md
  write     summary + eval meta.yaml (immutable once written) + `scores` symlink in every gen subentry + INDEX rows to paste
Numbers are recorded neutrally: level, n, cell; no verdicts.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from datetime import date
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/grid_v3"))
sys.path.insert(0, str(REPO / "eval_ladder"))
from launch_gen import ARMS, TIERS, harness_arm, registry, subentry  # noqa: E402
import run_eval  # noqa: E402

EVALDIR = REPO / "misc/2026-09-07_eval_grid_v2/eval"
LEDGER = EVALDIR / "score_ledger.json"
CELL_ORDER = ["seen", "unseen", "zero_shot"]
CONTENT_ORDER = ["same", "cross", "foreign"]


def eval_entry() -> Path:
    hits = sorted((REPO / "store/evals").glob("028_grid_v3_paper_arms__dai__*"))
    assert len(hits) == 1, f"expected exactly one 028 eval entry, found {[h.name for h in hits]} (merge first)"
    return hits[0]


def all_arms():
    for arm in ARMS:
        for tier in TIERS:
            for fam in ("hf", "ed"):
                yield arm, tier, fam, harness_arm(arm, tier, fam)


def arm_summary(ha: str, arm: str, tier: str, fam: str, entry: Path) -> dict | None:
    scores = entry / ha
    if not scores.is_dir():
        return None
    rows = {r["item_id"]: r for r in map(json.loads, filter(str.strip, registry(arm, tier, fam).read_text().splitlines())) if r["arm"] == ha}
    pool = run_eval.pool_means(scores)
    if not pool:
        return None
    ceil = run_eval.ceilings()
    pct = run_eval.item_pct(pool, rows, ceil)
    copy_by_item: dict[str, list[float]] = collections.defaultdict(list)
    n_items = 0
    for f in sorted(scores.glob("*/items.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            n_items += 1
            if r.get("copy_max") is not None:
                copy_by_item[r["item_id"].rpartition("__ref_")[0].rpartition("__s")[0]].append(float(r["copy_max"]))
    cells: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    ptype: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    copies: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for item, v in pct.items():
        r = rows[item]
        key = (r["ref_novelty"], r["content"])
        cells[key].append(v)
        ptype[key].add(r["pct_type"])
        if item in copy_by_item:
            copies[key].append(st.mean(copy_by_item[item]))
    out_cells = {}
    for key in sorted(cells, key=lambda k: (CELL_ORDER.index(k[0]) if k[0] in CELL_ORDER else 9, CONTENT_ORDER.index(k[1]) if k[1] in CONTENT_ORDER else 9)):
        pt = ptype[key]
        assert len(pt) == 1, f"{ha} {key}: mixed % types {pt}"
        vals = cells[key]
        out_cells[f"{key[0]}|{key[1]}"] = {"novelty": key[0], "content": key[1], "pct_type": pt.pop(), "n": len(vals),
                                           "level": round(st.mean(vals), 4), "sd": round(st.pstdev(vals), 4) if len(vals) > 1 else None,
                                           "copy_max_mean": round(st.mean(copies[key]), 4) if copies[key] else None}
    same = [v for k, vs in cells.items() if k[1] == "same" for v in vs]
    return {"arm": ha, "rows": len(rows), "items_scored": len(pct), "score_rows": n_items,
            "headline_pct_same": round(st.mean(same), 4) if same else None, "n_same": len(same), "cells": out_cells}


def summary(write: bool) -> dict:
    entry = eval_entry()
    out = {}
    for arm, tier, fam, ha in all_arms():
        s = arm_summary(ha, arm, tier, fam, entry)
        if s:
            out[ha] = s
    lines = [f"# grid v3 — paper arms, v4 pool-% (eval entry `{entry.name}`)", "",
             "pool-% = mean app_ref vs the row's GT pool ÷ class ceiling (certified matrix; ceilings_v3.json overlay for new classes); "
             "%_same cells are cross-class comparable, (%_proxy) cells are ranking-only. Levels, not verdicts. "
             "ED81 arms: 81 f, frame-0 anchor — copy/core flags not comparable with 121 f.", "",
             "| arm | rows scored | %_same headline (n) | " + " | ".join(f"{n}/{c}" for n in CELL_ORDER for c in CONTENT_ORDER) + " |",
             "|---|---|---|" + "---|" * (len(CELL_ORDER) * len(CONTENT_ORDER))]
    for ha, s in out.items():
        cells = []
        for n in CELL_ORDER:
            for c in CONTENT_ORDER:
                x = s["cells"].get(f"{n}|{c}")
                cells.append("—" if not x else (f"{x['level']*100:.1f} ({x['n']})" if x["pct_type"] == "same" else f"({x['level']*100:.1f}) ({x['n']})"))
        head = f"{s['headline_pct_same']*100:.1f} ({s['n_same']})" if s["headline_pct_same"] is not None else "—"
        lines.append(f"| `{ha}` | {s['items_scored']}/{s['rows']} | {head} | " + " | ".join(cells) + " |")
    lines += ["", "copy_max mean per cell is in summary.json."]
    text = "\n".join(lines) + "\n"
    print(text)
    if write:
        (entry / "summary.json").write_text(json.dumps({"created": date.today().isoformat(), "arms": out}, indent=1))
        (EVALDIR / "REPORT.md").write_text(text)
        print(f"[write] {entry.relative_to(REPO)}/summary.json + {EVALDIR.relative_to(REPO)}/REPORT.md")
    return out


def write_meta(entry: Path, summ: dict) -> None:
    meta_p = entry / "meta.yaml"
    if meta_p.exists():
        print(f"[meta] exists, immutable: {meta_p.relative_to(REPO)}")
        return
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    arms = {}
    for arm, tier, fam, ha in all_arms():
        sub = REPO / subentry(arm, tier, fam)
        gm = yaml.safe_load((sub / "meta.yaml").read_text()) if (sub / "meta.yaml").exists() else {}
        s = summ.get(ha)
        arms[ha] = {"gen": f"gens/{sub.parent.name}/{sub.name}", "run": ARMS[arm]["run"], "step": ARMS[arm].get("step"),
                    "rows": s["rows"] if s else None, "items_scored": s["items_scored"] if s else None,
                    "pct_same_headline": s["headline_pct_same"] if s else None, "n_same": s["n_same"] if s else None,
                    "frames": 81 if fam == "ed" else 121, "prefix_frames": 1 if fam == "ed" else 9,
                    "prompt_family": gm.get("prompt_family"), "slurm_job": ledger.get(ha)}
    seq = int(entry.name.split("_")[0])
    meta = {
        "id": entry.name, "seq": seq, "shelf": "evals", "created": date.today().isoformat(),
        "machine": "dai — NCSA DeltaAI GH200 (aarch64), ghx4, bhwp-dtai-gh + bgjg-dtai-gh",
        "harness": "transition-eval v4.0.0 (eval-v4-cert worktree, branch eval/v4-metrics @a2b97c6 — the --reference-corpus amendment; [UNCERTIFIED] by design)",
        "instrument": {
            "reference_v4_sha256": "459fd9a71bb50ef81dcbd1d881aecf6a6c70e18855899b37c0a64b7167e606a8",
            "reference_corpus": "data/processed/transitions_std121/corpus_manifest_v1_222.json  # the certified 222-clip manifest the reference pin is verified against",
            "corpus": "data/processed/transitions_std121/corpus_manifest.json  # 677 clips / 97 classes (datasets/006) — supplies the grid-v3 pool references",
            "ceilings": "certified matrix m1a_S3 for the original classes + eval_ladder/ceilings_v3.json overlay (deployed kernel vs frozen populations; calibration ratio 0.9965 on 5 classes)",
            "tau_copy": 0.858, "matrix": "m1a_S3",
            "cache": "misc/refvfx_baseline/probe/cache",
            "env": "envs-aarch64/refvfx (module miniforge3_pytorch/2.10.0); PYTHONPATH=.claude/worktrees/eval-v4-cert/src",
            "job": "misc/2026-09-07_eval_grid_v2/score_v3.sbatch (16 shards/arm; EffectData arms GEN_PREFIX_FRAMES=1)",
        },
        "grid": {"design": "misc/2026-09-07_eval_grid_v2/PROPOSAL.md (v3.0.0)", "registry": "eval_ladder/registry_v3.jsonl (384 rows: 139 kept from the 152-row grid + 245 new)",
                 "families": {"neutral_hf": "prompts/010_ctt_v3_neutral", "effect_hf": "prompts/011_ctt_v3_effect",
                              "neutral_ed": "prompts/012_ctt_v3ed_neutral", "effect_ed": "prompts/013_ctt_v3ed_effect"},
                 "cells": "reference novelty {seen, unseen, zs_higgsfield, zs_effectdata} x endpoint content {same, cross, foreign}; %_same headline, %_proxy ranking-only",
                 "seeds": [42, 43]},
        "source": "Grid v3 (superset of the 152-row CTT grid) scored for the four paper arms x neutral/effect prompt tiers x Higgsfield-121f / EffectData-81f families in ONE pass, one machine. Kept rows reuse the byte-identical earlier generations (hardlinks) and are RE-SCORED here against the enlarged pools (datasets/006 caveat: 17 topped-up classes). The EffectData tier is zero-shot for every arm here (none trained on S6). Levels only; margins are computed by the paper tables.",
        "arms_scored": arms,
        "summary": "summary.json (per arm x cell: pool-% level, n, sd, copy_max mean); misc/2026-09-07_eval_grid_v2/eval/REPORT.md",
    }
    meta_p.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=140))
    print(f"[meta] wrote {meta_p.relative_to(REPO)}")


def link_scores(entry: Path) -> None:
    for arm, tier, fam, ha in all_arms():
        sub = REPO / subentry(arm, tier, fam)
        if not (entry / ha).is_dir() or not (sub / "meta.yaml").exists():
            continue
        link = sub / "scores"
        target = Path("../../../evals") / entry.name / ha
        if link.is_symlink() and link.readlink() == target:
            continue
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)
        print(f"[link] {link.relative_to(REPO)} -> {target}")


def index_rows(entry: Path, summ: dict) -> None:
    print("\n=== INDEX rows (gens: append to each arm's line; evals: new numbered row) ===")
    for arm in ARMS:
        parts = []
        for tier in TIERS:
            for fam in ("hf", "ed"):
                ha = harness_arm(arm, tier, fam)
                sub = REPO / subentry(arm, tier, fam)
                s = summ.get(ha)
                head = f" {s['headline_pct_same']*100:.1f}" if s and s["headline_pct_same"] is not None else ""
                parts.append(f"`{sub.name}`{head}")
        print(f"  {ARMS[arm]['gen_dir']}: grid v3 → " + " · ".join(parts) + f" (v4 → evals/{entry.name.split('__')[0]})")
    seq = int(entry.name.split("_")[0])
    heads = ", ".join(f"{ha.replace('_v3','').replace('ed81','·ED')} {s['headline_pct_same']*100:.1f}" for ha, s in summ.items() if s["headline_pct_same"] is not None)
    print(f"{seq}. `{entry.name}` — v4 on DeltaAI, grid v3 (384 rows, seeds 42/43; 139 kept + 245 new; ED tier 81 f) — 16 arms in ONE pass: "
          f"{{base_cond, ic_gen, dualforce_control, dualforce_dcg_w6}} x {{neutral, effect}} x {{HF-121f, ED-81f}}. %_same headline: {heads}. "
          f"Pools from datasets/006 (677 clips), ceilings overlay ceilings_v3.json; --reference-corpus amendment (a2b97c6). summary.json + eval/REPORT.md.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summary", "write"])
    a = ap.parse_args()
    entry = eval_entry()
    summ = summary(write=a.cmd == "write")
    if a.cmd == "write":
        write_meta(entry, summ)
        link_scores(entry)
        index_rows(entry, summ)


if __name__ == "__main__":
    main()
