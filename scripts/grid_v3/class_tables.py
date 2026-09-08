#!/usr/bin/env python
"""grid v3 — per-class diagnostic tables requested by the owner (2026-09-08), appended to eval/REPORT.md.

Per donor class over ALL its rows (same + cross + foreign pooled; %_proxy rows included -> ranking only):
  T1  base_cond neutral / effect, Δ = effect − neutral, the reference arms' effect levels, class ceiling — sorted by Δ
  T2  score = (DCG w6 effect − base effect) − (base effect − base neutral): high where the effect clause adds little
      and the reference (with DCG) adds much — sorted descending
  T3  the neutral/effect novelty tables restricted to the classes with score > 0 (a diagnostic slice chosen on
      these same scores — stated as such, never an unbiased estimate)
Levels only; n = rows; 2-3-row classes carry the ~12 pp per-generation seed SD.
"""
from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval_ladder")); sys.path.insert(0, str(REPO / "scripts/grid_v3"))
import run_eval  # noqa: E402
import closeout  # noqa: E402
from launch_gen import ARMS  # noqa: E402

LAB = {"base_cond": "base_cond", "ic_gen": "ic_gen", "dualforce_control": "dualforce control", "dualforce_dcg_w6": "dualforce DCG w=6"}


def nov_of(r):
    c = r["cell"]
    return "seen" if c in ("G-fit", "G-memo-probe") else "unseen" if c.startswith("G-unseen") else "zero_shot"


def main():
    ceil = run_eval.ceilings(); E = closeout.eval_entry()

    def pct(ha, keep=None):
        rows = {r["item_id"]: r for r in map(json.loads, filter(str.strip, open(REPO / f"eval_ladder/registry_{ha}.jsonl")))
                if r["arm"] == ha and (keep is None or r["gt_pool_class"] in keep)}
        return rows, run_eval.item_pct(run_eval.pool_means(E / ha), rows, ceil)

    def per_class(ha):
        rows, x = pct(ha); d = collections.defaultdict(list)
        for i, v in x.items():
            d[rows[i]["gt_pool_class"]].append(v)
        return {c: (st.mean(v) * 100, len(v)) for c, v in d.items()}

    def pooled(rows, x, nov=None):
        v = [x[i] for i in x if nov is None or nov_of(rows[i]) == nov]
        return f"{st.mean(v)*100:.1f} ({len(v)})" if v else "—"

    out = ["", "## Per-class diagnostics (owner request 2026-09-08)", "",
           "Per donor class over all its rows (same + cross + foreign pooled, so ranking only). Levels, n = rows; 2-3-row classes carry the ~12 pp seed SD."]
    selected = {}
    for fam, label in (("v3", "Higgsfield 121 f"), ("v3ed81", "EffectData 81 f")):
        bn = per_class(f"base_cond_neutral_{fam}"); be = per_class(f"base_cond_effect_{fam}")
        dn = per_class(f"dualforce_control_neutral_{fam}"); de = per_class(f"dualforce_control_effect_{fam}"); ge = per_class(f"dualforce_dcg_w6_effect_{fam}")
        out += ["", f"### T1 {label} — sorted by base_cond Δ = effect − neutral", "",
                "| class | n | base neutral | base effect | Δ eff−neu | dualforce ctrl neutral | dualforce ctrl effect | DCG w6 effect | ceiling |", "|---|---|---|---|---|---|---|---|---|"]
        for c in sorted(be, key=lambda c: be[c][0] - bn[c][0]):
            out.append(f"| {c} | {be[c][1]} | {bn[c][0]:.1f} | {be[c][0]:.1f} | {be[c][0]-bn[c][0]:+.1f} | {dn[c][0]:.1f} | {de[c][0]:.1f} | {ge[c][0]:.1f} | {ceil.get(c, float('nan')):.2f} |")
        out += ["", f"### T2 {label} — score = (DCG w6 effect − base effect) − (base effect − base neutral), descending", "",
                "| class | n | base neutral | base effect | Δ eff−neu | DCG w6 effect | DCG − base effect | ctrl − base effect | score |", "|---|---|---|---|---|---|---|---|---|"]
        scored = sorted(((ge[c][0] - be[c][0]) - (be[c][0] - bn[c][0]), c) for c in be)
        for s, c in reversed(scored):
            out.append(f"| {c} | {be[c][1]} | {bn[c][0]:.1f} | {be[c][0]:.1f} | {be[c][0]-bn[c][0]:+.1f} | {ge[c][0]:.1f} | {ge[c][0]-be[c][0]:+.1f} | {de[c][0]-be[c][0]:+.1f} | {s:+.1f} |")
        selected[fam] = {c for s, c in scored if s > 0}
    out += ["", "### T3 novelty tables restricted to the score > 0 classes (diagnostic slice, selected on these scores)", "",
            f"Higgsfield: {sorted(selected['v3'])} · EffectData: {sorted(selected['v3ed81'])}"]
    for tier in ("neutral", "effect"):
        out += ["", f"**{tier} prompts — selected classes only**", "",
                "| arm | seen (HF) | unseen (HF) | zero-shot (HF) | all HF rows | zero-shot (ED 81 f) |", "|---|---|---|---|---|---|"]
        for arm in ARMS:
            r, x = pct(f"{arm}_{tier}_v3", selected["v3"]); re_, xe = pct(f"{arm}_{tier}_v3ed81", selected["v3ed81"])
            out.append(f"| {LAB[arm]} | {pooled(r, x, 'seen')} | {pooled(r, x, 'unseen')} | {pooled(r, x, 'zero_shot')} | {pooled(r, x)} | {pooled(re_, xe)} |")
    rep = closeout.EVALDIR / "REPORT.md"
    text = rep.read_text()
    marker = "## Per-class diagnostics"
    text = text[: text.index(marker)].rstrip("\n") + "\n" if marker in text else text
    rep.write_text(text + "\n".join(out) + "\n")
    print(f"appended {len(out)} lines to {rep.relative_to(REPO)}")


if __name__ == "__main__":
    main()
