#!/usr/bin/env python
"""Two-criterion gapper selection (owner 2026-09-12): LOWEST base_cond effect AND HIGHEST DCG w6 effect.

Candidates = zero-shot classes with a DCG effect measurement: grid v3 HF zero-shot (17, 2 seeds, all content rows),
grid v3 EffectData (34, 2 seeds, all rows), screen stage B (40 EffectData effects, ONE same row, seed 42).
Outputs: Pareto fronts, a threshold grid (how many effects satisfy base ≤ b and DCG ≥ d), and the shortlist.
Selection uses DCG (an outcome arm): diagnostic slice; the reportable numbers come from NEW rows (grid v4).
Appends to misc/2026-09-07_eval_grid_v2/eval/HARDNESS.md.
"""
from __future__ import annotations
import collections, csv, json, statistics as st, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval_ladder")); sys.path.insert(0, str(REPO / "scripts/grid_v3"))
import run_eval, closeout  # noqa: E402

def main():
    ceil = run_eval.ceilings(); E = closeout.eval_entry()
    def per_class(ha, nov=None):
        rows = {r["item_id"]: r for r in map(json.loads, filter(str.strip, open(REPO / f"eval_ladder/registry_{ha}.jsonl"))) if r["arm"] == ha}
        x = run_eval.item_pct(run_eval.pool_means(E / ha), rows, ceil); d = collections.defaultdict(list)
        for i, v in x.items():
            if nov is None or rows[i]["cell"].startswith(nov): d[rows[i]["gt_pool_class"]].append(v)
        return {c: (st.mean(v) * 100, len(v)) for c, v in d.items()}
    cand = []  # (name, source, base_eff, dcg_eff, dcg_neu, n, ceiling)
    for fam, src, nov in (("v3", "v3 HF zs", "G-zs"), ("v3ed81", "v3 ED", None)):
        be = per_class(f"base_cond_effect_{fam}", nov); ge = per_class(f"dualforce_dcg_w6_effect_{fam}", nov); gn = per_class(f"dualforce_dcg_w6_neutral_{fam}", nov)
        for c in be: cand.append((c, src, be[c][0], ge[c][0], gn[c][0], be[c][1], ceil.get(c, float("nan"))))
    lv = {r["effect"]: float(r["prompt_only_level"]) for r in csv.DictReader(open(REPO / "misc/2026-09-08_ed_gapper_screen/eval/screen_levels.csv"))}
    for l in (REPO / "misc/2026-09-08_ed_gapper_screen/eval/STAGE_B.md").read_text().splitlines():
        c = [x.strip() for x in l.strip("|").split("|")]
        if len(c) >= 6 and c[0] in lv:
            try: cand.append((c[0], "screen B", float(c[1]), float(c[2]), float("nan"), 1, float(c[4])))
            except ValueError: pass
    # Pareto fronts: minimise base_eff, maximise dcg_eff
    rest = list(cand); fronts = []
    while rest:
        f = [a for a in rest if not any((b[2] <= a[2] and b[3] >= a[3]) and (b[2] < a[2] or b[3] > a[3]) for b in rest)]
        fronts.append(sorted(f, key=lambda a: a[2])); rest = [a for a in rest if a not in f]
    out = ["", "## Two-criterion selection: lowest base effect AND highest DCG w6 effect (owner 2026-09-12)", "",
           f"Candidates: {sum(1 for a in cand if a[1]=='v3 HF zs')} v3 HF zero-shot classes + {sum(1 for a in cand if a[1]=='v3 ED')} v3 EffectData effects (2 seeds, all rows) + {sum(1 for a in cand if a[1]=='screen B')} screen effects (1 same row, seed 42). DCG neutral only exists on v3. Selection on an outcome arm → diagnostic; report on new rows.", "",
           "### Threshold grid — number of effects with base effect ≤ b and DCG effect ≥ d (v3 HF / v3 ED / screen)", "",
           "| base ≤ \\ DCG ≥ | 70 | 80 | 90 |", "|---|---|---|---|"]
    for b in (40, 50, 60, 70, 80):
        cells = []
        for d in (70, 80, 90):
            n = collections.Counter(a[1] for a in cand if a[2] <= b and a[3] >= d)
            cells.append(f"{n['v3 HF zs']} / {n['v3 ED']} / {n['screen B']} = **{sum(n.values())}**")
        out.append(f"| {b} | " + " | ".join(cells) + " |")
    out += ["", "### Pareto fronts 1–2 (non-dominated on the two criteria), then the rest with base ≤ 60 and DCG ≥ 80", "",
            "| effect | source | n | base eff | DCG eff | gap | DCG neu | ceiling | front |", "|---|---|---|---|---|---|---|---|---|"]
    shown = set()
    for fi, f in enumerate(fronts[:2], 1):
        for a in f:
            out.append(f"| {a[0]} | {a[1]} | {a[5]} | {a[2]:.1f} | {a[3]:.1f} | {a[3]-a[2]:+.1f} | {'—' if a[4]!=a[4] else f'{a[4]:.1f}'} | {a[6]:.2f} | {fi} |"); shown.add(a[0])
    for a in sorted(cand, key=lambda a: -(a[3] - a[2])):
        if a[0] not in shown and a[2] <= 60 and a[3] >= 80:
            out.append(f"| {a[0]} | {a[1]} | {a[5]} | {a[2]:.1f} | {a[3]:.1f} | {a[3]-a[2]:+.1f} | {'—' if a[4]!=a[4] else f'{a[4]:.1f}'} | {a[6]:.2f} | · |"); shown.add(a[0])
    sel = [a for a in cand if a[2] <= 60 and a[3] >= 80]
    out += ["", f"Shortlist rule base ≤ 60 & DCG ≥ 80 → {len(sel)} effects: mean base {st.mean(a[2] for a in sel):.1f}, mean DCG {st.mean(a[3] for a in sel):.1f}, gap {st.mean(a[3]-a[2] for a in sel):+.1f}; by source {dict(collections.Counter(a[1] for a in sel))}.",
            f"Same rule on v3 only (two seeds): {[a[0] for a in sel if a[1]!='screen B']}"]
    rep = closeout.EVALDIR / "HARDNESS.md"; text = rep.read_text(); m = "## Two-criterion selection"
    text = text[: text.index(m)].rstrip("\n") + "\n" if m in text else text
    rep.write_text(text + "\n".join(out) + "\n"); print("\n".join(out))

if __name__ == "__main__":
    main()
