#!/usr/bin/env python
"""grid v3 — hardness selectors compared per class (owner 2026-09-12).

Three per-class quantities over grid-v3 rows (levels, pool-% of m1a; 2 seeds):
  A  prompt-only hardness      = base_cond EFFECT level            (lower = harder; uses no compared arm)
  B  owner's target            = DCG w6 NEUTRAL − base_cond EFFECT  (reference alone vs best text; uses DCG = outcome)
  C  clause contribution       = base_cond EFFECT − base_cond NEUTRAL
Per class means use ALL content rows (same+cross+foreign; ranking only); tier readouts use SAME rows.
Writes misc/2026-09-07_eval_grid_v2/eval/HARDNESS.md.
"""
from __future__ import annotations
import collections, json, statistics as st, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval_ladder")); sys.path.insert(0, str(REPO / "scripts/grid_v3"))
import run_eval, closeout  # noqa: E402
from launch_gen import ARMS  # noqa: E402
LAB = {"base_cond": "base", "ic_gen": "ic_gen", "dualforce_control": "ctrl", "dualforce_dcg_w6": "DCG"}
K = 10  # classes per family selected under each approach

def nov_of(r):
    c = r["cell"]; return "seen" if c in ("G-fit", "G-memo-probe") else "unseen" if c.startswith("G-unseen") else "zero_shot"

def spearman(x, y):
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]: j += 1
            for k in range(i, j + 1): r[s[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rk(x), rk(y); n = len(x); mx, my = st.mean(rx), st.mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); vx = sum((a - mx) ** 2 for a in rx); vy = sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5 if vx and vy else float("nan")

def main():
    ceil = run_eval.ceilings(); E = closeout.eval_entry()
    def load(ha):
        rows = {r["item_id"]: r for r in map(json.loads, filter(str.strip, open(REPO / f"eval_ladder/registry_{ha}.jsonl"))) if r["arm"] == ha}
        return rows, run_eval.item_pct(run_eval.pool_means(E / ha), rows, ceil)
    out = ["# Hardness selectors compared (grid v3, evals/028; levels; 2 seeds)", "",
           "A = base_cond effect level (prompt-only hardness, lower = harder). B = DCG w6 neutral − base_cond effect (owner target: reference alone vs best text). C = base_cond effect − base_cond neutral (clause contribution).",
           "Per-class means over ALL content rows (ranking only); tier readouts over SAME rows. B and C use outcome arms → selecting on them is selection on the outcome; A is the only selector that does not.", ""]
    summary = {}
    for fam, label in (("v3", "Higgsfield 121 f"), ("v3ed81", "EffectData 81 f")):
        per = {}; same = {}; novs = collections.defaultdict(set); nall = collections.Counter(); nsame = collections.Counter()
        for arm in ARMS:
            for tier in ("neutral", "effect"):
                rows, x = load(f"{arm}_{tier}_{fam}"); d = collections.defaultdict(list); ds = collections.defaultdict(list)
                for i, v in x.items():
                    c = rows[i]["gt_pool_class"]; d[c].append(v); novs[c].add(nov_of(rows[i]))
                    if rows[i]["content"] == "same": ds[c].append(v)
                for c in d:
                    per[(arm, tier, c)] = st.mean(d[c]) * 100; same[(arm, tier, c)] = st.mean(ds[c]) * 100 if ds[c] else float("nan")
                    if arm == "base_cond" and tier == "effect": nall[c] = len(d[c]); nsame[c] = len(ds[c])
        classes = sorted({c for (_, _, c) in per})
        g = lambda arm, tier, c: per[(arm, tier, c)]
        A = {c: g("base_cond", "effect", c) for c in classes}
        B = {c: g("dualforce_dcg_w6", "neutral", c) - g("base_cond", "effect", c) for c in classes}
        C = {c: g("base_cond", "effect", c) - g("base_cond", "neutral", c) for c in classes}
        out += [f"## {label} — per class, sorted by A (hardest first)", "",
                "| class | novelty | n all/same | base neu | base eff (A) | ic_gen neu | ctrl neu | DCG neu | DCG eff | B = DCGneu−base eff | C = eff−neu | ceiling |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in sorted(classes, key=lambda c: A[c]):
            out.append(f"| {c} | {'/'.join(sorted(novs[c]))} | {nall[c]}/{nsame[c]} | {g('base_cond','neutral',c):.1f} | {A[c]:.1f} | {g('ic_gen','neutral',c):.1f} | {g('dualforce_control','neutral',c):.1f} | {g('dualforce_dcg_w6','neutral',c):.1f} | {g('dualforce_dcg_w6','effect',c):.1f} | {B[c]:+.1f} | {C[c]:+.1f} | {ceil.get(c, float('nan')):.2f} |")
        xs = classes
        out += ["", f"Spearman over {len(xs)} classes: ρ(A, B) = {spearman([A[c] for c in xs],[B[c] for c in xs]):+.2f} · ρ(C, B) = {spearman([C[c] for c in xs],[B[c] for c in xs]):+.2f} · ρ(A, C) = {spearman([A[c] for c in xs],[C[c] for c in xs]):+.2f}", ""]
        zs = [c for c in classes if novs[c] == {"zero_shot"}]
        selA = sorted(zs, key=lambda c: A[c])[:K]; selB = sorted(zs, key=lambda c: -B[c])[:K]; selC = sorted(zs, key=lambda c: C[c])[:K]
        out += [f"### Zero-shot candidates ({len(zs)} classes) — top-{K} under each selector", "",
                f"- A (lowest base effect): {selA}", f"- B (largest DCG neutral − base effect): {selB}", f"- C (smallest clause contribution): {selC}",
                f"- overlap A∩B = {len(set(selA)&set(selB))}/{K} · A∩C = {len(set(selA)&set(selC))}/{K} · B∩C = {len(set(selB)&set(selC))}/{K}", ""]
        def tier_row(sel, name):
            cells = []
            for arm in ARMS:
                for tier in ("neutral", "effect"):
                    vals = [same[(arm, tier, c)] for c in sel if same[(arm, tier, c)] == same[(arm, tier, c)]]
                    cells.append(f"{st.mean(vals):.1f}" if vals else "—")
            bn, be, dn = [float(cells[i]) for i in (0, 1, 6)]
            return f"| {name} | {len(sel)} | {sum(nsame[c] for c in sel)} | " + " | ".join(cells) + f" | {dn-be:+.1f} |"
        out += ["Tier readout on SAME rows (levels), classes as selected above:", "",
                "| selection | classes | same rows | base neu | base eff | ic_gen neu | ic_gen eff | ctrl neu | ctrl eff | DCG neu | DCG eff | DCG neu − base eff |", "|---|---|---|---|---|---|---|---|---|---|---|---|",
                tier_row(zs, "all zero-shot (v3 tier)"), tier_row(selA, f"A top-{K}"), tier_row(selB, f"B top-{K}"), tier_row(selC, f"C top-{K}"),
                tier_row([c for c in zs if c not in selA], "zero-shot minus A"), ""]
        summary[fam] = dict(A=A, B=B, C=C, zs=zs, selA=selA, selB=selB, selC=selC)
    rep = closeout.EVALDIR / "HARDNESS.md"; rep.write_text("\n".join(out) + "\n"); print(rep.relative_to(REPO))
    json.dump({f: {k: v for k, v in s.items()} for f, s in summary.items()}, open(rep.with_suffix(".json"), "w"), indent=1)

if __name__ == "__main__":
    main()
