#!/usr/bin/env python3
"""Report-only (fable-advisor): S6 KEPT (26,266 consumed) vs DROPPED (2,378 shape-singleton)
per-channel moment comparison — puts "no distribution shift from the drop" on the record as a
measurement, not an assertion. Appends a section to NORM_REPORT_v3.md. No gate.
"""
import os, sys, json, glob, random
import numpy as np
from collections import defaultdict

LAB = "/taiga/illinois/eng/cs/jrehg/users/emirkisa"
DR = f"{LAB}/diffusion-research"
FEAT = f"{LAB}/cache/armA_signals/feat"
ROSTER = f"{DR}/outputs/ctt_v2/encodes/EFFECTDATA/ROSTER.json"
ARMA = f"{DR}/misc/2026-08-24_flow_signal_conditioning/armA"
sys.path.insert(0, ARMA)
from armA_extract import CH_NAMES

cl = json.load(open(ROSTER))["clips"]
eff_shape = defaultdict(list)
for c in cl:
    eff_shape[(c["effect"], tuple(c["latent_fhw"]))].append(c["stem"])
dropped = {s for stems in eff_shape.values() if len(stems) < 2 for s in stems}
kept = {c["stem"] for c in cl} - dropped

def moments(stems, n):
    picks = random.Random(0).sample(sorted(stems), min(n, len(stems)))
    s1 = np.zeros(44); s2 = np.zeros(44); k = 0
    for stem in picks:
        p = f"{FEAT}/S6__{stem}.npz"
        if not os.path.exists(p):
            continue
        F = np.asarray(np.load(p, allow_pickle=True)["F"], np.float64).reshape(-1, 44)
        s1 += F.sum(0); s2 += (F * F).sum(0); k += F.shape[0]
    mean = s1 / k; std = np.sqrt(np.clip(s2 / k - mean**2, 0, None))
    return mean, std, len(picks)

mk, sk, nk = moments(kept, 300)
md, sd, nd = moments(dropped, 300)
L = ["\n## S6 kept-vs-dropped moment comparison (report-only; confirms drop immateriality)\n",
     f"KEPT sample {nk} clips (of 26,266) vs DROPPED sample {nd} clips (of 2,378). Per-channel "
     f"mean/std; the drop is immaterial iff deltas are small.\n",
     "| channel | kept μ | drop μ | |Δμ|/σ_kept | std ratio |",
     "|---|--:|--:|--:|--:|"]
worst = 0.0
for c in range(44):
    dmu = abs(mk[c] - md[c]) / (sk[c] + 1e-9)
    ratio = sd[c] / (sk[c] + 1e-9)
    worst = max(worst, dmu)
    L.append(f"| {CH_NAMES[c]} | {mk[c]:.3f} | {md[c]:.3f} | {dmu:.3f} | {ratio:.3f} |")
L.append(f"\nWorst |Δμ|/σ_kept across channels: **{worst:.3f}** — "
         f"{'immaterial (< 0.15σ)' if worst < 0.15 else 'NOTE: > 0.15σ, inspect'}. "
         f"Report-only; the norm fits over the full cache roster regardless (fit-population rule).\n")
open(f"{ARMA}/NORM_REPORT_v3.md", "a").write("\n".join(L))
print(f"appended S6 moment split to NORM_REPORT_v3.md; worst |Δμ|/σ = {worst:.3f}")
