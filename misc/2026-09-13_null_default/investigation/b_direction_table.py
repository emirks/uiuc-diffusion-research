"""Step B — guidance-direction table.

For each stratum (reference = GT twin for S-GRID; R1 witness for S-PROBE, tiers separate; R2 as alt),
space, descriptor d and candidate null N: does pushing the default AWAY from N move d TOWARD the reference?
  correct_dir  := sign(ref - default) == sign(default - N)      (default lies between N and ref, or N beyond)
  brief_crit   := sign(ref - N)       == sign(ref - default)    (the brief's wording; necessary, not sufficient)
  headroom     := |ref - default| / |default - N|                (small = little to push against / overshoot)
Group level uses medians; paired level uses per-endpoint (S-GRID) / per-(prompt,seed) (S-PROBE) units.
Also: empirical check on the Sep-08 re-measure — DCG (pixel-crossfade null of the demo, mechanism (a)) on
dualforce_control, paired by endpoint: did DR / M move in the direction the table predicts?
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd
from pathlib import Path

LAB = Path("/taiga/illinois/eng/cs/jrehg/users/emirkisa"); REPO = LAB / "diffusion-research"
CAMP = REPO / "misc/2026-09-13_null_default"; INV = CAMP / "investigation"
pc = pd.read_csv(CAMP / "results/per_clip.csv", keep_default_na=False, low_memory=False)
num = ["DR", "M", "cross", "nu_max", "nu_mean", "step_share", "path_over_gap", "swap_sharp", "local_at_peak",
       "trans_mean", "explained", "gap", "n_interior"]
for c in num:
    pc[c] = pd.to_numeric(pc[c], errors="coerce")

POINT_D = ["DR", "M", "cross", "nu_max", "path_over_gap", "step_share"]
TRANS_D = ["nu_max", "swap_sharp", "local_at_peak", "trans_mean"]


def sgn(x, eps=1e-9):
    return 0 if abs(x) < eps else (1 if x > 0 else -1)


def dino_lerp_analytic(n):
    """Straight line in DINO space, n interior points uniformly spaced: exact descriptor values."""
    alpha = (np.arange(n) + 1) / (n + 1)
    return dict(DR=0.0, M=float(((alpha >= 0.25) & (alpha <= 0.75)).mean()), cross=float(np.where(alpha >= 0.5)[0][0] / n),
                nu_max=float(np.minimum(alpha, 1 - alpha).max()), path_over_gap=1.0, step_share=float(1.0 / (n + 1)))


def unit_table(space, stratum, tier, ref_group, def_group, ref_key, def_key):
    """Return per-unit frame with ref/default/null descriptor values, unit = endpoint (S-GRID) or prompt_id+seed (S-PROBE)."""
    sub = pc[(pc.space == space) & (pc.stratum == stratum)]
    if tier:
        sub = sub[sub.tier == tier]
    descs = TRANS_D if space == "TRANS" else POINT_D
    ref = sub[(sub.group == ref_group) & (sub.kind == "main")]
    dfl = sub[(sub.group == def_group) & (sub.kind == "main")]
    if stratum == "S-GRID":
        # unit = endpoint; default averaged over seeds; landmarks belong to the GT twin (landmark_owner)
        r = ref.groupby("endpoint")[descs].median().add_prefix("ref_")
        d = dfl.groupby("endpoint")[descs].median().add_prefix("def_")
        lm = sub[sub.kind != "main"]
        lms = {k: lm[lm.kind == k].groupby("endpoint")[descs].median().add_prefix(f"{k}_") for k in lm.kind.unique()}
        out = r.join(d, how="inner")
    else:
        # unit = prompt_id + seed
        def key(x):
            return x["prompt_id"].astype(str) + "|" + x["seed"].astype(str)
        ref = ref.assign(unit=key(ref)); dfl = dfl.assign(unit=key(dfl))
        r = ref.set_index("unit")[descs].add_prefix("ref_")
        d = dfl.set_index("unit")[descs].add_prefix("def_")
        lm = sub[sub.kind != "main"].copy(); lm = lm.assign(unit=key(lm))
        lms = {k: lm[lm.kind == k].set_index("unit")[descs].add_prefix(f"{k}_") for k in lm.kind.unique()}
        out = r.join(d, how="inner")
    for k, v in lms.items():
        out = out.join(v, how="left")
    if space == "DINO":
        n = int(sub[sub.kind == "main"]["n_interior"].median())
        an = dino_lerp_analytic(n)
        for dsc in descs:
            out[f"DINOLERP_{dsc}"] = an[dsc]
    return out, descs


rows = []
CASES = [  # space-agnostic list of (stratum, tier, ref_group, def_group)
    ("S-GRID", "", "GT", "NULLGEN"),
    ("S-PROBE", "high", "R1", "R3"),
    ("S-PROBE", "high", "R2", "R3"),
    ("S-PROBE", "inplace", "R1", "R3"),
    ("S-PROBE", "inplace", "R2", "R3"),
]
for space in ("PIX", "DINO", "VAE", "TRANS"):
    for stratum, tier, refg, defg in CASES:
        ut, descs = unit_table(space, stratum, tier, refg, defg, refg, defg)
        nulls = [k for k in ("LERP", "CUT50", "FREEZE", "LATLERP", "DINOLERP") if f"{k}_{descs[0]}" in ut.columns]
        for dsc in descs:
            ref_med, def_med = ut[f"ref_{dsc}"].median(), ut[f"def_{dsc}"].median()
            for N in nulls:
                nv = ut[f"{N}_{dsc}"]
                n_med = nv.median()
                d_def = ref_med - def_med           # where the default must move
                d_null = def_med - n_med            # where "away from N" pushes
                correct = sgn(d_def) == sgn(d_null) and sgn(d_def) != 0
                brief = sgn(ref_med - n_med) == sgn(d_def) and sgn(d_def) != 0
                # paired: per unit sign agreement
                pd_def = ut[f"ref_{dsc}"] - ut[f"def_{dsc}"]; pd_null = ut[f"def_{dsc}"] - nv
                ok = (np.sign(pd_def) == np.sign(pd_null)) & (pd_def.abs() > 1e-9) & (pd_null.abs() > 1e-9)
                valid = pd_def.notna() & pd_null.notna()
                frac = float(ok[valid].mean()) if valid.any() else np.nan
                headroom = abs(d_def) / (abs(d_null) + 1e-9)
                verdict = ("negligible(|def-N|<0.02)" if abs(d_null) < 0.02 else
                           ("RIGHT" if correct else ("WRONG(N between default and ref)" if brief else "WRONG")))
                if abs(d_def) < 0.02:
                    verdict = "no-gap(|ref-def|<0.02)"
                rows.append(dict(space=space, stratum=stratum, tier=tier, ref=refg, default=defg, descriptor=dsc, null=N,
                                 n_units=int(valid.sum()), ref_med=round(ref_med, 4), def_med=round(def_med, 4),
                                 null_med=round(n_med, 4), ref_minus_def=round(d_def, 4), def_minus_null=round(d_null, 4),
                                 correct_dir=bool(correct), brief_criterion=bool(brief), paired_frac_correct=round(frac, 3),
                                 headroom_ratio=round(headroom, 3), verdict=verdict))
tab = pd.DataFrame(rows)
tab.to_csv(INV / "direction_table.csv", index=False)

# compact markdown: S-GRID (GT ref) and S-PROBE high (R1 ref) side by side for LERP / LATLERP / DINOLERP
def compact(stratum, tier, refg):
    t = tab[(tab.stratum == stratum) & (tab.tier == tier) & (tab.ref == refg) & (tab.null.isin(["LERP", "LATLERP", "DINOLERP", "CUT50"]))]
    lines = [f"### {stratum} {tier} (reference = {refg}, default = {t.default.iloc[0]})",
             "| space | descriptor | ref | default | LERP | verdict(LERP) | paired | LATLERP/DINOLERP | verdict | CUT50 verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (sp, d), g in t.groupby(["space", "descriptor"], sort=False):
        L = g[g.null == "LERP"].iloc[0]
        alt = g[g.null.isin(["LATLERP", "DINOLERP"])]
        C = g[g.null == "CUT50"]
        alt_s = f"{alt.iloc[0].null_med:.3f} ({alt.iloc[0].null})" if len(alt) else "—"
        alt_v = alt.iloc[0].verdict if len(alt) else "—"
        c_v = C.iloc[0].verdict if len(C) else "—"
        lines.append(f"| {sp} | {d} | {L.ref_med:.3f} | {L.def_med:.3f} | {L.null_med:.3f} | {L.verdict} | {L.paired_frac_correct:.2f} | {alt_s} | {alt_v} | {c_v} |")
    return "\n".join(lines)

md = "\n\n".join([compact("S-GRID", "", "GT"), compact("S-PROBE", "high", "R1"), compact("S-PROBE", "inplace", "R1")])
(INV / "direction_table.md").write_text(md + "\n")
print(md)

# ------------------------------------------------------------------ empirical check: DCG (mechanism a) on dualforce_control
rm = pd.read_csv(REPO / "misc/2026-09-08_collapse_remeasure/per_clip.csv", keep_default_na=False, low_memory=False)
for c in ("DR_med", "M", "PR", "seed", "static", "foreign", "two_sided", "error"):
    if c in rm.columns:
        rm[c] = pd.to_numeric(rm[c], errors="coerce") if c in ("DR_med", "M", "PR", "seed") else rm[c]
def clean(x):
    x = x[(x.error.astype(str) == "") | (x.error.astype(str) == "nan")]
    x = x[(x.static.astype(str).str.lower() != "true") & (x.foreign.astype(str).str.lower() != "true")]
    return x
ctrl = clean(rm[(rm.arm == "dualforce_control") & (rm.variant == "01_neutral__dai") & (rm.two_sided.astype(str).str.lower() == "true") & (rm.seed == 42)])
ctrl = ctrl.drop_duplicates("md5").set_index("endpoint")
lines = ["### Empirical check — DCG (null = pixel crossfade of the demo's endpoints, mechanism (a)) vs dualforce_control, neutral, both anchors, seed 42, paired by endpoint",
         "| arm | n pairs | ΔDR med (dcg−ctrl) | n_pos/n_neg | ΔM med | n_pos/n_neg | ΔPR med | ctrl DR / M med | dcg DR / M med | GT PIX DR / M (S-GRID, 19) |", "|---|---|---|---|---|---|---|---|---|---|"]
gt = pc[(pc.space == "PIX") & (pc.group == "GT") & (pc.kind == "main")]
for arm in ("dualforce_dcg_w1", "dualforce_dcg_w1p5", "dualforce_dcg_w3", "dualforce_dcg_w6"):
    d = clean(rm[(rm.arm == arm) & (rm.variant == "01_neutral__dai") & (rm.two_sided.astype(str).str.lower() == "true") & (rm.seed == 42)])
    d = d.drop_duplicates("md5").set_index("endpoint")
    j = ctrl[["DR_med", "M", "PR"]].join(d[["DR_med", "M", "PR"]], how="inner", rsuffix="_dcg")
    dDR = j.DR_med_dcg - j.DR_med; dM = j.M_dcg - j.M; dPR = j.PR_dcg - j.PR
    lines.append(f"| {arm} | {len(j)} | {dDR.median():+.3f} | {(dDR>0).sum()}/{(dDR<0).sum()} | {dM.median():+.3f} | {(dM>0).sum()}/{(dM<0).sum()} | {dPR.median():+.2f} | {j.DR_med.median():.3f} / {j.M.median():.3f} | {j.DR_med_dcg.median():.3f} / {j.M_dcg.median():.3f} | {gt.DR.median():.3f} / {gt.M.median():.3f} |")
emd = "\n".join(lines)
(INV / "dcg_empirical_check.md").write_text(emd + "\n")
print("\n" + emd)
