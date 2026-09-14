"""Extra numbers for FINDINGS: swap duration (frames with tau in (0.2,0.8)), aligned-curve values, TRANS values,
per-clip spread, trained control vs GT on the 19 shared endpoints (PIX, Sep-08 re-measure)."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, pandas as pd
from pathlib import Path
LAB = Path("/taiga/illinois/eng/cs/jrehg/users/emirkisa"); REPO = LAB / "diffusion-research"
CAMP = REPO / "misc/2026-09-13_null_default"; INV = CAMP / "investigation"; CACHE = INV / "cache"
K = 48; SG = np.linspace(0, 1, K)
rec = pd.read_csv(INV / "curves_per_clip.csv", keep_default_na=False)
for c in ("s_cross", "jump1", "jump3", "step_share", "DR", "M", "nu_max", "path_over_gap"):
    rec[c] = pd.to_numeric(rec[c], errors="coerce")
out = ["## Extra numbers", ""]
# transit duration in frames: fraction of interior s-grid with tau in (0.2,0.8) x n_interior (PIX/DINO n=104; VAE 13 latents)
out += ["### Swap duration: interior time with τ ∈ (0.2, 0.8), as a fraction of the interior (PIX/DINO: ×104 frames; VAE ×13 latents); per-clip spread = std of τ(s=0.5)", "",
        "| space | group | n | transit frac med [IQR] | ≈ frames | share with transit ≤ 10% of interior | std τ(0.5) across clips | std ρ(0.5) |", "|---|---|---|---|---|---|---|---|"]
for sp in ("PIX", "DINO", "VAE"):
    z = np.load(CACHE / f"curves_{sp}.npz", allow_pickle=True); keys = list(z["keys"]); idx = {k: i for i, k in enumerate(keys)}
    tau = z["tau"]; rho = z["rho"]
    r = rec[(rec.space == sp) & (rec.kind == "main")]
    n_int = 104 if sp != "VAE" else 13
    for g in ["S-GRID:NULLGEN", "S-GRID:GT", "S-GRID-F:NULLGEN", "S-PROBE:R3:high", "S-PROBE:R1:high", "S-PROBE:R2:high", "S-PROBE:R3:inplace", "S-SWEEP:A_empty"]:
        s = r[r.glabel == g]
        if len(s) == 0:
            continue
        ii = [idx[k] for k in s.key]; T = tau[ii]
        transit = ((T > 0.2) & (T < 0.8)).mean(1)
        out.append(f"| {sp} | {g} | {len(s)} | {np.median(transit):.2f} [{np.quantile(transit,.25):.2f}, {np.quantile(transit,.75):.2f}] | {np.median(transit)*n_int:.0f} | {(transit <= 0.10).mean():.2f} | {T[:, K//2].std():.2f} | {rho[ii][:, K//2].std():.2f} |")
    # landmarks
    for kind in ("LERP", "CUT50", "LATLERP"):
        s = rec[(rec.space == sp) & (rec.kind == kind) & (rec.glabel == "S-GRID:LM")]
        if len(s) == 0:
            continue
        ii = [idx[k] for k in s.key]; T = tau[ii]; transit = ((T > 0.2) & (T < 0.8)).mean(1)
        out.append(f"| {sp} | LM:{kind} | {len(s)} | {np.median(transit):.2f} | {np.median(transit)*n_int:.0f} | {(transit <= 0.10).mean():.2f} | {T[:, K//2].std():.2f} | {rho[ii][:, K//2].std():.2f} |")
# aligned curve values
ms = pd.read_csv(INV / "curves_mean_std.csv")
out += ["", "### Crossing-aligned mean curves (τ=0.5 at s=0.5): τ at aligned s = 0.3 / 0.4 / 0.6 / 0.7; ρ at aligned s = 0.5; ν at aligned 0.5", "",
        "| space | group | τ_al(0.3/0.4/0.6/0.7) | ρ_al(0.5) | ν_al(0.5) | n aligned |", "|---|---|---|---|---|---|"]
def val(sp, g, curve, s):
    m = ms[(ms.space == sp) & (ms.group == g) & (ms.curve == curve)]
    if len(m) == 0:
        return np.nan, 0
    j = int(round(s * (K - 1))); row = m.iloc[j]
    return row["mean"], int(row["n"])
for sp in ("PIX", "DINO", "VAE"):
    for g in ["S-GRID:NULLGEN", "S-GRID:GT", "S-PROBE:R3:high", "S-PROBE:R1:high", "S-GRID:LM:LERP", "S-GRID:LM:CUT50", "S-GRID:LM:LATLERP"]:
        v = [val(sp, g, "tau_al", s)[0] for s in (.3, .4, .6, .7)]
        if np.isnan(v[0]):
            continue
        rr, n = val(sp, g, "rho_al", .5); nn, _ = val(sp, g, "nu_al", .5)
        out.append(f"| {sp} | {g} | {' / '.join(f'{x:.2f}' for x in v)} | {rr:.2f} | {nn:.2f} | {n} |")
# TRANS values
mt = pd.read_csv(INV / "curves_mean_std_TRANS.csv"); KT = 25
out += ["", "### TRANS mean curves at s = 0.25 / 0.5 / 0.75 of the latent window", "", "| group | swap progress | nu_t | local_t | inplace share/step | n |", "|---|---|---|---|---|---|"]
for g in ["S-GRID:NULLGEN", "S-GRID:GT", "S-PROBE:R3:high", "S-PROBE:R1:high", "S-PROBE:R3:inplace", "S-GRID:LM:LERP", "S-GRID:LM:CUT50"]:
    def tv(curve):
        m = mt[(mt.group == g) & (mt.curve == curve)]
        if len(m) == 0:
            return "—", 0
        return " / ".join(f"{m.iloc[int(round(s*(KT-1)))]['mean']:.2f}" for s in (.25, .5, .75)), int(m.iloc[0]["n"])
    a, n = tv("swapp"); b, _ = tv("nu"); c, _ = tv("local"); d, _ = tv("inplace")
    out.append(f"| {g} | {a} | {b} | {c} | {d} | {n} |")
# trained control vs GT on the 19 S-GRID endpoints (PIX, re-measure)
rm = pd.read_csv(REPO / "misc/2026-09-08_collapse_remeasure/per_clip.csv", keep_default_na=False, low_memory=False)
for c in ("DR_med", "M", "PR", "seed"):
    rm[c] = pd.to_numeric(rm[c], errors="coerce")
pc = pd.read_csv(CAMP / "results/per_clip.csv", keep_default_na=False, low_memory=False)
for c in ("DR", "M", "path_over_gap"):
    pc[c] = pd.to_numeric(pc[c], errors="coerce")
gt = pc[(pc.space == "PIX") & (pc.group == "GT") & (pc.kind == "main")].set_index("endpoint")[["DR", "M", "path_over_gap"]]
nul = pc[(pc.space == "PIX") & (pc.group == "NULLGEN") & (pc.stratum == "S-GRID") & (pc.kind == "main")].groupby("endpoint")[["DR", "M", "path_over_gap"]].median()
def arm_rows(arm, variants):
    x = rm[(rm.arm == arm) & (rm.variant.isin(variants)) & (rm.two_sided.astype(str).str.lower() == "true") & (rm.foreign.astype(str).str.lower() != "true") & ((rm.error.astype(str) == "") | (rm.error.astype(str) == "nan"))]
    x = x.drop_duplicates("md5")
    return x.groupby("endpoint")[["DR_med", "M", "PR"]].median()
out += ["", "### PIX, the 19 S-GRID endpoints with GT twins: GT vs base default vs trained control vs DCG w6 (Sep-08 re-measure rows, both anchors, neutral, non-foreign, md5-dedup, medians over seeds then endpoints)", "",
        "| arm | n endpoints | DR med | M med | path/gap or PR med | paired ΔDR vs GT med (n_pos/n_neg) | paired ΔM vs GT med |", "|---|---|---|---|---|---|---|"]
out.append(f"| GT twin | {len(gt)} | {gt.DR.median():.3f} | {gt.M.median():.3f} | {gt.path_over_gap.median():.2f} | — | — |")
j = nul.join(gt, rsuffix="_gt", how="inner")
out.append(f"| base_cond neutral (NULLGEN) | {len(j)} | {j.DR.median():.3f} | {j.M.median():.3f} | {j.path_over_gap.median():.2f} | {(j.DR-j.DR_gt).median():+.3f} ({((j.DR-j.DR_gt)>0).sum()}/{((j.DR-j.DR_gt)<0).sum()}) | {(j.M-j.M_gt).median():+.3f} |")
for arm, variants in (("dualforce_control", ["01_neutral__dai", "03_neutral_v3__dai"]), ("dualforce_dcg_w1", ["01_neutral__dai"]), ("dualforce_dcg_w1p5", ["01_neutral__dai"]), ("dualforce_dcg_w3", ["01_neutral__dai"]), ("dualforce_dcg_w6", ["01_neutral__dai", "03_neutral_v3__dai"])):
    a = arm_rows(arm, variants); j = a.join(gt, how="inner", rsuffix="_gt")
    if len(j) == 0:
        continue
    out.append(f"| {arm} | {len(j)} | {j.DR_med.median():.3f} | {j.M.median():.3f} | {j.PR.median():.2f} | {(j.DR_med-j.DR).median():+.3f} ({((j.DR_med-j.DR)>0).sum()}/{((j.DR_med-j.DR)<0).sum()}) | {(j.M-j.M_gt).median():+.3f} |")
(INV / "extra_numbers.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
