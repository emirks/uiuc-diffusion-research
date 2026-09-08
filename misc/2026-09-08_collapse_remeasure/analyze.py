"""Analysis of per_clip.csv + endpoint_covariates.csv -> tables (TABLES.md, summary.json) + figures.

Principles (owner, 2026-09-08): continuous statistics first (median/IQR/ECDF of the confinement residual,
mid-band coverage), the on-line share only as a description at the stated cut point with a sensitivity
curve; unit = endpoint pair (cluster bootstrap); duplicates removed by md5; paired contrasts wherever the
same (endpoint, seed) or (item, seed) exists in both conditions.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from instrument import THETA

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(REPO)
rng = np.random.default_rng(0)
NBOOT = 2000
OUT = {}
MD = []


def md(s=""):
    MD.append(s)


def fmt(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


# ------------------------------------------------------------------ load + dedupe
df = pd.read_csv(f"{HERE}/per_clip.csv")
cov = pd.read_csv(f"{HERE}/endpoint_covariates.csv") if os.path.exists(f"{HERE}/endpoint_covariates.csv") else None
df = df[df.get("error").isna()] if "error" in df else df
df["seed"] = df["seed"].astype(str)
# grid-row key shared across arms: item_id = <cell>__<arm stamp>__<endpoint>__ref_<reference>; drop the arm stamp
import re as _re
def _row_key(s):
    s = _re.sub(r"__dfw[0-9p]+(_e)?$", "", s)         # v2 dualforce_dcg items carry a trailing __dfw6 / __dfw1p5 / __dfw6_e token
    p = s.split("__")
    return "__".join([p[0]] + p[2:]) if len(p) >= 3 else s
df["row_key"] = df["item"].map(_row_key)
# duplicates: identical files inside one (arm, variant) (base model ignores the reference -> same video)
df["dup_rank"] = df.groupby(["arm", "variant", "md5"]).cumcount()
df["is_dup"] = df["dup_rank"] > 0
# cross-variant duplicates (grid v3 hardlinks 139 v2 rows): for POOLED analyses keep one copy per (arm, md5)
df["is_dup_any"] = df.groupby(["arm", "md5"]).cumcount() > 0
df["clean"] = (~df["foreign"].astype(bool)) & (~df["static"].astype(bool)) & df["DR_med"].notna()
u = df[(~df["is_dup"]) & df["clean"]].copy()          # analysis frame: unique within variant, clean
u2 = df[(~df["is_dup_any"]) & df["clean"]].copy()     # pooled-analysis frame: unique across variants, clean
T2 = "tier2_start__dai"
u_lvl = u[u["variant"] != T2]                          # level tables: grid rows only (the tier-2 regen enters the paired probe only)
OUT["n_rows"] = int(len(df)); OUT["n_unique_clean"] = int(len(u))
OUT["n_dup"] = int(df["is_dup"].sum()); OUT["n_foreign"] = int(df["foreign"].sum()); OUT["n_static"] = int(df["static"].sum())

md("# Collapse re-measurement on existing store generations (2026-09-08)")
md(f"\nRows scored: {len(df)} · byte-identical duplicates: {OUT['n_dup']} · foreign/davis: {OUT['n_foreign']} · "
   f"static (gap too small): {OUT['n_static']} · **unique clean clips analysed: {len(u)}**")
md(f"\nConfinement residual DR = median normalised off-endpoint-line residual of interior frames (0 = on the line). "
   f"On-line = DR ≤ {THETA} (descriptive cut point, kept from the Aug-24 campaign; sensitivity below). "
   f"M = mid-band coverage of the projection coordinate (dissolve 0.5, cut/freeze 0). "
   f"CIs are 95% cluster bootstraps over endpoint pairs.")


def cboot(frame, stat, n=NBOOT):
    """Cluster bootstrap over endpoint: resample endpoints with replacement, recompute stat(frame).
    Array-based (row-index lists per endpoint + one iloc per replicate) instead of pandas concat."""
    frame = frame.reset_index(drop=True)
    _, inv = np.unique(frame["endpoint"].astype(str).values, return_inverse=True)
    k = inv.max() + 1
    idx_by_ep = [np.flatnonzero(inv == j) for j in range(k)]
    vals = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, k, k)
        idx = np.concatenate([idx_by_ep[j] for j in pick])
        vals[i] = stat(frame.iloc[idx])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def share_online(f):
    return float(f["online"].mean()) if len(f) else np.nan


def med(col):
    return lambda f: float(f[col].median()) if len(f) else np.nan


# ------------------------------------------------------------------ A. level table
md("\n## A. Levels per arm × grid × prompt × conditioning (unique clean clips)\n")
md("| arm | grid | prompt | cond | n clips | n static (excl.) | n endpoints | DR median [IQR] | DR CI | M median | on-line share [CI] | on-line n | DISS/CUT/FRZ |")
md("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
levels = []
for (arm, grid, prompt, cond), g in u_lvl.groupby(["arm", "grid", "prompt", "condition"]):
    ns = int(((df["arm"] == arm) & (df["grid"] == grid) & (df["prompt"] == prompt) & (df["condition"] == cond)
              & (~df["is_dup"]) & (~df["foreign"].astype(bool)) & df["static"].astype(bool) & (df["variant"] != T2)).sum())
    q = g["DR_med"].quantile([.25, .5, .75]).values
    lo, hi = cboot(g, med("DR_med"))
    so = share_online(g); slo, shi = cboot(g, share_online)
    onl = g[g["online"]]
    dist = f"{(onl['cls']=='DISSOLVE').sum()}/{(onl['cls']=='CUT').sum()}/{(onl['cls']=='FREEZE').sum()}"
    row = dict(arm=arm, grid=grid, prompt=prompt, condition=cond, n=len(g), n_endpoints=g["endpoint"].nunique(),
               DR_med=q[1], DR_q25=q[0], DR_q75=q[2], DR_ci=[lo, hi], M_med=float(g["M"].median()),
               online_share=so, online_ci=[slo, shi], online_n=int(g["online"].sum()), online_frac_med=float(g["online_frac"].median()),
               dist=dist)
    levels.append(row)
    row["n_static"] = ns
    md(f"| {arm} | {grid} | {prompt} | {cond} | {len(g)} | {ns} | {row['n_endpoints']} | {fmt(q[1])} [{fmt(q[0])}, {fmt(q[2])}] | "
       f"[{fmt(lo)}, {fmt(hi)}] | {fmt(row['M_med'],2)} | {so*100:.1f}% [{slo*100:.1f}, {shi*100:.1f}] | {row['online_n']} | {dist} |")
OUT["levels"] = levels

# ------------------------------------------------------------------ B. sensitivity (base_cond)
md("\n## B. Cut-point sensitivity, base_cond (share of unique clean clips with DR ≤ θ)\n")
ths = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25]
md("| grid | prompt | cond | n | " + " | ".join(f"θ={t}" for t in ths) + " |")
md("|---|---|---|---|" + "---|" * len(ths))
sens = []
for (grid, prompt, cond), g in u_lvl[u_lvl["arm"] == "base_cond"].groupby(["grid", "prompt", "condition"]):
    vals = [float((g["DR_med"] <= t).mean()) for t in ths]
    sens.append(dict(grid=grid, prompt=prompt, condition=cond, n=len(g), thetas=ths, shares=vals))
    md(f"| {grid} | {prompt} | {cond} | {len(g)} | " + " | ".join(f"{v*100:.0f}%" for v in vals) + " |")
OUT["sensitivity"] = sens


# ------------------------------------------------------------------ C. paired contrasts
def paired(a, b, keys, label, note=""):
    """a, b: frames; join on keys; delta = a - b (DR). Returns dict + markdown row."""
    j = a.merge(b, on=keys, suffixes=("_a", "_b"))
    j = j[j["DR_med_a"].notna() & j["DR_med_b"].notna()]
    if len(j) < 5:
        md(f"| {label} | {len(j)} | – | – | – | – | – | too few pairs |")
        return None
    d = (j["DR_med_a"] - j["DR_med_b"]).values
    w = stats.wilcoxon(d, alternative="two-sided", zero_method="zsplit") if np.any(d != 0) else None
    cliff = float(np.mean(np.sign(d)))
    fa, fb = j["online_a"].astype(bool), j["online_b"].astype(bool)
    flips_a = int((fa & ~fb).sum()); flips_b = int((~fa & fb).sum())
    p_sign = stats.binomtest(flips_a, flips_a + flips_b, 0.5).pvalue if (flips_a + flips_b) else np.nan
    j["endpoint"] = j["endpoint_a"] if "endpoint_a" in j else j["endpoint"]
    lo, hi = cboot(j.assign(delta=d), lambda f: float(np.median(f["delta"])))
    res = dict(label=label, n_pairs=len(j), n_endpoints=int(j["endpoint"].nunique()), delta_median=float(np.median(d)),
               delta_ci=[lo, hi], delta_mean=float(d.mean()), cliff=cliff, wilcoxon_p=(float(w.pvalue) if w else np.nan),
               online_a=float(fa.mean()), online_b=float(fb.mean()), flips_a_only=flips_a, flips_b_only=flips_b,
               sign_p=float(p_sign), note=note)
    md(f"| {label} | {len(j)} ({res['n_endpoints']} ep) | {fmt(res['delta_median'])} [{fmt(lo)}, {fmt(hi)}] | {fmt(cliff,2)} | "
       f"{fmt(res['wilcoxon_p'],4)} | {fa.mean()*100:.1f}% → {fb.mean()*100:.1f}% | {flips_a}/{flips_b} (p={fmt(p_sign,3)}) | {note} |")
    return res


md("\n## C. Paired contrasts (same endpoint and seed; unique clean clips)\n")
md("ΔDR = first − second (positive = first is further from the endpoint line). Cliff's δ = mean sign(Δ). "
   "Flips = on-line in first only / on-line in second only, with exact sign test.\n")
md("| contrast | pairs | ΔDR median [CI] | Cliff δ | Wilcoxon p | on-line share first → second | flips (p) | note |")
md("|---|---|---|---|---|---|---|---|")
pairs = {}
bc = u[u["arm"] == "base_cond"]
# C1 end anchor: v2 neutral two-sided vs tier-2 start-only regen (same rows, same seeds)
a = bc[(bc["variant"] == "02_neutral__dai") & (bc["condition"] == "both")].drop_duplicates(["endpoint", "seed"])
b = bc[(bc["variant"] == "tier2_start__dai")].drop_duplicates(["endpoint", "seed"])
pairs["end_anchor_v2_neutral"] = paired(a, b, ["endpoint", "seed"], "END ANCHOR: base both − base start-only (v2 neutral, tier-2 regen)",
                                        "causal probe; anchor is the only difference")
# C2 prompt: neutral vs effect, same endpoint/seed/condition, per grid (effect rows carry different clauses -> several per endpoint)
for grid in ["v2", "v3", "v3ed81"]:
    for cond in ["both", "start"]:
        n_ = bc[(bc["grid"] == grid) & (bc["prompt"] == "neutral") & (bc["condition"] == cond) & (bc["variant"] != "tier2_start__dai")].drop_duplicates(["endpoint", "seed"])
        e_ = bc[(bc["grid"] == grid) & (bc["prompt"] == "effect") & (bc["condition"] == cond)]
        if len(n_) and len(e_):
            pairs[f"prompt_{grid}_{cond}"] = paired(e_, n_, ["endpoint", "seed"], f"PROMPT: base effect − base neutral ({grid}, {cond})",
                                                   "text describes the transition vs neutral")
# C3 training: base_cond vs dualforce_control, same item & seed (same grid row)
for grid in ["v2", "v3", "v3ed81"]:
    for prompt in ["neutral", "effect"]:
        for cond in ["both", "start"]:
            a = u[(u["arm"] == "base_cond") & (u["grid"] == grid) & (u["prompt"] == prompt) & (u["condition"] == cond) & (u["variant"] != "tier2_start__dai")]
            b = u[(u["arm"] == "dualforce_control") & (u["grid"] == grid) & (u["prompt"] == prompt) & (u["condition"] == cond)]
            # base duplicates were dropped -> re-expand: join on (endpoint, seed) for base, item for trained
            a2 = df[(df["arm"] == "base_cond") & (df["grid"] == grid) & (df["prompt"] == prompt) & (df["condition"] == cond) & (df["variant"] != "tier2_start__dai") & df["clean"]]
            if len(a2) and len(b):
                pairs[f"train_{grid}_{prompt}_{cond}"] = paired(a2, b, ["row_key", "seed"], f"TRAINING: base − dualforce_control ({grid}, {prompt}, {cond})",
                                                               "same grid row; base duplicates kept so every trained clip has its partner")
# C4 guidance: dualforce_dcg_w* − dualforce_control, same item & seed
for arm_w, wlab in [("dualforce_dcg_w1", "w=1"), ("dualforce_dcg_w1p5", "w=1.5"), ("dualforce_dcg_w3", "w=3"), ("dualforce_dcg_w6", "w=6")]:
    for grid in ["v2", "v3", "v3ed81"]:
        for prompt in ["neutral", "effect"]:
            for cond in ["both", "start"]:
                a = u[(u["arm"] == arm_w) & (u["grid"] == grid) & (u["prompt"] == prompt) & (u["condition"] == cond)]
                b = u[(u["arm"] == "dualforce_control") & (u["grid"] == grid) & (u["prompt"] == prompt) & (u["condition"] == cond)]
                if len(a) and len(b):
                    pairs[f"guid_{wlab}_{grid}_{prompt}_{cond}"] = paired(a, b, ["row_key", "seed"], f"GUIDANCE {wlab}: dcg − control ({grid}, {prompt}, {cond})", "")
OUT["paired"] = {k: v for k, v in pairs.items() if v}

# ------------------------------------------------------------------ D. unpaired both vs start (levels)
md("\n## D. Unpaired both-endpoint vs start-only within arm × grid × prompt (different endpoints per stratum)\n")
md("| arm | grid | prompt | n both / start | DR median both / start | Δmedian [CI] | Cliff δ (MWU) | on-line both / start | Δshare [CI] |")
md("|---|---|---|---|---|---|---|---|---|")
unp = []
for (arm, grid, prompt), g in u_lvl.groupby(["arm", "grid", "prompt"]):
    gb, gs = g[g["condition"] == "both"], g[g["condition"] == "start"]
    if len(gb) < 5 or len(gs) < 5:
        continue
    x, y = gb["DR_med"].values, gs["DR_med"].values
    mwu = stats.mannwhitneyu(x, y, alternative="two-sided")
    cliff = float(2 * mwu.statistic / (len(x) * len(y)) - 1)
    def dmed(f):
        return float(f[f["condition"] == "both"]["DR_med"].median() - f[f["condition"] == "start"]["DR_med"].median())
    def dshare(f):
        return float(f[f["condition"] == "both"]["online"].mean() - f[f["condition"] == "start"]["online"].mean())
    lo, hi = cboot(g, dmed); slo, shi = cboot(g, dshare)
    r = dict(arm=arm, grid=grid, prompt=prompt, n_both=len(gb), n_start=len(gs), DR_both=float(np.median(x)), DR_start=float(np.median(y)),
             dmed=dmed(g), dmed_ci=[lo, hi], cliff=cliff, mwu_p=float(mwu.pvalue), online_both=share_online(gb), online_start=share_online(gs),
             dshare=dshare(g), dshare_ci=[slo, shi])
    unp.append(r)
    md(f"| {arm} | {grid} | {prompt} | {len(gb)} / {len(gs)} | {fmt(r['DR_both'])} / {fmt(r['DR_start'])} | {fmt(r['dmed'])} [{fmt(lo)}, {fmt(hi)}] | "
       f"{fmt(cliff,2)} (p={fmt(r['mwu_p'],4)}) | {r['online_both']*100:.1f}% / {r['online_start']*100:.1f}% | {r['dshare']*100:+.1f} pp [{slo*100:.1f}, {shi*100:.1f}] |")
OUT["unpaired"] = unp

# ------------------------------------------------------------------ E. per-endpoint propensity (base_cond both)
md("\n## E. Per-endpoint collapse propensity, base_cond both-endpoint (pooling seeds and both prompts)\n")
md("| grid | endpoints | ≥1 on-line gen | all gens on-line | median per-endpoint on-line share |")
md("|---|---|---|---|---|")
prop = []
for grid, g in bc[bc["condition"] == "both"].groupby("grid"):
    pe = g.groupby("endpoint")["online"].agg(["mean", "size"])
    r = dict(grid=grid, n_endpoints=len(pe), any=float((pe["mean"] > 0).mean()), all=float((pe["mean"] == 1).mean()), med_share=float(pe["mean"].median()),
             per_endpoint=pe["mean"].round(3).to_dict())
    prop.append(r)
    md(f"| {grid} (both prompts) | {len(pe)} | {r['any']*100:.0f}% | {r['all']*100:.0f}% | {r['med_share']*100:.0f}% |")
for grid, g in bc[(bc["condition"] == "both") & (bc["prompt"] == "neutral") & (bc["variant"] != "tier2_start__dai")].groupby("grid"):
    pe = g.groupby("endpoint")["online"].agg(["mean", "size"])
    r = dict(grid=grid + "_neutral", n_endpoints=len(pe), any=float((pe["mean"] > 0).mean()), all=float((pe["mean"] == 1).mean()),
             med_share=float(pe["mean"].median()), per_endpoint=pe["mean"].round(3).to_dict())
    prop.append(r)
    md(f"| {grid} (neutral only, 2 seeds) | {len(pe)} | {r['any']*100:.0f}% | {r['all']*100:.0f}% | {r['med_share']*100:.0f}% |")
OUT["propensity"] = prop

# ------------------------------------------------------------------ F. covariates
if cov is not None:
    md("\n## F. Endpoint covariates vs confinement (base_cond, unique clean clips)\n")
    m = bc.merge(cov, on="endpoint", how="left", suffixes=("", "_cov"))
    md("Spearman ρ of DR with each covariate (per grid × prompt × cond); DINO/CLIP distances are between the two anchor frames "
       "(GT clip frames where the real transition exists, else the conditioned frames of the generation).\n")
    md("| grid | prompt | cond | n (with pair cov) | ρ DINO dist | ρ CLIP dist | ρ pixel gap | ρ motion prefix | ρ motion suffix | on-line share by DINO tercile (low/mid/high) |")
    md("|---|---|---|---|---|---|---|---|---|---|")
    covres = []
    for (grid, prompt, cond), g in m.groupby(["grid", "prompt", "condition"]):
        gg = g[g["dino_dist"].notna()]
        if len(gg) < 8:
            continue
        def rho(col):
            s = gg[[col, "DR_med"]].dropna()
            return (float(stats.spearmanr(s[col], s["DR_med"]).statistic), float(stats.spearmanr(s[col], s["DR_med"]).pvalue), len(s)) if len(s) >= 8 else (np.nan, np.nan, len(s))
        r = dict(grid=grid, prompt=prompt, condition=cond, n=len(gg))
        for col in ["dino_dist", "clip_dist", "pixel_gap_rel", "motion_prefix", "motion_suffix"]:
            r[col] = rho(col)
        try:
            gg = gg.assign(terc=pd.qcut(gg["dino_dist"], 3, labels=["low", "mid", "high"], duplicates="drop"))
            terc = gg.groupby("terc", observed=True)["online"].mean()
            r["online_by_dino_tercile"] = {str(k): float(v) for k, v in terc.items()}
            tstr = "/".join(f"{v*100:.0f}%" for v in terc.values)
        except Exception:
            tstr = "–"
        covres.append(r)
        cell = lambda t: f"{fmt(t[0],2)} (p={fmt(t[1],3)})" if np.isfinite(t[0]) else "–"
        md(f"| {grid} | {prompt} | {cond} | {len(gg)} | {cell(r['dino_dist'])} | {cell(r['clip_dist'])} | {cell(r['pixel_gap_rel'])} | "
           f"{cell(r['motion_prefix'])} | {cell(r['motion_suffix'])} | {tstr} |")
    OUT["covariates"] = covres

    # cluster-bootstrap OLS on the POOLED frame (unique across grids, HF only, base_cond)
    m2 = u2[(u2["arm"] == "base_cond") & (u2["variant"] != "tier2_start__dai") & u2["grid"].isin(["v2", "v3"])].merge(cov, on="endpoint", how="left", suffixes=("", "_cov"))

    def run_ols(frame, cols, names, title):
        f0 = frame.dropna(subset=cols + ["DR_med"]).reset_index(drop=True)
        if len(f0) < 25:
            md(f"\n{title}: too few rows ({len(f0)}).\n"); return None
        def ols(f):
            X = np.column_stack([np.ones(len(f))] + [f[c].values.astype(float) for c in cols])
            beta, *_ = np.linalg.lstsq(X, f["DR_med"].values, rcond=None)
            return beta
        beta = ols(f0)
        _, inv = np.unique(f0["endpoint"].astype(str).values, return_inverse=True); k = inv.max() + 1
        idx_by_ep = [np.flatnonzero(inv == j) for j in range(k)]
        B = np.array([ols(f0.iloc[np.concatenate([idx_by_ep[j] for j in rng.integers(0, k, k)])]) for _ in range(1000)])
        md(f"\n{title} (n={len(f0)} clips, {k} endpoints; cluster-bootstrap 95% CIs over endpoints)\n")
        md("| term | coef | 95% CI |"); md("|---|---|---|")
        reg = {}
        for i, nme in enumerate(["intercept"] + names):
            lo, hi = np.percentile(B[:, i], [2.5, 97.5]); reg[nme] = [float(beta[i]), float(lo), float(hi)]
            md(f"| {nme} | {beta[i]:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
        return dict(n=len(f0), n_endpoints=int(k), coef=reg)

    m2["is_both"] = (m2["condition"] == "both").astype(float)
    m2["is_effect"] = (m2["prompt"] == "effect").astype(float)
    m2["both_x_neutral"] = m2["is_both"] * (1 - m2["is_effect"])
    OUT["regression"] = {}
    OUT["regression"]["neutral_both_vs_start"] = run_ols(
        m2[m2["prompt"] == "neutral"], ["is_both", "dino_dist", "motion_prefix"], ["both-endpoint", "DINO dist", "motion_prefix"],
        "**Regression R1 — neutral prompt, both vs start:** DR = b0 + b1·[both] + b2·DINO + b3·motion_prefix")
    OUT["regression"]["all_with_interaction"] = run_ols(
        m2, ["is_both", "is_effect", "both_x_neutral", "dino_dist", "motion_prefix"],
        ["both-endpoint", "effect prompt", "both × neutral", "DINO dist", "motion_prefix"],
        "**Regression R2 — both prompts with interaction:** DR = b0 + b1·[both] + b2·[effect] + b3·[both×neutral] + b4·DINO + b5·motion_prefix")
    OUT["regression"]["both_neutral_stratum"] = run_ols(
        m2[(m2["prompt"] == "neutral") & (m2["condition"] == "both")], ["dino_dist", "motion_prefix", "motion_suffix"],
        ["DINO dist", "motion_prefix", "motion_suffix"],
        "**Regression R3 — within base both-endpoint neutral:** DR = b0 + b1·DINO + b2·motion_prefix + b3·motion_suffix")

# ------------------------------------------------------------------ G. figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
cal = json.load(open("misc/2026-08-24_lerp_collapse/phase2_calibration.json"))["cal"]["synth"]
FIG = f"{HERE}/figs"; os.makedirs(FIG, exist_ok=True)

# fig1: ECDF of DR, base_cond, per grid; lines by cond × prompt
grids = [g for g in ["v2", "v3", "v3ed81"] if (bc["grid"] == g).any()]
fig, axes = plt.subplots(1, len(grids), figsize=(5 * len(grids), 4), sharey=True)
axes = np.atleast_1d(axes)
for ax, grid in zip(axes, grids):
    for (cond, prompt), g in bc[(bc["grid"] == grid) & (bc["variant"] != "tier2_start__dai")].groupby(["condition", "prompt"]):
        x = np.sort(g["DR_med"].values); ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post",
                                                  label=f"{cond} · {prompt} (n={len(x)})", lw=1.8 if cond == "both" else 1.2,
                                                  ls="-" if prompt == "neutral" else "--")
    t2 = bc[(bc["grid"] == grid) & (bc["variant"] == "tier2_start__dai")]
    if len(t2):
        x = np.sort(t2["DR_med"].values); ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color="k", lw=1, ls=":", label=f"start-only regen of the both rows (n={len(x)})")
    ax.axvline(cal["dissolve"]["DR_p95"], color="grey", lw=0.8); ax.axvline(THETA, color="red", lw=0.8, ls="--")
    if cov is not None and cov["gt_DR_med"].notna().any():
        ax.axvspan(cov["gt_DR_med"].quantile(.05), cov["gt_DR_med"].quantile(.95), color="green", alpha=0.08, label="real transitions p5–p95" if ax is axes[0] else None)
    ax.set_title(f"base LTX-2, grid {grid}"); ax.set_xlabel("confinement residual DR (0 = on the endpoint line)"); ax.set_xlim(0, 0.8); ax.grid(alpha=.3)
    ax.legend(fontsize=7)
axes[0].set_ylabel("ECDF")
plt.tight_layout(); plt.savefig(f"{FIG}/fig1_ecdf_base_cond.png", dpi=150); plt.close()

# fig2: DR x M scatter: base both vs start (HF grids), trained control both; synthetic anchors + GT cloud
fig, ax = plt.subplots(figsize=(7, 5.2))
hf = u2[u2["grid"].isin(["v2", "v3"])]          # pooled across grids -> cross-variant unique
for (arm, cond), g, c, mk in [(("base_cond", "both"), None, "tab:red", "o"), (("base_cond", "start"), None, "tab:orange", "^"),
                              (("dualforce_control", "both"), None, "tab:blue", "s"), (("dualforce_dcg_w6", "both"), None, "tab:purple", "D")]:
    g = hf[(hf["arm"] == arm) & (hf["condition"] == cond) & (hf["variant"] != "tier2_start__dai")]
    ax.scatter(g["DR_med"], g["M"], s=14, alpha=.45, c=c, marker=mk, label=f"{arm} · {cond} (n={len(g)})", edgecolors="none")
if cov is not None:
    gg = cov.dropna(subset=["gt_DR_med", "gt_M"]); ax.scatter(gg["gt_DR_med"], gg["gt_M"], s=10, c="green", alpha=.5, marker="x", label=f"real transitions (n={len(gg)})")
for k, c, lab in [("dissolve", "grey", "synthetic dissolve"), ("cut", "black", "synthetic cut"), ("freeze", "brown", "synthetic freeze")]:
    ax.scatter([cal[k]["DR_p95"]], [cal[k]["M_med"]], s=120, marker="*", c=c, label=lab, zorder=5)
ax.axvline(THETA, color="red", lw=0.8, ls="--"); ax.axhline(0.25, color="grey", lw=0.6, ls=":")
ax.set_xlabel("confinement residual DR"); ax.set_ylabel("mid-band coverage M (dissolve 0.5 · cut/freeze 0)"); ax.set_xlim(-0.01, 0.9); ax.set_ylim(-0.02, 0.85)
ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=.3); ax.set_title("Where in the null family does a clip sit? (HF grids, unique clean)")
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_dr_m_scatter.png", dpi=150); plt.close()

# fig3: DR vs DINO endpoint distance (base_cond), by cond, HF grids
if cov is not None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, cond in zip(axes, ["both", "start"]):
        g = m[(m["condition"] == cond) & m["dino_dist"].notna() & m["grid"].isin(["v2", "v3"]) & (m["variant"] != "tier2_start__dai")]
        for prompt, c in [("neutral", "tab:red"), ("effect", "tab:blue")]:
            gp = g[g["prompt"] == prompt]
            if len(gp) >= 8:
                rho = stats.spearmanr(gp["dino_dist"], gp["DR_med"])
                ax.scatter(gp["dino_dist"], gp["DR_med"], s=14, alpha=.5, c=c, label=f"{prompt} prompt (n={len(gp)}, ρ={rho.statistic:.2f}, p={rho.pvalue:.3f})", edgecolors="none")
        ax.axhline(THETA, color="red", lw=0.8, ls="--"); ax.set_title(f"base LTX-2 · {cond}"); ax.set_xlabel("DINOv2 distance between anchor frames (1 − cos)"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    axes[0].set_ylabel("confinement residual DR")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig3_dr_vs_dino.png", dpi=150); plt.close()

# fig4: guidance sweep on dualforce (v2 neutral, seed 42): median DR and on-line share by w
sw = u[(u["grid"] == "v2") & (u["prompt"] == "neutral") & (u["seed"] == "42") & u["arm"].isin(["dualforce_control", "dualforce_dcg_w1", "dualforce_dcg_w1p5", "dualforce_dcg_w3", "dualforce_dcg_w6"])]
if len(sw):
    wmap = {"dualforce_control": 0, "dualforce_dcg_w1": 1, "dualforce_dcg_w1p5": 1.5, "dualforce_dcg_w3": 3, "dualforce_dcg_w6": 6}
    fig, ax = plt.subplots(figsize=(6, 4))
    sweep = {}
    for cond, c in [("both", "tab:blue"), ("start", "tab:orange")]:
        g = sw[sw["condition"] == cond].groupby("arm")["DR_med"].agg(["median", "size"]).reset_index()
        g["w"] = g["arm"].map(wmap); g = g.sort_values("w")
        sweep[cond] = g[["arm", "w", "median", "size"]].to_dict("records")
        ax.plot(g["w"], g["median"], "o-", c=c, label=f"{cond} (n≈{int(g['size'].median())} per point)")
    ax.set_xticks([0, 1, 1.5, 3, 6]); ax.set_xticklabels(["no guid.", "1", "1.5", "3", "6"]); ax.set_xlabel("guidance weight w (dualforce, v2 neutral, seed 42)"); ax.set_ylabel("median DR")
    ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_title("Guidance moves the middle away from / toward the endpoint line?")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig4_guidance_sweep.png", dpi=150); plt.close()
    OUT["guidance_sweep"] = sweep

json.dump(OUT, open(f"{HERE}/summary.json", "w"), indent=1, default=float)
open(f"{HERE}/TABLES.md", "w").write("\n".join(MD) + "\n")
print("\n".join(MD))
print(f"\n[analyze] wrote summary.json, TABLES.md, figs/ -> {FIG}")
