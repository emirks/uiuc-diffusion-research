"""Step A (2/2) — aggregate the per-clip curves: mean ± std per group per space (raw s and crossing-aligned),
per-clip spread, k-means prototypes, position-free swap tail; TRANS native curves from trans_profiles.csv.
Outputs: curves_mean_std.csv, clusters.csv, swap_tail.csv, fig_traj_{PIX,DINO,VAE,TRANS}.png, fig_clusters_PIX.png,
         fig_clusters_DINO.png, step_a_summary.md
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import sys, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

LAB = Path("/taiga/illinois/eng/cs/jrehg/users/emirkisa"); REPO = LAB / "diffusion-research"
CAMP = REPO / "misc/2026-09-13_null_default"; INV = CAMP / "investigation"; CACHE = INV / "cache"
sys.path.insert(0, str(CAMP / "scripts")); import common
K = 48; SG = np.linspace(0, 1, K)
rec = pd.read_csv(INV / "curves_per_clip.csv", keep_default_na=False)
for c in ("gap", "n", "s_cross", "jump1", "jump3", "step_share", "path_over_gap", "DR", "M", "nu_max"):
    rec[c] = pd.to_numeric(rec[c], errors="coerce")

COL = {"S-GRID:NULLGEN": "#d62728", "S-GRID:GT": "#1f77b4", "S-GRID-F:NULLGEN": "#e377c2",
       "S-PROBE:R3:high": "#ff7f0e", "S-PROBE:R1:high": "#2ca02c", "S-PROBE:R2:high": "#17becf",
       "S-PROBE:R3:inplace": "#bcbd22", "S-PROBE:R1:inplace": "#8c564b",
       "S-SWEEP:A_empty": "#9467bd", "S-SWEEP:A_word": "#7f7f7f",
       "LERP": "#000000", "CUT50": "#555555", "FREEZE": "#999999", "LATLERP": "#c49c94"}
MAIN_ORDER = ["S-GRID:NULLGEN", "S-GRID:GT", "S-GRID-F:NULLGEN", "S-PROBE:R3:high", "S-PROBE:R1:high", "S-PROBE:R2:high",
              "S-PROBE:R3:inplace", "S-PROBE:R1:inplace", "S-SWEEP:A_empty", "S-SWEEP:A_word"]


def load(space):
    z = np.load(CACHE / f"curves_{space}.npz", allow_pickle=True)
    keys = list(z["keys"]); idx = {k: i for i, k in enumerate(keys)}
    arr = {n: z[n] for n in z.files if n != "keys"}
    return idx, arr


def sel_rows(space, glabel=None, kind=None, lm_stratum=None):
    r = rec[rec.space == space]
    if glabel is not None:
        r = r[r.glabel == glabel]
    if kind is not None:
        r = r[r.kind == kind]
    if lm_stratum is not None:
        r = r[r.glabel.str.startswith(lm_stratum + ":LM")]
    return r


ms_rows = []
def mean_std(space, idx, arr, rows, label, kind_label):
    ii = [idx[k] for k in rows.key]
    out = {}
    for name in ("tau", "rho", "nu", "dA", "dB", "tau_al", "rho_al", "nu_al"):
        A = arr[name][ii]
        m = np.nanmean(A, 0); s = np.nanstd(A, 0)
        out[name] = (m, s, int(np.isfinite(A[:, K // 2]).sum()))
        for j in range(K):
            ms_rows.append(dict(space=space, group=label, kind=kind_label, curve=name, s=round(SG[j], 4), mean=m[j], std=s[j], n=out[name][2]))
    return out


def fig_space(space):
    idx, arr = load(space)
    groups = [g for g in MAIN_ORDER if (rec[(rec.space == space) & (rec.glabel == g)].shape[0] > 0)]
    curves = {g: mean_std(space, idx, arr, sel_rows(space, glabel=g), g, "main") for g in groups}
    lm = {}
    for kind in ("LERP", "CUT50", "FREEZE", "LATLERP"):
        rr = rec[(rec.space == space) & (rec.kind == kind) & (rec.glabel == "S-GRID:LM")]
        if len(rr):
            lm[kind] = mean_std(space, idx, arr, rr, f"S-GRID:LM:{kind}", kind)
        rr = rec[(rec.space == space) & (rec.kind == kind) & (rec.glabel == "S-PROBE:LM:high")]
        if len(rr):
            mean_std(space, idx, arr, rr, f"S-PROBE:LM:high:{kind}", kind)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), dpi=150)
    panels = [("tau", "progress τ(s) along A→B chord"), ("rho", "off-chord deviation ρ(s) [gap units]"), ("nu", "novelty ν(s)=min(dA,dB)/gap"),
              ("tau_al", "τ, aligned at crossing (τ=0.5 → s=0.5)"), ("rho_al", "ρ, crossing-aligned"), ("nu_al", "ν, crossing-aligned")]
    show = ["S-GRID:NULLGEN", "S-GRID:GT", "S-PROBE:R3:high", "S-PROBE:R1:high", "S-PROBE:R3:inplace", "S-SWEEP:A_empty"]
    for ax, (name, title) in zip(axes.ravel(), panels):
        for g in show:
            if g not in curves:
                continue
            m, s, n = curves[g][name]
            ax.plot(SG, m, color=COL[g], lw=2, label=f"{g} (n={n})")
            ax.fill_between(SG, m - s, m + s, color=COL[g], alpha=0.12)
        for kind, c in lm.items():
            m, s, n = c[name]
            ax.plot(SG, m, color=COL[kind], lw=1.2, ls="--" if kind != "LATLERP" else ":", label=f"{kind} (S-GRID landmarks)")
        ax.set_title(title, fontsize=10); ax.set_xlabel("s (interior, 0=A anchor, 1=B anchor)"); ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{space}: anchor-relative trajectories, mean ± 1 std per group (S-GRID landmarks dashed)", fontsize=12)
    plt.tight_layout(); plt.savefig(INV / f"fig_traj_{space}.png"); plt.close(fig)
    return curves


summary = ["## Step A — per-space summary (medians over clips unless stated)", ""]
all_curves = {}
for sp in ("PIX", "DINO", "VAE"):
    all_curves[sp] = fig_space(sp)
pd.DataFrame(ms_rows).to_csv(INV / "curves_mean_std.csv", index=False)

# ---------------- per-clip spread + swap tail (position-free): jump3 >= 0.5 gap in tau within 3 frames (PIX/DINO), jump1 >= 0.5 (VAE, 1 latent step = 8 frames)
sw_rows = []
for sp in ("PIX", "DINO", "VAE"):
    r = rec[(rec.space == sp) & (rec.kind == "main")]
    for g, s in r.groupby("glabel"):
        jump = s.jump1 if sp == "VAE" else s.jump3
        sw_rows.append(dict(space=sp, group=g, n=len(s), swap_tail_frac=float((jump >= 0.5).mean()), jump_med=float(jump.median()),
                            s_cross_med=float(s.s_cross.median()), never_crossed_frac=float(s.s_cross.isna().mean()),
                            s_cross_iqr=float(s.s_cross.quantile(.75) - s.s_cross.quantile(.25)),
                            DR_med=float(s.DR.median()), M_med=float(s.M.median()), nu_max_med=float(s.nu_max.median()),
                            step_share_med=float(s.step_share.median()), path_over_gap_med=float(s.path_over_gap.median())))
sw = pd.DataFrame(sw_rows); sw.to_csv(INV / "swap_tail.csv", index=False)

# ---------------- clusters: k-means (k=3) on [tau(s), nu(s)] in PIX and DINO; fit on generated + reference groups; landmarks assigned
def kmeans(X, k, seed=0, iters=200):
    rng = np.random.default_rng(seed)
    # k-means++ init
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(C)
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1)
        C2 = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(C2, C):
            break
        C = C2
    inertia = float(((X - C[lab]) ** 2).sum())
    return lab, C, inertia

cl_rows = []
for sp in ("PIX", "DINO"):
    idx, arr = load(sp)
    r = rec[(rec.space == sp) & (rec.kind == "main")]
    ii = [idx[k] for k in r.key]
    X = np.concatenate([arr["tau"][ii], arr["nu"][ii]], 1)
    best = None
    for seed in range(8):
        lab, C, inert = kmeans(X, 3, seed)
        if best is None or inert < best[2]:
            best = (lab, C, inert)
    lab, C, _ = best
    # order clusters by crossing time of the prototype tau
    order = np.argsort([np.argmax(C[j][:K] >= 0.5) if (C[j][:K] >= 0.5).any() else K for j in range(3)])
    remap = {old: new for new, old in enumerate(order)}
    lab = np.array([remap[l] for l in lab]); C = C[order]
    r = r.assign(cluster=lab)
    # landmarks -> nearest prototype
    rl = rec[(rec.space == sp) & (rec.kind != "main") & (rec.glabel == "S-GRID:LM")]
    il = [idx[k] for k in rl.key]; XL = np.concatenate([arr["tau"][il], arr["nu"][il]], 1)
    labl = np.argmin(((XL[:, None, :] - C[None]) ** 2).sum(-1), 1)
    for j in range(3):
        proto = C[j]; tau_p, nu_p = proto[:K], proto[K:]
        cr = SG[np.argmax(tau_p >= 0.5)] if (tau_p >= 0.5).any() else np.nan
        desc = dict(space=sp, cluster=j, proto_cross_s=round(float(cr), 3), proto_tau_at_0p25=round(float(tau_p[K // 4]), 3),
                    proto_tau_at_0p75=round(float(tau_p[3 * K // 4]), 3), proto_nu_max=round(float(nu_p.max()), 3),
                    proto_max_dtau_3=round(float(np.max(np.abs(tau_p[3:] - tau_p[:-3]))), 3))
        for g, s in r.groupby("glabel"):
            desc[f"share:{g}"] = round(float((s.cluster == j).mean()), 3)
        for kind in ("LERP", "CUT50", "FREEZE"):
            m = rl.kind.values == kind
            desc[f"lm:{kind}"] = round(float((labl[m] == j).mean()), 2) if m.any() else np.nan
        # within-cluster medians of jump3 / step_share for the S-GRID null + R3
        s = r[(r.cluster == j) & (r.glabel.isin(["S-GRID:NULLGEN", "S-PROBE:R3:high"]))]
        desc["null_members_n"] = len(s); desc["null_jump3_med"] = round(float(s.jump3.median()), 3) if len(s) else np.nan
        desc["null_step_share_med"] = round(float(s.step_share.median()), 3) if len(s) else np.nan
        desc["null_DR_med"] = round(float(s.DR.median()), 3) if len(s) else np.nan
        cl_rows.append(desc)
    # figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=150)
    for j in range(3):
        ax = axes[j]
        for g in ["S-GRID:NULLGEN", "S-GRID:GT", "S-PROBE:R3:high", "S-PROBE:R1:high"]:
            s = r[(r.cluster == j) & (r.glabel == g)]
            if len(s) == 0:
                continue
            A = arr["tau"][[idx[k] for k in s.key]]
            for row in A[:25]:
                ax.plot(SG, row, color=COL[g], alpha=0.18, lw=0.8)
            ax.plot(SG, A.mean(0), color=COL[g], lw=2, label=f"{g}: {len(s)}/{(r.glabel == g).sum()}")
        ax.plot(SG, C[j][:K], color="k", lw=2, ls="--", label="prototype τ")
        ax.set_title(f"{sp} cluster {j} (crossing s={cl_rows[-3 + j]['proto_cross_s']})", fontsize=10); ax.set_ylim(-0.3, 1.4); ax.grid(alpha=0.2); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(INV / f"fig_clusters_{sp}.png"); plt.close(fig)
pd.DataFrame(cl_rows).to_csv(INV / "clusters.csv", index=False)

# ---------------- TRANS native curves from trans_profiles.csv (window-normalised time), mean ± std per group; swap-crossing aligned
tp = pd.read_csv(CAMP / "results/trans_profiles.csv", keep_default_na=False, low_memory=False)
for c in ["latent_t", "in_window", "nu_t", "sA_t", "sB_t", "swap_t", "trans_t", "inplace_t", "local_t", "ent_t"]:
    tp[c] = pd.to_numeric(tp[c], errors="coerce")
man = common.load_manifest().set_index("clip_id")
pcm = pd.read_csv(CAMP / "results/per_clip.csv", keep_default_na=False, usecols=["clip_id", "kind", "space", "landmark_owner", "tier"])
pcm = pcm[(pcm.kind == "main") & (pcm.space == "TRANS")].drop_duplicates("clip_id").set_index("clip_id")
KT = 25; SGT = np.linspace(0, 1, KT)
tr_rows = []; tcurves = {}
def trans_group(label, clip_ids, kind="main"):
    acc = {n: [] for n in ("nu", "swapp", "local", "inplace", "trans", "swapp_al", "nu_al", "local_al")}
    for cid in clip_ids:
        d = tp[(tp.clip_id == cid) & (tp.kind == kind) & (tp.in_window == 1)].sort_values("latent_t")
        if len(d) < 3:
            continue
        s = (d.latent_t.values - d.latent_t.values[0]) / (d.latent_t.values[-1] - d.latent_t.values[0])
        sw = d.swap_t.values; den = sw[-1] - sw[0]
        swapp = (sw - sw[0]) / den if abs(den) > 1e-6 else np.full_like(sw, np.nan)
        for n, v in (("nu", d.nu_t.values), ("swapp", swapp), ("local", d.local_t.values), ("inplace", d.inplace_t.values / (d.inplace_t.values.sum() + 1e-8)), ("trans", d.trans_t.values)):
            acc[n].append(np.interp(SGT, s, v))
        ge = np.where(swapp >= 0.5)[0]
        if len(ge) and 0 < s[ge[0]] < 1:
            sc = s[ge[0]]; src = np.where(SGT <= 0.5, sc * 2 * SGT, sc + (SGT - 0.5) * 2 * (1 - sc))
            acc["swapp_al"].append(np.interp(src, s, swapp)); acc["nu_al"].append(np.interp(src, s, d.nu_t.values)); acc["local_al"].append(np.interp(src, s, d.local_t.values))
    out = {}
    for n, L in acc.items():
        if not L:
            continue
        A = np.stack(L); out[n] = (A.mean(0), A.std(0), len(L))
        for j in range(KT):
            tr_rows.append(dict(space="TRANS", group=label, kind=kind, curve=n, s=round(SGT[j], 4), mean=out[n][0][j], std=out[n][1][j], n=len(L)))
    tcurves[label] = out

for g in ["S-GRID:NULLGEN", "S-GRID:GT", "S-PROBE:R3:high", "S-PROBE:R1:high", "S-PROBE:R2:high", "S-PROBE:R3:inplace", "S-SWEEP:A_empty"]:
    st, grp = g.split(":")[0], g.split(":")[1]; tier = g.split(":")[2] if len(g.split(":")) > 2 else None
    ids = man[(man.stratum == st) & (man.group == grp) & ((man.tier == tier) if tier else True)].index.tolist()
    trans_group(g, ids)
gt_ids = man[(man.stratum == "S-GRID") & (man.group == "GT")].index.tolist()
for kind in ("LERP", "CUT50", "FREEZE"):
    trans_group(f"S-GRID:LM:{kind}", gt_ids, kind=kind)
pd.DataFrame(tr_rows).to_csv(INV / "curves_mean_std_TRANS.csv", index=False)
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), dpi=150)
panels = [("swapp", "swap progress (sB−sA, normalised 0→1)"), ("nu", "TRANS novelty nu_t (ch nu, cell mean)"), ("local", "local_t: frac cells with 1−cos > q90"),
          ("swapp_al", "swap progress, crossing-aligned"), ("nu_al", "nu_t, crossing-aligned"), ("inplace", "in-place change share per step (Σ=1)")]
for ax, (name, title) in zip(axes.ravel(), panels):
    for g in ["S-GRID:NULLGEN", "S-GRID:GT", "S-PROBE:R3:high", "S-PROBE:R1:high", "S-PROBE:R3:inplace", "S-SWEEP:A_empty"]:
        if g in tcurves and name in tcurves[g]:
            m, s, n = tcurves[g][name]; ax.plot(SGT, m, color=COL[g], lw=2, label=f"{g} (n={n})"); ax.fill_between(SGT, m - s, m + s, color=COL[g], alpha=0.12)
    for kind in ("LERP", "CUT50", "FREEZE"):
        lab = f"S-GRID:LM:{kind}"
        if lab in tcurves and name in tcurves[lab]:
            m, s, n = tcurves[lab][name]; ax.plot(SGT, m, color=COL[kind], lw=1.2, ls="--", label=f"{kind} (S-GRID landmarks)")
    ax.set_title(title, fontsize=10); ax.set_xlabel("s (latent window a→b)"); ax.grid(alpha=0.2)
axes[0, 0].legend(fontsize=7)
fig.suptitle("TRANS: native 44-ch signal time courses, mean ± 1 std per group", fontsize=12)
plt.tight_layout(); plt.savefig(INV / "fig_traj_TRANS.png"); plt.close(fig)

# ---------------- summary md
def curve_at(space, g, name, s_list):
    c = all_curves[space].get(g)
    if c is None:
        return "—"
    m = c[name][0]
    return " / ".join(f"{m[int(round(s * (K - 1)))]:.2f}" for s in s_list)
summary += ["### Mean τ(s) at s = 0.1 / 0.25 / 0.5 / 0.75 / 0.9 (raw time) and mean ρ(s) at the same s", "",
            "| space | group | n | τ(s) | ρ(s) | ν(s) | swap-tail frac (jump≥0.5 gap in 3 frames / 1 latent) | s_cross med (IQR) | never crossed |", "|---|---|---|---|---|---|---|---|---|"]
for sp in ("PIX", "DINO", "VAE"):
    for g in ["S-GRID:NULLGEN", "S-GRID:GT", "S-GRID-F:NULLGEN", "S-PROBE:R3:high", "S-PROBE:R1:high", "S-PROBE:R2:high", "S-PROBE:R3:inplace", "S-PROBE:R1:inplace", "S-SWEEP:A_empty", "S-SWEEP:A_word"]:
        row = sw[(sw.space == sp) & (sw.group == g)]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        summary.append(f"| {sp} | {g} | {row.n} | {curve_at(sp, g, 'tau', [.1, .25, .5, .75, .9])} | {curve_at(sp, g, 'rho', [.1, .25, .5, .75, .9])} | {curve_at(sp, g, 'nu', [.1, .25, .5, .75, .9])} | {row.swap_tail_frac:.2f} | {row.s_cross_med:.2f} ({row.s_cross_iqr:.2f}) | {row.never_crossed_frac:.2f} |")
summary += ["", "### Landmarks read in each space (S-GRID owners): τ / ρ / ν at s = 0.25 / 0.5 / 0.75", ""]
for sp in ("PIX", "DINO", "VAE"):
    for kind in ("LERP", "CUT50", "FREEZE", "LATLERP"):
        rr = rec[(rec.space == sp) & (rec.kind == kind) & (rec.glabel == "S-GRID:LM")]
        if len(rr) == 0:
            continue
        idx, arr = load(sp); ii = [idx[k] for k in rr.key]
        t, r_, n_ = arr["tau"][ii].mean(0), arr["rho"][ii].mean(0), arr["nu"][ii].mean(0)
        f = lambda v: " / ".join(f"{v[int(round(s * (K - 1)))]:.2f}" for s in (.25, .5, .75))
        summary.append(f"- {sp} {kind}: τ {f(t)} · ρ {f(r_)} · ν {f(n_)} · DR med {rr.DR.median():.3f} · path/gap {rr.path_over_gap.median():.2f}")
summary += ["", "### Clusters (k-means k=3 on [τ(s), ν(s)], generated + reference clips; ordered by prototype crossing time)", ""]
cl = pd.DataFrame(cl_rows)
for _, c in cl.iterrows():
    shares = ", ".join(f"{k.split(':',1)[1]} {v:.2f}" for k, v in c.items() if k.startswith("share:") and k.split(':',1)[1] in ("S-GRID:NULLGEN", "S-GRID:GT", "S-PROBE:R3:high", "S-PROBE:R1:high", "S-PROBE:R3:inplace", "S-SWEEP:A_empty"))
    summary.append(f"- {c.space} cluster {c.cluster}: prototype crosses at s={c.proto_cross_s}, τ(0.25)={c.proto_tau_at_0p25}, τ(0.75)={c.proto_tau_at_0p75}, max 3-pt Δτ={c.proto_max_dtau_3}, ν_max={c.proto_nu_max}; null members n={c.null_members_n} jump3 med {c.null_jump3_med} step_share med {c.null_step_share_med} DR med {c.null_DR_med}; landmarks→ LERP {c['lm:LERP']} CUT50 {c['lm:CUT50']} FREEZE {c['lm:FREEZE']}; shares: {shares}")
(INV / "step_a_summary.md").write_text("\n".join(summary) + "\n")
print("\n".join(summary))
