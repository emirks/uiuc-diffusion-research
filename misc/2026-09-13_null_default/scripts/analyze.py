"""C2 - SPEC 5.2-5.4 tables, collapse measures, and figures.

Reads results/per_clip.csv (+ trans_profiles.csv), writes results/TABLES.md,
results/collapse_measures.csv, results/fig_*.png and results/strips/. Partial-safe: every stat
guards empty/small groups (returns NaN); nothing here decodes GPU features. Bootstrap is 2,000
resamples clustered by endpoint (S-PROBE: by prompt_id).

    python misc/2026-09-13_null_default/scripts/analyze.py [--no-strips]
"""
from __future__ import annotations

import argparse
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

RESULTS = C.CAMP / "results"
EMB = (["DR", "M", "cross", "nu_max", "nu_mean", "explained", "step_share", "path_over_gap"]
       + [f"tau_{i}" for i in range(10)])              # 18-d (SPEC 3)
POINT_SPACES = ["PIX", "DINO", "VAE"]
LM_KINDS = ("LERP", "CUT50", "FREEZE", "LATLERP")
REF_GROUPS = ("GT", "R1")
NULL_GROUPS = ("NULLGEN", "R2", "R3", "A_empty", "A_word", "B", "C", "D", "E")
TABLE_METRICS = ["DR", "M", "cross", "nu_max", "explained", "step_share", "path_over_gap"]
TRANS_METRICS = ["nu_max", "nu_mean", "swap_sharp", "swap_pos", "trans_mean", "trans_peak",
                 "inplace_peak_share", "local_at_peak", "ent_mean"]
RNG = np.random.default_rng(0)


def cluster_key(df):
    """endpoint, but prompt_id for S-PROBE (SPEC 5.2)."""
    return np.where(df["stratum"].values == "S-PROBE", df["prompt_id"].values, df["endpoint"].values)


def boot_ci(vals, clusters, n=2000, stat=np.median):
    vals = np.asarray(vals, float)
    ok = np.isfinite(vals)
    vals, clusters = vals[ok], np.asarray(clusters)[ok]
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan, 0)
    uc = np.unique(clusters)
    if len(uc) < 2:
        return (float(stat(vals)), np.nan, np.nan, len(vals))
    idx = {c: np.where(clusters == c)[0] for c in uc}
    est = []
    for _ in range(n):
        pick = RNG.choice(uc, size=len(uc), replace=True)
        s = np.concatenate([idx[c] for c in pick])
        est.append(stat(vals[s]))
    return (float(stat(vals)), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)), len(vals))


def table_group(df):
    return np.where(df["kind"].values == "main", df["group"].values, df["kind"].values)


# --------------------------------------------------------------------------- 5.2 tables
def write_tables(df):
    df = df.copy()
    df["tgroup"] = table_group(df)
    lines = ["# TABLES — null-default (SPEC 5.2)", "",
             "Median [bootstrap 95% CI], 2000 resamples clustered by endpoint (S-PROBE: prompt_id).",
             "Groups: GT/NULLGEN/R1/R2/R3/A_*..E (kind=main) + LERP/CUT50/FREEZE/LATLERP (landmarks).", ""]
    for space in POINT_SPACES + ["TRANS"]:
        metrics = TRANS_METRICS if space == "TRANS" else TABLE_METRICS
        lines += [f"## {space}", "", "| stratum | group | n | " + " | ".join(metrics) + " |",
                  "|" + "---|" * (len(metrics) + 3)]
        sub = df[df.space == space]
        for stratum in ["S-GRID", "S-GRID-F", "S-GRID-START", "S-PROBE", "S-SWEEP"]:
            ss = sub[sub.stratum == stratum]
            for g in sorted(ss.tgroup.unique()):
                gg = ss[ss.tgroup == g]
                ck = cluster_key(gg)
                cells = []
                for m in metrics:
                    med, lo, hi, nn = boot_ci(gg[m].values, ck)
                    cells.append(f"{med:.3f} [{lo:.3f}, {hi:.3f}]" if np.isfinite(med) else "—")
                lines.append(f"| {stratum} | {g} | {len(gg)} | " + " | ".join(cells) + " |")
        lines.append("")
    (RESULTS / "TABLES.md").write_text("\n".join(lines) + "\n")
    print(f"[analyze] wrote TABLES.md", flush=True)


# --------------------------------------------------------------------------- embedding pool
def space_matrix(df, space):
    s = df[(df.space == space)].copy()
    X = s[EMB].apply(pd.to_numeric, errors="coerce").values.astype(float)
    return s.reset_index(drop=True), X


def build_pool(s, X):
    """reference (GT/R1 main) + all landmark rows -> scaler + standardized pool + labels."""
    is_ref = (s.kind == "main") & (s.group.isin(REF_GROUPS))
    is_lm = s.kind.isin(LM_KINDS)
    pool_mask = (is_ref | is_lm).values & np.isfinite(X).all(1)
    Xp = X[pool_mask]
    mu = Xp.mean(0)
    sd = Xp.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    labels = np.where(s.kind.isin(LM_KINDS).values, s.kind.values,
                      np.where(s.group.isin(REF_GROUPS).values, "REF:" + s.group.values, "OTHER"))
    return mu, sd, Z, pool_mask, labels, is_ref.values


def nearest_label(zq, Zpool, pool_labels, exclude_row=None):
    d = np.linalg.norm(Zpool - zq, axis=1)
    if exclude_row is not None:
        d[exclude_row] = np.inf
    j = int(np.argmin(d))
    return pool_labels[j]


# --------------------------------------------------------------------------- 5.3 collapse measures
def collapse_measures(df):
    from sklearn.cluster import KMeans
    recs = []
    agree = {}                      # clip_id -> count of point spaces where landmark-nearest
    for space in POINT_SPACES:
        s, X = space_matrix(df, space)
        if len(s) == 0:
            continue
        mu, sd, Z, pool_mask, labels, is_ref = build_pool(s, X)
        pool_idx = np.where(pool_mask)[0]
        Zpool = Z[pool_idx]
        pool_labels = labels[pool_idx]
        finite = np.isfinite(Z).all(1)
        # ---- landmark_nearest for null rows
        for stratum in ["S-GRID", "S-GRID-F", "S-GRID-START", "S-PROBE", "S-SWEEP"]:
            for g in NULL_GROUPS:
                rows = s[(s.stratum == stratum) & (s.group == g) & (s.kind == "main")]
                rows = rows[finite[rows.index]]
                if len(rows) == 0:
                    continue
                fams = {"LERP": 0, "CUT50": 0, "FREEZE": 0, "LATLERP": 0, "REF": 0}
                hits = []
                for i in rows.index:
                    lab = nearest_label(Z[i], Zpool, pool_labels)
                    islm = lab in LM_KINDS
                    hits.append(1 if islm else 0)
                    fams[lab if islm else "REF"] += 1
                    if islm:
                        agree[s.loc[i, "clip_id"]] = agree.get(s.loc[i, "clip_id"], 0) + 1
                ck = cluster_key(rows)
                med, lo, hi, nn = boot_ci(np.array(hits), ck, stat=np.mean)
                recs.append(dict(measure="landmark_nearest", space=space, stratum=stratum, group=g,
                                 n=len(rows), frac=med, lo=lo, hi=hi,
                                 **{f"fam_{k}": v for k, v in fams.items()}))
        # ---- reference LOO baseline
        for g in REF_GROUPS:
            refs = s[(s.kind == "main") & (s.group == g)]
            refs = refs[finite[refs.index]]
            if len(refs) == 0:
                continue
            hits = []
            for i in refs.index:
                excl = np.where(pool_idx == i)[0]
                lab = nearest_label(Z[i], Zpool, pool_labels, exclude_row=(excl[0] if len(excl) else None))
                hits.append(1 if lab in LM_KINDS else 0)
            ck = cluster_key(refs)
            med, lo, hi, nn = boot_ci(np.array(hits), ck, stat=np.mean)
            recs.append(dict(measure="landmark_nearest_LOObaseline", space=space, stratum="REF",
                             group=g, n=len(refs), frac=med, lo=lo, hi=hi))
        # ---- occupancy entropy (k-means k=6 on pool)
        if len(Zpool) >= 6:
            km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(Zpool)
            def norm_ent(assign):
                if len(assign) == 0:
                    return np.nan
                p = np.bincount(assign, minlength=6) / len(assign)
                p = p[p > 0]
                return float(-(p * np.log(p)).sum() / np.log(6))
            ref_assign = km.predict(Zpool[np.isin(pool_labels, ["REF:GT", "REF:R1"])]) \
                if np.isin(pool_labels, ["REF:GT", "REF:R1"]).any() else np.array([], int)
            ref_own = norm_ent(ref_assign)
            for stratum in ["S-GRID", "S-GRID-F", "S-GRID-START", "S-PROBE", "S-SWEEP"]:
                for g in NULL_GROUPS:
                    rows = s[(s.stratum == stratum) & (s.group == g) & (s.kind == "main")]
                    rows = rows[finite[rows.index]]
                    if len(rows) == 0:
                        continue
                    a = km.predict(Z[rows.index])
                    ck = cluster_key(rows)
                    _, lo, hi, _ = boot_ci(a.astype(float), ck, stat=lambda v: norm_ent(v.astype(int)))
                    recs.append(dict(measure="occupancy_entropy", space=space, stratum=stratum,
                                     group=g, n=len(rows), frac=norm_ent(a), lo=lo, hi=hi,
                                     ref_own=ref_own))
    coll = pd.DataFrame(recs)
    coll.to_csv(RESULTS / "collapse_measures.csv", index=False)
    print(f"[analyze] wrote collapse_measures.csv ({len(coll)} rows)", flush=True)
    return coll, agree


# --------------------------------------------------------------------------- cross-space agreement + TRANS sig
def cross_space(df, agree):
    tr = df[df.space == "TRANS"]
    gt = tr[(tr.kind == "main") & (tr.group == "GT")]
    nu_p5 = np.nanpercentile(gt["nu_max"].astype(float), 5) if len(gt) else np.nan
    sw_p95 = np.nanpercentile(gt["swap_sharp"].astype(float), 95) if len(gt) else np.nan
    rows = []
    nulls = df[(df.kind == "main") & (df.group.isin(NULL_GROUPS))][["clip_id", "stratum", "group"]].drop_duplicates()
    trmap = {(r.clip_id): (r.nu_max, r.swap_sharp) for r in tr[tr.kind == "main"].itertuples()}
    for r in nulls.itertuples():
        n_agree = agree.get(r.clip_id, 0)
        nm, ss = trmap.get(r.clip_id, (np.nan, np.nan))
        swaplike = int(np.isfinite(nm) and np.isfinite(ss) and nm < nu_p5 and ss > sw_p95) \
            if (np.isfinite(nu_p5) and np.isfinite(sw_p95)) else 0
        rows.append(dict(clip_id=r.clip_id, stratum=r.stratum, group=r.group,
                         n_point_landmark_nearest=n_agree, trans_swaplike=swaplike))
    ag = pd.DataFrame(rows)
    ag.to_csv(RESULTS / "cross_space_agreement.csv", index=False)
    # counts table
    summ = ag.groupby(["stratum", "group"]).agg(
        n=("clip_id", "size"),
        agree0=("n_point_landmark_nearest", lambda x: int((x == 0).sum())),
        agree1=("n_point_landmark_nearest", lambda x: int((x == 1).sum())),
        agree2=("n_point_landmark_nearest", lambda x: int((x == 2).sum())),
        agree3=("n_point_landmark_nearest", lambda x: int((x == 3).sum())),
        swaplike=("trans_swaplike", "sum")).reset_index()
    print(f"[analyze] cross-space agreement (nu_p5={nu_p5:.3f}, swap_p95={sw_p95:.3f}):", flush=True)
    print(summ.to_string(), flush=True)
    return ag, summ, nu_p5, sw_p95


# --------------------------------------------------------------------------- 5.3 S-PROBE paired
def probe_paired(df):
    from scipy.stats import wilcoxon
    recs = []
    for space, metric in [("PIX", "DR"), ("DINO", "DR"), ("VAE", "DR"),
                          ("PIX", "nu_max"), ("PIX", "step_share"), ("TRANS", "swap_sharp")]:
        sub = df[(df.space == space) & (df.stratum == "S-PROBE") & (df.kind == "main")]
        piv = sub.pivot_table(index=["prompt_id", "seed", "tier"], columns="group", values=metric, aggfunc="first")
        for a, b in [("R3", "R1"), ("R3", "R2")]:
            if a not in piv or b not in piv:
                continue
            d = (piv[a] - piv[b]).dropna()
            if len(d) == 0:
                continue
            try:
                p = float(wilcoxon(d.values).pvalue) if np.any(d.values != 0) else np.nan
            except ValueError:
                p = np.nan
            recs.append(dict(space=space, metric=metric, contrast=f"{a}-{b}", n=len(d),
                             median=float(np.median(d)), n_neg=int((d < 0).sum()),
                             n_pos=int((d > 0).sum()), wilcoxon_p=p))
    pp = pd.DataFrame(recs)
    pp.to_csv(RESULTS / "probe_paired.csv", index=False)
    print(f"[analyze] wrote probe_paired.csv ({len(pp)} rows)", flush=True)
    return pp


# --------------------------------------------------------------------------- figures
def figures(df, no_strips=False):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    STRATA = ["S-GRID", "S-GRID-F", "S-GRID-START", "S-PROBE", "S-SWEEP"]
    cmap = plt.get_cmap("tab10")
    # fig_pca per point space
    for space in POINT_SPACES:
        s, X = space_matrix(df, space)
        if len(s) < 5:
            continue
        mu, sd, Z, pool_mask, labels, is_ref = build_pool(s, X)
        fin = np.isfinite(Z).all(1)
        if pool_mask.sum() < 3:
            continue
        pca = PCA(n_components=2, random_state=0).fit(Z[pool_mask & fin])
        P = pca.transform(Z[fin])
        sfin = s[fin].reset_index(drop=True)
        fig, axes = plt.subplots(1, len(STRATA), figsize=(4 * len(STRATA), 4), squeeze=False)
        groups = sorted(set(sfin["group"]).union(LM_KINDS))
        gcol = {g: cmap(i % 10) for i, g in enumerate(groups)}
        for axi, stratum in enumerate(STRATA):
            ax = axes[0][axi]
            mask = (sfin["stratum"] == stratum).values
            for g in groups:
                gm = mask & (np.where(sfin["kind"].values == "main", sfin["group"].values,
                                      sfin["kind"].values) == g)
                if gm.sum() == 0:
                    continue
                hollow = g in LM_KINDS
                ax.scatter(P[gm, 0], P[gm, 1], s=18, label=g, color=gcol[g],
                           facecolors="none" if hollow else gcol[g], edgecolors=gcol[g], alpha=0.7)
            ax.set_title(f"{space} · {stratum}"); ax.grid(alpha=0.2)
        axes[0][0].legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(RESULTS / f"fig_pca_{space}.png", dpi=150); plt.close(fig)
    # fig_tau per point space
    for space in POINT_SPACES:
        s = df[df.space == space]
        if len(s) == 0:
            continue
        s = s.copy(); s["tgroup"] = table_group(s)
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, g in enumerate(sorted(s.tgroup.unique())):
            gg = s[s.tgroup == g][[f"tau_{k}" for k in range(10)]].apply(pd.to_numeric, errors="coerce")
            if gg.dropna(how="all").empty:
                continue
            med = gg.median(); q1 = gg.quantile(.25); q3 = gg.quantile(.75)
            xs = np.arange(10)
            ax.plot(xs, med.values, label=g, color=cmap(i % 10))
            ax.fill_between(xs, q1.values, q3.values, color=cmap(i % 10), alpha=0.12)
        ax.set_xlabel("interior bin"); ax.set_ylabel("tau"); ax.set_title(f"tau profile · {space}")
        ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.2)
        fig.tight_layout(); fig.savefig(RESULTS / f"fig_tau_{space}.png", dpi=150); plt.close(fig)
    # fig_trans time courses
    tp = RESULTS / "trans_profiles.csv"
    if tp.exists():
        prof = pd.read_csv(tp, keep_default_na=False)
        prof["tgroup"] = np.where(prof["kind"].values == "main", prof["group"].values, prof["kind"].values)
        chans = ["nu_t", "swap_t", "trans_t", "inplace_t", "local_t"]
        fig, axes = plt.subplots(1, len(chans), figsize=(4 * len(chans), 4), squeeze=False)
        for ci, ch in enumerate(chans):
            ax = axes[0][ci]
            for i, g in enumerate(sorted(prof.tgroup.unique())):
                gg = prof[prof.tgroup == g]
                if gg.empty:
                    continue
                m = gg.groupby("latent_t")[ch].median()
                ax.plot(m.index, m.values, label=g, color=cmap(i % 10))
            ax.set_title(ch); ax.grid(alpha=0.2)
        axes[0][0].legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(RESULTS / "fig_trans.png", dpi=150); plt.close(fig)
    print("[analyze] wrote figures (fig_pca_*, fig_tau_*, fig_trans)", flush=True)


def fig_agreement(summ):
    import matplotlib.pyplot as plt
    if summ is None or summ.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [f"{r.stratum}/{r.group}" for r in summ.itertuples()]
    bottom = np.zeros(len(summ))
    for k in ["agree0", "agree1", "agree2", "agree3"]:
        ax.bar(labels, summ[k].values, bottom=bottom, label=k)
        bottom += summ[k].values
    ax.set_ylabel("clips"); ax.set_title("point-space landmark-nearest agreement (0-3)")
    ax.legend(fontsize=7); plt.xticks(rotation=90, fontsize=6)
    fig.tight_layout(); fig.savefig(RESULTS / "fig_agreement.png", dpi=150); plt.close(fig)
    print("[analyze] wrote fig_agreement.png", flush=True)


# --------------------------------------------------------------------------- strips (serial cv2)
def strips(df, agree):
    import cv2
    cv2.setNumThreads(1)
    outdir = RESULTS / "strips"; outdir.mkdir(exist_ok=True)
    man = C.load_manifest().set_index("clip_id")
    # nearest-landmark family per null clip in PIX (representative space)
    s, X = space_matrix(df, "PIX")
    if len(s) == 0:
        return
    mu, sd, Z, pool_mask, labels, is_ref = build_pool(s, X)
    pool_idx = np.where(pool_mask)[0]; Zpool = Z[pool_idx]; pool_labels = labels[pool_idx]
    fin = np.isfinite(Z).all(1)
    picks = {}   # (stratum, family) -> [clip_ids]
    for i in s.index:
        r = s.loc[i]
        if r["kind"] != "main" or r["group"] not in NULL_GROUPS or not fin[i]:
            continue
        lab = nearest_label(Z[i], Zpool, pool_labels)
        if lab not in LM_KINDS:
            continue
        key = (r["stratum"], lab)
        picks.setdefault(key, [])
        if len(picks[key]) < 3:
            picks[key].append(r["clip_id"])
    made = 0
    for (stratum, fam), cids in picks.items():
        for cid in cids:
            if cid not in man.index:
                continue
            path = C.REPO / man.loc[cid, "path"]
            try:
                cap = cv2.VideoCapture(str(path)); frames = []
                while True:
                    ok, f = cap.read()
                    if not ok:
                        break
                    frames.append(f)
                cap.release()
                if not frames:
                    continue
                idxs = np.linspace(0, len(frames) - 1, 12).astype(int)
                strip = np.concatenate([cv2.resize(frames[j], (128, 128)) for j in idxs], axis=1)
                cv2.imwrite(str(outdir / f"{stratum}__{fam}__{cid}.png"), strip)
                made += 1
            except Exception:
                continue
    print(f"[analyze] wrote {made} strips -> {outdir}", flush=True)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-strips", action="store_true")
    args = ap.parse_args()
    df = pd.read_csv(RESULTS / "per_clip.csv", keep_default_na=False)
    for c in EMB + TRANS_METRICS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"[analyze] per_clip rows={len(df)}, spaces={df.space.unique().tolist()}", flush=True)

    write_tables(df)
    coll, agree = collapse_measures(df)
    ag, summ, nu_p5, sw_p95 = cross_space(df, agree)
    pp = probe_paired(df)
    figures(df, no_strips=args.no_strips)
    fig_agreement(summ)
    if not args.no_strips:
        strips(df, agree)
    print("[analyze] DONE", flush=True)


if __name__ == "__main__":
    main()
