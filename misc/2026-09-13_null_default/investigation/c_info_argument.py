"""Step C — the information argument made precise.

(1) What a pixel LERP 'adds' when read in DINO / VAE / TRANS: DR (distance from that space's straight line),
    nu_max (novelty), path/gap excess, M — per stratum, from per_clip.csv.
(2) LATLERP vs the VAE-encoded pixel LERP: per-timestep distance in gap units (how far the model's view of a real
    dissolve is from the latent straight line), and the latent-norm deficit along LATLERP (off-shell indicator)
    compared with LERP-encoded, GT and NULLGEN latents.
(3) DINO-space lerp: the chord midpoint norm sqrt((1+cos)/2) from the anchors' cosine (a chord point is not a
    unit vector, hence not the CLS of any image), per stratum.
Outputs: info_argument.csv (per-clip rows), info_argument.md (summary), fig_info_latent.png
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import sys, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

LAB = Path("/taiga/illinois/eng/cs/jrehg/users/emirkisa"); REPO = LAB / "diffusion-research"
CAMP = REPO / "misc/2026-09-13_null_default"; INV = CAMP / "investigation"
sys.path.insert(0, str(CAMP / "scripts")); import common
F = common.FEATURES

pc = pd.read_csv(CAMP / "results/per_clip.csv", keep_default_na=False, low_memory=False)
for c in ["DR", "M", "nu_max", "path_over_gap", "swap_sharp", "local_at_peak", "trans_mean", "step_share"]:
    pc[c] = pd.to_numeric(pc[c], errors="coerce")

# ---------------- (1) pixel LERP read in other spaces, per stratum (landmark-source clips)
rows = []
for sp in ("DINO", "VAE", "TRANS"):
    for st, g in pc[(pc.space == sp) & (pc.kind == "LERP")].groupby("stratum"):
        d = dict(space=sp, stratum=st, n=len(g))
        if sp == "TRANS":
            d.update(nu_max=g.nu_max.median(), swap_sharp=g.swap_sharp.median(), local_at_peak=g.local_at_peak.median())
        else:
            d.update(DR=g.DR.median(), M=g.M.median(), nu_max=g.nu_max.median(), path_over_gap=g.path_over_gap.median(), step_share=g.step_share.median())
        rows.append(d)
lerp_other = pd.DataFrame(rows)

# ---------------- (2) latent geometry: LATLERP vs encoded LERP vs GT vs NULLGEN
man = common.load_manifest()
pcm = pc[(pc.kind == "main") & (pc.space == "VAE")].drop_duplicates("clip_id").set_index("clip_id")["landmark_owner"]
sel = man[((man.stratum == "S-GRID") & (man.group.isin(["GT", "NULLGEN"]))) | ((man.stratum == "S-PROBE") & (man.group == "R3"))]
lat_rows = []
def norms(v):
    return np.linalg.norm(v.reshape(v.shape[0], -1).astype(np.float64), axis=1)
for r in sel.itertuples():
    own = pcm.get(r.clip_id, r.clip_id)
    la, lb = common.lat_index(r.a_idx), common.lat_index(r.b_idx)
    z = np.load(F / f"{r.clip_id}.npz"); v_main = z["vae"].astype(np.float32).reshape(z["vae"].shape[0], -1)
    zl = np.load(F / f"{own}__LERP.npz"); v_lerp = zl["vae"].astype(np.float32).reshape(-1, v_main.shape[1])
    zz = np.load(F / f"{own}__LATLERP.npz"); v_lat = zz["vae"].astype(np.float32).reshape(-1, v_main.shape[1])
    zc = np.load(F / f"{own}__CUT50.npz"); v_cut = zc["vae"].astype(np.float32).reshape(-1, v_main.shape[1])
    a, b = v_lat[la].astype(np.float64), v_lat[lb].astype(np.float64); gap = np.linalg.norm(b - a)
    anchor_norm = 0.5 * (np.linalg.norm(a) + np.linalg.norm(b))
    inter = slice(la + 1, lb)
    # distance encoded-LERP -> LATLERP per interior timestep (gap units); both share the same anchors up to encoder noise
    d_enc_lat = np.linalg.norm(v_lerp[inter].astype(np.float64) - v_lat[inter].astype(np.float64), axis=1) / gap
    # norm profiles (relative to anchor norm)
    n_lat = norms(v_lat[inter]) / anchor_norm; n_lerp = norms(v_lerp[inter]) / anchor_norm
    n_main = norms(v_main[inter]) / anchor_norm; n_cut = norms(v_cut[inter]) / anchor_norm
    # DINO chord midpoint norm from CLS anchors
    dn = z["dino"].astype(np.float64); ca, cb = dn[r.a_idx], dn[r.b_idx]
    cos = float(ca @ cb); mid_norm = float(np.sqrt(max(0.0, (1 + cos) / 2)))
    lat_rows.append(dict(clip_id=r.clip_id, stratum=r.stratum, group=r.group, owner=own,
                         gap_over_anchor_norm=float(gap / anchor_norm),
                         d_encLERP_to_LATLERP_med=float(np.median(d_enc_lat)), d_encLERP_to_LATLERP_mid=float(d_enc_lat[len(d_enc_lat) // 2]),
                         latlerp_norm_min=float(n_lat.min()), enclerp_norm_min=float(n_lerp.min()),
                         main_norm_min=float(n_main.min()), main_norm_med=float(np.median(n_main)), cut_norm_min=float(n_cut.min()),
                         dino_cos_AB=cos, dino_chord_mid_norm=mid_norm,
                         anchor_a_norm=float(np.linalg.norm(a)), anchor_b_norm=float(np.linalg.norm(b))))
lat = pd.DataFrame(lat_rows)
lat.to_csv(INV / "info_argument.csv", index=False)

def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + " |")
    return "\n".join(out)

def q(s):
    return f"{s.median():.3f} [{s.quantile(.25):.3f}, {s.quantile(.75):.3f}]"
md = ["## Step C numbers", "", "### (1) The pixel LERP read in the other spaces (landmark-source clips; medians)", "",
      md_table(lerp_other.round(3)), "",
      "### (2) VAE latent geometry (interior latent timesteps; norms relative to the mean anchor-latent norm)", "",
      "| stratum·group | n | gap / anchor-norm | d(enc-LERP, LATLERP) med (gap units) | at mid | LATLERP min norm | enc-LERP min norm | real clip min norm | real clip med norm | CUT50 min norm |", "|---|---|---|---|---|---|---|---|---|---|"]
for (st, g), s in lat.groupby(["stratum", "group"]):
    md.append(f"| {st}·{g} | {len(s)} | {q(s.gap_over_anchor_norm)} | {q(s.d_encLERP_to_LATLERP_med)} | {q(s.d_encLERP_to_LATLERP_mid)} | {q(s.latlerp_norm_min)} | {q(s.enclerp_norm_min)} | {q(s.main_norm_min)} | {q(s.main_norm_med)} | {q(s.cut_norm_min)} |")
md += ["", "### (3) DINO chord (feature-space lerp) midpoint norm — a unit-sphere CLS space; chord points are not CLS vectors of any image", "",
       "| stratum·group | cos(A,B) med [IQR] | chord midpoint norm med [IQR] |", "|---|---|---|"]
for (st, g), s in lat.groupby(["stratum", "group"]):
    md.append(f"| {st}·{g} | {q(s.dino_cos_AB)} | {q(s.dino_chord_mid_norm)} |")
(INV / "info_argument.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

# ---------------- figure: norm profiles along the interior (S-GRID owners)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
own_list = lat[lat.stratum == "S-GRID"].owner.unique()[:19]
prof = {"LATLERP": [], "enc-LERP": [], "GT": [], "CUT50": []}
for own in own_list:
    r = man[man.clip_id == own].iloc[0]
    la, lb = common.lat_index(r.a_idx), common.lat_index(r.b_idx); inter = slice(la, lb + 1)
    vs = {"LATLERP": np.load(F / f"{own}__LATLERP.npz")["vae"], "enc-LERP": np.load(F / f"{own}__LERP.npz")["vae"],
          "GT": np.load(F / f"{own}.npz")["vae"], "CUT50": np.load(F / f"{own}__CUT50.npz")["vae"]}
    an = 0.5 * (np.linalg.norm(vs["GT"][la].astype(np.float64)) + np.linalg.norm(vs["GT"][lb].astype(np.float64)))
    for k, v in vs.items():
        prof[k].append(norms(v[inter].astype(np.float32)) / an)
cols = {"LATLERP": "#d62728", "enc-LERP": "#ff7f0e", "GT": "#1f77b4", "CUT50": "#7f7f7f"}
for k, P in prof.items():
    P = np.stack(P); s = np.linspace(0, 1, P.shape[1]); m, sd = P.mean(0), P.std(0)
    axes[0].plot(s, m, color=cols[k], label=k); axes[0].fill_between(s, m - sd, m + sd, color=cols[k], alpha=0.15)
axes[0].set_xlabel("s (latent window a→b)"); axes[0].set_ylabel("‖latent_t‖ / mean anchor norm"); axes[0].set_title("VAE latent norm along the interior (S-GRID, 19 owners)"); axes[0].legend(fontsize=8)
axes[1].hist(lat[lat.stratum == "S-GRID"].dino_chord_mid_norm, bins=15, color="#9467bd", alpha=0.8, label="S-GRID")
axes[1].hist(lat[lat.stratum == "S-PROBE"].dino_chord_mid_norm, bins=15, color="#2ca02c", alpha=0.5, label="S-PROBE R3")
axes[1].axvline(1.0, color="k", ls="--", lw=1); axes[1].set_xlabel("DINO chord midpoint norm (unit sphere = 1)"); axes[1].set_title("DINO-space lerp leaves the CLS sphere"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(INV / "fig_info_latent.png"); print("fig saved")
