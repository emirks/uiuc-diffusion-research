"""Metric-search experiment runner. Executor side: build candidate distance
matrices, score them through the FIXED exam, report exactly what the advisor asked
for (acc / Cohen's d / coverage / misretrieved / hubness + per-class recall on the
named confusion clusters), and append to a persistent results log. The advisor picks
the next batch from these numbers.
"""

from __future__ import annotations

import json
import time

import numpy as np

from . import harness as H
from . import kernels as K

# The confusion-cluster classes the advisor tracks (Experiment 1 readout list).
WATCH = ["shadow", "shadow_smoke", "gas_transformation", "money_rain", "color_rain",
         "wireframe", "polygon", "water_element", "fire_element",
         "cotton_cloud", "firelava", "mystification", "giant_grab", "saint_glow"]

RESULTS_LOG = H.paths.WB_OUT / "search" / "results_log.jsonl"
SE_ACC = 0.028      # binomial SE on accuracy at n=223 (advisor's tie threshold ~1 SE)


def per_class(r: dict, classes: list[str]) -> dict:
    pcr = r["per_class_recall"]
    return {c: pcr.get(c) for c in classes}


def report(ctx: dict, name: str, D: np.ndarray, note: str = "",
           reasons=None, save: bool = True) -> dict:
    r = H.run(ctx, name, D, reasons=reasons)
    v = H.verdict(r, ctx)
    inc = ctx["incumbent"]
    dacc = r["accuracy_1nn"] - inc["accuracy_1nn"]
    dd = r["separation_cohens_d"] - inc["separation_cohens_d"]
    print(f"   Δacc {dacc:+.4f} ({dacc/SE_ACC:+.1f} SE)   Δd {dd:+.4f}   "
          f"genuinely_beats={v['genuinely_beats']}")
    wc = per_class(r, WATCH)
    print("   watch:", "  ".join(f"{c}={wc[c]:.2f}" if wc[c] is not None else f"{c}=NA"
                                 for c in WATCH))
    rec = {
        "name": name, "note": note,
        "accuracy": r["accuracy_1nn"], "cohens_d": r["separation_cohens_d"],
        "coverage": r["coverage"], "misretrieved": r["misretrieved"],
        "n_correct": int(r["n_clips"] - r["misretrieved"]),
        "hubness_pass": r["hubness"]["pass"],
        "hubness": {k: r["hubness"]["stats"][k] for k in
                    ("hubness_skew", "pred_entropy_norm", "max_pred_class_share")},
        "delta_acc": dacc, "delta_d": dd, "genuinely_beats": v["genuinely_beats"],
        "watch_recall": wc,
        "incumbent": {"accuracy": inc["accuracy_1nn"], "cohens_d": inc["separation_cohens_d"],
                      "misretrieved": inc["misretrieved"]},
    }
    if save:
        RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_LOG, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
    return {"result": r, "verdict": v, "record": rec, "D": D}


# --- Experiment 1: the space test (C1b centered, C1 LW-whitened) --------------

def exp1(ctx: dict, sub: dict) -> dict:
    print("\n=== EXPERIMENT 1 — the space test (raw / centered / LW-whitened) ===")
    C_raw = H.cores(sub, "sided")

    t0 = time.time()
    D_raw = K.soft_chamfer_matrix(C_raw)
    print(f"[build raw] {time.time()-t0:.1f}s")
    out = {}
    out["raw"] = report(ctx, "m1a_raw_sided", D_raw,
                        "sanity: must reproduce the incumbent exactly")

    tf_c = K.center_transform(C_raw)
    C_c = K.apply_transform(C_raw, tf_c)
    t0 = time.time()
    D_c = K.soft_chamfer_matrix(C_c)
    print(f"[build centered] {time.time()-t0:.1f}s")
    out["centered"] = report(ctx, "C1b_centered_chamfer", D_c,
                             "(f-mu)/||.|| pooled-corpus mean removed")

    tf_w, art = K.lw_transform(C_raw)
    C_w = K.apply_transform(C_raw, tf_w)
    t0 = time.time()
    D_w = K.soft_chamfer_matrix(C_w)
    print(f"[build LW] {time.time()-t0:.1f}s  (shrinkage δ={art['shrinkage']:.6f}, "
          f"cond {art['condition_number_raw']:.2e}->{art['condition_number_shrunk']:.2e})")
    out["lw"] = report(ctx, "C1_lw_whitened_chamfer", D_w,
                       f"Ledoit-Wolf ZCA, δ={art['shrinkage']:.6f}, no eig floor")
    return out


CHANNELS_NPZ = H.paths.WB_CACHE / "search" / "channels.npz"


def build_channels(ctx: dict, sub: dict, save: bool = True) -> dict:
    """Compute + cache the cheap base distance matrices (chamfer-based), so any
    stack/fusion the advisor specifies is an instant ECDF-average of cached channels.
    EMD is held out (16 min) until greenlit."""
    C_raw = H.cores(sub, "sided")
    feats, msk = sub["feats"], sub["mask_sided"]
    npre, nsuf = sub["n_prefix"], sub["n_suffix"]
    n = len(C_raw)

    def dset(order, horizon):
        return [K.diff_sets(feats[i], msk[i], int(npre[i]), int(nsuf[i]),
                            order=order, horizon=horizon) for i in range(n)]

    # appearance spaces
    tf_c = K.center_transform(C_raw)
    C_cen = K.apply_transform(C_raw, tf_c)
    # debiased appearance: project out each clip's own endpoint directions e_A,e_B
    C_deb = []
    for i in range(n):
        eA, eB = K.endpoint_anchors(feats[i], int(npre[i]), int(nsuf[i]))
        C_deb.append(K.unit(K.project_out(C_raw[i], np.stack([eA, eB]))))

    ch = {}
    ch["app_raw"] = K.soft_chamfer_matrix(C_raw)
    ch["app_centered"] = K.soft_chamfer_matrix(C_cen)
    ch["app_debiased"] = K.soft_chamfer_matrix(C_deb)
    ch["vel"] = K.soft_chamfer_matrix(dset(1, 1))
    ch["accel"] = K.soft_chamfer_matrix(dset(2, 1))
    ch["horizon2"] = K.soft_chamfer_matrix(dset(1, 2))
    ch["horizon4"] = K.soft_chamfer_matrix(dset(1, 4))
    if save:
        CHANNELS_NPZ.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CHANNELS_NPZ, **ch)
        print(f"[channels] cached {list(ch)} -> {CHANNELS_NPZ}")
    return ch


def load_channels() -> dict:
    z = np.load(CHANNELS_NPZ, allow_pickle=True)
    return {k: z[k] for k in z.keys()}


def stack(ctx: dict, ch: dict, combo: list[str], name: str = "",
          weights=None) -> dict:
    """ECDF-fuse a named subset of channels and score it."""
    name = name or ("stack[" + "+".join(combo) + "]")
    D = K.ecdf_fuse([ch[c] for c in combo], weights=weights)
    return report(ctx, name, D, note="ECDF-avg of " + ", ".join(combo))


def g1_debiased_centered(sub: dict) -> list[np.ndarray]:
    """G1 (fable spec): centered space, project out the rank-2 endpoint subspace
    {e_A^c, e_B^c}, renormalize. Keeps the full 768-D frame minus the endpoint plane
    — NOT a rank-1 departure collapse. Coverage guard: empty clip -> centered core."""
    C_raw = H.cores(sub, "sided")
    feats, npre, nsuf = sub["feats"], sub["n_prefix"], sub["n_suffix"]
    mu = K.pooled(C_raw).mean(axis=0)
    out = []
    for i in range(len(C_raw)):
        eA, eB = K.endpoint_anchors(feats[i], int(npre[i]), int(nsuf[i]))
        eAc, eBc = eA - mu, eB - mu
        u1 = eAc / (np.linalg.norm(eAc) + 1e-12)
        u2 = eBc - (eBc @ u1) * u1
        n2 = np.linalg.norm(u2)
        B = np.stack([u1, u2 / n2]) if n2 > 1e-6 else u1[None]
        Fc = C_raw[i] - mu
        Fp = Fc - (Fc @ B.T) @ B
        nrm = np.linalg.norm(Fp, axis=1)
        keep = nrm > 1e-6
        if keep.sum() < 1:
            out.append(K.unit(Fc))                     # coverage fallback
        else:
            out.append(Fp[keep] / nrm[keep, None])
    return out


# --- Experiment 3 (Batch 3): debiasing G1, accel G2, fusion F, re-rank R1/R2 ----

def exp3(ctx: dict, sub: dict) -> dict:
    print("\n=== EXPERIMENT 3 (Batch 3) — debias / accel / fusion / k-reciprocal ===")
    ch = load_channels()
    out = {}

    # base fuse(centered, velocity) — the R1 substrate
    D_cv = K.ecdf_fuse([ch["app_centered"], ch["vel"]])
    out["fuse_cv"] = report(ctx, "fuse_centered+vel", D_cv, "R1 substrate")

    # R1 — k-reciprocal re-rank of fuse(centered, velocity)
    D_r1 = K.re_ranking(D_cv, k1=20, k2=6, lam=0.3)
    out["R1"] = report(ctx, "R1_krecip(fuse_cv)", D_r1,
                       "Zhong2017 k1=20,k2=6,λ=0.3")

    # G1 — centered endpoint-debiased appearance
    D_g1 = K.soft_chamfer_matrix(g1_debiased_centered(sub))
    out["G1"] = report(ctx, "G1_debiased_centered", D_g1, "project out {eA,eB} plane")

    # G2 — acceleration (already cached as 'accel')
    out["G2"] = report(ctx, "G2_accel", ch["accel"], "2nd-difference dynamics")

    # F — equal-weight fusion of {centered, velocity} ∪ hubness-passing {G1, G2}
    members = {"app_centered": ch["app_centered"], "vel": ch["vel"]}
    for nm, D in [("G1", D_g1), ("G2", ch["accel"])]:
        r = H.run(ctx, nm, D, quiet=True)
        if r["hubness"]["pass"]:
            members[nm] = D
        else:
            print(f"   [F] excluding {nm} — fails hubness standalone")
    D_F = K.ecdf_fuse(list(members.values()))
    out["F"] = report(ctx, "F_" + "+".join(members), D_F,
                      "4-channel ECDF fusion (hubness-gated members)")

    # R2 — k-reciprocal re-rank of F (if F >= fuse_cv)
    if out["F"]["result"]["accuracy_1nn"] >= out["fuse_cv"]["result"]["accuracy_1nn"]:
        D_r2 = K.re_ranking(D_F, k1=20, k2=6, lam=0.3)
        out["R2"] = report(ctx, "R2_krecip(F)", D_r2, "re-rank of F")
    return out


def _diff_cores(sub, order, horizon):
    feats, msk = sub["feats"], sub["mask_sided"]
    npre, nsuf = sub["n_prefix"], sub["n_suffix"]
    return [K.diff_sets(feats[i], msk[i], int(npre[i]), int(nsuf[i]),
                        order=order, horizon=horizon) for i in range(len(feats))]


def make_sub(feats: np.ndarray, base: dict) -> dict:
    """A substrate dict with new per-frame feats but the base-space masks / endpoint
    indices / sidedness reused verbatim (fable's spec for the large track: do NOT
    recompute masks in the new backbone's similarity scale)."""
    s = dict(base)
    s["feats"] = np.asarray(feats, dtype=np.float32)
    return s


def stack_pipeline(ctx: dict, sub: dict, tag: str) -> dict:
    """Fable Batch-4 recipe on ANY substrate: 6 channels -> balanced App/Dyn
    composites -> D_STACK -> k-reciprocal D_FINAL. Used for both CLS and large."""
    print(f"\n=== STACK PIPELINE [{tag}] — 6-channel balanced stack + re-rank ===")
    C_raw = H.cores(sub, "sided")
    out = {}

    P1 = K.soft_chamfer_matrix(K.apply_transform(C_raw, K.center_transform(C_raw)))
    P2 = K.soft_chamfer_matrix(g1_debiased_centered(sub))
    Vsets = _diff_cores(sub, 1, 1)
    V1 = K.soft_chamfer_matrix(Vsets)
    t0 = time.time(); V1e = K.emd_matrix_parallel(Vsets)
    print(f"[{tag} V1e velocity-EMD] {time.time()-t0:.1f}s")
    V4 = K.soft_chamfer_matrix(_diff_cores(sub, 1, 4))
    A2 = K.soft_chamfer_matrix(_diff_cores(sub, 2, 1))
    feats, msk = sub["feats"], sub["mask_sided"]
    npre, nsuf = sub["n_prefix"], sub["n_suffix"]
    Vspeed = K.wasserstein1d_matrix(
        [K.speed_set(feats[i], msk[i], int(npre[i]), int(nsuf[i]))
         for i in range(len(feats))])

    named = {"P1_centered": P1, "P2_debiased": P2, "V1_velocity": V1,
             "V1e_vel_emd": V1e, "V4_horizon": V4, "A2_accel": A2,
             "Vspeed": Vspeed}
    passes = {}
    for nm, D in named.items():
        res = report(ctx, f"{tag}:{nm}", D)
        passes[nm] = res["result"]["hubness"]["pass"]
        out[nm] = res

    app = [D for nm, D in [("P1_centered", P1), ("P2_debiased", P2)] if passes[nm]]
    dyn = [D for nm, D in [("V1_velocity", V1), ("V1e_vel_emd", V1e),
                           ("V4_horizon", V4), ("A2_accel", A2),
                           ("Vspeed", Vspeed)] if passes[nm]]
    D_STACK = 0.5 * K.ecdf_compose(app) + 0.5 * K.ecdf_compose(dyn)
    np.fill_diagonal(D_STACK, 0.0)
    out["D_STACK"] = report(ctx, f"D_STACK_{tag}", D_STACK,
                            "0.5*App(P1,P2)+0.5*Dyn(V1,V1e,V4,A2)")
    D_FINAL = K.re_ranking(D_STACK, k1=20, k2=6, lam=0.3)
    out["D_FINAL"] = report(ctx, f"D_FINAL_{tag}", D_FINAL, "k-reciprocal re-rank")
    np.savez(H.paths.WB_CACHE / "search" / f"stack_{tag}.npz",
             D_STACK=D_STACK, D_FINAL=D_FINAL)
    return out


LARGE_FEATS = H.paths.WB_CACHE / "search" / "large_feats.npz"


def exp5_large(ctx: dict, sub_base: dict) -> dict:
    """The GPU-track stack: same recipe in DINOv2-large space, base masks reused."""
    z = np.load(LARGE_FEATS, allow_pickle=True)
    assert list(z["keys"]) == list(ctx["keys"]), "large key order mismatch"
    large = make_sub(z["feats"], sub_base)
    print(f"[large] feats {z['feats'].shape} model {z['model']}")
    return stack_pipeline(ctx, large, "large")


def exp4(ctx: dict, sub: dict, channels: dict | None = None,
         tag: str = "cls") -> dict:
    """channels: optional {P1,P2,V1,V4,A2} distance matrices to override the CLS
    cache (used by the DINOv2-large GPU track). V1e (velocity-EMD) is computed here
    from the velocity sets unless supplied."""
    print(f"\n=== EXPERIMENT 4 (Batch 4, {tag}) — 6-channel balanced stack + re-rank ===")
    ch = channels or load_channels()
    out = {}

    P1 = ch["app_centered"]
    P2 = ch.get("G1") if channels else K.soft_chamfer_matrix(g1_debiased_centered(sub))
    V1 = ch["vel"]
    V4 = ch["horizon4"]
    A2 = ch["accel"]
    Vsets = ch.get("_vel_sets") if channels else _diff_cores(sub, 1, 1)
    t0 = time.time()
    V1e = ch.get("V1e") if channels else K.emd_matrix_parallel(Vsets)
    print(f"[V1e velocity-EMD] {time.time()-t0:.1f}s")

    named = {"P1_centered": P1, "P2_debiased": P2, "V1_velocity": V1,
             "V1e_vel_emd": V1e, "V4_horizon": V4, "A2_accel": A2}
    # per-channel exam + hubness gate (label-free inclusion rule)
    passes = {}
    for nm, D in named.items():
        res = report(ctx, nm, D)
        passes[nm] = res["result"]["hubness"]["pass"]
        out[nm] = res

    app_members = [D for nm, D in [("P1_centered", P1), ("P2_debiased", P2)] if passes[nm]]
    dyn_members = [D for nm, D in [("V1_velocity", V1), ("V1e_vel_emd", V1e),
                                   ("V4_horizon", V4), ("A2_accel", A2)] if passes[nm]]
    App = K.ecdf_compose(app_members)
    Dyn = K.ecdf_compose(dyn_members)
    D_STACK = 0.5 * App + 0.5 * Dyn
    np.fill_diagonal(D_STACK, 0.0)
    out["D_STACK"] = report(ctx, f"D_STACK_{tag}", D_STACK,
                            "0.5*App(P1,P2) + 0.5*Dyn(V1,V1e,V4,A2), re-ECDF composites")

    D_FINAL = K.re_ranking(D_STACK, k1=20, k2=6, lam=0.3)
    out["D_FINAL"] = report(ctx, f"D_FINAL_{tag}", D_FINAL,
                            "k-reciprocal re-rank of D_STACK")
    # persist the two headline matrices for the record / GPU-track reuse
    np.savez(H.paths.WB_CACHE / "search" / f"stack_{tag}.npz",
             D_STACK=D_STACK, D_FINAL=D_FINAL)
    return out


def _velocity_cores(sub: dict) -> list[np.ndarray]:
    feats, msk = sub["feats"], sub["mask_sided"]
    npre, nsuf = sub["n_prefix"], sub["n_suffix"]
    return [K.velocity_sets(feats[i], msk[i], int(npre[i]), int(nsuf[i]))
            for i in range(len(feats))]


# --- Experiment 2 (refuted branch, pre-declared): C4 IDF, C5 energy, C3 dyn ----

def exp2(ctx: dict, sub: dict) -> dict:
    print("\n=== EXPERIMENT 2 — distributional + dynamics (raw space, pre-declared) ===")
    C_raw = H.cores(sub, "sided")
    out = {}

    # C4 — corpus-IDF weighted Chamfer
    t0 = time.time()
    W = K.idf_weights(C_raw)
    D_idf = K.weighted_soft_chamfer_matrix(C_raw, W)
    print(f"[build C4 idf] {time.time()-t0:.1f}s")
    out["C4_idf"] = report(ctx, "C4_idf_chamfer_raw", D_idf,
                           "corpus-genericity-weighted soft-Chamfer")

    # C5 — energy distance
    t0 = time.time()
    D_en = K.energy_matrix(C_raw)
    print(f"[build C5 energy] {time.time()-t0:.1f}s")
    out["C5_energy"] = report(ctx, "C5_energy_raw", D_en,
                              "energy distance, Euclidean ground cost")

    # C3 — velocity channel + ECDF fusion with the best appearance so far
    t0 = time.time()
    Cv = _velocity_cores(sub)
    D_vel = K.soft_chamfer_matrix(Cv)
    print(f"[build C3 velocity] {time.time()-t0:.1f}s  "
          f"(median |V|={int(np.median([len(v) for v in Cv]))})")
    out["C3_vel"] = report(ctx, "C3_velocity_alone", D_vel,
                           "soft-Chamfer on unit embedding-velocities")

    D_m1a = K.soft_chamfer_matrix(C_raw)
    D_fuse_raw = K.ecdf_fuse([D_m1a, D_vel])
    out["C3_fuse_raw"] = report(ctx, "C3_fuse_m1a+vel", D_fuse_raw,
                                "ECDF-avg fusion of m1a(raw) and velocity")

    C_c = K.apply_transform(C_raw, K.center_transform(C_raw))
    D_c = K.soft_chamfer_matrix(C_c)
    D_fuse_c = K.ecdf_fuse([D_c, D_vel])
    out["C3_fuse_centered"] = report(ctx, "C3_fuse_centered+vel", D_fuse_c,
                                     "ECDF-avg fusion of centered-chamfer and velocity")
    return out


if __name__ == "__main__":
    import sys
    ctx = H.load_context()
    sub = H.load_substrate(ctx)
    which = sys.argv[1] if len(sys.argv) > 1 else "exp1"
    if which == "exp1":
        exp1(ctx, sub)
    elif which == "exp2":
        exp2(ctx, sub)
    elif which == "both":
        exp1(ctx, sub); exp2(ctx, sub)
