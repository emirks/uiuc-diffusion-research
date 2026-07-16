"""Corpus-reference statistics + v4 metric kernels (M1a=S3, M1b=D_ZPR, M1c=CSLS).

v4's three transfer metrics are corpus-relative: each maps raw pair measurements
through reference statistics fitted ONCE on the pinned 223-clip corpus — ECDF
populations over the 24,753 unordered pairwise distances (per channel and per
fused composite), the pooled-core centering mean mu, and the CSLS neighborhood
means r_j. Per SPEC §4 these are *pinned instrument constants*: built by
`build_reference` from the pinned corpus, frozen as a committed artifact
(REFERENCE_PATH), stamped by corpus hash + artifact sha256 (versioning.PINS),
and changed only via a version bump + re-certification.

Layering: this module implements the pair kernels and reference machinery; the
deployed per-item metrics in m1_transfer.py and the certification exam BOTH
import from here (the exam never reimplements a statistic).

Provenance: the constructions are ports of the health-validated metric-search
deliverables (workbench/search REPORT_{m1a,m1b,m1c,health_validation}.md —
aliases S3, D_ZPR, CSLS). Port parity vs the validated matrices was a blocking
pre-freeze gate of the v4.0.0 certification campaign.

Orientation convention: every function here returns DISTANCES (lower = more
similar). m1_transfer.py owns the headline orientation of reported fields.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from .appearance import set_similarity
from .morph import dtw_distance, resample_curve, znorm

REFERENCE_PATH = pathlib.Path(__file__).parent / "reference_v4.npz"

K_CSLS = 10                      # CSLS neighborhood size (pinned to the exam hubness k)
N_STEPS = 64                     # trajectory resample length (incumbent m1b convention)
S3_APP_WEIGHT = 0.5              # S3 = w*App + (1-w)*Dyn, frozen equal weights

# RMS radius of the 20x20 CoTracker query grid on [0,1]^2 (geometric constant of
# the track substrate; converts dlog_scale/dtheta into typical-point displacement).
_g = np.stack(np.meshgrid((np.arange(20) + 0.5) / 20,
                          (np.arange(20) + 0.5) / 20), -1).reshape(-1, 2)
RGRID = float(np.sqrt(((_g - _g.mean(0)) ** 2).sum(1).mean()))
del _g


def _unit(F: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    F = np.asarray(F, dtype=np.float64)
    n = np.linalg.norm(F, axis=1, keepdims=True)
    return F / np.maximum(n, eps)


# ---------------------------------------------------------------------------
# M1a channel representations (per clip)
# ---------------------------------------------------------------------------

def pooled_core_mean(core_sets: list[np.ndarray]) -> np.ndarray:
    """mu: mean over the pooled corpus sided-core CLS frames (float64)."""
    return np.concatenate([np.asarray(c, dtype=np.float64) for c in core_sets],
                          axis=0).mean(axis=0)


def centered_set(core_feats: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """P1 representation: core frames minus mu, re-unit-normalized."""
    return _unit(np.asarray(core_feats, dtype=np.float64) - mu)


def endpoint_anchors(feats: np.ndarray, n_prefix: int, n_suffix: int,
                     eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """e_A, e_B: unit-normalized mean of the first n_prefix / last n_suffix frames.
    Computed in the input dtype (float32 cache features) — the validated search
    construction; do not upcast before the mean."""
    feats = np.asarray(feats)
    eA = feats[:n_prefix].mean(axis=0)
    eA = eA / (np.linalg.norm(eA) + eps)
    eB = feats[-n_suffix:].mean(axis=0) if n_suffix else eA
    eB = eB / (np.linalg.norm(eB) + eps)
    return eA, eB


def debiased_set(core_feats: np.ndarray, feats: np.ndarray, n_prefix: int,
                 n_suffix: int, mu: np.ndarray) -> np.ndarray:
    """P2 representation: centered core frames with the rank-2 endpoint plane
    span{e_A - mu, e_B - mu} projected out, renormalized; rows that vanish are
    dropped; empty clip falls back to the centered representation."""
    eA, eB = endpoint_anchors(feats, n_prefix, n_suffix)
    eAc, eBc = eA - mu, eB - mu
    u1 = eAc / (np.linalg.norm(eAc) + 1e-12)
    u2 = eBc - (eBc @ u1) * u1
    n2 = np.linalg.norm(u2)
    B = np.stack([u1, u2 / n2]) if n2 > 1e-6 else u1[None]
    Fc = np.asarray(core_feats, dtype=np.float64) - mu
    Fp = Fc - (Fc @ B.T) @ B
    nrm = np.linalg.norm(Fp, axis=1)
    keep = nrm > 1e-6
    if keep.sum() < 1:
        return _unit(Fc)                              # coverage fallback
    return Fp[keep] / nrm[keep, None]


def velocity_set(feats: np.ndarray, core_mask: np.ndarray, n_prefix: int,
                 n_suffix: int, eps: float = 1e-6) -> np.ndarray:
    """V1/V1e representation: unit-normalized consecutive CLS differences
    f_{t+1}-f_t fully inside the unconditioned window, kept at core positions
    (mask on the earlier frame). Coverage guards: empty core set -> all window
    velocities; all-below-eps -> raw window velocities. Differencing runs in the
    input dtype (float32 cache features) — the validated search construction;
    the final unit-normalization upcasts to float64."""
    feats = np.asarray(feats)
    T = len(feats)
    win = np.zeros(T, dtype=bool)
    win[n_prefix:T - n_suffix] = True
    v = feats[1:] - feats[:-1]
    vt_win = win[:-1] & win[1:]
    keep = vt_win & np.asarray(core_mask, dtype=bool)[:-1]
    if keep.sum() < 1:
        keep = vt_win
    V = v[keep]
    nv = np.linalg.norm(V, axis=1)
    V = V[nv >= eps]
    if len(V) < 1:
        V = v[vt_win]
    return _unit(V)


# ---------------------------------------------------------------------------
# Pair distances
# ---------------------------------------------------------------------------

def chamfer_distance(A: np.ndarray, B: np.ndarray) -> float:
    """1 - symmetric mean-of-max cosine — exactly the deployed set_similarity."""
    return 1.0 - float(set_similarity(A, B))


def emd_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Exact 1-Wasserstein between uniform empirical measures on unit-row sets,
    ground cost 1-cos (scipy HiGHS transportation LP; solver pinned in
    versioning.PINS). Degenerate one-row sets -> mean cost (exact)."""
    from scipy.optimize import linprog
    from scipy import sparse
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    ni, nj = len(A), len(B)
    if ni == 0 or nj == 0:
        return float("nan")
    if ni == 1 or nj == 1:
        return float((1.0 - A @ B.T).mean())
    cost = (1.0 - A @ B.T).ravel()
    a = np.full(ni, 1.0 / ni)
    b = np.full(nj, 1.0 / nj)
    rows = sparse.kron(sparse.eye(ni), np.ones((1, nj)))
    cols = sparse.kron(np.ones((1, ni)), sparse.eye(nj)).tocsr()
    Aeq = sparse.vstack([rows, cols[:-1]]).tocsr()
    beq = np.concatenate([a, b[:-1]])
    res = linprog(cost, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
    return float(res.fun) if res.success else float("nan")


# ---------------------------------------------------------------------------
# M1b view representations (per clip) — Z / P / R over the deployed camera fit
# ---------------------------------------------------------------------------

def traj_view(params: np.ndarray, valid: np.ndarray, scheme: str) -> np.ndarray:
    """[N_STEPS, 4] resampled camera trajectory under a commensuration scheme.
    Inherits the incumbent preprocessing exactly (zero-invalid -> x*len ->
    resample). Z: per-channel z-norm (trajectory SHAPE). P: physical units, with
    dlog_scale/dtheta scaled by RGRID (AMPLITUDE)."""
    cols = []
    for c in (0, 1, 2, 3):
        # keep params' native dtype (float32 from the deployed camera fit) through
        # the per-duration scaling — matches deployed camera_match AND the validated
        # workbench Z/P views bit-for-bit (resample_curve upcasts via np.interp).
        x = params[:, c].copy()
        x[~valid] = 0.0
        x = x * len(x)
        cols.append(resample_curve(x, N_STEPS))
    A = np.stack(cols, axis=1)
    if scheme == "Z":
        A = np.stack([znorm(A[:, i]) for i in range(A.shape[1])], axis=1)
    elif scheme == "P":
        A = A.copy()
        A[:, 2] = A[:, 2] * RGRID
        A[:, 3] = A[:, 3] * RGRID
    else:
        raise ValueError(scheme)
    return A


def residual_energy_view(tracks: np.ndarray, vis: np.ndarray,
                         cam: dict) -> tuple[np.ndarray, bool]:
    """R view: [N_STEPS, 1] resampled per-step non-rigid residual-energy fraction
    (the share of point motion the rigid similarity fit can't explain — effect
    turbulence), plus the cam-valid flag. Steps with an invalid fit contribute 0.
    Construction pinned from the validated search deliverable (REPORT_m1b §4)."""
    from .m1_transfer import _smooth_tracks
    tr = _smooth_tracks(tracks)
    T = len(tr)
    e = np.zeros(T - 1)
    for s in range(T - 1):
        if not cam["valid"][s]:
            continue
        ok = (vis[s] >= 0.5) & (vis[s + 1] >= 0.5)
        P, Q = tr[s][ok], tr[s + 1][ok]
        if len(P) < 1:
            continue
        pred = P @ cam["Ms"][s].T + cam["ts"][s]
        num = float(((Q - pred) ** 2).sum())
        den = float(((Q - P) ** 2).sum())
        e[s] = float(np.clip(num / den, 0.0, 1.0)) if den > 1e-12 else 0.0
    return resample_curve(e, N_STEPS)[:, None], bool(np.asarray(cam["valid"]).mean() > 0.5)


# ---------------------------------------------------------------------------
# ECDF machinery
# ---------------------------------------------------------------------------

def ecdf_rank_matrix(D: np.ndarray) -> np.ndarray:
    """Map each off-diagonal entry to its ECDF value over the off-diagonal finite
    upper-triangle population (global monotone, symmetry-preserving rank
    transform); NaN preserved, diagonal 0. Verbatim semantics of the validated
    search transform."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    vals = D[iu, ju].astype(float)
    finite = np.isfinite(vals)
    out = np.full_like(vals, np.nan)
    order = np.argsort(vals[finite], kind="mergesort")
    ranks = np.empty(order.shape[0])
    ranks[order] = (np.arange(order.shape[0]) + 1) / order.shape[0]
    out[finite] = ranks
    M = np.zeros((n, n))
    M[iu, ju] = out
    M[ju, iu] = out
    return M


def ecdf_fuse_matrices(mats: list[np.ndarray]) -> np.ndarray:
    """Equal-weight average of ECDF-mapped matrices; NaN wherever any input is
    non-finite; diagonal 0."""
    E = [ecdf_rank_matrix(m) for m in mats]
    out = np.zeros_like(E[0])
    for e in E:
        out += e / len(E)
    nan_mask = np.zeros_like(out, dtype=bool)
    for m in mats:
        nan_mask |= ~np.isfinite(np.asarray(m, dtype=float))
    out[nan_mask] = np.nan
    np.fill_diagonal(out, 0.0)
    return out


def ecdf_compose_matrices(mats: list[np.ndarray]) -> np.ndarray:
    """re-ECDF( mean( ECDF(m) ) ) — rank-space composite in [0,1]."""
    return ecdf_rank_matrix(ecdf_fuse_matrices(mats))


def population(D: np.ndarray) -> np.ndarray:
    """Sorted finite off-diagonal upper-triangle values — the frozen reference
    population an out-of-corpus measurement is ranked against."""
    D = np.asarray(D, dtype=float)
    iu, ju = np.triu_indices(D.shape[0], k=1)
    vals = D[iu, ju]
    return np.sort(vals[np.isfinite(vals)])


def ecdf_lookup(pop: np.ndarray, x: float) -> tuple[float, bool]:
    """Percentile of x against a frozen sorted population: P(pop <= x), i.e.
    searchsorted-right / n — identical to ecdf_rank_matrix for in-population
    values (tie-free populations; asserted at build). Returns (percentile,
    saturated) where saturated flags x outside the fitted support (percentile
    clipped at 0 or 1) — reference populations were fitted on real-corpus pairs;
    behavior outside their support is flagged, not certified (SPEC §6.5)."""
    if not np.isfinite(x):
        return float("nan"), False
    n = len(pop)
    idx = int(np.searchsorted(pop, x, side="right"))
    sat = bool(x < pop[0] or x > pop[-1])
    return idx / n, sat


# ---------------------------------------------------------------------------
# CSLS (M1c) — de-hubbed object-motion distance
# ---------------------------------------------------------------------------

def csls_matrix(D: np.ndarray, k: int = K_CSLS) -> np.ndarray:
    """CSLS de-hubbing of a distance matrix (Conneau et al. 2018): S = 1-D;
    r_i = mean of i's k highest finite similarities; CSLS = 2S - r_i - r_j,
    returned as the distance -CSLS. NaN preserved, diagonal 0."""
    D = np.asarray(D, dtype=float)
    nan = ~np.isfinite(D)
    S = 1.0 - D
    n = S.shape[0]
    Sw = S.copy()
    np.fill_diagonal(Sw, -np.inf)
    Sw[nan] = -np.inf
    r = np.empty(n)
    for i in range(n):
        row = Sw[i][np.isfinite(Sw[i])]
        r[i] = np.mean(np.sort(row)[-k:]) if len(row) >= 1 else 0.0
    out = -(2.0 * S - r[:, None] - r[None, :])
    out[nan] = np.nan
    np.fill_diagonal(out, 0.0)
    return out


def csls_r(sims: np.ndarray, k: int = K_CSLS) -> float:
    """Neighborhood mean r for one clip given its similarities to the reference
    corpus (NaN entries ignored; the clip itself must not be included)."""
    row = np.asarray(sims, dtype=float)
    row = row[np.isfinite(row)]
    return float(np.mean(np.sort(row)[-k:])) if len(row) >= 1 else 0.0


def csls_distance(s: float, r_a: float, r_b: float) -> float:
    """Pair CSLS distance from a raw similarity and the two neighborhood means."""
    if not np.isfinite(s):
        return float("nan")
    return -(2.0 * s - r_a - r_b)


# ---------------------------------------------------------------------------
# Corpus matrix builders (exam + reference build; import-deployed throughout)
# ---------------------------------------------------------------------------

def m1a_channel_matrices(feats: list[np.ndarray], core_masks: list[np.ndarray],
                         n_prefix: list[int], n_suffix: list[int],
                         n_jobs: int = 8) -> dict:
    """The four S3 channel distance matrices over a clip set. Serial chamfer
    (fast); fork-parallel exact EMD (bit-identical to serial)."""
    cores = [np.asarray(f, dtype=np.float64)[m] for f, m in zip(feats, core_masks)]
    mu = pooled_core_mean(cores)
    cen = [centered_set(c, mu) for c in cores]
    deb = [debiased_set(c, f, int(p), int(s), mu)
           for c, f, p, s in zip(cores, feats, n_prefix, n_suffix)]
    vel = [velocity_set(f, m, int(p), int(s))
           for f, m, p, s in zip(feats, core_masks, n_prefix, n_suffix)]

    def _cham(sets):
        n = len(sets)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = chamfer_distance(sets[i], sets[j])
        return D

    P1 = _cham(cen)
    P2 = _cham(deb)
    V1 = _cham(vel)
    V1e = _emd_matrix_parallel(vel, n_jobs=n_jobs)
    return {"P1": P1, "P2": P2, "V1": V1, "V1e": V1e, "mu": mu}


_EMD_STATE: dict = {}


def _emd_pair_idx(pair):
    i, j = pair
    C = _EMD_STATE["C"]
    return emd_distance(C[i], C[j])


def _emd_matrix_parallel(sets: list[np.ndarray], n_jobs: int = 8) -> np.ndarray:
    import concurrent.futures
    import itertools
    import multiprocessing
    C = [np.ascontiguousarray(c, dtype=np.float64) for c in sets]
    n = len(C)
    pairs = list(itertools.combinations(range(n), 2))
    _EMD_STATE["C"] = C
    try:
        ctx = multiprocessing.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs,
                                                    mp_context=ctx) as ex:
            vals = list(ex.map(_emd_pair_idx, pairs, chunksize=64))
    finally:
        _EMD_STATE.clear()
    D = np.zeros((n, n))
    for (i, j), v in zip(pairs, vals):
        D[i, j] = D[j, i] = v
    return D


def m1b_view_matrices(tracks: np.ndarray, vis: np.ndarray) -> dict:
    """Z / P / R view distance matrices from CoTracker tracklets via the deployed
    camera fit. NaN wherever either clip's fit is not cam-valid (>50% valid
    steps) — coverage identical to the incumbent m1b."""
    from .m1_transfer import camera_trajectory
    n = len(tracks)
    cams = [camera_trajectory(tracks[i], vis[i]) for i in range(n)]
    cv = [bool(np.asarray(c["valid"]).mean() > 0.5) for c in cams]
    Zf = [traj_view(c["params"], c["valid"], "Z") for c in cams]
    Pf = [traj_view(c["params"], c["valid"], "P") for c in cams]
    Rf = [residual_energy_view(tracks[i], vis[i], cams[i])[0] for i in range(n)]

    def _dtw(feats_list):
        D = np.full((n, n), np.nan)
        for i in range(n):
            if not cv[i]:
                continue
            D[i, i] = 0.0
            for j in range(i + 1, n):
                if cv[j]:
                    D[i, j] = D[j, i] = dtw_distance(feats_list[i], feats_list[j])
        return D

    return {"Z": _dtw(Zf), "P": _dtw(Pf), "R": _dtw(Rf), "cam_valid": np.array(cv)}


def s3_matrix(P1, P2, V1, V1e) -> np.ndarray:
    """M1a deliverable: 0.5*App(P1,P2) + 0.5*Dyn(V1e,V1), each an ecdf_compose."""
    D = (S3_APP_WEIGHT * ecdf_compose_matrices([P1, P2])
         + (1.0 - S3_APP_WEIGHT) * ecdf_compose_matrices([V1e, V1]))
    np.fill_diagonal(D, 0.0)
    return D


def dzpr_matrix(Z, P, R) -> np.ndarray:
    """M1b deliverable: equal-weight ECDF fusion of the three camera views."""
    return ecdf_fuse_matrices([Z, P, R])


# ---------------------------------------------------------------------------
# Reference artifact: build / save / load / verify
# ---------------------------------------------------------------------------

def build_reference_from_parts(channels: dict, views: dict, object_D: np.ndarray,
                               keys: list[str], corpus_sha: str) -> dict:
    """Assemble the reference artifact from already-built corpus matrices (the
    channel matrices P1/P2/V1/V1e + mu, the view matrices Z/P/R, and the deployed
    1-object_match distance matrix). Cheap: only population() extraction + the
    CSLS neighborhood means. The certify exam reuses this with the matrices it
    already computed (no second EMD pass)."""
    a, b = channels, views
    fuse_app = ecdf_fuse_matrices([a["P1"], a["P2"]])
    fuse_dyn = ecdf_fuse_matrices([a["V1e"], a["V1"]])

    S = 1.0 - np.asarray(object_D, dtype=float)
    nan = ~np.isfinite(object_D)
    Sw = S.copy()
    np.fill_diagonal(Sw, -np.inf)
    Sw[nan] = -np.inf
    r_obj = np.array([csls_r(Sw[i][np.isfinite(Sw[i])]) for i in range(len(S))])

    ref = {
        "keys": np.array(keys),
        "corpus_sha": np.array(corpus_sha),
        "mu": a["mu"],
        "pop_P1": population(a["P1"]),
        "pop_P2": population(a["P2"]),
        "pop_V1": population(a["V1"]),
        "pop_V1e": population(a["V1e"]),
        "pop_App": population(fuse_app),
        "pop_Dyn": population(fuse_dyn),
        "pop_Z": population(b["Z"]),
        "pop_P": population(b["P"]),
        "pop_R": population(b["R"]),
        "r_obj": r_obj,
        "k_csls": np.array(K_CSLS),
        "rgrid": np.array(RGRID),
        "s3_app_weight": np.array(S3_APP_WEIGHT),
    }
    for k, v in ref.items():
        if k.startswith("pop_") and len(np.unique(v)) != len(v):
            dup = len(v) - len(np.unique(v))
            print(f"[reference] WARNING {k}: {dup} tied values "
                  f"(lookup vs rank may differ by ties/n on those cells)")
    return ref


def build_reference(feats, core_masks, n_prefix, n_suffix, tracks, vis,
                    object_D: np.ndarray, keys: list[str], corpus_sha: str,
                    n_jobs: int = 8) -> dict:
    """Build every pinned reference statistic from the pinned corpus (the
    canonical rebuild path — recomputes all channel/view matrices). object_D is
    the deployed 1-object_match distance matrix (from the certify exam's motion
    builder, which imports m1_transfer.object_match)."""
    a = m1a_channel_matrices(feats, core_masks, n_prefix, n_suffix, n_jobs=n_jobs)
    b = m1b_view_matrices(tracks, vis)
    return build_reference_from_parts(a, b, object_D, keys, corpus_sha)


def loo_m1a_reference(channels: dict, drop_idx: int) -> dict:
    """Leave-own-clip-out M1a reference (bar-2 robustness clause, advisor Q2):
    the M1a ECDF populations recomputed with corpus clip `drop_idx`'s row/column
    removed from every channel matrix, so re-scoring that clip cannot draw on the
    ~222 population pairs it contributes (worst-case in-sample ECDF inflation ≈0.9
    percentile pts — the leakage channel that could manufacture a bar-2 PASS in
    S3's percentile-compressed regime). Only the ECDF populations are masked; the
    centering mean μ is kept full-corpus (its 1/223 perturbation is not the leakage
    channel and is not outcome-coupled — disclosed). Returns exactly the keys
    m1a_pair reads, so the DEPLOYED lookup kernel is reused unchanged."""
    def drop(M):
        return np.delete(np.delete(np.asarray(M, float), drop_idx, 0), drop_idx, 1)
    P1, P2, V1, V1e = (drop(channels[k]) for k in ("P1", "P2", "V1", "V1e"))
    fuse_app = ecdf_fuse_matrices([P1, P2])
    fuse_dyn = ecdf_fuse_matrices([V1e, V1])
    return {
        "mu": channels["mu"],
        "pop_P1": population(P1), "pop_P2": population(P2),
        "pop_V1": population(V1), "pop_V1e": population(V1e),
        "pop_App": population(fuse_app), "pop_Dyn": population(fuse_dyn),
        "s3_app_weight": np.array(S3_APP_WEIGHT),
    }


def compare_reference(fresh: dict, committed: dict, tol: float = 1e-6) -> dict:
    """Rebuild-parity clause (SPEC §4/§7): a reference rebuilt from the pinned
    corpus must reproduce the committed artifact within tolerance. Returns per-
    array max abs deltas + an overall pass; keys/shape mismatches fail loudly."""
    out = {"per_array": {}, "pass": True, "mismatch": []}
    arrays = [k for k in fresh if k not in ("keys", "corpus_sha")]
    if [str(x) for x in fresh["keys"]] != [str(x) for x in committed["keys"]]:
        out["pass"] = False
        out["mismatch"].append("keys order differs")
    for k in arrays:
        if k not in committed:
            out["pass"] = False
            out["mismatch"].append(f"{k} missing from committed")
            continue
        fa, ca = np.asarray(fresh[k], float).ravel(), np.asarray(committed[k], float).ravel()
        if fa.shape != ca.shape:
            out["pass"] = False
            out["mismatch"].append(f"{k} shape {fa.shape} vs {ca.shape}")
            continue
        d = float(np.nanmax(np.abs(fa - ca))) if fa.size else 0.0
        out["per_array"][k] = d
        if d > tol:
            out["pass"] = False
            out["mismatch"].append(f"{k} max|Δ|={d:.2e} > {tol:.0e}")
    return out


def save_reference(ref: dict, path: pathlib.Path = REFERENCE_PATH) -> str:
    np.savez_compressed(path, **ref)
    return sha256_of(path)


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference(path: pathlib.Path = REFERENCE_PATH,
                   expect_corpus_sha: str | None = None) -> dict:
    """Load the frozen artifact; refuse a corpus mismatch loudly."""
    z = np.load(path, allow_pickle=False)
    ref = {k: z[k] for k in z.files}
    if expect_corpus_sha is not None:
        got = str(ref["corpus_sha"])
        if got != expect_corpus_sha:
            raise RuntimeError(
                f"reference_v4 artifact was built for corpus {got[:16]}… but the "
                f"loaded corpus hashes to {expect_corpus_sha[:16]}… — rebuild + "
                f"re-certify (SPEC §4/§7)")
    return ref


# ---------------------------------------------------------------------------
# Per-item pair scoring (consumed by m1_transfer.py; distances out)
# ---------------------------------------------------------------------------

def m1a_pair(gen_feats, gen_core, gen_pre, gen_suf,
             ref_feats, ref_core, ref_pre, ref_suf, R: dict) -> dict:
    """S3 distance for one (gen, ref) pair against the frozen reference."""
    mu = R["mu"]
    gc = np.asarray(gen_feats, dtype=np.float64)[gen_core]
    rc = np.asarray(ref_feats, dtype=np.float64)[ref_core]
    reps = {
        "P1": (centered_set(gc, mu), centered_set(rc, mu)),
        "P2": (debiased_set(gc, gen_feats, gen_pre, gen_suf, mu),
               debiased_set(rc, ref_feats, ref_pre, ref_suf, mu)),
    }
    vg = velocity_set(gen_feats, gen_core, gen_pre, gen_suf)
    vr = velocity_set(ref_feats, ref_core, ref_pre, ref_suf)
    raw = {k: chamfer_distance(a, b) for k, (a, b) in reps.items()}
    raw["V1"] = chamfer_distance(vg, vr)
    raw["V1e"] = emd_distance(vg, vr)

    sat = False
    pct = {}
    for k in ("P1", "P2", "V1", "V1e"):
        pct[k], s = ecdf_lookup(R[f"pop_{k}"], raw[k])
        sat |= s
    app_f = 0.5 * (pct["P1"] + pct["P2"])
    dyn_f = 0.5 * (pct["V1e"] + pct["V1"])
    app, sa = ecdf_lookup(R["pop_App"], app_f)
    dyn, sd = ecdf_lookup(R["pop_Dyn"], dyn_f)
    sat |= sa or sd
    w = float(R["s3_app_weight"])
    return {"s3": w * app + (1.0 - w) * dyn, "app": app, "dyn": dyn,
            "raw": raw, "saturated": bool(sat)}


def m1b_pair(gen_tracks, gen_vis, gen_cam, ref_tracks, ref_vis, ref_cam,
             R: dict) -> dict:
    """D_ZPR distance for one (gen, ref) pair. NaN (never imputed) unless both
    camera fits are cam-valid, matching the corpus coverage rule."""
    gv = bool(np.asarray(gen_cam["valid"]).mean() > 0.5)
    rv = bool(np.asarray(ref_cam["valid"]).mean() > 0.5)
    if not (gv and rv):
        return {"dzpr": float("nan"), "saturated": False, "cam_valid": False}
    out = {}
    sat = False
    for view, pop in (("Z", "pop_Z"), ("P", "pop_P")):
        d = dtw_distance(traj_view(gen_cam["params"], gen_cam["valid"], view),
                         traj_view(ref_cam["params"], ref_cam["valid"], view))
        out[view], s = ecdf_lookup(R[pop], d)
        sat |= s
    rg, _ = residual_energy_view(gen_tracks, gen_vis, gen_cam)
    rr, _ = residual_energy_view(ref_tracks, ref_vis, ref_cam)
    d = dtw_distance(rg, rr)
    out["R"], s = ecdf_lookup(R["pop_R"], d)
    sat |= s
    return {"dzpr": (out["Z"] + out["P"] + out["R"]) / 3.0, "views": out,
            "saturated": bool(sat), "cam_valid": True}


def m1c_pair(sim_gen_ref: float, sims_gen_corpus: np.ndarray, ref_key_index: int,
             R: dict) -> dict:
    """CSLS distance for one (gen, ref) pair: the gen clip's neighborhood mean is
    computed against the frozen 223-clip reference corpus; the reference clip's
    r_j is a frozen artifact constant."""
    r_gen = csls_r(sims_gen_corpus, int(R["k_csls"]))
    r_ref = float(R["r_obj"][ref_key_index])
    return {"csls": csls_distance(sim_gen_ref, r_gen, r_ref),
            "r_gen": r_gen, "r_ref": r_ref}
