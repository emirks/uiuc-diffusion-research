"""Reusable set-distance kernels + representation transforms for the metric search.

All builders take a list of per-clip frame arrays (each [n_i, d]) and return a
symmetric 223x223 distance matrix. Kept numpy-only and dependency-light so a whole
factorial of candidates runs in minutes on the login node.

soft_chamfer_matrix reproduces m1a's appearance.set_similarity EXACTLY on unit-row
inputs (asserted by the harness base-touch), so a candidate differs from the
incumbent only in the representation it is fed or the set-distance chosen — never in
a silently different implementation of the same idea.
"""

from __future__ import annotations

import numpy as np


def unit(F: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize rows so cosine == dot."""
    F = np.asarray(F, dtype=np.float64)
    n = np.linalg.norm(F, axis=1, keepdims=True)
    return F / np.maximum(n, eps)


# --- representation transforms (fit label-free on pooled corpus core frames) ---

def pooled(cores: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(c, dtype=np.float64) for c in cores], axis=0)


def center_transform(cores: list[np.ndarray]) -> callable:
    """C1b: subtract the pooled corpus mean, renormalize (common-mode / cone removal)."""
    mu = pooled(cores).mean(axis=0)
    return lambda F: unit(np.asarray(F, dtype=np.float64) - mu)


def lw_transform(cores: list[np.ndarray]) -> tuple[callable, dict]:
    """C1: Ledoit-Wolf ZCA whiten (parameter-free, no eigenvalue floor), renormalize."""
    from .. import lw
    art = lw.fit(pooled(cores))
    return (lambda F: unit(lw.whiten(art, np.asarray(F, dtype=np.float64)))), art


def apply_transform(cores: list[np.ndarray], tf: callable) -> list[np.ndarray]:
    return [tf(c) for c in cores]


# --- set distances -------------------------------------------------------------

def soft_chamfer_matrix(cores: list[np.ndarray]) -> np.ndarray:
    """D[i,j] = 1 - 0.5*(mean_a max_b cos + mean_b max_a cos). Unit rows -> cos=dot.

    Exactly m1a's kernel; per-pair matmul (no 14k x 14k Gram materialized)."""
    C = [np.ascontiguousarray(c, dtype=np.float64) for c in cores]
    n = len(C)
    D = np.zeros((n, n))
    for i in range(n):
        Ci = C[i]
        for j in range(i + 1, n):
            S = Ci @ C[j].T                       # cosine block
            sim = 0.5 * (S.max(axis=1).mean() + S.max(axis=0).mean())
            D[i, j] = D[j, i] = 1.0 - sim
    return D


def idf_weights(cores: list[np.ndarray]) -> list[np.ndarray]:
    """C4: per-frame corpus-genericity weight w_a = max(0, 1 - g_a), where g_a =
    mean cosine of frame a to all core frames of OTHER clips. Because cosine is a
    dot product, mean-cos-to-a-set = dot with that set's mean, so g_a is O(N*d) with
    no 14k x 14k Gram: g_a = (f_a . total - f_a . clipsum) / (N - n_clip)."""
    C = [np.ascontiguousarray(c, dtype=np.float64) for c in cores]
    clip_sum = [c.sum(axis=0) for c in C]
    n_clip = np.array([len(c) for c in C])
    total = np.sum(clip_sum, axis=0)
    N = int(n_clip.sum())
    W = []
    for c, cs, nc in zip(C, clip_sum, n_clip):
        denom = max(N - nc, 1)
        g = (c @ total - c @ cs) / denom          # mean cos to other-clip frames
        W.append(np.maximum(0.0, 1.0 - g))
    return W


def weighted_soft_chamfer_matrix(cores: list[np.ndarray],
                                 weights: list[np.ndarray]) -> np.ndarray:
    """m1a's soft-Chamfer with per-frame weights on the mean legs (C4)."""
    C = [np.ascontiguousarray(c, dtype=np.float64) for c in cores]
    Wn = []
    for w in weights:
        s = w.sum()
        Wn.append(w / s if s > 0 else np.full_like(w, 1.0 / len(w)))   # uniform fallback
    n = len(C)
    D = np.zeros((n, n))
    for i in range(n):
        Ci, wi = C[i], Wn[i]
        for j in range(i + 1, n):
            S = Ci @ C[j].T
            sim = 0.5 * (wi @ S.max(axis=1) + Wn[j] @ S.max(axis=0))
            D[i, j] = D[j, i] = 1.0 - sim
    return D


# --- C3: embedding-velocity (dynamics) channel + ECDF fusion ------------------

def velocity_sets(feats: np.ndarray, mask: np.ndarray, n_prefix: int, n_suffix: int,
                  eps: float = 1e-6) -> np.ndarray:
    """v_t = f_{t+1}-f_t on the unconditioned window, kept at core positions,
    unit-normalized. Fall back to all window velocities if the core set is empty.
    Differencing consecutive frames is NOT endpoint normalization (no anchor)."""
    T = len(feats)
    win = np.zeros(T, dtype=bool)
    win[n_prefix:T - n_suffix] = True
    v = feats[1:] - feats[:-1]                    # [T-1, d]; v[t] = f[t+1]-f[t]
    vt_win = win[:-1] & win[1:]                   # velocity fully inside the window
    keep = vt_win & mask[:-1]                     # core position = mask on earlier frame
    if keep.sum() < 1:
        keep = vt_win
    V = v[keep]
    nv = np.linalg.norm(V, axis=1)
    V = V[nv >= eps]
    if len(V) < 1:
        V = v[vt_win]                             # last-ditch coverage guard
    return unit(V)


def ecdf_map(D: np.ndarray) -> np.ndarray:
    """Map each off-diagonal entry to its ECDF value over off-diagonal finite
    entries — a global monotone, symmetry-preserving rank transform. Diagonal 0."""
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


def ecdf_fuse(mats: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Equal-weight (default) average of ECDF-mapped matrices. NaN if any input NaN."""
    E = [ecdf_map(m) for m in mats]
    w = weights or [1.0 / len(E)] * len(E)
    out = np.zeros_like(E[0])
    for wi, e in zip(w, E):
        out += wi * e
    # propagate NaN where any constituent was undefined
    nan_mask = np.zeros_like(out, dtype=bool)
    for m in mats:
        nan_mask |= ~np.isfinite(m)
    out[nan_mask] = np.nan
    np.fill_diagonal(out, 0.0)
    return out


# --- endpoint anchors + debiasing (owner-reopened branch) ---------------------

def endpoint_anchors(feats: np.ndarray, n_prefix: int, n_suffix: int,
                     eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """e_A, e_B exactly as certify.probes.endpoint_vecs: unit-normalized mean of the
    first n_prefix / last n_suffix CLS frames."""
    eA = feats[:n_prefix].mean(axis=0)
    eA = eA / (np.linalg.norm(eA) + eps)
    eB = feats[-n_suffix:].mean(axis=0) if n_suffix else eA
    eB = eB / (np.linalg.norm(eB) + eps)
    return eA, eB


def project_out(F: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the component of each row in span(basis) (basis rows = directions).
    Orthonormalize basis first; returns the residual (NOT renormalized)."""
    B = np.atleast_2d(np.asarray(basis, dtype=np.float64))
    Q, _ = np.linalg.qr(B.T)                      # columns orthonormal
    F = np.asarray(F, dtype=np.float64)
    return F - (F @ Q) @ Q.T


# --- higher-order / horizon dynamics ------------------------------------------

def diff_sets(feats: np.ndarray, mask: np.ndarray, n_prefix: int, n_suffix: int,
              order: int = 1, horizon: int = 1, eps: float = 1e-6) -> np.ndarray:
    """Unit-normalized k-th-order / horizon differences at core positions.
    order=1,horizon=1 -> velocity; order=2 -> acceleration; horizon=h -> f_{t+h}-f_t."""
    T = len(feats)
    win = np.zeros(T, dtype=bool)
    win[n_prefix:T - n_suffix] = True
    X = feats
    for _ in range(order):
        X = X[horizon:] - X[:-horizon]            # successive differencing
    step = order * horizon
    vt_win = win[:T - step] & win[step:]
    keep = vt_win & mask[:T - step]
    if keep.sum() < 1:
        keep = vt_win
    V = X[keep]
    nv = np.linalg.norm(V, axis=1)
    V = V[nv >= eps]
    if len(V) < 1:
        V = X[vt_win]
    return unit(V)


# --- hubness reduction (CSLS, mutual proximity) — de-sinking a distance matrix --

def csls(D: np.ndarray, k: int = 10) -> np.ndarray:
    """CSLS de-hubbing (Conneau et al. 2018). S=1-D similarity; r(i)=mean of i's k
    highest-S neighbors; CSLS[i,j]=2S[i,j]-r(i)-r(j); returned as a distance -CSLS.
    Penalizes any point similar to many others -> demotes a universal hub. k pinned
    to the exam's hubness k. NaN cells preserved."""
    nan = ~np.isfinite(D)
    S = 1.0 - np.asarray(D, dtype=float)
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


def mutual_proximity(D: np.ndarray) -> np.ndarray:
    """Empirical mutual proximity (Schnitzer et al. 2012). MP[i,j] = fraction of
    points farther from BOTH i and j than i,j are from each other; distance 1-MP.
    Parameter-free de-hubber. NaN cells preserved."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    nan = ~np.isfinite(D)
    Dw = D.copy()
    Dw[nan] = np.inf
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            dij = Dw[i, j]
            if not np.isfinite(dij):
                continue
            fi = Dw[i] > dij
            fj = Dw[j] > dij
            both = fi & fj
            both[i] = both[j] = False
            valid = np.isfinite(Dw[i]) & np.isfinite(Dw[j])
            valid[i] = valid[j] = False
            denom = valid.sum()
            mp = both.sum() / denom if denom > 0 else 0.0
            out[i, j] = out[j, i] = 1.0 - mp
    np.fill_diagonal(out, 0.0)
    return out


def dominant_sink(D: np.ndarray, labels) -> tuple:
    """The class that is the 1-NN for the most clips (the 'sink'), and its share."""
    import collections
    labels = np.asarray(labels)
    M = D.copy().astype(float)
    np.fill_diagonal(M, np.inf)
    M[np.isnan(M)] = np.inf
    ok = np.isfinite(M).any(1)
    preds = [labels[int(np.argmin(M[i]))] for i in range(len(labels)) if ok[i]]
    c = collections.Counter(preds)
    top, cnt = c.most_common(1)[0]
    return top, cnt / len(preds)


# --- self-tuning local scaling (Zelnik-Manor & Perona 2004; published k) --------

def local_scaling(D: np.ndarray, k: int = 7) -> np.ndarray:
    """D_ls[i,j] = D[i,j] / sqrt(sigma_i sigma_j), sigma_i = distance to i's k-th NN.
    Label-free (sigma from D's own geometry); shrinks sparse-region (small-class)
    distances so true classmates beat dense big-class intruders. k=7 published."""
    M = np.asarray(D, dtype=float).copy()
    n = M.shape[0]
    off = M.copy()
    np.fill_diagonal(off, np.inf)
    off[np.isnan(off)] = np.inf
    sig = np.empty(n)
    for i in range(n):
        finite = off[i][np.isfinite(off[i])]
        if len(finite) == 0:
            sig[i] = 1.0
        else:
            kk = min(k, len(finite)) - 1
            sig[i] = np.partition(finite, kk)[kk]        # k-th smallest (0-indexed k-1)
    sig = np.maximum(sig, 1e-12)
    scale = np.sqrt(np.outer(sig, sig))
    out = D / scale
    np.fill_diagonal(out, 0.0)
    return out


# --- k-reciprocal re-ranking (Zhong et al. CVPR 2017; published constants) ------

def re_ranking(D: np.ndarray, k1: int = 20, k2: int = 6, lam: float = 0.3) -> np.ndarray:
    """k-reciprocal re-ranking on a full symmetric distance matrix (every clip is
    both query and gallery). Canonical algorithm, constants frozen from the paper
    (k1=20, k2=6, λ=0.3). Output symmetrized (D*+D*ᵀ)/2. Label-free."""
    orig = np.asarray(D, dtype=np.float32).copy()
    orig[~np.isfinite(orig)] = orig[np.isfinite(orig)].max()
    n = orig.shape[0]
    # normalize columns as in the reference implementation
    od = np.transpose(orig / np.max(orig, axis=0))
    V = np.zeros_like(od, dtype=np.float32)
    initial_rank = np.argsort(od, axis=1).astype(np.int32)

    for i in range(n):
        fwd = initial_rank[i, :k1 + 1]
        bwd = initial_rank[fwd, :k1 + 1]
        fi = np.where(bwd == i)[0]
        krecip = fwd[fi]
        krecip_exp = krecip.copy()
        for cand in krecip:
            c_fwd = initial_rank[cand, :int(round(k1 / 2)) + 1]
            c_bwd = initial_rank[c_fwd, :int(round(k1 / 2)) + 1]
            fic = np.where(c_bwd == cand)[0]
            c_krecip = c_fwd[fic]
            if len(np.intersect1d(c_krecip, krecip)) > 2.0 / 3.0 * len(c_krecip):
                krecip_exp = np.append(krecip_exp, c_krecip)
        krecip_exp = np.unique(krecip_exp)
        w = np.exp(-od[i, krecip_exp])
        V[i, krecip_exp] = w / np.sum(w)

    if k2 != 1:
        Vqe = np.zeros_like(V, dtype=np.float32)
        for i in range(n):
            Vqe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = Vqe

    invIndex = [np.where(V[:, i] != 0)[0] for i in range(n)]
    jaccard = np.zeros_like(od, dtype=np.float32)
    for i in range(n):
        tmp = np.zeros((1, n), dtype=np.float32)
        nz = np.where(V[i, :] != 0)[0]
        imgs = [invIndex[ind] for ind in nz]
        for j in range(len(nz)):
            tmp[0, imgs[j]] += np.minimum(V[i, nz[j]], V[imgs[j], nz[j]])
        jaccard[i] = 1 - tmp / (2.0 - tmp)

    final = jaccard * (1 - lam) + od * lam
    final = 0.5 * (final + final.T)              # symmetrize
    np.fill_diagonal(final, 0.0)
    return final.astype(np.float64)


# --- V-speed: angular-speed magnitude distribution (1D Wasserstein) -----------

def speed_set(feats: np.ndarray, mask: np.ndarray, n_prefix: int, n_suffix: int,
              eps: float = 1e-9) -> np.ndarray:
    """Bag of per-step angular speeds ‖f_{t+1}-f_t‖ at core positions (magnitude
    the unit-normalized velocity channels discard). Same coverage guard."""
    T = len(feats)
    win = np.zeros(T, dtype=bool)
    win[n_prefix:T - n_suffix] = True
    v = feats[1:] - feats[:-1]
    vt_win = win[:-1] & win[1:]
    keep = vt_win & mask[:-1]
    if keep.sum() < 1:
        keep = vt_win
    s = np.linalg.norm(v[keep], axis=1)
    return s if len(s) else np.linalg.norm(v[vt_win], axis=1)


def wasserstein1d_matrix(sets: list[np.ndarray]) -> np.ndarray:
    """Exact 1-D Wasserstein-1 between bags of scalars (scipy)."""
    from scipy.stats import wasserstein_distance
    n = len(sets)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(wasserstein_distance(sets[i], sets[j]))
    return D


# --- exact EMD (scipy HiGHS transportation LP) --------------------------------

def emd_pair(Ci: np.ndarray, Cj: np.ndarray) -> float:
    """Exact 1-Wasserstein between uniform empirical measures on unit-row sets,
    ground cost 1-cos. scipy linprog HiGHS transportation LP (no Sinkhorn)."""
    from scipy.optimize import linprog
    from scipy import sparse
    ni, nj = len(Ci), len(Cj)
    if ni == 0 or nj == 0:
        return float("nan")
    if ni == 1 or nj == 1:                         # degenerate -> mean cost (exact)
        return float((1.0 - Ci @ Cj.T).mean())
    cost = (1.0 - Ci @ Cj.T).ravel()
    a = np.full(ni, 1.0 / ni)
    b = np.full(nj, 1.0 / nj)
    # row-sum and col-sum equality constraints (drop one redundant row)
    rows = sparse.kron(sparse.eye(ni), np.ones((1, nj)))       # [ni, ni*nj]
    cols = sparse.kron(np.ones((1, ni)), sparse.eye(nj)).tocsr()   # [nj, ni*nj]
    A = sparse.vstack([rows, cols[:-1]]).tocsr()
    beq = np.concatenate([a, b[:-1]])
    res = linprog(cost, A_eq=A, b_eq=beq, bounds=(0, None), method="highs")
    return float(res.fun) if res.success else float("nan")


def emd_matrix(cores: list[np.ndarray], progress: int = 0) -> np.ndarray:
    C = [np.ascontiguousarray(c, dtype=np.float64) for c in cores]
    n = len(C)
    D = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = emd_pair(C[i], C[j])
            k += 1
            if progress and k % progress == 0:
                print(f"    emd {k}/{n*(n-1)//2}", flush=True)
    return D


_EMD_STATE: dict = {}


def _emd_pair_idx(pair):
    i, j = pair
    C = _EMD_STATE["C"]
    return emd_pair(C[i], C[j])


def emd_pair_w(Ci: np.ndarray, Cj: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Exact EMD with GIVEN marginals a, b (each sums to 1), cost 1-cos."""
    from scipy.optimize import linprog
    from scipy import sparse
    ni, nj = len(Ci), len(Cj)
    if ni == 0 or nj == 0:
        return float("nan")
    cost = (1.0 - Ci @ Cj.T)
    if ni == 1 or nj == 1:
        return float((a[:, None] * b[None, :] * cost).sum())
    rows = sparse.kron(sparse.eye(ni), np.ones((1, nj)))
    cols = sparse.kron(np.ones((1, ni)), sparse.eye(nj)).tocsr()
    A = sparse.vstack([rows, cols[:-1]]).tocsr()
    beq = np.concatenate([a, b[:-1]])
    res = linprog(cost.ravel(), A_eq=A, b_eq=beq, bounds=(0, None), method="highs")
    return float(res.fun) if res.success else float("nan")


def _emd_pair_w_idx(pair):
    i, j = pair
    C, W = _EMD_STATE["C"], _EMD_STATE["W"]
    return emd_pair_w(C[i], C[j], W[i], W[j])


def emd_matrix_weighted(cores: list[np.ndarray], weights: list[np.ndarray],
                        n_jobs: int = 8) -> np.ndarray:
    """Fork-parallel exact EMD with per-clip marginals (speed-proportional mass)."""
    import concurrent.futures, itertools, multiprocessing
    C = [np.ascontiguousarray(c, dtype=np.float64) for c in cores]
    W = []
    for w in weights:
        s = np.asarray(w, dtype=np.float64)
        W.append(s / s.sum() if s.sum() > 0 else np.full(len(s), 1.0 / max(len(s), 1)))
    n = len(C)
    pairs = list(itertools.combinations(range(n), 2))
    _EMD_STATE.update(C=C, W=W)
    try:
        ctx = multiprocessing.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
            vals = list(ex.map(_emd_pair_w_idx, pairs, chunksize=64))
    finally:
        _EMD_STATE.clear()
    D = np.full((n, n), np.nan)
    for i in range(n):
        if len(C[i]):
            D[i, i] = 0.0
    for (i, j), v in zip(pairs, vals):
        D[i, j] = D[j, i] = v
    return D


def emd_matrix_parallel(cores: list[np.ndarray], n_jobs: int | None = None) -> np.ndarray:
    """Fork-parallel exact EMD over independent pairs (bit-identical to emd_matrix)."""
    import concurrent.futures
    import itertools
    import multiprocessing
    import os

    C = [np.ascontiguousarray(c, dtype=np.float64) for c in cores]
    n = len(C)
    pairs = list(itertools.combinations(range(n), 2))
    if n_jobs is None:
        n_jobs = 8                               # single-thread BLAS per worker; keep modest on login node
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


# --- hierarchical ECDF fusion (fable Batch 4 operator) ------------------------

def ecdf_compose(mats: list[np.ndarray]) -> np.ndarray:
    """re-ECDF( mean( ECDF(m) for m in mats ) ) — a rank-space composite in [0,1]."""
    return ecdf_map(ecdf_fuse(mats))


def energy_matrix(cores: list[np.ndarray]) -> np.ndarray:
    """C5: energy distance between frame sets with Euclidean ground cost.
    D_E = 2*mean_{a,b} ||fa-fb|| - mean_{a,a'} - mean_{b,b'}  (V-statistics)."""
    C = [np.ascontiguousarray(c, dtype=np.float64) for c in cores]
    n = len(C)
    # self terms mean_{a,a'} ||fa-fa'|| per clip
    def mean_pdist(A, B):
        # mean over all pairs of ||a-b|| between row-sets A,B
        # ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b
        aa = np.einsum("ij,ij->i", A, A)
        bb = np.einsum("ij,ij->i", B, B)
        G = A @ B.T
        d2 = aa[:, None] + bb[None, :] - 2.0 * G
        np.maximum(d2, 0.0, out=d2)
        return float(np.sqrt(d2).mean())
    self_mean = np.array([mean_pdist(c, c) for c in C])
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            cross = mean_pdist(C[i], C[j])
            v = 2.0 * cross - self_mean[i] - self_mean[j]
            D[i, j] = D[j, i] = max(v, 0.0)
    return D
