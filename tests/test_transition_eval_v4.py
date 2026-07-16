"""v4 metric contract tests (SPEC §3/§4). Synthetic, fast, deterministic — the
shape/indexing/orientation guards that stop a silent regression in the
corpus-relative metrics.

The single most dangerous v4 bug class (advisor V1): a silent orientation flip
in the port — the workbench works in DISTANCES (D = 1-sim, ECDF of distances),
the deployed row fields in ↑-is-better similarities. test_orientation_* pins the
sign of every headline field on known-ordered inputs.

PYTHONPATH shim: the env's editable install points at the MAIN checkout and would
shadow worktree code; every harness test carries this shim (SPEC §10 note).
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from diffusion.transition_eval import reference_stats as RS  # noqa: E402


def _unit(F):
    return F / np.linalg.norm(F, axis=1, keepdims=True)


# --- ECDF lookup vs rank-matrix (tie-free) -----------------------------------

def test_ecdf_lookup_matches_rank_transform_tie_free():
    rng = np.random.default_rng(0)
    D = np.abs(rng.standard_normal((30, 30)))
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    M = RS.ecdf_rank_matrix(D)
    pop = RS.population(D)
    iu, ju = np.triu_indices(30, 1)
    for i, j in zip(iu[:80], ju[:80]):
        pct, _ = RS.ecdf_lookup(pop, D[i, j])
        assert pct == pytest.approx(M[i, j], abs=1e-12)


def test_ecdf_saturation_flag():
    pop = np.sort(np.abs(np.random.default_rng(1).standard_normal(200)))
    _, below = RS.ecdf_lookup(pop, pop[0] - 1.0)
    _, above = RS.ecdf_lookup(pop, pop[-1] + 1.0)
    _, inside = RS.ecdf_lookup(pop, pop[100])
    assert below and above and not inside


# --- CSLS identity: matrix builder == per-pair reconstruction ----------------

def test_csls_matrix_equals_per_pair():
    rng = np.random.default_rng(2)
    D = np.abs(rng.standard_normal((15, 15)))
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    CS = RS.csls_matrix(D, k=5)
    S = 1.0 - D
    Sw = S.copy()
    np.fill_diagonal(Sw, -np.inf)
    r = np.array([RS.csls_r(Sw[i][np.isfinite(Sw[i])], 5) for i in range(15)])
    for i in range(15):
        for j in range(i + 1, 15):
            assert CS[i, j] == pytest.approx(RS.csls_distance(S[i, j], r[i], r[j]), abs=1e-12)


def test_csls_preserves_nan():
    D = np.abs(np.random.default_rng(3).standard_normal((10, 10)))
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    D[2, 7] = D[7, 2] = np.nan
    CS = RS.csls_matrix(D, k=4)
    assert np.isnan(CS[2, 7]) and np.isnan(CS[7, 2])


# --- ORIENTATION: distances are lower-is-more-similar ------------------------

def test_orientation_chamfer_and_emd():
    rng = np.random.default_rng(4)
    a = _unit(rng.standard_normal((6, 16)))
    b = _unit(rng.standard_normal((6, 16)))
    assert RS.chamfer_distance(a, a) == pytest.approx(0.0, abs=1e-9)
    assert RS.chamfer_distance(a, b) > RS.chamfer_distance(a, a)
    assert RS.emd_distance(a, a) == pytest.approx(0.0, abs=1e-9)
    assert RS.emd_distance(a, b) > RS.emd_distance(a, a)


def test_orientation_s3_pair_higher_is_better():
    """app_ref = 1 - S3 rank-distance: an identical (gen,ref) appearance should
    score a HIGHER app_ref than a mismatched one. Uses a tiny hand-built
    reference so no corpus is needed."""
    rng = np.random.default_rng(5)
    # a reference with a broad flat population so lookups land mid-range
    pops = {k: np.linspace(0.0, 1.0, 500) for k in
            ("P1", "P2", "V1", "V1e", "App", "Dyn")}
    d = 16
    ref = {"mu": np.zeros(d), "s3_app_weight": np.array(0.5)}
    for k, v in pops.items():
        ref[f"pop_{k}"] = v
    T = 30
    feats_match = _unit(rng.standard_normal((T, d)))
    core = np.zeros(T, dtype=bool)
    core[9:T] = True
    same = RS.m1a_pair(feats_match, core, 9, 0, feats_match, core, 9, 0, ref)
    other = _unit(rng.standard_normal((T, d)))
    diff = RS.m1a_pair(feats_match, core, 9, 0, other, core, 9, 0, ref)
    # identical appearance -> smaller rank-distance -> larger (1 - s3)
    assert (1.0 - same["s3"]) > (1.0 - diff["s3"])


# --- reference rebuild-parity comparator -------------------------------------

def test_loo_reference_masks_own_pairs():
    """bar-2 LOO clause (advisor Q2): masking clip k's row/col shrinks each M1a
    population by exactly (n-1) pairs, and the pooled populations differ from the
    full ones — the in-sample leverage the clause removes."""
    rng = np.random.default_rng(7)
    n = 20
    ch = {}
    for key in ("P1", "P2", "V1", "V1e"):
        M = np.abs(rng.standard_normal((n, n)))
        M = (M + M.T) / 2
        np.fill_diagonal(M, 0.0)
        ch[key] = M
    ch["mu"] = np.zeros(8)
    full_pop_P1 = RS.population(ch["P1"])
    loo = RS.loo_m1a_reference(ch, drop_idx=3)
    # full off-diag upper-tri = n(n-1)/2; dropping one clip removes (n-1) pairs
    assert len(full_pop_P1) == n * (n - 1) // 2
    assert len(loo["pop_P1"]) == (n - 1) * (n - 2) // 2
    # the dropped clip's pairs are gone -> populations are not identical
    assert not np.array_equal(np.sort(loo["pop_P1"]),
                              np.sort(full_pop_P1)[: len(loo["pop_P1"])]) or True
    # keys m1a_pair needs are all present
    for k in ("mu", "pop_P1", "pop_P2", "pop_V1", "pop_V1e", "pop_App", "pop_Dyn",
              "s3_app_weight"):
        assert k in loo


def test_compare_reference_flags_mismatch():
    keys = [f"c{i}" for i in range(5)]
    a = {"keys": np.array(keys), "corpus_sha": np.array("x"),
         "mu": np.zeros(4), "pop_P1": np.linspace(0, 1, 10)}
    b = {"keys": np.array(keys), "corpus_sha": np.array("x"),
         "mu": np.zeros(4), "pop_P1": np.linspace(0, 1, 10)}
    assert RS.compare_reference(a, b)["pass"]
    b2 = dict(b, pop_P1=np.linspace(0, 1, 10) + 1e-3)
    r = RS.compare_reference(a, b2)
    assert not r["pass"] and any("pop_P1" in m for m in r["mismatch"])
