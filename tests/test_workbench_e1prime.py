"""Unit tests for E1' — the gamma-scalar signature (E1PRIME_DIRECTIVE §2).

Synthetic only — no corpus, no GPU, no cache.

Two of these tests are not shape checks. They are the EVIDENCE for two claims the
frozen pre-registration rests on, and they were written before the candidate ran:

  test_per_channel_arclength_linearizes_monotone
      PREREG §P1 rejects the per-channel arc-length reading of sigma BY DERIVATION:
      reparameterizing a monotone scalar channel by its own arc length maps it to a
      STRAIGHT LINE for every clip, which would annihilate a_hat corpus-wide. If that
      claim were false, the rejected reading would be back on the table and the gating
      sigma would be an unforced executor choice. So it is proved, not asserted.

  test_make_lerp_is_idempotent_on_its_own_endpoints
      PREREG §P5a's disclosure: every "nothing" object has m~ == 0 EXACTLY, which is
      why IV1/IV2 certify less than their names suggest. That disclosure is only honest
      if the idempotence is real. It is proved here on the deployed builder.
"""

from __future__ import annotations

import numpy as np
import pytest

from diffusion.transition_eval.controls import make_lerp
from diffusion.transition_eval.workbench import curves, e1prime, lw

RNG = np.random.default_rng(20260714)


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


# --- P1: why the per-channel arc-length reading of sigma was rejected ----------

def test_per_channel_arclength_linearizes_monotone():
    """The claim PREREG §P1 rejects a reading on: arc-length reparameterization of a
    MONOTONE scalar channel makes it a straight line, for ANY monotone input.

    a_hat (progress along the chord) is near-monotone by construction, so under the
    rejected reading every clip's a_hat channel would collapse to the same ramp shape
    and carry only its endpoints. That is not a resampling convention; it is the
    deletion of a channel."""
    for _ in range(20):
        # any strictly increasing curve, however wildly shaped
        v = np.cumsum(RNG.uniform(0.01, 3.0, size=40))
        got = curves.resample(v[:, None], 64)[:, 0]      # per-channel arc length
        line = np.linspace(v[0], v[-1], 64)
        assert np.allclose(got, line, atol=1e-9), "monotone channel did NOT linearize"

    # and it is NOT a degenerate property of resampling in general: the SHARED-sigma
    # reading (P1's gating choice) preserves the channel's shape.
    v = np.cumsum(RNG.uniform(0.01, 3.0, size=40))
    w = np.sin(np.linspace(0, 4 * np.pi, 40))            # a second, non-monotone channel
    X = np.stack([v, w], axis=1)
    shared = curves.resample(X, 64)[:, 0]                # sigma from the JOINT curve
    line = np.linspace(v[0], v[-1], 64)
    assert not np.allclose(shared, line, atol=1e-3), \
        "shared-sigma reading also linearized the channel — P1's derivation would be void"


def test_resample_on_matches_curves_resample_for_own_sigma():
    """_resample_on(X, arclen(X)) is exactly curves.resample(X) — the explicit form is
    only needed because the non-gating sensitivity column uses a DIFFERENT sigma."""
    X = RNG.normal(size=(37, 3))
    a = curves.resample(X, 64)
    b = e1prime._resample_on(X, curves.arc_length_sigma(X), 64)
    assert np.allclose(a, b, atol=0, rtol=0)


# --- P5a: the m~ == 0 tautology, proved on the deployed builder ----------------

def test_make_lerp_is_idempotent_on_its_own_endpoints():
    """A rendered null's own rendered null IS that null, bit-for-bit. Hence every
    "nothing" object has m~ = m_lerp - m_lerp == 0 exactly (PREREG §P5a)."""
    prefix = RNG.integers(0, 256, size=(9, 16, 16, 3), dtype=np.uint8)
    suffix = RNG.integers(0, 256, size=(8, 16, 16, 3), dtype=np.uint8)
    null = make_lerp(prefix, suffix, 121)
    assert len(null) == 121
    again = make_lerp(null[:9], null[-8:], 121)
    assert np.array_equal(null, again), "make_lerp is NOT idempotent — P5a's disclosure "\
                                        "would be wrong"


def test_m_tilde_is_exactly_zero_when_clip_is_its_own_null():
    """The signature-level consequence: feed a null's frames as both the source and the
    null, and the m~ channel is identically zero."""
    d = 32
    eA, eB = _unit(RNG.normal(size=d)), _unit(RNG.normal(size=d))
    Q = e1prime.sided_basis(eA, eB, "twosided")
    D = float(np.linalg.norm(eB - eA))
    feats = _unit(RNG.normal(size=(20, d)))
    m = e1prime.sided_m(feats, Q, D)
    assert np.allclose(m - m, 0.0, atol=0), "m - m must be exactly 0"
    assert np.all(m >= 0)


# --- P2: the sided residual ---------------------------------------------------

def test_sided_basis_rank_and_orthonormality():
    d = 24
    eA, eB = _unit(RNG.normal(size=d)), _unit(RNG.normal(size=d))
    Q1 = e1prime.sided_basis(eA, eB, "onesided")
    Q2 = e1prime.sided_basis(eA, eB, "twosided")
    assert Q1.shape == (d, 1) and Q2.shape == (d, 2)
    assert np.allclose(Q1.T @ Q1, np.eye(1), atol=1e-12)
    assert np.allclose(Q2.T @ Q2, np.eye(2), atol=1e-12)
    # the one-sided basis spans e_A; the two-sided one spans BOTH anchors
    assert np.allclose(Q1 @ (Q1.T @ eA), eA, atol=1e-10)
    for e in (eA, eB):
        assert np.allclose(Q2 @ (Q2.T @ e), e, atol=1e-10)


def test_sided_basis_degenerate_returns_none_never_rank1_fallback():
    """§1.5: undefined is not zero — and it is not a quiet fallback either."""
    d = 16
    eA = _unit(RNG.normal(size=d))
    assert e1prime.sided_basis(eA, eA.copy(), "twosided") is None      # collinear
    assert e1prime.sided_basis(eA, 3.0 * eA, "twosided") is None       # parallel
    assert e1prime.sided_basis(np.zeros(d), eA, "onesided") is None    # zero anchor


def test_sided_m_is_the_residual_off_the_span():
    d = 20
    eA, eB = _unit(RNG.normal(size=d)), _unit(RNG.normal(size=d))
    Q = e1prime.sided_basis(eA, eB, "twosided")
    D = float(np.linalg.norm(eB - eA))
    # a frame lying INSIDE the span has zero residual...
    inside = 0.3 * eA + 0.7 * eB
    assert e1prime.sided_m(inside[None], Q, D)[0] == pytest.approx(0.0, abs=1e-12)
    # ...and one orthogonal to it has residual == its own norm / D
    rest = np.linalg.qr(np.column_stack([eA, eB, RNG.normal(size=(d, 3))]))[0][:, 2]
    assert e1prime.sided_m(rest[None], Q, D)[0] == pytest.approx(1.0 / D, rel=1e-9)
    # the residual is orthogonal to the span, by construction
    f = RNG.normal(size=(7, d))
    rho = f - (f @ Q) @ Q.T
    assert np.allclose(rho @ Q, 0.0, atol=1e-10)


# --- distances ----------------------------------------------------------------

def test_banded_dtw_batch_is_bit_identical_to_the_scalar_form():
    """The batch DTW is an OPTIMIZATION, never a change of semantics. If it drifted
    from curves.banded_dtw the candidate would be graded by a different distance than
    the one that is frozen."""
    for _ in range(25):
        n = 64
        A = RNG.normal(size=(5, n))
        B = RNG.normal(size=(5, n))
        band = max(1, int(round(0.10 * n)))
        batch = e1prime.banded_dtw_batch(A, B, band)
        for p in range(5):
            scalar = curves.banded_dtw(A[p][:, None], B[p][:, None], band_frac=0.10)
            assert batch[p] == pytest.approx(scalar, rel=0, abs=1e-12)


def test_distance_matrix_equal_weight_mean_and_nan_discipline():
    sigs = [RNG.normal(size=(64, 3)) for _ in range(6)]
    sigs[2] = None                                        # an undefined clip
    D = e1prime.distance_matrix(sigs)
    assert D.shape == (6, 6)
    assert np.allclose(D, D.T, equal_nan=True)            # symmetric
    assert np.all(np.isnan(D[2, :])) and np.all(np.isnan(D[:, 2]))   # NaN, never 0
    assert np.diag(D)[0] == 0.0
    # the combination rule is the EQUAL-WEIGHT MEAN of the per-channel DTWs
    band = max(1, int(round(0.10 * 64)))
    want = np.mean([curves.banded_dtw(sigs[0][:, c:c + 1], sigs[1][:, c:c + 1], 0.10)
                    for c in range(3)])
    assert D[0, 1] == pytest.approx(want, abs=1e-12)


def test_undefined_clip_is_not_scored_as_zero_distance():
    """§1.5, the failure mode that would silently flatter a candidate: an undefined
    clip must come out NaN (which the frozen kernel reads as 'cannot retrieve' and
    drops from coverage), never 0.0 (which would make it everyone's nearest
    neighbour)."""
    sigs = [None, RNG.normal(size=(64, 3))]
    D = e1prime.distance_matrix(sigs)
    assert np.isnan(D[0, 1])
    assert not np.isfinite(D[0, 1])


# --- arm C: Ledoit-Wolf (no free parameter — that is the point) ---------------

def test_ledoit_wolf_shrinkage_is_a_valid_intensity():
    X = RNG.normal(size=(200, 40)) @ RNG.normal(size=(40, 40))
    r = lw.shrinkage(X)
    assert 0.0 <= r["shrinkage"] <= 1.0


def test_ledoit_wolf_preserves_the_trace_exactly():
    """Sigma* = d*m*I + (1-d)*S  =>  tr(Sigma*)/p == tr(S)/p == m, exactly. This is an
    algebraic identity of the estimator, so it is a REAL check on the implementation
    (a wrong delta or a wrong target would break it), not a tautology."""
    X = RNG.normal(size=(150, 30)) @ RNG.normal(size=(30, 30))
    r = lw.shrinkage(X)
    p = r["p"]
    assert np.trace(r["Sigma"]) / p == pytest.approx(r["m"], rel=1e-12)
    assert np.trace(r["Sigma"]) / p == pytest.approx(np.trace(r["emp_cov"]) / p, rel=1e-12)


def test_ledoit_wolf_shrinks_toward_zero_as_n_grows():
    """delta -> 0 as n -> inf with p fixed: with enough samples the empirical covariance
    needs no help. If this failed, the arm would be whitening with a made-up target."""
    A = RNG.normal(size=(20, 20))
    small = lw.shrinkage(RNG.normal(size=(40, 20)) @ A)["shrinkage"]
    large = lw.shrinkage(RNG.normal(size=(8000, 20)) @ A)["shrinkage"]
    assert large < small
    assert large < 0.05


def test_ledoit_wolf_whitener_needs_no_eigenvalue_floor():
    """Sigma* is positive-definite by construction (lambda_min >= delta*m > 0), so arm C
    introduces NO regularization parameter — the whole reason it can answer escalation
    (a) without a second executor choice."""
    X = RNG.normal(size=(60, 50)) @ np.diag(np.r_[np.ones(45), 1e-9 * np.ones(5)])
    art = lw.fit(X)
    assert art["eig_floor_ratio"] is None
    assert art["n_floored"] == 0
    assert art["eigvals"].min() > 0
    assert art["condition_number_shrunk"] < art["condition_number_raw"]


def test_ledoit_wolf_whitens_a_well_conditioned_population():
    """On a WELL-CONDITIONED population (every eigenvalue comparable to the mean, so the
    shrinkage term delta*m is negligible against all of them), LW is a near-exact
    whitener: cov(whitened) ~ I. It really is a whitener, not a no-op.

    The qualifier is not a fudge — it is the estimator's designed behaviour, and the
    next test turns it into the actual experiment: where eigenvalues are NOT comparable
    to the mean, LW deliberately under-corrects, which is exactly how it differs from an
    eigenvalue floor."""
    p = 20
    Q = np.linalg.qr(RNG.normal(size=(p, p)))[0]
    scale = Q @ np.diag(np.sqrt(np.linspace(1.0, 4.0, p))) @ Q.T   # cond(cov) = 4
    X = RNG.normal(size=(8000, p)) @ scale
    art = lw.fit(X)
    assert art["shrinkage"] < 0.05
    C = np.cov(lw.whiten(art, X).T, bias=True)
    assert np.allclose(np.diag(C), 1.0, atol=0.10)
    assert np.abs(C - np.diag(np.diag(C))).max() < 0.10


def test_ledoit_wolf_does_not_amplify_near_null_directions_but_the_eig_floor_does():
    """THE PROPERTY ARM C EXISTS TO PROBE, isolated.

    A ZCA whitener with an eigenvalue FLOOR divides the near-null directions by
    sqrt(floor), which is tiny — so pure noise in those directions is amplified to unit
    variance and ends up dominating every downstream distance. Ledoit-Wolf divides them
    by sqrt(delta*m) instead — a substantial fraction of the mean variance — so they
    stay small.

    This is exactly the difference escalation (a) left open: whether E1's collapse
    belongs to WHITENING, or to the executor-chosen FLOOR. The two whiteners are
    therefore not interchangeable, and arm C is a real experiment rather than a
    restatement. (This test asserts the mechanism, not any outcome on the corpus.)"""
    from diffusion.transition_eval.workbench import whitening

    n_null = 5
    X = RNG.normal(size=(200, 50)) @ np.diag(np.r_[np.ones(45), 1e-6 * np.ones(n_null)])

    v_lw = np.var(lw.whiten(lw.fit(X), X), axis=0)[-n_null:]
    zca = whitening.fit_zca(X, eig_floor_ratio=1e-6)
    v_zca = np.var(whitening.whiten(zca, X), axis=0)[-n_null:]

    assert v_lw.max() < 1e-3, "Ledoit-Wolf amplified a near-null direction"
    assert v_zca.max() > 10 * v_lw.max(), (
        "the eig-floor whitener did NOT amplify the near-null directions more than "
        "LW — the two arms would then be probing the same thing")
