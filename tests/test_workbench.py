"""Unit tests for the metric workbench's shared infrastructure.

Synthetic only — no corpus, no GPU, no cache. The bundle factory is the
certified suite's own (tests/test_certify_v3.py::fake_bundle), so the modules
are exercised against exactly the bundle shape the deployed pipeline emits.

What these tests are actually for: hubness.py has no precedent anywhere in the
certified tree and is a TERMINAL gate (a candidate that fails it is dead
regardless of accuracy), so it is tested against a PLANTED sink — the pathology
it exists to catch — not merely for shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from diffusion.transition_eval.workbench import anchors, curves, hubness, whitening

from test_certify_v3 import fake_bundle

RNG = np.random.default_rng(20260714)


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


# --- whitening (§1.1) ---------------------------------------------------------

def test_zca_whitens_an_anisotropic_cloud():
    """The point of §1.1: after ZCA the covariance is I, so inner products and
    chords are measured with a straight ruler."""
    d = 12
    A = RNG.normal(size=(d, d))
    X = RNG.normal(size=(4000, d)) @ A + 3.0        # anisotropic, off-center
    z = whitening.fit_zca(X)
    s = whitening.sanity(z, X)
    assert s["mean_abs_whitened_mean"] < 1e-9       # centered
    assert abs(s["mean_diag"] - 1.0) < 1e-6         # unit variance per axis
    assert s["max_abs_offdiag"] < 1e-6              # decorrelated


def test_zca_is_symmetric_and_deterministic():
    X = RNG.normal(size=(500, 8))
    z1, z2 = whitening.fit_zca(X), whitening.fit_zca(X)
    assert np.array_equal(z1["W"], z2["W"])                     # bitwise
    assert np.allclose(z1["W"], z1["W"].T, atol=1e-12)          # ZCA, not PCA
    assert np.allclose(z1["W"] @ z1["W_inv"], np.eye(8), atol=1e-8)


def test_zca_eigenvalue_floor_tames_a_degenerate_direction():
    """A near-zero eigendirection must not be amplified into pure noise."""
    X = RNG.normal(size=(500, 5))
    X[:, 4] *= 1e-9                                  # a dead axis
    z = whitening.fit_zca(X)
    assert int(z["n_floored"]) >= 1
    assert np.isfinite(whitening.whiten(z, X)).all()
    assert np.abs(whitening.whiten(z, X)).max() < 1e6


def test_whiten_is_affine_so_differences_drop_the_mean():
    """Why E1's delta is insensitive to the ZCA mean: mu cancels in a difference."""
    X = RNG.normal(size=(200, 6))
    z = whitening.fit_zca(X)
    a, b = RNG.normal(size=6), RNG.normal(size=6)
    lhs = whitening.whiten(z, a[None]) - whitening.whiten(z, b[None])
    rhs = (a - b) @ z["W"].T
    assert np.allclose(lhs[0], rhs, atol=1e-12)


# --- curves (§1.3) ------------------------------------------------------------

def test_arc_length_ignores_dwell_and_lurch():
    """Two clips tracing the same PATH at different speeds resample identically —
    that is the whole reason §1.3 parameterizes by arc length, not time."""
    steady = np.linspace(0, 1, 50)[:, None] * np.array([1.0, 0.0])
    t = np.concatenate([np.linspace(0, 0.2, 40), np.linspace(0.2, 1.0, 10)])
    lurching = t[:, None] * np.array([1.0, 0.0])     # same path, ugly schedule
    a = curves.resample(steady, 64)
    b = curves.resample(lurching, 64)
    assert np.abs(a - b).max() < 1e-9


def test_resample_shapes_and_degenerate_curve():
    assert curves.resample(RNG.normal(size=(37, 4)), 64).shape == (64, 4)
    frozen = np.tile(RNG.normal(size=(1, 4)), (30, 1))   # a clip that never moves
    out = curves.resample(frozen, 32)
    assert out.shape == (32, 4) and np.isfinite(out).all()


def test_l2_and_dtw_are_nan_propagating():
    """Undefined must stay undefined — never silently 0 (§1.5)."""
    a = np.zeros((8, 2))
    bad = np.full((8, 2), np.nan)
    assert np.isnan(curves.l2(a, bad))
    assert np.isnan(curves.banded_dtw(a, bad))
    assert curves.l2(a, a) == 0.0
    assert curves.banded_dtw(a, a) == pytest.approx(0.0)


def test_banded_dtw_absorbs_small_warps_but_not_large_ones():
    base = np.sin(np.linspace(0, 3.14, 64))[:, None]
    small = np.roll(base, 3)      # inside a 10% band of 64 (=6)
    large = np.roll(base, 30)     # far outside it
    assert curves.banded_dtw(base, small) < curves.banded_dtw(base, large)


def test_distance_matrix_is_symmetric_with_nan_for_undefined():
    cs = [np.zeros((8, 2)), np.ones((8, 2)), None]
    D = curves.distance_matrix(cs)
    assert D.shape == (3, 3)
    assert np.allclose(D[:2, :2], D[:2, :2].T)
    assert np.isnan(D[0, 2]) and np.isnan(D[2, 0])
    assert D[0, 0] == 0.0


# --- anchors (§1.2) -----------------------------------------------------------

def _zca_for(bundles):
    X = np.concatenate([b["feats"] for b in bundles]).astype(np.float64)
    return whitening.fit_zca(X)


def test_anchors_use_the_conditioned_windows_and_chord_is_positive():
    eA_dir, eB_dir = _unit(np.array([1.0, 0, 0] + [0] * 21)), _unit(np.array([0, 1.0, 0] + [0] * 21))
    core = _unit(np.array([0, 0, 1.0] + [0] * 21))
    b = fake_bundle(core, eA_dir, eB_dir)
    z = _zca_for([b])
    a = anchors.clip_anchors(b, z)
    assert a["D"] > 0
    assert np.isfinite(a["D_norenorm_diagnostic"]) and a["D_norenorm_diagnostic"] > 0
    assert a["chord"].shape == a["e_A"].shape


def test_min_d_floor_flags_the_bottom_5_percent_and_never_zeroes():
    dirs = _unit(RNG.normal(size=(40, 24)))
    bundles = [fake_bundle(_unit(RNG.normal(size=24)), d, _unit(d + 0.05 * RNG.normal(size=24)))
               for d in dirs]
    z = _zca_for(bundles)
    out = anchors.corpus_anchors(bundles, z)
    assert out["D"].shape == (40,)
    assert 0 <= int(out["n_low_D"]) <= 4                     # ~5% of 40
    assert (out["D"][out["low_D"]] < out["min_d_floor"]).all()
    assert (out["D"] > 0).all()                              # flagged, never zeroed


def test_endpoint_progress_coordinates_are_chord_normalized():
    """A frame sitting exactly on the chord has zero residual; one at the far
    endpoint has a_hat 1 and b_hat 0."""
    d = 8
    eA = np.zeros(d)
    eB = np.zeros(d); eB[0] = 4.0
    on_chord = np.stack([eA, 0.5 * eB, eB])
    p = anchors.endpoint_progress(on_chord, eA, eB)
    assert np.allclose(p["a_hat"], [0.0, 0.5, 1.0], atol=1e-12)
    assert np.allclose(p["b_hat"], [1.0, 0.5, 0.0], atol=1e-12)
    assert np.allclose(p["m"], 0.0, atol=1e-12)              # no excursion

    off = eB.copy(); off[1] = 4.0                            # 90 degrees off the chord
    p2 = anchors.endpoint_progress(np.stack([off]), eA, eB)
    assert p2["m"][0] == pytest.approx(1.0)                  # ||rho|| / D == 1


def test_endpoint_progress_undefined_when_chord_collapses():
    d = 6
    eA = np.zeros(d)
    p = anchors.endpoint_progress(RNG.normal(size=(5, d)), eA, eA.copy())
    assert np.isnan(p["a_hat"]).all() and np.isnan(p["m"]).all()


# --- hubness (§1.4) — the terminal gate, tested against a planted sink --------

def _healthy_matrix(n_per=6, n_cls=12, d=10):
    centers = RNG.normal(size=(n_cls, d)) * 6.0
    X = np.concatenate([c + RNG.normal(size=(n_per, d)) for c in centers])
    labels = [f"c{i}" for i in range(n_cls) for _ in range(n_per)]
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    return D, labels


def _sink_matrix(n_per=6, n_cls=12, d=10):
    """The M1c/polygon pathology, planted: one class's clips sit near the origin
    with tiny residual descriptors, so almost everyone's nearest neighbour is
    that class."""
    D, labels = _healthy_matrix(n_per, n_cls, d)
    sink = np.flatnonzero(np.array(labels) == "c0")
    D[:, sink] = 0.05                      # everyone is close to the sink
    D[sink, :] = 0.05
    D[np.ix_(sink, sink)] = 0.01
    np.fill_diagonal(D, 0.0)
    return D, labels


GATES = {"hubness": {"max_hubness_skew": 3.0, "min_pred_entropy_norm": 0.70,
                     "max_pred_class_share": 0.25}}


def test_healthy_space_passes_the_hubness_gate():
    D, labels = _healthy_matrix()
    v = hubness.gate(hubness.hubness_stats(D, labels), GATES)
    assert v["pass"], v
    assert v["stats"]["pred_entropy_norm"] > 0.9


def test_planted_sink_fails_the_hubness_gate_regardless_of_anything_else():
    D, labels = _sink_matrix()
    s = hubness.hubness_stats(D, labels)
    v = hubness.gate(s, GATES)
    assert not v["pass"]
    assert s["sink_class"] == "c0"                     # it names the culprit
    assert s["max_pred_class_share"] > 0.25
    assert not v["checks"]["entropy_ok"]


def test_kocc_and_entropy_ignore_undefined_rows():
    """An undefined clip retrieves no one and predicts nothing — it must not be
    counted as agreeing with anybody (§1.5)."""
    D, labels = _healthy_matrix()
    D[3, :] = np.nan
    D[:, 3] = np.nan
    s = hubness.hubness_stats(D, labels)
    assert s["n_undefined_rows"] == 1
    assert s["n_graded"] == len(labels) - 1
    assert np.isfinite(s["hubness_skew"])


def test_skew_of_a_flat_distribution_is_zero():
    assert hubness.skew(np.ones(50)) == 0.0


def test_partial_nan_pool_still_gates_and_reports_coverage():
    """Definedness gates NaN rows BY DESIGN (§3.3's energy gate is supposed to
    remove near-static clips). A shrunken pool concentrates hub statistics; that
    is a property of the candidate, not an artifact to correct away. The gate
    must still compute, and coverage must be reported beside it — no
    renormalization (that would be a new bar form)."""
    D, labels = _healthy_matrix()
    for i in (2, 5, 9, 14):
        D[i, :] = np.nan
        D[:, i] = np.nan
    s = hubness.hubness_stats(D, labels)
    v = hubness.gate(s, GATES)
    assert s["n_undefined_rows"] == 4
    assert s["coverage"] == pytest.approx(1 - 4 / len(labels))
    assert np.isfinite(s["hubness_skew"]) and v["pass"]


def test_only_the_gating_k_gates_but_k_sensitivity_is_persisted():
    D, labels = _sink_matrix()
    s = hubness.hubness_stats(D, labels)
    assert set(s["skew_by_k_diagnostic"]) == {"1", "5", "10"}
    assert s["skew_by_k_diagnostic"]["10"] == pytest.approx(s["hubness_skew"])
    assert hubness.GATING_K == 10


# --- the frozen calibration, as a self-auditing regression fixture ------------
#
# gates.yaml's hubness thresholds were calibrated on these six FROZEN INCUMBENT
# matrices before any candidate existed. The RUNBOOK §0 fixes where the gate must
# sit: M1c is "hub-collapsed (polygon column = sink artifact)" and MFS is
# collapsed by the same pattern -> both must FAIL; M1b is merely weak
# ("discrimination only inside camera-tagged strata") -> it must PASS, as must
# the healthy M1a family. If an implementation drift in hubness.py ever moves
# these, the gate has been silently rewritten and this test says so.

INCUMBENT_HUBNESS = {   # k=10, computed at step 0 from distance_matrices.npz
    "m1a__v3_sided":    (0.839, 0.9111, 0.0807, True),
    "m1a__v2_envelope": (0.967, 0.9059, 0.0807, True),
    "m1a__all_frames":  (1.188, 0.9280, 0.0673, True),
    "m1b_camera":       (2.517, 0.9173, 0.1019, True),
    "m_incumbent":      (3.386, 0.5094, 0.3901, False),
    "m1c_object":       (4.325, 0.4469, 0.5811, False),
}


@pytest.mark.parametrize("name", sorted(INCUMBENT_HUBNESS))
def test_frozen_gate_reproduces_the_runbook_diagnosis(name):
    skew, H, share, should_pass = INCUMBENT_HUBNESS[name]
    stats = {"hubness_skew": skew, "pred_entropy_norm": H, "max_pred_class_share": share}
    v = hubness.gate(stats, GATES)
    assert v["pass"] is should_pass, (
        f"{name}: gate says pass={v['pass']}, RUNBOOK §0 says pass={should_pass}")


def test_thresholds_sit_in_the_empty_gap_between_the_two_populations():
    """The derivation rule, not the numbers, is what is frozen: each threshold is
    the midpoint of the gap between the pass population and the dead population
    (the deployed tau_copy convention — probes.grade_splices)."""
    passing = [v for v in INCUMBENT_HUBNESS.values() if v[3]]
    dead = [v for v in INCUMBENT_HUBNESS.values() if not v[3]]
    g = GATES["hubness"]
    # skew: every passing metric below the bound, every dead one above
    assert max(v[0] for v in passing) < g["max_hubness_skew"] < min(v[0] for v in dead)
    # entropy: passing above, dead below
    assert min(v[1] for v in passing) > g["min_pred_entropy_norm"] > max(v[1] for v in dead)
    # prediction share: passing below, dead above
    assert max(v[2] for v in passing) < g["max_pred_class_share"] < min(v[2] for v in dead)
