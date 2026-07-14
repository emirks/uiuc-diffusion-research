"""Phase 1 motion metrics against CONSTRUCTED TRUTH — no GPU, no corpus.

Every test here builds a flow field from a KNOWN similarity (or a known camera +
known object motion) and asks whether the metric recovers it. This is the same
discipline §3.4 imposes on the real pipeline: a metric that produces
self-consistent numbers has proved nothing; the incumbent M1c retrieved polygon
58% of the time and was measuring nothing at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from diffusion.transition_eval.workbench import acceptance, curves, m1b_flow, m1c_flow

H, W = 64, 48                 # small, same aspect as the real 432x320
DIAG = float(np.hypot(H, W))


def synth_flow(tx=0.0, ty=0.0, log_s=0.0, theta=0.0, h=H, w=W) -> np.ndarray:
    """The exact dense flow field of a known similarity — the ground truth."""
    return m1c_flow.camera_field(np.array([tx, ty, log_s, theta]), h, w)


def fit(flow):
    A, grid = m1b_flow.design_matrix(flow.shape[0], flow.shape[1], stride=1)
    return m1b_flow.fit_similarity(flow, A, grid, stride=1)


# --- the fitter recovers constructed camera motion ----------------------------

@pytest.mark.parametrize("truth", [
    (3.0, 0.0, 0.0, 0.0),          # pan x
    (0.0, -2.5, 0.0, 0.0),         # pan y
    (0.0, 0.0, 0.05, 0.0),         # zoom in
    (0.0, 0.0, -0.03, 0.0),        # zoom out
    (0.0, 0.0, 0.0, 0.04),         # rotation
    (2.0, 1.0, 0.02, -0.03),       # compound
])
def test_similarity_fit_recovers_known_camera(truth):
    r = fit(synth_flow(*truth))
    assert r["defined"] and r["inlier_frac"] > 0.99
    assert np.allclose(r["params"], np.array(truth), atol=1e-6), r["params"]


def test_zero_flow_is_the_identity_not_a_failure():
    """A static camera is a perfectly good camera fit — it is M1c's ENERGY gate,
    not M1b's, that removes near-static content."""
    r = fit(np.zeros((H, W, 2)))
    assert r["defined"]
    assert np.allclose(r["params"], np.zeros(4), atol=1e-9)


def test_huber_survives_a_moderate_moving_object():
    """§3.2's operative rule: no spatial mask exists, so the effect region must be
    down-weighted as OUTLIER motion rather than masked out."""
    truth = (2.0, -1.0, 0.01, 0.02)
    flow = synth_flow(*truth)
    flow[:, : int(W * 0.15)] += np.array([9.0, -7.0])    # a fast object, 15% of frame
    r = fit(flow)
    assert r["defined"]
    assert np.allclose(r["params"], np.array(truth), atol=0.25), r["params"]
    assert r["inlier_frac"] < 1.0                        # it DID see the outliers


def test_huber_breakdown_is_bounded_and_the_definedness_gate_catches_the_rest():
    """A MEASURED PROPERTY of the estimator §3.2 pins, asserted here so it cannot
    drift silently. Huber is bounded-influence but LOW-BREAKDOWN: a big coherent
    effect region pulls the camera fit toward itself, and no initialization fixes
    it (a high-breakdown median-flow start converges to the identical fixed point).

    What protects the metric is the DEFINEDNESS gate, not the estimator: once the
    contamination is large enough to really corrupt the fit, the inlier fraction
    collapses below §3.2's 40% floor and the frame exits as UNDEFINED — wrong
    numbers are not emitted, they are withheld."""
    truth = np.array([2.0, -1.0, 0.01, 0.02])
    err = {}
    for frac in (0.15, 0.25, 0.33, 0.45):
        flow = synth_flow(*truth)
        flow[:, : int(W * frac)] += np.array([9.0, -7.0])
        r = fit(flow)
        err[frac] = (float(np.abs(r["params"] - truth).max()), r["inlier_frac"], r["defined"])

    # bias grows with contamination, monotonically — no cliff, no surprise
    assert err[0.15][0] < err[0.25][0] < err[0.33][0]
    assert err[0.33][0] < 1.0                    # still a usable camera at a third
    # and the catastrophic case is refused rather than reported
    assert err[0.45][1] < m1b_flow.MIN_INLIER_FRAC
    assert err[0.45][2] is False


def test_pure_noise_flow_is_undefined_not_confidently_wrong():
    rng = np.random.default_rng(0)
    flow = rng.normal(0, 12.0, size=(H, W, 2))     # no coherent camera at all
    r = fit(flow)
    assert r["inlier_frac"] < m1b_flow.MIN_INLIER_FRAC
    assert not r["defined"]                        # undefined, never a fake number


# --- clip-level definedness (§3.2) --------------------------------------------

def _traj(n_defined=20, n_bad=0):
    T = n_defined + n_bad
    flow = np.stack([synth_flow(0.2 * i, 0.0, 0.0, 0.0) for i in range(n_defined)]
                    + [np.random.default_rng(i).normal(0, 12, (H, W, 2)) for i in range(n_bad)])
    return m1b_flow.clip_camera_trajectory(flow, stride=2)


def test_clip_undefined_when_too_much_of_the_core_fails():
    t = _traj(n_defined=6, n_bad=6)                # 50% undefined > 30% cap
    d = m1b_flow.clip_descriptor(t, np.ones(12, bool))
    assert not d["defined"] and d["curve"] is None
    assert "undefined" in d["reason"]


def test_clip_defined_below_the_cap_and_descriptor_has_the_right_shape():
    t = _traj(n_defined=10, n_bad=2)               # ~17% undefined < 30%
    d = m1b_flow.clip_descriptor(t, np.ones(12, bool))
    assert d["defined"]
    assert d["curve"].shape == (m1b_flow.N_RESAMPLE, 4)


def test_texture_gate_can_undefine_a_frame_the_fit_liked():
    """A confident fit to a textureless frame is still a fit to noise (§3.2)."""
    T = 10
    flow = np.stack([synth_flow(0.3 * i) for i in range(T)])
    tex = np.ones(T, bool)
    tex[:5] = False
    t = m1b_flow.clip_camera_trajectory(flow, texture_ok=tex, stride=2)
    assert t["defined"][:5].sum() == 0            # gated despite a clean fit
    assert t["defined"][5:].all()
    assert np.isnan(t["params"][:5]).all()        # undefined, not zeroed


# --- M1c: the residual and the energy gate (§3.3) ------------------------------

def test_pure_camera_leaves_no_residual_so_the_energy_gate_fires():
    """THE DESIGNED KILL. A clip whose motion is entirely camera has a near-zero
    residual; that is not an object descriptor, it is noise. It must exit as
    UNDEFINED rather than collapse onto a shared sink (the polygon pattern)."""
    truth = np.array([2.0, 1.0, 0.01, 0.02])
    flow = m1c_flow.camera_field(truth, H, W)
    res = m1c_flow.residual_field(flow, truth)
    assert m1c_flow.frame_energy(res, DIAG) < 1e-9

    T = 12
    flows = np.stack([flow] * T)
    traj = m1b_flow.clip_camera_trajectory(flows, stride=2)
    out = m1c_flow.clip_curve(flows, traj, np.ones(T, bool), eps=1e-4)
    assert not out["defined"]                      # gated out, as designed
    assert out["n_gated"] == T
    assert out["curve"] is None                    # NaN row, never a zero vector


def test_real_object_motion_survives_the_energy_gate_and_describes():
    cam = np.array([1.0, 0.0, 0.0, 0.0])
    T = 12
    flows = []
    for _ in range(T):
        f = m1c_flow.camera_field(cam, H, W)
        f[H // 4:H // 2, W // 4:W // 2] += np.array([6.0, 4.0])   # an object moves
        flows.append(f)
    flows = np.stack(flows)
    traj = m1b_flow.clip_camera_trajectory(flows, stride=2)
    out = m1c_flow.clip_curve(flows, traj, np.ones(T, bool), eps=1e-4)
    assert out["defined"] and out["n_gated"] == 0
    assert out["curve"].shape == (m1c_flow.N_RESAMPLE, 3 * 3 * 8 + 3)
    assert np.isfinite(out["curve"]).all()


def test_descriptor_is_resolution_normalized():
    """Magnitudes are divided by the image diagonal, so the same motion at two
    resolutions must not produce different descriptors (§3.3: no resolution leak)."""
    cam = np.zeros(4)
    small = m1c_flow.camera_field(cam, 64, 48)
    small[10:20, 10:20] += np.array([3.0, 3.0])
    big = m1c_flow.camera_field(cam, 128, 96)
    big[20:40, 20:40] += np.array([6.0, 6.0])      # same motion, 2x scale
    ds = m1c_flow.frame_descriptor(small, float(np.hypot(64, 48)))
    db = m1c_flow.frame_descriptor(big, float(np.hypot(128, 96)))
    # orientation histogram is scale-free; the magnitude channel is diagonal-normalized
    assert ds[-1] == pytest.approx(db[-1], rel=0.10)


def test_energy_is_nan_when_the_camera_fit_failed_never_zero():
    rng = np.random.default_rng(3)
    flows = rng.normal(0, 12, (6, H, W, 2))
    traj = m1b_flow.clip_camera_trajectory(flows, stride=2)
    st = m1c_flow.clip_residual_stats(flows, traj)
    assert np.isnan(st["energy"][~traj["defined"]]).all()


# --- §3.4 acceptance machinery -------------------------------------------------

def test_relative_params_compose_back_to_the_cumulative_pose():
    cum = acceptance.trajectory("pan_zoom", 8, amp=4.0)
    rel = acceptance.relative_params(cum)
    M = acceptance.sim_matrix(*cum[0])
    for p in rel:
        M = acceptance.sim_matrix(*p) @ M
    assert np.allclose(M, acceptance.sim_matrix(*cum[-1]), atol=1e-9)


def test_injected_trajectory_is_recovered_from_its_own_flow():
    """The §3.4 test, end to end, on analytic flow: warp -> flow -> fit -> grade."""
    cum = acceptance.trajectory("pan_x", 12, amp=6.0)
    truth = acceptance.relative_params(cum)
    flows = np.stack([synth_flow(*p) for p in truth])
    traj = m1b_flow.clip_camera_trajectory(flows, stride=1)
    g = acceptance.grade_injection(traj["params"], truth, traj["defined"])
    assert g["pass"], g
    assert g["params"]["tx"]["corr"] >= acceptance.CORR_MIN
    assert g["params"]["tx"]["amp_err"] <= acceptance.AMP_ERR_MAX


def test_grade_injection_fails_a_halved_amplitude_even_at_perfect_correlation():
    """A trajectory that tracks the truth in SHAPE but at half the amplitude has
    not recovered the camera — correlation alone would pass it."""
    cum = acceptance.trajectory("zoom", 12, amp=0.4)
    truth = acceptance.relative_params(cum)
    g = acceptance.grade_injection(truth * 0.5, truth, np.ones(len(truth), bool))
    assert g["params"]["log_scale"]["corr"] == pytest.approx(1.0, abs=1e-9)
    assert not g["params"]["log_scale"]["amp_ok"]
    assert not g["pass"]


def test_reversal_ground_truth_negates_rotation_and_zoom():
    """Under time reversal each step inverts: log-scale and rotation negate."""
    p = np.array([[1.0, 0.0, 0.05, 0.03]] * 5)
    e = acceptance.expected_reversed_params(p)
    assert np.allclose(e[:, 2], -0.05, atol=1e-9)      # zoom negates
    assert np.allclose(e[:, 3], -0.03, atol=1e-9)      # rotation negates


def test_reversal_probe_catches_a_direction_blind_metric():
    """THE BLIND SPOT. A metric whose 'reversed' fit equals the forward fit (no
    sign flip) must FAIL parameter negation even though its descriptor distance
    might look fine."""
    T = 10
    fwd_params = np.stack([np.array([0.5, 0.0, 0.02, 0.01])] * T)
    fwd = {"params": fwd_params, "defined": np.ones(T, bool)}
    blind = {"params": fwd_params.copy(), "defined": np.ones(T, bool)}   # no flip
    g = acceptance.grade_reversal(fwd, blind, np.zeros((32, 4)), np.ones((32, 4)),
                                  median_within_class=0.1)
    assert not g["parameter_negation"]["pass"]
    assert not g["pass"]

    honest = {"params": acceptance.expected_reversed_params(fwd_params),
              "defined": np.ones(T, bool)}
    g2 = acceptance.grade_reversal(fwd, honest, np.zeros((32, 4)), np.ones((32, 4)),
                                   median_within_class=0.1)
    assert g2["parameter_negation"]["pass"]
    assert g2["pass"]


def test_palindromic_move_is_excluded_from_reversal_grading():
    """A pan out and back genuinely IS its own reverse — grading it grades noise."""
    static = np.zeros((12, 4))
    assert not acceptance.reversal_sensitive(static, np.ones(12, bool))
