"""Shape, indexing, and determinism checks for diffusion.transition_eval.

Pure numpy — runs on the login node without torch or GPU. Synthetic feature
trajectories stand in for DINO embeddings: unit vectors moving between two
anchor directions.
"""

import pathlib
import sys

import numpy as np
import pytest

# Resolve the package from THIS checkout, not the env's editable install —
# in a worktree the editable install points at the main checkout and would
# silently test the wrong code (first import of `diffusion` binds the path).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from diffusion.transition_eval.controls import make_lerp
from diffusion.transition_eval.morph import (
    core_mask, derived_scalars, dtw_distance, morph_profile, profile_distance,
    resample_curve, znorm,
)
from diffusion.transition_eval.motion import motion_fidelity
from diffusion.transition_eval.report import retrieval_eval


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _traj(weights_a, dim=32, seed=0):
    """Feature trajectory blending two fixed anchor directions by weights_a."""
    rng = np.random.default_rng(seed)
    ea, eb = _unit(rng.normal(size=dim)), _unit(rng.normal(size=dim))
    w = np.asarray(weights_a)[:, None]
    return _unit(w * ea + (1 - w) * eb)


def synthetic_crossfade(T=121):
    """Linear blend A->B: the degenerate transition."""
    return _traj(np.linspace(1.0, 0.0, T))


def synthetic_metamorphosis(T=121, dim=32, seed=0):
    """Hold A, dive to a third 'effect' direction in the middle, reform as B."""
    rng = np.random.default_rng(seed)
    ea, eb, ec = _unit(rng.normal(size=dim)), _unit(rng.normal(size=dim)), _unit(rng.normal(size=dim))
    t = np.linspace(0.0, 1.0, T)
    wa = np.clip(1 - 4 * np.maximum(t - 0.25, 0), 0, 1)      # A until 25%, gone by 50%
    wb = np.clip(1 - 4 * np.maximum(0.75 - t, 0), 0, 1)      # B from 75%, absent before 50%
    wc = np.clip(1 - wa - wb, 0, 1)
    return _unit(wa[:, None] * ea + wb[:, None] * eb + wc[:, None] * ec)


def test_crossfade_has_near_zero_depth():
    p = morph_profile(synthetic_crossfade())
    s = derived_scalars(p)
    assert s["depth"] < 0.35, s
    assert s["core_frac"] < 0.1


def test_metamorphosis_has_high_depth_and_core():
    p = morph_profile(synthetic_metamorphosis())
    s = derived_scalars(p)
    assert s["depth"] > 0.7, s
    fade = derived_scalars(morph_profile(synthetic_crossfade()))
    assert s["core_frac"] > max(0.1, fade["core_frac"])
    assert s["depart"] > 0.15  # A holds before dissolving
    assert s["arrive"] < 0.85  # B present only near the end


def test_core_mask_fallback_is_single_frame():
    p = morph_profile(synthetic_crossfade())
    m = core_mask(p)
    assert m.sum() >= 1
    assert not m[:9].any() and not m[-8:].any()


def test_profile_distance_separates_shapes():
    meta1 = morph_profile(synthetic_metamorphosis(T=121, seed=1))
    meta2 = morph_profile(synthetic_metamorphosis(T=242, seed=2))  # length-invariant
    fade = morph_profile(synthetic_crossfade())
    d_same = profile_distance(meta1, meta2)
    d_diff = profile_distance(meta1, fade)
    assert d_same["dtw"] < d_diff["dtw"]
    assert d_same["pearson"] > d_diff["pearson"]


def test_one_endpoint_profile_drops_b():
    p = morph_profile(synthetic_metamorphosis(), n_endpoints=1)
    assert p["b_hat"] is None
    s = derived_scalars(p)
    assert "arrive" not in s
    q = morph_profile(synthetic_metamorphosis(seed=3), n_endpoints=1)
    assert np.isfinite(profile_distance(p, q)["dtw"])


def test_dtw_identity_and_determinism():
    x = znorm(resample_curve(np.sin(np.linspace(0, 6, 100)), 96))[:, None]
    assert dtw_distance(x, x) == pytest.approx(0.0, abs=1e-9)
    y = znorm(resample_curve(np.cos(np.linspace(0, 6, 150)), 96))[:, None]
    assert dtw_distance(x, y) == dtw_distance(x, y)  # deterministic
    assert dtw_distance(x, y) > 0


def test_motion_fidelity_matches_same_motion():
    T, N = 60, 25
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.normal(scale=0.01, size=(T, 1, 2)), axis=0)
    tracks_a = 0.5 + base + rng.normal(scale=1e-4, size=(T, N, 2))
    tracks_b = 0.3 + base + rng.normal(scale=1e-4, size=(T, N, 2))  # same motion, shifted
    tracks_c = tracks_a[::-1]                                        # reversed motion
    vis = np.ones((T, N), dtype=np.float32)
    same = motion_fidelity(tracks_a, vis, tracks_b, vis)
    diff = motion_fidelity(tracks_a, vis, tracks_c, vis)
    assert same > 0.9
    assert same > diff


def test_motion_fidelity_nan_on_static():
    T, N = 60, 10
    static = np.full((T, N, 2), 0.5)
    vis = np.ones((T, N), dtype=np.float32)
    assert np.isnan(motion_fidelity(static, vis, static, vis))


def test_lerp_control_shape_and_endpoints():
    pre = np.full((9, 64, 48, 3), 255, dtype=np.uint8)
    suf = np.zeros((8, 32, 24, 3), dtype=np.uint8)  # different geometry
    v = make_lerp(pre, suf, 121)
    assert v.shape == (121, 64, 48, 3)
    assert (v[:9] == 255).all() and (v[-8:] == 0).all()
    mid = v[9:-8].mean(axis=(1, 2, 3))
    assert (np.diff(mid) < 0).all()  # monotone fade


def test_retrieval_eval_perfect_and_chance():
    labels = ["x"] * 3 + ["y"] * 3
    D = np.ones((6, 6))
    for i in range(6):
        for j in range(6):
            if labels[i] == labels[j]:
                D[i, j] = 0.1
    np.fill_diagonal(D, 0.0)
    r = retrieval_eval(D, labels)
    assert r["accuracy_1nn"] == 1.0
    assert r["separation_cohens_d"] > 5
    assert r["chance"] == pytest.approx(0.5)


# --- exp_053 additions --------------------------------------------------------

def test_wilson_interval_basic():
    from diffusion.transition_eval.report import wilson_interval
    lo, hi = wilson_interval(38, 41)
    assert 0.75 < lo < 0.93 < hi < 1.0
    lo0, hi0 = wilson_interval(0, 10)
    assert lo0 == 0.0 and hi0 > 0.2
    assert np.isnan(wilson_interval(0, 0)[0])


def test_retrieval_eval_reports_wilson():
    labels = ["x"] * 3 + ["y"] * 3
    D = np.ones((6, 6)); np.fill_diagonal(D, 0.0)
    for i in range(6):
        for j in range(6):
            if labels[i] == labels[j]:
                D[i, j] = 0.1
    r = retrieval_eval(D, labels)
    lo, hi = r["accuracy_wilson95"]
    assert lo < r["accuracy_1nn"] <= hi <= 1.0


def test_cross_high_flag():
    from diffusion.transition_eval.morph import morph_profile
    rng = np.random.default_rng(0)
    base = rng.normal(size=64); base /= np.linalg.norm(base)
    # near-identical endpoints -> cross_high fires
    feats = np.tile(base, (30, 1)) + rng.normal(scale=0.01, size=(30, 64))
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    p = morph_profile(feats, n_prefix=9, n_suffix=8)
    assert p["cross_high"] is True
    # orthogonal endpoints -> no flag
    other = rng.normal(size=64); other -= other @ base * base; other /= np.linalg.norm(other)
    feats2 = np.concatenate([np.tile(base, (15, 1)), np.tile(other, (15, 1))])
    feats2 += rng.normal(scale=0.01, size=feats2.shape)
    feats2 /= np.linalg.norm(feats2, axis=1, keepdims=True)
    p2 = morph_profile(feats2, n_prefix=9, n_suffix=8)
    assert p2["cross_high"] is False


def test_trust_flags_logic():
    from diffusion.transition_eval.report import trust_flags
    val = {"retrieval": {"motion_fidelity": {"per_class_recall":
           {"water": 0.75, "earth": 0.0, "flame": 0.5}}}}
    counts = {"water": 4, "earth": 5, "flame": 2}
    tf = trust_flags(val, counts, motion_recall_min=0.5, min_ceiling_clips=4)
    assert tf["water"]["motion_trusted"] and tf["water"]["ceiling_trusted"]
    assert not tf["earth"]["motion_trusted"] and tf["earth"]["ceiling_trusted"]
    assert tf["flame"]["motion_trusted"] and not tf["flame"]["ceiling_trusted"]


def test_judge_pass_rate_all_pass():
    from diffusion.transition_eval.rubric import judge_pass_rate
    good = {q: {"answer": q not in ("q4_leakage", "q5_artifacts")}
            for q in ("q1_same_type", "q2_dynamics", "q3_endpoints", "q4_leakage", "q5_artifacts")}
    bad = {**good, "q5_artifacts": {"answer": True}}
    r = judge_pass_rate([good, bad])
    assert r["all_pass"] == pytest.approx(0.5)
    assert r["q5_artifacts"] == pytest.approx(0.5)
    assert r["n_parsed"] == 2
