"""v3 metric math contract tests — synthetic, deterministic, no GPU.
Covers: sidedness-aware core mask + flagged fallback (S), similarity-fit
recovery + camera/object decomposition (M1b/c), copy/intrusion/memorization
(M2a/b/c), static-hold control, manifest derivations (tier, sidedness)."""

import json
import pathlib
import sys

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diffusion.transition_eval import manifests_v3 as MF          # noqa: E402
from diffusion.transition_eval.controls import make_static_hold    # noqa: E402
from diffusion.transition_eval.m1_transfer import (                # noqa: E402
    _fit_similarity, camera_match, camera_trajectory, object_match,
)
from diffusion.transition_eval.m2_integrity import (               # noqa: E402
    copy_score, intrusion_margin, memorization_score, mid_mask,
)
from diffusion.transition_eval.s_structure import core_mask_v3     # noqa: E402

RNG = np.random.default_rng(7)


def fake_profile(a_hat, b_hat=None, n_pre=9, n_suf=8):
    return {"a_hat": np.asarray(a_hat, dtype=float),
            "b_hat": None if b_hat is None else np.asarray(b_hat, dtype=float),
            "cross": 0.2, "cross_high": False,
            "n_prefix": n_pre, "n_suffix": n_suf, "n_endpoints": 2}


# --- S ---------------------------------------------------------------------------

def test_core_mask_sidedness_modes_differ():
    T = 60
    t = np.linspace(0, 1, T)
    a = 1 - t                       # departs A linearly
    b = t                           # approaches B linearly
    p = fake_profile(a, b, n_pre=5, n_suf=5)
    two, meta2 = core_mask_v3(p, "twosided")
    one, meta1 = core_mask_v3(p, "onesided")
    # one-sided ignores b̂: late frames (near B, far from A) join the core
    assert one.sum() > two.sum()
    assert meta1["mode"] == "a_only" and meta2["mode"] == "envelope"
    late = np.zeros(T, dtype=bool)
    late[45:55] = True   # far from A, near B, still OUTSIDE the suffix window
    assert one[late].all() and not two[late].any()


def test_core_mask_fallback_flags_and_sizes():
    T = 50
    env = np.full(T, 0.9)
    env[24:27] = 0.55               # shallow valley, never below 0.5
    p = fake_profile(env, env, n_pre=5, n_suf=5)
    mask, meta = core_mask_v3(p, "twosided")
    assert meta["core_degenerate"] is True
    assert mask.sum() >= 3          # valley expansion, not a single frame
    assert not mask[:5].any() and not mask[-5:].any()


def test_core_mask_strict_path_unflagged():
    T = 60
    env = np.full(T, 0.9)
    env[20:40] = 0.1
    p = fake_profile(env, env, n_pre=5, n_suf=5)
    mask, meta = core_mask_v3(p, "twosided")
    assert meta["core_degenerate"] is False and mask.sum() == 20


# --- M1b / M1c --------------------------------------------------------------------

def test_similarity_fit_recovers_transform():
    P = RNG.uniform(0, 1, (40, 2))
    s, th = 1.05, 0.1
    M_true = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    t_true = np.array([0.03, -0.02])
    Q = P @ M_true.T + t_true
    M, t = _fit_similarity(P, Q)
    assert np.allclose(M, M_true, atol=1e-9) and np.allclose(t, t_true, atol=1e-9)


def _tracks_from_motion(T, N, pan=None, objects=None):
    """Synthetic normalized tracks: global pan per step + optional moving objects
    (list of (point_slice, velocity))."""
    base = RNG.uniform(0.1, 0.9, (N, 2))
    tr = np.zeros((T, N, 2), dtype=np.float32)
    tr[0] = base
    for s in range(1, T):
        step = np.zeros((N, 2))
        if pan is not None:
            step += pan
        if objects:
            for sl, v in objects:
                step[sl] += v
        tr[s] = tr[s - 1] + step
    return tr, np.ones((T, N), dtype=np.float32)


def test_camera_trajectory_recovers_pan():
    tr, vis = _tracks_from_motion(40, 50, pan=np.array([0.004, -0.002]))
    cam = camera_trajectory(tr, vis)
    assert cam["valid"].all()
    mid = cam["params"][5:-5]       # box-smoothing bleeds at the edges
    assert np.allclose(mid[:, 0], 0.004, atol=5e-4)
    assert np.allclose(mid[:, 1], -0.002, atol=5e-4)
    assert np.abs(mid[:, 2:]).max() < 5e-3   # no zoom/rot


def test_camera_match_same_vs_opposite_pan():
    trA, visA = _tracks_from_motion(40, 50, pan=np.array([0.004, 0.0]))
    trB, visB = _tracks_from_motion(40, 50, pan=np.array([0.004, 0.0]))
    trC, visC = _tracks_from_motion(40, 50, pan=np.array([-0.004, 0.0]))
    same = camera_match(camera_trajectory(trA, visA), camera_trajectory(trB, visB))
    opp = camera_match(camera_trajectory(trA, visA), camera_trajectory(trC, visC))
    assert same["cam_valid"] and opp["cam_valid"]
    assert same["cam_corr"] > opp["cam_corr"]


def test_object_match_removes_camera_and_sees_objects():
    obj = (slice(0, 8), np.array([0.006, 0.006]))
    trA, visA = _tracks_from_motion(40, 60, pan=np.array([0.004, 0.0]), objects=[obj])
    trB, visB = _tracks_from_motion(40, 60, pan=np.array([-0.004, 0.001]), objects=[obj])
    # same object motion under DIFFERENT camera -> residual match high
    m_same_obj = object_match(trA, visA, trB, visB)
    # pure pans, no objects -> nan (no residually-moving tracklets)
    trC, visC = _tracks_from_motion(40, 60, pan=np.array([0.004, 0.0]))
    trD, visD = _tracks_from_motion(40, 60, pan=np.array([0.002, 0.001]))
    assert np.isnan(object_match(trC, visC, trD, visD))
    assert m_same_obj > 0.8


# --- M2 ---------------------------------------------------------------------------

def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_copy_score_flags_and_provenance():
    D = 32
    ref = _unit(RNG.normal(size=(50, D))).astype(np.float32)
    ref_core = np.zeros(50, dtype=bool)
    ref_core[20:30] = True
    gen = _unit(RNG.normal(size=(40, D))).astype(np.float32)
    gen[25] = ref[5]                       # copy a NON-core ref frame into gen mid
    mm = mid_mask(40, 9, 8)
    r = copy_score(gen, mm, ref, ref_core, tau=0.88)
    assert r["near_copy"] is True and r["copy_max"] >= 0.999
    assert r["copy_gen_frame"] == 25 and r["copy_ref_frame"] == 5
    # matching a ref CORE frame must NOT trip copy (that's M1a's business)
    gen2 = _unit(RNG.normal(size=(40, D))).astype(np.float32)
    gen2[25] = ref[24]
    r2 = copy_score(gen2, mm, ref, ref_core, tau=0.88)
    assert r2["near_copy"] is False


def test_intrusion_names_the_intruder():
    D = 16
    smoke = _unit(RNG.normal(size=(30, D))).astype(np.float32)
    gas = _unit(RNG.normal(size=(30, D))).astype(np.float32)
    gen = _unit(smoke[:20] + 0.05 * RNG.normal(size=(20, D))).astype(np.float32)
    pools = {"gas": gas, "smoke": smoke, "melt": _unit(RNG.normal(size=(30, D))).astype(np.float32)}
    r = intrusion_margin(gen, np.ones(20, dtype=bool), pools, target="gas")
    assert r["margin"] < 0 and r["intruder"] == "smoke"


def test_memorization_attributes_clip():
    D = 16
    pool_a = _unit(RNG.normal(size=(20, D))).astype(np.float32)
    pool_b = _unit(RNG.normal(size=(20, D))).astype(np.float32)
    gen = _unit(RNG.normal(size=(30, D))).astype(np.float32)
    gen[10] = pool_b[3]
    r = memorization_score(gen, np.ones(30, dtype=bool),
                           {"cls/a.mp4": pool_a, "cls/b.mp4": pool_b})
    assert r["mem_clip"] == "cls/b.mp4" and r["mem_max"] >= 0.999


# --- controls / manifests -----------------------------------------------------------

def test_static_hold_shapes_and_content():
    pre = RNG.integers(0, 255, (9, 32, 24, 3), dtype=np.uint8)
    suf = RNG.integers(0, 255, (8, 32, 24, 3), dtype=np.uint8)
    v = make_static_hold(pre, suf, 60)
    assert v.shape == (60, 32, 24, 3)
    assert (v[9:52] == pre[-1]).all()
    v2 = make_static_hold(pre, None, 60)
    assert v2.shape == (60, 32, 24, 3) and (v2[9:] == pre[-1]).all()


def _mini_corpus():
    return {"classes": {"gas": {"sidedness": "onesided", "tags": ["object"], "n_clips": 2}},
            "clips": {"gas/gas_0.mp4": {}, "gas/gas_1.mp4": {}, "gas/gas_2.mp4": {}},
            "std_contract": {}, "corpus_root": "corpus"}


def test_tier_derivation_abc():
    corpus = _mini_corpus()
    item = MF.EvalItemV3(item_id="x", generated_video="g.mp4",
                         reference_video="corpus/gas/gas_0.mp4", style="gas",
                         condition_prefix=MF.Condition("corpus/gas/gas_1.mp4", 9))
    assert MF.derive_tier(item, corpus, {"clips": ["melt/m0.mp4"], "_clipset": {"melt/m0.mp4"}}) == "A"
    assert MF.derive_tier(item, corpus, {"clips": ["gas/gas_2.mp4"], "_clipset": {"gas/gas_2.mp4"}}) == "B"
    assert MF.derive_tier(item, corpus, {"clips": ["gas/gas_0.mp4"], "_clipset": {"gas/gas_0.mp4"}}) == "C"
    assert MF.derive_tier(item, corpus, None) is None


def test_eval_manifest_rejects_unknown_and_duplicate(tmp_path):
    good = {"item_id": "a", "generated_video": "g.mp4",
            "reference_video": "r.mp4", "style": "gas"}
    p = tmp_path / "m.json"
    p.write_text(json.dumps([good, {**good, "bogus_key": 1}]))
    with pytest.raises(TypeError):
        MF.load_eval_manifest(p)
    p.write_text(json.dumps([good, good]))
    with pytest.raises(ValueError):
        MF.load_eval_manifest(p)
