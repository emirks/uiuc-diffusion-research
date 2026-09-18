"""Tests for scripts/copy_metrics.py — the M2a copy rate vs each gen's OWN reference.

Synthetic feature fixtures exercise the copy_score contract as copy_metrics wires it:
  - a gen whose MID frames equal the reference's NON-core frames -> copy_max ~= 1, near_copy True;
  - a gen with disjoint features -> low copy_max, near_copy False;
  - mid_mask sizes per grid type (HF one/two-sided, ED, external).

The scored path reuses the harness copy_score / mid_mask verbatim; these tests pin the wiring
(mask construction, reference core via morph_profile + core_mask_v3, the row schema) rather than
re-deriving the metric.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diffusion.transition_eval.m2_integrity import TAU_COPY, copy_score, mid_mask
from diffusion.transition_eval.morph import morph_profile
from diffusion.transition_eval.s_structure import core_mask_v3

# Load scripts/copy_metrics.py as a module (scripts/ is not a package).
_spec = importlib.util.spec_from_file_location(
    "copy_metrics", REPO_ROOT / "scripts" / "copy_metrics.py")
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _ref_onesided(T: int = 60, D: int = 32, seed: int = 0):
    """A one-sided reference: endpoint A frames (first 9) point one way, a distinct 'medium'
    fills the core, and the tail drifts toward a second appearance. core_mask_v3 keeps the
    'neither endpoint' middle; the NON-core frames are the endpoint scenes A/B."""
    rng = np.random.default_rng(seed)
    dirA = _unit(rng.normal(size=D))
    dirMed = _unit(rng.normal(size=D))
    dirB = _unit(rng.normal(size=D))
    feats = np.zeros((T, D), dtype=np.float32)
    for t in range(T):
        if t < 9:
            base = dirA
        elif t > T - 9:
            base = dirB
        else:
            base = dirMed
        f = base + 0.02 * rng.normal(size=D)
        feats[t] = _unit(f)
    return feats


def test_gen_mid_equals_ref_noncore_is_copy():
    """MID frames of the gen copied verbatim from the reference's NON-core frames -> copy."""
    ref = _ref_onesided()
    profile = morph_profile(ref, n_prefix=9, n_suffix=8, n_endpoints=2)
    ref_core, _ = core_mask_v3(profile, "onesided")
    noncore_idx = np.flatnonzero(~ref_core)
    assert len(noncore_idx) > 0

    # HF one-sided gen: n_pre=9, n_suf=0. Build a gen whose mid frames literally reuse the
    # reference's non-core (endpoint) frames -> the max cosine is 1.0 -> near_copy True.
    T_g = 40
    gen = _unit(np.random.default_rng(1).normal(size=(T_g, ref.shape[1]))).astype(np.float32)
    gen_mid = mid_mask(T_g, 9, 0)
    mid_idx = np.flatnonzero(gen_mid)
    for j, gi in enumerate(mid_idx):
        gen[gi] = ref[noncore_idx[j % len(noncore_idx)]]

    res = copy_score(gen, gen_mid, ref, ref_core, TAU_COPY)
    assert res["copy_max"] > 0.999
    assert res["near_copy"] is True
    assert res["copy_ref_frame"] in set(noncore_idx.tolist())


def test_disjoint_features_low_copy():
    """A gen on an orthogonal subspace from the reference -> low copy_max, near_copy False."""
    ref = _ref_onesided(seed=2)
    D = ref.shape[1]
    profile = morph_profile(ref, n_prefix=9, n_suffix=8, n_endpoints=2)
    ref_core, _ = core_mask_v3(profile, "onesided")

    # Place the reference in the first half of the dims and the gen in the second half so every
    # cross cosine is ~0.
    ref2 = np.zeros_like(ref)
    ref2[:, : D // 2] = ref[:, : D // 2]
    ref2 = _unit(ref2).astype(np.float32)
    gen = np.zeros((30, D), dtype=np.float32)
    gen[:, D // 2:] = np.random.default_rng(3).normal(size=(30, D - D // 2))
    gen = _unit(gen).astype(np.float32)
    gen_mid = mid_mask(30, 9, 0)

    res = copy_score(gen, gen_mid, ref2, ref_core, TAU_COPY)
    assert res["copy_max"] < 0.3
    assert res["near_copy"] is False


def test_windows_and_gridtype_rules():
    """n_pre/n_suf follow the handoff_metrics rule; grid_type routes arm/variant correctly."""
    # grid classification
    assert cm.grid_type("refvfx", "author_native") == "external"
    assert cm.grid_type("vap", "author_native") == "external"
    assert cm.grid_type("ic_gen", "neutral_v3") == "HF"
    assert cm.grid_type("ic_gen", "neutral_v3ed81") == "ED"
    assert cm.grid_type("dualforce_control", "effect_v3ed81") == "ED"
    # windows per grid type / sidedness
    assert cm.windows("HF", "one") == (9, 0)
    assert cm.windows("HF", "two") == (9, 8)
    assert cm.windows("ED", "one") == (1, 0)
    assert cm.windows("external", "one") == (1, 0)


def test_mid_mask_sizes():
    """mid_mask leaves n_pre..T (one-sided) / n_pre..T-n_suf (two-sided)."""
    # HF one-sided, T=121: mid = 121-9 = 112 frames
    assert int(mid_mask(121, 9, 0).sum()) == 121 - 9
    # HF two-sided, T=121: mid = 121-9-8 = 104
    assert int(mid_mask(121, 9, 8).sum()) == 121 - 9 - 8
    # ED, T=81, n_pre=1: mid = 80
    assert int(mid_mask(81, 1, 0).sum()) == 80
    # external refVFX-length, T=33, n_pre=1: mid = 32
    assert int(mid_mask(33, 1, 0).sum()) == 32


def test_compute_row_missing_when_no_reference():
    """A grid row whose reference is unresolvable yields a NaN row + `missing` entries, no crash."""
    class _FS:
        def has(self, *a, **k):
            return False

        def get(self, *a, **k):  # never reached (has -> False)
            raise AssertionError

    feats = cm.Feats(_FS())

    class _Corpus:
        def resolve(self, row):
            return None, None, None

    row = {"reference": "does_not_exist", "sided": "one"}
    r = cm.compute_row(feats, _Corpus(), Path("videos/foo__s42.mp4"), row, "HF")
    assert r["item_id"] == "foo" and r["seed"] == 42
    assert r["n_pre"] == 9 and r["n_suf"] == 0
    assert np.isnan(r["copy_max"]) and r["near_copy"] is None
    assert "reference_unresolved" in r["missing"]
    assert f"{cm.DINO_NS}:gen" in r["missing"]
    # row schema is exactly the fixed set of keys
    assert set(r.keys()) == {"item_id", "seed", "arm", "n_pre", "n_suf", "n_mid",
                             "ref_core_frac", "copy_max", "near_copy",
                             "copy_gen_frame", "copy_ref_frame", "missing"}
