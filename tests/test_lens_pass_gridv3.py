"""Unit tests for the Op-4 competitor-lens deliverables (CPU, no GPU, no store extraction):

  * scripts/lens_pass_gridv3.py  — harness-arm derivation, the (item_id, seed) join that reconciles
    score_batch's filename-stem item_id to the GRID item_id via Op-2's per_gen.jsonl, and the
    collect/merge of per-gen JSON into rows.jsonl.
  * scripts/aesthetic_from_store.py — the LAION head loads (shape-checked) and scores a random
    feature array to a finite value.
  * scripts/lens_gate_check.py — the pure per-lens max|Δ| comparator + the exit-1 (embedding > 1e-3)
    / report-only (det_motion_fidelity, 1e-2) decision logic.

Run: pytest tests/test_lens_pass_gridv3.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import lens_pass_gridv3 as L      # noqa: E402
import lens_gate_check as G       # noqa: E402
import aesthetic_from_store as A  # noqa: E402


# --------------------------------------------------------------------------- #
# harness-arm derivation (strip rule fallback + real-population agreement)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant_rel,expect", [
    ("store/gens/999_ic_gen/03_neutral_v3__dai", "ic_gen_neutral_v3"),
    ("store/gens/999_ic_gen/04_neutral_v3ed81__dai", "ic_gen_neutral_v3ed81"),
    ("store/gens/999_base_cond/06_effect_v3__dai", "base_cond_effect_v3"),
    ("store/gens/999_dualforce_dcg_w6/03_neutral_v3__dai", "dualforce_dcg_w6_neutral_v3"),
    ("store/gens/999_vap/05_author_native__dai", "vap_author_native"),
    ("store/gens/999_refvfx/03_author_native__eps", "refvfx_author_native"),
])
def test_harness_arm_strip_rule(variant_rel, expect):
    # 999_ dirs do not exist -> no meta.yaml -> the strip rule is exercised.
    assert L.variant_harness_arm(variant_rel) == expect


def test_harness_arm_matches_population_meta():
    """Every real gridv3 variant's meta.yaml harness_arm is what the join uses (matches Op-2/Op-3)."""
    pop = L.load_population()
    arms = [L.variant_harness_arm(v) for v in pop["gen_variants"]]
    assert len(arms) == 19 and len(set(arms)) == 19
    # a few frozen expectations
    m = dict(zip(pop["gen_variants"], arms))
    assert m["store/gens/011_vap/05_author_native__dai"] == "vap_author_native"
    assert m["store/gens/001_ic_gen/03_neutral_v3__dai"] == "ic_gen_neutral_v3"
    assert m["store/gens/032_dualforce_dcg_w6/06_effect_v3ed81__dai"] == "dualforce_dcg_w6_effect_v3ed81"


# --------------------------------------------------------------------------- #
# item-id reconciliation
# --------------------------------------------------------------------------- #
def test_grid_ids_from_stem():
    assert L._grid_ids_from_stem(
        "G-fit__ic_gen_neutral_v3__animalization_3__ref_animalization_1__s42", None
    ) == ("G-fit__ic_gen_neutral_v3__animalization_3__ref_animalization_1", 42)
    # no seed token -> stem verbatim, seed falls back to score_batch's parsed seed
    assert L._grid_ids_from_stem("weird_name_no_seed", 7) == ("weird_name_no_seed", 7)


def _fake_scored(stem, arm_group="011_vap", arm_variant="05_author_native__dai",
                 warnings=None, **lenses):
    gen = str(REPO / "store" / "gens" / arm_group / arm_variant / "videos" / f"{stem}.mp4")
    row = {"item_id": stem, "gen": gen, "seed": int(stem.rsplit("__s", 1)[1]),
           "warnings": warnings or [], "impl_sha": "8a808635e8a08cb4"}
    for k in L.LENS_KEYS:
        row[k] = lenses.get(k, 0.5)
    return row, gen


def test_merge_row_pergen_join():
    stem = "G-zs-cross__vap_author_native__animalization_0__ref_acid_0__s42"
    scored, gen = _fake_scored(stem, clip_sim_ref=0.72, det_motion_fidelity=0.33)
    grid_id = "G-zs-cross__vap_author_native__animalization_0__ref_acid_0"
    idx = {L._relpath(gen): (grid_id, 42)}
    row, status = L.merge_row(scored, idx, "vap_author_native")
    assert status == "pergen"
    assert row["item_id"] == grid_id and row["seed"] == 42
    assert row["arm"] == "vap_author_native"
    assert row["clip_sim_ref"] == 0.72 and row["det_motion_fidelity"] == 0.33
    assert set(L.LENS_KEYS).issubset(row) and row["impl_sha"] == "8a808635e8a08cb4"
    assert "no_pergen_join" not in row["warnings"]


def test_merge_row_fallback_and_warning_passthrough():
    stem = "G-zs-cross__vap_author_native__animalization_2__ref_acid_1__s43"
    scored, _ = _fake_scored(stem, warnings=["missing input_frame /x/y.png"])
    row, status = L.merge_row(scored, {}, "vap_author_native")   # empty index -> fallback
    assert status == "fallback"
    assert row["item_id"] == "G-zs-cross__vap_author_native__animalization_2__ref_acid_1"
    assert row["seed"] == 43
    assert "no_pergen_join" in row["warnings"]
    assert any(w.startswith("missing input_frame") for w in row["warnings"])  # preserved


def test_collect_arm_end_to_end(tmp_path):
    arm = "vap_author_native"
    eval_dir = tmp_path / "039_lenses_gridv3__dai__2026-09-18"
    rows_dir = eval_dir / arm / "rows"
    rows_dir.mkdir(parents=True)
    idx = {}
    # gen A: joined, all lenses finite, no missing frame
    stemA = "G-zs-cross__vap_author_native__animalization_0__ref_acid_0__s42"
    sA, genA = _fake_scored(stemA, clip_sim_ref=0.7, videoprism_sim_ref=0.95)
    (rows_dir / f"{stemA}.json").write_text(json.dumps(sA))
    idx[L._relpath(genA)] = ("G-zs-cross__vap_author_native__animalization_0__ref_acid_0", 42)
    # gen B: joined, det_motion_fidelity None (NaN), missing input frame warning
    stemB = "G-zs-cross__vap_author_native__animalization_0__ref_acid_0__s43"
    sB, genB = _fake_scored(stemB, warnings=["missing input_frame /z.png"], det_motion_fidelity=None)
    (rows_dir / f"{stemB}.json").write_text(json.dumps(sB))
    idx[L._relpath(genB)] = ("G-zs-cross__vap_author_native__animalization_0__ref_acid_0", 43)

    st = L.collect_arm(eval_dir, "store/gens/011_vap/05_author_native__dai", arm, pergen_index=idx)
    assert st["n"] == 2 and st["pergen_join"] == 2 and st["fallback_join"] == 0
    assert st["missing_input_frame"] == 1
    assert st["lens_finite"]["clip_sim_ref"] == 2
    assert st["lens_finite"]["det_motion_fidelity"] == 1   # B is None -> not finite

    out = (eval_dir / arm / "rows.jsonl").read_text().splitlines()
    got = [json.loads(x) for x in out]
    assert len(got) == 2
    # sorted by (item_id, seed): both share item_id, so seed 42 then 43
    assert [r["seed"] for r in got] == [42, 43]
    assert all(r["arm"] == arm for r in got)
    assert all(set(["item_id", "seed", "arm", "gen", "impl_sha", "warnings"]).issubset(r) for r in got)


# --------------------------------------------------------------------------- #
# aesthetic head
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not A.DEFAULT_HEAD.exists(), reason="LAION head not cached")
def test_aesthetic_head_finite():
    head = A.load_aesthetic_head(device="cpu")   # also asserts the 5 layer shapes
    rng = np.random.default_rng(0)
    x = rng.standard_normal((24, 768)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    s = A.aesthetic_from_feats(x, head)
    assert np.isfinite(s)
    # single frame is finite; empty is NaN
    assert np.isfinite(A.aesthetic_from_feats(x[0], head))
    assert np.isnan(A.aesthetic_from_feats(np.zeros((0, 768), np.float32), head))
    # the head re-normalizes internally, so an unnormalized copy scores identically
    assert abs(A.aesthetic_from_feats(x, head) - A.aesthetic_from_feats(3.0 * x, head)) < 1e-4


# --------------------------------------------------------------------------- #
# gate-check comparator
# --------------------------------------------------------------------------- #
def _pair(frozen_vals, rec_vals):
    return (dict(frozen_vals), dict(rec_vals))


def test_gate_compare_pass():
    # all embedding deltas < 1e-4
    pairs = [_pair(
        {"clip_sim_ref": 0.7000000, "videoprism_sim_ref": 0.95, "motion_smoothness": 0.98,
         "clip_sim_input": 0.9, "videoprism_sim_input": 0.97, "dynamic_degree_mean_mag": 1.98,
         "det_motion_fidelity": 0.33},
        {"clip_sim_ref": 0.7000050, "videoprism_sim_ref": 0.95, "motion_smoothness": 0.98,
         "clip_sim_input": 0.9, "videoprism_sim_input": 0.97, "dynamic_degree_mean_mag": 1.98,
         "det_motion_fidelity": 0.33})]
    rep = G.compare(pairs)
    assert rep["fail"] is False
    assert rep["lenses"]["clip_sim_ref"]["n_over_tol"] == 0   # 5e-6 < 1e-4
    assert rep["lenses"]["clip_sim_ref"]["n"] == 1


def test_gate_compare_fail_on_embedding():
    pairs = [_pair({"clip_sim_ref": 0.70}, {"clip_sim_ref": 0.70 + 2e-3})]
    rep = G.compare(pairs)
    assert rep["fail"] is True
    assert rep["worst_emb"][0] == "clip_sim_ref"
    assert rep["worst_emb"][1] == pytest.approx(2e-3, abs=1e-9)
    assert rep["lenses"]["clip_sim_ref"]["n_over_tol"] == 1   # over the 1e-4 report tol


def test_gate_compare_track_does_not_gate():
    # det_motion_fidelity off by 2e-2 (> its 1e-2 report tol) but it must NOT set fail.
    pairs = [_pair({"det_motion_fidelity": 0.30, "clip_sim_ref": 0.70},
                   {"det_motion_fidelity": 0.32, "clip_sim_ref": 0.70})]
    rep = G.compare(pairs)
    assert rep["fail"] is False
    assert rep["lenses"]["det_motion_fidelity"]["n_over_tol"] == 1   # 2e-2 > 1e-2
    # a 5e-3 track delta is within its own 1e-2 tol
    rep2 = G.compare([_pair({"det_motion_fidelity": 0.30}, {"det_motion_fidelity": 0.305})])
    assert rep2["lenses"]["det_motion_fidelity"]["n_over_tol"] == 0
    assert rep2["fail"] is False


def test_gate_compare_skips_none_and_nan():
    pairs = [
        _pair({"clip_sim_ref": None, "videoprism_sim_ref": 0.9},
              {"clip_sim_ref": 0.7, "videoprism_sim_ref": 0.9}),          # clip skipped (None)
        _pair({"clip_sim_ref": 0.7, "videoprism_sim_ref": float("nan")},
              {"clip_sim_ref": 0.7, "videoprism_sim_ref": 0.9}),          # vp skipped (NaN)
    ]
    rep = G.compare(pairs)
    assert rep["lenses"]["clip_sim_ref"]["n"] == 1        # only the second pair counted
    assert rep["lenses"]["videoprism_sim_ref"]["n"] == 1  # only the first pair counted
    assert rep["fail"] is False
