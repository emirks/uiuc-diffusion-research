"""Synthetic contract tests for certify/ (SPEC §6) — CPU, no GPU, no corpus.
Loads the REAL bars.yaml so schema drift between bars and graders fails here,
not mid-certification."""

import json
import pathlib
import sys

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

yaml = pytest.importorskip("yaml")

from diffusion.transition_eval.certify import blockc, probes           # noqa: E402
from diffusion.transition_eval.certify.exam import (                    # noqa: E402
    class_sign_test, pool_margin_exam, sign_test_p, variant_core,
)

BARS = yaml.safe_load(
    (REPO_ROOT / "src/diffusion/transition_eval/certify/bars.yaml").read_text())
RNG = np.random.default_rng(11)


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def fake_bundle(core_dir, eA_dir, eB_dir, T=40, n_pre=5, n_suf=5, dim=24, noise=0.02):
    """Synthetic bundle: prefix frames near eA, suffix near eB, middle near core_dir."""
    f = np.concatenate([
        np.tile(eA_dir, (n_pre, 1)), np.tile(core_dir, (T - n_pre - n_suf, 1)),
        np.tile(eB_dir, (n_suf, 1))])
    f = _unit(f + noise * RNG.normal(size=f.shape)).astype(np.float32)
    a_hat = np.concatenate([np.ones(n_pre), np.zeros(T - n_pre - n_suf), np.ones(n_suf)])
    return {"feats": f,
            "profile": {"a_hat": a_hat, "b_hat": a_hat.copy(), "cross": 0.1,
                        "cross_high": False, "n_prefix": n_pre, "n_suffix": n_suf,
                        "n_endpoints": 2}}


# --- exam: sign test + adoption machinery -------------------------------------------

def test_sign_test_p_exact():
    assert sign_test_p(8, 0) == pytest.approx(1 / 256)
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 5) == pytest.approx(0.6230, abs=1e-4)
    assert sign_test_p(9, 1) < 0.05 < sign_test_p(7, 3)


def test_class_sign_test_counts_and_eligibility():
    new = {"a": 0.9, "b": 0.8, "c": 0.5, "d": 0.5, "e": 1.0}
    old = {"a": 0.5, "b": 0.9, "c": 0.5, "d": 0.2, "e": 0.9}
    r = class_sign_test(new, old, eligible={"a", "b", "c", "d"})   # e excluded
    assert (r["wins"], r["losses"], r["ties"]) == (2, 1, 1)
    assert r["p_one_sided"] == sign_test_p(2, 1)


def test_variant_core_all_frames_excludes_conditioned_windows():
    b = fake_bundle(_unit(RNG.normal(size=24)), _unit(RNG.normal(size=24)),
                    _unit(RNG.normal(size=24)))
    m, meta = variant_core(b, "onesided", "all_frames")
    assert not m[:5].any() and not m[-5:].any() and m[5:-5].all()
    assert meta["core_degenerate"] is False


def test_pool_margin_exam_loo_and_singletons():
    dim = 24
    dirs = {c: _unit(RNG.normal(size=dim)) for c in ("gas", "smoke", "lone")}
    eA, eB = _unit(RNG.normal(size=dim)), _unit(RNG.normal(size=dim))
    bundles, labels = [], []
    for c, n in (("gas", 3), ("smoke", 3), ("lone", 1)):
        for _ in range(n):
            bundles.append(fake_bundle(dirs[c], eA, eB))
            labels.append(c)
    r = pool_margin_exam(bundles, labels, ["twosided"] * len(labels), "all_frames")
    assert r["n_singleton_excluded"] == 1
    assert r["accuracy"] == 1.0                      # own-pool (LOO) still nearest
    assert r["per_class_recall"] == {"gas": 1.0, "smoke": 1.0}


# --- sibling selection + audit --------------------------------------------------------

def _mini_corpus(bundles_by_key):
    classes = {}
    for k in bundles_by_key:
        classes.setdefault(k.split("/")[0], {"sidedness": "twosided", "tags": [],
                                             "n_clips": 0})
        classes[k.split("/")[0]]["n_clips"] += 1
    return {"corpus_root": "data/x",
            "clips": {k: {"class": k.split("/")[0]} for k in bundles_by_key},
            "classes": classes}


def test_bar_pair_is_max_endpoint_distance():
    dim = 24
    core = _unit(RNG.normal(size=dim))
    e1, e2 = _unit(RNG.normal(size=dim)), _unit(RNG.normal(size=dim))
    e_far = _unit(-e1 + 0.1 * RNG.normal(size=dim))
    bk = {"c/a.mp4": fake_bundle(core, e1, e2), "c/b.mp4": fake_bundle(core, e1, e2),
          "c/z.mp4": fake_bundle(core, e_far, e2)}
    pairs = probes.sibling_pairs(bk, _mini_corpus(bk))
    assert set(pairs["c"]["bar_pair"]) & {"c/z.mp4"}   # the far-endpoint clip is in it
    assert len(pairs["c"]["pairs"]) == 3


def test_content_invariance_audit_centering():
    """Two classes at different mean levels but zero within-class relation ->
    pooled partial correlation ~0 (the per-class centering is the 'partial')."""
    dim = 32
    bundles, corpus_clips = {}, {}
    for cls, base_style in (("p", 0), ("q", 1)):
        style_dir = _unit(RNG.normal(size=dim))
        for i in range(6):
            eA = _unit(RNG.normal(size=dim))          # content: random per clip
            key = f"{cls}/{i}.mp4"
            bundles[key] = fake_bundle(style_dir, eA, eA, noise=0.05)
            corpus_clips[key] = {"class": cls}
    corpus = {"corpus_root": "d", "clips": corpus_clips,
              "classes": {"p": {"sidedness": "twosided", "tags": [], "n_clips": 6},
                          "q": {"sidedness": "twosided", "tags": [], "n_clips": 6}}}
    pairs = probes.sibling_pairs(bundles, corpus)
    audit = probes.content_invariance_audit(bundles, corpus, pairs)
    assert audit["n_pairs"] == 30
    assert abs(audit["pooled_partial_corr"]) < 0.45


# --- constructed probe videos ---------------------------------------------------------

def _write_clip(path, value, T=40, hw=(64, 48)):
    from diffusion.transition_eval.video_io import write_video
    frames = np.full((T, *hw, 3), value, dtype=np.uint8)
    write_video(frames, path)
    return path


def test_build_splice_uses_noncore_and_perturbs(tmp_path):
    from diffusion.transition_eval.video_io import load_frames
    gen = _write_clip(tmp_path / "gen.mp4", 100)
    ref_frames = np.full((40, 64, 48, 3), 50, dtype=np.uint8)   # core value 50
    ref_frames[:10] = 200                                       # non-core value 200
    ref_frames[-10:] = 200
    from diffusion.transition_eval.video_io import write_video
    write_video(ref_frames, tmp_path / "ref.mp4")
    ref_core = np.zeros(40, dtype=bool)
    ref_core[10:30] = True
    out = probes.build_splice(gen, tmp_path / "ref.mp4", ref_core,
                              tmp_path / "spl.mp4", segment_frames=12,
                              n_prefix=5, n_suffix=5)
    frames, _ = load_frames(out, short_side=None)
    mid = frames[len(frames) // 2].mean()
    assert abs(mid - 200) < 25 and abs(mid - 50) > 60   # non-core content, not core
    pert = BARS["probes"]["copy_splices"]["perturbation"]
    out2 = probes.build_splice(gen, tmp_path / "ref.mp4", ref_core,
                               tmp_path / "spl2.mp4", segment_frames=12,
                               n_prefix=5, n_suffix=5, perturb=pert)
    frames2, _ = load_frames(out2, short_side=None)
    assert not np.array_equal(frames[len(frames) // 2], frames2[len(frames2) // 2])


def test_build_hard_cut_switches_at_handoff(tmp_path):
    from diffusion.transition_eval.video_io import load_frames
    a = _write_clip(tmp_path / "a.mp4", 10)
    b = _write_clip(tmp_path / "b.mp4", 240)
    out = probes.build_hard_cut(a, b, tmp_path / "cut.mp4", n_prefix=9)
    frames, _ = load_frames(out, short_side=None)
    assert frames[7].mean() < 60 and frames[10].mean() > 180


# --- reversal ------------------------------------------------------------------------

def _cam(params):
    T = len(params)
    return {"params": np.asarray(params, dtype=np.float32),
            "Ms": np.tile(np.eye(2, dtype=np.float32), (T, 1, 1)),
            "ts": np.zeros((T, 2), dtype=np.float32),
            "n_points": np.full(T, 50), "valid": np.ones(T, dtype=bool)}


def test_reversal_sensitivity_asymmetric_vs_invariant():
    """The deployed statistic z-norms channels, so time-antisymmetric velocity
    profiles (constant pan, palindrome) are reversal-INVARIANT and must score
    ~0; a time-asymmetric profile (early burst) must score clearly above them.
    Absolute threshold validation happens on real cached tracks pre-freeze."""
    T = 60
    burst = np.zeros((T, 4)); burst[:8, 0] = 0.02                   # early event
    const = np.zeros((T, 4)); const[:, 0] = 0.004                   # steady pan
    pal = np.zeros((T, 4)); pal[:T // 2, 0] = 0.004; pal[T // 2:, 0] = -0.004
    s_burst = probes.reversal_sensitivity(_cam(burst))
    s_const = probes.reversal_sensitivity(_cam(const))
    s_pal = probes.reversal_sensitivity(_cam(pal))
    assert s_const == pytest.approx(0.0, abs=1e-6)
    assert s_pal == pytest.approx(0.0, abs=1e-6)
    assert s_burst > 10 * max(s_const, s_pal) and s_burst > 0.2


def test_grade_reversal_all_must_drop_branch():
    T = 60
    burst_a = np.zeros((T, 4)); burst_a[:8, 0] = 0.02       # early-event move
    burst_b = np.zeros((T, 4)); burst_b[1:9, 0] = 0.019     # same move, slight offset
    cams = {"c/a": _cam(burst_a), "c/b": _cam(burst_b)}
    rev_cams = {"c/b": probes.reversed_cam(_cam(burst_b))}
    pairs = [{"class": "c", "gen": "c/a", "ref": "c/b", "self_reversal": {}}]
    r = probes.grade_reversal(pairs, cams, rev_cams, BARS)
    assert r["pass"] is True and r["wins"] == 1 and "all-must-drop" in r["rule"]


# --- graders over rows -----------------------------------------------------------------

def _bars_rows(cls="c"):
    return {
        f"sib__{cls}": {"item_id": f"sib__{cls}", "arm": "probe_sibling",
                        "app_ref": 0.7, "near_copy": False, "copy_max": 0.62,
                        "prefix_dino": 0.99, "sidedness": "twosided"},
        f"control_lerp__sib__{cls}": {"item_id": f"control_lerp__sib__{cls}",
                                      "arm": "control_lerp", "app_ref": 0.4,
                                      "core_degenerate": True, "sidedness": "twosided"},
        f"splice_verbatim__{cls}": {"item_id": f"splice_verbatim__{cls}",
                                    "arm": "probe_splice_verbatim", "copy_max": 0.97},
        f"splice_perturbed__{cls}": {"item_id": f"splice_perturbed__{cls}",
                                     "arm": "probe_splice_perturbed", "copy_max": 0.93},
        f"swap__{cls}": {"item_id": f"swap__{cls}", "arm": "probe_swap",
                         "prefix_dino": 0.55},
        f"hardcut__{cls}": {"item_id": f"hardcut__{cls}", "arm": "probe_hardcut",
                            "max_seam_z": 9.4},
    }


def test_graders_pass_and_fail_paths():
    import copy
    bars = copy.deepcopy(BARS)
    for key in ("siblings", "controls"):
        bars["probes"][key][f"bar{2 if key == 'siblings' else 3}"]["min_classes"] = 1
    bars["probes"]["m3_panel"]["bar6_endpoint_swap"]["min_classes"] = 1
    bars["probes"]["m3_panel"]["bar6_hard_cut"]["min_classes"] = 1
    rows = _bars_rows()
    assert probes.grade_siblings(rows, ["c"], bars)["pass"]
    assert probes.grade_controls(rows, ["c"], bars)["pass"]
    spl = probes.grade_splices(rows, ["c"], bars)
    assert spl["pass"] and spl["tau_recalibrated"] == pytest.approx(0.5 * (0.93 + 0.62))
    assert probes.grade_m3_panel(rows, ["c"], bars)["pass"]
    bad = {**rows, f"sib__c": {**rows["sib__c"], "near_copy": True}}
    assert not probes.grade_siblings(bad, ["c"], bars)["pass"]
    bad2 = {**rows, "splice_perturbed__c": {**rows["splice_perturbed__c"], "copy_max": 0.70}}
    assert not probes.grade_splices(bad2, ["c"], bars)["pass"]


# --- Block C ---------------------------------------------------------------------------

def test_convert_v2_manifest_recovers_reference_and_excludes_loudly(tmp_path):
    gen = tmp_path / "outputs/videos/g.mp4"
    gen.parent.mkdir(parents=True)
    gen.write_bytes(b"x")
    man = [{"item_id": "ok", "generated_video": "outputs/videos/g.mp4",
            "style": "hero", "n_endpoints": 2,
            "condition_prefix": {"video": "exp/c.mp4", "num_frames": 9},
            "condition_suffix": None, "arm": "ic",
            "notes": "endpoints=hero_1 (hero); reference=hero_4 (hero; camera)"},
           {"item_id": "missing_video", "generated_video": "outputs/videos/nope.mp4",
            "style": "hero", "notes": "reference=hero_4"},
           {"item_id": "no_ref", "generated_video": "outputs/videos/g.mp4",
            "style": "hero", "notes": "no marker here"}]
    (tmp_path / "m.json").write_text(json.dumps(man))
    corpus = {"corpus_root": "data/t", "clips": {"hero/hero_4.mp4": {"class": "hero"}},
              "classes": {"hero": {"sidedness": "onesided", "tags": [], "n_clips": 1}}}
    items, excluded = blockc.convert_v2_manifest(tmp_path / "m.json", corpus, tmp_path)
    assert len(items) == 1 and items[0]["item_id"] == "ok"
    assert items[0]["reference_video"].endswith("hero/hero_4.mp4")
    assert {e["item_id"] for e in excluded} == {"missing_video", "no_ref"}


def test_grade_copy_twins_requires_all():
    rows = {f"t{i}": {"near_copy": True, "max_seam_z": 0.1, "copy_max": 0.97}
            for i in range(11)}
    ids = [f"t{i}" for i in range(11)]
    assert blockc.grade_copy_twins(rows, ids)["pass"]
    rows["t3"] = {"near_copy": False, "max_seam_z": 1.2, "copy_max": 0.7}
    r = blockc.grade_copy_twins(rows, ids)
    assert not r["pass"] and r["n_pass"] == 10


# --- draft.8 minimal reliability fixes ---------------------------------------------------

from diffusion.transition_eval.m1_transfer import (                     # noqa: E402
    camera_trajectory, object_match,
)


def test_object_match_empty_keep_returns_nan():
    """The draft.7 killer: low-texture video -> keep-filter empties -> must be
    NaN, not an apply_along_axis crash (killed bars 4/6/7's data)."""
    T, N = 30, 12
    tracks = RNG.normal(size=(T, N, 2)).astype(np.float32)
    vis_dead = np.zeros((T, N), dtype=np.float32)
    vis_ok = np.ones((T, N), dtype=np.float32)
    assert np.isnan(object_match(tracks, vis_dead, tracks, vis_ok))
    assert np.isnan(object_match(tracks, vis_ok, tracks, vis_dead))


def test_camera_trajectory_zero_tracklets_no_crash():
    T = 20
    cam = camera_trajectory(np.zeros((T, 0, 2), dtype=np.float32),
                            np.zeros((T, 0), dtype=np.float32))
    assert cam["valid"].sum() == 0 and len(cam["params"]) == T - 1


def test_anchor_ids_dedup_across_strata():
    from diffusion.transition_eval.certify.run_certification import anchor_ids
    corpus = {"classes": {
        "aa": {"sidedness": "twosided", "n_clips": 4, "tags": ["camera"]},
        "bb": {"sidedness": "onesided", "n_clips": 5, "tags": []},
        "cc": {"sidedness": "twosided", "n_clips": 6, "tags": ["camera"]},
    }}
    pairs = {"aa": {}, "bb": {}, "cc": {}}
    c_items = [{"item_id": "exp_057__base_x", "arm": "base"},
               {"item_id": "exp_057__ic_x", "arm": "ic"}]
    ids = anchor_ids(pairs, corpus, c_items, BARS)
    assert len(set(ids)) == 6
    assert "sib__aa" in ids and "sib__bb" in ids and "sib__cc" in ids


def test_grade_controls_floor_only():
    """draft.8 bar 3: the floor claim alone gates; core_degenerate is
    descriptive (owner decision — vacuous conjunct removed)."""
    import copy
    bars = copy.deepcopy(BARS)
    bars["probes"]["controls"]["bar3"]["min_classes"] = 1
    sib = {"item_id": "sib__c", "arm": "probe_sibling", "app_ref": 0.7}
    ctrl = {"item_id": "control_lerp__sib__c", "arm": "control_lerp",
            "app_ref": 0.4, "core_degenerate": False}   # draft.7 failure mode
    assert probes.grade_controls({"sib__c": sib, ctrl["item_id"]: ctrl},
                                 ["c"], bars)["pass"]
    high = {**ctrl, "app_ref": 0.8}                     # floor clause still gates
    assert not probes.grade_controls({"sib__c": sib, high["item_id"]: high},
                                     ["c"], bars)["pass"]
    err = {"item_id": "control_lerp__sib__c", "arm": "control_lerp",
           "error": "boom"}                             # error row = documented miss
    assert not probes.grade_controls({"sib__c": sib, err["item_id"]: err},
                                     ["c"], bars)["pass"]


# --- perf changes: lazy decode + parallel motion matrices (numeric no-ops) ----------

def test_process_video_file_skips_decode_when_warm(tmp_path, monkeypatch):
    """need_frames=False + warm caches -> no decode, same bundle content."""
    from diffusion.transition_eval import pipeline
    from diffusion.transition_eval.features import feature_cache_path, file_key

    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"not-a-real-video")          # never decoded on the warm path

    class StubExtractor:
        model_name = "stub-model"
        def extract(self, frames):                # must never run
            raise AssertionError("extract called despite warm cache")

    ext = StubExtractor()
    key = file_key(vid, ext.model_name, "256")
    T, D = 12, 8
    feats = np.random.default_rng(0).random((T, D)).astype(np.float32)
    np.savez_compressed(feature_cache_path(key, tmp_path), feats=feats, src=key)

    calls = {"decode": 0}
    def fake_load(path, short_side=256):
        calls["decode"] += 1
        raise AssertionError("load_frames called despite need_frames=False + warm cache")
    monkeypatch.setattr(pipeline, "load_frames", fake_load)
    monkeypatch.setattr(pipeline, "probe_fps", lambda p: 24.0)

    b, frames = pipeline.process_video_file(vid, tmp_path, ext, tracker=None,
                                            short_side=256, need_frames=False,
                                            n_prefix=2, n_suffix=2, n_endpoints=1)
    assert frames is None and calls["decode"] == 0
    assert np.array_equal(b["feats"], feats) and b["fps"] == 24.0

    # cold cache must still decode (flag is opportunistic, never wrong)
    vid2 = tmp_path / "v2.mp4"
    vid2.write_bytes(b"x")
    def fake_load2(path, short_side=256):
        return np.zeros((T, 4, 4, 3), dtype=np.uint8), 24.0
    monkeypatch.setattr(pipeline, "load_frames", fake_load2)
    class StubExtractor2(StubExtractor):
        def extract(self, frames):
            return np.ones((len(frames), D), dtype=np.float32)
    b2, f2 = pipeline.process_video_file(vid2, tmp_path, StubExtractor2(),
                                         tracker=None, short_side=256,
                                         need_frames=False,
                                         n_prefix=2, n_suffix=2, n_endpoints=1)
    assert f2 is not None and b2["feats"].shape == (T, D)


def test_motion_distance_matrices_parallel_matches_serial():
    """Fork-pool pairwise loop must be bit-identical to the serial loop."""
    from diffusion.transition_eval.certify.exam import motion_distance_matrices

    rng = np.random.default_rng(7)
    bundles = []
    for _ in range(9):
        T, P = 20, 30
        base = rng.random((1, P, 2)).astype(np.float32)
        drift = rng.normal(0, 0.01, (T, P, 2)).astype(np.float32).cumsum(0)
        bundles.append({"tracks": np.clip(base + drift, 0, 1),
                        "vis": np.ones((T, P), dtype=np.float32)})
    serial = motion_distance_matrices(bundles, n_jobs=1)
    parallel = motion_distance_matrices(bundles, n_jobs=2, min_pairs_for_pool=1)
    for k in serial:
        np.testing.assert_array_equal(serial[k], parallel[k])


# ---- diagnostics / representation layer ---------------------------------------------

def test_per_clip_rows_margins_and_masking():
    from diffusion.transition_eval.certify import diagnostics as dg
    keys, labels = ["k0", "k1", "k2"], ["a", "a", "b"]
    D = np.array([[0.0, 0.2, 0.5],
                  [0.2, 0.0, 0.1],
                  [0.5, 0.1, 0.0]])
    rows = dg.per_clip_rows(D, keys, labels)
    assert rows[0]["pred"] == "a" and rows[0]["nn_key"] == "k1"
    assert abs(rows[0]["margin"] - 0.3) < 1e-12
    # clip 1's nearest neighbour is the cross-class k2: misretrieved, margin < 0
    assert rows[1]["pred"] == "b" and abs(rows[1]["margin"] + 0.1) < 1e-12
    D2 = D.copy()
    D2[2, :] = np.nan
    assert dg.per_clip_rows(D2, keys, labels)[2]["pred"] is None


def test_tag_accuracy_coarse_and_patterns():
    from diffusion.transition_eval.certify import diagnostics as dg
    clips = [
        {"key": "k0", "class": "a", "sidedness": "twosided", "tags": ["object"]},
        {"key": "k1", "class": "a", "sidedness": "twosided", "tags": ["object"]},
        {"key": "k2", "class": "b", "sidedness": "onesided", "tags": ["style", "camera"]},
    ]
    rows = [{"key": "k0", "label": "a", "pred": "a"},
            {"key": "k1", "label": "a", "pred": "b"},
            {"key": "k2", "label": "b", "pred": "b"}]
    bt = dg.tag_accuracy({"m": rows}, clips)
    coarse = {r["group"]: r for r in bt["coarse"]}
    assert coarse["twosided"]["n"] == 2 and coarse["twosided"]["m"] == 0.5
    assert coarse["object"]["m"] == 0.5
    assert coarse["camera"]["m"] == 1.0 and coarse["camera"]["n"] == 1
    pats = {r["group"]: r for r in bt["patterns"]}
    assert set(pats) == {"twosided_object", "onesided_style_camera"}
    assert pats["twosided_object"]["m"] == 0.5


def test_build_and_write_analysis_schema(tmp_path):
    from diffusion.transition_eval.certify import diagnostics as dg
    corpus = {"clips": {
        "alpha/a0.mp4": {"class": "alpha", "source": "d/twosided_object_alpha/a0.mp4"},
        "alpha/a1.mp4": {"class": "alpha", "source": "d/twosided_object_alpha/a1.mp4"},
        "beta/b0.mp4": {"class": "beta", "source": "d/onesided_style_beta/b0.mp4"},
    }}
    keys = sorted(corpus["clips"])
    labels = [corpus["clips"][k]["class"] for k in keys]
    sidedness = ["twosided", "twosided", "onesided"]
    D = np.array([[0.0, 0.1, 0.6], [0.1, 0.0, 0.5], [0.6, 0.5, 0.0]])
    r1 = {"m": {"accuracy_1nn": 1.0, "chance": 0.5,
                "confusion": {"alpha": {"alpha": 2}, "beta": {"alpha": 1}}}}
    r2 = {"accuracy": 1.0, "n_graded": 2, "n_singleton_excluded": 1,
          "per_class_recall": {"alpha": 1.0}, "margins_mean": 0.2,
          "rows": [{"label": "alpha", "margin": 0.2, "correct": True},
                   {"label": "alpha", "margin": 0.2, "correct": True},
                   {"label": "beta", "margin": None, "correct": None}]}
    ana = dg.build_analysis(corpus, keys, labels, sidedness, r1, {"m": D}, r2, "v3_sided")
    assert [c["tags"] for c in ana["clips"]] == [["object"], ["object"], ["style"]]
    assert ana["metrics"]["m"]["retrieval"] is r1["m"]
    assert ana["metrics"]["m"]["rows"][0]["key"] == keys[0]
    assert ana["r2"]["rows"][0]["key"] == keys[0]
    assert ana["r2"]["winner_mask"] == "v3_sided"
    assert {r["group"] for r in ana["by_tag"]["patterns"]} == \
        {"twosided_object", "onesided_style"}
    out = dg.write_analysis(tmp_path, ana, {"m": D}, keys)
    assert out.exists()
    z = np.load(tmp_path / "distance_matrices.npz")
    assert list(z["keys"]) == keys and np.array_equal(z["m"], D)
    loaded = json.loads(out.read_text())
    assert set(loaded) == {"clips", "metrics", "r2", "by_tag"}
