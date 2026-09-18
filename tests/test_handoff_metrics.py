"""Tests for scripts/handoff_metrics.py — the hand-off metric definitions.

Synthetic features with known cosines and a coherent (identical vs reversed) track set. No GPU,
no real video decode (``probe_fps`` is monkeypatched); features are written through the real
``FeatureStore`` into a temp repo so the path-is-identity resolution is exercised end to end.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import handoff_metrics as hm  # noqa: E402
from diffusion.feature_store import FeatureStore  # noqa: E402
from diffusion.transition_eval.motion import motion_fidelity  # noqa: E402

DINO, TRACK = hm.DINO_NS, hm.TRACK_NS


# --- pure helpers -----------------------------------------------------------
def test_grid_type():
    assert hm.grid_type("ic_gen", "neutral_v3") == "HF"
    assert hm.grid_type("ic_gen", "effect_v3") == "HF"
    assert hm.grid_type("ic_gen", "neutral_v3ed81") == "ED"
    assert hm.grid_type("base_cond", "effect_v3ed81") == "ED"
    for a in ("refvfx", "vap", "vfxmaster"):
        assert hm.grid_type(a, "author_native") == "external"


def test_windows():
    assert hm.windows("HF", "one") == (9, 0)
    assert hm.windows("HF", "two") == (9, 8)
    assert hm.windows("ED", "one") == (1, 0)
    assert hm.windows("external", "one") == (1, 0)


def test_parse_and_match_stem():
    stem = "G-fit__ic_gen_neutral_v3__animalization_3__ref_animalization_1__s42"
    iid, seed = hm._parse_stem(stem)
    assert iid == "G-fit__ic_gen_neutral_v3__animalization_3__ref_animalization_1"
    assert seed == 42
    vidset = {stem}
    v4_id = stem + "__ref_animalization_0"
    assert hm._match_stem(v4_id, vidset) == stem
    assert hm._match_stem("nope__s99__ref_x", vidset) is None


def test_cos_window():
    u = np.zeros((20, 4), np.float32)
    u[:, 0] = 1.0                                  # every frame = +x unit vector
    anchor = np.array([1, 0, 0, 0], np.float32)
    assert hm._cos_window(u, anchor, 5, 9) == pytest.approx(1.0, abs=1e-6)
    ortho = np.array([0, 1, 0, 0], np.float32)
    assert hm._cos_window(u, ortho, 5, 9) == pytest.approx(0.0, abs=1e-6)
    at60 = np.array([0.5, math.sqrt(3) / 2, 0, 0], np.float32)   # 60 deg from +x
    assert hm._cos_window(u, at60, 5, 9) == pytest.approx(0.5, abs=1e-6)
    assert math.isnan(hm._cos_window(u, anchor, 25, 30))         # empty window -> NaN


def test_motion_fidelity_identical_vs_reversed():
    T, N = 20, 25
    p0 = np.random.default_rng(1).uniform(50, 300, size=(N, 2)).astype(np.float32)
    vel = np.zeros((N, 2), np.float32)
    vel[:, 0] = 3.0                                # coherent global +x motion
    t = np.arange(T)[:, None, None]
    tracks = (p0[None] + vel[None] * t).astype(np.float32)
    vis = np.ones((T, N), np.float32)
    rev = tracks[::-1].copy()
    assert motion_fidelity(tracks, vis, tracks, vis) == pytest.approx(1.0, abs=1e-3)
    assert motion_fidelity(tracks, vis, rev, vis) == pytest.approx(-1.0, abs=1e-3)


# --- store-backed compute_row ----------------------------------------------
def _unit(dim=8):
    u = np.zeros(dim, np.float32)
    u[0] = 1.0
    return u


def _coherent_tracks(T, N=25, seed=0):
    p0 = np.random.default_rng(seed).uniform(50, 300, size=(N, 2)).astype(np.float32)
    vel = np.zeros((N, 2), np.float32)
    vel[:, 0] = 3.0
    t = np.arange(T)[:, None, None]
    tracks = (p0[None] + vel[None] * t).astype(np.float32)
    return tracks, np.ones((T, N), np.float32)


def _put(fs, video, *, dino=None, tracks=None, vis=None):
    meta = {"origin": "extracted", "host": "test", "code_sha": "test"}
    if dino is not None:
        fs.put(video, DINO, {"feats": dino}, meta, video_sha256="0" * 64)
    if tracks is not None:
        fs.put(video, TRACK, {"tracks": tracks, "vis": vis}, meta, video_sha256="0" * 64)


def _setup(tmp_path, monkeypatch, *, two_sided, gen_feats, condA_feats, condB_feats,
           gen_tracks=None, condA_tracks=None, condB_tracks=None, endpoint="acid_0"):
    """Build a temp repo with one gen + its condition clips and (optionally) features."""
    root = tmp_path
    conds = root / "eval_ladder" / "conds"
    conds.mkdir(parents=True)
    monkeypatch.setattr(hm, "CONDS_DIR", conds)
    monkeypatch.setattr(hm, "probe_fps", lambda p: 24.0)     # -> K = 8

    stem = f"G-fit__ic_gen_neutral_v3__{endpoint}__ref_x__s42"
    vdir = root / "store" / "gens" / "tst" / "videos"
    vdir.mkdir(parents=True)
    gen = vdir / f"{stem}.mp4"
    gen.write_bytes(b"\x00")
    condA = conds / f"{endpoint}_start9.mp4"
    condA.write_bytes(b"\x00")
    condB = conds / f"{endpoint}_end9.mp4"
    if two_sided:
        condB.write_bytes(b"\x00")

    fs = FeatureStore(root)
    if gen_feats is not None:
        _put(fs, gen, dino=gen_feats, tracks=gen_tracks,
             vis=(np.ones(gen_tracks.shape[:2], np.float32) if gen_tracks is not None else None))
    if condA_feats is not None:
        _put(fs, condA, dino=condA_feats, tracks=condA_tracks,
             vis=(np.ones(condA_tracks.shape[:2], np.float32) if condA_tracks is not None else None))
    if two_sided and condB_feats is not None:
        _put(fs, condB, dino=condB_feats, tracks=condB_tracks,
             vis=(np.ones(condB_tracks.shape[:2], np.float32) if condB_tracks is not None else None))
    return fs, gen


def test_compute_row_one_sided_identity_and_motion(tmp_path, monkeypatch):
    T = 121
    u = _unit()
    gen_feats = np.tile(u, (T, 1))                 # every gen frame = u
    condA_feats = np.tile(u, (9, 1))               # start9: 9 frames = u; anchor = u
    gtr, _ = _coherent_tracks(T)
    ctr, _ = _coherent_tracks(9)
    fs, gen = _setup(tmp_path, monkeypatch, two_sided=False,
                     gen_feats=gen_feats, condA_feats=condA_feats, condB_feats=None,
                     gen_tracks=gtr, condA_tracks=ctr)
    feats = hm.Feats(fs)
    row = hm.compute_row(feats, gen, "acid_0", "one", "HF",
                         {gen.stem: (1.0, 0.5)})
    assert (row["n_pre"], row["n_suf"], row["K"]) == (9, 0, 8)
    assert row["identity_A"] == pytest.approx(1.0, abs=1e-6)
    assert math.isnan(row["identity_B"])           # one-sided
    assert math.isnan(row["motion_B"])
    assert row["motion_A"] == pytest.approx(1.0, abs=1e-3)
    assert row["seam_free"] == 1.0                 # prefix 1.0 <= 3
    assert row["missing"] == []


def test_compute_row_two_sided_identity_B(tmp_path, monkeypatch):
    T = 121
    u = _unit()
    gen_feats = np.tile(u, (T, 1))
    condA_feats = np.tile(u, (9, 1))
    condB_feats = np.zeros((9, 8), np.float32)
    condB_feats[1] = u                             # anchor index = T_B - n_suf = 9 - 8 = 1
    gtr, _ = _coherent_tracks(T)
    ctrA, _ = _coherent_tracks(9)
    ctrB, _ = _coherent_tracks(9, seed=2)
    fs, gen = _setup(tmp_path, monkeypatch, two_sided=True,
                     gen_feats=gen_feats, condA_feats=condA_feats, condB_feats=condB_feats,
                     gen_tracks=gtr, condA_tracks=ctrA, condB_tracks=ctrB)
    feats = hm.Feats(fs)
    row = hm.compute_row(feats, gen, "acid_0", "two", "HF",
                         {gen.stem: (1.0, 2.0)})
    assert (row["n_pre"], row["n_suf"]) == (9, 8)
    assert row["identity_A"] == pytest.approx(1.0, abs=1e-6)
    assert row["identity_B"] == pytest.approx(1.0, abs=1e-6)
    assert row["motion_A"] == pytest.approx(1.0, abs=1e-3)
    assert row["motion_B"] == pytest.approx(1.0, abs=1e-3)
    assert row["seam_free"] == 1.0                 # prefix 1.0 & suffix 2.0 both <= 3
    assert row["missing"] == []


def test_seam_free_thresholds(tmp_path, monkeypatch):
    T = 121
    u = _unit()
    fs, gen = _setup(tmp_path, monkeypatch, two_sided=True,
                     gen_feats=np.tile(u, (T, 1)), condA_feats=np.tile(u, (9, 1)),
                     condB_feats=np.tile(u, (9, 1)))
    feats = hm.Feats(fs)
    # one-sided uses prefix only
    r = hm.compute_row(feats, gen, "acid_0", "one", "HF", {gen.stem: (5.0, 0.0)})
    assert r["seam_free"] == 0.0                    # prefix 5 > 3
    r = hm.compute_row(feats, gen, "acid_0", "one", "HF", {gen.stem: (2.9, 9.0)})
    assert r["seam_free"] == 1.0                    # suffix ignored one-sided
    # two-sided needs both <= 3
    r = hm.compute_row(feats, gen, "acid_0", "two", "HF", {gen.stem: (1.0, 4.0)})
    assert r["seam_free"] == 0.0                    # suffix 4 > 3
    # absent seam row -> NaN + missing
    r = hm.compute_row(feats, gen, "acid_0", "one", "HF", {})
    assert math.isnan(r["seam_free"]) and "seam" in r["missing"]


def test_missing_features(tmp_path, monkeypatch):
    T = 121
    u = _unit()
    # gen features present, condition features ABSENT
    fs, gen = _setup(tmp_path, monkeypatch, two_sided=False,
                     gen_feats=np.tile(u, (T, 1)), condA_feats=None, condB_feats=None,
                     gen_tracks=_coherent_tracks(T)[0])
    feats = hm.Feats(fs)
    r = hm.compute_row(feats, gen, "acid_0", "one", "HF", {gen.stem: (1.0, 0.0)})
    assert math.isnan(r["identity_A"]) and f"{DINO}:condA" in r["missing"]
    assert math.isnan(r["motion_A"]) and f"{TRACK}:condA" in r["missing"]
    assert r["seam_free"] == 1.0                    # seam still available


def test_missing_gen_features(tmp_path, monkeypatch):
    # NO features at all (the externals-before-extraction case)
    fs, gen = _setup(tmp_path, monkeypatch, two_sided=False,
                     gen_feats=None, condA_feats=None, condB_feats=None)
    feats = hm.Feats(fs)
    r = hm.compute_row(feats, gen, "acid_0", "one", "HF", {})
    assert math.isnan(r["identity_A"]) and math.isnan(r["motion_A"])
    assert f"{DINO}:gen" in r["missing"]
    assert f"{TRACK}:gen" in r["missing"]


def test_ed_row_motion_a_undefined(tmp_path, monkeypatch):
    # ED grid: n_pre = 1 -> identity_A defined against frame 0; motion_A NaN by definition.
    T = 81
    u = _unit()
    gen_feats = np.tile(u, (T, 1))
    condA_feats = np.tile(u, (9, 1))               # frame 0 is the anchor (n_pre - 1 = 0)
    fs, gen = _setup(tmp_path, monkeypatch, two_sided=False,
                     gen_feats=gen_feats, condA_feats=condA_feats, condB_feats=None,
                     gen_tracks=_coherent_tracks(T)[0], condA_tracks=_coherent_tracks(9)[0])
    feats = hm.Feats(fs)
    r = hm.compute_row(feats, gen, "acid_0", "one", "ED", {gen.stem: (1.0, 0.0)})
    assert (r["n_pre"], r["n_suf"]) == (1, 0)
    assert r["identity_A"] == pytest.approx(1.0, abs=1e-6)
    assert math.isnan(r["motion_A"])               # n_pre < 9: undefined, NOT missing
    assert f"{TRACK}:condA" not in r["missing"]


def test_append_index_idempotent(tmp_path, monkeypatch):
    store = tmp_path / "store"
    store.mkdir()
    idx = store / "INDEX.md"
    idx.write_text("## evals\n1. `001_x` — foo\n\n## datasets\n1. `d`\n")
    monkeypatch.setattr(hm, "REPO_ROOT", tmp_path)
    eid = "038_handoff_gridv3__dai__2026-09-18"
    assert hm.append_index_line(eid, 19, 7242) is True
    assert hm.append_index_line(eid, 19, 7242) is False       # idempotent
    text = idx.read_text()
    assert text.count(f"`{eid}`") == 1
    # inserted inside the evals section, before ## datasets
    assert text.index(f"`{eid}`") < text.index("## datasets")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
