"""Feature-store I/O contract for the scoring path (SPEC §9, 4.0.1).

These pin the store-backed replacement of the hashed-filename caches: real
videos persist by path through a tmp FeatureStore; a hit returns exactly the
arrays a miss would have written (numeric identity); synthetic controls persist
next to the gen as `<ns>.ctl-<name>` and read back clean under `fsck`. CPU only,
fake extractor/tracker/scorer — no torch backbone, no GPU.

PYTHONPATH shim: the env's editable install points at the MAIN checkout and would
shadow worktree code; every harness test carries this shim (SPEC §10 note).
"""

import json
import pathlib
import sys

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diffusion.feature_store import FeatureStore                      # noqa: E402
from diffusion.transition_eval import store_io                        # noqa: E402
from diffusion.transition_eval.pipeline import (                      # noqa: E402
    process_video_file, process_video_store)


# --- fakes -------------------------------------------------------------------

class FakeExtractor:
    """Deterministic stand-in for DinoExtractor.extract: uint8 [T,H,W,3] ->
    L2-normalized [T,D] that depends on the frames, and counts its calls."""

    model_name = "stub-dino"

    def __init__(self, dim=8):
        self.dim = dim
        self.calls = 0

    def extract(self, frames):
        self.calls += 1
        T = len(frames)
        base = frames.reshape(T, -1).astype(np.float64).mean(axis=1)
        cols = np.arange(self.dim, dtype=np.float64)
        F = np.cos(np.outer(base, cols) * 0.01 + cols)
        F = F / np.linalg.norm(F, axis=1, keepdims=True)
        return F.astype(np.float32)


class FakeTracker:
    CACHE_TAG = "v2"

    def __init__(self, n=6):
        self.n = n
        self.calls = 0

    def track(self, frames):
        self.calls += 1
        T = len(frames)
        rng = np.random.default_rng(len(frames))
        tracks = rng.random((T, self.n, 2)).astype(np.float32)
        vis = np.ones((T, self.n), dtype=np.float32)
        return tracks, vis


class FakeLpips:
    def pairwise(self, f1, f2, batch_size=16):
        return np.mean(np.abs(f1.astype(float) - f2.astype(float)), axis=(1, 2, 3))


def _mk_video(path: pathlib.Path, nbytes: int = 64) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * nbytes)               # store.put stats + hashes it
    return path


def _frames(T=12, H=8, W=8):
    return np.broadcast_to(
        np.arange(T, dtype=np.uint8)[:, None, None, None], (T, H, W, 3)).copy()


# --- HarnessStore: real-video routing ----------------------------------------

def test_real_video_put_get_has_roundtrip(tmp_path):
    hs = store_io.HarnessStore(FeatureStore(tmp_path), code_sha="deadbeef")
    vid = _mk_video(tmp_path / "store" / "gens" / "A" / "videos" / "g.mp4")
    ident = store_io.RealVideo(vid)

    assert hs.get(ident, store_io.DINO_NS) is None and not hs.has(ident, store_io.DINO_NS)
    feats = np.random.default_rng(0).random((12, 8)).astype(np.float32)
    hs.put(ident, store_io.DINO_NS, {"feats": feats})

    assert hs.has(ident, store_io.DINO_NS)
    got = hs.get(ident, store_io.DINO_NS)
    np.testing.assert_array_equal(got["feats"], feats)
    # path rule: features/ sibling of videos/, one folder per stem
    npz = tmp_path / "store" / "gens" / "A" / "features" / "g" / f"{store_io.DINO_NS}.npz"
    side = npz.with_suffix(".json")
    assert npz.exists() and side.exists()
    sc = json.loads(side.read_text())
    assert sc["origin"] == "extracted" and sc["code_sha"] == "deadbeef"
    assert sc["video"] == "store/gens/A/videos/g.mp4"


def test_process_video_store_hit_skips_extract(tmp_path):
    """A store hit returns the SAME feats a miss wrote, without re-extracting —
    the numeric-identity guarantee of the store path."""
    hs = store_io.HarnessStore(FeatureStore(tmp_path))
    vid = _mk_video(tmp_path / "videos" / "g.mp4")
    ident = store_io.RealVideo(vid)
    ext, trk, frames = FakeExtractor(), FakeTracker(), _frames()

    b1 = process_video_store(frames, ident, hs, ext, trk,
                             n_prefix=2, n_suffix=2, n_endpoints=1)
    assert ext.calls == 1 and trk.calls == 1
    feats_ref = ext.extract(frames)                  # recompute independently
    ext.calls = 0

    b2 = process_video_store(None, ident, hs, ext, trk,     # frames=None: must hit
                             n_prefix=2, n_suffix=2, n_endpoints=1)
    assert ext.calls == 0                            # no re-extract on a hit
    np.testing.assert_array_equal(b1["feats"], b2["feats"])
    np.testing.assert_array_equal(b2["feats"], feats_ref)
    np.testing.assert_array_equal(b1["tracks"], b2["tracks"])
    assert b2["identity"] is ident


def test_process_video_file_store_branch_warm_skips_decode(tmp_path, monkeypatch):
    """need_frames=False + warm store (dino+tracks present) -> no decode."""
    from diffusion.transition_eval import pipeline
    hs = store_io.HarnessStore(FeatureStore(tmp_path))
    vid = _mk_video(tmp_path / "videos" / "g.mp4")
    ident = store_io.RealVideo(vid)
    hs.put(ident, store_io.DINO_NS,
           {"feats": np.ones((12, 8), dtype=np.float32)})
    hs.put(ident, store_io.TRACK_NS,
           {"tracks": np.zeros((12, 6, 2), dtype=np.float32),
            "vis": np.ones((12, 6), dtype=np.float32)})

    def boom(*a, **k):
        raise AssertionError("load_frames called despite need_frames=False + warm")
    monkeypatch.setattr(pipeline, "load_frames", boom)
    monkeypatch.setattr(pipeline, "probe_fps", lambda p: 24.0)

    class NoExtract(FakeExtractor):
        def extract(self, frames):
            raise AssertionError("extract called despite warm store")

    b, frames = process_video_file(vid, hs, NoExtract(), FakeTracker(),
                                   short_side=256, need_frames=False,
                                   n_prefix=2, n_suffix=2, n_endpoints=1)
    assert frames is None and b["fps"] == 24.0
    assert b["feats"].shape == (12, 8) and b["tracks"].shape == (12, 6, 2)


def test_process_video_file_store_cold_still_decodes(tmp_path, monkeypatch):
    from diffusion.transition_eval import pipeline
    hs = store_io.HarnessStore(FeatureStore(tmp_path))
    vid = _mk_video(tmp_path / "videos" / "g2.mp4")
    monkeypatch.setattr(pipeline, "load_frames", lambda p, short_side=256: (_frames(), 24.0))
    ext = FakeExtractor()
    b, frames = process_video_file(vid, hs, ext, FakeTracker(),
                                   short_side=256, need_frames=False,
                                   n_prefix=2, n_suffix=2, n_endpoints=1)
    assert frames is not None and ext.calls == 1        # cold -> decode + extract
    assert hs.has(store_io.RealVideo(vid), store_io.DINO_NS)   # and persisted


# --- synthetic controls: persisted next to the gen ---------------------------

def test_control_persists_next_to_gen_and_is_fsck_clean(tmp_path):
    hs = store_io.HarnessStore(FeatureStore(tmp_path), code_sha="cafef00d")
    videos_dir = tmp_path / "store" / "gens" / "A" / "videos"
    vid = _mk_video(videos_dir / "g.mp4")
    gen = store_io.RealVideo(vid)
    ext, trk = FakeExtractor(), FakeTracker()

    # gen's own features first (so the control sidecar can reuse the gen identity)
    process_video_store(_frames(), gen, hs, ext, trk, n_prefix=2, n_suffix=2, n_endpoints=1)
    # then the control, synthesized frames of a different length
    ctl = store_io.Control(vid, "lerp")
    cframes = _frames(T=14)
    cb = process_video_store(cframes, ctl, hs, ext, trk, n_prefix=2, n_suffix=2, n_endpoints=1)
    assert isinstance(cb["identity"], store_io.Control)

    feat_dir = tmp_path / "store" / "gens" / "A" / "features" / "g"
    npz = feat_dir / f"{store_io.DINO_NS}.ctl-lerp.npz"
    side = feat_dir / f"{store_io.DINO_NS}.ctl-lerp.json"
    assert npz.exists() and side.exists()
    sc = json.loads(side.read_text())
    assert sc["origin"] == "control:lerp" and sc["control"] == "lerp"
    assert sc["video"] == "store/gens/A/videos/g.mp4" and sc["code_sha"] == "cafef00d"
    # the control does NOT masquerade as the gen's real namespace file
    assert hs.has(gen, store_io.DINO_NS)
    got = hs.get(ctl, store_io.DINO_NS)
    np.testing.assert_array_equal(got["feats"], cb["feats"])
    assert cb["feats"].shape[0] == 14                # control's own length

    # fsck: the control lives inside the gen's item folder; the gen video's
    # identity in the sidecar keeps it out of the `stale` list; no orphans.
    rep = hs.store.fsck(videos_dir)
    assert rep["ok"] and not rep["stale"] and not rep["orphan_npz"] and not rep["orphan_json"]


def test_control_get_has_miss(tmp_path):
    hs = store_io.HarnessStore(FeatureStore(tmp_path))
    vid = _mk_video(tmp_path / "store" / "gens" / "A" / "videos" / "g.mp4")
    ctl = store_io.Control(vid, "hold")
    assert not hs.has(ctl, store_io.LPIPS_NS)
    assert hs.get(ctl, store_io.LPIPS_NS) is None
    hs.put(ctl, store_io.LPIPS_NS, {"d": np.arange(11, dtype=np.float32)})
    assert hs.has(ctl, store_io.LPIPS_NS)
    np.testing.assert_array_equal(hs.get(ctl, store_io.LPIPS_NS)["d"],
                                  np.arange(11, dtype=np.float32))


# --- temporal-LPIPS through the store (score._temporal_lpips_stored) ---------

def test_temporal_lpips_stored_roundtrip(tmp_path):
    from diffusion.transition_eval.score import _temporal_lpips_stored
    hs = store_io.HarnessStore(FeatureStore(tmp_path))
    vid = _mk_video(tmp_path / "videos" / "g.mp4")
    ident = store_io.RealVideo(vid)
    frames = _frames()
    d1 = _temporal_lpips_stored(hs, ident, frames, FakeLpips())
    assert d1.shape == (len(frames) - 1,)
    # a hit reads back the stored array; frames=None is allowed only on a hit
    d2 = _temporal_lpips_stored(hs, ident, None, FakeLpips())
    np.testing.assert_array_equal(d1, d2)
    # a miss with no frames is a caller bug
    ident2 = store_io.RealVideo(_mk_video(tmp_path / "videos" / "g2.mp4"))
    with pytest.raises(RuntimeError):
        _temporal_lpips_stored(hs, ident2, None, FakeLpips())
