"""Unit tests for the competitor-lens feature-store integration (score_batch.py
--store). Fake extractors, tmp FeatureStore roots, CPU only — no backbones.

The 20-row numeric reproduction of the paper's rows_v3 (to 1e-6) needs the real
GPU features and is P4; these tests pin the store read/write/extract-on-miss
routing and the input-frame embed wrappers.

Run from this directory:  python -m pytest -q test_store_lenses.py
"""

import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import score_batch as sb                                     # noqa: E402
from diffusion.feature_store import FeatureStore             # noqa: E402


def _mk_video(path: pathlib.Path, nbytes: int = 64) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * nbytes)
    return path


class FakeExtractor:
    """Stand-in for a REGISTRY extractor: extract(video_path) -> arrays dict."""

    def __init__(self, arrays_fn):
        self.arrays_fn = arrays_fn
        self.calls = 0

    def extract(self, video):
        self.calls += 1
        return self.arrays_fn(video)


def _lenses(tmp_path):
    L = sb.StoreLenses(tmp_path, device="cpu")
    return L


def test_video_hit_returns_store_arrays_no_extract(tmp_path):
    L = _lenses(tmp_path)
    vid = _mk_video(tmp_path / "videos" / "g.mp4")
    feats = np.random.default_rng(0).random((12, 512)).astype(np.float32)
    L.store.put(vid, sb.CLIP_NS, {"feats": feats},
                {"origin": "extracted", "host": "t", "code_sha": ""})
    # inject an extractor that would explode if called
    L._ext[sb.CLIP_NS] = FakeExtractor(lambda v: (_ for _ in ()).throw(AssertionError("extracted on a hit")))

    out = L.video(str(vid), sb.CLIP_NS)
    np.testing.assert_array_equal(out["feats"], feats)


def test_video_miss_extracts_and_writes_back(tmp_path):
    L = _lenses(tmp_path)
    vid = _mk_video(tmp_path / "videos" / "g.mp4")
    feats = np.arange(6 * 512, dtype=np.float32).reshape(6, 512)
    fake = FakeExtractor(lambda v: {"feats": feats})
    L._ext[sb.CLIP_NS] = fake

    assert not L.has(str(vid), sb.CLIP_NS)
    out1 = L.video(str(vid), sb.CLIP_NS)
    np.testing.assert_array_equal(out1["feats"], feats)
    assert fake.calls == 1
    assert L.has(str(vid), sb.CLIP_NS)          # written back
    out2 = L.video(str(vid), sb.CLIP_NS)        # now a hit
    np.testing.assert_array_equal(out2["feats"], feats)
    assert fake.calls == 1                      # not re-extracted

    # sidecar is self-describing (origin extracted)
    sc = L.store.read_meta(vid, sb.CLIP_NS)
    assert sc["origin"] == "extracted" and sc["shape"] == [6, 512]


def test_tracks_namespace_two_arrays(tmp_path):
    L = _lenses(tmp_path)
    vid = _mk_video(tmp_path / "videos" / "g.mp4")
    tracks = np.zeros((10, 6, 2), dtype=np.float32)
    vis = np.ones((10, 6), dtype=np.float32)
    L._ext[sb.TRACK_NS] = FakeExtractor(lambda v: {"tracks": tracks, "vis": vis})
    out = L.video(str(vid), sb.TRACK_NS)
    np.testing.assert_array_equal(out["tracks"], tracks)
    np.testing.assert_array_equal(out["vis"], vis)


# --- input-frame embed wrappers (reuse the REGISTRY model instances) ----------

class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeClipModel:
    def get_image_features(self, **inp):
        import torch
        # deterministic 512-d "embedding" from the pixel sum
        n = inp["pixel_values"].shape[0]
        base = torch.arange(1.0, 513.0)
        return base.repeat(n, 1)


class _FakeClipProc:
    def __call__(self, images, return_tensors=None):
        import torch
        return _FakeBatch(pixel_values=torch.zeros(len(images), 3, 4, 4))


class _FakeClipExt:
    def __init__(self):
        self.model = _FakeClipModel()
        self.proc = _FakeClipProc()
        self.device = "cpu"


def test_clip_frame_shape_and_l2_norm(tmp_path):
    L = _lenses(tmp_path)
    L._ext[sb.CLIP_NS] = _FakeClipExt()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    emb = L.clip_frame(frame)
    assert emb.shape == (1, 512) and emb.dtype == np.float32
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


class _FakeVPModel:
    def __call__(self, **inp):
        import torch
        # last_hidden_state [1, nf*npatch, D]; nf=4, npatch=5, D=7
        nf, npatch, D = 4, 5, 7
        lhs = torch.arange(1.0, 1 + nf * npatch * D).reshape(1, nf * npatch, D)
        return type("O", (), {"last_hidden_state": lhs})()


class _FakeVPProc:
    def __call__(self, videos, return_tensors=None):
        import torch
        return _FakeBatch(pixel_values=torch.zeros(1, 4, 3, 4, 4))


class _FakeVPExt:
    num_frames = 4

    def __init__(self):
        self.model = _FakeVPModel()
        self.proc = _FakeVPProc()
        self.device = "cpu"


def test_vp_frame_shape_and_norm(tmp_path):
    L = _lenses(tmp_path)
    L._ext[sb.VP_NS] = _FakeVPExt()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    emb = L.vp_frame(frame)
    assert emb.shape == (4, 7) and emb.dtype == np.float32       # [num_frames, D]
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def test_parse_item_still_derives_meta():
    # I/O change must not disturb the filename contract the rows carry
    p = pathlib.Path("x/store/gens/013_dualforce_control/03_neutral_v3__dai/videos/"
                     "G-fit__dualforce_control_neutral_v3__animalization_3__ref_animalization_1__s42.mp4")
    m = sb.parse_item(p)
    assert m["arm"] == "013_dualforce_control/03_neutral_v3__dai"
    assert m["endpoint"] == "animalization_3" and m["reference"] == "animalization_1"
    assert m["seed"] == 42
