"""CPU-only tests for the feature store (no GPU, no backbones, no real videos).

Covers the path rule (both cases), atomic put + sidecar-last, the hard-link
migrate (same st_ino) with a hand-computed legacy key, manifest rebuild, fsck
(orphan / stale / mixed hosts), coverage counts, write_meta_block text
preservation, and sha256sums skip logic. A fake extractor is injected for the
extract worker; the legacy cache files are hand-built npz.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from diffusion.feature_store import (  # noqa: E402
    FeatureStore, legacy_filenames, legacy_key, sha256_file)
import store_features as sf  # noqa: E402

NS = "dino_cls@dinov2b-r256"
NS_TRACKS = "cotracker3@g20-m384-v2"


def _mp4(path: Path, data: bytes = b"\x00\x01\x02fakevideo") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _feats(n=4, d=8):
    return {"feats": np.arange(n * d, dtype=np.float32).reshape(n, d)}


# --- path rule ---------------------------------------------------------------
def test_path_rule_store_gen(tmp_path):
    store = FeatureStore(tmp_path)
    v = tmp_path / "013_arm" / "03_var" / "videos" / "item__s42.mp4"
    assert store.path(v, NS) == (tmp_path / "013_arm" / "03_var" / "features"
                                 / "item__s42" / f"{NS}.npz")
    assert store.sidecar(v, NS).name == f"{NS}.json"


def test_path_rule_corpus(tmp_path):
    store = FeatureStore(tmp_path)
    v = tmp_path / "data" / "cls_a" / "clip7.mp4"          # not under videos/
    assert store.path(v, NS) == (tmp_path / "data" / "cls_a" / "features"
                                 / "clip7" / f"{NS}.npz")


# --- legacy key recipe -------------------------------------------------------
def test_legacy_key_recipe(tmp_path):
    v = _mp4(tmp_path / "videos" / "clip.mp4", b"abc123")
    st = v.stat()
    expect = hashlib.sha1("|".join([
        str(v.resolve()), str(st.st_mtime_ns), str(st.st_size),
        "facebook/dinov2-base", "256"]).encode()).hexdigest()[:16]
    assert legacy_key(v) == expect
    # filename forms hash the key again
    k = legacy_key(v)
    assert legacy_filenames(NS, k)[0] == \
        f"dino_arr_{hashlib.sha1(k.encode()).hexdigest()[:16]}.npz"
    assert legacy_filenames(NS, k)[1] == f"dino_{k}.npz"
    assert legacy_filenames(NS_TRACKS, k)[0] == \
        f"tracks_{hashlib.sha1((k + ':tracks:v2').encode()).hexdigest()[:16]}.npz"
    assert legacy_filenames("lpips_t@alex-r256", k)[0] == \
        f"lpips_{hashlib.sha1((k + ':tlpips:alex-v1').encode()).hexdigest()[:16]}.npz"


# --- atomic put + sidecar last -----------------------------------------------
def test_put_writes_pair_and_sidecar_last(tmp_path):
    store = FeatureStore(tmp_path)
    v = _mp4(tmp_path / "g" / "v" / "videos" / "a__s42.mp4")
    store.put(v, NS, _feats(), {"host": "h1", "code_sha": "abc", "origin": "extracted"})
    assert store.has(v, NS)
    npz, side = store.path(v, NS), store.sidecar(v, NS)
    assert npz.exists() and side.exists()
    # no tmp files left behind
    assert not list(npz.parent.glob("*.tmp-*"))
    got = store.get(v, NS)
    assert np.array_equal(got["feats"], _feats()["feats"])
    meta = json.loads(side.read_text())
    for k in ("ns", "video", "video_sha256", "video_size", "video_mtime_ns",
              "host", "code_sha", "created", "origin", "shape", "dtype", "bytes"):
        assert k in meta, k
    assert meta["ns"] == NS
    assert meta["video"] == "g/v/videos/a__s42.mp4"       # relative to repo root
    assert meta["shape"] == [4, 8] and meta["dtype"] == "float32"
    assert meta["video_sha256"] == sha256_file(v)


def test_lone_npz_is_not_has_and_is_orphan(tmp_path):
    store = FeatureStore(tmp_path)
    v = _mp4(tmp_path / "g" / "v" / "videos" / "a__s42.mp4")
    store.put(v, NS, _feats(), {"host": "h", "code_sha": "x", "origin": "extracted"})
    store.sidecar(v, NS).unlink()                          # simulate interrupted write
    assert not store.has(v, NS)
    rep = store.fsck(v.parent)
    assert rep["orphan_npz"] and not rep["ok"]


# --- migrate (hard link, same inode) -----------------------------------------
def test_migrate_hard_links_same_inode(tmp_path):
    store = FeatureStore(tmp_path)
    legacy = tmp_path / "legacy_cache"
    legacy.mkdir()
    v = _mp4(tmp_path / "g" / "v" / "videos" / "a__s42.mp4", b"video-bytes")
    key = legacy_key(v)
    legacy_file = legacy / legacy_filenames(NS, key)[0]
    np.savez_compressed(legacy_file, feats=_feats()["feats"], src="legacy")

    stats = sf.migrate_videos(store, [v], [legacy], host="dai", sha="s", dry_run=False)
    assert stats[NS]["hit"] == 1 and stats[NS]["linked"] == 1
    assert store.has(v, NS)
    # hard link => same device+inode as the legacy file
    a, b = store.path(v, NS).stat(), legacy_file.stat()
    assert (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)
    assert np.array_equal(store.get(v, NS)["feats"], _feats()["feats"])
    meta = json.loads(store.sidecar(v, NS).read_text())
    assert meta["origin"].startswith("migrated:")
    # migrated => extraction host unknown (null); migrating node recorded separately
    assert meta["host"] is None
    assert meta["migrated_by_host"] == "dai"       # the `host` arg passed to migrate
    cov = store.coverage(v.parent)
    assert cov[NS]["have"] == 1 and cov[NS]["legacy"] == 1 and cov[NS]["hosts"] == []
    rep = store.fsck(v.parent)
    assert NS in rep["legacy_ns"] and NS not in rep["mixed_hosts"] and rep["ok"]


def test_migrate_dry_run_counts_no_link(tmp_path):
    store = FeatureStore(tmp_path)
    legacy = tmp_path / "legacy_cache"
    legacy.mkdir()
    v = _mp4(tmp_path / "g" / "v" / "videos" / "a__s42.mp4", b"vv")
    key = legacy_key(v)
    np.savez_compressed(legacy / legacy_filenames(NS, key)[0], feats=_feats()["feats"])
    stats = sf.migrate_videos(store, [v], [legacy], host="h", sha="s", dry_run=True)
    assert stats[NS]["hit"] == 1 and stats[NS]["linked"] == 0
    assert not store.has(v, NS)                            # dry run wrote nothing


# --- extract worker with a fake extractor ------------------------------------
class _FakeExtractor:
    def extract(self, video):
        return {"feats": np.ones((3, 5), dtype=np.float32)}


def test_extract_fills_misses_only(tmp_path):
    store = FeatureStore(tmp_path)
    vids = [_mp4(tmp_path / "g" / "v" / "videos" / f"a{i}__s42.mp4", bytes([i]))
            for i in range(3)]
    store.put(vids[0], NS, _feats(), {"host": "h", "code_sha": "x", "origin": "extracted"})
    dry = sf.extract_videos(store, NS, vids, _FakeExtractor(), host="h", sha="s",
                            dry_run=True)
    assert dry["have"] == 1 and dry["miss"] == 2 and dry["filled"] == 0
    real = sf.extract_videos(store, NS, vids, _FakeExtractor(), host="h", sha="s")
    assert real["filled"] == 2
    assert all(store.has(v, NS) for v in vids)
    assert np.array_equal(store.get(vids[1], NS)["feats"], np.ones((3, 5), np.float32))


# --- manifest rebuild --------------------------------------------------------
def test_rebuild_manifest_equals_sidecars(tmp_path):
    store = FeatureStore(tmp_path)
    vids = [_mp4(tmp_path / "g" / "v" / "videos" / f"a{i}__s42.mp4", bytes([i]))
            for i in range(3)]
    for v in vids:
        store.put(v, NS, _feats(), {"host": "h", "code_sha": "x", "origin": "extracted"})
    n = store.rebuild_manifest(vids[0].parent)
    assert n == 3
    manifest = store._feat_root(vids[0].parent) / "manifest.jsonl"
    recs = [json.loads(ln) for ln in manifest.read_text().splitlines()]
    sidecars = [json.loads(store.sidecar(v, NS).read_text()) for v in vids]
    assert sorted(json.dumps(r, sort_keys=True) for r in recs) == \
           sorted(json.dumps(s, sort_keys=True) for s in sidecars)
    assert not store.fsck(vids[0].parent)["manifest_drift"]


# --- fsck: stale + mixed hosts + coverage ------------------------------------
def test_fsck_detects_stale(tmp_path):
    store = FeatureStore(tmp_path)
    v = _mp4(tmp_path / "g" / "v" / "videos" / "a__s42.mp4", b"orig")
    store.put(v, NS, _feats(), {"host": "h", "code_sha": "x", "origin": "extracted"})
    assert store.fsck(v.parent)["ok"]
    v.write_bytes(b"changed-size-and-mtime")               # video changed after extract
    rep = store.fsck(v.parent)
    assert rep["stale"] and not rep["ok"]


def test_fsck_detects_mixed_hosts_and_coverage(tmp_path):
    store = FeatureStore(tmp_path)
    v1 = _mp4(tmp_path / "g" / "v" / "videos" / "a1__s42.mp4", b"1")
    v2 = _mp4(tmp_path / "g" / "v" / "videos" / "a2__s42.mp4", b"2")
    v3 = _mp4(tmp_path / "g" / "v" / "videos" / "a3__s42.mp4", b"3")
    store.put(v1, NS, _feats(), {"host": "eps", "code_sha": "x", "origin": "extracted"})
    store.put(v2, NS, _feats(), {"host": "dai", "code_sha": "x", "origin": "extracted"})
    cov = store.coverage(v1.parent)
    assert cov[NS]["have"] == 2 and cov[NS]["of"] == 3
    assert cov[NS]["hosts"] == ["dai", "eps"] and cov[NS]["legacy"] == 0
    rep = store.fsck(v1.parent)
    assert NS in rep["mixed_hosts"] and not rep["ok"]


# --- write_meta_block: preserve unrelated yaml -------------------------------
META_TXT = """id: gens/013_x/03_y
seq: 3
arm: dualforce   # inline comment kept
# a standalone comment
videos: 2
notes: some note with: a colon
"""


def test_write_meta_block_preserves_text(tmp_path):
    store = FeatureStore(tmp_path)
    var = tmp_path / "g" / "v"
    (var / "videos").mkdir(parents=True)
    v1 = _mp4(var / "videos" / "a1__s42.mp4", b"1")
    v2 = _mp4(var / "videos" / "a2__s42.mp4", b"2")
    (var / "meta.yaml").write_text(META_TXT)
    for v in (v1, v2):
        store.put(v, NS, _feats(), {"host": "dai", "code_sha": "x", "origin": "extracted"})
    store.write_meta_block(var)
    txt = (var / "meta.yaml").read_text()
    # unrelated lines survive verbatim (comments included)
    for line in ("arm: dualforce   # inline comment kept", "# a standalone comment",
                 "notes: some note with: a colon", "seq: 3"):
        assert line in txt
    assert "features:" in txt
    assert f"  {NS}: {{have: 2, of: 2, hosts: [dai]}}" in txt
    # idempotent: a second call leaves exactly one features: block
    store.write_meta_block(var)
    assert (var / "meta.yaml").read_text().count("features:") == 1


# --- sha256sums skip logic ---------------------------------------------------
def test_sha256sums_skip_unchanged(tmp_path):
    store = FeatureStore(tmp_path)
    vd = tmp_path / "g" / "v" / "videos"
    v1 = _mp4(vd / "a1__s42.mp4", b"one")
    v2 = _mp4(vd / "a2__s42.mp4", b"two")
    r1 = sf.write_sha256sums(store, vd)
    assert r1 == {"n": 2, "computed": 2, "reused": 0}
    sums = (vd / "SHA256SUMS").read_text()
    assert f"{sha256_file(v1)}  a1__s42.mp4" in sums
    r2 = sf.write_sha256sums(store, vd)                    # nothing changed
    assert r2["reused"] == 2 and r2["computed"] == 0
    v1.write_bytes(b"one-changed")                         # one file changed
    r3 = sf.write_sha256sums(store, vd)
    assert r3["computed"] == 1 and r3["reused"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# --- population scoping ------------------------------------------------------
def test_population_targets_and_subset_sha256sums(tmp_path):
    """--population restricts corpus/conds to the listed clips; SHA256SUMS merges; fsck/coverage scope."""
    gen = tmp_path / "gens" / "001_arm" / "01_var__x"
    _mp4(gen / "videos" / "a__s42.mp4"); (gen / "meta.yaml").write_text("id: x\n")
    cls = tmp_path / "corpus" / "acid"
    for n in "acid_0 acid_1 acid_2".split():
        _mp4(cls / f"{n}.mp4", n.encode())
    conds = tmp_path / "conds"
    for n in "e_start9 e_end9 f_start9".split():
        _mp4(conds / f"{n}.mp4", n.encode())
    pop = {"gen_variants": [str(gen)],
           "corpus": {"files": [str(cls / "acid_0.mp4"), str(cls / "acid_2.mp4")]},
           "conds": {"files": [str(conds / "e_start9.mp4")]}}
    pf = tmp_path / "pop.json"; pf.write_text(json.dumps(pop))
    targets = sf.population_targets(pf)
    assert [x["label"] for x in targets] == ["gens/001_arm/01_var__x", "corpus/acid", "conds"]
    assert [v.name for v in targets[1]["videos"]] == ["acid_0.mp4", "acid_2.mp4"]
    assert [v.name for v in targets[2]["videos"]] == ["e_start9.mp4"]
    store = FeatureStore(tmp_path)
    r = sf.write_sha256sums(store, cls, videos=targets[1]["videos"])
    assert r["n"] == 2 and r["computed"] == 2
    r2 = sf.write_sha256sums(store, cls, videos=[cls / "acid_1.mp4"])
    names = [ln.split("  ", 1)[1] for ln in (cls / "SHA256SUMS").read_text().splitlines()]
    assert names == ["acid_0.mp4", "acid_1.mp4", "acid_2.mp4"] and r2["computed"] == 1
    # features for a clip OUTSIDE the population must not disturb scoped fsck/coverage
    store.put(cls / "acid_1.mp4", NS, _feats(), {"host": "h1"})
    store.put(cls / "acid_0.mp4", NS, _feats(), {"host": "h1"})
    cov = store.coverage(cls, videos=targets[1]["videos"])
    assert cov[NS]["of"] == 2 and cov[NS]["have"] == 1
    rep = store.fsck(cls, videos=targets[1]["videos"])
    assert rep["ok"] and rep["n_videos"] == 2 and rep["n_features"] == 1


# --- control variants (<ns>.ctl-<name>) first-class in put/coverage/fsck ------
def test_put_control_variant_roundtrip_and_sidecar(tmp_path):
    store = FeatureStore(tmp_path)
    videos = tmp_path / "gens" / "A" / "01_v__x" / "videos"
    gen = _mp4(videos / "g__s42.mp4", b"genbytes")
    store.put(gen, NS, _feats(), {"host": "dai", "code_sha": "c0", "origin": "extracted"})
    gsha = json.loads(store.sidecar(gen, NS).read_text())["video_sha256"]
    ctl = {"feats": np.ones((5, 8), np.float32)}
    p = store.put(gen, NS, ctl, {"host": "dai", "code_sha": "c0",
                                 "origin": "control:lerp"}, variant="lerp",
                  video_sha256=gsha)
    assert p.name == f"{NS}.ctl-lerp.npz"
    assert store.has(gen, NS, variant="lerp") and store.has(gen, NS)   # separate files
    assert np.array_equal(store.get(gen, NS, variant="lerp")["feats"], ctl["feats"])
    sc = store.read_meta(gen, NS, variant="lerp")
    assert sc["control"] == "lerp" and sc["origin"] == "control:lerp"
    assert sc["video"] == "gens/A/01_v__x/videos/g__s42.mp4"
    assert sc["video_sha256"] == gsha and sc["shape"] == [5, 8]
    # coverage: the real-ns cell counts only the real file; controls report apart
    cov = store.coverage(videos)
    assert cov[NS]["have"] == 1
    assert cov["controls"]["have"] == 1 and cov["controls"]["names"] == ["lerp"]
    # fsck: control is first-class + clean + counted apart; the ".ctl-lerp"
    # pseudo-namespace never enters the host tracking (no spurious mixed/legacy)
    rep = store.fsck(videos, rehash=True)
    assert rep["ok"] and rep["n_features"] == 1 and rep["n_controls"] == 1
    assert not rep["stale"] and not rep["mixed_hosts"] and not rep["legacy_ns"]


def test_control_orphan_npz_is_flagged(tmp_path):
    store = FeatureStore(tmp_path)
    videos = tmp_path / "g" / "v" / "videos"
    gen = _mp4(videos / "g__s42.mp4")
    store.put(gen, NS, _feats(), {"host": "h", "code_sha": "x", "origin": "extracted"})
    store.put(gen, NS, {"feats": np.ones((2, 8), np.float32)},
              {"host": "h", "origin": "control:hold"}, variant="hold")
    store.sidecar(gen, NS, variant="hold").unlink()       # interrupted control write
    assert not store.has(gen, NS, variant="hold")
    rep = store.fsck(videos)
    assert f"g__s42/{NS}.ctl-hold.npz" in rep["orphan_npz"] and not rep["ok"]


def test_control_stale_against_gen_identity(tmp_path):
    store = FeatureStore(tmp_path)
    videos = tmp_path / "g" / "v" / "videos"
    gen = _mp4(videos / "g__s42.mp4", b"orig")
    store.put(gen, NS, _feats(), {"host": "h", "origin": "extracted"})
    gsha = json.loads(store.sidecar(gen, NS).read_text())["video_sha256"]
    store.put(gen, NS, {"feats": np.ones((3, 8), np.float32)},
              {"host": "h", "origin": "control:lerp"}, variant="lerp", video_sha256=gsha)
    assert store.fsck(videos)["ok"]
    gen.write_bytes(b"changed-after-extract")             # gen video changes
    rep = store.fsck(videos)                              # control stale-checks vs the gen mp4
    assert any(".ctl-lerp (stat drift)" in s for s in rep["stale"]) and not rep["ok"]


# --- sha_from_sums + precomputed-sha put (skip re-hash) ------------------------
def test_sha_from_sums_and_skip_rehash(tmp_path):
    store = FeatureStore(tmp_path)
    vd = tmp_path / "g" / "v" / "videos"
    v = _mp4(vd / "a__s42.mp4", b"payload")
    assert store.sha_from_sums(v) is None                 # no SHA256SUMS yet
    sf.write_sha256sums(store, vd)
    assert store.sha_from_sums(v) == sha256_file(v)
    # an explicit (deliberately wrong) sha is honored verbatim => no re-hash
    store.put(v, NS, _feats(), {"host": "h", "origin": "extracted"},
              video_sha256="deadbeef")
    assert store.read_meta(v, NS)["video_sha256"] == "deadbeef"
    # feeding sha_from_sums matches a fresh hash
    v2 = _mp4(vd / "b__s42.mp4", b"payload2")
    sf.write_sha256sums(store, vd)
    store.put(v2, NS, _feats(), {"host": "h", "origin": "extracted"},
              video_sha256=store.sha_from_sums(v2))
    assert store.read_meta(v2, NS)["video_sha256"] == sha256_file(v2)


def test_sha_from_sums_clip_folder(tmp_path):
    """corpus clips (not under a videos/ dir) read the folder's own SHA256SUMS."""
    store = FeatureStore(tmp_path)
    cls = tmp_path / "corpus" / "acid"
    c = _mp4(cls / "acid_0.mp4", b"acidbytes")
    sf.write_sha256sums(store, cls)
    assert store.sha_from_sums(c) == sha256_file(c)
