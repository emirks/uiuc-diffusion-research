"""Versioning/stamping contract tests (SPEC §7): pure stdlib, login-node safe."""

import json
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diffusion.transition_eval import versioning as V  # noqa: E402


def test_version_matches_file():
    ver = V.version()
    assert ver == (V.VERSION_PATH.read_text().strip())
    assert re.match(r"^\d+\.\d+\.\d+", ver), ver


def test_spec_hash_present_and_stable():
    h1, h2 = V.spec_sha(), V.spec_sha()
    assert h1 == h2
    assert h1 is not None and re.fullmatch(r"[0-9a-f]{64}", h1)


def test_corpus_sha_is_canonical(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('{"x": 1, "y": [2, 3]}')
    b.write_text('{\n  "y": [2, 3],\n  "x": 1\n}')  # same content, different form
    assert V.corpus_sha(a) == V.corpus_sha(b)
    b.write_text('{"y": [2, 3], "x": 2}')
    assert V.corpus_sha(a) != V.corpus_sha(b)
    assert V.corpus_sha(tmp_path / "missing.json") is None


def test_stamp_shape_and_draft_is_uncertified():
    s = V.stamp()
    for key in ("harness", "version", "certified", "uncertified_reasons",
                "git", "spec_sha256", "corpus_sha256", "pins", "env"):
        assert key in s, key
    assert s["harness"] == f"transition-eval/{s['version']}"
    # current VERSION is a draft -> must never stamp as certified
    if V.is_draft():
        assert s["certified"] is False
        assert s["uncertified_reasons"]
    json.dumps(s)  # stamp must be JSON-serializable as-is


def test_git_state_tolerates_any_checkout():
    g = V.git_state()
    assert set(g) == {"commit", "commit_short", "branch", "exact_tag",
                      "package_dirty", "dirty_files"}
    if g["commit"] is not None:
        assert re.fullmatch(r"[0-9a-f]{40}", g["commit"])


def test_open_pins_block_certification():
    ok, reasons = V.certification(V.git_state(), "3.0.0")
    if any(v is None for v in V.PINS.values()):
        assert not ok and any("OPEN dependency pins" in r for r in reasons)
