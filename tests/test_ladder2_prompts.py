"""ladder2 — render_prompt unit tests (seatbelt #2: prompts are rendered, never authored).

Run: pytest tests/test_ladder2_prompts.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments/ladder2"))

import prompts  # noqa: E402

TOKEN = "sksz"
LEAK_WORDS = (
    "transforms into",
    "transitions",
    "transition into",
    "morphs into",
    "turns into",
    "becomes a",
    "the scene changes",
)


def test_one_sided_drops_the_outcome():
    out = prompts.render_prompt("color_rain_0", "one", TOKEN)
    s1, s2 = prompts.split_caption("color_rain_0")
    assert out == f"{s1}. {TOKEN}."
    assert s2 not in out
    assert prompts.MARKER not in out


def test_two_sided_keeps_both_scenes_with_token_between():
    out = prompts.render_prompt("shadow_smoke_7", "two", TOKEN)
    s1, s2 = prompts.split_caption("shadow_smoke_7")
    assert out == f"{s1}. {TOKEN}. {s2}."
    assert out.index(TOKEN) > out.index(s1[:20])
    assert out.index(TOKEN) < out.index(s2[:20])


def test_no_double_punctuation():
    for clip in ("color_rain_0", "shadow_smoke_7", "earth_element_6", "money_rain_1"):
        for sided in ("one", "two"):
            out = prompts.render_prompt(clip, sided, TOKEN)
            assert ".." not in out and " ." not in out and ",." not in out


def test_marker_never_survives_any_render():
    for clip in list(prompts.captions())[:60]:
        for sided in ("one", "two"):
            assert prompts.MARKER not in prompts.render_prompt(clip, sided, TOKEN)


def test_rendered_prompts_carry_no_transition_verbs():
    """The audited-endpoint guarantee, re-asserted mechanically on every render."""
    for clip in sorted(prompts.audited_clips()):
        sided = prompts.sidedness()[prompts.clip_class(clip)]
        low = prompts.render_prompt(clip, sided, TOKEN).lower()
        for bad in LEAK_WORDS:
            assert bad not in low, f"{clip}: leak word {bad!r}"


def test_sidedness_sources_agree_and_are_complete():
    side = prompts.sidedness()
    assert set(side.values()) == {"one", "two"}
    assert len(side) == 39
    assert sum(v == "two" for v in side.values()) == 15


def test_every_split_clip_renders():
    import json

    split = json.loads((REPO_ROOT / "data/processed/transitions_std121/split_v1.2.json").read_text())
    missing = []
    for cls, entry in split["classes"].items():
        for clip in entry["train"] + entry["test"]:
            try:
                prompts.render_for_clip(clip, TOKEN)
            except KeyError:
                missing.append(clip)
    assert missing == ["hole_transition_1"], missing  # sole known caption gap (2-clip held-in class)


def test_token_is_substituted_verbatim():
    for tok in ("sksz", "qvtr"):
        assert f" {tok}." in prompts.render_prompt("color_rain_0", "one", tok)


@pytest.mark.parametrize("sided", ["", "One", "1", None])
def test_bad_sidedness_rejected(sided):
    with pytest.raises((ValueError, TypeError)):
        prompts.render_prompt("color_rain_0", sided, TOKEN)
