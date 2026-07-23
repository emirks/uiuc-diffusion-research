"""ladder2 — the ONE place a prompt is made (training AND inference).

Every prompt in the campaign is RENDERED here from the endpoint captions; none is ever
hand-authored. That is seatbelt #2 ("prompt rendered, not authored") and it is also what
makes train == inference: `build_registry.py` emits the training captions with the same
`render_prompt()` that renders the registry rows, so a specialist never sees at train time
a sentence shape it will not see at generation time.

Caption corpus: every clip's caption is a single string `"<S1>. The scene transforms into <S2>."`
(verified: 222/222 clips carry the marker across the four sources below). S1 is a static
snapshot of the START scene, S2 of the END scene. The OUTCOME half is what leaked the
transition in every prior ladder, so the renderer keeps:

    one-sided  ->  "{S1}. {token}."
    two-sided  ->  "{S1}. {token}. {S2}."

For a one-sided class the end frames ARE the effect, so S2 is dropped entirely: the model
gets the start scene, a neutral token in the transition slot, and nothing else.

Sidedness is read from the owner-final taxonomy (`corpus_manifest.json`) and cross-checked
against `class_axes_v2.yaml`; a disagreement is a hard error, never a silent pick.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STD = REPO_ROOT / "data/processed/transitions_std121"

#: the sentence that separates the start scene from the outcome in every corpus caption
MARKER = "The scene transforms into "

#: caption sources, highest authority first. The first file is the one whose 83 endpoints
#: were audited one-by-one against their frames (0 transition leaks); the rest cover the
#: remaining training clips. Where they overlap they are byte-identical (verified 49/49).
CAPTION_SOURCES = (
    "docs/eval_ladder/clip_captions.json",
    "experiments/exp_058_ic_lora_diverse_retrain/dataset/captions.json",
    "data/processed/transitions_std121/dataset_exp058.json",
    "data/processed/transitions_std121/dataset_exp064_missing.json",
    "data/processed/transitions_std121/dataset.json",
)

#: clips whose caption was audited against the actual frames (leak-free, eval-eligible)
AUDITED_SOURCE = "docs/eval_ladder/clip_captions.json"
#: promoted to test by split v1.2 after the 81-endpoint audit; audited separately 2026-07-22
AUDITED_EXTRA = ("earth_element_6", "money_rain_1")


def _load_any(path: Path) -> dict[str, str]:
    """Read a caption file in either shape: {clip: caption} or [{video, caption}]."""
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if isinstance(v, str)}
    return {Path(r["video"]).stem: r["caption"] for r in data if r.get("caption")}


@lru_cache(maxsize=1)
def captions() -> dict[str, str]:
    """clip -> full caption, first source wins."""
    out: dict[str, str] = {}
    for rel in CAPTION_SOURCES:
        for clip, cap in _load_any(REPO_ROOT / rel).items():
            out.setdefault(clip, cap)
    return out


@lru_cache(maxsize=1)
def audited_clips() -> frozenset[str]:
    """Clips whose caption was checked against its frames — the only eval-eligible endpoints."""
    return frozenset(_load_any(REPO_ROOT / AUDITED_SOURCE)) | frozenset(AUDITED_EXTRA)


@lru_cache(maxsize=1)
def sidedness() -> dict[str, str]:
    """class -> 'one' | 'two', from corpus_manifest.json, cross-checked against class_axes_v2."""
    import yaml

    corpus = json.loads((STD / "corpus_manifest.json").read_text())["classes"]
    axes = yaml.safe_load((REPO_ROOT / "outputs/taxonomy/class_axes_v2.yaml").read_text())["classes"]
    out = {}
    for cls, entry in corpus.items():
        side = {"onesided": "one", "twosided": "two"}[entry["sidedness"]]
        ax = axes.get(cls, {}).get("sidedness")
        if ax is not None:
            expect = {"A_only": "one", "two_sided": "two"}[ax]
            if expect != side:
                raise AssertionError(f"sidedness disagreement for {cls}: manifest={side} axes={expect}")
        out[cls] = side
    return out


@lru_cache(maxsize=1)
def clip_index() -> dict[str, str]:
    """clip -> class, read from the frozen split (never string-split the clip name).

    Clip names do NOT reliably encode the class: `action_run_setonfire_6` belongs to class
    `run_set_on_fire` and `flame_transition_0` to class `flame`. Deriving the class from the
    name silently mislabels those rows — which is exactly the kind of hand-derived fact the
    ladder2 redesign exists to eliminate.
    """
    split = json.loads((STD / "split_v1.2.json").read_text())["classes"]
    return {clip: cls for cls, entry in split.items() for clip in entry["paths"]}


def clip_class(clip: str) -> str:
    """Authoritative clip -> class."""
    try:
        return clip_index()[clip]
    except KeyError:
        raise KeyError(f"{clip} is not in split_v1.2") from None


def split_caption(clip: str) -> tuple[str, str]:
    """clip -> (S1, S2), both stripped of trailing punctuation/whitespace."""
    cap = captions().get(clip)
    if cap is None:
        raise KeyError(f"no caption for {clip}")
    if MARKER not in cap:
        raise ValueError(f"caption for {clip} lacks the marker {MARKER!r}")
    s1, s2 = cap.split(MARKER, 1)
    return _trim(s1), _trim(s2)


def _trim(text: str) -> str:
    return text.strip().rstrip(".,;: ").strip()


def render_prompt(clip: str, sided: str, token: str) -> str:
    """The campaign's only prompt renderer.

    `sided` is the sidedness of the CONDITIONING (one endpoint or two) — always the
    endpoint clip's own sidedness, since strict sidedness-matching guarantees it equals
    the transition source's.
    """
    if sided not in ("one", "two"):
        raise ValueError(f"sided must be 'one' or 'two', got {sided!r}")
    s1, s2 = split_caption(clip)
    if sided == "one":
        return f"{s1}. {token}."
    return f"{s1}. {token}. {s2}."


def render_for_clip(clip: str, token: str) -> str:
    """Render using the clip's own class sidedness (the training case)."""
    return render_prompt(clip, sidedness()[clip_class(clip)], token)


if __name__ == "__main__":  # tiny CLI: eyeball a few renders
    import sys

    tok = sys.argv[1] if len(sys.argv) > 1 else "sksz"
    for clip in sys.argv[2:] or ["color_rain_0", "shadow_smoke_7", "earth_element_6"]:
        print(f"--- {clip} [{sidedness()[clip_class(clip)]}-sided]\n{render_for_clip(clip, tok)}\n")
