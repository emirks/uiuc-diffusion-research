"""Shared lexicons, tokenisers and leakage filters for the CTT v2 caption pipeline.

Implements advisor A4 (round 8, 2026-07-27) Q1 / Q3 / Q4, with every measurement
function calibrated to reproduce the M1 corpus numbers pinned in
`misc/ctt_v2_final/DOSSIER.md` §4 exactly.  Calibration results (corpus n=171
descriptions from 139 certified captions):

    word count        `text.split()`                -> reproduces
                                                       M1_length_empirical.json bit-for-bit
    determiner rate   first token in {a, an, the}   -> A 134/139, B 31/32  (pinned)
    -ing participle   r"\\b\\w+ing\\b"                -> B 29/32            (pinned)
    audio words       AUDIO_WORDS                   -> 0/171               (pinned)
    camera phrase     CAMERA_PHRASES                -> 6/171 = 3.51%       (pinned 3.5%)
    colour density    COLOUR_WORDS                  -> 3.199 (pinned 3.158, +1.3%);
                                                       >=1 colour term 158/171 exact

The colour lexicon is the only detector that does not reproduce its pinned value
to the digit; gate #10 is REVIEW-only with a 0.5x-1.5x band so the 1.3% offset is
immaterial.  It is reported explicitly rather than tuned to fit.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[3]
LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")

CORPUS_CAPTIONS = LAB / "diffusion-research/eval_ladder/dataset/captions/dataset_captions.json"
CORPUS_MANIFEST = REPO / "data/processed/transitions_std121/corpus_manifest.json"
STRIPS_INDEX = REPO / "data/processed/caption_strips/strips_index.json"
SHADER_DIR = LAB / "misc/gl-transitions/transitions"
REFVFX_LORA_INDEX = LAB / "diffusion-research/data/raw/refvfx/_viewer_index/lora.jsonl.gz"
LENGTH_EMPIRICAL = LAB / "misc/ctt_v2_final/M1_length_empirical.json"


# --------------------------------------------------------------------------
# Tokenisation / corpus loading
# --------------------------------------------------------------------------
def word_count(text: str) -> int:
    """Whitespace word count.  Calibrated: reproduces M1_length_empirical.json."""
    return len(text.split())


def load_corpus_descriptions():
    """Return (A_descriptions, B_descriptions) from the 139 certified captions.

    Caption grammar is `"{descrA}. sksz."` or `"{descrA}. sksz. {descrB}."`.
    Descriptions are returned WITH their trailing period stripped, matching the
    storage convention of the new description store.
    """
    rows = json.loads(CORPUS_CAPTIONS.read_text())
    a_side, b_side = [], []
    for r in rows:
        parts = r["caption"].strip().split(" sksz.")
        a_side.append(parts[0].strip().rstrip(".").strip())
        rest = parts[1].strip() if len(parts) > 1 else ""
        if rest:
            b_side.append(rest.strip().rstrip(".").strip())
    return a_side, b_side


def load_length_empirical():
    return json.loads(LENGTH_EMPIRICAL.read_text())


# --------------------------------------------------------------------------
# Measurement lexicons (gate battery, A4 Q4)
# --------------------------------------------------------------------------
DETERMINERS = {"a", "an", "the"}


def opens_with_determiner(text: str) -> bool:
    toks = text.split()
    if not toks:
        return False
    return toks[0].lower().strip(",.;:") in DETERMINERS


ING_RE = re.compile(r"\b\w+ing\b", re.I)


def has_ing_participle(text: str) -> bool:
    return bool(ING_RE.search(text))


COLOUR_WORDS = set(
    """red orange yellow green blue purple violet pink brown black white gray grey
    gold golden silver beige tan cream teal turquoise magenta crimson scarlet maroon
    navy olive amber ivory bronze copper charcoal lavender indigo peach coral mustard
    burgundy khaki emerald ruby sapphire jade rust blonde blond auburn ginger
    platinum""".split()
)


def colour_count(text: str) -> int:
    n = 0
    for w in text.split():
        t = re.sub(r"[^a-z\-]", "", w.lower())
        for part in t.split("-"):
            if part in COLOUR_WORDS:
                n += 1
    return n


CAMERA_PHRASES = [
    "overhead view", "high angle", "low angle", "close-up", "close up", "closeup",
    "bird's-eye", "bird's eye", "aerial view", "wide shot", "medium shot",
    "point of view", "top-down", "over-the-shoulder", "from above", "from below",
    "eye-level",
]


def has_camera_phrase(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in CAMERA_PHRASES)


# A4 Q4 gate #6 bans the LTX-2 "audio layer" (corpus has 0/171).  The list below
# is restricted to words that denote SOUND.  Visible speech/singing ACTIONS
# ("a man singing into a microphone") are legitimate visual content and are
# tracked separately in SPEECH_ACTION_WORDS, reported but not gated.  Both lists
# score 0/171 on the corpus, so the pinned calibration is unaffected either way.
AUDIO_WORDS = [
    "sound", "sounds", "music", "musical", "voice", "voices", "says", "said",
    "audio", "noise", "noises", "laughter", "whisper", "whispers", "audible",
    "song", "songs", "melody", "soundtrack", "humming", "hums", "echo", "echoes",
]
SPEECH_ACTION_WORDS = [
    "singing", "sings", "talking", "talks", "speaking", "speaks", "shouting",
    "shouts",
]
_SPEECH_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in SPEECH_ACTION_WORDS) + r")\b", re.I
)


def speech_action_hits(text: str):
    return sorted(set(m.group(0).lower() for m in _SPEECH_RE.finditer(text)))
_AUDIO_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in AUDIO_WORDS) + r")\b", re.I)


def audio_hits(text: str):
    return sorted(set(m.group(0).lower() for m in _AUDIO_RE.finditer(text)))


_MARKUP_RE = re.compile(r'[*_`#\[\]<>"]|\bStyle:')


def markup_hits(text: str):
    return sorted(set(m.group(0) for m in _MARKUP_RE.finditer(text)))


def jaccard(a: str, b: str) -> float:
    sa = set(w.lower().strip(".,;:") for w in a.split())
    sb = set(w.lower().strip(".,;:") for w in b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# --------------------------------------------------------------------------
# Function-word vocabulary for gate #8 (content words EXCLUDED)
# --------------------------------------------------------------------------
FUNCTION_WORDS = sorted(set("""
a an the this that these those
i you he she it we they me him her us them my your his its our their mine yours
hers ours theirs myself yourself himself herself itself ourselves themselves
who whom whose which what where when why how
and or but nor so yet for if then than as because although though while whereas
unless until since whether either neither both
of in on at by to from with without within into onto upon over under above below
beneath beside besides between among around across along through throughout
behind before after against toward towards near beyond off out up down about
during despite per via inside outside
is are was were be been being am
do does did done doing
have has had having
can could shall should will would may might must
not no none nor never nothing
all any each every few many more most much several some such other others another
one two three four five six seven eight nine ten
very too also just only even still already yet again once
there here now then thus hence however moreover therefore
s t re ve ll d m
""".split()))

PUNCT_FEATURES = [",", ".", ";", ":", "-", "'", "(", ")", "/"]

_TOKEN_RE = re.compile(r"[A-Za-z']+")


def function_word_tokens(text: str):
    """Tokens restricted to the closed-class vocabulary; content words dropped."""
    fw = set(FUNCTION_WORDS)
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() in fw]


def all_word_tokens(text: str):
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# --------------------------------------------------------------------------
# Leakage filter vocabularies (A4 Q3, Layer 1)
# --------------------------------------------------------------------------
def load_shader_basenames():
    return sorted(p.stem for p in SHADER_DIR.glob("*.glsl"))


def load_class_names():
    m = json.loads(CORPUS_MANIFEST.read_text())
    return sorted(m["classes"].keys())


def load_refvfx_triggers():
    """All distinct refVFX trigger strings from the FULL 6,995-clip raw pool.

    `lora.jsonl.gz` has 6,995 rows and 47 distinct non-empty `et` (effect-type /
    trigger) strings -- a superset of the 47 in `s4_refvfx/selection.json`
    (42 train + 5 held-out).
    """
    import gzip
    trigs = set()
    with gzip.open(REFVFX_LORA_INDEX, "rt") as fh:
        for line in fh:
            et = json.loads(line).get("et") or ""
            et = et.strip()
            if et:
                trigs.add(et)
    return sorted(trigs)


# Shader basenames that are ordinary English words plausibly describing real
# scene content.  Per A4 Q3(iv) these route to Tier-2 REVIEW instead of Tier-1
# HARD.  A4 named "Slides, Swap, Doorway, ..."; the list below is the operator's
# completion of the "...".  Words that are simultaneously A4 mechanism words
# (dissolve, morph, glitch) are deliberately NOT escaped -- Tier-1 (vi) keeps them.
SHADER_ESCAPE = {
    "angular", "bounce", "box", "burn", "burn0", "chessboard", "circle", "crosshatch",
    "cube", "directional", "doorway", "dreamy", "fade", "fold", "fragment", "heart",
    "mosaic", "perlin", "pinwheel", "polar", "radial", "rectangle", "ripple", "rolls",
    "slides", "squeeze", "static", "swap", "swirl", "wind",
}

# A4 Q3 (vi): mechanism words that are never legitimate in a static 9-frame
# description.  Prefix patterns are expressed as regexes.
MECHANISM_PATTERNS = [
    r"\btransform\w*\b", r"\btransition\w*\b", r"\bmorph\w*\b", r"\bdissolv\w*\b",
    r"\bmetamorpho\w*\b", r"\bteleport\w*\b", r"\bturns? into\b", r"\bchanges? into\b",
    r"\bbecomes? a\b", r"\bvfx\b", r"\bcgi\b", r"\bshader\b", r"\boverlay\w*\b",
    r"\bglitch\w*\b",
]

# A4 Q3 (ii): leetspeak pattern (catches cr34sh-family drift)
LEETSPEAK_RE = re.compile(r"\b[a-z]*\d+[a-z]+\w*\b", re.I)

OUTCOME_MARKER = "The scene transforms into "

# A4 Q3 Tier-2: stylistic-drift regexes
DRIFT_RE = re.compile(r"\b(shift|fad|melt|split|warp|reveal|emerg|appear|vanish)\w*\b", re.I)
INCHOATIVE_RE = re.compile(r"\b(begin(s|ning)? to|starts? to|about to)\b", re.I)


def _word_boundary_re(s: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(s) + r"(?![A-Za-z0-9_])", re.I)


class LeakFilter:
    """A4 Q3 Layer-1: Tier-1 HARD (auto-reject) + Tier-2 REVIEW (auto-flag)."""

    def __init__(self):
        self.shaders = load_shader_basenames()
        self.classes = load_class_names()
        self.triggers = load_refvfx_triggers()

        # ---- Tier-1 -----------------------------------------------------
        # (i) refVFX trigger strings, plus their leading leet code tokens
        self.t1_trigger_strings = sorted(set(self.triggers))
        self.t1_trigger_tokens = sorted(
            {t.split()[0].strip(".,") for t in self.triggers if t.split()}
        )
        # (iv) shader basenames as whole tokens, minus the escape list
        self.t1_shaders = [s for s in self.shaders if s.lower() not in SHADER_ESCAPE]
        # (v) class names as snake_case tokens AND de-underscored word sequences.
        #     Single-word class names have no de-underscored form and are ordinary
        #     English (flame, shadow, portal); A4's Tier-2 rule explicitly owns
        #     "single constituent words of class names", so they go to Tier-2.
        self.t1_classes = []
        self.t2_class_words = set()
        for c in self.classes:
            if "_" in c:
                self.t1_classes.append(c)                  # snake_case token
                self.t1_classes.append(c.replace("_", " "))  # de-underscored phrase
                self.t2_class_words.update(c.split("_"))
            else:
                self.t2_class_words.add(c)

        self._t1_exact = [
            ("trigger", s) for s in self.t1_trigger_strings
        ] + [
            ("trigger_token", s) for s in self.t1_trigger_tokens
        ] + [
            ("shader", s) for s in self.t1_shaders
        ] + [
            ("class", s) for s in self.t1_classes
        ]
        self._t1_exact_re = [(kind, s, _word_boundary_re(s)) for kind, s in self._t1_exact]
        self._mech_re = [re.compile(p, re.I) for p in MECHANISM_PATTERNS]

        # ---- Tier-2 -----------------------------------------------------
        self.t2_class_words = {w for w in self.t2_class_words if len(w) > 2}
        self._t2_class_re = [(w, _word_boundary_re(w)) for w in sorted(self.t2_class_words)]
        self._t2_shader_re = [
            (s, _word_boundary_re(s)) for s in sorted(SHADER_ESCAPE)
        ]

    # -- Tier 1 -----------------------------------------------------------
    def tier1(self, text: str):
        hits = []
        for kind, s, rx in self._t1_exact_re:
            if rx.search(text):
                hits.append(f"{kind}:{s}")
        for rx in self._mech_re:
            m = rx.search(text)
            if m:
                hits.append(f"mechanism:{m.group(0).lower()}")
        for m in LEETSPEAK_RE.finditer(text):
            hits.append(f"leetspeak:{m.group(0)}")
        if _word_boundary_re("sksz").search(text):
            hits.append("sksz")
        if OUTCOME_MARKER.lower() in text.lower():
            hits.append("outcome_marker")
        return sorted(set(hits))

    # -- Tier 2 -----------------------------------------------------------
    def tier2(self, text: str):
        flags = []
        for w, rx in self._t2_class_re:
            if rx.search(text):
                flags.append(f"class_word:{w}")
        for s, rx in self._t2_shader_re:
            if rx.search(text):
                flags.append(f"shader_escape:{s}")
        for m in DRIFT_RE.finditer(text):
            flags.append(f"drift:{m.group(0).lower()}")
        for m in INCHOATIVE_RE.finditer(text):
            flags.append(f"inchoative:{m.group(0).lower()}")
        return sorted(set(flags))


# --------------------------------------------------------------------------
# Format asserts (A4 Q1 assembly contract)
# --------------------------------------------------------------------------
def format_violations(text: str, role: str):
    """Return a list of format-assert violations for a stored description."""
    v = []
    if not text:
        return ["empty"]
    if ". " in text:
        v.append("multi_sentence")
    if text.endswith("."):
        v.append("trailing_period_not_stripped")
    if role == "A" and not text[0].isupper():
        v.append("A_not_uppercase_initial")
    if role == "B" and not text[0].islower():
        v.append("B_not_lowercase_initial")
    if _word_boundary_re("sksz").search(text):
        v.append("sksz_present")
    if markup_hits(text):
        v.append("markup:" + ",".join(markup_hits(text)))
    if audio_hits(text):
        v.append("audio:" + ",".join(audio_hits(text)))
    n = word_count(text)
    if n < 8 or n > 70:
        v.append(f"length_out_of_sanity_range:{n}")
    return v
