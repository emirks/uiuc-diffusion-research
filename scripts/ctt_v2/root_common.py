"""CTT v2 — shared machinery for root assembly and the pre-launch assert battery.

Authority: `misc/ctt_v2_final/advisors/A5_SYNTHESIS_RULING_VERBATIM.md` RULING 4 (mix +
pairing + holdouts) and RULING 9 (the stamp + the HARD asserts), and
`misc/ctt_v2_final/REF_root_format.md` (the on-disk root contract).

Nothing in here makes a judgement call: every constant traces to a ruling, and every
derived set is recomputed from the frozen sources on disk (never a hand-kept list).

Root layout produced by `assemble_root.py` (5 dirs, identical relative path in each):

    <root>/<dir>/<stratum>_r<NN>/<group>/<target>__ref_<reference>.pt

`<dir>` in {latents, conditions, cond_clean_latents, masks, reference_latents}.
`<stratum>_r<NN>` is the *replica* directory: mix weights are realised by duplicating
whole stratum replicas, so the realised ratio is a property of the root on disk and can
be counted (A3-F8.3 / RULING 4).  `<group>` is the class (S0) or the op id (S1/S2/S4) —
the unit the ring-offset pairing runs inside.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
DOSSIER_DIR = LAB / "misc/ctt_v2_final"
#: the checkout the worktree hangs off.  Bulk precompute trees (`.precomputed_clips/`) are
#: gitignored, so they exist only there; the ic_gen root's own symlinks point at them.
MAIN_REPO = LAB / "diffusion-research"


#: `.claude/worktrees/<name>/` — an ephemeral checkout.  A recorded SOURCE path must never
#: contain one: `REPO_ROOT` is `Path(__file__).resolve().parents[1]`, so an inventory built
#: from a worktree bakes that worktree into every source path it records, and the paths break
#: the moment the worktree is removed.  The data itself is fine (the worktree's files are
#: symlinks into the main checkout) — it is the RECORD that rots, silently, later.
_WORKTREE_RE = re.compile(r"/\.claude/worktrees/[^/]+/")


def canonical_source(p: str | Path) -> str:
    """Strip any ephemeral-worktree component from a source path, and prove the result exists.

    Deliberately NOT `os.path.realpath`: that resolves through the storage mount and yields a
    different prefix (`/taiga/...`) than the `$LAB` convention every other path in this repo
    uses, which would trade one portability problem for another.  Collapsing the worktree
    component keeps the standard prefix and is exactly reversible by inspection.
    """
    s = str(p)
    out = _WORKTREE_RE.sub("/", s)
    if out != s and not Path(out).exists():
        raise SystemExit(
            f"[canonical_source] stripping the worktree component from\n  {s}\ngives\n  {out}\n"
            f"which does NOT exist. Refusing to record either: the worktree-prefixed path "
            f"would rot when the worktree is removed, and the stripped one is wrong now.")
    return out


def assert_no_worktree_paths(obj, where: str) -> None:
    """HARD: no recorded source path may reference an ephemeral worktree."""
    hits = []

    def walk(o, trail):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{trail}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{trail}[{i}]")
        elif isinstance(o, str) and _WORKTREE_RE.search(o):
            hits.append(f"{trail}: {o}")

    walk(obj, where)
    if hits:
        raise SystemExit(
            f"[worktree-path] {len(hits)} recorded path(s) point into an ephemeral "
            f"`.claude/worktrees/` checkout and would break when it is removed. "
            f"Run the builder from the MAIN checkout, or route the path through "
            f"`canonical_source()`. First 5:\n  " + "\n  ".join(hits[:5]))


def resolve_repo_rel(rel: str | Path) -> Path:
    """A repo-relative path, resolved against the worktree first, then the main checkout.

    READ-ONLY: the main tree is never written to.
    """
    rel = Path(rel)
    if rel.is_absolute():
        return rel
    here = REPO_ROOT / rel
    if here.exists():
        return here
    there = MAIN_REPO / rel
    return there if there.exists() else here

# --------------------------------------------------------------------------------------
# Frozen constants (RULING 4 / RULING 9 / REF_root_format.md)
# --------------------------------------------------------------------------------------
ROOT_DIRS = ("latents", "conditions", "cond_clean_latents", "masks", "reference_latents")

#: RULING 4 — ring offset within op, k = min(3, n-1), applied EVERYWHERE
MAX_REFS_PER_TARGET = 3
PAIRING_RULE = "ring_offset_within_op__k=min(3,n-1)"

#: A9 (2026-07-28, DOSSIER §12) — S4 REINSTATED, reversing A5 RULING 2.
#: A12 (2026-07-28, `advisors/A12_prorata_s2_split_VERBATIM.md`) — the S2a:S2b split is
#: DERIVED, never declared.
#:
#: ══ THE MIX CONTRACT ══  **S0 15 / S1 6 / S2 total 69 / S4 10**, and the S2a:S2b split is
#: derived PRO-RATA from the assembled POST-EXCLUSION base pair counts.  Only these
#: stratum-level weights are fixed numbers; there is no fixed number for either S2 half.
#:
#: A9's operative clause reads "S2 total 69, split pro-rata to the A5-ratified assembled
#: counts, which are ~equal": **"pro-rata" is the instruction, "~equal" was an observation**
#: of counts that had not yet met the exclusions.  Post-exclusion they are NOT equal
#: (S2a 22,731 vs S2b 23,577 base pairs — S2a loses 333 clips incl. the 8 inline-OOD ops,
#: S2b loses 131), and forcing an equal SHARE onto unequal BASES can only be realised by
#: differentially duplicating the two halves — which is precisely the "extra reweighting
#: knob" A1b excluded by name ("uniform per-sample weight within S2; equal counts make this
#: automatic; no extra reweighting knob").  Forced-equal also breaks A9's own stated
#: rationale for S2 = 69, which is per-op exposure (~4.3 draws/op): it hands every surviving
#: S2a op ~3.7 % more expected draws than every S2b op, purely by which half its sibling
#: exclusions happened to fall in.  Pro-rata — multiplier 1 on both halves — is the unique
#: implementation of A1b's spec under the actual counts.
#:
#: The value here previously read `{S2a 34.5, S2b 34.5}`, and before that
#: `{S0 15, S1 6, S2a 39.5, S2b 39.5, S4 0.0}` from the SUPERSEDED "S4 OUT" ruling.  Both
#: were live landmines of the SAME kind: assert A3 validates the realized mix AGAINST these
#: constants, so an assembly run would have built the wrong mix and then certified it as
#: correct.  The fix is the same in both cases — **derive, never restate**.  The split is
#: computed exactly once, in `expand_prorata_weights()`, from the counts the assembler
#: actually produced; its inputs are frozen in `PREREG_mix_inputs.json`.
STRATUM_WEIGHTS_PCT = {"S0": 15.0, "S1": 6.0, "S2": 69.0, "S4": 10.0}

#: Aggregate weight -> the member strata it splits across, PRO-RATA to their assembled
#: post-exclusion base pair counts (A12).  S1 and S4 are deliberately NOT in here: they are
#: independent strata with their own weight rationales, not a pro-rata pair.
PRORATA_GROUPS = {"S2": ("S2a", "S2b")}

#: The strata the mix names, aggregates expanded — the assembly / manifest key space.
MIX_STRATA = ("S0", "S1", "S2a", "S2b", "S4")

MIX_TOLERANCE_PP = 0.5


def weight_owner(stratum: str) -> str:
    """The mix-contract weight that owns this stratum.  'S2a' -> 'S2'; 'S0' -> 'S0'."""
    for _g, _members in PRORATA_GROUPS.items():
        if stratum in _members:
            return _g
    return stratum


def expand_strata(names) -> tuple:
    """Mix-contract space -> assembly space: ('S0','S2') -> ('S0','S2a','S2b')."""
    out = []
    for n in names:
        out.extend(PRORATA_GROUPS.get(n, (n,)))
    return tuple(out)


#: Guard: these weights are a RULING, not a preference. If they are edited again, the sum
#: must still be 100, every pro-rata group must own a declared weight, and the expansion
#: must reproduce exactly the strata the assembly names.
assert abs(sum(STRATUM_WEIGHTS_PCT.values()) - 100.0) < 1e-9, (
    f"STRATUM_WEIGHTS_PCT must sum to 100, got {sum(STRATUM_WEIGHTS_PCT.values())}"
)
assert set(PRORATA_GROUPS) <= set(STRATUM_WEIGHTS_PCT), (
    f"every pro-rata group must own a declared stratum weight; "
    f"{sorted(set(PRORATA_GROUPS) - set(STRATUM_WEIGHTS_PCT))} does not"
)
assert set(expand_strata(STRATUM_WEIGHTS_PCT)) == set(MIX_STRATA), (
    f"MIX_STRATA {sorted(MIX_STRATA)} != the expansion of STRATUM_WEIGHTS_PCT "
    f"{sorted(expand_strata(STRATUM_WEIGHTS_PCT))}"
)
assert len(expand_strata(STRATUM_WEIGHTS_PCT)) == len(MIX_STRATA), (
    "a stratum appears in more than one pro-rata group"
)

#: A9's PRE-REGISTERED CONTINGENCY BRANCHES, ratified verbatim by A11 item 3 (2026-07-28)
#: and RESTATED by A12 in mix-contract space — "the equal-halves numbers there are the same
#: illustration convention":
#:   "S1-fail -> S0 15 / S2 total 73 / S4 12;  S4-cutoff -> S0 15 / S1 6 / S2 total 79;
#:    both -> S0 15 / S2 total 85"  — each S2 total split PRO-RATA, exactly as the headline.
#: A5 RULING 3's original S1-drop literal (15 / 42.5 / 42.5) was computed while S4 was OUT
#: and the three shares had to absorb all 85 pp; it is now the *both-absent* branch, not the
#: S1-only one.  Keyed by ",".join(sorted(absent)) — exactly the key `assemble_root.py`
#: builds — so the branch is SELECTED, never re-derived by hand at assembly time.
ABSENT_BRANCH_WEIGHTS_PCT = {
    "S1":    {"S0": 15.0, "S2": 73.0, "S4": 12.0},
    "S4":    {"S0": 15.0, "S1": 6.0, "S2": 79.0},
    "S1,S4": {"S0": 15.0, "S2": 85.0},
}

#: Guard: every branch must sum to 100 and cover EXACTLY the complement of its own key.
#: A branch key is in ASSEMBLY space (it names the strata that are absent from disk), so it
#: may only remove WHOLE pro-rata groups — half a group absent has no pre-registered branch
#: and falls to ABSENT_POLICY, which is the honest outcome rather than an invented split.
for _absent_key, _branch in ABSENT_BRANCH_WEIGHTS_PCT.items():
    _gone = set(_absent_key.split(","))
    assert _gone <= set(MIX_STRATA), f"unknown stratum in branch key {_absent_key!r}"
    _gone_w = {weight_owner(s) for s in _gone}
    for _g in _gone_w:
        assert set(PRORATA_GROUPS.get(_g, (_g,))) <= _gone, (
            f"ABSENT_BRANCH_WEIGHTS_PCT[{_absent_key!r}] removes only part of the pro-rata "
            f"group {_g!r} ({sorted(PRORATA_GROUPS.get(_g, ()))}); a partial removal has no "
            f"pre-registered branch — ABSENT_POLICY applies instead"
        )
    assert set(_branch) == set(STRATUM_WEIGHTS_PCT) - _gone_w, (
        f"ABSENT_BRANCH_WEIGHTS_PCT[{_absent_key!r}] must name exactly "
        f"{sorted(set(STRATUM_WEIGHTS_PCT) - _gone_w)}, got {sorted(_branch)}"
    )
    assert abs(sum(_branch.values()) - 100.0) < 1e-9, (
        f"ABSENT_BRANCH_WEIGHTS_PCT[{_absent_key!r}] must sum to 100, got {sum(_branch.values())}"
    )
del _absent_key, _branch, _gone, _gone_w, _g

#: Fallback for an absent-set A9 did NOT pre-register a branch for (e.g. S2b alone).
ABSENT_POLICY = "renormalize_proportionally"

# ══════════════════════════════════════════════════════════════════════════════════════
# MULTI-DATASET MIX CONTRACTS (advisor 2026-08-28, misc/2026-08-28_effectdata_s6/BUILD.md §7)
# ══════════════════════════════════════════════════════════════════════════════════════
# The globals ABOVE are, and remain, the certified 002_ctt_v2 contract (A5/A9/A11/A12). A new
# dataset registers its OWN contract here WITHOUT mutating them, so 002's ROOT_MANIFEST always
# re-verifies against source at HEAD and assert A3 never becomes a liar (the exact drift class
# this file already fought twice). `assemble_root --contract` selects one, DEFAULT 002_ctt_v2 —
# no existing command can silently build a new mix. Every registered contract runs the SAME
# guard battery as the globals. Derive, never restate: 003 is a pure function of 002.
def _scale_add_stratum(base_w: dict, base_absent: dict, new: str, w_new: float):
    """Additive contract: every base weight x (1 - w_new/100), plus `new` at w_new pp. Absent
    branches map by the same rule; the `new`-absent branch renormalizes back to EXACTLY base."""
    f = (100.0 - w_new) / 100.0
    weights = {s: round(v * f, 6) for s, v in base_w.items()}
    weights[new] = float(w_new)
    absent = {}
    for k, br in base_absent.items():                 # base branches, scaled, + new
        absent[k] = {s: round(v * f, 6) for s, v in br.items()}
        absent[k][new] = float(w_new)
    absent[new] = {s: float(v) for s, v in base_w.items()}   # new absent -> exactly base
    return weights, absent


_S6_W, _S6_ABSENT = _scale_add_stratum(STRATUM_WEIGHTS_PCT, ABSENT_BRANCH_WEIGHTS_PCT, "S6", 20.0)

MIX_CONTRACTS = {
    "002_ctt_v2": {
        "weights": dict(STRATUM_WEIGHTS_PCT),
        "prorata": {k: tuple(v) for k, v in PRORATA_GROUPS.items()},
        "mix_strata": tuple(MIX_STRATA),
        "absent": {k: dict(v) for k, v in ABSENT_BRANCH_WEIGHTS_PCT.items()},
        "authority": "A5/A9/A11/A12 — the certified contract; identical to the module globals.",
    },
    "003_ctt_v2plus": {
        "weights": _S6_W,                              # S0 12 / S1 4.8 / S2 55.2 / S4 8 / S6 20
        "prorata": {"S2": ("S2a", "S2b")},
        "mix_strata": ("S0", "S1", "S2a", "S2b", "S4", "S6"),
        "absent": _S6_ABSENT,
        "authority": "advisor 2026-08-28: additive one-sided EffectData (S6) at 20 pp (owner "
                     "ceiling <=25-30); 002 weights x0.80, DERIVED not restated; S6-absent -> "
                     "exactly 002. Escalation to 25-28 pre-registered iff the paired-arm gate "
                     "shows breadth benefit absent AND core non-regression.",
    },
    "005_ctt_v2plus_s6reshape": {
        "weights": _S6_W,                              # IDENTICAL to 003_ctt_v2plus
        "prorata": {"S2": ("S2a", "S2b")},
        "mix_strata": ("S0", "S1", "S2a", "S2b", "S4", "S6"),
        "absent": _S6_ABSENT,
        "authority": "S6 reshape (misc/2026-08-30_s6_reshape DOSSIER Round 1, advisor "
                     "2026-08-30): weights IDENTICAL to 003_ctt_v2plus; S6 re-encoded at two "
                     "orientation grids (11,16,26)/(11,26,16) and paired within effect x grid",
    },
}


def _validate_contract(cid: str, c: dict) -> None:
    w, pr, ms, ab = c["weights"], c["prorata"], c["mix_strata"], c["absent"]

    def owner(s):
        return next((g for g, mem in pr.items() if s in mem), s)

    def expand(names):
        out = []
        for n in names:
            out.extend(pr.get(n, (n,)))
        return tuple(out)

    assert abs(sum(w.values()) - 100.0) < 1e-9, f"{cid}: weights sum {sum(w.values())} != 100"
    assert set(pr) <= set(w), f"{cid}: a pro-rata group owns no declared weight"
    assert set(expand(w)) == set(ms), f"{cid}: mix_strata != expand(weights)"
    assert len(expand(w)) == len(ms), f"{cid}: a stratum is in >1 pro-rata group"
    for ak, br in ab.items():
        gone = set(ak.split(","))
        assert gone <= set(ms), f"{cid}: unknown stratum in branch key {ak!r}"
        gone_w = {owner(s) for s in gone}
        for g in gone_w:
            assert set(pr.get(g, (g,))) <= gone, (
                f"{cid}: branch {ak!r} removes only part of pro-rata group {g!r}")
        assert set(br) == set(w) - gone_w, (
            f"{cid}: branch {ak!r} names {sorted(br)} != {sorted(set(w) - gone_w)}")
        assert abs(sum(br.values()) - 100.0) < 1e-9, (
            f"{cid}: branch {ak!r} sums to {sum(br.values())} != 100")


for _cid, _c in MIX_CONTRACTS.items():
    _validate_contract(_cid, _c)
#: the 002 registry entry MUST equal the module globals — they can never drift apart
assert MIX_CONTRACTS["002_ctt_v2"]["weights"] == STRATUM_WEIGHTS_PCT
assert MIX_CONTRACTS["002_ctt_v2"]["mix_strata"] == tuple(MIX_STRATA)
assert MIX_CONTRACTS["002_ctt_v2"]["absent"] == ABSENT_BRANCH_WEIGHTS_PCT
del _cid, _c


def mix_contract(dataset_id: str = "002_ctt_v2") -> dict:
    """The mix contract (weights / prorata / mix_strata / absent) for a dataset id."""
    if dataset_id not in MIX_CONTRACTS:
        raise KeyError(f"no mix contract for {dataset_id!r}; known: {sorted(MIX_CONTRACTS)}")
    return MIX_CONTRACTS[dataset_id]


TRIGGER = "sksz"
TRIGGER_SENTENCE = f" {TRIGGER}."
OUTCOME_MARKER = "The scene transforms into "

SEED = 42
N_INLINE_OOD_OPS = 8

# --------------------------------------------------------------------------------------
# TWO SHAPES (A9 §3 + §5 "extend the root asserts to two shapes")
#
# S4 is 33-frame / 16 fps material; every other stratum is 121-frame / 24 fps.  The root
# therefore holds two latent shapes at once, and the trainer's noise schedule is a
# DETERMINISTIC FUNCTION of the token count, so the two shapes train at two different
# shifts.  That is the disclosed-and-controlled confound of A9 §3 — it is only controlled
# if the shapes present in the root are asserted, not assumed.
#
# ⚠ The token/shift numbers in A9's prose (1,500 tokens / shift 1.120) are WRONG and were
# corrected from disk in DOSSIER §13.2: the encoded S4 latent is (128,5,14,26) @ fps 16,
# i.e. 5*14*26 = 1,820 tokens => shift 1.2350.  A9's pre-written smoke-gate assert
# `shifts in {1.120, 2.302}` would FAIL on a correct encode.  These constants are derived
# from the shape, never restated, so the arithmetic cannot drift again.
# --------------------------------------------------------------------------------------
#: `ltx_trainer/timestep_samplers.py:122-134` verbatim defaults — NOT clamped upstream,
#: so 4,800 tokens legitimately extrapolates past max_tokens.
SHIFT_MIN_TOKENS, SHIFT_MAX_TOKENS = 1024, 4096
SHIFT_MIN, SHIFT_MAX = 0.95, 2.05

#: latent (F, H/32, W/32) -> the ruled provenance of that shape.  `px` is (W, H, frames),
#: matching `scripts/ctt_v2/encode/encode_strata.py`'s bucket convention.
RULED_SHAPES = {
    (16, 20, 15): {"name": "corpus_121f", "px": (480, 640, 121), "fps": 24.0,
                   "strata": ("S0", "S1", "S2a", "S2b"), "prefix_latents": 2,
                   "authority": "REF_root_format.md (verified by loading)"},
    (5, 14, 26): {"name": "s4_33f", "px": (832, 448, 33), "fps": 16.0, "strata": ("S4",),
                  "prefix_latents": 1,
                  "authority": "DOSSIER §13.2 — verified on disk; 832x448 is a pure "
                               "16-row centre crop, 464 is not a multiple of 32"},
    # EffectData (S6) — one-sided breadth stratum, 81f/24fps clips whose NATIVE resolutions are
    # already VAE-legal (no crop, unlike S4).  The top-2k roster has exactly these 4 native shapes
    # (two transpose pairs).  prefix_latents=1 conditions video frame 0 ALONE, as S4 does — a 9-frame
    # prefix (the 121f default of 2) would reach into the effect onset and mismatch the frame-0
    # caption.  Same frame-0 rationale as S4's owner decision 2026-07-28; see TEXT_LIFECYCLE.md §8.1.
    (11, 22, 39): {"name": "effd_1248x704_81f", "px": (1248, 704, 81), "fps": 24.0,
                   "strata": ("S6",), "prefix_latents": 1,
                   "authority": "EffectData native (VAE-legal); frame-0 anchor per S4 precedent"},
    (11, 39, 22): {"name": "effd_704x1248_81f", "px": (704, 1248, 81), "fps": 24.0,
                   "strata": ("S6",), "prefix_latents": 1,
                   "authority": "EffectData native (VAE-legal); frame-0 anchor per S4 precedent"},
    (11, 33, 22): {"name": "effd_704x1056_81f", "px": (704, 1056, 81), "fps": 24.0,
                   "strata": ("S6",), "prefix_latents": 1,
                   "authority": "EffectData native (VAE-legal); frame-0 anchor per S4 precedent"},
    (11, 22, 33): {"name": "effd_1056x704_81f", "px": (1056, 704, 81), "fps": 24.0,
                   "strata": ("S6",), "prefix_latents": 1,
                   "authority": "EffectData native (VAE-legal); frame-0 anchor per S4 precedent"},
    # EffectData S6 RESHAPE (dataset 005) — the 4 native S6 grids re-encoded to 2 orientation grids
    # at 832x512 / 512x832 px x 81f: 4,576 tokens (95.3% of corpus 4,800), shift 2.222, ~half the
    # per-row compute of the native 7,986-9,438-token zoo, and same-orientation pairing returns
    # 2,286 of the 004 shape-singletons.  Aspect-preserving scale-to-COVER + center-crop (<=4.13%/edge),
    # frame-0 anchor unchanged (prefix_latents=1, sided one), as the native S6 grids.
    (11, 16, 26): {"name": "effd_832x512_81f", "px": (832, 512, 81), "fps": 24.0,
                   "strata": ("S6",), "prefix_latents": 1,
                   "authority": "S6 reshape spec, misc/2026-08-30_s6_reshape/DOSSIER.md Round 1"},
    (11, 26, 16): {"name": "effd_512x832_81f", "px": (512, 832, 81), "fps": 24.0,
                   "strata": ("S6",), "prefix_latents": 1,
                   "authority": "S6 reshape spec, misc/2026-08-30_s6_reshape/DOSSIER.md Round 1"},
}

#: Fallback for an unruled shape.  2 is the 121f value, so an unruled shape behaves like the
#: corpus rather than like S4 — the conservative direction, since 2 conditions MORE.
DEFAULT_PREFIX_LATENTS = 2


def prefix_latents(fhw) -> int:
    """How many leading LATENT frames the prefix anchor conditions, for this shape.

    A shape property, not a constant.  At 121f, latent frame 0 is the causal single-frame
    latent and latent frame 1 covers video frames 1-8, so `prefix_latents=2` is the 9-frame
    endpoint window the corpus captions describe.  S4 conditions **video frame 0 alone**
    (owner decision 2026-07-28), which is latent frame 0 alone, so `prefix_latents=1`.

    This is why the number lives here: at f=5 a 2-latent prefix would condition 40 % of the
    clip, and the S4 caption then describes 9 frames of a 33-frame transition.  One latent
    frame is 20 %, and the caption describes exactly the conditioned pixel.
    """
    return RULED_SHAPES.get(tuple(int(x) for x in fhw), {}).get(
        "prefix_latents", DEFAULT_PREFIX_LATENTS)


def mask_store_name(f: int, h: int, w: int, sided: str) -> str:
    """THE mask-store filename.  Every other site delegates here; nobody restates it.

    `p{prefix}` is in the name on purpose: the prefix width became a SHAPE property when S4
    moved to frame-0 conditioning, so an f5 mask written under the old fixed-2 rule shares
    `(f,h,w,sided)` with one written under the new rule and would be reused SILENTLY.  Naming it
    forces a regeneration instead.

    It is a function because the hand-rolled copies kept drifting.  Three sites had their own
    f-string; the two that computed the pre-rename name failed loudly (`build_smoke_root.py`
    could not find any mask; A11e reported all three real masks simultaneously "missing" and
    "stale" on a root whose every per-sample mask B2 had just verified).  `assemble_root.
    mask_store_path` is the path-returning wrapper around this.
    """
    return f"f{f}_h{h}_w{w}_p{prefix_latents((f, h, w))}_{sided}sided.pt"


def latent_tokens(fhw) -> int:
    """Sequence length the timestep sampler sees = the product of the latent dims."""
    f, h, w = (int(x) for x in fhw)
    return f * h * w


def shift_for_tokens(n_tokens: int) -> float:
    """`ShiftedLogitNormalTimestepSampler._get_shift_for_sequence_length`, verbatim."""
    m = (SHIFT_MAX - SHIFT_MIN) / (SHIFT_MAX_TOKENS - SHIFT_MIN_TOKENS)
    b = SHIFT_MIN - m * SHIFT_MIN_TOKENS
    return m * n_tokens + b


def shape_record(fhw) -> dict:
    """Everything determinable about one shape, derived — never restated."""
    key = tuple(int(x) for x in fhw)
    n = latent_tokens(key)
    ruled = RULED_SHAPES.get(key)
    return {
        "latent_fhw": list(key),
        "tokens": n,
        "shift": round(shift_for_tokens(n), 6),
        "ruled": ruled is not None,
        "name": (ruled or {}).get("name"),
        "px_whf": list((ruled or {}).get("px", ())) or None,
        "fps": (ruled or {}).get("fps"),
        "authority": (ruled or {}).get("authority"),
    }

#: frozen sources
SPLIT_PATH = REPO_ROOT / "data/processed/transitions_std121/split_v1.2.json"
CORPUS_MANIFEST = REPO_ROOT / "data/processed/transitions_std121/corpus_manifest.json"
REGISTRY = REPO_ROOT / "eval_ladder/registry.jsonl"
DAVIS_YAML = REPO_ROOT / "eval_ladder/davis.yaml"
HOLDOUT_S2 = REPO_ROOT / "experiments/exp_082_s2_humanvid/HOLDOUT_S2_UNION.json"
CONTENT_POOL = REPO_ROOT / "data/processed/ctt_v2_strata/CONTENT_POOL_union.json"
SHADER_DIR = LAB / "misc/gl-transitions/transitions"
COPY_GATE_VERDICT = DOSSIER_DIR / "VERIFY_copy_ref_discriminator.md"
PREREG_INLINE_OOD = DOSSIER_DIR / "PREREG_inline_ood_ops_s2a.json"
#: A12 — the frozen inputs of the DERIVED S2a:S2b split.  The registered object is the
#: RULE ("S2 total 69, split pro-rata to the assembled post-exclusion counts"); freezing
#: its inputs is what removes the experimenter degree of freedom the rule would otherwise
#: carry.  Written by `assemble_root.py --write-prereg-mix-inputs`.
PREREG_MIX_INPUTS = DOSSIER_DIR / "PREREG_mix_inputs.json"
#: the M3 pool-drop adjudication.  `CONTENT_POOL_union.json` is deliberately BYTE-UNCHANGED
#: (nothing may desynchronise from the S2b render), so the adjudication lives here instead
#: and `role_scoped_exclusions_for_caption_store` is the operative instruction.
POOL_DROPS = REPO_ROOT / "data/processed/ctt_v2_strata/POOL_DROPS_M3_ADJUDICATION.json"

INVENTORY_SCHEMA = "ctt_v2_stratum_inventory/1"
STRATA_MANIFEST_SCHEMA = "ctt_v2_strata_manifest/1"
ROOT_MANIFEST_SCHEMA = "ctt_v2_root_manifest/1"


# --------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------
def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def _prompts():
    """`eval_ladder/prompts.py` — the ONLY authority for clip -> class."""
    p = str(REPO_ROOT / "eval_ladder")
    if p not in sys.path:
        sys.path.insert(0, p)
    import prompts  # noqa: PLC0415

    return prompts


# --------------------------------------------------------------------------------------
# RULING 4 — pairing
# --------------------------------------------------------------------------------------
def ring_pairs(stems: list[str], max_refs: int = MAX_REFS_PER_TARGET) -> list[tuple[str, str]]:
    """Ring offset within a group, ref != target, k = min(max_refs, n-1).

    Identical semantics to `eval_ladder/train/inventory.py:make_pairs` (validated on the
    385-pair ic_gen root); groups with n < 2 produce nothing.
    """
    n = len(stems)
    if n < 2:
        return []
    k = min(max_refs, n - 1)
    out = []
    for i, target in enumerate(stems):
        for j in range(1, k + 1):
            out.append((target, stems[(i + j) % n]))
    return out


# --------------------------------------------------------------------------------------
# RULING 4 — realise the mix by integer replica duplication
# --------------------------------------------------------------------------------------
def expand_prorata_weights(stratum_weights_pct: dict[str, float],
                           base_counts: dict[str, int],
                           groups: dict | None = None) -> tuple[dict, dict]:
    """A12 — mix-contract weights -> per-stratum weights, PRO-RATA to the base counts.

    Input is mix-contract space (`S0`/`S1`/**`S2`**/`S4`); output is assembly space
    (`S0`/`S1`/**`S2a`**/**`S2b`**/`S4`).  This is the ONE place the S2a:S2b split exists:
    it is computed from the counts the assembler actually produced, so it can never be
    restated as a literal and drift out of agreement with the root on disk — the exact
    failure mode that produced the stale-mix landmine of DATASET §11.1.

    A member with a zero/absent base count takes no share and its siblings absorb the whole
    aggregate; that is what "pro-rata" means when one half is not on disk.  An aggregate
    that carries weight but has NO member on disk is a hard error, not a silent zero.

    Returns `(weights_pct, derivation)`; `derivation` is the audit record that goes into
    `ROOT_MANIFEST.json` and `PREREG_mix_inputs.json`.
    """
    groups = PRORATA_GROUPS if groups is None else groups
    out: dict[str, float] = {}
    derivation: dict[str, dict] = {}
    for name, w in stratum_weights_pct.items():
        members = groups.get(name)
        if not members:
            out[name] = float(w)
            continue
        counts = {m: int(base_counts.get(m, 0) or 0) for m in members}
        total = sum(counts.values())
        if total <= 0:
            raise ValueError(
                f"pro-rata group {name!r} carries weight {w} but every member has a zero "
                f"base count ({counts}); the split is undefined. Drop {name!r} from the "
                f"branch weights instead of splitting nothing.")
        shares = {m: float(w) * counts[m] / total for m, c in counts.items() if c > 0}
        out.update(shares)
        derivation[name] = {
            "rule": "pro-rata to the assembled POST-EXCLUSION base pair counts (A12)",
            "aggregate_weight_pct": float(w),
            "base_counts": counts,
            "base_total": total,
            "derived_weight_pct": {m: round(v, 6) for m, v in sorted(shares.items())},
        }
    return out, derivation


def solve_multipliers(base_counts: dict[str, int], weights_pct: dict[str, float],
                      tol_pp: float = MIX_TOLERANCE_PP, max_mult: int = 200,
                      groups: dict | None = None) -> dict:
    """Choose integer replica counts so the *counted* mix lands within `tol_pp`.

    `base_counts` is ASSEMBLY space (per stratum, S2a and S2b separately); `weights_pct` is
    MIX-CONTRACT space (S2 as one aggregate).  A12: **a pro-rata group is solved as ONE
    unit**, so its members necessarily receive the SAME multiplier and their realized shares
    land pro-rata to their base counts.  Uniform per-sample weight within the group is
    therefore STRUCTURAL — it is not a constraint bolted on after the search — and there is
    no representable state in which the solver hands the two halves different multipliers.
    Assert A3b re-checks the property by counting replica dirs on the assembled root, which
    is what makes it evidence rather than a comment.

    Deterministic sweep: for each weight taken as the anchor and each anchor multiplier,
    derive the others by rounding, then keep the (max-deviation, total-size) minimum.
    Deviations are measured PER STRATUM, against the derived per-stratum targets — exactly
    the quantity assert A3 counts off the root — so the solver cannot pass a mix that A3
    would then reject.  Raises if nothing lands inside tolerance.
    """
    groups = PRORATA_GROUPS if groups is None else groups
    members: dict[str, list[str]] = {}
    for name, w in weights_pct.items():
        if w <= 0:
            continue
        live = sorted(m for m in groups.get(name, (name,)) if base_counts.get(m, 0) > 0)
        if live:
            members[name] = live
    names = sorted(members)
    if not names:
        raise ValueError("no stratum with a positive base count and a positive weight")

    #: aggregate base counts — the unit the search runs over
    agg = {n: sum(base_counts[m] for m in members[n]) for n in names}
    total_w = sum(weights_pct[n] for n in names)
    w = {n: weights_pct[n] / total_w for n in names}
    #: per-stratum targets, DERIVED (never restated): the renormalised aggregate weight
    #: split pro-rata to the assembled counts.
    intended_pct, split = expand_prorata_weights(
        {n: 100.0 * w[n] for n in names}, base_counts, groups)
    intended = {m: v / 100.0 for m, v in intended_pct.items()}

    # Objective: the SMALLEST root that lands inside tolerance.  (Ranking by deviation
    # alone always prefers a bigger root — finer rounding grain — and would silently
    # inflate a 60 k-sample mix into a 500 k-sample one.)
    best = None
    for anchor in names:
        for m in range(1, max_mult + 1):
            target_total = agg[anchor] * m / w[anchor]
            mult = {n: max(1, int(round(w[n] * target_total / agg[n]))) for n in names}
            mult[anchor] = m
            tot = sum(agg[n] * mult[n] for n in names)
            dev = {s: 100.0 * base_counts[s] * mult[n] / tot - 100.0 * intended[s]
                   for n in names for s in members[n]}
            md = round(max(abs(v) for v in dev.values()), 9)
            key = (0, tot, md) if md <= tol_pp else (1, md, tot)
            if best is None or key < best[0]:
                best = (key, dict(mult), dev, tot)

    key, agg_mult, dev, tot = best
    mult = {s: agg_mult[n] for n in names for s in members[n]}
    max_dev = max(abs(v) for v in dev.values())
    rec = {
        "multipliers": mult,
        "aggregate_multipliers": agg_mult,
        "prorata_groups": {n: members[n] for n in names if len(members[n]) > 1},
        "split_rule": "A12 — a pro-rata group is solved as ONE unit, so its members share "
                      "one multiplier and their shares follow their base counts; the "
                      "stratum-level weights (S0 15 / S1 6 / S2 total 69 / S4 10) are the "
                      "only fixed numbers",
        "prorata_split": split,
        "stratum_weights_pct": {n: round(weights_pct[n], 6) for n in names},
        "stratum_weights_pct_renormalized": {n: round(100.0 * w[n], 6) for n in names},
        "base_counts": {s: base_counts[s] for s in mult},
        "aggregate_base_counts": agg,
        "realized_counts": {s: base_counts[s] * mult[s] for s in mult},
        "total": tot,
        "intended_pct": {s: round(intended_pct[s], 6) for s in mult},
        "realized_pct": {s: round(100.0 * base_counts[s] * mult[s] / tot, 6) for s in mult},
        "deviation_pp": {s: round(dev[s], 6) for s in mult},
        "max_deviation_pp": round(max_dev, 6),
        "tolerance_pp": tol_pp,
    }
    if max_dev > tol_pp:
        raise AssertionError(
            f"no integer replica assignment lands the mix within +-{tol_pp} pp "
            f"(best {max_dev:.4f} pp): {rec}")
    return rec


# --------------------------------------------------------------------------------------
# Holdouts / exclusions (RULING 4 + A3-F5b)
# --------------------------------------------------------------------------------------
@dataclass
class Exclusions:
    holdout_shaders: set = field(default_factory=set)          # 10 S2 shader families
    inline_ood_ops: set = field(default_factory=set)           # 8 pre-registered S2a ops
    reserved_pool_clips: set = field(default_factory=set)      # 120 reserved union-pool clips
    zs_classes: set = field(default_factory=set)               # 10 S0 zero-shot classes
    eval_endpoints: set = field(default_factory=set)           # eval + zs-audited + 42 test
    #: {clip_id: {"A", "B"}} — per-(clip, role) caption-store exclusions (M3 adjudication)
    role_scoped_captions: dict = field(default_factory=dict)
    clip_level_captions: set = field(default_factory=set)
    provenance: dict = field(default_factory=dict)

    def as_record(self) -> dict:
        return {
            "holdout_shaders": sorted(self.holdout_shaders),
            "inline_ood_ops": sorted(self.inline_ood_ops),
            "n_reserved_pool_clips": len(self.reserved_pool_clips),
            "reserved_pool_clips": sorted(self.reserved_pool_clips),
            "zs_classes": sorted(self.zs_classes),
            "n_eval_endpoints": len(self.eval_endpoints),
            "eval_endpoints": sorted(self.eval_endpoints),
            "role_scoped_caption_store_exclusions":
                {k: sorted(v) for k, v in sorted(self.role_scoped_captions.items())},
            "clip_level_caption_store_exclusions": sorted(self.clip_level_captions),
            "provenance": self.provenance,
        }

    def caption_store_hits(self, sources) -> list[str]:
        """Which excluded (clip, role) descriptions a sample's caption draws on."""
        hits = []
        for clip, role in sources:
            if clip in self.clip_level_captions:
                hits.append(f"{clip}:*")
            elif role in self.role_scoped_captions.get(clip, ()):
                hits.append(f"{clip}:{role}")
        return sorted(set(hits))


def load_caption_store_exclusions(path: Path | None = None) -> tuple[dict, set, dict]:
    """(role_scoped, clip_level, provenance) from the M3 pool-drop adjudication.

    ABSENT IS A FAILURE, not an empty set: the whole point of the adjudication is that the
    pool file was left byte-unchanged, so this file is the only carrier of the instruction.
    A silently-vacuous exclusion is exactly the class of defect the campaign keeps meeting,
    so the caller is told, in `provenance["error"]`, and the assert battery hard-fails.

    Two keys carry the same disposition — A10's authoritative `role_scoped_exclusions`
    (which also carries `enforced_at`) and the earlier `role_scoped_exclusions_for_caption_
    store`.  Both are read and they must AGREE; a disagreement is a hard error, because a
    half-updated sidecar is how a role exclusion silently narrows to one consumption channel.
    """
    p = Path(path) if path else POOL_DROPS
    if not p.exists():
        return {}, set(), {"file": str(p), "error": "ABSENT — the role-scoped exclusion "
                           "cannot be honoured and would be silently vacuous"}
    rec = read_json(p)
    ruling = {c: set(v.get("excluded_roles") or ()) for c, v in
              (rec.get("role_scoped_exclusions") or {}).items()}
    legacy = {c: set(v) for c, v in
              (rec.get("role_scoped_exclusions_for_caption_store") or {}).items()}
    if ruling and legacy and ruling != legacy:
        raise AssertionError(
            f"{p}: `role_scoped_exclusions` {ruling} disagrees with "
            f"`role_scoped_exclusions_for_caption_store` {legacy} — a half-updated sidecar "
            f"would enforce the exclusion on one consumption channel and not the other")
    role = ruling or legacy
    clip = set(rec.get("clip_level_exclusions_for_caption_store") or [])
    prov = {"file": str(p), "sha256": sha256_file(p),
            "authority": (rec.get("ruling_of_record") or {}).get("advisor") or rec.get("authority"),
            "verdict": (rec.get("ruling_of_record") or {}).get("verdict"),
            "standing_rule": (rec.get("ruling_of_record") or {}).get("standing_rule"),
            "status": rec.get("status"),
            "enforced_at": {c: (v.get("enforced_at") or []) for c, v in
                            (rec.get("role_scoped_exclusions") or {}).items()}}
    return role, clip, prov


# --------------------------------------------------------------------------------------
# THE A16 KEYED-JOIN RULE, IN CODE
# (advisors/A16_29_orphaned_s2a_clips_VERBATIM.md §Q4 items 1 and 4; items 2-3 are
#  dossier-review rules and live in the DOSSIER, not here.)
#
# Three separate incidents in this campaign share ONE mechanism: **an empty query result was
# read as a validated zero.**
#   1. `build_mass_pair_list.py` looked up `endpoint_a`/`endpoint_b` in S2a's rendered meta,
#      whose keys are `A`/`B`.  The strict lookup matched 0 of 454 pairs, and that empty set
#      read as the reassuring "S2a needs no descriptions" — 36 absent (clip, role) pairs.
#   2. A10 verified "all 37 field B, zero field A" first-hand and was right — for S2b.  The
#      universe was never enumerated, so S2a was never scanned: 29 rows consume field A.
#   3. The operator queried `descriptions.get(clip)` against a `clip|role`-keyed store, got
#      `None` for both roles, and briefly concluded the B-role description was missing too.
#
# So: an absent key is an EXCEPTION, never a `None`; a cross-source join that matches nothing
# is a FAILURE, never information; and a lookup key's SHAPE is validated against the store's
# own self-declaration BEFORE any result is interpreted.  `.get()` against a keyed store is
# banned in this lane — every one of the three incidents is a `.get()`-shaped read.
# --------------------------------------------------------------------------------------
#: a key's SHAPE, with all content erased: `davis_breakdance|B` -> `*|*`,
#: `air_bending/action_x.mp4` -> `*/*`, `s2_0000_c00` -> `*`.  Two keys with different
#: signatures cannot be keys of the same store, which is exactly what the three incidents
#: failed to notice.
_KEY_FIELD = re.compile(r"[^|/]+")
#: the first quoted token of a `keying` declaration is the key TEMPLATE (`'clip_id|role'`)
_KEYING_QUOTED = re.compile(r"['\"`]([^'\"`]+)['\"`]")


def key_shape_signature(key: str) -> str:
    """The separator/field signature of a store key or of a key template."""
    return _KEY_FIELD.sub("*", str(key))


def declared_key_signature(keying: str) -> str:
    """The signature a store's `keying` self-declaration promises.

    The declaration is prose with the key template quoted inside it (CAPTION_STORE.json:
    ``"'clip_id|role'. A-role describes frames 0-8, ..."``).  A declaration with no quoted
    template cannot be validated against, and that is a hard error rather than a shrug:
    an unvalidatable self-declaration is the same species of decoration as an exclusion no
    code reads.
    """
    m = _KEYING_QUOTED.search(str(keying))
    if not m:
        raise AssertionError(
            f"keyed store declares `keying` = {keying!r} but no key template is quoted in "
            f"it, so no lookup key can be validated against it. Declare the template, e.g. "
            f"\"'clip_id|role'. <prose>\".")
    return key_shape_signature(m.group(1))


def require_keying_declaration(obj: dict, where: str) -> str:
    """`keying` is MANDATORY on a keyed store artifact (A16 item 4).  Returns it."""
    if not isinstance(obj, dict) or not (obj.get("keying") or "").strip():
        raise AssertionError(
            f"{where}: keyed store artifacts must carry a non-empty `keying` field naming "
            f"their key template (A16 item 4). Without it a lookup key's shape cannot be "
            f"validated, and a wrong-shaped lookup returns an empty result that reads as "
            f"'nothing needed' — the mechanism behind all three key-shape incidents.")
    return obj["keying"]


def assert_key_shape(keys, lookup_key: str, where: str, keying: str | None = None,
                     sample: int = 200) -> dict:
    """Validate a lookup key's SHAPE against the store, BEFORE any result is interpreted.

    Two independent authorities, and both must agree with the lookup:
      * the store's self-declaration (`keying`), when the artifact carries one — A16 item 4;
      * a sample of the store's OWN keys, which is the ground truth the declaration claims
        to describe (a stale declaration is itself a defect worth catching).

    Returns the evidence record so a caller can archive it; raises on any mismatch.
    """
    keys = list(keys)
    if not keys:
        raise AssertionError(f"{where}: the keyed store is EMPTY — every lookup against it "
                             f"would return 'absent', which is instrument failure, not data")
    want = key_shape_signature(lookup_key)
    seen = sorted({key_shape_signature(k) for k in keys[:sample]})
    rec = {"where": where, "lookup_key": lookup_key, "lookup_signature": want,
           "store_key_signatures_sampled": seen, "n_keys": len(keys),
           "declared_keying": keying}
    if want not in seen:
        raise AssertionError(
            f"{where}: KEY-SHAPE MISMATCH — the lookup key {lookup_key!r} has shape {want!r} "
            f"but the store's own keys have shape(s) {seen} (sampled {min(sample, len(keys))} "
            f"of {len(keys)}). Every lookup would return 'absent' and the empty result would "
            f"read as 'nothing needed'. This is the A16 guard; fix the key shape, do not "
            f"interpret the result.")
    if keying is not None:
        decl = declared_key_signature(keying)
        rec["declared_signature"] = decl
        if decl != want:
            raise AssertionError(
                f"{where}: the store DECLARES key template shape {decl!r} ({keying!r}) but "
                f"the lookup key {lookup_key!r} has shape {want!r} — validate the key shape "
                f"before interpreting any result (A16 item 4)")
        if decl not in seen:
            raise AssertionError(
                f"{where}: the store's `keying` declaration ({decl!r}) disagrees with the "
                f"store's own keys ({seen}) — a stale self-declaration is a defect: it would "
                f"certify a wrong-shaped lookup as correct")
    return rec


def assert_join_nonvacuous(name: str, left, right, expect_min: int = 1,
                           left_label: str = "left", right_label: str = "right") -> dict:
    """A cross-source join that matches NOTHING is a FAILURE, never information (A16 item 1).

    Reports the two sides' key SHAPES on failure, because a vacuous join between two
    non-empty key sets is almost always a key-shape mismatch — incident 1, exactly.
    """
    L, R = set(left), set(right)
    inter = L & R
    rec = {"join": name, left_label: len(L), right_label: len(R),
           "intersection": len(inter), "expect_min": expect_min}
    if len(inter) < expect_min:
        ls = sorted({key_shape_signature(k) for k in list(L)[:200]})
        rs = sorted({key_shape_signature(k) for k in list(R)[:200]})
        raise AssertionError(
            f"[join {name}] VACUOUS JOIN: {len(L)} {left_label} keys x {len(R)} "
            f"{right_label} keys intersect in {len(inter)} (need >= {expect_min}). An empty "
            f"join result is a failure, never information (A16 item 1). {left_label} key "
            f"shapes {ls}; {right_label} key shapes {rs}"
            + ("  <-- the two sides are keyed DIFFERENTLY" if set(ls) != set(rs) else ""))
    return rec


class KeyedStore:
    """A keyed artifact whose accessors RAISE on absent keys.  There is no `.get()`.

    Wraps a `{key: value}` mapping and enforces, on the first lookup, that the lookup key's
    shape matches the store's own keys and the store's `keying` self-declaration.  The only
    sanctioned optional-presence query is `has()`, which validates key shape first — so a
    wrong-shaped probe can never answer "absent".  `require()` raises `KeyError` with the
    key shapes in the message; the caller decides whether that is a `SystemExit`.
    """

    __slots__ = ("_d", "name", "keying", "_checked")

    def __init__(self, mapping: dict, name: str, keying: str | None = None):
        if not isinstance(mapping, dict):
            raise AssertionError(f"{name}: a keyed store must be a mapping, got "
                                 f"{type(mapping).__name__}")
        self._d = mapping
        self.name = name
        self.keying = keying
        self._checked: set[str] = set()

    def __len__(self) -> int:
        return len(self._d)

    def __contains__(self, key) -> bool:      # shape-checked; see has()
        return self.has(key)

    def keys(self):
        return self._d.keys()

    def _check(self, key: str) -> None:
        sig = key_shape_signature(key)
        if sig in self._checked:
            return
        assert_key_shape(list(self._d), key, where=self.name, keying=self.keying)
        self._checked.add(sig)

    def has(self, key: str) -> bool:
        """Shape-validated presence test — the ONLY sanctioned optional-presence query."""
        self._check(key)
        return key in self._d

    def require(self, key: str):
        """The raising accessor.  An absent key is an exception, never a `None`."""
        self._check(key)
        if key not in self._d:
            raise KeyError(
                f"{self.name}: no entry for key {key!r} (shape {key_shape_signature(key)}; "
                f"store holds {len(self._d)} keys of shape(s) "
                f"{sorted({key_shape_signature(k) for k in list(self._d)[:200]})}). An "
                f"absent key is an exception, never a fallback (A16 item 1).")
        return self._d[key]

    def __getitem__(self, key):
        return self.require(key)

    def join_nonvacuous(self, wanted, name: str | None = None, expect_min: int = 1) -> dict:
        """Assert a batch of lookup keys actually intersects this store."""
        return assert_join_nonvacuous(name or f"{self.name}", wanted, self._d.keys(),
                                      expect_min=expect_min,
                                      left_label="wanted", right_label="store")


def load_keyed_store(path: str | Path, payload_key: str | None = None,
                     name: str | None = None, keying: str | None = None) -> KeyedStore:
    """Load a keyed store artifact.  `keying` is MANDATORY — declared or passed in.

    `payload_key` names the sub-object holding the entries (`"descriptions"` for
    `CAPTION_STORE.json`); omit it for a flat `{key: value}` file, which must then be
    accompanied by an explicit `keying=` (a flat file has nowhere to declare it).
    """
    p = Path(path)
    obj = read_json(p)
    label = name or p.name
    if payload_key is not None:
        keying = keying or require_keying_declaration(obj, str(p))
        payload = obj[payload_key]
    else:
        payload = obj
        if isinstance(obj, dict) and obj.get("keying"):
            keying = keying or obj["keying"]
        if not keying:
            raise AssertionError(
                f"{p}: a flat keyed store carries no `keying` declaration, so one must be "
                f"passed in (A16 item 4 — key shape is validated before interpretation)")
    return KeyedStore(payload, name=label, keying=keying)


#: A10 (2026-07-28) — the standing role-scoped exclusion, DERIVED from the sidecar so the
#: two can never drift.  Authority: `data/processed/ctt_v2_strata/
#: POOL_DROPS_M3_ADJUDICATION.json:role_scoped_exclusions` (verdict "ROLE-SCOPED EXCLUSION.
#: Whole-clip drop DENIED as written. Confidence 0.93.").
#:
#: `openvid_T1MiFx98l3g_0_50to156` has a blank-white A-anchor (ffprobe YMIN=YMAX=YAVG=232)
#: and a healthy B-anchor (Y 12-242).  The pool file and the endpoint frame cache stay
#: BYTE-UNCHANGED — a completed render and the running encodes are keyed to them — which is
#: exactly why the exclusion has to live in code that every consumer reads.
#:
#: 🔴 A16 CORRECTION to A10's occupancy claim.  A10 recorded "field B in 37/37 rendered rows,
#: byte-pure on every one, so the blank window is structurally unreachable in the role it
#: occupies."  True — for S2b, the only stratum it scanned.  The enumerated universe (A16 §Q3,
#: universal join run first-hand):  S2a 7,990 rows -> **29 field A**, 0 field B;
#: S2b 7,990 rows -> 0 field A, 37 field B;  S1 390 rows -> 0.  So the clip IS consumed in
#: its excluded role, and those 29 S2a rows are DROPPED at consumption (A16), derived from
#: this constant at build time and recorded in the root manifest's drop record.  The
#: exclusion itself stands unchanged; only the belief "it never appears in role A" was wrong.
#:
#: A10's standing rule, campaign-wide: **defects are dispositioned at the unit of
#: CONSUMPTION — (clip, role) — not the unit of storage.**  Whole-clip drops are reserved
#: for role-independent defects (leakage, duplication, provenance) or clips not yet
#: consumed anywhere.  `enforced_at` in the sidecar enumerates the consumption channels;
#: a recorded exclusion that no code reads is a landmine.
try:
    _role_excl, _clip_excl, ROLE_EXCLUSIONS_PROVENANCE = load_caption_store_exclusions()
    ROLE_EXCLUSIONS: dict = {c: tuple(sorted(r)) for c, r in _role_excl.items()}
except OSError as _exc:  # pragma: no cover — the sidecar is on /projects, not /tmp
    ROLE_EXCLUSIONS = {}
    ROLE_EXCLUSIONS_PROVENANCE = {"file": str(POOL_DROPS), "error": f"OSError: {_exc}"}


def require_role_exclusions(where: str = "") -> dict:
    """A VACUOUS standing exclusion is a failure, never "nothing to exclude" (A16 item 1).

    `data/processed/` is gitignored, so the adjudication sidecar travels with the working
    tree and not with the branch.  Found the hard way: on the freshly-consolidated `main`
    checkout `POOL_DROPS_M3_ADJUDICATION.json` was simply ABSENT (it existed only in the
    worktree), which made `ROLE_EXCLUSIONS` an empty dict — every `role_excluded()` call
    answered False, the A10 exclusion was silently vacuous, and the A16 drop would have
    dropped 0 of the 29 while every assert reported PASS.  That is precisely the
    `INTENDED_WEIGHTS_PCT` landmine class, and precisely the failure direction A16 forbids:
    an empty result read as information.  So any consumer of the exclusion calls this first.
    """
    prov = ROLE_EXCLUSIONS_PROVENANCE or {}
    if prov.get("error") or not ROLE_EXCLUSIONS:
        raise SystemExit(
            f"[role-exclusions] {where or 'this consumer'} depends on the standing A10 "
            f"role-scoped exclusion, and it is VACUOUS: {prov.get('error') or 'no entries'} "
            f"({prov.get('file', POOL_DROPS)}). An empty exclusion is instrument failure, not "
            f"'nothing to exclude' — every role_excluded() call would answer False and the "
            f"A16 drop would silently drop nothing. `data/processed/` is gitignored, so check "
            f"that the sidecar is present in THIS checkout.")
    return prov


def role_excluded(clip: str, role: str) -> bool:
    """Is this (clip, role) — this unit of CONSUMPTION — excluded? (A10 standing rule.)"""
    return role in ROLE_EXCLUSIONS.get(clip, ())


def caption_lookup(store: dict, clip: str, role: str, key_fmt: str = "{clip}|{role}") -> str:
    """(clip, role)-keyed caption-store lookup.  Hard-fail; NO cross-role fallback.

    A10, verbatim intent: a request for a role-excluded clip's A-caption means the clip
    leaked into A-role somewhere upstream, and that **must crash, not silently degrade** —
    a cross-role fallback would substitute the healthy B description and produce a caption
    that describes the wrong nine frames, with nothing in any log to show it happened.
    """
    if role_excluded(clip, role):
        raise SystemExit(
            f"[caption] REFUSING to look up ({clip!r}, role {role!r}): that (clip, role) is "
            f"role-excluded by A10 ({POOL_DROPS.name}). A request for it means the clip "
            f"leaked into role {role} upstream — fix the caller, do not fall back.")
    key = key_fmt.format(clip=clip, role=role)
    # A16 item 4: the lookup key's SHAPE is validated against the store's own keys (and its
    # `keying` declaration when it carries one) BEFORE the result is interpreted.  Incident 3
    # was `descriptions.get(clip)` against a `clip|role`-keyed store: `None` for both roles,
    # briefly read as "the B-role is missing too".  A shape check makes that impossible.
    ks = store if isinstance(store, KeyedStore) else KeyedStore(
        store, name=f"caption store (key template {key_fmt!r})", keying=f"'{key_fmt}'")
    try:
        return ks.require(key)
    except KeyError as exc:
        raise SystemExit(f"[caption] {exc.args[0]} ((clip, role) = ({clip!r}, {role!r})); "
                         f"no cross-role fallback exists") from None


def caption_sources(entry: dict, sided: str, kind: str) -> list[tuple[str, str]]:
    """The (clip, role) descriptions an ASSEMBLED caption draws on.

    An explicit `caption_sources` on the inventory entry always wins.  Otherwise it is
    derived from the S2/S1 render contract (DOSSIER §13.1, verified against `render_s2.py`):
    `build_from_stream(start9)` puts the A endpoint's FIRST 9 frames at the head, so the A
    endpoint supplies the **role-A** description; `build_to_stream(end9)` puts the B
    endpoint's LAST 9 frames at the tail, so the B endpoint supplies the **role-B**
    description.  `swap` inverts the shader progress argument only — it never exchanges the
    A/B content assignment, which is why role scoping is meaningful at all.

    A one-sided clip's caption has no suffix sentence (`{S1}. sksz.`), so it draws on no
    role-B description.  `kind == "corpus"` (S0) draws on the 139 certified corpus captions,
    not on the per-(clip, role) store at all.
    """
    if entry.get("caption_sources") is not None:
        return [(c, r) for c, r in entry["caption_sources"]]
    if kind == "corpus":
        return []
    eps = list(entry.get("endpoints") or [])
    if not eps:
        return []
    out = [(eps[0], "A")]
    if sided == "two" and len(eps) > 1:
        out.append((eps[1], "B"))
    return out


def zs_classes() -> set:
    """The 10 zero-shot held-out classes — from the frozen split, never a literal list."""
    return set(read_json(SPLIT_PATH)["generalist_holdout"])


def test_clips() -> set:
    """The 42 pre-registered test clips — from the frozen split."""
    split = read_json(SPLIT_PATH)["classes"]
    return {c for e in split.values() for c in e.get("test", [])}


def _davis_pool_ids() -> set:
    """DAVIS *sequences* used by the eval foreign roster, mapped to union-pool clip ids.

    `eval_ladder/davis.yaml` names sequences (`bear`, `mallard-water`); the union content
    pool names the same footage `davis_<sequence>`.  Comparing the composite pseudo-clip
    names (`davis_bear_elephant`) against pool ids would silently miss every real overlap.
    """
    import yaml  # noqa: PLC0415

    cfg = yaml.safe_load(DAVIS_YAML.read_text())
    seqs = set()
    for e in (cfg.get("one_sided") or {}).values():
        seqs.add(e["sequence"])
    for e in (cfg.get("two_sided") or {}).values():
        seqs.add(e["prefix"]["sequence"])
        seqs.add(e["suffix"]["sequence"])
    return {f"davis_{s}" for s in seqs}


def eval_endpoint_universe() -> tuple[set, dict]:
    """{eval endpoints} u {zs audited endpoints} u {the 42 test clips}  (A3-F5b).

    Class membership is resolved with `eval_ladder/prompts.py:clip_class()` — never by
    string-splitting a clip name (`action_run_setonfire_6` is class `run_set_on_fire`).
    """
    prompts = _prompts()
    zs = zs_classes()

    reg_endpoints, reg_refs = set(), set()
    for line in REGISTRY.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("endpoint"):
            reg_endpoints.add(row["endpoint"])
        if row.get("reference"):
            reg_refs.add(row["reference"])

    davis_ids = _davis_pool_ids()

    zs_audited = set()
    for clip in prompts.audited_clips():
        try:
            cls = prompts.clip_class(clip)
        except KeyError:
            continue  # davis pseudo-clips are handled through `davis_ids`
        if cls in zs:
            zs_audited.add(clip)

    tests = test_clips()

    universe = reg_endpoints | reg_refs | davis_ids | zs_audited | tests
    prov = {
        "registry": str(REGISTRY.relative_to(REPO_ROOT)),
        "n_registry_endpoints": len(reg_endpoints),
        "n_registry_references": len(reg_refs),
        "davis_eval_sequences_as_pool_ids": sorted(davis_ids),
        "n_zs_audited_endpoints": len(zs_audited),
        "zs_audited_endpoints": sorted(zs_audited),
        "n_test_clips": len(tests),
        "class_resolution": "eval_ladder/prompts.py:clip_class()",
    }
    return universe, prov


def load_exclusions(prereg_inline_ood: Path | None = None,
                    pool_drops: Path | None = None) -> Exclusions:
    holdout = read_json(HOLDOUT_S2)
    pool = read_json(CONTENT_POOL)
    universe, prov = eval_endpoint_universe()
    role_caps, clip_caps, cap_prov = load_caption_store_exclusions(pool_drops)
    ex = Exclusions(
        holdout_shaders=set(holdout["holdout_shaders"]),
        inline_ood_ops=set(),
        reserved_pool_clips={r["clip_id"] for r in pool["reserved"]},
        zs_classes=zs_classes(),
        eval_endpoints=universe,
        role_scoped_captions=role_caps,
        clip_level_captions=clip_caps,
        provenance={
            "holdout_shaders": str(HOLDOUT_S2.relative_to(REPO_ROOT)),
            "reserved_pool_clips": str(CONTENT_POOL.relative_to(REPO_ROOT)),
            "zs_classes": str(SPLIT_PATH.relative_to(REPO_ROOT)),
            "eval_endpoints": prov,
            "caption_store_exclusions": cap_prov,
        },
    )
    path = Path(prereg_inline_ood) if prereg_inline_ood else PREREG_INLINE_OOD
    if path.exists():
        rec = read_json(path)
        ex.inline_ood_ops = set(rec["op_ids"])
        ex.provenance["inline_ood_ops"] = {"file": str(path), "status": rec.get("status")}
    return ex


def select_inline_ood_ops(groups: dict, exclusions: Exclusions, n: int = N_INLINE_OOD_OPS,
                          seed: int = SEED) -> dict:
    """RULING 4 — n ops from n DISTINCT shaders, RNG seed 42, drawn from the otherwise
    trainable S2a ops.  Deterministic: same inputs => same draw, forever.
    """
    import random  # noqa: PLC0415

    eligible: dict[str, list[str]] = {}
    for gid, g in sorted(groups.items()):
        shader = g.get("shader")
        if shader is None or shader in exclusions.holdout_shaders:
            continue
        eligible.setdefault(shader, []).append(gid)
    shaders = sorted(eligible)
    if len(shaders) < n:
        raise AssertionError(f"only {len(shaders)} eligible shaders, need {n} distinct")
    rng = random.Random(seed)
    chosen_shaders = sorted(rng.sample(shaders, n))
    ops = {}
    for sh in chosen_shaders:
        ops[sh] = rng.choice(sorted(eligible[sh]))
    return {
        "authority": "A5 RULING 4 — 8 pre-registered S2a inline-OOD ops (8 distinct shaders, seed 42)",
        "seed": seed,
        "n": n,
        "selector": "root_common.select_inline_ood_ops",
        "status": "operator-derived, awaiting owner ratification",
        "shader_to_op": ops,
        "op_ids": sorted(ops.values()),
        "eligible_shaders": shaders,
        "eligible_op_count": sum(len(v) for v in eligible.values()),
    }


# --------------------------------------------------------------------------------------
# A11 item 1 — freeze the inline-OOD draw as a PRE-REGISTRATION artefact
# --------------------------------------------------------------------------------------
#: A11 item 1, verbatim.  The declaration is the whole evidentiary content of the file: it
#: states WHEN the draw happened relative to the only two events that could contaminate it.
PREREG_TIMING_DECLARATION = (
    "written after S2a data existed, before any training step and before any candidate was "
    "scored; selection is a blind seed-42 draw over the sorted op list, referencing no "
    "measured property of any op."
)
#: A11 item 1, verbatim.  Assembly does NOT wait for the countersign.
PREREG_STATUS = ("advisor-ratified 2026-07-28; owner countersign folded into the stamp "
                 "sign-off batch (DATASET §13.3 item 8)")
#: A11 item 1 — the two OOD tiers are complementary, and confusing them is the live risk.
PREREG_AUTHORITY = (
    "A5 RULING 4 (8 pre-registered S2a inline-OOD ops, 8 distinct shaders, seed 42), "
    "ratified by A11 item 1 (2026-07-28). RELATION TO THE 10 HELD-OUT SHADER FAMILIES "
    "(H2): complementary tiers, NOT overlap. H2 is a FAMILY-level holdout with ZERO "
    "rendered clips — it cannot supply inline demos without new rendering and it stays "
    "eval-side. These 8 are OP-level near-OOD (a novel op drawn from a shader family the "
    "model DOES train on), with clips already rendered and encoded. The 8 ops are EXCLUDED "
    "from the assembled root — not merely held — while their encodes stay on disk for the "
    "inline lane's diagnostic demos. Per A2 the inline scores never gate anything. "
    "S2a-ONLY by ruling: S2b's operators are all-new, so no S2b op shares an excluded "
    "op's id and the excluded ops remain globally unseen at op level. The draw is "
    "RATIFIED AS-IS and is NEVER post-filtered (e.g. to avoid full-occlusion shaders) — "
    "post-draw curation is exactly the cherry-picking the seed-42 procedure precludes."
)


def freeze_inline_ood_prereg(inventory_path: str | Path, exclusions: Exclusions,
                             out_path: str | Path | None = None,
                             when: str | None = None) -> dict:
    """Derive the 8 inline-OOD ops from a FROZEN S2a inventory and write the artefact.

    Everything in the written record is derived from the inventory on disk — the 8 op ids,
    the 80 clip ids and the inventory's own sha256 — so the pre-registration can be
    re-verified against the exact bytes it was drawn from, forever.  Re-running is a no-op
    unless the inventory changed, in which case the sha256 in the file stops matching and
    assert A6 says so instead of silently re-drawing.
    """
    import time  # noqa: PLC0415

    inv_path = Path(inventory_path)
    inv = read_json(inv_path)
    if inv.get("stratum") != "S2a":
        raise ValueError(f"{inv_path}: the inline-OOD pre-registration is S2a-only, "
                         f"got stratum {inv.get('stratum')!r}")
    rec = select_inline_ood_ops(inv["groups"], exclusions)
    clips = sorted(c for op in rec["op_ids"] for c in inv["groups"][op]["clips"])
    rec["authority"] = PREREG_AUTHORITY
    rec["status"] = PREREG_STATUS
    rec["timing_declaration"] = PREREG_TIMING_DECLARATION
    rec["written"] = when or time.strftime("%Y-%m-%d")
    rec["source_inventory"] = str(inv_path)
    rec["source_inventory_sha256"] = sha256_file(inv_path)
    rec["source_inventory_counts"] = dict(inv.get("counts") or {})
    rec["n_clips"] = len(clips)
    rec["clip_ids"] = clips
    if out_path is not None:
        write_json(out_path, rec)
    return rec


# --------------------------------------------------------------------------------------
# A12 — freeze the INPUTS of the derived S2a:S2b split
#
# The registered object is the RULE, not a number: "S2 total 69, split pro-rata to the
# assembled post-exclusion base pair counts".  A derived split is a feature rather than a
# hazard ONLY if its inputs are frozen before training, so that no experimenter degree of
# freedom survives.  This record is that freeze: every exclusion list that determines the
# assembled counts, by path and sha256; the counts themselves; the split those counts
# produce; and the amendment rule that makes any later change to the inputs visible.
# --------------------------------------------------------------------------------------
MIX_INPUTS_SCHEMA = "ctt_v2_prereg_mix_inputs/1"

#: A12, verbatim intent.  The counts may move — exclusions can legitimately change — but
#: only through a logged amendment, never silently.
MIX_AMENDMENT_RULE = (
    "ANY change to any input listed under `exclusion_inputs` or `inventories` — a new "
    "exclusion, a withdrawn one, a re-rendered or re-audited stratum, a redrawn "
    "inline-OOD pre-registration — CHANGES THE ASSEMBLED COUNTS AND THEREFORE THE DERIVED "
    "S2a:S2b SPLIT. When that happens: (1) re-run "
    "`assemble_root.py --plan-only --write-prereg-mix-inputs`, which recomputes the split "
    "from the new counts; (2) log the recomputation as a DOSSIER AMENDMENT recording the "
    "old and new counts, the old and new derived shares, the multipliers and max_dev, and "
    "the input whose sha256 moved; (3) only then assemble. The counts in this file may "
    "move ONLY via such a logged amendment — never silently. The stratum-level weights "
    "(S0 15 / S1 6 / S2 total 69 / S4 10, and the pre-registered contingency branches) are "
    "NOT amendable this way: they are a ruling, and changing them requires a new ruling."
)

MIX_PREREG_TIMING_DECLARATION = (
    "written from a --plan-only assembly, before any training step, before any candidate "
    "was scored, and before the root was materialised; the split is a deterministic "
    "function of the recorded inputs and references no measured property of any sample."
)


def freeze_mix_inputs_prereg(*, strata_manifest_path: str | Path, manifest: dict,
                             inventories: dict, exclusions: "Exclusions", drops: dict,
                             base_counts: dict, mix: dict, weight_note: str,
                             branch_key: str | None, present: list, absent: list,
                             out_path: str | Path | None = None,
                             when: str | None = None) -> dict:
    """Write the A12 frozen-inputs record for the DERIVED S2a:S2b split.

    Everything here is derived from the same `--plan-only` computation that produced the
    split, so the file cannot describe a different assembly than the one it certifies.
    """
    import time  # noqa: PLC0415
    from collections import Counter  # noqa: PLC0415

    def _src(path, **extra) -> dict:
        p = Path(path)
        rec = {"path": str(p), "exists": p.exists()}
        if p.exists():
            rec["sha256"] = sha256_file(p)
            rec["bytes"] = p.stat().st_size
        rec.update(extra)
        return rec

    # every reason that removed a clip or a group, counted — this is the DERIVATION of the
    # per-stratum drop totals (e.g. S2b's 131), not a restatement of them
    drop_reasons = {}
    for s, d in sorted(drops.items()):
        cr = Counter(r.split(":", 1)[0] for c in d["dropped_clips"] for r in c["reasons"])
        gr = Counter(r.split(":", 1)[0] for g in d["dropped_groups"] for r in g["reasons"])
        drop_reasons[s] = {
            "n_groups_dropped": len(d["dropped_groups"]),
            "n_clips_dropped": len(d["dropped_clips"]),
            "clip_drop_reasons": dict(sorted(cr.items())),
            "group_drop_reasons": dict(sorted(gr.items())),
        }

    ex_rec = exclusions.as_record()
    prereg_ood = (exclusions.provenance.get("inline_ood_ops") or {}).get("file")

    rec = {
        "schema": MIX_INPUTS_SCHEMA,
        "authority": "A12 (2026-07-28) — misc/ctt_v2_final/advisors/"
                     "A12_prorata_s2_split_VERBATIM.md. VERDICT (a): the S2a:S2b split is "
                     "derived pro-rata from the post-exclusion assembled counts. Builds on "
                     "A9 §4 (S2 total 69, split pro-rata), A11 item 3 (34.5 is per S2 half "
                     "=> S2 total 69) and A1b (uniform per-sample weight within S2, no "
                     "extra reweighting knob).",
        "written": when or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "timing_declaration": MIX_PREREG_TIMING_DECLARATION,
        "amendment_rule": MIX_AMENDMENT_RULE,

        # ---- (0) the contract ---------------------------------------------------------
        "mix_contract": {
            "statement": "S0 15 / S1 6 / S2 total 69 / S4 10; the S2a:S2b split is DERIVED "
                         "pro-rata from the assembled post-exclusion base pair counts.",
            "stratum_weights_pct": dict(STRATUM_WEIGHTS_PCT),
            "prorata_groups": {k: list(v) for k, v in PRORATA_GROUPS.items()},
            "contingency_branches_pct": {k: dict(v)
                                         for k, v in ABSENT_BRANCH_WEIGHTS_PCT.items()},
            "contingency_note": "S2 total 73 / 79 / 85, each split pro-rata exactly as the "
                                "headline. S1 and S4 weights stay fixed numbers — they are "
                                "independent strata, not a pro-rata pair.",
            "absent_policy": ABSENT_POLICY,
            "tolerance_pp": MIX_TOLERANCE_PP,
            "implementation": "root_common.expand_prorata_weights + "
                              "root_common.solve_multipliers (a pro-rata group is solved "
                              "as one unit, so its members share one multiplier); "
                              "re-checked off the root by assert_root A3 + A3b.",
        },

        # ---- (i) the inputs that determine the assembled counts ------------------------
        "strata_manifest": _src(strata_manifest_path, strata_present=list(present),
                                strata_absent=list(absent), weight_note=weight_note,
                                branch_override_key=branch_key),
        "inventories": {
            s: _src(manifest["strata"][s]["inventory"],
                    kind=inventories[s].get("kind"),
                    n_groups=len(inventories[s]["groups"]),
                    n_clips=len(inventories[s]["clips"]),
                    endpoint_disjointness=inventories[s].get("endpoint_disjointness", True))
            for s in sorted(present)
        },
        "exclusion_inputs": {
            "role_scoped_pool_drops": _src(
                POOL_DROPS,
                derivation="root_common.load_caption_store_exclusions() reads "
                           "`role_scoped_exclusions` (A10, authoritative) and the legacy "
                           "`role_scoped_exclusions_for_caption_store`, and HARD-FAILS if "
                           "they disagree; an absent file is a hard failure, never an "
                           "empty exclusion",
                role_scoped=ex_rec["role_scoped_caption_store_exclusions"],
                clip_level=ex_rec["clip_level_caption_store_exclusions"],
                provenance=exclusions.provenance.get("caption_store_exclusions")),
            "inline_ood_prereg": _src(
                prereg_ood or PREREG_INLINE_OOD,
                derivation="seed-42 blind draw over the sorted S2a op list, 8 ops from 8 "
                           "DISTINCT shader families (root_common.select_inline_ood_ops); "
                           "8 ops x 10 clips = 80 clips excluded from the root",
                op_ids=ex_rec["inline_ood_ops"]),
            "holdout_s2_shader_families": _src(
                HOLDOUT_S2, n=len(ex_rec["holdout_shaders"]),
                shaders=ex_rec["holdout_shaders"]),
            "reserved_pool_clips": _src(
                CONTENT_POOL, n=ex_rec["n_reserved_pool_clips"],
                derivation="the `reserved` block of the union content pool"),
            "zs_classes_and_test_clips": _src(
                SPLIT_PATH, zs_classes=ex_rec["zs_classes"],
                derivation="`generalist_holdout` (10 zs classes) + the 42 pre-registered "
                           "test clips from `classes[*].test`"),
            "eval_endpoint_universe": {
                "n": ex_rec["n_eval_endpoints"],
                "sha256_of_sorted_ids": sha256_obj(ex_rec["eval_endpoints"]),
                "provenance": exclusions.provenance.get("eval_endpoints"),
                "registry": _src(REGISTRY),
                "davis": _src(DAVIS_YAML),
            },
        },
        "drops_that_produced_the_counts": drop_reasons,

        # ---- (ii) the frozen counts ----------------------------------------------------
        "frozen_assembled_base_pair_counts": {s: int(base_counts[s])
                                              for s in sorted(base_counts)},
        "frozen_counts_note": "post-exclusion base PAIRS (ring offset within group, "
                              "k = min(3, n-1)) — the unit the mix is measured in, not "
                              "clips. These are the numbers the split is pro-rata to.",

        # ---- (iii) the derived split, shares, multipliers, deviation --------------------
        "derived_split": {
            "weight_note": weight_note,
            "branch_override_key": branch_key,
            "stratum_weights_pct_applied": mix["stratum_weights_pct"],
            "stratum_weights_pct_renormalized": mix["stratum_weights_pct_renormalized"],
            "prorata_split": mix["prorata_split"],
            "intended_pct": mix["intended_pct"],
            "multipliers": mix["multipliers"],
            "aggregate_multipliers": mix["aggregate_multipliers"],
            "realized_counts": mix["realized_counts"],
            "realized_pct": mix["realized_pct"],
            "deviation_pp": mix["deviation_pp"],
            "max_deviation_pp": mix["max_deviation_pp"],
            "tolerance_pp": mix["tolerance_pp"],
            "total_base_pairs": mix["total"],
            "total_files": mix["total"] * len(ROOT_DIRS),
        },
    }
    if out_path is not None:
        write_json(out_path, rec)
    return rec


# --------------------------------------------------------------------------------------
# Tier-1 caption leak filter (import the caption pipeline's if present, else reimplement)
# --------------------------------------------------------------------------------------
def _word_re(s: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(s) + r"(?![A-Za-z0-9_])", re.I)


#: ordinary English words that are also shader basenames (A4 Q3 (iv) routes these to the
#: Tier-2 REVIEW lane, not Tier-1 HARD).  Kept byte-identical to
#: `scripts/ctt_v2/captions/caption_common.py:SHADER_ESCAPE` so the fallback matches the
#: real filter; the real filter is preferred whenever it is importable.
_SHADER_ESCAPE = {
    "angular", "bounce", "box", "burn", "burn0", "chessboard", "circle", "crosshatch",
    "cube", "directional", "doorway", "dreamy", "fade", "fold", "fragment", "heart",
    "mosaic", "perlin", "pinwheel", "polar", "radial", "rectangle", "ripple", "rolls",
    "slides", "squeeze", "static", "swap", "swirl", "wind",
}
_MECHANISM_PATTERNS = [
    r"\btransform\w*\b", r"\btransition\w*\b", r"\bmorph\w*\b", r"\bdissolv\w*\b",
    r"\bmetamorpho\w*\b", r"\bteleport\w*\b", r"\bturns? into\b", r"\bchanges? into\b",
    r"\bbecomes? a\b", r"\bvfx\b", r"\bcgi\b", r"\bshader\b", r"\boverlay\w*\b",
    r"\bglitch\w*\b",
]


class _FallbackLeakFilter:
    """Tier-1 only, used when `scripts/ctt_v2/captions/caption_common.py` is unavailable.

    Shader basenames from `misc/gl-transitions/transitions/*.glsl` and the 39 class names
    from `corpus_manifest.json`, snake_case AND de-underscored, plus the A4 mechanism
    words.  The refVFX trigger lexicon is *not* reproducible here (it needs the caption
    pipeline's index); `source` reports which filter ran so the gap is never silent.
    """

    source = "root_common._FallbackLeakFilter"

    def __init__(self):
        self.shaders = sorted({p.stem for p in SHADER_DIR.glob("*.glsl")})
        self.classes = sorted(read_json(CORPUS_MANIFEST)["classes"])
        pats = []
        for s in self.shaders:
            if s.lower() in _SHADER_ESCAPE:
                continue
            pats.append(("shader", s, _word_re(s)))
        for c in self.classes:
            if "_" in c:
                pats.append(("class", c, _word_re(c)))
                pats.append(("class", c.replace("_", " "), _word_re(c.replace("_", " "))))
        self._pats = pats
        self._mech = [re.compile(p, re.I) for p in _MECHANISM_PATTERNS]

    def tier1(self, text: str) -> list[str]:
        hits = []
        for kind, s, rx in self._pats:
            if rx.search(text):
                hits.append(f"{kind}:{s}")
        for rx in self._mech:
            m = rx.search(text)
            if m:
                hits.append(f"mechanism:{m.group(0).lower()}")
        if OUTCOME_MARKER.lower() in text.lower():
            hits.append("outcome_marker")
        return sorted(set(hits))


def leak_filter():
    """Prefer the caption pipeline's own filter; fall back to the reimplementation."""
    cap = HERE / "captions"
    if (cap / "caption_common.py").exists():
        if str(cap) not in sys.path:
            sys.path.insert(0, str(cap))
        try:
            import caption_common  # noqa: PLC0415

            f = caption_common.LeakFilter()
            f.source = "scripts/ctt_v2/captions/caption_common.py:LeakFilter"
            return f
        except Exception as exc:  # noqa: BLE001 - never silently degrade
            print(f"[leak-filter] caption_common import failed ({exc!r}); using fallback",
                  file=sys.stderr)
    return _FallbackLeakFilter()


def caption_violations(caption: str, filt) -> list[str]:
    """RULING 9 caption asserts, applied to an ASSEMBLED training caption.

    * exactly one ` sksz.`
    * outcome marker absent
    * zero Tier-1 leak strings

    The trigger sentence is removed before the Tier-1 pass: the trigger is a leak string
    for a stored *description*, but it is mandatory in an assembled caption.
    """
    v = []
    n = caption.count(TRIGGER_SENTENCE)
    if n != 1:
        v.append(f"trigger_count:{n}")
    if OUTCOME_MARKER.lower() in caption.lower():
        v.append("outcome_marker")
    stripped = caption.replace(TRIGGER_SENTENCE, " ")
    for hit in filt.tier1(stripped):
        if hit == "sksz":
            continue
        v.append(f"tier1:{hit}")
    return v


# --------------------------------------------------------------------------------------
# copy-gate admissibility (RULING 1, training blocker)
# --------------------------------------------------------------------------------------
_VERDICT_RE = re.compile(r"^\s*\**\s*(?:admissibility\s*)?verdict\s*\**\s*[:=]\s*(.+)$",
                         re.I | re.M)


def copy_gate_verdict(path: Path | None = None) -> tuple[bool, str]:
    """Read the Day-0 copy-discriminator admissibility verdict.

    A5: the admissibility check BLOCKS training.  Default is FAIL — an absent file, an
    unreadable verdict, or anything that is not an unambiguous PASS fails the assert.
    A machine-readable `<stem>.json` sidecar with {"admissibility": "PASS"} wins if present.
    """
    p = Path(path) if path else COPY_GATE_VERDICT
    sidecar = p.with_suffix(".json")
    if sidecar.exists():
        try:
            rec = read_json(sidecar)
        except Exception as exc:  # noqa: BLE001
            return False, f"sidecar {sidecar} unreadable: {exc!r}"
        val = str(rec.get("admissibility", rec.get("verdict", ""))).strip()
        ok = val.upper() == "PASS"
        return ok, f"sidecar {sidecar}: admissibility={val!r}"
    if not p.exists():
        return False, (f"{p} is ABSENT — A5 RULING 1 makes the Day-0 copy-discriminator "
                       f"admissibility check a training blocker")
    text = p.read_text()
    m = _VERDICT_RE.search(text)
    if not m:
        return False, f"{p} carries no parseable 'Verdict:' line"
    verdict = m.group(1).strip()
    head = verdict.upper()
    negative = any(t in head for t in ("FAIL", "NOT PASS", "NO PASS", "STOP", "ESCALATE"))
    positive = re.search(r"(?<![A-Z])PASS(?:ED|ES)?(?![A-Z])", head) is not None
    ok = positive and not negative
    return ok, f"{p}: verdict={verdict[:160]!r}"


# --------------------------------------------------------------------------------------
# root path scheme
# --------------------------------------------------------------------------------------
#: `\d{2,}` and NOT `\d{2}`.  `replica_dir` pads to a MINIMUM of two digits, so a multiplier
#: above 100 emits three: S0's realized multiplier is 153, giving `S0_r00 .. S0_r99` followed
#: by `S0_r100 .. S0_r152`.  Those names are correct and unambiguous (the stratum part cannot
#: contain `_`, so `S0_r100` can only read as rep 100), but the two-digit regex could not parse
#: them and `assert_root` CRASHED — not failed, crashed — on the first S0 replica past 99:
#:     ValueError: relative path does not start with a <stratum>_r<NN> dir:
#:                 'S0_r100/air_bending/air_bending_0__ref_air_bending_2.pt'
#: Every previous root had every multiplier under 100, so the ceiling was never reached.  The
#: fix is in the PARSER, never in `replica_dir`: re-padding to `:03d` would rename all 2,021,295
#: symlinks for cosmetics and invalidate a root that A1 has already certified.
_REPLICA_RE = re.compile(r"^(?P<stratum>[A-Za-z0-9]+)_r(?P<rep>\d{2,})$")


def replica_dir(stratum: str, rep: int) -> str:
    return f"{stratum}_r{rep:02d}"


def parse_replica(rel: str) -> tuple[str, int]:
    """'S2a_r03/BookFlip_x/a__ref_b.pt' -> ('S2a', 3).  Raises on anything else.

    Handles >=3-digit replica indices (`S0_r152`); see `_REPLICA_RE` for why that matters.
    """
    first = rel.split(os.sep, 1)[0]
    m = _REPLICA_RE.match(first)
    if not m:
        raise ValueError(f"relative path does not start with a <stratum>_r<NN> dir: {rel!r}")
    return m.group("stratum"), int(m.group("rep"))


def rel_paths(root: Path, sub: str) -> set:
    base = root / sub
    return {str(p.relative_to(base)) for p in base.glob("**/*.pt")}


# --------------------------------------------------------------------------------------
# group-id slugging (A11 σ/S4-weight ruling item 3)
# --------------------------------------------------------------------------------------
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug_group(gid: str) -> str:
    """A path-safe group id: lowercase, non-alphanumeric -> '_', runs collapsed.

    S4's group ids are refVFX effect strings (`0rb4it 360 degree orbit`), so the raw id
    carries spaces.  "the trainer globs fine" is not the bar — robustness across shells,
    `rsync`, and future tooling is, and the slug costs nothing.
    """
    return _SLUG_RE.sub("_", str(gid).lower()).strip("_")


def slug_map(gids) -> tuple[dict, list]:
    """(slug -> raw, collisions).  A collision is a hard error, never a silent merge."""
    out: dict[str, str] = {}
    collisions = []
    for gid in sorted(gids):
        s = slug_group(gid)
        if s in out and out[s] != gid:
            collisions.append(f"{s!r} <- both {out[s]!r} and {gid!r}")
        else:
            out[s] = gid
    return out, collisions


# --------------------------------------------------------------------------------------
# NOMINAL vs EFFECTIVE weights (A11 σ/S4-weight ruling item 2)
# --------------------------------------------------------------------------------------
#: `assemble_root.ensure_mask`: m[:prefix_latents(shape)] = 1 (the prefix anchor);
#: m[-1] = 1 iff two-sided (the suffix anchor).  mask == 1 => the token is conditioned at
#: timestep 0 and EXCLUDED FROM LOSS (`flexible.py:502-546`).  So the loss-bearing token
#: count is a function of the shape and the sidedness, and nothing else.
def conditioned_frames(fhw, sided: str) -> int:
    """Latent frames conditioned = the shape's prefix anchor, +1 for a suffix anchor."""
    return prefix_latents(fhw) + (1 if sided == "two" else 0)


def loss_bearing_tokens(fhw, sided: str) -> int:
    """Target tokens that actually carry loss for one sample of this shape + sidedness.

    Derived, not tabulated — the values are 121f one-sided 4,200 (4,800 x 0.875), 121f
    two-sided 3,900 (4,800 x 0.8125) and S4 1,456 (1,820 x 0.80), and they fall out of
    `1 - conditioned_frames/F` exactly.  S4's ONE-frame anchor conditions 20 % of its tokens
    against 12.5 % at 121f, so the geometric discount on S4's effective share is now small;
    the remaining discount is its lower training shift.  (Under the earlier fixed 2-frame
    anchor this was 40 % / 1,092 tokens — the owner's frame-0 decision is what moved it, and
    it moves because the number is derived from `prefix_latents`, not restated here.)
    """
    f, h, w = (int(x) for x in fhw)
    return (f - conditioned_frames(fhw, sided)) * h * w


def effective_weights(rows: list[dict], replicas: dict) -> dict:
    """NOMINAL sample-count shares vs EFFECTIVE loss-bearing-token shares.

    A11: the nominal vector stays the PRE-REGISTERED quantity — it is what the manifest
    pins and what the contingency branches operate on.  The effective vector is stamped as a
    DERIVED DISCLOSURE only: right for disclosure, wrong as a control variable, because
    pre-registering it would force the nominal weights to chase every geometry change.
    """
    n_samples: dict[str, int] = {}
    n_tokens: dict[str, int] = {}
    for r in rows:
        s = r["stratum"]
        m = int(replicas.get(s, r.get("replicas", 1)))
        n_samples[s] = n_samples.get(s, 0) + m
        n_tokens[s] = n_tokens.get(s, 0) + m * loss_bearing_tokens(r["shape"], r["sided"])
    tot_s = sum(n_samples.values()) or 1
    tot_t = sum(n_tokens.values()) or 1
    return {
        "basis": "nominal = sample count; effective = loss-bearing target tokens "
                 "(mask==0), derived from each sample's shape and ACTUAL sidedness",
        "authority": "A11 (σ / S4-weight ruling) item 2 — nominal is pre-registered, "
                     "effective is a derived disclosure and never a control variable",
        "per_sample_loss_bearing_tokens": {
            f"{tuple(v)}|{sd}": loss_bearing_tokens(v, sd)
            for v, sd in sorted({(tuple(r["shape"]), r["sided"]) for r in rows})},
        "n_samples": n_samples,
        "n_loss_bearing_tokens": n_tokens,
        "nominal_pct": {s: round(100.0 * v / tot_s, 4) for s, v in sorted(n_samples.items())},
        "effective_pct": {s: round(100.0 * v / tot_t, 4) for s, v in sorted(n_tokens.items())},
        "total_samples": tot_s,
        "total_loss_bearing_tokens": tot_t,
    }


# --------------------------------------------------------------------------------------
# two-shape assert — prefer an external importable module, else the fallback below
# --------------------------------------------------------------------------------------
def _fallback_check_shapes(root: Path, manifest: dict, rows: list[dict]) -> list[dict]:
    """Fallback two-shape assert.  Same result contract as an external module.

    Each returned dict is {"name", "ok", "detail", "offenders"} and every one of them is a
    HARD check.  What is asserted:

    * every sample's shape is one of `RULED_SHAPES` (an unruled shape means the encode
      drifted from the bucket, which no other check would notice);
    * a stratum has exactly ONE shape (a mixed-shape stratum means two encodes got merged);
    * the shape set realised on disk == the shape set the root manifest declares;
    * a two-shape root actually contains BOTH shapes when S4 is in the mix (and only one
      when it is not) — i.e. the S4 cutoff branch is visible in the root, not just in prose;
    * the mask store carries exactly the (shape, sidedness) combinations the samples use,
      and no stale mask from another shape (a reused 16-frame mask is a `RuntimeError` at
      `flexible.py:533`, which is a good failure mode but far too late).
    """
    out: list[dict] = []
    by_stratum: dict[str, set] = {}
    unruled, need_masks = [], set()
    for r in rows:
        fhw = tuple(int(x) for x in r["shape"])
        by_stratum.setdefault(r["stratum"], set()).add(fhw)
        need_masks.add((fhw, r["sided"]))
        if fhw not in RULED_SHAPES:
            unruled.append(f"{r['rel']}: latent shape {list(fhw)} is not a ruled shape "
                           f"({sorted(list(k) for k in RULED_SHAPES)})")
    out.append({"name": "A11a_shapes_are_ruled", "ok": not unruled,
                "detail": f"{len(rows)} sample rows carry only ruled latent shapes"
                          if not unruled else f"{len(unruled)} unruled shapes",
                "offenders": unruled})

    mixed = [f"{s}: {sorted(list(x) for x in shapes)}"
             for s, shapes in sorted(by_stratum.items()) if len(shapes) != 1]
    out.append({"name": "A11b_one_shape_per_stratum", "ok": not mixed,
                "detail": " | ".join(f"{s} {sorted(list(x) for x in v)[0]}"
                                     for s, v in sorted(by_stratum.items()))
                          if not mixed else f"{len(mixed)} strata carry >1 shape",
                "offenders": mixed})

    realized = {fhw for shapes in by_stratum.values() for fhw in shapes}
    declared = {tuple(int(x) for x in v["latent_fhw"])
                for v in (manifest.get("shapes", {}).get("per_shape") or [])}
    diff = sorted(list(x) for x in (realized ^ declared))
    out.append({"name": "A11c_declared_shapes_match_disk", "ok": not diff,
                "detail": f"{len(realized)} distinct shape(s) on disk == declared"
                          if not diff else "declared vs realised shape sets differ",
                "offenders": [f"symmetric difference: {diff}"] if diff else []})

    s4 = bool(manifest.get("s4_in_mix"))
    want_n = 2 if s4 else 1
    ok_n = len(realized) == want_n
    out.append({"name": "A11d_two_shapes_iff_s4", "ok": ok_n,
                "detail": (f"s4_in_mix={s4} and the root holds {len(realized)} shape(s) "
                           f"{sorted(list(x) for x in realized)} — as ruled"),
                "offenders": [] if ok_n else
                [f"s4_in_mix={s4} implies {want_n} shape(s), found {len(realized)}: "
                 f"{sorted(list(x) for x in realized)}"]})

    store = root / "_mask_store"
    have = {p.name for p in store.glob("*.pt")} if store.is_dir() else set()
    #: `mask_store_name`, never a restated f-string.  This line used to build
    #: `f{f}_h{h}_w{w}_{sided}sided.pt` itself, which predates the `p{prefix}` component and made
    #: A11e report all three real masks "missing" AND all three "stale" simultaneously — a root
    #: whose every per-sample mask B2 had just verified.  Fourth site to carry a hand-rolled copy
    #: of this name (after regen_masks.py and build_smoke_root.py); hence one function, here.
    want = {mask_store_name(f, h, w, sided) for (f, h, w), sided in need_masks}
    bad = [f"missing mask {n}" for n in sorted(want - have)]
    bad += [f"stale mask {n} (no sample uses it)" for n in sorted(have - want)]
    out.append({"name": "A11e_mask_store_matches_shapes", "ok": not bad,
                "detail": f"mask store carries exactly the {len(want)} (shape, sidedness) "
                          f"combinations the samples use" if not bad
                          else f"{len(bad)} mask-store problems", "offenders": bad})

    # SAMPLES.jsonl is written by the assembler, so trusting it alone would only prove the
    # assembler is self-consistent.  Load ONE real tensor per stratum and confirm the shape
    # the row claims is the shape on disk.  (The dry-run epoch does this for every sample;
    # this is the cheap version that runs inside the battery.)
    drift = []
    try:
        import torch  # noqa: PLC0415

        for stratum, shapes in sorted(by_stratum.items()):
            row = next(r for r in rows if r["stratum"] == stratum)
            p = root / "latents" / row["rel"]
            d = torch.load(os.path.realpath(p), map_location="cpu", weights_only=True)
            disk = (int(d["num_frames"]), int(d["height"]), int(d["width"]))
            if disk != tuple(int(x) for x in row["shape"]):
                drift.append(f"{stratum}: SAMPLES.jsonl says {row['shape']} but "
                             f"{p} holds {list(disk)}")
        detail = (f"one tensor per stratum loaded; the declared shape is the shape on disk "
                  f"({len(by_stratum)} strata spot-checked)")
    except Exception as exc:  # noqa: BLE001 — an unreadable tensor is itself a failure
        drift.append(f"could not verify shapes against disk: {exc!r}")
        detail = "shape spot-check could not run"
    out.append({"name": "A11f_declared_shape_is_the_disk_shape", "ok": not drift,
                "detail": detail if not drift else f"{len(drift)} shape disagreements",
                "offenders": drift})
    return out


_fallback_check_shapes.source = "root_common._fallback_check_shapes"


def _external_shape_module():
    """Find the separate two-shape module, if one exists.  Returns (module, entrypoint).

    Coordination point: the two-shape work is deliberately owned by a separate module so
    that neither agent edits the other's file.  Two entrypoints are accepted —
    ``check_shapes(root, manifest, rows) -> list`` of result records, or
    ``assert_two_shapes(root, ...) -> int`` (the `assert_root_shapes.py` contract, whose
    per-check records are read back out of the report it writes).
    """
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    for path in sorted(HERE.glob("*shape*.py")):
        if path.name.startswith("_"):
            continue
        try:
            mod = __import__(path.stem)
        except Exception as exc:  # noqa: BLE001 — never silently degrade
            print(f"[shapes] {path.name} import failed ({exc!r}); trying the next candidate",
                  file=sys.stderr)
            continue
        for entry in ("check_shapes", "assert_two_shapes"):
            if callable(getattr(mod, entry, None)):
                return mod, entry
    return None, None


def _run_external(mod, entry: str, root: Path, manifest: dict, rows: list[dict]) -> list[dict]:
    """Run the external two-shape battery and return its per-check records.

    `expected_classes` is NARROWED (never widened) to the shapes this root's manifest
    declares, because the module's default expectation names both ruled shapes and would
    therefore false-FAIL on the pre-registered S4-cutoff branch, where the root legitimately
    holds ONE shape.  Narrowing keeps every check intact — an undeclared or unruled shape
    still has no expectation to match and still fails — while letting the ruled branch pass.
    Fixing the expectation rather than the check is the discipline; A11d (below) is what
    stops the narrowing from being a hole, by asserting the branch itself.
    """
    if entry == "check_shapes":
        return list(mod.check_shapes(root, manifest, rows))

    declared = {tuple(int(x) for x in s["latent_fhw"])
                for s in (manifest.get("shapes", {}).get("per_shape") or [])}
    expected = getattr(mod, "EXPECTED_SHAPE_CLASSES", None)
    narrowed = ({k: v for k, v in expected.items() if k in declared} or None) if expected else None
    report = root / "SHAPE_ASSERT_REPORT.json"
    code = mod.assert_two_shapes(root, expected_classes=narrowed, report_path=report)
    if not report.exists():
        return [{"name": "A11x_external_two_shape_module", "ok": code == 0,
                 "detail": f"{mod.__name__}.assert_two_shapes returned {code} but wrote no "
                           f"report at {report}", "offenders": []}]
    rec = read_json(report)
    out = list(rec.get("results") or [])
    if not out:
        out = [{"name": "A11x_external_two_shape_module", "ok": code == 0,
                "detail": f"{mod.__name__}.assert_two_shapes returned {code}",
                "offenders": []}]
    return out


def shape_assert():
    """The two-shape assert to run: the record-level clauses PLUS the external module.

    They are complementary, not redundant, and that is why both run:

    * the clauses in `_fallback_check_shapes` read the assembler's OWN record
      (`SAMPLES.jsonl` + `ROOT_MANIFEST.json`) and ask whether it agrees with the ruling and
      with one loaded tensor per stratum — i.e. *did the assembler tell the truth about what
      it built*, which no tensor-level pass can answer because a stale `_shape_cache.json`
      makes the assembler self-consistent and wrong;
    * the external module opens EVERY tensor in all five trees and checks per-shape set
      equality, per-sample geometry agreement across the trees, token-count collisions and
      the trainer's own index line — i.e. *is the media itself coherent*.

    If the external module is absent the record-level clauses still run, and the returned
    `.source` says so, so a degraded run is never silent.
    """
    mod, entry = _external_shape_module()

    def run(root: Path, manifest: dict, rows: list[dict]) -> list[dict]:
        out = _fallback_check_shapes(root, manifest, rows)
        if mod is None:
            out.append({
                "name": "A11x_external_two_shape_module_present", "ok": False,
                "detail": "no importable two-shape module was found — only the record-level "
                          "clauses ran, so no tensor in the root was cross-checked against "
                          "any other. A9 §5(iv) asks for the tensor-level pass.",
                "offenders": [f"expected a module in {HERE} exposing check_shapes() or "
                              f"assert_two_shapes()"]})
            return out
        try:
            out.extend(_run_external(mod, entry, root, manifest, rows))
        except Exception as exc:  # noqa: BLE001 — a crash there is a FAIL, not a traceback
            out.append({"name": "A11x_external_two_shape_module", "ok": False,
                        "detail": f"{mod.__name__}.{entry} raised", "offenders": [repr(exc)]})
        return out

    run.source = (f"root_common._fallback_check_shapes + scripts/ctt_v2/{mod.__name__}.py:{entry}"
                  if mod else "root_common._fallback_check_shapes (NO external module found)")
    return run
