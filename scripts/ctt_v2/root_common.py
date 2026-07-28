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
#: Ruled weights: S0 15 / S1 6 / S2a 34.5 / S2b 34.5 / S4 10.
#: The previous value here was {S0 15, S1 6, S2a 39.5, S2b 39.5, S4 0.0} from the
#: SUPERSEDED "S4 OUT" ruling. That was a live landmine: assert A3 validates the
#: realized mix AGAINST these constants, so an assembly run would have built the
#: wrong mix and then certified it as correct. Caught by the DATASET.md audit.
INTENDED_WEIGHTS_PCT = {"S0": 15.0, "S1": 6.0, "S2a": 34.5, "S2b": 34.5, "S4": 10.0}
MIX_TOLERANCE_PP = 0.5

#: Guard: these weights are a RULING, not a preference. If they are edited again,
#: the sum must still be 100 and every stratum the mix names must be present.
assert abs(sum(INTENDED_WEIGHTS_PCT.values()) - 100.0) < 1e-9, (
    f"INTENDED_WEIGHTS_PCT must sum to 100, got {sum(INTENDED_WEIGHTS_PCT.values())}"
)

#: A9's PRE-REGISTERED CONTINGENCY BRANCHES, ratified verbatim by A11 item 3 (2026-07-28):
#:   "S1-fail -> 15/0/36.5/36.5/12; S4-cutoff -> 15/6/39.5/39.5/0; both -> 15/42.5/42.5
#:    (i.e. A9's 15/73/12, 15/79, 15/85 split pro-rata per half)."
#: A5 RULING 3's original S1-drop literal (15 / 42.5 / 42.5) was computed while S4 was OUT
#: and the three shares had to absorb all 85 pp; it is now the *both-absent* branch, not the
#: S1-only one.  Keyed by ",".join(sorted(absent)) — exactly the key `assemble_root.py`
#: builds — so the branch is SELECTED, never re-derived by hand at assembly time.
ABSENT_BRANCH_WEIGHTS_PCT = {
    "S1":    {"S0": 15.0, "S2a": 36.5, "S2b": 36.5, "S4": 12.0},
    "S4":    {"S0": 15.0, "S1": 6.0, "S2a": 39.5, "S2b": 39.5},
    "S1,S4": {"S0": 15.0, "S2a": 42.5, "S2b": 42.5},
}

#: Guard: every branch must sum to 100 and cover EXACTLY the complement of its own key.
for _absent_key, _branch in ABSENT_BRANCH_WEIGHTS_PCT.items():
    _gone = set(_absent_key.split(","))
    assert _gone <= set(INTENDED_WEIGHTS_PCT), f"unknown stratum in branch key {_absent_key!r}"
    assert set(_branch) == set(INTENDED_WEIGHTS_PCT) - _gone, (
        f"ABSENT_BRANCH_WEIGHTS_PCT[{_absent_key!r}] must name exactly "
        f"{sorted(set(INTENDED_WEIGHTS_PCT) - _gone)}, got {sorted(_branch)}"
    )
    assert abs(sum(_branch.values()) - 100.0) < 1e-9, (
        f"ABSENT_BRANCH_WEIGHTS_PCT[{_absent_key!r}] must sum to 100, got {sum(_branch.values())}"
    )
del _absent_key, _branch, _gone

#: Fallback for an absent-set A9 did NOT pre-register a branch for (e.g. S2b alone).
ABSENT_POLICY = "renormalize_proportionally"

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
                   "strata": ("S0", "S1", "S2a", "S2b"),
                   "authority": "REF_root_format.md (verified by loading)"},
    (5, 14, 26): {"name": "s4_33f", "px": (832, 448, 33), "fps": 16.0, "strata": ("S4",),
                  "authority": "DOSSIER §13.2 — verified on disk; 832x448 is a pure "
                               "16-row centre crop, 464 is not a multiple of 32"},
}


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
def solve_multipliers(base_counts: dict[str, int], weights_pct: dict[str, float],
                      tol_pp: float = MIX_TOLERANCE_PP, max_mult: int = 200) -> dict:
    """Choose integer replica counts so the *counted* mix lands within `tol_pp`.

    Deterministic sweep: for each stratum taken as the anchor and each anchor multiplier,
    derive the others by rounding, then keep the (max-deviation, total-size) minimum.
    Returns the decision record; raises if nothing lands inside tolerance.
    """
    names = sorted(k for k in base_counts if base_counts[k] > 0 and weights_pct.get(k, 0) > 0)
    if not names:
        raise ValueError("no stratum with a positive base count and a positive weight")
    total_w = sum(weights_pct[s] for s in names)
    w = {s: weights_pct[s] / total_w for s in names}

    # Objective: the SMALLEST root that lands inside tolerance.  (Ranking by deviation
    # alone always prefers a bigger root — finer rounding grain — and would silently
    # inflate a 60 k-sample mix into a 500 k-sample one.)
    best = None
    for anchor in names:
        for m in range(1, max_mult + 1):
            target_total = base_counts[anchor] * m / w[anchor]
            mult = {s: max(1, int(round(w[s] * target_total / base_counts[s]))) for s in names}
            mult[anchor] = m
            tot = sum(base_counts[s] * mult[s] for s in names)
            dev = {s: 100.0 * base_counts[s] * mult[s] / tot - 100.0 * w[s] for s in names}
            md = round(max(abs(v) for v in dev.values()), 9)
            key = (0, tot, md) if md <= tol_pp else (1, md, tot)
            if best is None or key < best[0]:
                best = (key, dict(mult), dev, tot)

    key, mult, dev, tot = best
    max_dev = max(abs(v) for v in dev.values())
    rec = {
        "multipliers": mult,
        "base_counts": {s: base_counts[s] for s in names},
        "realized_counts": {s: base_counts[s] * mult[s] for s in names},
        "total": tot,
        "intended_pct": {s: round(100.0 * w[s], 6) for s in names},
        "realized_pct": {s: round(100.0 * base_counts[s] * mult[s] / tot, 6) for s in names},
        "deviation_pp": {s: round(dev[s], 6) for s in names},
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


#: A10 (2026-07-28) — the standing role-scoped exclusion, DERIVED from the sidecar so the
#: two can never drift.  Authority: `data/processed/ctt_v2_strata/
#: POOL_DROPS_M3_ADJUDICATION.json:role_scoped_exclusions` (verdict "ROLE-SCOPED EXCLUSION.
#: Whole-clip drop DENIED as written. Confidence 0.93.").
#:
#: `openvid_T1MiFx98l3g_0_50to156` has a blank-white A-anchor (ffprobe YMIN=YMAX=YAVG=232)
#: and a healthy B-anchor (Y 12-242); it is field B in 37/37 rendered S2b rows, byte-pure on
#: every one, so the render contract makes the blank window structurally unreachable in the
#: role it occupies.  The pool file and the endpoint frame cache stay BYTE-UNCHANGED — a
#: completed render and the running encodes are keyed to them — which is exactly why the
#: exclusion has to live in code that every consumer reads.
#:
#: A10's standing rule, campaign-wide: **defects are dispositioned at the unit of
#: CONSUMPTION — (clip, role) — not the unit of storage.**  Whole-clip drops are reserved
#: for role-independent defects (leakage, duplication, provenance) or clips not yet
#: consumed anywhere.  `enforced_at` in the sidecar enumerates the consumption channels;
#: a recorded exclusion that no code reads is a landmine.
try:
    ROLE_EXCLUSIONS: dict = {c: tuple(sorted(r))
                             for c, r in load_caption_store_exclusions()[0].items()}
except OSError:  # pragma: no cover — the sidecar is on /projects, not /tmp
    ROLE_EXCLUSIONS = {}


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
    if key not in store:
        raise SystemExit(f"[caption] caption store has no entry for key {key!r} "
                         f"((clip, role) = ({clip!r}, {role!r})); no cross-role fallback exists")
    return store[key]


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
_REPLICA_RE = re.compile(r"^(?P<stratum>[A-Za-z0-9]+)_r(?P<rep>\d{2})$")


def replica_dir(stratum: str, rep: int) -> str:
    return f"{stratum}_r{rep:02d}"


def parse_replica(rel: str) -> tuple[str, int]:
    """'S2a_r03/BookFlip_x/a__ref_b.pt' -> ('S2a', 3).  Raises on anything else."""
    first = rel.split(os.sep, 1)[0]
    m = _REPLICA_RE.match(first)
    if not m:
        raise ValueError(f"relative path does not start with a <stratum>_r<NN> dir: {rel!r}")
    return m.group("stratum"), int(m.group("rep"))


def rel_paths(root: Path, sub: str) -> set:
    base = root / sub
    return {str(p.relative_to(base)) for p in base.glob("**/*.pt")}


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
    want = {f"f{f}_h{h}_w{w}_{sided}sided.pt" for (f, h, w), sided in need_masks}
    bad = [f"missing mask {n}" for n in sorted(want - have)]
    bad += [f"stale mask {n} (no sample uses it)" for n in sorted(have - want)]
    out.append({"name": "A11e_mask_store_matches_shapes", "ok": not bad,
                "detail": f"mask store carries exactly the {len(want)} (shape, sidedness) "
                          f"combinations the samples use" if not bad
                          else f"{len(bad)} mask-store problems", "offenders": bad})
    return out


_fallback_check_shapes.source = "root_common._fallback_check_shapes"


def shape_assert():
    """The two-shape assert implementation to use.

    A separate importable module is preferred if one exists — coordination point, so the
    two-shape work is never written twice.  Contract: any module in this directory whose
    name mentions `shape` and which exposes ``check_shapes(root, manifest, rows) -> list``
    of ``{"name", "ok", "detail", "offenders"}`` records.  `rows` are the SAMPLES.jsonl
    rows (they carry `rel`, `stratum`, `sided`, `shape`).  The chosen implementation is
    printed and recorded, so a fallback is never silent.
    """
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    for path in sorted(HERE.glob("*shape*.py")):
        if path.name.startswith("_"):
            continue
        try:
            mod = __import__(path.stem)
            fn = getattr(mod, "check_shapes", None)
        except Exception as exc:  # noqa: BLE001 — never silently degrade
            print(f"[shapes] {path.name} import failed ({exc!r}); trying the next candidate",
                  file=sys.stderr)
            continue
        if callable(fn):
            fn.source = f"scripts/ctt_v2/{path.name}:check_shapes"
            return fn
    return _fallback_check_shapes
