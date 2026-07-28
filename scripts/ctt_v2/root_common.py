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

#: RULING 3 pre-registered fallback: if S1 drops, the mix renormalises (S0 15 / S2 85)
ABSENT_POLICY = "renormalize_proportionally"

TRIGGER = "sksz"
TRIGGER_SENTENCE = f" {TRIGGER}."
OUTCOME_MARKER = "The scene transforms into "

SEED = 42
N_INLINE_OOD_OPS = 8

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
            "provenance": self.provenance,
        }


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


def load_exclusions(prereg_inline_ood: Path | None = None) -> Exclusions:
    holdout = read_json(HOLDOUT_S2)
    pool = read_json(CONTENT_POOL)
    universe, prov = eval_endpoint_universe()
    ex = Exclusions(
        holdout_shaders=set(holdout["holdout_shaders"]),
        inline_ood_ops=set(),
        reserved_pool_clips={r["clip_id"] for r in pool["reserved"]},
        zs_classes=zs_classes(),
        eval_endpoints=universe,
        provenance={
            "holdout_shaders": str(HOLDOUT_S2.relative_to(REPO_ROOT)),
            "reserved_pool_clips": str(CONTENT_POOL.relative_to(REPO_ROOT)),
            "zs_classes": str(SPLIT_PATH.relative_to(REPO_ROOT)),
            "eval_endpoints": prov,
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
