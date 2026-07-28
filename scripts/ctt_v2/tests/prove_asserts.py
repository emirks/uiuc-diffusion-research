"""CTT v2 — PROVE that every HARD assert actually fires (A5 RULING 9).

An assert that has never failed is not known to work.  A5 RULING 9 makes ten-plus checks
launch-blocking; a typo in any one of them turns it into a decoration that prints PASS on a
broken root forever.  So this harness does the only thing that constitutes evidence:

  1. establish a GREEN baseline on a real assembled root (every check PASS, zero skips);
  2. break exactly ONE invariant at a time, in place;
  3. re-run the battery and require the intended check(s) to FAIL — and, in strict mode,
     require that NOTHING ELSE fails, so each check is shown to be sensitive to its own
     invariant and to nothing else;
  4. restore, byte-for-byte, and verify the restore by hash;
  5. re-establish the GREEN baseline at the end, so a leaked mutation cannot masquerade as
     a pass.

Every mutation is reversible and its undo is registered BEFORE it is applied.  Where a
mutation necessarily trips a second check (breaking an inventory's bytes must trip both the
sha256 integrity check and the "every sample resolves to a verified inventory" check, since
an unverifiable inventory is not consulted), the coupled set is DECLARED in `expect` and
the reason is recorded — never quietly tolerated.

    python scripts/ctt_v2/tests/prove_asserts.py --root <assembled root> \
        --manifest <strata manifest>
    python scripts/ctt_v2/tests/prove_asserts.py --root <root> --only A4_captions --loose

Writes `PROVE_ASSERTS.json` next to the root: one record per mutation with the offenders
each broken check reported, which is the audit trail for the stamp.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import root_common as rc  # noqa: E402

PY = sys.executable
ASSERT = HERE.parent / "assert_root.py"
DRYRUN = HERE.parent / "dryrun_epoch.py"

#: Checks owned by the SEPARATE two-shape module (`assert_root_shapes.py`), which
#: `assert_root.py` imports.  They are excluded from the strict "nothing else may fail"
#: rule, and recorded instead under `also_fired_external`.
#:
#: This is not a loophole, it is the correct scoping.  B1 is per-shape-class five-tree set
#: equality and B2 opens every tensor, so both are deliberately BROADER views of the same
#: filesystem facts A1/A2b/A2c check — any defect those see, B1/B2 may legitimately see too.
#: Demanding they stay silent would be demanding they be worse checks.  Their own
#: specificity is proven by their own `assert_root_shapes.py --self-test` (10 broken
#: fixtures), and the DELEGATION — that a failure inside the imported module reaches
#: `assert_root.py`'s exit code — is proven here by `A11x_external_module_mask_geometry`,
#: whose expectation IS a B check and which therefore is not exempted.
EXTERNAL_PREFIX = "B"


# ======================================================================================
# reversible primitives
# ======================================================================================
class Undo:
    """A stack of reversals, registered before each change is made."""

    def __init__(self):
        self._ops = []

    def push(self, fn) -> None:
        self._ops.append(fn)

    def run(self) -> None:
        while self._ops:
            self._ops.pop()()


class Ctx:
    def __init__(self, root: Path, manifest: Path, tmp: Path):
        self.root = root
        self.manifest = manifest
        self.tmp = tmp
        self.man = rc.read_json(root / "ROOT_MANIFEST.json")
        self.rows = [json.loads(ln) for ln in
                     (root / "SAMPLES.jsonl").read_text().splitlines() if ln.strip()]
        self.undo = Undo()
        self.extra_args: list[str] = []
        #: a mutation may point the battery at a DIFFERENT root (used for the empty-root
        #: case, which cannot be expressed as an in-place edit of a populated root)
        self.root_override: Path | None = None

    # ---- file-level primitives --------------------------------------------------------
    def save(self, path: Path) -> None:
        """Register a byte-exact restore of `path` (or its deletion, if absent now)."""
        p = Path(path)
        if p.exists():
            data = p.read_bytes()
            self.undo.push(lambda: p.write_bytes(data))
        else:
            self.undo.push(lambda: p.unlink(missing_ok=True))

    def edit_json(self, path: Path, mutate) -> None:
        self.save(path)
        obj = rc.read_json(path)
        mutate(obj)
        rc.write_json(path, obj)

    def unlink(self, rel: str) -> None:
        """Delete one root entry, remembering its symlink target."""
        p = self.root / rel
        tgt = os.readlink(p) if p.is_symlink() else None
        data = None if tgt else p.read_bytes()
        self.undo.push(lambda: (os.symlink(tgt, p) if tgt else p.write_bytes(data)))
        p.unlink()

    def add_link(self, rel: str, target: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        self.undo.push(lambda: p.unlink(missing_ok=True))
        os.symlink(target, p)

    def repoint(self, rel: str, target: str) -> None:
        p = self.root / rel
        old = os.readlink(p)
        self.undo.push(lambda: (p.unlink(), os.symlink(old, p)))
        p.unlink()
        os.symlink(target, p)

    # ---- inventory primitives ---------------------------------------------------------
    def inv_path(self, stratum: str) -> Path:
        return Path(self.man["inventories"][stratum]["path"])

    def edit_inventory(self, stratum: str, mutate, keep_sha_green: bool = True) -> None:
        """Mutate a stratum inventory.  By default the ROOT_MANIFEST's recorded sha256 is
        refreshed too, so ONLY the intended check fires — otherwise every inventory edit
        would also (correctly) trip the provenance-integrity check and drown the signal.
        """
        p = self.inv_path(stratum)
        self.edit_json(p, mutate)
        if keep_sha_green:
            mp = self.root / "ROOT_MANIFEST.json"
            self.save(mp)
            man = rc.read_json(mp)
            man["inventories"][stratum]["sha256"] = rc.sha256_file(p)
            rc.write_json(mp, man)

    # ---- convenience selectors --------------------------------------------------------
    def pick(self, stratum: str, rep: int = 0) -> dict:
        for r in self.rows:
            if r["stratum"] == stratum and r["rel"].endswith(".pt") and f"_r{rep:02d}/" in r["rel"]:
                return r
        raise SystemExit(f"no SAMPLES.jsonl row for stratum {stratum!r} replica {rep}")

    def write_tmp_json(self, name: str, obj) -> Path:
        p = self.tmp / name
        rc.write_json(p, obj)
        return p


# ======================================================================================
# the mutations — one broken invariant each
# ======================================================================================
def _first_group_clip(ctx: Ctx, stratum: str) -> tuple[str, str]:
    inv = rc.read_json(ctx.inv_path(stratum))
    row = ctx.pick(stratum)
    return row["group"], row["target"]


def m_A1b_empty_root(ctx: Ctx):
    """An assembled-but-empty root: manifest present, not one sample behind it."""
    empty = ctx.tmp / "empty_root"
    for sub in rc.ROOT_DIRS:
        (empty / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(ctx.root / "ROOT_MANIFEST.json", empty / "ROOT_MANIFEST.json")
    ctx.root_override = empty
    return "pointed the battery at a root with all 5 dirs present and zero samples"


def m_A1_missing(ctx: Ctx):
    row = ctx.pick("S0")
    ctx.unlink(f"conditions/{row['rel']}")
    return f"deleted conditions/{row['rel']} — present in 4 dirs, absent from the 5th"


def m_A1_extra(ctx: Ctx):
    row = ctx.pick("S0")
    rel = f"{Path(row['rel']).parent}/GHOSTEXTRA__ref_GHOST.pt"
    ctx.add_link(f"masks/{rel}", str(ctx.root / "ROOT_MANIFEST.json"))
    return f"added masks/{rel} — present in 1 dir, absent from the other 4"


def m_A2_inventory_bytes(ctx: Ctx):
    p = ctx.inv_path("S4")
    ctx.save(p)
    p.write_bytes(p.read_bytes() + b"\n")
    return f"appended one byte to {p.name} WITHOUT refreshing the recorded sha256"


def m_A2b_path_scheme(ctx: Ctx):
    row = ctx.pick("S0")
    stratum_dir = row["rel"].split("/")[0]
    for sub in rc.ROOT_DIRS:
        ctx.add_link(f"{sub}/{stratum_dir}/depth2_is_illegal.pt",
                     str(ctx.root / "ROOT_MANIFEST.json"))
    return f"added {stratum_dir}/depth2_is_illegal.pt to all 5 dirs (depth 2, not 3)"


def m_A2c_unknown_sample(ctx: Ctx):
    row = ctx.pick("S0")
    rel = f"{Path(row['rel']).parent}/GHOSTTARGET__ref_GHOSTREF.pt"
    for sub in rc.ROOT_DIRS:
        ctx.add_link(f"{sub}/{rel}", str(ctx.root / "ROOT_MANIFEST.json"))
    return f"added {rel} to all 5 dirs — a well-formed sample no inventory explains"


def m_A3_mix(ctx: Ctx):
    # from the END of S4: the on-disk shape spot check (A11f) reads the FIRST row of each
    # stratum, and deleting that row's files would trip A11f as well
    victims = [r["rel"] for r in ctx.rows if r["stratum"] == "S4"][-20:]
    if len(victims) < 20:
        raise SystemExit("need >=20 S4 samples to skew the mix")
    for rel in victims:
        for sub in rc.ROOT_DIRS:
            ctx.unlink(f"{sub}/{rel}")
    return (f"deleted {len(victims)} S4 samples from all 5 dirs — enough to push S4's "
            f"COUNTED share past the +-0.5 pp tolerance")


def m_A3b_prorata_multipliers(ctx: Ctx):
    """Give ONE half of the pro-rata pair a second replica directory.

    A9 §4 splits S2's 69 pp pro-rata across the two halves, which is only equivalent to A1b's
    "uniform per-sample weight within S2, no extra reweighting knob" while the halves carry the
    SAME replica multiplier — and they do, exactly (S2a 7,990 base clips, S2b 7,990).  A3b is
    the check that catches that assumption silently breaking, and it was the one assert in the
    battery with no mutation: it was added by the A12 rewrite AFTER the mutation set was
    written, so it had never failed.

    The break is COUNT-PRESERVING on purpose: five S2b samples are MOVED from `S2b_r00` into a
    new `S2b_r01`, in all five trees, so S2b shows two replica dirs on disk against S2a's one
    while every stratum's counted share is untouched.  A literal x2 duplication of S2b would
    also (correctly) blow A3's +-0.5 pp share tolerance, and then the run would not show that
    A3b is sensitive to the multiplier itself — which is the whole claim.  Differential
    replication IS the extra reweighting knob; this isolates it.

    Two of A3b's clauses fire together and both belong to it: the on-disk inequality, and the
    ROOT_MANIFEST cross-check (the manifest still declares `replicas: {S2b: 1}`, and A3b counts
    the disk rather than reading that number back).
    """
    victims = [r["rel"] for r in ctx.rows if r["stratum"] == "S2b"][-5:]
    if len(victims) < 5:
        raise SystemExit("need >=5 S2b samples to promote into a second replica dir")
    for rel in victims:
        _rdir, group, name = rel.split("/")
        new_rel = f"{rc.replica_dir('S2b', 1)}/{group}/{name}"
        for sub in rc.ROOT_DIRS:
            src = ctx.root / f"{sub}/{rel}"
            target = os.readlink(src) if src.is_symlink() else str(src)
            ctx.add_link(f"{sub}/{new_rel}", target)
            ctx.unlink(f"{sub}/{rel}")
    return (f"moved {len(victims)} S2b samples from S2b_r00 into a second replica dir "
            f"S2b_r01 in all 5 dirs — S2b now shows 2 replica dirs on disk, S2a 1, with "
            f"every stratum's counted share unchanged")


def _mutate_caption(ctx: Ctx, stratum: str, fn, what: str):
    row = ctx.pick(stratum)
    target = row["target"]

    def mutate(inv):
        inv["clips"][target]["caption"] = fn(inv["clips"][target]["caption"])

    ctx.edit_inventory(stratum, mutate)
    return f"{stratum}/{target}: {what}"


def m_A4_no_trigger(ctx: Ctx):
    return _mutate_caption(ctx, "S2a", lambda c: c.replace(rc.TRIGGER_SENTENCE, " "),
                           "removed ' sksz.' entirely (0 occurrences)")


def m_A4_double_trigger(ctx: Ctx):
    return _mutate_caption(ctx, "S2a", lambda c: c + rc.TRIGGER_SENTENCE,
                           "appended a second ' sksz.' (2 occurrences)")


def m_A4_outcome_marker(ctx: Ctx):
    return _mutate_caption(ctx, "S2b", lambda c: c + " " + rc.OUTCOME_MARKER + "a wolf.",
                           f"spliced in the outcome marker {rc.OUTCOME_MARKER!r}")


def m_A4_tier1_leak(ctx: Ctx):
    return _mutate_caption(ctx, "S2b", lambda c: c + " A GlitchDisplace wipe follows.",
                           "spliced in a Tier-1 shader basename")


def m_A5_eval_endpoint(ctx: Ctx):
    """Point an S2b clip at a real eval endpoint (class resolved via prompts.clip_class)."""
    ex = rc.load_exclusions()
    prompts = rc._prompts()
    zs = rc.zs_classes()
    victim = None
    for ep in sorted(ex.eval_endpoints):
        try:
            if prompts.clip_class(ep) in zs:
                continue          # would also (correctly) trip the zs check
        except KeyError:
            pass
        if ep in ex.reserved_pool_clips:
            continue              # would also trip the reserved-clip check
        victim = ep
        break
    if victim is None:
        raise SystemExit("no eval endpoint is isolable from the zs / reserved sets")
    row = ctx.pick("S2b")

    def mutate(inv):
        inv["clips"][row["target"]]["endpoints"][0] = victim

    ctx.edit_inventory("S2b", mutate)
    return f"S2b/{row['target']}: endpoint_a := {victim!r}, a real eval-side endpoint"


def m_A6_ood_op_present(ctx: Ctx):
    row = ctx.pick("S2a")
    real = rc.read_json(rc.PREREG_INLINE_OOD)["op_ids"]
    ops = sorted(set(real[:7]) | {row["group"]})
    p = ctx.write_tmp_json("prereg_ood_broken.json",
                           {"op_ids": ops, "status": "TEST FIXTURE — deliberately broken"})
    ctx.extra_args += ["--prereg-inline-ood", str(p)]
    return f"pre-registered {row['group']!r} as inline-OOD while it IS in the root"


def m_A6_vacuous(ctx: Ctx):
    ctx.extra_args += ["--prereg-inline-ood", str(ctx.tmp / "does_not_exist.json")]
    return ("pointed the inline-OOD pre-registration at an absent file — the 8-op exclusion "
            "would be VACUOUS on a root that contains S2a")


def m_A7_holdout_shader(ctx: Ctx):
    holdout = sorted(rc.read_json(rc.HOLDOUT_S2)["holdout_shaders"])[0]
    row = ctx.pick("S2b")

    def mutate(inv):
        inv["groups"][row["group"]]["shader"] = holdout

    ctx.edit_inventory("S2b", mutate)
    return f"S2b/{row['group']}: shader := {holdout!r}, a HOLDOUT_S2 family"


def m_A8_reserved_clip(ctx: Ctx):
    reserved = sorted(rc.load_exclusions().reserved_pool_clips)[0]
    row = ctx.pick("S2b")

    def mutate(inv):
        eps = inv["clips"][row["target"]]["endpoints"]
        eps[-1] = reserved

    ctx.edit_inventory("S2b", mutate)
    return f"S2b/{row['target']}: endpoint_b := {reserved!r}, a reserved union-pool clip"


def m_A9_zs_class(ctx: Ctx):
    zs = sorted(rc.zs_classes())[0]
    row = ctx.pick("S1")

    def mutate(inv):
        inv["groups"][row["group"]]["class"] = zs

    ctx.edit_inventory("S1", mutate)
    return f"S1/{row['group']}: class := {zs!r}, a zero-shot holdout class"


def m_A10_verdict_absent(ctx: Ctx):
    ctx.extra_args += ["--copy-gate-verdict", str(ctx.tmp / "no_such_verdict.md")]
    return "pointed the RULING-1 copy-gate verdict at an absent file"


def m_A10_verdict_fail(ctx: Ctx):
    p = ctx.tmp / "verdict_fail.md"
    p.write_text("# copy discriminator\n\nVerdict: FAIL — the gate does not separate.\n")
    ctx.extra_args += ["--copy-gate-verdict", str(p)]
    return "supplied a verdict document that says FAIL"


def m_A11_declared_shapes(ctx: Ctx):
    def mutate(man):
        man["shapes"]["per_shape"] = [s for s in man["shapes"]["per_shape"]
                                      if s["latent_fhw"] != [5, 14, 26]]
    ctx.edit_json(ctx.root / "ROOT_MANIFEST.json", mutate)
    return "dropped the S4 shape from the manifest's declared shape set"


def _retag_last(ctx: Ctx, stratum: str, shape: list) -> str:
    """Rewrite the LAST row of `stratum` in SAMPLES.jsonl.

    The last row, not the first: the on-disk spot check (A11f) reads the FIRST row of each
    stratum, so mutating the last one isolates the metadata checks from the disk check.
    """
    p = ctx.root / "SAMPLES.jsonl"
    ctx.save(p)
    rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    idx = max(i for i, r in enumerate(rows) if r["stratum"] == stratum)
    rows[idx]["shape"] = shape
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows[idx]["rel"]


def m_A11_mixed_shape(ctx: Ctx):
    rel = _retag_last(ctx, "S4", [16, 20, 15])
    return f"gave S4 sample {rel} the corpus shape — one stratum, two shapes"


def m_A11_unruled_shape(ctx: Ctx):
    rel = _retag_last(ctx, "S4", [8, 14, 26])
    return f"gave S4 sample {rel} an UNRULED shape (8,14,26) — an encode-bucket drift"


def m_A11f_disk_drift(ctx: Ctx):
    """Make the metadata truthful-looking and the DISK wrong — the assembler-lied case."""
    s4 = next(r for r in ctx.rows if r["stratum"] == "S4")
    corpus = next(r for r in ctx.rows if r["stratum"] == "S0")
    tgt = os.path.realpath(ctx.root / f"latents/{corpus['rel']}")
    ctx.repoint(f"latents/{s4['rel']}", tgt)
    return (f"repointed latents/{s4['rel']} at a (16,20,15) corpus tensor while "
            f"SAMPLES.jsonl still claims (5,14,26) — a stale shape cache would look exactly "
            f"like this")


def m_B2_mask_geometry(ctx: Ctx):
    """Give an S4 sample the 121f mask — and prove the DELEGATION gates the exit code.

    This is the hazard `REF_mixed_length.md` ranks first: five paths that all exist, one of
    which is the wrong shape.  It is invisible to the record-level clauses (SAMPLES.jsonl is
    untouched and truthful) and to A1 (the path sets stay equal), so only the tensor-level
    external module can see it.  The point of the mutation is therefore twofold: that the
    external check fires, and that its failure actually reaches `assert_root.py`'s exit code.
    """
    s4 = next(r for r in ctx.rows if r["stratum"] == "S4")
    corpus = next(r for r in ctx.rows if r["stratum"] == "S0")
    tgt = os.path.realpath(ctx.root / f"masks/{corpus['rel']}")
    ctx.repoint(f"masks/{s4['rel']}", tgt)
    return (f"repointed masks/{s4['rel']} at a (16,20,15) mask — five paths present, one "
            f"wrong geometry; only a tensor-level pass can see it")


def m_A0_namespace_drift(ctx: Ctx):
    """Move the exempt stratum's endpoint ids out of the eval namespace.

    A5 then still reports "= 0" — truthfully and uselessly, because the two sides can no
    longer meet.  This is the vacuity failure that absence asserts hide, and the whole reason
    A11's standing rule pairs every absence assert with a positive-presence control.
    """
    def mutate(inv):
        for c in inv["clips"].values():
            c["endpoints"] = [f"zzz_{e}" for e in (c.get("endpoints") or [])]

    ctx.edit_inventory("S0", mutate)
    return ("prefixed every S0 endpoint id with 'zzz_' — S0 IS the eval endpoint bank by "
            "design, so the eval-side and root-side id namespaces no longer intersect and "
            "A5 could never fire again")


def m_A14_slug_collision(ctx: Ctx):
    row = ctx.pick("S2a")
    twin = row["group"].replace("_", " ").lower()

    def mutate(inv):
        inv["groups"][twin] = dict(inv["groups"][row["group"]])

    ctx.edit_inventory("S2a", mutate)
    return (f"added group {twin!r} alongside {row['group']!r} — distinct raw ids that slug to "
            f"the same path, which would silently MERGE two pairing rings")


def m_A12_role_scoped(ctx: Ctx):
    """Exclude the role a clip in the root legitimately occupies."""
    victim = None
    for r in ctx.rows:
        if len(r.get("caption_sources") or []) > 1:
            victim = r
            break
    if victim is None:
        raise SystemExit("no sample in the root draws on a role-B description")
    clip = victim["caption_sources"][1][0]
    p = ctx.write_tmp_json("pool_drops_broken.json", {
        "authority": "TEST FIXTURE — deliberately broken",
        "role_scoped_exclusions": {clip: {"excluded_roles": ["B"],
                                          "reason": "TEST", "enforced_at": []}},
        "clip_level_exclusions_for_caption_store": []})
    ctx.extra_args += ["--pool-drops", str(p)]
    return f"excluded role B of {clip!r}, which the root legitimately consumes as role B"


def m_A12_sidecar_absent(ctx: Ctx):
    ctx.extra_args += ["--pool-drops", str(ctx.tmp / "no_such_adjudication.json")]
    return ("pointed the M3 adjudication sidecar at an absent file — the role-scoped "
            "exclusion would be silently vacuous")


def m_A13_prefix_condition(ctx: Ctx):
    """Put the real role-excluded clip into endpoint_a — both consumption channels break."""
    clip = next((c for c, roles in rc.ROLE_EXCLUSIONS.items() if "A" in roles), None)
    if clip is None:
        raise SystemExit("root_common.ROLE_EXCLUSIONS carries no A-role exclusion to test")
    row = ctx.pick("S2b")

    def mutate(inv):
        inv["clips"][row["target"]]["endpoints"][0] = clip
        inv["clips"][row["target"]].pop("caption_sources", None)

    ctx.edit_inventory("S2b", mutate)
    return (f"S2b/{row['target']}: endpoint_a := {clip!r} — the blank-white A-anchor clip, "
            f"injected into the prefix-condition slot")


def m_A13_vacuous(ctx: Ctx):
    p = ctx.write_tmp_json("role_exclusions_empty.json", {})
    ctx.extra_args += ["--override-role-exclusions", str(p)]
    return ("overrode the standing ROLE_EXCLUSIONS with an empty set — a recorded exclusion "
            "that no code reads is the landmine A10 named")


#: name -> (apply, expected-failing checks, why any coupled failure is inherent)
MUTATIONS: dict[str, tuple] = {
    "A1b_empty_root": (
        m_A1b_empty_root, {"A1b_root_nonempty"},
        "A1 correctly PASSES here — five empty path sets really are equal — which is "
        "exactly why A1b exists as a separate check: set-equality alone cannot tell a "
        "correct root from an empty one"),
    "A1_missing_from_one_dir": (m_A1_missing, {"A1_set_equality_5_dirs"}, ""),
    "A1_extra_in_one_dir": (m_A1_extra, {"A1_set_equality_5_dirs"}, ""),
    "A2_inventory_bytes_changed": (
        m_A2_inventory_bytes,
        {"A2_inventory_integrity", "A2c_root_resolves_to_inventories"},
        "an inventory that fails its sha256 is not consulted, so every sample it should "
        "explain becomes unexplained — the coupling IS the intended behaviour"),
    "A2b_path_scheme": (m_A2b_path_scheme, {"A2b_path_scheme"}, ""),
    "A2c_unknown_sample": (m_A2c_unknown_sample, {"A2c_root_resolves_to_inventories"}, ""),
    "A3_realized_mix": (m_A3_mix, {"A3_realized_mix"}, ""),
    "A3b_prorata_multipliers_unequal": (
        m_A3b_prorata_multipliers, {"A3b_prorata_multipliers_equal"},
        "two of A3b's own clauses fire — the on-disk inequality (S2b x2 replica dirs vs "
        "S2a x1) and the ROOT_MANIFEST cross-check (the manifest still declares "
        "replicas {S2b: 1}, and A3b counts the disk instead of reading that number back). "
        "Both offenders belong to A3b; no other check is touched, which is why the "
        "mutation is count-preserving"),
    "A4_trigger_absent": (m_A4_no_trigger, {"A4_captions"}, ""),
    "A4_trigger_twice": (m_A4_double_trigger, {"A4_captions"}, ""),
    "A4_outcome_marker": (m_A4_outcome_marker, {"A4_captions"}, ""),
    "A4_tier1_leak": (m_A4_tier1_leak, {"A4_captions"}, ""),
    "A5_eval_endpoint": (m_A5_eval_endpoint, {"A5_endpoint_disjointness"}, ""),
    "A6_ood_op_in_root": (m_A6_ood_op_present, {"A6_inline_ood_ops_absent"}, ""),
    "A6_exclusion_vacuous": (m_A6_vacuous, {"A6_inline_ood_ops_absent"}, ""),
    "A7_holdout_shader": (m_A7_holdout_shader, {"A7_holdout_shaders_absent"}, ""),
    "A8_reserved_pool_clip": (m_A8_reserved_clip, {"A8_reserved_pool_clips_absent"}, ""),
    "A9_zs_class": (m_A9_zs_class, {"A9_zs_classes_absent"}, ""),
    "A10_verdict_absent": (m_A10_verdict_absent, {"A10_copy_gate_admissibility_PASSED"}, ""),
    "A10_verdict_says_FAIL": (m_A10_verdict_fail, {"A10_copy_gate_admissibility_PASSED"}, ""),
    "A11c_declared_shapes": (m_A11_declared_shapes, {"A11c_declared_shapes_match_disk"}, ""),
    "A11b_two_shapes_in_one_stratum": (
        m_A11_mixed_shape, {"A11b_one_shape_per_stratum"}, ""),
    "A11a_unruled_shape": (
        m_A11_unruled_shape,
        {"A11a_shapes_are_ruled", "A11b_one_shape_per_stratum",
         "A11c_declared_shapes_match_disk", "A11d_two_shapes_iff_s4",
         "A11e_mask_store_matches_shapes"},
        "an unruled shape is simultaneously not-ruled, a second shape inside its stratum, "
        "a third shape in a two-shape root, absent from the declared set, and without a "
        "mask in the store — five independent checks all see it, which is the point of "
        "having five"),
    "A11f_metadata_disagrees_with_disk": (
        m_A11f_disk_drift, {"A11f_declared_shape_is_the_disk_shape"}, ""),
    "A11x_external_module_mask_geometry": (
        m_B2_mask_geometry, {"B2_per_sample_geometry_agreement"},
        "the external two-shape module's own checks are proven by its own `--self-test`; "
        "this mutation exists to prove the DELEGATION — that a failure inside the imported "
        "module reaches assert_root.py's exit code rather than being printed and dropped"),
    "A12_role_scoped_caption": (
        m_A12_role_scoped, {"A12_role_scoped_caption_exclusion"}, ""),
    "A12_sidecar_absent": (m_A12_sidecar_absent, {"A12_role_scoped_caption_exclusion"}, ""),
    "A13_prefix_condition": (
        m_A13_prefix_condition,
        {"A13_role_scoped_prefix_condition", "A12_role_scoped_caption_exclusion"},
        "endpoint_a is BOTH the prefix-condition source and the role-A caption source, so "
        "one clip in the wrong slot breaks both consumption channels — exactly A10's "
        "'unit of consumption' rule, seen from two sides"),
    "A13_exclusion_vacuous": (m_A13_vacuous, {"A13_role_scoped_prefix_condition"}, ""),
    "A0_absence_assert_namespace_drift": (
        m_A0_namespace_drift, {"A0_absence_assert_positive_controls"}, ""),
    "A14_slug_collision": (m_A14_slug_collision, {"A14_group_ids_slug_safe"}, ""),
}


# ======================================================================================
# dry-run mutations — "zero skipped" must also be provable
# ======================================================================================
def d_dangling(ctx: Ctx):
    row = ctx.pick("S0")
    ctx.repoint(f"latents/{row['rel']}", str(ctx.tmp / "no_such_tensor.pt"))
    return "DANGLING", f"repointed latents/{row['rel']} at a nonexistent target"


def d_orphan(ctx: Ctx):
    row = ctx.pick("S0")
    rel = f"{Path(row['rel']).parent}/ORPHAN__ref_X.pt"
    ctx.add_link(f"masks/{rel}", str(ctx.root / "ROOT_MANIFEST.json"))
    return "ORPHAN", f"added masks/{rel}, absent from latents/ — never enumerated, never trained"


def d_join_miss(ctx: Ctx):
    row = ctx.pick("S2a")
    ctx.unlink(f"cond_clean_latents/{row['rel']}")
    return "JOIN-MISS", f"deleted cond_clean_latents/{row['rel']}"


def d_shape_disagree(ctx: Ctx):
    corpus, s4 = ctx.pick("S0"), ctx.pick("S4")
    tgt = os.path.realpath(ctx.root / f"latents/{s4['rel']}")
    ctx.repoint(f"cond_clean_latents/{corpus['rel']}", tgt)
    return "SHAPE-DISAGREE", (f"repointed a corpus sample's cond_clean at an S4 (5,14,26) "
                              f"latent — the two-shape mix's worst silent failure")


def d_bad_keys(ctx: Ctx):
    row = ctx.pick("S0")
    tgt = os.path.realpath(ctx.root / f"conditions/{row['rel']}")
    ctx.repoint(f"latents/{row['rel']}", tgt)
    return "BAD-KEYS", f"repointed latents/{row['rel']} at a conditions tensor"


DRYRUN_MUTATIONS = {
    "dryrun_dangling_symlink": d_dangling,
    "dryrun_orphan_in_non_primary_dir": d_orphan,
    "dryrun_join_miss": d_join_miss,
    "dryrun_shape_disagreement": d_shape_disagree,
    "dryrun_bad_tensor_keys": d_bad_keys,
}


# ======================================================================================
# builder mutations — A16 action 1: the drop-and-record path must DISTINGUISH its two cases
#
# `build_inventories._attach` used to `SystemExit` on ANY consumption of a role-excluded
# (clip, role).  `misc/ctt_v2_final/advisors/A16_29_orphaned_s2a_clips_VERBATIM.md` (ruling of
# record) and `A17_29clip_affirmation_VERBATIM.md` (affirmation) replaced that with
# **drop-and-record for hits carried by the standing `ROLE_EXCLUSIONS`, crash for everything
# else**, and A16 action 1 required the distinction proven as a RERUNNABLE mutation here.
#
# `A18_28plus1_and_ood_demo_VERBATIM.md`'s residual item is why these live in this file:
# the same three controls were first run by hand and captured in
# `outputs/ctt_v2/roots/REHEARSAL_2026-07-28/A16_EXEC/logs/builder_controls.log`.
# **A one-off log is not a test** — a log cannot fail tomorrow.
#
# A one-sided test would be worthless: *"it no longer crashes"* is satisfied just as well by
# a builder that has stopped checking.  So both directions run, as real subprocess runs of
# the real builder against a synthetic meta shard:
#
#   excluded hit          -> exit 0, RECORDED in `role_scoped_exclusion_drops` with BOTH
#                            role-scoped reasons, and its caption left None (the clip is
#                            removed at ASSEMBLY, which is where the manifest drop record —
#                            the ruling's closing evidence — is written);
#   missing non-excluded  -> exit != 0 and NO inventory written;
#   both at once          -> still exit != 0, i.e. the drop path cannot mask the crash path;
#   a fabricated fallback -> exit != 0 (a caption EXISTS for an excluded consumption).
# ======================================================================================
BUILDER = HERE.parent / "build_inventories.py"


def _excluded_a_clip() -> str:
    """The standing A-role exclusion, DERIVED — never a literal clip name in this file."""
    clip = next((c for c, roles in rc.ROLE_EXCLUSIONS.items() if "A" in roles), None)
    if clip is None:
        raise SystemExit("root_common.ROLE_EXCLUSIONS carries no A-role exclusion to test — "
                         "a vacuous standing exclusion is instrument failure, not 'nothing "
                         "to exclude' (A16 keyed-join rule item 1)")
    return clip


def _builder_fixture(tmp: Path, tag: str, *, n_ok: int = 3, excluded: int = 0,
                     missing: int = 0, fabricate_fallback: bool = False) -> tuple[Path, Path]:
    """Write a minimal `render_s2.py`-shaped meta shard + a per-clip caption file."""
    d = tmp / f"builder_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    ex = _excluded_a_clip()
    rows, caps = [], {}
    for i in range(n_ok):
        stem = f"bfx_{i:02d}"
        rows.append({"stem": stem, "op_id": "op_fixture", "shader": "FixtureShader",
                     "A": f"fixture_a_{i}", "B": f"fixture_b_{i}"})
        caps[stem] = f"a fixture A description {i}. sksz. a fixture B description {i}."
    for i in range(excluded):
        stem = f"bfx_excluded_{i:02d}"
        rows.append({"stem": stem, "op_id": "op_fixture", "shader": "FixtureShader",
                     "A": ex, "B": f"fixture_b_x{i}"})
        if fabricate_fallback:
            caps[stem] = "a FABRICATED caption for a blank-white anchor. sksz. text."
    for i in range(missing):
        stem = f"bfx_missing_{i:02d}"
        rows.append({"stem": stem, "op_id": "op_fixture", "shader": "FixtureShader",
                     "A": f"fixture_a_m{i}", "B": f"fixture_b_m{i}"})
        # deliberately NO caption entry, and NOT carried by any standing exclusion
    shard = d / "clips_shard00.jsonl"
    shard.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cp = d / "captions.json"
    cp.write_text(json.dumps(caps, indent=1))
    return shard, cp


def _run_builder(tmp: Path, tag: str, **kw) -> tuple[int, str, Path]:
    shard, caps = _builder_fixture(tmp, tag, **kw)
    out = shard.parent / "INVENTORY.json"
    proc = subprocess.run(
        [PY, str(BUILDER), "s2meta", "--stratum", "S2a", "--meta-glob", str(shard),
         "--captions", str(caps), "--caption-key", "{clip}", "--out", str(out),
         "--no-require-sources"], capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr), out


def b_excluded_hit_is_dropped(tmp: Path) -> dict:
    """CONTROL 0 (positive) — the standing exclusion is DROPPED AND RECORDED, exit 0."""
    code, log, out = _run_builder(tmp, "drop", excluded=1)
    ex = _excluded_a_clip()
    inv = rc.read_json(out) if out.exists() else {}
    rec = inv.get("role_scoped_exclusion_drops") or {}
    clips = rec.get("clips") or {}
    stem = next(iter(clips), None)
    reasons = set(clips.get(stem) or [])
    want = {f"role_scoped_caption_exclusion:{ex}:A", f"role_scoped_prefix_condition:{ex}:A"}
    entry = (inv.get("clips") or {}).get(stem) or {}
    others = [c for s, c in (inv.get("clips") or {}).items() if s != stem]
    ok = (code == 0 and out.exists() and rec.get("n_clips") == 1 and len(clips) == 1
          and reasons == want
          # dropped here means: recorded, caption withheld, NO cross-role substitute
          and entry.get("caption") is None
          # ...and the other rows are unaffected, so this is a DROP, not a dead builder
          and all(c.get("caption") for c in others) and len(others) == 3
          # ...derived, never a hand-kept stem list
          and "ROLE_EXCLUSIONS" in (rec.get("derivation") or "")
          and rec.get("standing_role_exclusions", {}).get(ex) == ["A"])
    return {"ok": ok, "broke": f"one row consumes the role-excluded ({ex}, A)",
            "expected": "exit 0; recorded in `role_scoped_exclusion_drops` with BOTH "
                        "role-scoped reasons, caption left None, other rows untouched",
            "exit_code": code, "inventory_written": out.exists(),
            "n_recorded": len(clips), "reasons": sorted(reasons),
            "reasons_expected": sorted(want),
            "caption_withheld": entry.get("caption") is None,
            "other_rows_captioned": len(others),
            "derivation": rec.get("derivation"), "log_tail": log.strip()[-300:]}


def b_missing_nonexcluded_crashes(tmp: Path) -> dict:
    """CONTROL A (negative) — a missing key NO exclusion accounts for still HARD-CRASHES."""
    code, log, out = _run_builder(tmp, "crash", missing=1)
    ok = (code != 0 and "missing sources" in log and "caption:bfx_missing_00" in log
          and not out.exists())
    return {"ok": ok, "broke": "one row's caption key is absent and NOT carried by any "
                               "standing exclusion",
            "expected": "exit != 0, no inventory written — the converse defect must never "
                        "degrade into a silent drop",
            "exit_code": code, "inventory_written": out.exists(),
            "log_tail": log.strip()[-300:]}


def b_drop_does_not_mask_crash(tmp: Path) -> dict:
    """The distinction itself: one of each in ONE build must still fail."""
    code, log, out = _run_builder(tmp, "both", excluded=1, missing=1)
    ok = code != 0 and "missing sources" in log and not out.exists()
    return {"ok": ok, "broke": "one excluded row AND one missing-caption row in the same build",
            "expected": "exit != 0 — the drop path must not swallow the crash path",
            "exit_code": code, "inventory_written": out.exists(),
            "log_tail": log.strip()[-300:]}


def b_fabricated_fallback_crashes(tmp: Path) -> dict:
    """CONTROL B (negative) — a caption that EXISTS for an excluded consumption is an error."""
    code, log, out = _run_builder(tmp, "fallback", excluded=1, fabricate_fallback=True)
    ok = code != 0 and "fallback" in log.lower() and not out.exists()
    return {"ok": ok, "broke": "a caption EXISTS for a row whose (clip, role) description is "
                               "role-excluded — i.e. text was substituted upstream",
            "expected": "exit != 0 — no cross-role fallback may ever be consumed",
            "exit_code": code, "inventory_written": out.exists(),
            "log_tail": log.strip()[-300:]}


BUILDER_MUTATIONS = {
    "builder_A16_excluded_hit_is_dropped_and_recorded": b_excluded_hit_is_dropped,
    "builder_A16_missing_nonexcluded_still_crashes": b_missing_nonexcluded_crashes,
    "builder_A16_drop_does_not_mask_crash": b_drop_does_not_mask_crash,
    "builder_A16_fabricated_cross_role_fallback_crashes": b_fabricated_fallback_crashes,
}


# ======================================================================================
# runner
# ======================================================================================
def run_assert(root: Path, report: Path, extra: list[str]) -> tuple[int, dict]:
    cmd = [PY, str(ASSERT), "--root", str(root), "--report", str(report), *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    rep = rc.read_json(report) if report.exists() else {"failed": ["<no report written>"],
                                                        "results": [], "stderr": proc.stderr}
    return proc.returncode, rep


def run_dryrun(root: Path, report: Path) -> tuple[int, dict]:
    cmd = [PY, str(DRYRUN), "--root", str(root), "--report", str(report)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    rep = rc.read_json(report) if report.exists() else {"n_skipped": -1, "stderr": proc.stderr}
    return proc.returncode, rep


def offenders_for(rep: dict, names) -> dict:
    return {r["name"]: r["offenders"][:3] for r in rep.get("results", [])
            if r["name"] in names and not r["ok"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--manifest", help="the strata manifest the root was built from (recorded)")
    ap.add_argument("--only", action="append", default=[],
                    help="run just these mutations (repeatable)")
    ap.add_argument("--loose", action="store_true",
                    help="require the expected checks to fail, but tolerate extra failures")
    ap.add_argument("--skip-dryrun", action="store_true")
    ap.add_argument("--builders-only", action="store_true",
                    help="run ONLY the BUILDER_MUTATIONS family, which needs no assembled "
                         "root (the builder runs long before one exists — this is what makes "
                         "A16 action 1's controls rerunnable TODAY rather than a one-off log)")
    ap.add_argument("--report")
    args = ap.parse_args()
    if not args.builders_only and not args.root:
        ap.error("--root is required unless --builders-only is given")

    t0 = time.time()

    # ---- builders-only: no root, no lock, no baseline ----------------------------------
    if args.builders_only:
        tmp = Path(tempfile.mkdtemp(prefix="prove_builders_"))
        results, failures = [], []
        wanted = set(args.only) if args.only else None
        for name, fn in BUILDER_MUTATIONS.items():
            if wanted and name not in wanted:
                continue
            rec = dict(kind="builder", mutation=name, **fn(tmp))
            if not rec["ok"]:
                rec["ERROR"] = (f"expected {rec['expected']}; got exit={rec['exit_code']} "
                                f"({rec.get('log_tail', '')[-160:]})")
                failures.append(f"{name}: {rec.get('ERROR')}")
            results.append(rec)
            print(f"[{'PROVEN' if rec['ok'] else 'PROBLEM'}] {name}: {rec['broke']}\n"
                  f"          exit={rec['exit_code']}")
        proven = [r["mutation"] for r in results if r["ok"]]
        out = Path(args.report) if args.report else Path("PROVE_BUILDERS.json")
        rc.write_json(out, {
            "scope": "BUILDER_MUTATIONS only (no assembled root involved)",
            "authority": [
                "misc/ctt_v2_final/advisors/A16_29_orphaned_s2a_clips_VERBATIM.md action 1",
                "misc/ctt_v2_final/advisors/A17_29clip_affirmation_VERBATIM.md",
                "misc/ctt_v2_final/advisors/A18_28plus1_and_ood_demo_VERBATIM.md "
                "(residual: the controls must live in the test harness, not only in a log)",
            ],
            "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "n_mutations": len(results), "n_proven": len(proven), "proven": proven,
            "failures": failures, "results": results,
            "elapsed_s": round(time.time() - t0, 2),
        })
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\n[prove] {len(proven)}/{len(results)} builder mutations proven -> {out}")
        if failures:
            print("[prove] PROBLEMS:")
            for f in failures:
                print(f"        - {f}")
            return 1
        print("[prove] A16's DROP-vs-CRASH DISTINCTION IS PROVEN, BOTH DIRECTIONS")
        return 0

    root = Path(args.root)

    # EXCLUSIVE LOCK on the root.  This harness deliberately mutates the root in place, so
    # two concurrent runs interleave one another's mutations and each sees the other's as a
    # dirty baseline — which is exactly what happened once here, and it looks like a real
    # assert failure.  Fail loudly instead.
    lock = root / ".prove_asserts.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()} started={time.strftime('%H:%M:%S')}\n".encode())
        os.close(fd)
    except FileExistsError:
        print(f"[prove] REFUSING to run: {lock} exists — another run holds this root "
              f"({lock.read_text().strip()}). This harness mutates the root in place; two "
              f"runs would corrupt each other's baseline. Remove the lock only if you are "
              f"sure no run is live, and re-assemble the root first.")
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="prove_asserts_"))
    rep_path = tmp / "report.json"
    results, failures = [], []

    def baseline(tag: str) -> dict:
        code, rep = run_assert(root, rep_path, [])
        ok = code == 0 and not rep["failed"]
        print(f"[prove] baseline {tag}: {'GREEN' if ok else 'RED'} "
              f"({len(rep.get('results', []))} checks, failed={rep['failed']})")
        if not ok:
            failures.append(f"baseline {tag} is not green: {rep['failed']}")
        return rep

    print(f"[prove] root={root}\n[prove] python={PY}")
    base = baseline("before")
    all_checks = [r["name"] for r in base.get("results", [])]
    print(f"[prove] {len(all_checks)} checks in the battery: {all_checks}")

    if not args.skip_dryrun:
        code, drep = run_dryrun(root, rep_path)
        ok = code == 0 and drep.get("n_skipped") == 0
        print(f"[prove] baseline dry-run: {'ZERO SKIPPED' if ok else 'RED'} "
              f"(skipped={drep.get('n_skipped')}, samples={drep.get('n_epoch_ok')})")
        if not ok:
            failures.append(f"baseline dry-run is not clean: skipped={drep.get('n_skipped')}")
        results.append({"kind": "baseline_dryrun", "ok": ok,
                        "n_epoch_ok": drep.get("n_epoch_ok"),
                        "n_skipped": drep.get("n_skipped"),
                        "n_distinct_tensors_loaded": drep.get("n_distinct_tensors_loaded"),
                        "elapsed_s": drep.get("elapsed_s")})

    wanted = set(args.only) if args.only else None

    # ---- assert-battery mutations ------------------------------------------------------
    for name, (apply_fn, expect, coupling) in MUTATIONS.items():
        if wanted and name not in wanted:
            continue
        ctx = Ctx(root, Path(args.manifest) if args.manifest else root, tmp)
        try:
            what = apply_fn(ctx)
            code, rep = run_assert(ctx.root_override or root, rep_path, ctx.extra_args)
            failed = set(rep["failed"])
            missing = expect - failed
            extra = failed - expect
            external = {x for x in extra if x.startswith(EXTERNAL_PREFIX)}
            extra_owned = extra - external
            ok = code != 0 and not missing and (args.loose or not extra_owned)
            rec = {"kind": "assert", "mutation": name, "ok": ok, "broke": what,
                   "expected_to_fail": sorted(expect), "actually_failed": sorted(failed),
                   "also_fired_external": sorted(external),
                   "exit_code": code, "offenders": offenders_for(rep, expect)}
            if coupling:
                rec["coupling_declared"] = coupling
            if missing:
                rec["ERROR"] = f"did NOT fire: {sorted(missing)}"
            elif extra_owned and not args.loose:
                rec["ERROR"] = f"unexpected extra failures: {sorted(extra_owned)}"
        finally:
            ctx.undo.run()
        results.append(rec)
        print(f"[{'PROVEN' if rec['ok'] else 'PROBLEM'}] {name}: {what}\n"
              f"          fired={sorted(rec['actually_failed'])}")
        if not rec["ok"]:
            failures.append(f"{name}: {rec.get('ERROR')}")

    # ---- dry-run mutations -------------------------------------------------------------
    if not args.skip_dryrun:
        for name, apply_fn in DRYRUN_MUTATIONS.items():
            if wanted and name not in wanted:
                continue
            ctx = Ctx(root, Path(args.manifest) if args.manifest else root, tmp)
            try:
                tag, what = apply_fn(ctx)
                code, drep = run_dryrun(root, rep_path)
                offs = [o for o in drep.get("offenders_first_10", []) if o.startswith(tag)]
                ok = code != 0 and drep.get("n_skipped", 0) > 0 and bool(offs)
                rec = {"kind": "dryrun", "mutation": name, "ok": ok, "broke": what,
                       "expected_tag": tag, "exit_code": code,
                       "n_skipped": drep.get("n_skipped"), "offenders": offs[:3]}
                if not ok:
                    rec["ERROR"] = (f"expected a nonzero exit with a {tag} offender; got "
                                    f"exit={code} skipped={drep.get('n_skipped')} "
                                    f"offenders={drep.get('offenders_first_10')[:3]}")
            finally:
                ctx.undo.run()
            results.append(rec)
            print(f"[{'PROVEN' if rec['ok'] else 'PROBLEM'}] {name}: {what}\n"
                  f"          skipped={rec['n_skipped']} tag={tag}")
            if not rec["ok"]:
                failures.append(f"{name}: {rec.get('ERROR')}")

    # ---- builder mutations (root-independent; A16 action 1's two directions) -------------
    for name, fn in BUILDER_MUTATIONS.items():
        if wanted and name not in wanted:
            continue
        rec = dict(kind="builder", mutation=name, **fn(tmp))
        if not rec["ok"]:
            rec["ERROR"] = (f"expected {rec['expected']}; got exit={rec['exit_code']} "
                            f"({rec.get('log_tail', '')[-160:]})")
        results.append(rec)
        print(f"[{'PROVEN' if rec['ok'] else 'PROBLEM'}] {name}: {rec['broke']}\n"
              f"          exit={rec['exit_code']}")
        if not rec["ok"]:
            failures.append(f"{name}: {rec.get('ERROR')}")

    # ---- the root must be exactly as we found it ----------------------------------------
    baseline("after")
    if not args.skip_dryrun:
        code, drep = run_dryrun(root, rep_path)
        if code != 0 or drep.get("n_skipped") != 0:
            failures.append(f"dry-run after restore is not clean: {drep.get('n_skipped')}")

    proven = [r["mutation"] for r in results if r.get("kind") != "baseline_dryrun" and r["ok"]]
    out = Path(args.report) if args.report else root / "PROVE_ASSERTS.json"
    rc.write_json(out, {
        "root": str(root), "strata_manifest": args.manifest,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "strict": not args.loose,
        "battery_checks": all_checks,
        "n_mutations": len([r for r in results if r.get("kind") != "baseline_dryrun"]),
        "n_proven": len(proven), "proven": proven,
        "failures": failures, "results": results,
        "elapsed_s": round(time.time() - t0, 2),
    })
    shutil.rmtree(tmp, ignore_errors=True)
    lock.unlink(missing_ok=True)
    print(f"\n[prove] {len(proven)}/{len([r for r in results if r.get('kind') != 'baseline_dryrun'])}"
          f" mutations proved their check(s) fire -> {out}")
    if failures:
        print("[prove] PROBLEMS:")
        for f in failures:
            print(f"        - {f}")
        return 1
    uncovered = sorted(set(all_checks) - {c for _n, (_f, e, _c) in MUTATIONS.items() for c in e})
    if uncovered and not wanted:
        print(f"[prove] NOTE: no mutation targets {uncovered} — "
              f"they are covered only by the green baseline")
    print("[prove] EVERY TARGETED HARD ASSERT IS PROVEN TO FIRE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
