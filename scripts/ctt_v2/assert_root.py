"""CTT v2 — the HARD pre-launch assert battery (A5 RULING 9).

Every check here is a hard failure.  There is no warnings-only mode, no `--skip`, no
severity flag: the script exits non-zero if a single check fails, and the launch scripts
are expected to gate on that exit code.

    A1  set-equality of RELATIVE PATHS across all 5 root dirs (A3-F8.1 — counts are NOT
        sufficient; the trainer joins by relative path and silently drops mismatches)
    A2  inventory integrity — the inventories the root was built from are unchanged
    A3  realised mix within +-0.5 pp of intended, COUNTED from the assembled root.
        Two clauses:
        A3   every stratum's counted share is within tolerance of its target — and for a
             pro-rata member (S2a/S2b) the target is itself DERIVED from the assembled
             counts, not a declared number (A12);
        A3b  the members of a pro-rata group carry the SAME replica multiplier, counted
             from the replica dirs on disk.  This is A1b's "uniform per-sample weight
             within S2 ... no extra reweighting knob" as an exact integer identity —
             strictly stronger than a share tolerance on the halves, since differential
             duplication is the only mechanism that can force a share the base counts do
             not already produce.
    A4  every caption contains ` sksz.` exactly once; the outcome marker is absent;
        zero Tier-1 leak strings
    A5  S1/S2 endpoints n {eval endpoints, zs audited endpoints, the 42 test clips} = 0
        (classes resolved with eval_ladder/prompts.py:clip_class(), never by name-splitting)
    A6  the 8 pre-registered S2a inline-OOD ops are absent from the root
    A7  the 10 HOLDOUT_S2 shader families are absent from the root
    A8  the 120 reserved union-pool clips are absent from the root
    A9  the S0 zero-shot classes are absent from the root
    A10 the Day-0 copy-gate admissibility check is recorded as PASSED (RULING 1 makes it
        a training blocker; absent file or non-PASS verdict => FAIL)
    A11 the TWO SHAPES (A9 §3/§5): every shape is ruled, one shape per stratum, the
        declared set == the set on disk, both shapes present iff S4 is in the mix, and the
        mask store carries exactly the (shape, sidedness) combinations in use.  Delegated
        to an external `check_shapes` module if one exists (see `rc.shape_assert`).
    A12 no assembled caption draws on an excluded (clip, role) description
    A13 no assembled sample's PREFIX-CONDITION source (endpoint_a) is a role-excluded clip

    A12/A13 are the two consumption channels of A10's role-scoped exclusion — the caption
    store and the conditioning path.  A10's standing rule is that defects are dispositioned
    at the unit of CONSUMPTION, (clip, role), not the unit of storage; that is the whole
    mechanism by which `openvid_T1MiFx98l3g_0_50to156` keeps its 37 rendered S2b rows while
    its blank-white A-anchor can never reach a prefix or a caption.  Both checks fail if
    the exclusion set is empty, because a vacuous exclusion is the failure mode.

Every check has been PROVEN TO FIRE by `tests/prove_asserts.py`, which breaks one
invariant at a time on a real assembled fixture root and requires exactly the intended
check(s) to fail.  An assert that has never failed is not known to work.

    python scripts/ctt_v2/assert_root.py --root <root>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import root_common as rc  # noqa: E402

MAX_SHOW = 10


class Battery:
    def __init__(self):
        self.results: list[dict] = []

    def check(self, name: str, ok: bool, detail: str = "", offenders=None) -> bool:
        rec = {"name": name, "ok": bool(ok), "detail": detail,
               "offenders": list(offenders or [])[:200],
               "n_offenders": len(offenders or [])}
        self.results.append(rec)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}: {detail}")
        if not ok and offenders:
            for o in list(offenders)[:MAX_SHOW]:
                print(f"        - {o}")
            if len(offenders) > MAX_SHOW:
                print(f"        ... and {len(offenders) - MAX_SHOW} more")
        return ok

    @property
    def failed(self) -> list[str]:
        return [r["name"] for r in self.results if not r["ok"]]


# --------------------------------------------------------------------------------------
def parse_rel(rel: str) -> tuple[str, int, str, str]:
    """'S2a_r03/BookFlip_x/a__ref_b.pt' -> (stratum, rep, group, filename)."""
    parts = rel.split("/")
    if len(parts) != 3:
        raise ValueError(f"unexpected relative path depth: {rel!r}")
    stratum, rep = rc.parse_replica(rel)
    return stratum, rep, parts[1], parts[2]


def split_name(name: str) -> tuple[str, str]:
    stem = name[:-3] if name.endswith(".pt") else name
    if "__ref_" not in stem:
        raise ValueError(f"sample name is not <target>__ref_<reference>: {name!r}")
    tgt, ref = stem.split("__ref_", 1)
    return tgt, ref


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--copy-gate-verdict", help="override the RULING-1 verdict document path")
    ap.add_argument("--prereg-inline-ood", help="override the inline-OOD pre-registration file")
    ap.add_argument("--pool-drops", help="override the M3 pool-drop adjudication file")
    ap.add_argument("--override-role-exclusions", metavar="JSON",
                    help="TEST ONLY — replace the standing root_common.ROLE_EXCLUSIONS with "
                         "{clip: [roles]} from this file.  Recorded in the report as "
                         "TEST_ONLY_OVERRIDES so a real run can never hide it; exists so "
                         "tests/prove_asserts.py can prove the vacuity branch of A13 fires.")
    ap.add_argument("--report", help="where to write ASSERT_REPORT.json (default: <root>/)")
    args = ap.parse_args()

    if args.override_role_exclusions:
        rc.ROLE_EXCLUSIONS = {c: tuple(r) for c, r in
                              rc.read_json(args.override_role_exclusions).items()}
        print(f"[assert] ⚠ TEST-ONLY OVERRIDE: ROLE_EXCLUSIONS := {rc.ROLE_EXCLUSIONS} "
              f"(from {args.override_role_exclusions})")

    t0 = time.time()
    root = Path(args.root)
    b = Battery()

    mpath = root / "ROOT_MANIFEST.json"
    if not mpath.exists():
        print(f"[FAIL] root_manifest: {mpath} is absent — the root was never assembled")
        return 2
    man = rc.read_json(mpath)
    print(f"[assert] root={root}\n[assert] manifest={mpath} "
          f"(strata {man['strata_present']}, S4 in mix: {man['s4_in_mix']})")

    # ---- A1 set-equality of relative paths across the 5 dirs --------------------------
    sets = {}
    for sub in rc.ROOT_DIRS:
        if not (root / sub).is_dir():
            b.check("A1_set_equality_5_dirs", False, f"root dir {sub!r} does not exist")
            sets = None
            break
        sets[sub] = rc.rel_paths(root, sub)
    if sets:
        ref_name = "latents"
        ref = sets[ref_name]
        diffs = []
        for sub, s in sets.items():
            if sub == ref_name:
                continue
            sym = s ^ ref
            for r in sorted(sym)[:200]:
                diffs.append(f"{sub}: {'only-in-' + sub if r in s else 'missing-from-' + sub} {r}")
        b.check("A1_set_equality_5_dirs", not diffs,
                "identical relative-path sets in all 5 dirs "
                f"({len(ref)} each)" if not diffs else
                f"symmetric difference across dirs: {len(diffs)} entries "
                f"(counts {{{', '.join(f'{k}:{len(v)}' for k, v in sets.items())}}})",
                diffs)
        rels = sorted(ref)
    else:
        rels = []

    if not rels:
        b.check("A1b_root_nonempty", False, "no samples found under latents/")
        _finish(b, root, args, t0)
        return 1
    b.check("A1b_root_nonempty", True, f"{len(rels)} samples")

    # ---- A2 inventory integrity -------------------------------------------------------
    invs, bad = {}, []
    for s, meta in man["inventories"].items():
        p = Path(meta["path"])
        if not p.exists():
            bad.append(f"{s}: inventory missing at {p}")
            continue
        sha = rc.sha256_file(p)
        if sha != meta["sha256"]:
            bad.append(f"{s}: inventory sha256 changed since assembly ({p})")
            continue
        invs[s] = rc.read_json(p)
    b.check("A2_inventory_integrity", not bad,
            f"{len(invs)} inventories match their recorded sha256" if not bad
            else "inventories changed since assembly — the root's provenance is broken", bad)

    # ---- decode the root ---------------------------------------------------------------
    parsed, malformed = [], []
    for rel in rels:
        try:
            stratum, rep, group, name = parse_rel(rel)
            tgt, ref_clip = split_name(name)
            parsed.append((rel, stratum, rep, group, tgt, ref_clip))
        except ValueError as exc:
            malformed.append(f"{rel}: {exc}")
    b.check("A2b_path_scheme", not malformed,
            "every relative path is <stratum>_r<NN>/<group>/<target>__ref_<reference>.pt"
            if not malformed else "malformed relative paths", malformed)

    # ---- A3 realised mix, COUNTED from the root ---------------------------------------
    counts = Counter(p[1] for p in parsed)
    total = sum(counts.values())
    intended = man["weights"]["intended_pct"]
    tol = float(man["weights"]["tolerance_pp"])
    rows, off = [], []
    for s in sorted(set(intended) | set(counts)):
        real = 100.0 * counts.get(s, 0) / total
        want = float(intended.get(s, 0.0))
        dev = real - want
        rows.append(f"{s} counted {counts.get(s, 0)} = {real:.3f}% (intended {want:.3f}%, "
                    f"{dev:+.3f} pp)")
        if abs(dev) > tol:
            off.append(rows[-1])
    b.check("A3_realized_mix", not off,
            f"all strata within +-{tol} pp | " + " | ".join(rows), off)

    # ---- A3b the PRO-RATA split's multipliers are EQUAL, COUNTED from the root ---------
    # A12's machine-checkable form of A1b's "uniform per-sample weight within S2 ... no
    # extra reweighting knob".  Unequal replica multipliers on the two S2 halves ARE that
    # knob: differential duplication is the only way to force a share the base counts do
    # not already produce, and it is what a forced-equal 34.5/34.5 would have required over
    # counts of 22,731 vs 23,577.  Stronger and simpler than a share tolerance on the
    # halves: it is an exact integer identity, so it cannot be satisfied "within tolerance".
    #
    # Counted from the ROOT — the distinct replica directories actually on disk — never
    # read back from the manifest the assembler wrote; the manifest's own claim is then
    # cross-checked against that count, so a manifest that lies is also a failure.
    # The pro-rata grouping comes from the RULING in root_common, not from the manifest,
    # so a manifest that simply omits the group cannot make this check vacuous.
    reps_on_disk: dict[str, set] = defaultdict(set)
    for _p in parsed:
        reps_on_disk[_p[1]].add(_p[2])
    declared_reps = (man.get("counts") or {}).get("replicas") or {}
    manifest_groups = {k: sorted(v) for k, v in
                       ((man.get("weights") or {}).get("prorata_groups") or {}).items()}
    tie_off, tie_rows, checked = [], [], []
    for gname, gmembers in sorted(rc.PRORATA_GROUPS.items()):
        here = [m for m in gmembers if reps_on_disk.get(m)]
        for m in gmembers:
            if m in man["strata_present"] and m not in here:
                tie_off.append(f"{gname}: {m} is declared present but contributes no sample "
                               f"to the root — the pro-rata split it is a member of is not "
                               f"realised")
        if len(here) < 2:
            tie_rows.append(f"{gname}: {here or 'no'} member(s) in the root, split not "
                            f"exercised this branch")
            continue
        checked.append(gname)
        got = {m: len(reps_on_disk[m]) for m in here}
        tie_rows.append(f"{gname}: " + ", ".join(f"{m} x{got[m]} replicas" for m in here))
        if len(set(got.values())) != 1:
            tie_off.append(
                f"{gname}: replica multipliers differ ON DISK {got} — the halves are being "
                f"differentially duplicated, which is the extra reweighting knob A1b "
                f"excluded and A12 forbids. The split must come from the base counts "
                f"(A12: 'S2 total 69, split pro-rata to the assembled post-exclusion "
                f"counts'), not from unequal replication.")
        for m in here:
            d = declared_reps.get(m)
            if d is not None and int(d) != got[m]:
                tie_off.append(f"{gname}: {m} has {got[m]} replica dirs on disk but "
                               f"ROOT_MANIFEST declares {d}")
        if manifest_groups and manifest_groups.get(gname) != sorted(gmembers):
            tie_off.append(f"{gname}: ROOT_MANIFEST declares members "
                           f"{manifest_groups.get(gname)}, the ruling says {sorted(gmembers)}")
    b.check("A3b_prorata_multipliers_equal", not tie_off,
            (f"pro-rata groups exercised: {checked or 'NONE'} | " + " | ".join(tie_rows))
            if not tie_off else f"{len(tie_off)} pro-rata split problem(s)", tie_off)

    # ---- captions / endpoints, resolved through the (sha-verified) inventories ---------
    filt = rc.leak_filter()
    print(f"[assert] Tier-1 leak filter: {getattr(filt, 'source', type(filt).__name__)}")
    ex = rc.load_exclusions(
        Path(args.prereg_inline_ood) if args.prereg_inline_ood else None,
        Path(args.pool_drops) if args.pool_drops else None)

    cap_off, missing_src, role_off, prefix_off = [], [], [], []
    seen_endpoints: set = set()
    seen_shaders: set = set()
    seen_classes: set = set()
    exempt_endpoints: set = set()
    spath0 = root / "SAMPLES.jsonl"
    rows_for_weights = ([json.loads(ln) for ln in spath0.read_text().splitlines() if ln.strip()]
                        if spath0.exists() else [])
    endpoint_off, reserved_off, zs_off, shader_off, ood_off = [], [], [], [], []
    seen_captions: dict[str, list[str]] = defaultdict(list)
    prompts = rc._prompts()

    for rel, stratum, rep, group, tgt, ref_clip in parsed:
        inv = invs.get(stratum)
        if inv is None:
            missing_src.append(f"{rel}: no verified inventory for stratum {stratum}")
            continue
        entry = inv["clips"].get(tgt)
        gmeta = inv["groups"].get(group)
        if entry is None or gmeta is None:
            missing_src.append(f"{rel}: target {tgt!r} / group {group!r} not in the {stratum} "
                               f"inventory")
            continue
        if rep == 0:
            cap = entry.get("caption")
            if not cap:
                cap_off.append(f"{rel}: no caption in the inventory")
            else:
                seen_captions[cap].append(rel)
        # A12 — the M3 role-scoped caption-store exclusion (A11 item 5, machine check 3).
        # Derived from POOL_DROPS_M3_ADJUDICATION.json, never a hand-kept list.
        hits = ex.caption_store_hits(
            rc.caption_sources(entry, gmeta.get("sided", "two"),
                               inv.get("kind", "synthetic_op")))
        if hits:
            role_off.append(f"{rel}: caption draws on excluded (clip, role) {hits}")
        # A13 — the CONDITIONING channel of the same (clip, role) exclusion (A10
        # `enforced_at[2]`).  The prefix anchor is built from the A endpoint's frames 0-8
        # (`streams.py:56-72`, `build_from_stream(start9)`), so endpoint_a IS the
        # prefix-condition source.  A blank-white A-anchor would be injected straight into
        # conditioning and would also corrupt the S1 mechanical gate, which compares against
        # that same start9 anchor.
        eps_list = entry.get("endpoints") or []
        if eps_list and rc.role_excluded(eps_list[0], "A"):
            prefix_off.append(f"{rel}: prefix-condition source (endpoint_a) {eps_list[0]!r} "
                              f"is role-excluded for role A")
        # A6 inline-OOD ops
        if group in ex.inline_ood_ops:
            ood_off.append(f"{rel}: group {group} is a pre-registered inline-OOD op")
        # A7 holdout shaders
        if gmeta.get("shader") and gmeta["shader"] in ex.holdout_shaders:
            shader_off.append(f"{rel}: shader {gmeta['shader']} is a HOLDOUT_S2 family")
        # A8/A5/A9 endpoints
        eps = set()
        for clip in (tgt, ref_clip):
            e = inv["clips"].get(clip, {}).get("endpoints") or []
            eps.update(e)
        hit_res = sorted(eps & ex.reserved_pool_clips)
        if hit_res:
            reserved_off.append(f"{rel}: reserved union-pool clips {hit_res}")
        if inv.get("endpoint_disjointness", True):
            hit_eval = sorted(eps & ex.eval_endpoints)
            if hit_eval:
                endpoint_off.append(f"{rel}: endpoints in the eval sets {hit_eval}")
        # positive-presence controls: accumulate the namespaces the absence asserts compare
        # against, so "= 0" can be distinguished from "compared nothing"
        seen_endpoints.update(eps)
        seen_endpoints.update({tgt, ref_clip})
        if gmeta.get("shader"):
            seen_shaders.add(gmeta["shader"])
        if gmeta.get("class"):
            seen_classes.add(gmeta["class"])
        if not inv.get("endpoint_disjointness", True):
            exempt_endpoints.update(eps)
        # A9 zs classes: group class (S0) and any corpus-resolvable endpoint
        if gmeta.get("class") and gmeta["class"] in ex.zs_classes:
            zs_off.append(f"{rel}: group class {gmeta['class']} is a zero-shot holdout class")
        for ep in eps | {tgt, ref_clip}:
            try:
                cls = prompts.clip_class(ep)
            except KeyError:
                continue
            if cls in ex.zs_classes:
                zs_off.append(f"{rel}: clip {ep} resolves to zero-shot class {cls}")

    b.check("A2c_root_resolves_to_inventories", not missing_src,
            f"every sample resolves to a verified inventory entry ({len(parsed)} samples)"
            if not missing_src else "samples the inventories cannot explain", missing_src)

    # ---- A4 captions -------------------------------------------------------------------
    for cap, where in sorted(seen_captions.items()):
        v = rc.caption_violations(cap, filt)
        if v:
            cap_off.append(f"{where[0]} (+{len(where) - 1} more): {v} :: {cap[:90]!r}")
    stored = rc.read_json(root / "CAPTIONS.json") if (root / "CAPTIONS.json").exists() else {}
    stored_set = set(stored.values())
    drift = [c[:90] for c in seen_captions if c not in stored_set]
    if drift:
        cap_off.append(f"{len(drift)} captions in the inventories are absent from "
                       f"CAPTIONS.json (first: {drift[0]!r})")
    b.check("A4_captions", not cap_off,
            f"{len(seen_captions)} distinct captions: ' sksz.' exactly once, outcome marker "
            f"absent, zero Tier-1 leak strings" if not cap_off else
            f"{len(cap_off)} caption violations", cap_off)

    # ---- A5..A9 --------------------------------------------------------------------------
    b.check("A5_endpoint_disjointness", not endpoint_off,
            f"S1/S2 endpoints n {{eval, zs-audited, 42 test}} = 0 "
            f"({len(ex.eval_endpoints)} eval-side ids checked)" if not endpoint_off
            else f"{len(endpoint_off)} samples train on eval endpoints", endpoint_off)
    # A6 is vacuous unless the 8 ops are actually frozen; a vacuous exclusion on a root
    # that contains S2a is itself a hard failure (RULING 4 lists them as excluded).
    if "S2a" in set(counts) and len(ex.inline_ood_ops) != rc.N_INLINE_OOD_OPS:
        ood_off.append(f"the root contains S2a but only {len(ex.inline_ood_ops)} inline-OOD ops "
                       f"are pre-registered (expected {rc.N_INLINE_OOD_OPS}) — the exclusion "
                       f"would be vacuous; see {rc.PREREG_INLINE_OOD}")
    # A11 item 1 spells A6 out in five clauses: file exists · n == 8 · 8 DISTINCT shaders ·
    # all 8 op_ids RESOLVE in the S2a inventory · none of their clips is in the root.  The
    # middle three are what stop a stale or hand-edited pre-registration passing quietly: a
    # file naming ops that no longer exist would otherwise exclude NOTHING and still read
    # "8 pre-registered ops", which is the vacuous-exclusion failure in a new disguise.
    prereg_path = (Path(args.prereg_inline_ood) if args.prereg_inline_ood
                   else rc.PREREG_INLINE_OOD)
    if "S2a" in set(counts):
        if not prereg_path.exists():
            ood_off.append(f"{prereg_path} is ABSENT — the 8 ops are not pre-registered")
        else:
            rec = rc.read_json(prereg_path)
            shaders = rec.get("shader_to_op") or {}
            if len(set(shaders)) != rc.N_INLINE_OOD_OPS:
                ood_off.append(f"{prereg_path}: {len(set(shaders))} distinct shaders, expected "
                               f"{rc.N_INLINE_OOD_OPS} — the ops must come from DISTINCT shaders")
            if sorted(shaders.values()) != sorted(rec.get("op_ids") or []):
                ood_off.append(f"{prereg_path}: shader_to_op and op_ids disagree")
            s2a_inv = invs.get("S2a") or {}
            unresolved = sorted(o for o in (rec.get("op_ids") or [])
                                if o not in (s2a_inv.get("groups") or {}))
            if unresolved:
                ood_off.append(f"{prereg_path}: {len(unresolved)} pre-registered op_ids do not "
                               f"resolve in the S2a inventory — the pre-registration is stale "
                               f"and excludes nothing: {unresolved[:5]}")
            sha = rec.get("source_inventory_sha256")
            src = rec.get("source_inventory")
            if sha and src and Path(src).exists() and rc.sha256_file(src) != sha:
                ood_off.append(f"{prereg_path}: the S2a inventory has CHANGED since the draw "
                               f"({src}) — re-verify the pre-registration, never re-draw it")
    b.check("A6_inline_ood_ops_absent", not ood_off,
            f"none of the {len(ex.inline_ood_ops)} pre-registered inline-OOD ops is in the root"
            if not ood_off else f"{len(ood_off)} inline-OOD problems", ood_off)
    b.check("A7_holdout_shaders_absent", not shader_off,
            f"none of the {len(ex.holdout_shaders)} HOLDOUT_S2 shader families is in the root"
            if not shader_off else f"{len(shader_off)} samples use a held-out shader", shader_off)
    b.check("A8_reserved_pool_clips_absent", not reserved_off,
            f"none of the {len(ex.reserved_pool_clips)} reserved union-pool clips is in the root"
            if not reserved_off else f"{len(reserved_off)} samples use a reserved clip",
            reserved_off)
    b.check("A9_zs_classes_absent", not zs_off,
            f"none of the {len(ex.zs_classes)} zero-shot classes is in the root"
            if not zs_off else f"{len(zs_off)} samples touch a zero-shot class", zs_off)

    # ---- A0 POSITIVE-PRESENCE CONTROLS for every absence assert --------------------------
    # Standing campaign rule (A11, the σ / S4-weight ruling): an absence assert may only
    # PASS if the instrument that would have carried the thing was positively found and
    # parsed.  A5/A7/A8/A9 all report "= 0", and every one of them reports "= 0" just as
    # happily when the two sides are in DIFFERENT NAMESPACES and the comparison could never
    # have matched anything.  That is the same failure shape as a log grep that never
    # matches, and it is silent.  These controls make it loud.
    pc = []
    if not ex.eval_endpoints:
        pc.append("A5: the eval-endpoint universe is EMPTY — the disjointness check would "
                  "compare against nothing")
    elif exempt_endpoints and not (exempt_endpoints & ex.eval_endpoints):
        pc.append(f"A5: no endpoint of any disjointness-EXEMPT stratum is in the eval "
                  f"universe. Those strata (e.g. S0) ARE the eval endpoint bank by design, "
                  f"so zero overlap means the two sides are in different id namespaces and "
                  f"A5 can never fire ({len(exempt_endpoints)} exempt ids checked)")
    shader_vocab = {p.stem for p in rc.SHADER_DIR.glob("*.glsl")}
    if not ex.holdout_shaders:
        pc.append("A7: the HOLDOUT_S2 shader list is EMPTY")
    elif shader_vocab and (miss := sorted(ex.holdout_shaders - shader_vocab)):
        pc.append(f"A7: {len(miss)} held-out shader names do not resolve to a real "
                  f"`.glsl` in {rc.SHADER_DIR} — a typo'd holdout name matches nothing "
                  f"and excludes nothing: {miss[:5]}")
    if seen_shaders and not (seen_shaders & shader_vocab):
        pc.append("A7: no shader named in the root resolves in the shader vocabulary — the "
                  "root's shader ids and the holdout list are in different namespaces")
    if not ex.reserved_pool_clips:
        pc.append("A8: the reserved union-pool clip list is EMPTY")
    else:
        pool = rc.read_json(rc.CONTENT_POOL)
        pool_ids = {r["clip_id"] for r in pool.get("training", [])} | ex.reserved_pool_clips
        if seen_endpoints and not (seen_endpoints & pool_ids):
            pc.append(f"A8: not one of the root's {len(seen_endpoints)} endpoint ids is a "
                      f"union-pool clip id — the reserved-clip check is comparing across "
                      f"namespaces and can never fire")
    corpus_classes = set(rc.read_json(rc.CORPUS_MANIFEST)["classes"])
    if not ex.zs_classes:
        pc.append("A9: the zero-shot class list is EMPTY")
    elif miss := sorted(ex.zs_classes - corpus_classes):
        pc.append(f"A9: {len(miss)} zero-shot class names are not corpus classes — they "
                  f"would match nothing: {miss[:5]}")
    if seen_classes and not (seen_classes & corpus_classes):
        pc.append("A9: no group class in the root is a corpus class — the class namespaces "
                  "do not meet, so A9 can never fire")
    b.check("A0_absence_assert_positive_controls", not pc,
            f"every absence assert compares in a live, shared namespace: eval universe "
            f"{len(ex.eval_endpoints)} ids (overlapping {len(exempt_endpoints & ex.eval_endpoints)} "
            f"exempt-stratum endpoints) · {len(ex.holdout_shaders)} holdout shaders all "
            f"resolving in {len(shader_vocab)} `.glsl` files · {len(ex.reserved_pool_clips)} "
            f"reserved pool clips in the same id space as the root's {len(seen_endpoints)} "
            f"endpoints · {len(ex.zs_classes)} zs classes all corpus classes"
            if not pc else f"{len(pc)} absence asserts could not have fired", pc)

    # ---- A14 group ids survive path-safe slugging, uniquely -------------------------------
    # A11 item 3.  S4's raw group ids are refVFX effect strings with spaces
    # (`0rb4it 360 degree orbit`), and "the trainer globs fine" is not the bar — robustness
    # across shells, rsync and future tooling is.  Two raw ids that slug to the same string
    # would silently MERGE two operator groups into one pairing ring, which is a design
    # change disguised as a path fix, so uniqueness is asserted per stratum and the raw->slug
    # mapping is recorded rather than inferred.
    slug_off, slug_tables = [], {}
    for s, inv in sorted(invs.items()):
        table, collisions = rc.slug_map(inv["groups"])
        slug_tables[s] = table
        for c in collisions:
            slug_off.append(f"{s}: slug collision {c}")
        for slug, raw in sorted(table.items()):
            if not slug:
                slug_off.append(f"{s}: group {raw!r} slugs to the empty string")
    b.check("A14_group_ids_slug_safe", not slug_off,
            "every group id slugs to a unique, non-empty, path-safe string "
            + " | ".join(f"{s} {len(t)} ids" for s, t in sorted(slug_tables.items()))
            if not slug_off else f"{len(slug_off)} slugging problems", slug_off)

    # ---- A15 nominal vs effective weights, recorded (derived disclosure) ------------------
    if rows_for_weights:
        eff = rc.effective_weights(rows_for_weights, man["counts"]["replicas"])
        s4_eff = eff["effective_pct"].get("S4")
        b.check("A15_effective_weights_recorded", True,
                "nominal (sample count) vs EFFECTIVE (loss-bearing tokens): "
                + " | ".join(f"{s} {eff['nominal_pct'][s]:.2f}% -> {eff['effective_pct'][s]:.2f}%"
                             for s in sorted(eff["nominal_pct"]))
                + (f"  [S4 effective {s4_eff:.2f}%]" if s4_eff is not None else "")
                + ". Nominal is the pre-registered quantity; effective is a derived "
                  "disclosure (A11 item 2), never a control variable.")

    # ---- A10 copy-gate admissibility ----------------------------------------------------
    ok, detail = rc.copy_gate_verdict(Path(args.copy_gate_verdict) if args.copy_gate_verdict
                                      else None)
    b.check("A10_copy_gate_admissibility_PASSED", ok, detail)

    # ---- A11 the TWO SHAPES (A9 §3/§5(iv)) ----------------------------------------------
    spath = root / "SAMPLES.jsonl"
    if not spath.exists():
        b.check("A11_two_shapes", False, f"{spath} is absent — shapes cannot be counted")
    else:
        rows = [json.loads(ln) for ln in spath.read_text().splitlines() if ln.strip()]
        missing = [r.get("rel") for r in rows if not r.get("shape") or not r.get("sided")]
        if missing:
            b.check("A11_two_shapes", False,
                    f"{len(missing)} SAMPLES.jsonl rows carry no shape/sidedness",
                    missing[:200])
        else:
            check_shapes = rc.shape_assert()
            print(f"[assert] two-shape assert: "
                  f"{getattr(check_shapes, 'source', check_shapes.__name__)}")
            for rec in check_shapes(root, man, rows):
                b.check(rec["name"], rec["ok"], rec.get("detail", ""), rec.get("offenders"))

    # ---- A12 role-scoped caption-store exclusion ----------------------------------------
    cap_prov = ex.provenance.get("caption_store_exclusions", {})
    if cap_prov.get("error"):
        role_off.insert(0, f"{cap_prov.get('file')}: {cap_prov['error']}")
    b.check("A12_role_scoped_caption_exclusion", not role_off,
            f"no assembled caption draws on an excluded (clip, role) description "
            f"({sum(len(v) for v in ex.role_scoped_captions.values())} role-scoped + "
            f"{len(ex.clip_level_captions)} clip-level exclusions enforced from "
            f"{cap_prov.get('file')})" if not role_off
            else f"{len(role_off)} samples draw on an excluded (clip, role) description",
            role_off)

    # ---- A13 role-scoped PREFIX-CONDITION exclusion (A10) --------------------------------
    if not rc.ROLE_EXCLUSIONS:
        prefix_off.insert(0, f"root_common.ROLE_EXCLUSIONS is EMPTY — the A10 role-scoped "
                             f"exclusion would be silently vacuous; see {rc.POOL_DROPS}")
    b.check("A13_role_scoped_prefix_condition", not prefix_off,
            f"no assembled sample's prefix-condition source (endpoint_a) is a role-excluded "
            f"clip ({len(rc.ROLE_EXCLUSIONS)} standing (clip, role) exclusion(s): "
            f"{ {c: list(r) for c, r in rc.ROLE_EXCLUSIONS.items()} })" if not prefix_off
            else f"{len(prefix_off)} samples condition on a role-excluded prefix",
            prefix_off)

    return _finish(b, root, args, t0)


def _finish(b: Battery, root: Path, args, t0: float) -> int:
    rep = {"root": str(root), "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "elapsed_s": round(time.time() - t0, 2),
           "n_checks": len(b.results), "failed": b.failed, "results": b.results}
    out = Path(args.report) if args.report else root / "ASSERT_REPORT.json"
    try:
        rc.write_json(out, rep)
    except OSError as exc:
        print(f"[assert] WARNING: could not write {out}: {exc}")
    n_fail = len(b.failed)
    print(f"\n[assert] {len(b.results) - n_fail}/{len(b.results)} checks passed "
          f"in {rep['elapsed_s']}s -> {out}")
    if n_fail:
        print(f"[assert] HARD FAILURES: {b.failed}")
        return 1
    print("[assert] ALL HARD PRE-LAUNCH ASSERTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
