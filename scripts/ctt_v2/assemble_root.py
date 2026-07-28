"""CTT v2 — assemble the training root from the stratum inventories (pure filesystem, no GPU).

A5 RULING 4 / RULING 9.  Three properties are structural, not conventions:

1. **The mix is realised by SYMLINK DUPLICATION, not by a sampler.**  Each stratum is
   emitted `m` times into replica directories `<stratum>_r00 .. r<m-1>`, so the realised
   ratio is a property of the root on disk and `assert_root.py` can *count* it (A3-F8.3).
   The MIX CONTRACT is **S0 15 / S1 6 / S2 total 69 / S4 10**; the S2a:S2b split is
   DERIVED pro-rata from the assembled post-exclusion base pair counts (A12), so the two
   halves are solved as one unit and necessarily share a multiplier.  No per-half weight
   literal exists anywhere; the split's inputs are frozen in `PREREG_mix_inputs.json`.
2. **Pairing is ring offset within op, k = min(3, n-1), everywhere** — the same function
   for S0 classes and S1/S2/S4 ops (RULING 4, A1b Q5).
3. **Holdouts are removed here, once**, and every removal is recorded with its reason in
   `ROOT_MANIFEST.json`.  The excluded sets: the 10 HOLDOUT_S2 shader families, the 8
   pre-registered S2a inline-OOD ops, the 120 reserved union-pool clips, the S0 zs
   classes, and (A3-F5b) any S1/S2 clip whose content endpoints touch the eval sets.

The script is idempotent: it computes the desired (path -> target) map, deletes anything
in the root that is not in it, and creates only what is missing.  Re-running is a no-op.

    python scripts/ctt_v2/assemble_root.py --manifest <strata manifest>
    python scripts/ctt_v2/assemble_root.py --init-manifest <path>   # write a default
    python scripts/ctt_v2/assemble_root.py --manifest m.json --set-present S4=true
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import root_common as rc  # noqa: E402

REPO = rc.REPO_ROOT
DEFAULT_ROOT = REPO / "outputs/ctt_v2/roots/ctt_v2_mix"

#: Does this stratum's generated media EXIST ON DISK right now?  This is the ONLY
#: hand-edited part of the default manifest — weights come from rc.STRATUM_WEIGHTS_PCT and
#: the S2a:S2b split is DERIVED from the assembled counts (A12), never written here.
#: Updated 2026-07-28 by the DATASET.md audit, which found S2b still marked absent after
#: it had rendered + passed acceptance + passed its blind audit, and S4 still marked
#: absent after A9 reinstated it.
_S_PRESENT = {
    "S0": True,    # certified root, 385 pairs
    "S1": False,   # pilot only (33 clips); full 390-clip grid gated on the credit-blocked pilot gate
    "S2a": True,   # 7,990 clips / 799 ops, accepted
    "S2b": True,   # 7,990 clips / 799 ops, accept_s2 PASS, blind audit PASS (0 consensus-BAD)
    "S4": True,    # reinstated by A9; extraction + encode in flight
}
assert set(_S_PRESENT) == set(rc.MIX_STRATA), (
    "_S_PRESENT and rc.MIX_STRATA must name exactly the same strata"
)


# --------------------------------------------------------------------------------------
def default_manifest() -> dict:
    inv = REPO / "outputs/ctt_v2/inventories"
    return {
        "schema": rc.STRATA_MANIFEST_SCHEMA,
        "authority": "A5 RULING 2/3/4 (misc/ctt_v2_final/advisors/A5_SYNTHESIS_RULING_VERBATIM.md)",
        "root": str(DEFAULT_ROOT),
        "seed": rc.SEED,
        "pairing": {"rule": rc.PAIRING_RULE, "max_refs_per_target": rc.MAX_REFS_PER_TARGET},
        "mix_tolerance_pp": rc.MIX_TOLERANCE_PP,
        "absent_policy": rc.ABSENT_POLICY,
        # ---- THE MIX CONTRACT (A9 / A11 item 3 / A12) ---------------------------------
        # STRATUM-level weights are fixed numbers; `S2` is an AGGREGATE whose 69 pp splits
        # across S2a and S2b PRO-RATA to the assembled post-exclusion base pair counts.
        # There is deliberately NO per-half number anywhere in this manifest: A12 ruled
        # that "pro-rata" is A9's instruction and "which are ~equal" was an observation of
        # pre-exclusion counts, and A1b independently forbids the differential duplication
        # that a forced-equal share would require.  The split is computed at assembly time
        # from the counts the assembler produces, and its inputs are frozen in
        # `misc/ctt_v2_final/PREREG_mix_inputs.json`.
        "stratum_weights_pct": dict(rc.STRATUM_WEIGHTS_PCT),
        "prorata_groups": {k: list(v) for k, v in rc.PRORATA_GROUPS.items()},
        "mix_contract_authority":
            "A9 §4 + A11 item 3 (S2 total 69) + A12 (split derived pro-rata to the "
            "assembled post-exclusion base pair counts; A1b: uniform per-sample weight "
            "within S2, no extra reweighting knob) — root_common.STRATUM_WEIGHTS_PCT + "
            "root_common.PRORATA_GROUPS",
        # A9 §4 "Pre-registered branches (record now, before any gate returns)", transcribed
        # in mix-contract space per A12 ("S2 total 73 / 79 / 85, split pro-rata").
        # These are TOGGLES, not code paths: `--set-present S1=false` selects a branch and
        # the branch weights come from this table, so no assembly variant is ever hardcoded.
        #
        # History worth keeping: this dict previously held `{"S1": {S0 15, S2a 42.5,
        # S2b 42.5}}` and was flagged STALE — that is A9's BOTH-absent branch sitting under
        # the S1-only key, i.e. dropping S1 alone would have silently deleted S4 from the
        # mix as well.  A9 §4 pre-registers all three branches explicitly, so transcribing
        # them is not a design decision.
        #   S1 fails its gates       -> S0 15 / S2 total 73 / S4 12
        #   S4 misses the cutoff     -> A5's 15 / 6 / S2 total 79, unchanged
        #   both                     -> A5's registered 15 / S2 total 85
        "absent_weight_overrides": {k: dict(v)
                                    for k, v in rc.ABSENT_BRANCH_WEIGHTS_PCT.items()},
        "absent_weight_overrides_authority":
            "A9 §4, ratified verbatim by A11 item 3 and restated in mix-contract space by "
            "A12 — root_common.ABSENT_BRANCH_WEIGHTS_PCT (each branch guarded to sum to "
            "100 and to remove only WHOLE pro-rata groups; every S2 total splits pro-rata)",
        # Weights are NEVER duplicated here — they are read from the single ruled source,
        # rc.STRATUM_WEIGHTS_PCT.  Literals are exactly how the mix drifted twice already:
        # this block once carried A5 RULING 2's "S4 OUT" (39.5/39.5/0.0) after A9 reversed
        # it, and then carried a forced 34.5/34.5 after A9's "split pro-rata" said not to —
        # while assert A3 validated the realized mix against those same stale literals,
        # i.e. it would have built the wrong mix and then certified it correct.  Derive,
        # don't restate.
        #
        # `present` is a statement of FACT about what exists on disk right now, and is the
        # only thing that should ever be edited by hand here:
        #   S0  — certified root, 385 pairs.
        #   S1  — pilot only (33 clips); the full 390-clip grid is gated on the pilot gate,
        #         which hard-stops on the Gemini billing outage.  FALSE until that lands.
        #   S2a — rendered + accepted.
        #   S2b — rendered (7,990/799), accept_s2 verify PASS, blind audit PASS (0 consensus-BAD).
        #   S4  — reinstated by A9; extraction + encode in flight, captions blocked.
        "strata": {
            s: {
                "present": _S_PRESENT[s],
                # the mix-contract weight this stratum draws from.  For a pro-rata member
                # `weight_pct` is deliberately null — the number does not exist until the
                # counts do.
                "weight_group": rc.weight_owner(s),
                "weight_pct": (rc.STRATUM_WEIGHTS_PCT[s]
                               if s in rc.STRATUM_WEIGHTS_PCT else None),
                "weight_rule": ("fixed (ruled)" if s in rc.STRATUM_WEIGHTS_PCT else
                                f"DERIVED pro-rata within {rc.weight_owner(s)} from the "
                                f"assembled post-exclusion base pair counts (A12)"),
                "inventory": str(inv / f"{s}.json"),
            }
            for s in rc.MIX_STRATA
        },
    }


# --------------------------------------------------------------------------------------
def apply_exclusions(inv: dict, ex: rc.Exclusions) -> tuple[dict, dict]:
    """Return (kept_groups, drop_record).  Groups/clips are removed, never silently."""
    prompts = rc._prompts()
    stratum = inv["stratum"]
    kind = inv.get("kind", "synthetic_op")
    check_endpoints = bool(inv.get("endpoint_disjointness", True))
    dropped_groups, dropped_clips = [], []
    kept: dict[str, dict] = {}

    for gid, g in inv["groups"].items():
        reasons = []
        if g.get("shader") and g["shader"] in ex.holdout_shaders:
            reasons.append(f"holdout_shader:{g['shader']}")
        if gid in ex.inline_ood_ops:
            reasons.append("inline_ood_op")
        if g.get("class") and g["class"] in ex.zs_classes:
            reasons.append(f"zs_class:{g['class']}")
        if reasons:
            dropped_groups.append({"group": gid, "stratum": stratum, "reasons": reasons,
                                   "n_clips": len(g["clips"])})
            continue

        clips = []
        for clip in g["clips"]:
            entry = inv["clips"][clip]
            creasons = []
            eps = entry.get("endpoints") or []
            hit_reserved = sorted(set(eps) & ex.reserved_pool_clips)
            if hit_reserved:
                creasons.append("reserved_pool_clip:" + ",".join(hit_reserved))
            # M3 adjudication: a (clip, role) description may be excluded without the clip
            # being dropped.  `openvid_T1MiFx98l3g_0_50to156` has a blank A-anchor and a
            # healthy B-anchor, and occupies field B in all 10 rendered clips, so role-A is
            # excluded and the clip itself is KEPT — a whole-clip drop would discard 10 good
            # rendered clips to fix a defect that cannot manifest in the role it occupies.
            hit_caps = ex.caption_store_hits(
                rc.caption_sources(entry, g.get("sided", "two"), kind))
            if hit_caps:
                creasons.append("role_scoped_caption_exclusion:" + ",".join(hit_caps))
            # ...and the CONDITIONING channel of the same (clip, role) exclusion: the prefix
            # anchor is the A endpoint's frames 0-8, so endpoint_a is the prefix-condition
            # source (A10 `enforced_at[2]`).
            if eps and rc.role_excluded(eps[0], "A"):
                creasons.append(f"role_scoped_prefix_condition:{eps[0]}:A")
            if check_endpoints:
                hit_eval = sorted(set(eps) & ex.eval_endpoints)
                if hit_eval:
                    creasons.append("eval_endpoint:" + ",".join(hit_eval))
                for ep in eps:
                    try:
                        cls = prompts.clip_class(ep)
                    except KeyError:
                        continue           # pool/bank ids are not corpus clips
                    if cls in ex.zs_classes:
                        creasons.append(f"zs_class_endpoint:{ep}({cls})")
            if creasons:
                dropped_clips.append({"clip": clip, "group": gid, "stratum": stratum,
                                      "reasons": creasons})
                continue
            clips.append(clip)

        if len(clips) < 2:
            dropped_groups.append({"group": gid, "stratum": stratum,
                                   "reasons": [f"fewer_than_2_trainable_clips:{len(clips)}"],
                                   "n_clips": len(clips)})
            continue
        kept[gid] = dict(g, clips=sorted(clips))

    return kept, {"dropped_groups": dropped_groups, "dropped_clips": dropped_clips}


def build_samples(inv: dict, groups: dict, max_refs: int) -> list[dict]:
    out = []
    for gid, g in sorted(groups.items()):
        for tgt, ref in rc.ring_pairs(g["clips"], max_refs):
            out.append({"stratum": inv["stratum"], "group": gid, "target": tgt,
                        "reference": ref, "sided": g["sided"],
                        "name": f"{tgt}__ref_{ref}.pt"})
    return out


# --------------------------------------------------------------------------------------
class ShapeCache:
    """(F, H, W) of a latent tensor, cached by (realpath, size, mtime)."""

    def __init__(self, path: Path):
        self.path = path
        self.data = rc.read_json(path) if path.exists() else {}
        self.dirty = False

    def get(self, src: Path) -> tuple[int, int, int]:
        st = os.stat(src)
        key = f"{os.path.realpath(src)}|{st.st_size}|{int(st.st_mtime)}"
        if key not in self.data:
            import torch  # noqa: PLC0415

            d = torch.load(src, map_location="cpu", weights_only=True)
            self.data[key] = [int(d["num_frames"]), int(d["height"]), int(d["width"])]
            self.dirty = True
        return tuple(self.data[key])

    def save(self) -> None:
        if self.dirty:
            rc.write_json(self.path, self.data)


def mask_store_path(root: Path, f: int, h: int, w: int, sided: str) -> Path:
    return root / "_mask_store" / f"f{f}_h{h}_w{w}_{sided}sided.pt"


def ensure_mask(path: Path, f: int, h: int, w: int, sided: str) -> None:
    """mask = f(conditioning): [0:2]=1 always (prefix anchor); [-1]=1 iff two-sided."""
    if path.exists():
        return
    import torch  # noqa: PLC0415

    m = torch.zeros(f, h, w)
    m[:2] = 1.0
    if sided == "two":
        m[-1] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"mask": m}, tmp)
    tmp.replace(path)


# --------------------------------------------------------------------------------------
def materialize(root: Path, desired: dict[str, str], prune: bool = True,
                workers: int = 16) -> dict:
    """Make the root exactly `desired` ({rel path under root: symlink target}). Idempotent."""
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    created = replaced = kept = removed = 0
    existing: set = set()
    with ThreadPoolExecutor(max_workers=len(rc.ROOT_DIRS)) as pool:
        for sub, found in zip(rc.ROOT_DIRS,
                              pool.map(lambda s: list((root / s).glob("**/*.pt"))
                                       if (root / s).exists() else [], rc.ROOT_DIRS)):
            existing.update(str(p.relative_to(root)) for p in found)

    todo = []
    for rel, target in desired.items():
        if rel in existing:
            p = root / rel
            try:
                if p.is_symlink() and os.readlink(p) == target:
                    kept += 1
                    continue
            except OSError:
                pass
            p.unlink()
            replaced += 1
        else:
            created += 1
        todo.append((rel, target))

    for d in sorted({str((root / rel).parent) for rel, _ in todo}):
        os.makedirs(d, exist_ok=True)
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda rt: os.symlink(rt[1], root / rt[0]), todo))

    if prune:
        for rel in sorted(existing - set(desired)):
            (root / rel).unlink()
            removed += 1
        # sweep now-empty directories
        for sub in rc.ROOT_DIRS:
            base = root / sub
            if not base.exists():
                continue
            for d in sorted((p for p in base.glob("**/*") if p.is_dir()),
                            key=lambda p: len(p.parts), reverse=True):
                if not any(d.iterdir()):
                    d.rmdir()
    return {"created": created, "replaced": replaced, "unchanged": kept, "pruned": removed}


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest")
    ap.add_argument("--init-manifest", metavar="PATH",
                    help="write a default strata manifest and exit")
    ap.add_argument("--root", help="override the root path from the manifest")
    ap.add_argument("--set-present", action="append", default=[], metavar="S4=true",
                    help="toggle a stratum in/out without editing the manifest")
    ap.add_argument("--prereg-inline-ood", help="override the inline-OOD pre-registration file")
    ap.add_argument("--write-prereg-inline-ood", action="store_true",
                    help="derive and FREEZE the 8 inline-OOD ops if the file does not exist yet")
    ap.add_argument("--write-prereg-mix-inputs", nargs="?", const="", metavar="PATH",
                    default=None,
                    help="write the A12 frozen-inputs record for the DERIVED S2a:S2b split "
                         f"(default {rc.PREREG_MIX_INPUTS}). Works with --plan-only, which "
                         "is how it should be run: the record is written BEFORE the root is "
                         "materialised and before any training step.")
    ap.add_argument("--no-prune", action="store_true",
                    help="do not delete root entries that are no longer desired")
    ap.add_argument("--plan-only", action="store_true",
                    help="compute everything, write no symlinks and no manifest")
    args = ap.parse_args()

    if args.init_manifest:
        rc.write_json(args.init_manifest, default_manifest())
        print(f"[assemble] wrote default strata manifest -> {args.init_manifest}")
        return
    if not args.manifest:
        raise SystemExit("--manifest is required (or --init-manifest to create one)")

    t0 = time.time()
    man = rc.read_json(args.manifest)
    if man.get("schema") != rc.STRATA_MANIFEST_SCHEMA:
        raise SystemExit(f"bad manifest schema: {man.get('schema')!r}")
    for spec in args.set_present:
        k, _, v = spec.partition("=")
        if k not in man["strata"]:
            raise SystemExit(f"--set-present: unknown stratum {k!r}")
        man["strata"][k]["present"] = v.strip().lower() in ("1", "true", "yes", "on")

    root = Path(args.root or man["root"])
    max_refs = int(man["pairing"]["max_refs_per_target"])
    tol = float(man.get("mix_tolerance_pp", rc.MIX_TOLERANCE_PP))

    present = [s for s, c in man["strata"].items() if c.get("present")]
    absent = sorted(s for s in man["strata"] if s not in present)
    if not present:
        raise SystemExit("no stratum is present")
    print(f"[assemble] strata present: {present}  absent: {absent}")

    # ---- load inventories -------------------------------------------------------------
    invs = {}
    for s in present:
        p = Path(man["strata"][s]["inventory"])
        if not p.exists():
            raise SystemExit(f"[assemble] {s} is marked present but its inventory is missing: {p}")
        inv = rc.read_json(p)
        if inv.get("schema") != rc.INVENTORY_SCHEMA:
            raise SystemExit(f"[assemble] {p}: bad inventory schema {inv.get('schema')!r}")
        if inv["stratum"] != s:
            raise SystemExit(f"[assemble] {p}: stratum {inv['stratum']!r} != {s!r}")
        invs[s] = inv

    # ---- exclusions -------------------------------------------------------------------
    ex = rc.load_exclusions(args.prereg_inline_ood)
    prereg_path = Path(args.prereg_inline_ood) if args.prereg_inline_ood else rc.PREREG_INLINE_OOD
    if not ex.inline_ood_ops:
        s2a = invs.get("S2a")
        if s2a is None:
            print("[assemble] WARNING: no inline-OOD pre-registration and no S2a inventory; "
                  "the 8-op exclusion is VACUOUS this run")
        elif args.write_prereg_inline_ood:
            rec = rc.freeze_inline_ood_prereg(man["strata"]["S2a"]["inventory"], ex,
                                              out_path=prereg_path)
            ex.inline_ood_ops = set(rec["op_ids"])
            ex.provenance["inline_ood_ops"] = {"file": str(prereg_path),
                                               "status": rec["status"], "just_written": True}
            print(f"[assemble] FROZE the 8 inline-OOD ops -> {prereg_path}")
        else:
            raise SystemExit(
                f"[assemble] the 8 pre-registered S2a inline-OOD ops are not frozen yet "
                f"({prereg_path} absent). Re-run with --write-prereg-inline-ood to derive "
                f"them deterministically (seed {rc.SEED}), or point --prereg-inline-ood at "
                f"the owner-ratified file.")

    # ---- exclusions + pairing ---------------------------------------------------------
    kept_groups, drops, samples = {}, {}, {}
    for s, inv in invs.items():
        kept_groups[s], drops[s] = apply_exclusions(inv, ex)
        samples[s] = build_samples(inv, kept_groups[s], max_refs)
        print(f"[assemble] {s}: {len(kept_groups[s])}/{len(inv['groups'])} groups kept, "
              f"{len(drops[s]['dropped_clips'])} clips dropped, {len(samples[s])} base pairs")

    # ---- weights (MIX-CONTRACT space; the S2a:S2b split is DERIVED — A12) --------------
    # Stratum-level weights are fixed ruled numbers.  `S2` is one aggregate here and stays
    # one aggregate through the solver, which is what makes S2a and S2b share a multiplier
    # by construction rather than by assertion.
    stratum_weights = man.get("stratum_weights_pct")
    if stratum_weights is None:
        raise SystemExit(
            f"{args.manifest}: no `stratum_weights_pct` block. This manifest predates A12 "
            f"(it declares a per-half `weight_pct` for S2a/S2b, which A12 abolished). "
            f"Regenerate it: assemble_root.py --init-manifest <path>")
    groups = {k: tuple(v) for k, v in (man.get("prorata_groups")
                                       or rc.PRORATA_GROUPS).items()}
    if groups != {k: tuple(v) for k, v in rc.PRORATA_GROUPS.items()}:
        raise SystemExit(f"{args.manifest}: prorata_groups {groups} disagree with the ruled "
                         f"root_common.PRORATA_GROUPS {rc.PRORATA_GROUPS}")
    stale = sorted(s for s in man["strata"]
                   if man["strata"][s].get("weight_pct") is not None
                   and rc.weight_owner(s) != s)
    if stale:
        raise SystemExit(
            f"{args.manifest}: {stale} declare a literal `weight_pct` but are members of a "
            f"pro-rata group — A12 forbids a per-half literal, the split is derived from "
            f"the assembled counts. Regenerate the manifest.")

    #: the mix-contract weights that own at least one PRESENT stratum
    present_weights = sorted({rc.weight_owner(s) for s in present})
    overrides = man.get("absent_weight_overrides") or {}
    key = ",".join(absent)
    weight_note = "as-declared"
    branch_key = None
    intended = {n: float(stratum_weights[n]) for n in present_weights}
    if absent and key in overrides:
        ov = overrides[key]
        if sorted(ov) != present_weights:
            raise SystemExit(f"absent_weight_overrides[{key!r}] covers {sorted(ov)}, the "
                             f"present strata draw on weights {present_weights}")
        if abs(sum(float(v) for v in ov.values()) - 100.0) > 1e-9:
            raise SystemExit(f"absent_weight_overrides[{key!r}] sums to "
                             f"{sum(float(v) for v in ov.values())}, not 100")
        intended = {n: float(ov[n]) for n in present_weights}
        weight_note = f"pre-registered branch override for absent={key}"
        branch_key = key
    elif absent and abs(sum(intended.values()) - 100.0) > 1e-9:
        weight_note = (f"proportional renormalisation ({rc.ABSENT_POLICY}, absent={key}) "
                       f"— no pre-registered branch for this absent set")

    base_counts = {s: len(samples[s]) for s in present}
    mix = rc.solve_multipliers(base_counts, intended, tol_pp=tol, groups=groups)
    print(f"[assemble] mix weights ({weight_note}): "
          + ", ".join(f"{n} {intended[n]:g}" for n in present_weights))
    for gname, d in sorted(mix["prorata_split"].items()):
        print(f"[assemble] {gname} {d['aggregate_weight_pct']:.4f} pp split PRO-RATA to "
              f"{d['base_counts']} -> "
              + ", ".join(f"{m} {v:.4f}" for m, v in sorted(d["derived_weight_pct"].items())))
    print(f"[assemble] mix: multipliers={mix['multipliers']} total={mix['total']} "
          f"max_dev={mix['max_deviation_pp']:.4f} pp (tol {tol})")

    if args.write_prereg_mix_inputs is not None:
        out = Path(args.write_prereg_mix_inputs or rc.PREREG_MIX_INPUTS)
        rc.freeze_mix_inputs_prereg(
            strata_manifest_path=args.manifest, manifest=man, inventories=invs,
            exclusions=ex, drops=drops, base_counts=base_counts, mix=mix,
            weight_note=weight_note, branch_key=branch_key, present=present, absent=absent,
            out_path=out)
        print(f"[assemble] FROZE the A12 mix inputs -> {out}")

    # ---- desired filesystem -----------------------------------------------------------
    shapes = ShapeCache(root / "_shape_cache.json")
    desired: dict[str, str] = {}
    rows = []
    captions: dict[str, str] = {}
    for s in present:
        inv = invs[s]
        for rep in range(mix["multipliers"][s]):
            rdir = rc.replica_dir(s, rep)
            for smp in samples[s]:
                tgt, ref = inv["clips"][smp["target"]], inv["clips"][smp["reference"]]
                for k in ("latents", "cond_clean", "conditions"):
                    if not tgt.get(k):
                        raise SystemExit(f"[assemble] {s}/{smp['target']}: inventory has no "
                                         f"{k} source — the stratum is not assemblable yet")
                if not ref.get("latents"):
                    raise SystemExit(f"[assemble] {s}/{smp['reference']}: no reference latents")
                cap = tgt.get("caption")
                if not cap:
                    raise SystemExit(f"[assemble] {s}/{smp['target']}: inventory has no caption")

                rel = f"{rdir}/{smp['group']}/{smp['name']}"
                f, h, w = shapes.get(Path(tgt["latents"]))
                mpath = mask_store_path(root, f, h, w, smp["sided"])
                if not args.plan_only:
                    ensure_mask(mpath, f, h, w, smp["sided"])
                desired[f"latents/{rel}"] = os.path.realpath(tgt["latents"])
                desired[f"reference_latents/{rel}"] = os.path.realpath(ref["latents"])
                desired[f"cond_clean_latents/{rel}"] = os.path.realpath(tgt["cond_clean"])
                desired[f"conditions/{rel}"] = os.path.realpath(tgt["conditions"])
                desired[f"masks/{rel}"] = str(mpath)
                ckey = rc.sha256_text(cap)[:16]
                captions[ckey] = cap
                if rep == 0:
                    rows.append({"rel": f"{rdir}/{smp['group']}/{smp['name']}", "stratum": s,
                                 "group": smp["group"], "target": smp["target"],
                                 "reference": smp["reference"], "sided": smp["sided"],
                                 "caption_key": ckey, "shape": [f, h, w],
                                 "replicas": mix["multipliers"][s],
                                 "endpoints": tgt.get("endpoints") or [],
                                 "caption_sources": [list(x) for x in rc.caption_sources(
                                     tgt, smp["sided"], inv.get("kind", "synthetic_op"))]})
    shapes.save()

    # ---- the two shapes, derived from what the encodes actually are --------------------
    shape_counts: dict[tuple, int] = {}
    shape_by_stratum: dict[str, set] = {}
    for r in rows:
        key = tuple(r["shape"])
        shape_counts[key] = shape_counts.get(key, 0) + r["replicas"]
        shape_by_stratum.setdefault(r["stratum"], set()).add(key)
    shapes_block = {
        "note": "A9 §3/§5 — the root holds TWO SHAPES; tokens and shift are DERIVED from "
                "the latent shape via ltx_trainer/timestep_samplers.py, never restated. "
                "A9's prose figures (1,500 tokens / shift 1.120) are wrong; see DOSSIER §13.2.",
        "per_shape": [dict(rc.shape_record(k), n_samples=v,
                           strata=sorted(s for s, v2 in shape_by_stratum.items() if k in v2))
                      for k, v in sorted(shape_counts.items())],
        "per_stratum": {s: sorted(list(k) for k in v) for s, v in sorted(shape_by_stratum.items())},
        "sigma_distributions": "<PENDING: per-stratum sigma distributions, computed "
                               "analytically from these shifts (A9 §3.1) — owed by the "
                               "sigma-distribution lane>",
    }

    for sh in shapes_block["per_shape"]:
        print(f"[assemble] shape {sh['latent_fhw']} ({sh['name'] or 'UNRULED'}): "
              f"{sh['tokens']} tokens -> shift {sh['shift']}, {sh['n_samples']} samples, "
              f"strata {sh['strata']}")

    if args.plan_only:
        print(f"[assemble] PLAN ONLY: {len(desired)} files across {len(rc.ROOT_DIRS)} dirs, "
              f"{len(desired)//len(rc.ROOT_DIRS)} samples, {len(captions)} distinct captions "
              f"({time.time() - t0:.1f}s)")
        return

    root.mkdir(parents=True, exist_ok=True)
    fs = materialize(root, desired, prune=not args.no_prune)
    print(f"[assemble] filesystem: {fs}")

    # ---- records ----------------------------------------------------------------------
    samples_path = root / "SAMPLES.jsonl"
    samples_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rc.write_json(root / "CAPTIONS.json", captions)

    realized = {s: len(samples[s]) * mix["multipliers"][s] for s in present}
    total = sum(realized.values())
    manifest = {
        "schema": rc.ROOT_MANIFEST_SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "authority": man.get("authority"),
        "root": str(root),
        "seed": man.get("seed", rc.SEED),
        "strata_manifest": {"path": str(args.manifest), "sha256": rc.sha256_file(args.manifest)},
        "strata_present": present,
        "strata_absent": absent,
        "s4_in_mix": "S4" in present,
        "pairing": {"rule": rc.PAIRING_RULE, "max_refs_per_target": max_refs},
        "weights": {
            "note": weight_note,
            "contract": "S0 15 / S1 6 / S2 total 69 / S4 10; the S2a:S2b split is DERIVED "
                        "pro-rata from the assembled post-exclusion base pair counts (A12). "
                        "`intended_pct` below is that derivation's output, not a declared "
                        "number — assert A3 counts the root against it, and assert A3b "
                        "re-checks that the two S2 halves carry the SAME multiplier.",
            "authority": man.get("mix_contract_authority"),
            "branch": {"absent": absent, "override_key": branch_key,
                       "authority": man.get("absent_weight_overrides_authority")},
            "stratum_weights_pct": {n: round(intended[n], 6) for n in present_weights},
            "prorata_groups": {k: list(v) for k, v in groups.items()},
            "prorata_split": mix["prorata_split"],
            "prereg_mix_inputs": str(rc.PREREG_MIX_INPUTS),
            "intended_pct": {s: mix["intended_pct"][s] for s in present},
            "declared_pct": {s: man["strata"][s].get("weight_pct") for s in man["strata"]},
            "realized_pct": {s: round(100.0 * realized[s] / total, 6) for s in present},
            "deviation_pp": {s: round(100.0 * realized[s] / total - mix["intended_pct"][s], 6)
                             for s in present},
            "tolerance_pp": tol,
            "solver": mix,
        },
        "counts": {
            "base_pairs": {s: len(samples[s]) for s in present},
            "replicas": mix["multipliers"],
            "realized_samples": realized,
            "total_samples": total,
            "files_per_dir": total,
            "total_files": total * len(rc.ROOT_DIRS),
            "distinct_captions": len(captions),
            "groups_kept": {s: len(kept_groups[s]) for s in present},
        },
        "inventories": {s: {"path": man["strata"][s]["inventory"],
                            "sha256": rc.sha256_file(man["strata"][s]["inventory"]),
                            "kind": invs[s]["kind"],
                            "endpoint_disjointness": invs[s].get("endpoint_disjointness", True),
                            "groups": len(invs[s]["groups"]),
                            "clips": len(invs[s]["clips"])} for s in present},
        "shapes": shapes_block,
        "exclusions": ex.as_record(),
        "drops": {s: {"n_groups": len(drops[s]["dropped_groups"]),
                      "n_clips": len(drops[s]["dropped_clips"]),
                      "groups": drops[s]["dropped_groups"],
                      "clips": drops[s]["dropped_clips"]} for s in present},
        "filesystem": fs,
        "mask_store": {"dir": str(root / "_mask_store"),
                       "rule": "m[:2]=1 always (prefix anchor); m[-1]=1 iff two-sided",
                       "files": sorted(p.name for p in (root / "_mask_store").glob("*.pt"))},
        "records": {
            "samples_jsonl": str(samples_path),
            "samples_jsonl_sha256": rc.sha256_file(samples_path),
            "captions_json": str(root / "CAPTIONS.json"),
            "captions_json_sha256": rc.sha256_file(root / "CAPTIONS.json"),
        },
        "elapsed_s": round(time.time() - t0, 2),
    }
    rc.write_json(root / "ROOT_MANIFEST.json", manifest)
    print(f"[assemble] {total} samples x {len(rc.ROOT_DIRS)} dirs = {total * 5} files; "
          f"realized mix "
          + ", ".join(f"{s} {manifest['weights']['realized_pct'][s]:.3f}%" for s in present)
          + f"  ({time.time() - t0:.1f}s)")
    print(f"[assemble] wrote {root / 'ROOT_MANIFEST.json'}")


if __name__ == "__main__":
    main()
