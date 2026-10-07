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
3. **Group ids are SLUGGED at path construction** (A11 item 3) — `root_common.slug_group`,
   the same function assert A14 checks with, so the two cannot drift.  The raw→slug mapping is
   stored in `ROOT_MANIFEST.json:group_slugs`; a collision is a hard stop, never a silent
   merge of two pairing rings.  Symlink TARGETS are untouched absolute paths, so nothing
   already written under a raw id (the render, the encoded latents, the endpoint-frame cache)
   is re-keyed — the mapping bridges.
4. **Holdouts are removed here, once**, and every removal is recorded with its reason in
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
import collections
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

#: Per-contract root path (a new dataset writes a NEW root; never overwrites 002's).
CONTRACT_ROOTS = {
    "002_ctt_v2": DEFAULT_ROOT,
    "003_ctt_v2plus": REPO / "outputs/ctt_v2/roots/ctt_v2plus_mix",
    "005_ctt_v2plus_s6reshape": REPO / "outputs/ctt_v2/roots/ctt_v2plus_s6reshape_mix",
}
#: `present` is a disk FACT. S6 becomes present once its encode + conditions land.
_CONTRACT_PRESENT = {
    "002_ctt_v2": dict(_S_PRESENT),
    "003_ctt_v2plus": {**_S_PRESENT, "S6": True},
    # S1 restored 2026-08-29 (disk fact — 003's manifest carries S1 present by hand-edit;
    # 005 encodes it in code).
    "005_ctt_v2plus_s6reshape": {**_S_PRESENT, "S1": True, "S6": True},
}
#: S6 pairing reads each clip's shape + subject from a FROZEN per-contract ROSTER (never the
#: env-dependent ShapeCache). 005 re-encodes S6 at the r832 grids ⇒ a distinct roster.
CONTRACT_S6_ROSTER = {
    "002_ctt_v2": None,  # native (no S6 in 002; kept for symmetry, unused)
    "003_ctt_v2plus": REPO / "outputs/ctt_v2/encodes/EFFECTDATA/ROSTER.json",
    "005_ctt_v2plus_s6reshape": REPO / "outputs/ctt_v2/encodes/EFFECTDATA_r832/ROSTER.json",
}
#: Per-contract S6 inventory basename override (005's S6 latents live at r832 grids).
CONTRACT_S6_INVENTORY = {
    "005_ctt_v2plus_s6reshape": {"S6": "S6_r832.json"},
}
#: Per-contract code-side VERSION string.
CONTRACT_VERSION = {
    "005_ctt_v2plus_s6reshape": "3.1.0-ctt_v2plus_s6reshape-codeside",
}


# --------------------------------------------------------------------------------------
def default_manifest(contract_id: str = "002_ctt_v2") -> dict:
    c = rc.mix_contract(contract_id)               # weights / prorata / mix_strata / absent
    present = _CONTRACT_PRESENT[contract_id]
    assert set(present) == set(c["mix_strata"]), (
        f"{contract_id}: _CONTRACT_PRESENT != contract mix_strata")
    inv = REPO / "outputs/ctt_v2/inventories"
    return {
        "schema": rc.STRATA_MANIFEST_SCHEMA,
        "contract": contract_id,
        "authority": c["authority"],
        "root": str(CONTRACT_ROOTS[contract_id]),
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
        "stratum_weights_pct": dict(c["weights"]),
        "prorata_groups": {k: list(v) for k, v in c["prorata"].items()},
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
        "absent_weight_overrides": {k: dict(v) for k, v in c["absent"].items()},
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
                "present": present[s],
                # the mix-contract weight this stratum draws from.  For a pro-rata member
                # `weight_pct` is deliberately null — the number does not exist until the
                # counts do.
                "weight_group": rc.weight_owner(s),
                "weight_pct": (c["weights"][s] if s in c["weights"] else None),
                "weight_rule": ("fixed (ruled)" if s in c["weights"] else
                                f"DERIVED pro-rata within {rc.weight_owner(s)} from the "
                                f"assembled post-exclusion base pair counts (A12)"),
                "inventory": str(inv / CONTRACT_S6_INVENTORY.get(contract_id, {}).get(s, f"{s}.json")),
            }
            for s in c["mix_strata"]
        },
    }


# --------------------------------------------------------------------------------------
def apply_exclusions(inv: dict, ex: rc.Exclusions) -> tuple[dict, dict]:
    """Return (kept_groups, drop_record).  Groups/clips are removed, never silently.

    The drop record OPENS with the drops the inventory builder already made and recorded
    (`build_inventories._attach`, A16: role-scoped consumption hits are dropped-and-recorded
    rather than crashing).  Carrying them here is what makes `ROOT_MANIFEST.json`'s drop
    record the complete account of every clip that was rendered but not consumed — A16 named
    that record the closing evidence for the 29-clip gap, and a record that silently started
    at the inventory would show 0 of the 29.
    """
    prompts = rc._prompts()
    stratum = inv["stratum"]
    kind = inv.get("kind", "synthetic_op")
    check_endpoints = bool(inv.get("endpoint_disjointness", True))
    dropped_groups = []
    dropped_clips = [dict(d, stratum=d.get("stratum", stratum))
                     for d in (inv.get("build_drops") or {}).get("dropped_clips", [])]
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
            # M3 adjudication (A10): a (clip, role) description may be excluded without the
            # CLIP being dropped from the pool — defects are dispositioned at the unit of
            # CONSUMPTION, (clip, role), not the unit of storage.
            # `openvid_T1MiFx98l3g_0_50to156` has a blank-white A-anchor (frames 0-17 flat,
            # YMIN=YMAX=231) and a healthy B-anchor, so role A is excluded and the clip stays
            # in the pool for B-role use.
            #
            # 🔴 THE PREVIOUS VERSION OF THIS COMMENT SAID THE CLIP "occupies field B in all
            # 10 rendered clips".  THAT IS FALSE, and believing it caused all three key-shape
            # incidents (DOSSIER §22).  A10 checked S2b only and never enumerated the
            # universe.  The verified per-stratum table, A16 §Q3 (universal join, run
            # first-hand against the store's 1,403 keys):
            #
            #     stratum   rows scanned   field A   field B
            #     S2a           7,990        29         0     <- 29 rows consume it as A
            #     S2b           7,990         0        37
            #     S1              390         0         0
            #
            # So this clip IS consumed in the excluded role, by 29 S2a rows, and per A16 those
            # rows are DROPPED (at inventory build; see `build_inventories._attach`, whose
            # record is merged into `drops` below).  The two channels of the exclusion are
            # both still checked here, per-clip, because a clip can reach this function from
            # an inventory built by any path.
            #
            # Absence claims in this lane must declare their universe, enumerated (A16 item 3):
            # a per-universe table like the one above, so a missing cell shows up as a blank
            # instead of an implicit zero.
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
                                      "reasons": creasons, "dropped_at": "assembly"})
                continue
            clips.append(clip)

        if len(clips) < 2:
            dropped_groups.append({"group": gid, "stratum": stratum,
                                   "reasons": [f"fewer_than_2_trainable_clips:{len(clips)}"],
                                   "n_clips": len(clips)})
            continue
        kept[gid] = dict(g, clips=sorted(clips))

    return kept, {"dropped_groups": dropped_groups, "dropped_clips": dropped_clips}


def build_samples(inv: dict, groups: dict, max_refs: int, shape_of: dict | None = None) -> list[dict]:
    """Ring-offset pairs within each group. When `shape_of` is given (S6 only), pairing is
    restricted to SAME-SHAPE: each effect group is sub-grouped by the clip's frozen ROSTER shape
    and ring_pairs runs within each shape-subgroup independently. A shape-subgroup of size 1
    produces nothing (ring_pairs n<2 -> []), so that clip is dropped (no same-shape same-effect
    partner). Every resulting pair is same-shape by construction. shape_of=None leaves the path
    byte-identical to the original (all non-S6 strata are single-shape anyway)."""
    out = []
    for gid, g in sorted(groups.items()):
        if shape_of is None:
            pairs = rc.ring_pairs(g["clips"], max_refs)
        else:
            by_shape: dict = {}
            for c in g["clips"]:
                by_shape.setdefault(shape_of[c], []).append(c)
            pairs = []
            for shp in sorted(by_shape):                       # sorted shape-key order (determinism)
                pairs += rc.ring_pairs(sorted(by_shape[shp]), max_refs)
        for tgt, ref in pairs:
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


def conditions_provenance(present, samples, invs) -> dict:
    """Per stratum: where its text embeds come from, and how many DISTINCT ones to expect.

    `n_distinct_expected` is DERIVED from the inventory — the number of distinct `conditions`
    source paths the stratum's assembled samples actually reference — never from a tabulated
    number that could stop matching.  A21 Q2's assert compares it for equality against the
    realized distinct symlink-target count, which is what makes a shared placeholder visible:
    a stub collapses many expected targets into one.

    `real` is a claim about TEXT, not about file existence, so it is only true when the source
    is not the known placeholder and every referenced path exists.
    """
    out: dict[str, dict] = {}
    for s in present:
        inv = invs[s]
        srcs, missing = set(), 0
        for smp in samples[s]:
            p = (inv["clips"][smp["target"]] or {}).get("conditions")
            if not p:
                missing += 1
                continue
            srcs.add(os.path.realpath(p))
        dirs = sorted({str(Path(p).parent) for p in srcs})
        placeholderish = [d for d in dirs if "placeholder" in d.lower()]
        out[s] = {
            "n_samples": len(samples[s]),
            "n_distinct_expected": len(srcs),
            "source_dirs": dirs[:4],
            "n_source_dirs": len(dirs),
            "targets_missing_a_conditions_path": missing,
            "real": bool(srcs) and not placeholderish and missing == 0,
            "placeholder_dirs_seen": placeholderish,
        }
    return out


def mask_store_path(root: Path, f: int, h: int, w: int, sided: str) -> Path:
    #: The NAME is owned by `root_common.mask_store_name` and delegated to here, never restated:
    #: four sites had grown their own copy of it and two were computing the pre-`p{prefix}` form.
    #: This function is just the path wrapper.
    return root / "_mask_store" / rc.mask_store_name(f, h, w, sided)


def ensure_mask(path: Path, f: int, h: int, w: int, sided: str) -> None:
    """mask = f(conditioning): [:prefix_latents(shape)]=1; [-1]=1 iff two-sided."""
    if path.exists():
        return
    import torch  # noqa: PLC0415

    m = torch.zeros(f, h, w)
    m[:rc.prefix_latents((f, h, w))] = 1.0
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
    ap.add_argument("--contract", default="002_ctt_v2",
                    help="mix contract id for --init-manifest (root_common.MIX_CONTRACTS). "
                         "DEFAULT 002_ctt_v2 so no existing command silently builds a new mix; "
                         "003_ctt_v2plus must be asked for by name (adds EffectData S6 at 20 pp).")
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
    ap.add_argument("--sampler-mix", action="store_true",
                    help="emit ONE row per base pair and record the per-stratum target weights "
                         "for the trainer's sampler to realise, instead of duplicating symlinks "
                         "to bake the mix into the filesystem. 56,368 rows instead of 404,259, "
                         "exact weights instead of an integer-multiplier residual, and the mix "
                         "becomes changeable without touching the dataset. Requires the trainer's "
                         "stratified sampler + its consumed-count assert.")
    ap.add_argument("--no-prune", action="store_true",
                    help="do not delete root entries that are no longer desired")
    ap.add_argument("--plan-only", action="store_true",
                    help="compute everything, write no symlinks and no manifest")
    ap.add_argument("--code-side", action="store_true",
                    help="CODE-SIDE root: write samples.jsonl with ABSOLUTE realpaths + the "
                         "_mask_store only; NO per-row symlink trees (the trainer's SampleListDataset "
                         "resolves dataset_root/abspath == abspath). Skips materialize entirely.")
    args = ap.parse_args()

    if args.init_manifest:
        rc.write_json(args.init_manifest, default_manifest(args.contract))
        print(f"[assemble] wrote default strata manifest -> {args.init_manifest}")
        return
    if not args.manifest:
        raise SystemExit("--manifest is required (or --init-manifest to create one)")

    t0 = time.time()
    man = rc.read_json(args.manifest)
    if man.get("schema") != rc.STRATA_MANIFEST_SCHEMA:
        raise SystemExit(f"bad manifest schema: {man.get('schema')!r}")
    # A manifest's own `contract` field is authoritative: never let mix.json / ROOT_MANIFEST
    # record a contract other than the one the manifest was built for.
    if man.get("contract") and man["contract"] != args.contract:
        raise SystemExit(
            f"[assemble] manifest contract {man['contract']!r} != --contract {args.contract!r}")
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
    # A vacuous standing role exclusion is a failure, never "nothing to exclude" (A16 item 1).
    rc.require_role_exclusions("assemble_root")
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
    # S6 same-shape pairing needs each clip's shape + subject from the FROZEN ROSTER (never the
    # env-dependent ShapeCache). S6 pairs are restricted to same-shape same-effect; the lone
    # subject of a shape within an effect has no same-shape partner and is DROPPED (recorded).
    s6_shape, s6_subject = {}, {}
    if "S6" in invs:
        _roster_p = CONTRACT_S6_ROSTER[man.get("contract", args.contract)]
        _rj = rc.read_json(_roster_p)
        s6_shape = {c["stem"]: tuple(c["latent_fhw"]) for c in _rj["clips"]}
        s6_subject = {c["stem"]: c["subject"] for c in _rj["clips"]}
    kept_groups, drops, samples = {}, {}, {}
    for s, inv in invs.items():
        kept_groups[s], drops[s] = apply_exclusions(inv, ex)
        samples[s] = build_samples(inv, kept_groups[s], max_refs,
                                   shape_of=s6_shape if s == "S6" else None)
        if s == "S6":
            # invariants (advisor build-gates): every pair same-shape + different-subject
            for p in samples[s]:
                assert s6_shape[p["target"]] == s6_shape[p["reference"]], \
                    f"[assemble] S6 cross-shape pair leaked: {p['target']} vs {p['reference']}"
                assert s6_subject[p["target"]] != s6_subject[p["reference"]], \
                    f"[assemble] S6 same-subject pair: {p['target']} vs {p['reference']}"
            # record the shape-singleton drops (clips consumed nowhere) into build_drops
            grp_of = {c: gid for gid, g in kept_groups[s].items() for c in g["clips"]}
            consumed = {p["target"] for p in samples[s]} | {p["reference"] for p in samples[s]}
            for stem in sorted(set(grp_of) - consumed):
                drops[s]["dropped_clips"].append(
                    {"clip": stem, "group": grp_of[stem], "stratum": "S6",
                     "reasons": ["no_same_shape_same_effect_partner"]})
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
    if args.sampler_mix:
        # ---- SAMPLER MIX: one row per base pair; the mix moves into the TRAINER ------------
        # Physical duplication made the realised ratio countable off disk, which is why it was
        # chosen (A3 counts it).  It also cost 2,021,295 symlinks for 56,368 distinct pairs,
        # forced INTEGER multipliers (hence a 0.4289 pp residual the contract had to tolerate),
        # and produced the `S0_r100` class of bug because S0 needed x153.
        #
        # Here every base pair is emitted EXACTLY ONCE and the per-stratum target weights are
        # recorded in the manifest for the trainer's sampler to realise EXACTLY.  The path
        # scheme keeps its `<stratum>_r00/` component on purpose, so A1 set-equality, A2b path
        # scheme, B1 per-shape set-equality and `parse_replica` are all unchanged, and A3b is
        # trivially satisfied (every member of a pro-rata group carries the same multiplier, 1).
        #
        # ⚠ THE ONE PROPERTY THIS GIVES UP: the realised mix is no longer countable from the
        # root, so A3 cannot be evidence any more.  It is replaced, not dropped — the sampler
        # logs per-stratum CONSUMED counts and asserts them against these weights at train
        # start, failing closed.  A mix that is only a number in a config with nothing counting
        # it is exactly the silent-failure shape this campaign keeps getting bitten by.
        intended_pct, split = rc.expand_prorata_weights(intended, base_counts, groups)
        mix = {"multipliers": {s: 1 for s in present},
               "total": sum(base_counts.values()),
               "intended_pct": intended_pct,
               "prorata_split": split,
               "max_deviation_pp": 0.0,
               "mode": "sampler",
               "note": "one row per base pair; the mix is realised by the trainer's sampler "
                       "from `sampler_mix.weights_pct` below, not by symlink duplication"}
    else:
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

    # ---- group-id slugs, decided ONCE, here (A11 item 3) -------------------------------
    # A11: "sanitise deterministically before assembly — lowercase, non-alphanumeric ->
    # underscore, collapse runs — assert the 42 slugged ids remain unique, and store the
    # raw->slug mapping in the manifest.  The trainer globs fine is not the bar; path
    # robustness across shells, rsync and future tooling is."  S4's raw group ids are refVFX
    # effect strings with SPACES (`t2k1s takes off clothes revealing a lean muscular body`),
    # which is what put a space in every S4 path in the smoke root.
    #
    # `rc.slug_group` is the SAME function assert A14 uses — deliberately not a second
    # implementation, because two implementations of a path rule drift and the drift is silent.
    # Uniqueness is asserted over the groups actually assembled AND over the full inventory
    # (A14's scope), so a collision cannot appear later by including a group this run dropped.
    slug_tables: dict[str, dict] = {}
    for s in present:
        for scope, gids in (("assembled", kept_groups[s]), ("inventory", invs[s]["groups"])):
            table, collisions = rc.slug_map(gids)
            if collisions:
                raise SystemExit(
                    f"[assemble] {s}: group-id slug COLLISION over the {scope} groups: "
                    f"{collisions}. Two raw ids that slug to the same path would silently "
                    f"MERGE two operator groups into one pairing ring — a design change "
                    f"disguised as a path fix. Fix the ids; do not relax the slug.")
            empty = sorted(raw for slug, raw in table.items() if not slug)
            if empty:
                raise SystemExit(f"[assemble] {s}: group id(s) {empty} slug to the empty "
                                 f"string over the {scope} groups")
        slug_tables[s] = {rc.slug_group(g): g for g in sorted(kept_groups[s])}
        n_changed = sum(1 for slug, raw in slug_tables[s].items() if slug != raw)
        print(f"[assemble] {s}: {len(slug_tables[s])} group slugs unique "
              f"({n_changed} differ from the raw id)")

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

                #: THE path is built from the SLUG; the raw id survives in SAMPLES.jsonl and
                #: in the manifest's mapping table, which is what every consumer resolves
                #: through.  Nothing already written under a raw string is re-keyed: the
                #: symlink TARGETS are untouched absolute realpaths into the encode store,
                #: so the completed render, the ~16k encoded latents and the endpoint-frame
                #: cache keep exactly the names they have (A11: "the mapping bridges").
                gslug = rc.slug_group(smp["group"])
                rel = f"{rdir}/{gslug}/{smp['name']}"
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
                    rows.append({"rel": rel, "stratum": s,
                                 "group": smp["group"], "group_slug": gslug,
                                 "target": smp["target"],
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
        "note": f"the root holds {len(shape_counts)} shape classes; tokens and shift are "
                f"DERIVED from the latent shape via ltx_trainer/timestep_samplers.py, never "
                f"restated (A9's prose figures 1,500 tokens / shift 1.120 are wrong; see "
                f"DOSSIER §13.2)",
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

    if args.code_side:
        # CODE-SIDE: no per-row symlink trees. samples.jsonl carries the ABSOLUTE realpaths that
        # `desired` already resolved; the only materialized artifact is _mask_store (generated in
        # the loop above). The trainer reads dataset_root/abspath == abspath.
        TREES = list(rc.ROOT_DIRS)
        rc.write_json(root / "CAPTIONS.json", captions)
        if not (root / "captions.json").exists():
            (root / "captions.json").symlink_to("CAPTIONS.json")
        # PORTABILITY: samples.jsonl carries paths RELATIVE to the dataset root, so the same file
        # works on any device. Source tensors resolve through ONE re-pointable directory symlink
        # `_src` -> the repo (every source path is under $LAB/diffusion-research); masks live in the
        # in-root `_mask_store`. On a new device: copy the root + point `_src` at that device's repo.
        REPO_ABS = str(root.parents[3])                     # root = <repo>/outputs/ctt_v2/roots/<name>
        MASK_ABS = str(root / "_mask_store")

        def portable(p: str) -> str:
            if p.startswith(MASK_ABS):
                return "_mask_store/" + os.path.basename(p)
            if not p.startswith(REPO_ABS + "/"):
                raise SystemExit(f"[code-side] source path not under repo (not portable): {p}")
            return "_src/" + p[len(REPO_ABS) + 1:]

        srclink = root / "_src"
        if not (srclink.is_symlink() or srclink.exists()):
            srclink.symlink_to("../../../..")               # = <repo> here; re-point per device
        n = 0
        with (root / "samples.jsonl").open("w") as out:
            for r in rows:
                rel = r["rel"]
                out.write(json.dumps({
                    "id": f"{r['stratum']}/{r['group_slug']}/{r['target']}__ref_{r['reference']}",
                    "stratum": r["stratum"], "group": r["group"], "group_slug": r["group_slug"],
                    "target": r["target"], "reference": r["reference"], "sided": r["sided"],
                    "caption_key": r["caption_key"], "shape": r["shape"],
                    "paths": {t: portable(desired[f"{t}/{rel}"]) for t in TREES},   # root-relative
                    "endpoints": r["endpoints"], "caption_sources": r["caption_sources"],
                }) + "\n")
                n += 1
        s6_prov = ({"s6_rule": "ring_offset_within_op_shape__k=min(3,n-1)__s6_drop_shape_singletons",
                    "s6_same_shape_pairs": len(samples["S6"]),
                    "s6_shape_singletons_dropped": sum(
                        1 for d in drops["S6"]["dropped_clips"]
                        if "no_same_shape_same_effect_partner" in d.get("reasons", [])),
                    "s6_per_shape_census": {str(list(k)): v for k, v in sorted(collections.Counter(
                        s6_shape[p["target"]] for p in samples["S6"]).items())}}
                   if "S6" in present else {})
        rc.write_json(root / "mix.json", {
            "schema": "ctt_v2plus_mix/v2_code_side", "contract": args.contract,
            "form": "code_side — ROOT-RELATIVE paths in samples.jsonl (source via the re-pointable _src symlink; masks in _mask_store); no per-row symlink trees",
            "stratum_weights_pct": mix["intended_pct"], "prorata_split": mix.get("prorata_split"),
            "strata_present": present, "strata_absent": absent,
            "authority": "root_common.MIX_CONTRACTS; StratifiedEpochSampler realizes intended_pct"})
        # samples.jsonl is fully written and closed above; stamp its byte-sha + row count
        # into the manifest NATIVELY (identical names/types to the post-assembly stamp used
        # for ROOT4/ROOT5 in earlier rounds — see DOSSIER Round 4 §3 / Round 6 §1).
        samples_sha256 = rc.sha256_file(root / "samples.jsonl")
        samples_rows = n
        rc.write_json(root / "ROOT_MANIFEST.json", {
            "schema": rc.ROOT_MANIFEST_SCHEMA, "form": "code_side",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "root": str(root),
            "contract": args.contract, "seed": man.get("seed", rc.SEED),
            "strata_present": present, "strata_absent": absent, "total_samples": n,
            "samples_sha256": samples_sha256, "samples_rows": samples_rows,
            "pairing": {"rule": rc.PAIRING_RULE, "max_refs_per_target": max_refs, **s6_prov},
            "weights": {"note": weight_note, "intended_pct": mix["intended_pct"]},
            "shapes": shapes_block,
            "mask_store": {"dir": str(root / "_mask_store"),
                           "files": sorted(p.name for p in (root / "_mask_store").glob("*.pt"))},
            "drops": {s: {"n_clips": len(drops[s]["dropped_clips"]),
                          "clips": drops[s]["dropped_clips"]} for s in present},
            "note": "CODE-SIDE root: the per-row symlink trees are NOT materialized; samples.jsonl "
                    "carries ROOT-RELATIVE paths (source via _src symlink -> repo, re-point per device). Filesystem-count asserts (assert_root.py) do not "
                    "apply; verification is the trainer's own _verify_files + fewer-files-to-audit.",
        })
        (root / "VERSION").write_text(
            CONTRACT_VERSION.get(args.contract, "3.0.0-ctt_v2plus-codeside") + "\n")
        print(f"[assemble] CODE-SIDE: samples.jsonl {n} rows (absolute realpaths) + mix.json + "
              f"ROOT_MANIFEST + {len(list((root / '_mask_store').glob('*.pt')))} masks; "
              f"NO symlink trees ({time.time() - t0:.1f}s)")
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
        # §A19's requirement, in the manifest itself: a root whose manifest cannot tell a real
        # text embed from a shared stub is not auditable.  It found 168,184 samples pointing at
        # ONE `conditions_placeholder.pt` while every path-alignment assert passed, because a
        # placeholder aligns perfectly.  `assert_root.py:A1c` reads this block and compares the
        # ACTUAL distinct symlink-target count per stratum against `n_distinct_expected`.
        "conditions_provenance": conditions_provenance(present, samples, invs),
        "pairing": {"rule": rc.PAIRING_RULE, "max_refs_per_target": max_refs,
                    **({"s6_rule": "ring_offset_within_op_shape__k=min(3,n-1)__s6_drop_shape_singletons",
                        "s6_same_shape_pairs": len(samples["S6"]),
                        "s6_shape_singletons_dropped": sum(
                            1 for d in drops["S6"]["dropped_clips"]
                            if "no_same_shape_same_effect_partner" in d.get("reasons", [])),
                        "s6_per_shape_census": {
                            str(list(k)): v for k, v in sorted(collections.Counter(
                                s6_shape[p["target"]] for p in samples["S6"]).items())},
                        "s6_zero_same_subject_verified": True}
                       if "S6" in present else {})},
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
            "mix_mode": mix.get("mode", "replica_duplication"),
            "sampler_mix": ({
                "weights_pct": mix["intended_pct"],
                "base_pairs": {s: len(samples[s]) for s in present},
                "contract": "the trainer's sampler MUST realise these per-stratum shares exactly "
                            "and assert its own consumed counts against them at train start, "
                            "failing closed. With replica duplication A3 counted the realised mix "
                            "off disk; in sampler mode that evidence moves into the trainer.",
                "authority": "A9 §4 + A11 item 3 (S2 total 69) + A12 (pro-rata split); mechanism "
                             "translated from physical duplication to index expansion, identical "
                             "slot multiset per epoch, exact rather than integer-rounded",
            } if mix.get("mode") == "sampler" else None),
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
        # ---- A11 item 3: the raw->slug bridge, stored, not inferred ---------------------
        "group_slugs": {
            "rule": "lowercase, non-alphanumeric -> '_', runs collapsed, edges trimmed "
                    "(root_common.slug_group — the same function assert A14 checks with, and "
                    "the same call the path builder above makes)",
            "authority": "A11 (σ / S4-weight ruling), RULING 4's minor ratifications, recorded "
                         "in DOSSIER §15.5 — the same item `assert_root.py:A14`'s header calls "
                         "'A11 item 3': 'sanitise "
                         "deterministically before assembly, assert the 42 slugged ids remain "
                         "unique, and store the raw->slug mapping in the manifest. Do not "
                         "re-key anything already written under raw strings; the mapping "
                         "table bridges.'",
            "applies_to": "the GROUP component of every relative path inside this root. The "
                          "symlink TARGETS are untouched absolute paths into the encode / "
                          "render stores, so nothing already keyed to a raw id is re-keyed.",
            "n_slugged": {s: sum(1 for slug, raw in t.items() if slug != raw)
                          for s, t in sorted(slug_tables.items())},
            "slug_to_raw": {s: dict(sorted(t.items())) for s, t in sorted(slug_tables.items())},
            "raw_to_slug": {s: {raw: slug for slug, raw in sorted(t.items())}
                            for s, t in sorted(slug_tables.items())},
        },
        "exclusions": ex.as_record(),
        # Every clip that was rendered but is NOT consumed, with its reason.  `dropped_at`
        # separates the two stages: "inventory_build" drops are the standing A10 role-scoped
        # exclusion applied at build time (A16); "assembly" drops are the holdout/reserved/
        # eval-endpoint exclusions applied here.
        "drops": {s: {"n_groups": len(drops[s]["dropped_groups"]),
                      "n_clips": len(drops[s]["dropped_clips"]),
                      "n_clips_at_inventory_build":
                          sum(1 for d in drops[s]["dropped_clips"]
                              if d.get("dropped_at") == "inventory_build"),
                      "n_clips_at_assembly":
                          sum(1 for d in drops[s]["dropped_clips"]
                              if d.get("dropped_at") == "assembly"),
                      "inventory_build_record":
                          {k: v for k, v in (invs[s].get("build_drops") or {}).items()
                           if k != "dropped_clips"},
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
