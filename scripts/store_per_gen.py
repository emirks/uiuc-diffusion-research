#!/usr/bin/env python
"""One row per GENERATION from the existing v4 evals (store_per_gen).

For the 19 grid-v3 variants (misc/2026-09-17_feature_store/population_gridv3.json) this walks the
already-scored pool rows in `store/evals/<eval>/<arm>/c*/items.jsonl` and emits one row per
generation = (grid item_id, seed) to `store/evals/<eval>/<arm>/per_gen.jsonl`, next to the shards
it derives from (evals/028 for the 16 paper-arm variants, evals/030 for the 3 externals).

The headline scalar is `transport_pct` — the paper's "%_same" (capped pooled-%): the generation's
mean appearance similarity (app_ref) to the real videos of its operator (gt_pool_class), divided by
that operator's ceiling, as a percentage capped at 100. It carries the S3 / app_ref column (the
stored deployed-kernel appearance metric), NOT the size-free "Look_u" recompute.

The metric is reproduced from the CANONICAL implementation that wrote evals/028 summary.json —
eval_ladder/run_eval.py::{ceilings,pool_means,item_pct} as driven by scripts/grid_v3/closeout.py:
  * pool mean per (grid item_id, seed) = mean of app_ref over that generation's GT-pool references
    (dedup by full eval item_id; drop app_ref=None), exactly run_eval.pool_means;
  * ceiling per class = mean within-class off-diagonal similarity of the certified m1a_S3 matrix,
    with the eval_ladder/ceilings_v3.json overlay for the classes absent from it (new HF zero-shot
    + EffectData `ed.*`), exactly run_eval.ceilings();
  * summary.json `level` = mean over the cell's grid items of the per-item UNCAPPED ratio, where the
    per-item ratio = mean over its seeds of (per-seed pool mean / ceiling), exactly run_eval.item_pct.

Each row stores the raw parts (`transport_raw` = the per-(item,seed) pool mean of app_ref,
`transport_ceiling` = the class ceiling), so both the UNCAPPED summary.json levels and the CAPPED
per-generation `transport_pct` are recoverable. `--check` re-aggregates the written per_gen.jsonl
and prints (a) the per_gen -> summary.json table (exits 1 on any mismatch) and (b) the 183-triple
zero-shot slice vs the CHANGELOG 2026-09-17 23:23 numbers.

CPU only, pure python over jsonl (+ one cv2 frame-count decode per variant). Idempotent, per-arm.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval_ladder"))
import run_eval  # noqa: E402  (canonical ceilings; MATRIX=m1a_S3, MAX_POOL_REFS=8)

POPULATION = REPO / "misc/2026-09-17_feature_store/population_gridv3.json"

# scalars carried from items.jsonl. CONST_FIELDS are properties of the generated clip (identical
# across a generation's pool rows) — taken verbatim. The rest are per-pool-reference comparisons and
# vary across the rows, so they are reduced to one per generation: MEAN_FIELDS are averaged over the
# pool references (copy_max averaged this way is exactly what reproduces summary.json copy_max_mean),
# near_copy is OR'd (a near copy of ANY real reference).
CONST_FIELDS = ["max_seam_z", "prefix_seam_z", "suffix_seam_z", "prefix_dino", "prefix_lpips",
                "core_degenerate", "cross_high"]
MEAN_FIELDS = ["copy_max", "cam_zpr", "obj_csls"]
ANY_FIELDS = ["near_copy"]

ZS_CELLS = {"G-zs-same", "G-zs-cross", "G-zs-foreign"}  # the one-sided zero-shot slice cells

# 183-triple zero-shot table (CHANGELOG 2026-09-17 23:23): score_v3_zs.py S3 column, 222-based ceilings.
# score_v3_zs groups our arms as HF + ED together; the shared triples come from VAP's grid.
CHANGELOG_183 = {
    "VAP":                       (["011_vap/05_author_native__dai"],                                          81.1),
    "VFXMaster":                 (["012_vfxmaster/05_author_native__dai"],                                    85.7),
    "refVFX":                    (["003_refvfx/03_author_native__dai"],                                       62.2),
    "ic_gen neutral":            (["001_ic_gen/03_neutral_v3__dai", "001_ic_gen/04_neutral_v3ed81__dai"],     61.6),
    "dualforce_control neutral": (["013_dualforce_control/03_neutral_v3__dai", "013_dualforce_control/04_neutral_v3ed81__dai"], 87.6),
    "dualforce_dcg_w6 neutral":  (["032_dualforce_dcg_w6/03_neutral_v3__dai", "032_dualforce_dcg_w6/04_neutral_v3ed81__dai"],   92.6),
}
SHARED_183_GRID = "011_vap/05_author_native__dai"  # its grid defines the shared (endpoint, reference, cell) triples


# ------------------------------------------------------------------------ helpers
def transport(raw: float, ceiling: float) -> tuple[float, bool]:
    """(transport_pct, transport_capped): the capped pooled-%, and whether the cap bit.
    transport_pct = 100 * min(raw / ceiling, 1); capped when the uncapped ratio exceeds 1."""
    ratio = raw / ceiling
    return 100.0 * min(ratio, 1.0), ratio > 1.0


def parse_item_id(eval_item_id: str) -> tuple[str, int]:
    """`<grid item_id>__s<seed>__ref_<poolstem>` -> (grid item_id, seed). Same split as run_eval.pool_means."""
    head, _, _pref = eval_item_id.rpartition("__ref_")
    base, _, seed = head.rpartition("__s")
    return base, int(seed)


def n_frames_of(variant_dir: Path) -> int | None:
    """Frame count of the variant's generations (constant per variant); decode one video."""
    import cv2

    vids = sorted((variant_dir / "videos").glob("*.mp4"))
    if not vids:
        return None
    cap = cv2.VideoCapture(str(vids[0]))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def collect_generations(scores_dir: Path) -> dict[tuple[str, int], dict]:
    """One pass over c*/items.jsonl. Per (grid item_id, seed): app_ref list (for the pool mean),
    ref_in_v4_population list, and the constant per-generation scalars. Dedup by full eval item_id
    exactly as run_eval.pool_means (an item planned twice appears in two passes' shards)."""
    seen: set[str] = set()
    gens: dict[tuple[str, int], dict] = collections.defaultdict(
        lambda: {"app_ref": [], "ref_in_pop": [], "const": {},
                 "mean": collections.defaultdict(list), "any": collections.defaultdict(list)})
    for f in sorted(scores_dir.glob("*/items.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            iid = r["item_id"]
            if iid in seen:
                continue
            seen.add(iid)
            if r.get("app_ref") is None:
                continue
            key = parse_item_id(iid)
            g = gens[key]
            g["app_ref"].append(float(r["app_ref"]))
            if r.get("ref_in_v4_population") is not None:
                g["ref_in_pop"].append(bool(r["ref_in_v4_population"]))
            for k in CONST_FIELDS:
                if k not in g["const"] and r.get(k) is not None:
                    g["const"][k] = r[k]
            for k in MEAN_FIELDS:
                if r.get(k) is not None:
                    g["mean"][k].append(float(r[k]))
            for k in ANY_FIELDS:
                if r.get(k) is not None:
                    g["any"][k].append(bool(r[k]))
    return gens


def build_variant(variant_rel: str, ceil: dict[str, float]) -> tuple[Path, str, list[dict], tuple[int, int]]:
    """Return (per_gen.jsonl path, arm, rows, (n_control_skipped, n_no_ceiling_skipped)) for one variant."""
    variant_dir = REPO / variant_rel
    scores_dir = (variant_dir / "scores").resolve()
    arm = scores_dir.name  # canonical harness arm (e.g. ic_gen_neutral_v3 / vap_author_native)
    grid = {json.loads(l)["item_id"]: json.loads(l)
            for l in (variant_dir / "grid.jsonl").read_text().splitlines()}
    nf = n_frames_of(variant_dir)
    gens = collect_generations(scores_dir)

    rows: list[dict] = []
    skipped_no_grid = skipped_no_ceil = 0
    for (item_id, seed), g in sorted(gens.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        gi = grid.get(item_id)
        if gi is None:            # control_hold / control_lerp diagnostic pseudo-gens — not generations
            skipped_no_grid += 1
            continue
        cls = gi["gt_pool_class"]
        if cls not in ceil:       # class outside the (certified + ceilings_v3) ceiling set — excluded, as item_pct does
            skipped_no_ceil += 1
            continue
        raw = st.mean(g["app_ref"])
        c = ceil[cls]
        pct, is_capped = transport(raw, c)
        vid = variant_dir / "videos" / f"{item_id}__s{seed}.mp4"
        if not vid.exists():
            alt = variant_dir / "videos" / f"{item_id}__seed{seed}.mp4"
            vid = alt if alt.exists() else vid
        cst = g["const"]
        mean = {k: (st.mean(v) if v else None) for k, v in g["mean"].items()}
        anyf = {k: (any(v) if v else None) for k, v in g["any"].items()}
        rows.append({
            "item_id": item_id,
            "seed": seed,
            "arm": arm,
            "variant_dir": variant_rel,
            "gen_video": str(vid.relative_to(REPO)),
            "tier": gi["ref_novelty"],       # {seen, unseen, zero_shot}
            "sided": gi["sided"],            # {one, two}
            "cell": gi["cell"],
            "content": gi["content"],
            "pct_type": gi.get("pct_type") or ("same" if gi["content"] == "same" else "proxy"),
            "endpoint": gi["endpoint"],
            "reference": gi["reference"],
            "ref_class": cls,                # gt_pool_class = the operator whose real videos the gen is scored against
            "n_frames": nf,
            "transport_pct": pct,
            "transport_raw": raw,            # app_ref: per-generation mean over the GT-pool references
            "transport_ceiling": c,
            "transport_capped": is_capped,
            "copy_max": mean.get("copy_max"),      # mean over pool references (reproduces summary copy_max_mean)
            "near_copy": anyf.get("near_copy"),    # near copy of ANY real reference
            "max_seam_z": cst.get("max_seam_z"),
            "prefix_seam_z": cst.get("prefix_seam_z"),
            "suffix_seam_z": cst.get("suffix_seam_z"),
            "prefix_dino": cst.get("prefix_dino"),
            "prefix_lpips": cst.get("prefix_lpips"),
            "cam_zpr": mean.get("cam_zpr"),        # mean over pool references
            "obj_csls": mean.get("obj_csls"),      # mean over pool references
            "core_degenerate": cst.get("core_degenerate"),
            "cross_high": cst.get("cross_high"),
            # per-generation reduction of the per-pool-row flag: True iff every GT-pool reference used
            # for this generation's app_ref is in the certified v4 (222-clip) population.
            "ref_in_v4_population": (all(g["ref_in_pop"]) if g["ref_in_pop"] else None),
        })
    out = scores_dir / "per_gen.jsonl"
    return out, arm, rows, (skipped_no_grid, skipped_no_ceil)


def write_variants(variants: list[str]) -> None:
    ceil = run_eval.ceilings()
    for vr in variants:
        out, arm, rows, (sng, snc) = build_variant(vr, ceil)
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"[write] {out.relative_to(REPO)}  arm={arm}  gens={len(rows)}"
              f"  (skipped: controls={sng}, no-ceiling={snc})")


# ------------------------------------------------------------------------ --check
def load_per_gen(scores_dir: Path) -> list[dict]:
    p = scores_dir / "per_gen.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def check() -> int:
    """Re-aggregate the WRITTEN per_gen.jsonl. (a) reproduce evals/028 summary.json (exit 1 on
    mismatch). (b) print the 183-triple zero-shot table vs the CHANGELOG numbers (informational)."""
    pop = json.loads(POPULATION.read_text())
    variants = pop["gen_variants"]
    scores_of = {vr: (REPO / vr / "scores").resolve() for vr in variants}
    rows_of = {vr: load_per_gen(scores_of[vr]) for vr in variants}

    entry = sorted((REPO / "store/evals").glob("028_grid_v3_paper_arms__dai__*"))[0]
    summ = json.loads((entry / "summary.json").read_text())["arms"]

    print(f"=== (a) per_gen.jsonl  ->  {entry.name}/summary.json ===")
    print(f"{'arm':34s} {'items':>6s} {'headline':>9s} {'cells ok':>9s}")
    mismatches: list[str] = []
    for vr in variants:
        rows = rows_of[vr]
        if not rows:
            continue
        arm = rows[0]["arm"]
        if arm not in summ:      # externals (evals/030) have no summary.json
            continue
        # per grid item_id: mean over seeds of the UNCAPPED ratio (== run_eval.item_pct)
        by_item_ratio: dict[str, list[float]] = collections.defaultdict(list)
        by_item_cm: dict[str, list[float]] = collections.defaultdict(list)
        meta_of: dict[str, dict] = {}
        for r in rows:
            by_item_ratio[r["item_id"]].append(r["transport_raw"] / r["transport_ceiling"])
            if r.get("copy_max") is not None:
                by_item_cm[r["item_id"]].append(float(r["copy_max"]))
            meta_of[r["item_id"]] = r
        item_pct = {i: st.mean(v) for i, v in by_item_ratio.items()}
        cells: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
        copies: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
        for i, v in item_pct.items():
            m = meta_of[i]
            key = (m["tier"], m["content"])
            cells[key].append(v)
            if i in by_item_cm:
                copies[key].append(st.mean(by_item_cm[i]))
        S = summ[arm]
        if len(item_pct) != S["items_scored"]:
            mismatches.append(f"{arm}: items_scored {len(item_pct)} != {S['items_scored']}")
        same = [v for (t, c), vs in cells.items() if c == "same" for v in vs]
        hd = round(st.mean(same), 4) if same else None
        if hd != S["headline_pct_same"]:
            mismatches.append(f"{arm}: headline {hd} != {S['headline_pct_same']}")
        n_ok = 0
        for key, vals in cells.items():
            ck = f"{key[0]}|{key[1]}"
            c = S["cells"].get(ck)
            if c is None:
                mismatches.append(f"{arm}: extra cell {ck}")
                continue
            lvl = round(st.mean(vals), 4)
            sd = round(st.pstdev(vals), 4) if len(vals) > 1 else None
            cmm = round(st.mean(copies[key]), 4) if copies[key] else None
            ok = True
            if len(vals) != c["n"]:
                mismatches.append(f"{arm} {ck}: n {len(vals)} != {c['n']}"); ok = False
            if lvl != c["level"]:
                mismatches.append(f"{arm} {ck}: level {lvl} != {c['level']}"); ok = False
            if sd != c["sd"]:
                mismatches.append(f"{arm} {ck}: sd {sd} != {c['sd']}"); ok = False
            if cmm != c["copy_max_mean"]:
                mismatches.append(f"{arm} {ck}: copy_max_mean {cmm} != {c['copy_max_mean']}"); ok = False
            n_ok += ok
        print(f"{arm:34s} {len(item_pct):6d} {('%.4f' % hd) if hd is not None else '—':>9s} {f'{n_ok}/{len(cells)}':>9s}")

    # (b) the 183-triple zero-shot slice (S3 / app_ref, capped per generation)
    print(f"\n=== (b) 183-triple one-sided zero-shot, S3 capped pooled-% (CHANGELOG 2026-09-17 23:23) ===")
    rows_by_variant = {}
    for vr in variants:
        for r in rows_of[vr]:
            rows_by_variant.setdefault(vr, []).append(r)
    shared_grid = {json.loads(l)["item_id"]: json.loads(l)
                   for l in (REPO / "store/gens" / SHARED_183_GRID / "grid.jsonl").read_text().splitlines()}
    shared = {(r["endpoint"], r["reference"], r["cell"]) for r in shared_grid.values()}
    print(f"{'arm':28s} {'gens':>5s} {'per_gen':>8s} {'CHANGELOG':>10s} {'Δ':>6s}")
    for label, (parts, target) in CHANGELOG_183.items():
        capped: list[float] = []
        for part in parts:
            vr = "store/gens/" + part
            for r in rows_by_variant.get(vr, []):
                if r["cell"] not in ZS_CELLS:
                    continue
                if (r["endpoint"], r["reference"], r["cell"]) not in shared:
                    continue
                capped.append(r["transport_pct"] / 100.0)  # transport_pct already capped
        got = 100.0 * st.mean(capped) if capped else float("nan")
        print(f"{label:28s} {len(capped):5d} {got:8.2f} {target:10.1f} {got - target:+6.2f}")
    print("  note: per_gen carries the CERTIFIED m1a_S3 ceilings (reproduces summary.json exactly); the CHANGELOG"
          "\n  183-triple used the 222-based recompute (calibration ratio 0.9965) — VFXMaster 85.76->85.8 vs 85.73->85.7.")

    print()
    if mismatches:
        print(f"[FAIL] {len(mismatches)} summary.json mismatch(es):")
        for m in mismatches:
            print("   ", m)
        return 1
    print("[OK] per_gen.jsonl reproduces evals/028 summary.json exactly (level, n, sd, copy_max_mean, headline, items_scored).")
    return 0


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="re-aggregate written per_gen.jsonl; exit 1 on summary.json mismatch")
    ap.add_argument("--only", default=None, help="substring filter on the variant path (e.g. ic_gen, 030, refvfx)")
    a = ap.parse_args()
    pop = json.loads(POPULATION.read_text())
    variants = [v for v in pop["gen_variants"] if (a.only is None or a.only in v)]
    if a.check:
        sys.exit(check())
    write_variants(variants)


if __name__ == "__main__":
    main()
