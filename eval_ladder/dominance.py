"""ladder2 — Amendment-1 Pass A: clip-level dominance diagnostic.

WHY THIS EXISTS. The pre-registered generalist claim was a donor-pool margin vs the base twin.
That statistic is invalid for every reference-bearing cell: without the adapter, LTX-2 treats the
in-context clip as CONTENT TO CONTINUE and reproduces the demo's own scene instead of the
conditioned endpoint. Copying a donor-class clip scores well against the donor pool, so base is
rewarded for ignoring the task and ic_gen is penalised for doing it. The harness's absolute
`near_copy` flag (tau 0.858, calibrated for verbatim frame copies) cannot catch scene-level
reproduction sitting around cos 0.5.

WHAT THIS MEASURES (per Amendment-1, locked before any recipient-pool score existed):

    ep_align  = mean over GEN MIDDLE frames of ( max over ENDPOINT-clip frames of cos )
    ref_align = mean over GEN MIDDLE frames of ( max over REFERENCE NON-CORE frames of cos )
    ref_dominated := ref_align > ep_align          (threshold 0, pre-committed)

MEAN-of-best-match, not max-of-max: max aggregation is exactly why a whole reproduced scene slides
under tau_copy. The mean asks "what is the middle of this video actually made of?".

Masks follow the certified harness verbatim: `mid_mask` for the generation (everything outside the
conditioned windows) and `~core_mask` for the reference (the demo's own scenes A/B). `copy_max` is
a certified metric and is NOT touched — these are new columns beside it.

    python eval_ladder/dominance.py --mode passA
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
WT = REPO_ROOT / ".claude/worktrees/eval-v4-cert/src"
sys.path.insert(0, str(WT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import prompts  # noqa: E402
from diffusion.transition_eval.features import DinoExtractor, video_features  # noqa: E402
from diffusion.transition_eval.m2_integrity import mid_mask  # noqa: E402
from diffusion.transition_eval.morph import core_mask, morph_profile  # noqa: E402

STD = REPO_ROOT / "data/processed/transitions_std121"
# exp_078: env-driven so the premise test can target a side arm (b1/b1r) in a private video root.
# Unset => ladder2 / ic_gen, byte-identical to prior behavior.
GENS = Path(os.environ.get("LADDER_OUT_ROOT", REPO_ROOT / "outputs/videos/ladder2"))
DOM_ARM = os.environ.get("DOM_ARM", "ic_gen")          # the treatment arm to score for C/T
DOM_STEP = os.environ.get("LADDER_GEN_STEP")            # __ck<step> dir suffix for a diagnostic ckpt
DOM_OUT = os.environ.get("DOM_OUT", str(HERE))          # where passB manifests/results are written
CACHE = REPO_ROOT / "outputs/eval/cache"
OUT = HERE / "dominance_passA.jsonl"
N_PREFIX, N_SUFFIX = 9, 8


def _gen_dir(arm: str) -> str:
    """Directory an arm's videos live in, honoring the LADDER_GEN_STEP suffix for side arms."""
    if DOM_STEP and arm not in ("base", "ic_gen"):
        return f"{arm}__ck{DOM_STEP}"
    return arm


def feats_of(path: Path, ext: DinoExtractor) -> np.ndarray:
    f, _ = video_features(path, CACHE, ext)
    return f


def plan_b() -> None:
    """Amendment-1 Pass B manifests: score each cross-cell generation against the RECIPIENT pool
    (endpoint_class corpus, excluding the endpoint clip AND the reference clip), plus the four
    per-input_key ANCHORS (reference clip and endpoint clip scored against both pools, self-excluded).
    T and C are normalised by those anchors, so they must be scored with the same instrument."""
    rows = [json.loads(x) for x in (HERE / "registry.jsonl").read_text().splitlines() if x.strip()]
    base_by_key = {r["input_key"]: r for r in rows if r["arm"] == "base"}
    CROSS = {"G-unseen-cross", "G-zs-cross"}
    items, anchors_done = [], set()

    def pool(cls: str, banned: set) -> list[str]:
        clips = sorted(p.stem for p in (STD / cls).glob("*.mp4"))
        return [c for c in clips if c not in banned][:8]

    # exp_078: the arm whose C/T we are computing. For ic_gen it also scores the base twin (original
    # amendment-1 behavior). For a side arm (b1/b1r) we score ONLY that arm's generations vs the
    # recipient pool — the anchors (R_ref/R_ep/D_ref/D_ep) are arm-independent and reused from the
    # existing dominance_passB.json, and the ic_gen comparator C is already recorded there.
    for r in rows:
        if r["arm"] != DOM_ARM or r["cell"] not in CROSS:
            continue
        twin = base_by_key.get(r["input_key"])
        rec_cls = r["endpoint_class"]
        banned = {r["reference"], r["endpoint"]}
        # (a) generations vs the RECIPIENT pool — the treatment arm always; base twin only for ic_gen
        arm_pairs = [(r, DOM_ARM)]
        if DOM_ARM == "ic_gen" and twin is not None:
            arm_pairs.append((twin, "base"))
        for arm_row, arm in arm_pairs:
            if arm_row is None:
                continue
            for seed in (42, 43):
                g = GENS / _gen_dir(arm_row["arm"]) / f"{arm_row['item_id']}__s{seed}.mp4"
                if not g.exists():
                    continue
                for ref_clip in pool(rec_cls, banned):
                    items.append({
                        "item_id": f"RECP__{arm}__{r['item_id']}__s{seed}__ref_{ref_clip}",
                        "generated_video": str(g),
                        "reference_video": f"data/processed/transitions_std121/{rec_cls}/{ref_clip}.mp4",
                        "style": rec_cls, "arm": f"recp_{arm}",
                        "n_endpoints": 2 if r["sided"] == "two" else 1,
                        "notes": f"Amendment-1 PassB recipient-pool {r['cell']}"})
        # (b) anchors, once per input_key: the reference clip and the endpoint clip themselves,
        #     each scored against BOTH pools with self excluded
        if r["input_key"] in anchors_done:
            continue
        anchors_done.add(r["input_key"])
        for role, clip, cls_of in (("ref", r["reference"], prompts.clip_class(r["reference"])),
                                   ("ep", r["endpoint"], rec_cls)):
            src = STD / cls_of / f"{clip}.mp4"
            if not src.exists():
                continue
            for pool_name, pool_cls in (("D", r["donor_class"]), ("R", rec_cls)):
                for ref_clip in pool(pool_cls, {clip}):
                    items.append({
                        "item_id": f"ANCH__{pool_name}_{role}__{r['input_key']}__ref_{ref_clip}",
                        "generated_video": str(src),
                        "reference_video": f"data/processed/transitions_std121/{pool_cls}/{ref_clip}.mp4",
                        "style": pool_cls, "arm": f"anchor_{pool_name}{role}",
                        "n_endpoints": 2 if r["sided"] == "two" else 1,
                        "notes": "Amendment-1 PassB anchor"})
    d = Path(os.environ.get("DOM_EVAL_DIR", HERE / "eval_recp"))   # exp_078: private for b1/b1r
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("eval_c*.json"):
        old.unlink()
    for i in range(6):
        (d / f"eval_c{i}.json").write_text(json.dumps(items[i::6], indent=1))
    print(f"[planB] arm={DOM_ARM} {len(items)} rows ({len(anchors_done)} input_keys anchored) "
          f"-> 6 chunks in {d}")


def report_b() -> None:
    """Amendment-1 Pass B verdict. Formulas and thresholds were locked in the dossier BEFORE any
    recipient-pool score existed; nothing here was chosen after seeing a number."""
    import run_eval

    ceil = run_eval.ceilings()
    reg = {r["item_id"]: r for r in run_eval.load_registry()}
    base_by_key = {r["input_key"]: r for r in run_eval.load_registry() if r["arm"] == "base"}

    def pooled(dirs, prefix):
        """-> {key: mean app_ref} for item_ids starting with `prefix`."""
        acc = collections.defaultdict(list)
        for d in dirs:
            for f in Path(d).glob("*/items.jsonl"):
                for line in f.read_text().splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if not r["item_id"].startswith(prefix) or r.get("app_ref") is None:
                        continue
                    head, _, _ref = r["item_id"].rpartition("__ref_")
                    acc[head].append(r["app_ref"])
        return {k: st.mean(v) for k, v in acc.items()}

    # exp_078: env-overridable so the b1/b1r premise test reads its private recipient + donor scores.
    RECP = Path(os.environ.get("DOM_RECP_SCORES", REPO_ROOT / "outputs/eval/ladder2_recp"))
    DON = Path(os.environ.get("DOM_DON_SCORES", REPO_ROOT / "outputs/eval/ladder2"))
    recp = pooled([RECP], "RECP__")          # RECP__<arm>__<item_id>__s<seed>
    anch = pooled([RECP], "ANCH__")          # ANCH__<D|R>_<ref|ep>__<input_key>
    donor = pooled([DON], "")                # <item_id>__s<seed>

    rowsB, degen = [], 0
    for item_id, row in reg.items():
        if row["arm"] != DOM_ARM or row["cell"] not in ("G-unseen-cross", "G-zs-cross"):
            continue
        k = row["input_key"]
        rec_cls, don_cls = row["endpoint_class"], row["donor_class"]
        if don_cls not in ceil or rec_cls not in ceil:
            continue
        a = {n: anch.get(f"ANCH__{n}__{k}") for n in ("D_ref", "D_ep", "R_ref", "R_ep")}
        if any(v is None for v in a.values()):
            continue
        D_ref, D_ep = a["D_ref"] / ceil[don_cls], a["D_ep"] / ceil[don_cls]
        R_ref, R_ep = a["R_ref"] / ceil[rec_cls], a["R_ep"] / ceil[rec_cls]
        if abs(D_ref - D_ep) < 0.05 or abs(R_ep - R_ref) < 0.05:   # pre-committed 5pp guard
            degen += 1
            continue
        twin = base_by_key.get(k)
        per_arm = {}
        arm_ids = [(DOM_ARM, item_id)]
        if DOM_ARM == "ic_gen":
            arm_ids.append(("base", twin["item_id"] if twin else None))
        for arm, rid in arm_ids:
            if rid is None:
                continue
            Ds = [donor.get(f"{rid}__s{s}") for s in (42, 43)]
            Rs = [recp.get(f"RECP__{arm}__{item_id}__s{s}") for s in (42, 43)]
            Ds = [x for x in Ds if x is not None]; Rs = [x for x in Rs if x is not None]
            if not Ds or not Rs:
                continue
            D = st.mean(Ds) / ceil[don_cls]
            R = st.mean(Rs) / ceil[rec_cls]
            T = min(1.0, max(0.0, (D - D_ep) / (D_ref - D_ep)))
            C = min(1.0, max(0.0, (R - R_ref) / (R_ep - R_ref)))
            quad = ("transfer" if T >= .5 and C >= .5 else
                    "reference won" if T >= .5 else
                    "endpoint-prior won" if C >= .5 else "mush")
            per_arm[arm] = {"T": T, "C": C, "TI": min(T, C), "quad": quad}
        need = 2 if DOM_ARM == "ic_gen" else 1
        if len(per_arm) == need:
            rowsB.append({"item_id": item_id, "cell": row["cell"], "donor": don_cls, **per_arm})

    if not rowsB:
        sys.exit(f"[reportB] no complete {DOM_ARM} rows yet ({degen} anchor_degenerate)")

    print(f"\n=== AMENDMENT-1 PASS B — transfer index (n={len(rowsB)} paired items, "
          f"{degen} anchor_degenerate excluded) ===")
    print(f"{'cell':16s} {'arm':7s} {'n':>4s} {'T':>6s} {'C':>6s} {'TI':>6s}   quadrants")
    print("-" * 78)
    arms_here = ["ic_gen", "base"] if DOM_ARM == "ic_gen" else [DOM_ARM]
    for cell in ("G-unseen-cross", "G-zs-cross"):
        sub = [r for r in rowsB if r["cell"] == cell]
        if not sub:
            continue
        for arm in arms_here:
            q = collections.Counter(r[arm]["quad"] for r in sub)
            print(f"{cell:16s} {arm:7s} {len(sub):4d} {st.mean(r[arm]['T'] for r in sub):6.2f} "
                  f"{st.mean(r[arm]['C'] for r in sub):6.2f} {st.mean(r[arm]['TI'] for r in sub):6.2f}   "
                  + ", ".join(f"{k} {v}" for k, v in q.most_common()))
        # ΔTI vs base only defined for the ic_gen run (which scores both arms).
        if DOM_ARM == "ic_gen":
            per_donor = collections.defaultdict(list)
            for r in sub:
                per_donor[r["donor"]].append(r["ic_gen"]["TI"] - r["base"]["TI"])
            d = [r["ic_gen"]["TI"] - r["base"]["TI"] for r in sub]
            pos = sum(1 for v in per_donor.values() if st.mean(v) > 0)
            print(f"{'':16s} {'ΔTI':7s} {len(sub):4d} {'':6s} {'':6s} {st.mean(d)*100:+5.1f}pp  "
                  f"donors positive {pos}/{len(per_donor)}\n")
    out_path = Path(DOM_OUT) / ("dominance_passB.json" if DOM_ARM == "ic_gen"
                                else f"dominance_passB_{DOM_ARM}.json")
    out_path.write_text(json.dumps(rowsB, indent=1))
    print(f"[reportB] wrote {out_path}")
    print("TI = min(T, C), locked pre-scoring. T = donor manner arrived; C = endpoint content kept.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["passA", "planB", "reportB"], default="passA")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "planB":      # manifest-only; must not do Pass-A's globbing first
        plan_b()
        return
    if args.mode == "reportB":
        report_b()
        return

    rows = [json.loads(x) for x in (HERE / "registry.jsonl").read_text().splitlines() if x.strip()]
    base_by_key = {r["input_key"]: r for r in rows if r["arm"] == "base"}
    # every reference-bearing row, BOTH arms (advisor: apply symmetrically, including to cells
    # whose result looked flattering — G-ref-control's +4.1pp came from the same mechanism)
    work = []
    for r in rows:
        if r["arm"] != "ic_gen" or not r.get("reference"):
            continue
        twin = base_by_key.get(r["input_key"])
        for arm_row, arm in ((r, "ic_gen"), (twin, "base")):
            if arm_row is None:
                continue
            for seed in (42, 43):
                p = GENS / arm_row["arm"] / f"{arm_row['item_id']}__s{seed}.mp4"
                if p.exists():
                    work.append((r, arm, seed, p))
    if args.limit:
        work = work[: args.limit]
    print(f"[passA] {len(work)} (generation, arm) rows to measure")

    ext = DinoExtractor()
    out, n_err = [], 0
    for i, (row, arm, seed, gen_path) in enumerate(work, 1):
        try:
            ep_clip = STD / prompts.clip_class(row["endpoint"]) / f"{row['endpoint']}.mp4"
            ref_clip = STD / prompts.clip_class(row["reference"]) / f"{row['reference']}.mp4"
            g = feats_of(gen_path, ext)
            e = feats_of(ep_clip, ext)
            f = feats_of(ref_clip, ext)
            n_suf = N_SUFFIX if row["sided"] == "two" else 0
            gm = mid_mask(len(g), N_PREFIX, n_suf)
            rprof = morph_profile(f, n_prefix=N_PREFIX,
                                  n_suffix=N_SUFFIX if row["sided"] == "two" else N_SUFFIX,
                                  n_endpoints=2 if row["sided"] == "two" else 1)
            rnon = ~core_mask(rprof)
            gi = np.flatnonzero(gm)
            if len(gi) == 0 or not rnon.any():
                continue
            ep_align = float((g[gi] @ e.T).max(axis=1).mean())
            ref_align = float((g[gi] @ f[np.flatnonzero(rnon)].T).max(axis=1).mean())
            out.append({"item_id": row["item_id"], "cell": row["cell"], "arm": arm, "seed": seed,
                        "donor": row["donor_class"], "endpoint_class": row["endpoint_class"],
                        "ep_align": ep_align, "ref_align": ref_align,
                        "ref_dominated": bool(ref_align > ep_align)})
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            if n_err <= 3:
                print(f"  [err] {gen_path.name}: {type(exc).__name__}: {exc}")
        if i % 50 == 0:
            print(f"  {i}/{len(work)}")

    OUT.write_text("".join(json.dumps(o) + "\n" for o in out))
    print(f"[passA] wrote {len(out)} rows -> {OUT.relative_to(REPO_ROOT)} ({n_err} errors)")

    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for o in out:
        agg[(o["cell"], o["arm"])]["ep"].append(o["ep_align"])
        agg[(o["cell"], o["arm"])]["ref"].append(o["ref_align"])
        agg[(o["cell"], o["arm"])]["dom"].append(o["ref_dominated"])
    print(f"\n{'cell':18s} {'arm':7s} {'n':>4s} {'ep_align':>9s} {'ref_align':>10s} {'ref_dominated':>14s}")
    print("-" * 68)
    for (cell, arm) in sorted(agg):
        d = agg[(cell, arm)]
        rate = sum(d["dom"]) / len(d["dom"])
        print(f"{cell:18s} {arm:7s} {len(d['ep']):4d} {st.mean(d['ep']):9.3f} "
              f"{st.mean(d['ref']):10.3f} {rate:13.0%}")
    print("\nref_dominated = the generation's middle frames resemble the DEMO more than the ENDPOINT")
    print("Advisor gate: if base's rate is below ~50%, the leakage story is weaker than believed")
    print("             -> re-consult BEFORE publishing the amendment's interpretation.")


if __name__ == "__main__":
    main()
