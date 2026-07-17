"""exp_066 — build SPEC §2 eval manifests for every ladder-v3 arm.

One row per generated video (EvalItemV3: score.py-compatible). Conventions are
the PRE-REGISTERED ones — no new decisions here:
  - base + specialist arms (r0/r1/r1k/r1k_ext/r2/r3): reference_video = the
    item's own ground-truth clip (PLAN §5.2, "as in exp_061").
  - ic arms (ic2/ic3, tiers A/B/C): reference_video = the frozen grid/demo
    reference actually fed to the model (Amendment 2 table).
  - X arms (r3x + ic3_x): style/reference = RECIPIENT class; reference =
    recipient's grid reference clip, identical across the R3X/R4X twin
    (grid v2 r3x.scoring_contract; Amendment 1 §4).
  - condition videos point at the FULL endpoint clip; score.py slices
    first-9 / last-8 (exp_060/061 convention). Suffix condition present iff
    the row was generated two-sided.
Chunking: class-hash split keeps same-class rows (and thus all twin pairs,
which share class+clip+seed) in one chunk. Chunk sizes target <=70 rows so a
--controls auto run fits a 1h59 secondary job with margin.

Run:  python3 build_eval_manifests.py            # writes dataset/eval_*.json
      python3 build_eval_manifests.py --verify eval_r2r3   # existence check
"""

import argparse
import hashlib
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
EXP = pathlib.Path(__file__).resolve().parent
DS = EXP / "dataset"
STD = "data/processed/transitions_std121"
SEEDS = (42, 43, 44)

G2 = json.loads((REPO / "docs/eval_ladder/ladder_items_v2.json").read_text())
G3 = json.loads((REPO / "docs/eval_ladder/ladder_items_v3.json").read_text())
E61 = REPO / "experiments/exp_061_ladder_r0_r1"
V62 = REPO / "outputs/videos/exp_062_ladder_r2r3_specialists"
V63 = REPO / "outputs/videos/exp_063_ladder_r4r5_generalist"
V65 = REPO / "outputs/videos/exp_065_ladder_v3_grid"
M65 = REPO / "experiments/exp_065_ladder_v3_grid/dataset"


def gt(cls, clip):
    return f"{STD}/{cls}/{clip}.mp4"


def row(item_id, video, ref, style, arm, *, prefix=None, suffix=None,
        twin=None, notes=""):
    r = {"item_id": item_id, "generated_video": str(video),
         "reference_video": ref, "style": style,
         "n_endpoints": (1 if prefix else 0) + (1 if suffix else 0),
         "arm": arm, "twin_of": twin, "notes": notes}
    if prefix:
        r["condition_prefix"] = {"video": prefix, "num_frames": 9}
    if suffix:
        r["condition_suffix"] = {"video": suffix, "num_frames": 8}
    return r


def base_rows():
    rows = []
    for arm in ("r0", "r1"):  # exp_061 manifests, verbatim
        rows += json.loads((E61 / f"dataset/eval_manifest_{arm}.json").read_text())
    for cls in G2["r1k"]["classes"]:
        for clip in G2["classes"][cls]["test_items"]:
            for s in SEEDS:
                iid = f"R1K__{cls}__{clip}__s{s}"
                rows.append(row(iid, V62 / "R1K" / f"{iid}.mp4", gt(cls, clip),
                                cls, "r1k", prefix=gt(cls, clip),
                                notes=f"base PE-keyed one_sided; seed={s}"))
    return rows


def r1k_ext_rows():
    rows = []
    for r in json.loads((M65 / "manifest_base_ext.json").read_text())["rows"]:
        cls, clip, po = r["class"], r["clip"], r["prefix_only"]
        for s in SEEDS:
            iid = f"{r['id']}__s{s}"
            rows.append(row(iid, V65 / r["rung"] / f"{iid}.mp4", gt(cls, clip),
                            cls, "r1k_ext", prefix=gt(cls, clip),
                            suffix=None if po else gt(cls, clip),
                            notes=f"base PE-keyed extension; seed={s}"))
    return rows


def r2r3_rows():
    rows = []
    for cls, g in sorted(G2["classes"].items()):
        if "R2" not in g["rungs"]:
            continue
        sfx = bool(g["suffix_conditioning"])
        for rung, clips in (("R2", g["r2_items"]), ("R3", g["test_items"])):
            for clip in clips:
                for s in SEEDS:
                    for step in (250, 2000):
                        iid = f"{rung}__{cls}__{clip}__s{s}__ckpt{step}"
                        rows.append(row(
                            iid, V62 / rung / f"{iid}.mp4", gt(cls, clip), cls,
                            f"{rung.lower()}_ckpt{step}", prefix=gt(cls, clip),
                            suffix=gt(cls, clip) if sfx else None,
                            notes=f"specialist {rung} keyed; seed={s}"))
    return rows


def r3x_rows():
    rows = []
    for rcp, rec in sorted(G3["r3x"]["recipients"].items()):
        # v3 extension recipients carry no recipient_reference key; the frozen
        # grid-v2 reference is the same clip the ic3 X-manifest uses (asserted
        # at build: hero_flight_0 / shadow_smoke_3 / super_fast_run_1).
        ref_clip = rec.get("recipient_reference") or G2["classes"][rcp]["reference"]
        ref = gt(rcp, ref_clip)
        for d in rec["donors"]:
            for s in SEEDS:
                iid = f"R3X__{rcp}__{d['donor_clip']}__s{s}__ckpt2000"
                rows.append(row(
                    iid, V62 / "R3X" / f"{iid}.mp4", ref, rcp, "r3x",
                    prefix=gt(d["donor_class"], d["donor_clip"]),
                    notes=(f"foreign donor {d['donor_class']}/{d['donor_clip']}"
                           f" ({d.get('donor_clip_source') or 'ext-drawn'});"
                           f" no GT; seed={s}")))
    return rows


def ic_rows(manifest, out_root, arm_of, arm_prefix):
    rows = []
    doc = json.loads(manifest.read_text())
    for r in doc["rows"]:
        cls, ep, po = r["class"], r["endpoints"], r["prefix_only"]
        ep_cls = r.get("donor_class", cls)
        ref = gt(r["reference_class"], r["reference"])
        for s in SEEDS:
            iid = f"{r['id']}__s{s}"
            video = out_root / r["rung"] / f"{iid}.mp4"
            rows.append(row(iid, video, ref, cls, arm_of(r),
                            prefix=gt(ep_cls, ep),
                            suffix=None if po else gt(ep_cls, ep),
                            notes=(f"{arm_prefix} rung={r['rung']} "
                                   f"ref={r['reference']}; seed={s}")))
    return rows


def ic2_rows():
    doc = json.loads((REPO / "experiments/exp_063_ladder_r4r5_generalist"
                      / "dataset/ladder_r4r5.json").read_text())
    rows = []
    for r in doc["rows"]:
        for s in SEEDS:
            iid = f"{r['id']}__s{s}"
            video = V63 / r["rung"] / f"{iid}.mp4"
            if not video.exists():   # frozen arm: score exactly what survived
                continue
            rows.append(row(iid, video, gt(r["reference_class"], r["reference"]),
                            r["class"], f"ic2_{r['rung'].lower()}",
                            prefix=gt(r["class"], r["endpoints"]),
                            suffix=None if r["prefix_only"]
                            else gt(r["class"], r["endpoints"]),
                            notes=f"ic2 frozen comparison arm; seed={s}"))
    return rows


def sigma_recheck_rows():
    src = json.loads((REPO / "experiments/exp_060_sigma_seed"
                      / "dataset/eval_manifest.json").read_text())
    rows = []
    for r in src:
        if r.get("style") != "hero_flight":
            continue
        r = dict(r)
        r["item_id"] += "__recheck"
        r["arm"] = "sigma_hero_recheck"
        r["notes"] = (r.get("notes", "") + " | amendment-2 recheck: same rows, "
                      "corrected corpus (hero_flight now twosided)")
        rows.append(r)
    return rows


def chunk_by_class(rows, n):
    chunks = [[] for _ in range(n)]
    for r in rows:
        chunks[int(hashlib.sha1(r["style"].encode()).hexdigest(), 16) % n].append(r)
    return chunks


def training_manifest_ic3():
    pairs = json.loads((REPO / "experiments/exp_064_ic3_aligned_retrain"
                        / "dataset/pairs.json").read_text())
    clips = sorted({f"{p['class']}/{p[k]}.mp4" for p in pairs
                    for k in ("target", "reference")})
    return {"adapter_id": "ic3_step_05000", "base_model": "ltx-2-19b-dev",
            "clips": clips,
            "pairs": [{"target": f"{p['class']}/{p['target']}.mp4",
                       "reference": f"{p['class']}/{p['reference']}.mp4",
                       "conditioning": p["sidedness"]} for p in pairs]}


BUNDLES = {
    "base": (base_rows, 6), "r1k_ext": (r1k_ext_rows, 1),
    "r2r3": (r2r3_rows, 4), "r3x": (r3x_rows, 2),
    "ic3_abc": (lambda: ic_rows(M65 / "manifest_ic3.json", V65,
                                lambda r: f"ic3_{r['tier'].lower()}", "ic3"), 3),
    "ic3_x": (lambda: ic_rows(M65 / "manifest_ic3_x.json", V65,
                              lambda r: "ic3_x", "ic3"), 2),
    "ic2": (ic2_rows, 1), "sigma_hero_recheck": (sigma_recheck_rows, 1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", metavar="LABEL",
                    help="check all generated_video paths exist for eval_<LABEL>*.json")
    args = ap.parse_args()
    DS.mkdir(exist_ok=True)

    if args.verify:
        missing = []
        for f in sorted(DS.glob(f"eval_{args.verify}*.json")):
            for r in json.loads(f.read_text()):
                if not pathlib.Path(r["generated_video"]).exists():
                    missing.append(r["item_id"])
        print(f"[verify {args.verify}] missing: {len(missing)}")
        for m in missing[:20]:
            print("  -", m)
        raise SystemExit(1 if missing else 0)

    total = 0
    for name, (fn, n_chunks) in BUNDLES.items():
        rows = fn()
        total += len(rows)
        parts = chunk_by_class(rows, n_chunks) if n_chunks > 1 else [rows]
        for i, part in enumerate(parts):
            label = name if n_chunks == 1 else f"{name}_c{i}"
            (DS / f"eval_{label}.json").write_text(json.dumps(part, indent=1))
            print(f"[write] eval_{label}.json  {len(part)} rows")
    (DS / "training_manifest_ic3.json").write_text(
        json.dumps(training_manifest_ic3(), indent=1))
    print(f"[done] {total} eval rows total + training_manifest_ic3.json")


if __name__ == "__main__":
    main()
