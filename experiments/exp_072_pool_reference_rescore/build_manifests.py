"""exp_072 — build pool-reference re-scoring manifests (pilot + full chunks).

For each existing ladder eval item (arms r1, r2_ckpt2000, r3_ckpt2000, ic3_a/b/c)
emit one row per pool reference: pool = same-class corpus clips minus the item's
own endpoints clip minus the item's original reference (the IC demo / own GT),
deterministic first-8 by clip name. Everything else (generated_video, conditions)
is copied verbatim, so the certified scorer treats each (gen, ref) pair as a row.
"""
import glob
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
DS66 = REPO / "experiments/exp_066_ladder_v3_scoring/dataset"
OUT = pathlib.Path(__file__).resolve().parent / "dataset"
MAX_REFS = 8
PILOT_CLASSES = {"portal", "shadow_smoke", "super_fast_run"}
PILOT_MAX_REFS = 4
ARMS = {"r1", "r2_ckpt2000", "r3_ckpt2000", "ic3_a", "ic3_b", "ic3_c"}
SOURCES = ["eval_base_c*.json", "eval_r2r3_c*.json", "eval_ic3_abc_c*.json"]


def main():
    corpus = json.load(open(REPO / "data/processed/transitions_std121/corpus_manifest.json"))
    byc = {}
    for key, meta in corpus["clips"].items():
        byc.setdefault(meta["class"], []).append(key)  # "class/clip.mp4"
    for v in byc.values():
        v.sort()

    items = []
    for pat in SOURCES:
        for f in sorted(glob.glob(str(DS66 / pat))):
            for it in json.load(open(f)):
                if it["arm"] in ARMS:
                    items.append(it)
    # de-dup (base manifests can repeat an item across chunks)
    seen, uniq = set(), []
    for it in items:
        if it["item_id"] in seen:
            continue
        seen.add(it["item_id"])
        uniq.append(it)

    def pool_rows(it, max_refs):
        style = it["style"]
        own_clip = it["item_id"].split("__")[2]  # rung__style__clip__seed[__ckpt]
        orig_ref = pathlib.Path(it.get("reference_video") or "").stem
        rows = []
        for key in byc.get(style, []):
            stem = pathlib.Path(key).stem
            if stem in (own_clip, orig_ref):
                continue
            r = dict(it)
            r["item_id"] = f"{it['item_id']}__ref_{stem}"
            r["reference_video"] = f"data/processed/transitions_std121/{key}"
            r["notes"] = f"pool-ref lane (exp_072); source_ref={orig_ref}"
            rows.append(r)
            if len(rows) >= max_refs:
                break
        return rows

    OUT.mkdir(exist_ok=True)
    pilot = [r for it in uniq if it["style"] in PILOT_CLASSES
             for r in pool_rows(it, PILOT_MAX_REFS)]
    json.dump(pilot, open(OUT / "eval_pool_pilot.json", "w"), indent=1)

    full = [r for it in uniq for r in pool_rows(it, MAX_REFS)]
    n_chunks = 6
    per = (len(full) + n_chunks - 1) // n_chunks
    for i in range(n_chunks):
        chunk = full[i * per:(i + 1) * per]
        json.dump(chunk, open(OUT / f"eval_pool_c{i}.json", "w"), indent=1)
    arms = {}
    for r in full:
        arms[r["arm"]] = arms.get(r["arm"], 0) + 1
    print(f"[done] pilot={len(pilot)} rows; full={len(full)} rows in {n_chunks} chunks (~{per}/chunk)")
    print("       rows per arm:", arms)


if __name__ == "__main__":
    main()
