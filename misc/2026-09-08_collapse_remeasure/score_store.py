"""Score existing store generations with the revised null-family instrument (CPU only).

Scope (owner, 2026-09-08): base_cond (all variants + the Aug-24 tier-2 start-only regen),
dualforce_control (all variants), dualforce_dcg_w6 (all variants), dualforce_dcg_w1/w1p5/w3 (v2 neutral).

Output: per_clip.csv (one row per video) + profiles.npz (per-frame resid/tau, keyed by row index).
Run from repo root:  python misc/2026-09-08_collapse_remeasure/score_store.py [--nproc 24]
"""
import os, sys, re, json, glob, argparse, time
import numpy as np
import pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from instrument import load_matrix, measure, md5_file, THETA

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(REPO)

# ---- arms / variants in scope -------------------------------------------------------------
# windows: LTX HF 121f -> prefix 9, suffix 8 (two-sided rows) ; ED81 tier -> prefix 1, one-sided only
VARIANTS = []
def add(arm_dir, variants):
    for v in variants:
        VARIANTS.append((arm_dir, v))
add("005_base_cond", ["01_effect__dai", "02_neutral__dai", "04_neutral_v3__dai", "05_neutral_v3ed81__dai",
                      "06_effect_v3__dai", "07_effect_v3ed81__dai"])
add("013_dualforce_control", ["01_neutral__dai", "02_effect__dai", "03_neutral_v3__dai", "04_neutral_v3ed81__dai",
                              "05_effect_v3__dai", "06_effect_v3ed81__dai"])
add("032_dualforce_dcg_w6", ["01_neutral__dai", "02_effect__dai", "03_neutral_v3__dai", "04_neutral_v3ed81__dai",
                             "05_effect_v3__dai", "06_effect_v3ed81__dai"])
add("029_dualforce_dcg_w1", ["01_neutral__dai"])
add("030_dualforce_dcg_w1p5", ["01_neutral__dai"])
add("031_dualforce_dcg_w3", ["01_neutral__dai"])

TIER2 = dict(arm="base_cond", variant="tier2_start__dai", grid="v2", prompt="neutral",
             vdir="misc/2026-08-24_lerp_collapse/tier2_gen/out",
             gridfile="misc/2026-08-24_lerp_collapse/tier2_gen/tier2_startonly.jsonl")

SEED_RE = re.compile(r"__s(?:eed)?(\d+)\.mp4$")


def variant_meta(variant):
    grid = "v3ed81" if "ed81" in variant else ("v3" if "_v3" in variant else "v2")
    prompt = "effect" if "effect" in variant else "neutral"
    if grid == "v3ed81":
        prefix, suffix = 1, 0
    else:
        prefix, suffix = 9, 8
    return grid, prompt, prefix, suffix


def load_grid(path):
    g = {}
    for l in open(path):
        d = json.loads(l)
        g[d["item_id"]] = d
    return g


def is_foreign(row, item):
    ep = str(row.get("endpoint", ""))
    return (row.get("endpoint_source") == "davis") or ep.startswith("davis_") or ("foreign" in item)


def build_tasks():
    tasks = []
    for arm_dir, variant in VARIANTS:
        vdir = f"store/gens/{arm_dir}/{variant}"
        grid = load_grid(f"{vdir}/grid.jsonl")
        g, p, prefix, suffix = variant_meta(variant)
        arm = arm_dir.split("_", 1)[1]
        for v in sorted(glob.glob(f"{vdir}/videos/*.mp4")):
            base = os.path.basename(v)
            m = SEED_RE.search(base)
            if not m:
                continue
            item = base[: m.start()]
            row = grid.get(item)
            if row is None:
                continue
            sided = row.get("sided")
            two = (sided == "two") and suffix > 0
            cond = "both" if two else "start"
            tasks.append(dict(path=v, arm=arm, variant=variant, grid=g, prompt=p, prefix=prefix, suffix=suffix,
                              two_sided=two, condition=cond, item=item, seed=m.group(1), sided=sided,
                              endpoint=row.get("endpoint"), endpoint_source=row.get("endpoint_source"),
                              endpoint_class=row.get("endpoint_class"), cell=row.get("cell"),
                              reference=row.get("reference"), foreign=is_foreign(row, item)))
    # tier-2 paired start-only regen of the v2 neutral two-sided rows
    grid = load_grid(TIER2["gridfile"])
    for v in sorted(glob.glob(f"{TIER2['vdir']}/**/*.mp4", recursive=True)):
        base = os.path.basename(v)
        m = SEED_RE.search(base)
        if not m:
            continue
        item = base[: m.start()]
        row = grid.get(item, {})
        tasks.append(dict(path=v, arm="base_cond", variant=TIER2["variant"], grid="v2", prompt="neutral",
                          prefix=9, suffix=8, two_sided=False, condition="start", item=item, seed=m.group(1),
                          sided="one(regen)", endpoint=row.get("endpoint"), endpoint_source=row.get("endpoint_source"),
                          endpoint_class=row.get("endpoint_class"), cell=row.get("cell"),
                          reference=row.get("reference"), foreign=is_foreign(row, item)))
    return tasks


def work(task):
    try:
        M = load_matrix(task["path"])
        if M is None:
            return task, dict(error="decode"), None, None
        out, resid, tau = measure(M, task["prefix"], task["suffix"], task["two_sided"])
        out["md5"] = md5_file(task["path"])
        return task, out, resid, tau
    except Exception as e:  # keep going; record the failure
        return task, dict(error=f"{type(e).__name__}: {e}"), None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    tasks = build_tasks()
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[score] {len(tasks)} clips over {len(VARIANTS)} store variants + tier2", flush=True)
    t0 = time.time()
    rows, resids, taus = [], [], []
    with Pool(args.nproc) as pool:
        for i, (task, out, resid, tau) in enumerate(pool.imap_unordered(work, tasks, chunksize=4)):
            r = {k: v for k, v in task.items() if k != "path"}
            r["path"] = task["path"]
            r.update(out)
            r["row"] = len(rows)
            rows.append(r)
            resids.append(resid if resid is not None else np.zeros(0, np.float32))
            taus.append(tau if tau is not None else np.zeros(0, np.float32))
            if (i + 1) % 500 == 0:
                print(f"  ..{i+1}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).sort_values("row").reset_index(drop=True)
    # static guard from the Aug-24 calibration: absolute codec-noise residual of a re-encoded dissolve
    p0 = json.load(open("misc/2026-08-24_lerp_collapse/phase0_results.json"))
    noise = np.median([r["dr_floor"] * r["gap_floor"] for r in p0 if np.isfinite(r["dr_floor"])])
    df["floor_est"] = noise / df["gap"]
    df["static"] = df["floor_est"] > THETA
    from instrument import classify
    df["cls"] = [classify(a, b, c, d) for a, b, c, d in zip(df["DR_med"], df["M"], df["R"], df["static"])]
    df["online"] = (df["DR_med"] <= THETA) & ~df["static"]
    df.to_csv(f"{HERE}/per_clip.csv", index=False)
    np.savez_compressed(f"{HERE}/profiles.npz",
                        resid=np.array(resids, dtype=object), tau=np.array(taus, dtype=object))
    n_err = df["error"].notna().sum() if "error" in df else 0
    print(f"[score] done {len(df)} rows, {n_err} errors, {time.time()-t0:.0f}s -> per_clip.csv", flush=True)
    print(df.groupby(["arm", "variant", "condition"]).size().to_string())


if __name__ == "__main__":
    main()
