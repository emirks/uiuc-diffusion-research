#!/usr/bin/env python
"""Seam z from the STORED temporal LPIPS, at the PHYSICAL given windows (metrics v5, eval 048).
Per generation read ``lpips_t@alex-r256`` ``d`` (the consecutive-frame temporal LPIPS the v4 harness
cached), the physical window (n_pre, n_suf) = store_eval_common.windows(grid_type, sided), and
seam_scores(d, n_pre, max(n_suf, 1)) from diffusion.transition_eval.endpoints. This recomputes the
seam entirely from stored features, bit-identical to v4 where v4 used the physical window, and CORRECTS
evals/030's 9-frame planned prefix on the one-sided prior works to their 1-frame physical prefix
(owner 2026-09-23). Rows -> store/evals/<EVAL_ID>/<harness_arm>/rows.jsonl (+ meta.yaml). CPU, Pool(8).
"""
from __future__ import annotations
import argparse, json, math, sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src")); sys.path.insert(0, str(REPO_ROOT / "scripts"))
from store_eval_common import (parse_stem, harness_arm_of, grid_of, grid_type, windows, write_eval)  # noqa: E402

NS = "lpips_t@alex-r256"
ROSTER = REPO_ROOT / "scripts/metrics_v5/roster.json"
_fs = None


def _init():
    global _fs
    from diffusion.feature_store import FeatureStore
    _fs = FeatureStore(REPO_ROOT)


def work(task):
    from diffusion.transition_eval.endpoints import seam_scores
    vpath, sided, gtype = task
    v = Path(vpath); item_id, seed = parse_stem(v.stem)
    n_pre, n_suf = windows(gtype, sided)
    r = dict(item_id=item_id, seed=seed, arm=harness_arm_of(item_id), gtype=gtype, sided=sided,
             n_pre=n_pre, n_suf=n_suf, missing=[])
    if not _fs.has(v, NS):
        r["missing"].append(f"{NS}:gen"); return r
    d = _fs.get(v, NS)["d"].astype(np.float64)
    s = seam_scores(d, n_pre, max(n_suf, 1))
    pz, sz = s["prefix_seam_z"], s["suffix_seam_z"]
    r["prefix_seam_z"] = pz
    r["suffix_seam_z"] = sz
    r["max_seam_z"] = s["max_seam_z"]
    seam_z = max(pz, sz) if sided == "two" else pz     # two-sided uses both handoffs; one-sided the prefix only
    r["seam_z"] = seam_z
    r["seam_free"] = 1 if seam_z <= 3 else 0
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-id", default=None); ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--roster", default=str(ROSTER)); ap.add_argument("--no-index", action="store_true")
    args = ap.parse_args(argv)
    eval_id = args.eval_id or f"seam_gridv3__dai__{args.date}"
    roster = json.loads(Path(args.roster).read_text())
    variants = []
    for arm in roster["arms"]:
        for g in arm["gens"]:
            if g not in variants:
                variants.append(g)
    results = []
    with Pool(8, initializer=_init) as pool:
        for vrel in variants:
            vdir = REPO_ROOT / vrel
            grid = grid_of(vdir); tasks = []
            for v in sorted((vdir / "videos").glob("*.mp4")):
                item_id, _ = parse_stem(v.stem); g = grid.get(item_id)
                if not g:
                    continue
                gt = grid_type(vdir, harness_arm_of(item_id).split("_")[0])
                tasks.append((str(v), g.get("sided", "one"), gt))
            rows = list(pool.imap(work, tasks, chunksize=16))
            fin = lambda k: [r[k] for r in rows if isinstance(r.get(k), float) and math.isfinite(r[k])]
            defined = [r for r in rows if "seam_z" in r]
            sf = [r["seam_free"] for r in defined]
            cov = dict(n=len(rows), defined=len(defined), missing=sum(1 for r in rows if r["missing"]),
                       gtype=(rows[0]["gtype"] if rows else None),
                       seam_free_pct=(round(100.0 * float(np.mean(sf)), 2) if sf else None),
                       seam_z_med=(round(float(np.median([r["seam_z"] for r in defined])), 4) if defined else None))
            results.append(dict(harness_arm=rows[0]["arm"], gen=vrel, rows=rows, coverage=cov))
            print(f"[score] {rows[0]['arm']:<38} {cov}", flush=True)
    ed = write_eval(eval_id, results, created=args.date,
                    instrument="scripts/metrics_v5/seam_eval.py",
                    definition=[ln.strip() for ln in __doc__.splitlines()[1:9] if ln.strip()],
                    why="Seam-free share and seam z from the stored temporal LPIPS at the physical given windows; the v5 replacement for the v4 harness seam, correcting evals/030's 9-frame prefix on the one-sided prior works.",
                    caveat="seam_scores robust z of the handoff step vs the video's own steps (median/MAD); one-sided rows use the prefix seam only, two-sided the max of prefix and suffix. Missing lpips_t (e.g. the neutral twins, Round 2) -> seam None + a `missing` entry.",
                    extra={"namespace": NS, "window": "store_eval_common.windows (physical given windows)"},
                    index_line=None)
    print(f"[eval] wrote {ed}"); return 0


if __name__ == "__main__":
    sys.exit(main())
