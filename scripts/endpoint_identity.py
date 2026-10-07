#!/usr/bin/env python
"""Endpoint IDENTITY preservation (store eval): are the given endpoint frames kept in the output?
Per generation, DINOv2-B CLS cosine (namespace dino_cls@dinov2b-r256) between the output frames that SHOULD be the
given frames and the given frames themselves (same index):
  A_given_clip_mean  = mean_t<n_pre cos(out[t], start9[t])            (9 frames on HF rows; the single frame 0 on ED rows / prior works)
  A_given_last_frame = cos(out[n_pre-1], start9[n_pre-1])
  B_given_clip_mean  = mean_j<n_suf cos(out[T-n_suf+j], end9[1+j])     (two-sided rows only; 8 frames)
  TEG baselines (two-sided externals, 2026-09-20): refVFX / Wan FLF2V compare frame 0 with start9[0] and the last frame
  with end9[8]; the VACE first-last CLIP baseline (VACE16) compares out[0..5] / out[77..80] with the 16-fps resamples
  conds_16fps/<endpoint>_start6 / _end4 (start9 idx 0,2,3,5,6,8 / end9 idx 4,5,7,8).
  B_given_first_frame= cos(out[T-n_suf], end9[1])
  A_middle_frame     = cos(out[T//2], start9[n_pre-1])                 (how far the middle has moved from the start; descriptive)
The hand-off identity (first/last K generated frames vs the adjacent given frame) is eval 038 and is NOT recomputed here.
Rows -> store/evals/<EVAL_ID>/<harness_arm>/rows.jsonl (+ meta.yaml, INDEX line). CPU, 16 workers.
"""
from __future__ import annotations
import argparse, json, math, sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src")); sys.path.insert(0, str(REPO_ROOT / "scripts"))
from store_eval_common import (POP_GRIDV3, parse_stem, harness_arm_of, grid_of, grid_type, windows, cond_clips, write_eval)  # noqa: E402

NS = "dino_cls@dinov2b-r256"
_fs = None


def _init():
    global _fs
    from diffusion.feature_store import FeatureStore
    _fs = FeatureStore(REPO_ROOT)


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def work(task):
    vpath, endpoint, sided, gtype = task
    v = Path(vpath)
    n_pre, n_suf = windows(gtype, sided)
    item_id, seed = parse_stem(v.stem)
    r = dict(item_id=item_id, seed=seed, arm=harness_arm_of(item_id), endpoint=endpoint, sided=sided, gtype=gtype, n_pre=n_pre, n_suf=n_suf, missing=[])
    g = _fs.get(v, NS) if _fs.has(v, NS) else None
    cA_p, cB_p = cond_clips(gtype, endpoint)     # start9/end9, or the 16-fps start6/end4 resamples for VACE16
    cA = _fs.get(cA_p, NS) if (cA_p.exists() and _fs.has(cA_p, NS)) else None
    if g is None: r["missing"].append(f"{NS}:gen")
    if cA is None: r["missing"].append(f"{NS}:condA")
    if g is not None and cA is not None:
        fg, fA = g["feats"], cA["feats"]; T = fg.shape[0]; a = n_pre - 1
        r["A_given_clip_mean"] = float(np.mean([_cos(fg[t], fA[t]) for t in range(n_pre)]))
        r["A_given_last_frame"] = _cos(fg[a], fA[a])
        r["A_middle_frame"] = _cos(fg[T // 2], fA[a])
        if n_suf > 0:
            cB = _fs.get(cB_p, NS) if (cB_p.exists() and _fs.has(cB_p, NS)) else None
            if cB is None: r["missing"].append(f"{NS}:condB")
            else:
                fB = cB["feats"]; b = fB.shape[0] - n_suf
                r["B_given_clip_mean"] = float(np.mean([_cos(fg[T - n_suf + j], fB[b + j]) for j in range(n_suf)]))
                r["B_given_first_frame"] = _cos(fg[T - n_suf], fB[b])
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-id", default=None); ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--population", default=str(POP_GRIDV3)); ap.add_argument("--no-index", action="store_true")
    args = ap.parse_args(argv)
    eval_id = args.eval_id or f"endpoint_identity_gridv3__dai__{args.date}"   # DRAFT (unnumbered) until finalized
    pop = json.loads(Path(args.population).read_text())
    per_variant = []
    for vrel in pop["gen_variants"]:
        vdir = REPO_ROOT / vrel; grid = grid_of(vdir); tasks = []
        for v in sorted((vdir / "videos").glob("*.mp4")):
            item_id, _ = parse_stem(v.stem); g = grid.get(item_id)
            if g: tasks.append((str(v), g["endpoint"], g.get("sided", "one"), grid_type(vdir, harness_arm_of(item_id).split("_")[0])))
        per_variant.append((vrel, tasks))
    results = []
    with Pool(16, initializer=_init) as pool:
        for vrel, tasks in per_variant:
            rows = list(pool.imap(work, tasks, chunksize=8))
            fin = lambda k: [r[k] for r in rows if isinstance(r.get(k), float) and math.isfinite(r[k])]
            cov = dict(n=len(rows), A_defined=len(fin("A_given_clip_mean")), B_defined=len(fin("B_given_clip_mean")), missing=sum(1 for r in rows if r["missing"]),
                       A_given_clip_mean=(round(float(np.mean(fin("A_given_clip_mean"))), 4) if fin("A_given_clip_mean") else None),
                       B_given_clip_mean=(round(float(np.mean(fin("B_given_clip_mean"))), 4) if fin("B_given_clip_mean") else None))
            results.append(dict(harness_arm=rows[0]["arm"], gen=vrel, rows=rows, coverage=cov))
            print(f"[score] {rows[0]['arm']:<36} {cov}", flush=True)
    ed = write_eval(eval_id, results, created=args.date, instrument="scripts/endpoint_identity.py",
                    definition=[ln.strip() for ln in __doc__.splitlines()[2:10] if ln.strip()],
                    why="Endpoint fidelity, identity half: the given frames must survive conditioning (a VAE/conditioning round-trip check, complements the hand-off identity of eval 038).",
                    caveat="Whole-frame DINO CLS (no subject mask). Values near 1 are expected for every arm; the prior works and ED rows are one-frame conditioned, so A averages one frame there and B does not exist.",
                    extra={"namespace": NS},
                    index_line=None if (args.no_index or not eval_id[:3].isdigit()) else (f"{int(eval_id.split('_', 1)[0])}. `{eval_id}` — endpoint IDENTITY preservation on grid v3 ({len(results)} variants, {sum(r['coverage']['n'] for r in results)} gens): DINO cosine between the output frames that should be the given frames and the given start9/end9 frames (same index; A = given start clip, B = given end clip on two-sided rows), plus the middle-frame drift. CPU over stored features; rows.jsonl are store artifacts; definition in meta.yaml. scripts/endpoint_identity.py."))
    print(f"[eval] wrote {ed}"); return 0


if __name__ == "__main__":
    sys.exit(main())
