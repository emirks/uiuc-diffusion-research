#!/usr/bin/env python
"""Flow MSE + Action KL from the STORED whole-video features (metrics v5, eval 049).

Two transition-fidelity metrics of the reference clip vs the generated video, over the WHOLE video
(not the transition window), both sampled on their own at fixed fractions of their duration and
featurized once (pairing is a store lookup by the grid row's `reference`):
  flow_mse      = mean over [32,24,32,2] of (100*(flow_gen - flow_ref))^2   -- (% of frame diagonal)^2
  flow_mse_mag  = mean over [32,24,32]   of (100*(|flow_gen| - |flow_ref|))^2   -- magnitude-only (cached)
  action_kl     = KL(p_ref || p_gen) = sum p_ref * (log p_ref - log p_gen), log-probs via log_softmax
  action_kl_rev = KL(p_gen || p_ref) ;  action_js = Jensen-Shannon (nats)
flow from `flow_u32@raft-r256-g24x32`, action prob/logit from `action@swin3db-k400-u32`. Reference clip
= data/processed/transitions_std121/<clip_class(reference)>/<reference>.mp4 (the same resolver
`scripts/grid_v3/gen_dcg_v3.ref_clip_path` uses; grid.jsonl field `reference`). Rows ->
store/evals/<EVAL_ID>/<harness_arm>/rows.jsonl (+ meta.yaml). CPU, Pool(8).
"""
from __future__ import annotations
import argparse, json, math, sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "eval_ladder"))
from store_eval_common import parse_stem, harness_arm_of, grid_of  # noqa: E402

FLOW_NS = "flow_u32@raft-r256-g24x32"
ACT_NS = "action@swin3db-k400-u32"
STD = REPO_ROOT / "data/processed/transitions_std121"
ROSTER = REPO_ROOT / "scripts/metrics_v5/roster.json"
_fs = None
_clip_class = None
_ref_cache = None


def _init():
    global _fs, _clip_class, _ref_cache
    from diffusion.feature_store import FeatureStore
    from prompts import clip_class
    _fs = FeatureStore(REPO_ROOT)
    _clip_class = clip_class
    _ref_cache = {}


# ----- pure metric functions (shared by the workers and the self-checks) -----
def flow_mse_pair(fg: np.ndarray, fr: np.ndarray) -> float:
    d = 100.0 * (fg.astype(np.float64) - fr.astype(np.float64))
    return float(np.mean(d * d))


def flow_mse_mag_pair(fg: np.ndarray, fr: np.ndarray) -> float:
    mg = np.sqrt((fg.astype(np.float64) ** 2).sum(-1))       # [32,24,32]
    mr = np.sqrt((fr.astype(np.float64) ** 2).sum(-1))
    d = 100.0 * (mg - mr)
    return float(np.mean(d * d))


def _log_softmax(logit: np.ndarray) -> np.ndarray:
    x = logit.astype(np.float64)
    x = x - x.max()
    return x - math.log(float(np.exp(x).sum()))


def action_metrics(pg, lg, pr, lr) -> tuple[float, float, float]:
    """(action_kl = KL(p_ref||p_gen), action_kl_rev = KL(p_gen||p_ref), action_js in nats).
    p from the stored prob arrays; log-probs from log_softmax of the stored logits."""
    lpg, lpr = _log_softmax(lg), _log_softmax(lr)
    pg64, pr64 = pg.astype(np.float64), pr.astype(np.float64)
    kl_fwd = float(np.sum(pr64 * (lpr - lpg)))               # KL(p_ref || p_gen)
    kl_rev = float(np.sum(pg64 * (lpg - lpr)))               # KL(p_gen || p_ref)
    m = 0.5 * (pr64 + pg64)
    lm = np.log(np.clip(m, 1e-45, None))
    js = 0.5 * float(np.sum(pr64 * (lpr - lm))) + 0.5 * float(np.sum(pg64 * (lpg - lm)))
    return kl_fwd, kl_rev, js


def _ref_path(reference: str) -> Path:
    return STD / _clip_class(reference) / f"{reference}.mp4"


def _read(video, ns):
    return _fs.get(video, ns) if _fs.has(video, ns) else None


def _read_ref(reference: str):
    """(flow, prob, logit, missing_list) for a reference clip, cached per worker."""
    if reference in _ref_cache:
        return _ref_cache[reference]
    miss = []
    rp = _ref_path(reference)
    zf = _read(rp, FLOW_NS)
    za = _read(rp, ACT_NS)
    if zf is None:
        miss.append(f"{FLOW_NS}:ref")
    if za is None:
        miss.append(f"{ACT_NS}:ref")
    out = (None if zf is None else zf["flow"],
           None if za is None else za["prob"], None if za is None else za["logit"], miss)
    _ref_cache[reference] = out
    return out


def work(task):
    vpath, reference = task
    v = Path(vpath)
    item_id, seed = parse_stem(v.stem)
    r = dict(item_id=item_id, seed=seed, arm=harness_arm_of(item_id), reference=reference, missing=[])
    r.update(flow_mse=None, flow_mse_mag=None, action_kl=None, action_kl_rev=None, action_js=None)
    if reference is None:
        r["missing"].append("reference:grid")
        return r
    # gen features
    gf = _read(v, FLOW_NS)
    ga = _read(v, ACT_NS)
    if gf is None:
        r["missing"].append(f"{FLOW_NS}:gen")
    if ga is None:
        r["missing"].append(f"{ACT_NS}:gen")
    rflow, rprob, rlogit, rmiss = _read_ref(reference)
    r["missing"] += rmiss
    if gf is not None and rflow is not None:
        r["flow_mse"] = flow_mse_pair(gf["flow"], rflow)
        r["flow_mse_mag"] = flow_mse_mag_pair(gf["flow"], rflow)
    if ga is not None and rprob is not None:
        kl_fwd, kl_rev, js = action_metrics(ga["prob"], ga["logit"], rprob, rlogit)
        r["action_kl"], r["action_kl_rev"], r["action_js"] = kl_fwd, kl_rev, js
    return r


def _self_checks():
    """flow_mse(ref, ref) == 0 and action_kl(ref, ref) == 0 EXACTLY for 3 references with features."""
    from diffusion.feature_store import FeatureStore
    from prompts import clip_class
    fs = FeatureStore(REPO_ROOT)
    done, checked = 0, []
    for cls_dir in sorted(STD.iterdir()):
        if not cls_dir.is_dir():
            continue
        for v in sorted(cls_dir.glob("*.mp4")):
            if fs.has(v, FLOW_NS) and fs.has(v, ACT_NS):
                zf = fs.get(v, FLOW_NS)["flow"]
                za = fs.get(v, ACT_NS)
                fm = flow_mse_pair(zf, zf)
                fmm = flow_mse_mag_pair(zf, zf)
                kl_fwd, kl_rev, js = action_metrics(za["prob"], za["logit"], za["prob"], za["logit"])
                assert fm == 0.0 and fmm == 0.0, f"flow_mse(ref,ref)={fm}, mag={fmm} for {v.name}"
                # brief self-check: flow_mse(ref,ref)==0 and action_kl(ref,ref)==0 EXACTLY.
                assert kl_fwd == 0.0 and kl_rev == 0.0, f"action_kl(ref,ref)={kl_fwd}/{kl_rev} != 0 for {v.name}"
                # action_js(ref,ref) is only ~0 (float noise from log(mean) vs log_softmax); not required exact.
                assert abs(js) < 1e-6, f"action_js(ref,ref)={js} unexpectedly large for {v.name}"
                checked.append(v.name)
                done += 1
                break
        if done >= 3:
            break
    assert done >= 3, f"self-check found only {done} references with both namespaces"
    print(f"[self-check] flow_mse(ref,ref)==0 and action_kl(ref,ref)==0 (exact) on {checked}", flush=True)


def main(argv=None) -> int:
    from store_eval_common import write_eval
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-id", default=None)
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--roster", default=str(ROSTER))
    ap.add_argument("--no-index", action="store_true")
    args = ap.parse_args(argv)
    eval_id = args.eval_id or f"049_transition_flow_action_gridv3__dai__{args.date}"

    _init()                     # main-process store + clip_class for the self-checks + task build
    _self_checks()

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
            if not (vdir / "grid.jsonl").exists():
                print(f"[score] {vrel}: no grid.jsonl, skipped", file=sys.stderr)
                continue
            grid = grid_of(vdir)
            tasks = []
            for v in sorted((vdir / "videos").glob("*.mp4")):
                item_id, _ = parse_stem(v.stem)
                g = grid.get(item_id)
                if not g:
                    continue
                tasks.append((str(v), g.get("reference")))
            rows = list(pool.imap(work, tasks, chunksize=16))
            defined = [r for r in rows if not r["missing"]]
            fm = [r["flow_mse"] for r in rows if isinstance(r.get("flow_mse"), float) and math.isfinite(r["flow_mse"])]
            ak = [r["action_kl"] for r in rows if isinstance(r.get("action_kl"), float) and math.isfinite(r["action_kl"])]
            cov = dict(n=len(rows), defined=len(defined), missing=sum(1 for r in rows if r["missing"]),
                       flow_mse_mean=(round(float(np.mean(fm)), 4) if fm else None),
                       action_kl_mean=(round(float(np.mean(ak)), 4) if ak else None))
            results.append(dict(harness_arm=rows[0]["arm"], gen=vrel, rows=rows, coverage=cov))
            print(f"[score] {rows[0]['arm']:<40} {cov}", flush=True)
    ed = write_eval(eval_id, results, created=args.date,
                    instrument="scripts/metrics_v5/flow_action_eval.py",
                    definition=[ln.strip() for ln in __doc__.splitlines()[2:12] if ln.strip()],
                    why="Whole-video transition-fidelity metrics of the reference clip vs the generated video: optical-flow "
                        "MSE (Flow MSE) and Kinetics-400 action-class KL (Action KL), both from stored features (evals 049).",
                    caveat="Whole-video comparison (not the transition window), scene-layout dependent for flow (the pooled "
                           "24x32 flow grid assumes a shared frame layout; gens are duration- and resolution-matched to their "
                           "reference); the Kinetics-400 action classes are a video-motion SIGNATURE, not effect names. KL is "
                           "asymmetric: action_kl = KL(p_ref||p_gen); action_kl_rev / action_js also stored.",
                    extra={"flow_namespace": FLOW_NS, "action_namespace": ACT_NS,
                           "reference": "data/processed/transitions_std121/<clip_class(reference)>/<reference>.mp4"},
                    index_line=None)
    print(f"[eval] wrote {ed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
