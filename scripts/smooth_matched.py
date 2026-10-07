#!/usr/bin/env python
"""Motion smoothness at MATCHED temporal spacing (store eval): mean cosine between CLIP-B/32 frame embeddings
(namespace clip_b32@r256) of frames ~1/TARGET_FPS apart, stride = max(1, round(fps / TARGET_FPS)), TARGET_FPS = 8:
stride 3 for our 24-fps clips, 1 for the prior works (6.55-9.72 fps). Removes the frame-rate confound of the
consecutive-frame lens (motion_smoothness in eval 040), which favours higher-fps videos. Also emits the native-spacing
value for reference. fps per generation from eval 038 rows.
"""
from __future__ import annotations
import argparse, json, math, sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src")); sys.path.insert(0, str(REPO_ROOT / "scripts"))
from store_eval_common import POP_GRIDV3, parse_stem, harness_arm_of, load_jsonl, write_eval  # noqa: E402

NS = "clip_b32@r256"; TARGET_FPS = 8.0
_fs = None


def _init():
    global _fs
    from diffusion.feature_store import FeatureStore
    _fs = FeatureStore(REPO_ROOT)


def work(task):
    vpath, fps = task
    v = Path(vpath); item_id, seed = parse_stem(v.stem)
    r = dict(item_id=item_id, seed=seed, arm=harness_arm_of(item_id), fps=fps, target_fps=TARGET_FPS, missing=[])
    if fps is None or not math.isfinite(fps): r["missing"].append("fps"); return r
    if not _fs.has(v, NS): r["missing"].append(f"{NS}:gen"); return r
    e = _fs.get(v, NS)["feats"].astype(np.float32)
    stride = max(1, int(round(fps / TARGET_FPS))); r["stride"] = stride
    if len(e) <= stride: r["missing"].append("too_short"); return r
    r["smooth_matched"] = float((e[:-stride] * e[stride:]).sum(axis=1).mean())
    r["smooth_native"] = float((e[:-1] * e[1:]).sum(axis=1).mean())
    r["spacing_ms"] = 1000.0 * stride / fps
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-id", default=None); ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--population", default=str(POP_GRIDV3)); ap.add_argument("--no-index", action="store_true")
    args = ap.parse_args(argv)
    eval_id = args.eval_id or f"smooth_matched_gridv3__dai__{args.date}"   # DRAFT (unnumbered) until finalized
    e038 = sorted((REPO_ROOT / "store/evals").glob("038_handoff_gridv3*"))[-1]
    fps = {}
    for d in e038.iterdir():
        p = d / "rows.jsonl"
        if p.exists():
            for r in load_jsonl(p): fps[(r["item_id"], r["seed"])] = r.get("fps")
    pop = json.loads(Path(args.population).read_text()); results = []
    with Pool(16, initializer=_init) as pool:
        for vrel in pop["gen_variants"]:
            vdir = REPO_ROOT / vrel
            tasks = [(str(v), fps.get(parse_stem(v.stem))) for v in sorted((vdir / "videos").glob("*.mp4"))]
            rows = list(pool.imap(work, tasks, chunksize=16))
            ok = [r["smooth_matched"] for r in rows if "smooth_matched" in r]
            cov = dict(n=len(rows), defined=len(ok), missing=sum(1 for r in rows if r["missing"]), stride=(rows[0].get("stride")),
                       smooth_matched_mean=(round(float(np.mean(ok)), 4) if ok else None))
            results.append(dict(harness_arm=rows[0]["arm"], gen=vrel, rows=rows, coverage=cov))
            print(f"[score] {rows[0]['arm']:<36} {cov}", flush=True)
    ed = write_eval(eval_id, results, created=args.date, instrument="scripts/smooth_matched.py",
                    definition=[ln.strip() for ln in __doc__.splitlines()[1:6] if ln.strip()],
                    why="The prior works' motion smoothness compares consecutive frames, so a 24-fps clip takes smaller steps than a 10-fps one; matched spacing makes the arms comparable.",
                    caveat="Still rewards videos that do not move (a static video scores 1); descriptive quality metric, not a ranking target. Spacing is 125 ms for ours vs 103-153 ms for the prior works (their native step).",
                    extra={"namespace": NS, "target_fps": TARGET_FPS},
                    index_line=None if (args.no_index or not eval_id[:3].isdigit()) else (f"{int(eval_id.split('_', 1)[0])}. `{eval_id}` — motion smoothness at MATCHED temporal spacing on grid v3 ({len(results)} variants): CLIP-B/32 cosine between frames ~125 ms apart (stride round(fps/8)) from stored features; native-spacing value kept alongside. CPU; rows.jsonl are store artifacts; definition in meta.yaml. scripts/smooth_matched.py."))
    print(f"[eval] wrote {ed}"); return 0


if __name__ == "__main__":
    sys.exit(main())
