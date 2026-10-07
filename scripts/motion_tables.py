#!/usr/bin/env python
"""Motion tables (grid v3) in the standard two-table format: own arms by tier (HF+ED pooled) and prior works on
the shared one-sided zero-shot set. Columns (all from store evals over stored features):
  Endpoint Motion A/B   eval 041 (dense flow of the given windows)         -- '--' until the eval exists / where not measurable
  Motion fid. (vs ref)  eval 040 det_motion_fidelity (Yatim et al.)         -- NaN where nothing moves (n per cell)
  Motion smooth. (8fps) consecutive-frame CLIP-B/32 cosine at MATCHED temporal spacing (stride = round(fps/8)) from the stored features
  Dynamic degree (px/s) eval 040 dynamic_degree_mean_mag (RAFT px/step at 256 px) x fps    -- descriptive, no direction
  Seam-free %           per_gen max_seam_z <= 3 (evals 028/030)             -- and Seam z (median)
Writes papers_drafts/_preview/motion_gridv3/{tab_M1,tab_M2}.tex, TABLES.md, simple.pdf.
"""
from __future__ import annotations
import argparse, glob, json, math, os, subprocess, sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
EVALS = REPO_ROOT / "store" / "evals"
OUT = REPO_ROOT / "papers_drafts" / "_preview" / "motion_gridv3"
TARGET_FPS = 8.0
CLIP_NS = "clip_b32@r256"

OWN = [("LTX-2 (no reference)", "base_cond"), ("LTX-2 baseline LoRA", "ic_gen"), ("SEGUE w/o guidance", "dualforce_control"), ("SEGUE", "dualforce_dcg_w6")]
EXT = [("Video-As-Prompt", "vap"), ("VFXMaster", "vfxmaster"), ("refVFX", "refvfx")]
TEX = {"LTX-2 (no reference)": "LTX-2 (no reference)", "LTX-2 baseline LoRA": "LTX-2 baseline LoRA", "SEGUE w/o guidance": r"\segue{} w/o guidance",
       "SEGUE": r"\segue{}", "Video-As-Prompt": "Video-As-Prompt", "VFXMaster": "VFXMaster", "refVFX": "refVFX"}
GENS = {"base_cond": ("store/gens/005_base_cond/04_neutral_v3__dai", "store/gens/005_base_cond/05_neutral_v3ed81__dai"),
        "ic_gen": ("store/gens/001_ic_gen/03_neutral_v3__dai", "store/gens/001_ic_gen/04_neutral_v3ed81__dai"),
        "dualforce_control": ("store/gens/013_dualforce_control/03_neutral_v3__dai", "store/gens/013_dualforce_control/04_neutral_v3ed81__dai"),
        "dualforce_dcg_w6": ("store/gens/032_dualforce_dcg_w6/03_neutral_v3__dai", "store/gens/032_dualforce_dcg_w6/04_neutral_v3ed81__dai"),
        "vap": ("store/gens/011_vap/05_author_native__dai",), "vfxmaster": ("store/gens/012_vfxmaster/05_author_native__dai",),
        "refvfx": ("store/gens/003_refvfx/03_author_native__dai",)}
# (key, header, direction, fmt)
COLS = [("ep_A", "Endpoint Motion A", "max", "3"), ("ep_B", "Endpoint Motion B", "max", "3"),
        ("motfid", r"Motion fid.\ (vs ref)", "max", "3"), ("smooth8", r"Motion smooth.\ (8\,fps)", "max", "3"),
        ("dyn_pxs", "Dynamic degree (px/s)", "none", "1"), ("seam_free", "Seam-free \\%", "max", "1"), ("seam_med", "Seam $z$ (median)", "min", "2")]


def _read_meta(path: Path) -> dict:
    out = {}
    for ln in path.read_text().splitlines():
        if ln[:1].isspace() or ":" not in ln: continue
        k, v = ln.split(":", 1); out[k.strip()] = v.split("#", 1)[0].strip().strip("'\"")
    return out


def _parse_stem(stem: str):
    item, s = stem.rsplit("__s", 1); return item, int(s)


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _find_eval(prefix: str) -> Path | None:
    """numbered eval first; else the DRAFT under store/evals/_draft (unnumbered name = prefix without its NNN_)."""
    c = sorted(EVALS.glob(prefix + "*")) or sorted((EVALS / "_draft").glob(prefix.split("_", 1)[1] + "*")); return c[-1] if c else None


_fs = None
def _init():
    global _fs
    from diffusion.feature_store import FeatureStore
    _fs = FeatureStore(REPO_ROOT)


def _smooth8(task):
    vpath, fps = task
    v = Path(vpath)
    if not _fs.has(v, CLIP_NS) or fps is None or not math.isfinite(fps): return (vpath, float("nan"))
    e = _fs.get(v, CLIP_NS)["feats"].astype(np.float32)
    stride = max(1, int(round(fps / TARGET_FPS)))
    if len(e) <= stride: return (vpath, float("nan"))
    cos = (e[:-stride] * e[stride:]).sum(axis=1)
    return (vpath, float(cos.mean()))


def collect(with_smooth: bool = True) -> dict:
    """-> {(harness_arm, item_id, seed): {metrics..., tier, sided, cell, endpoint, reference, gtype}}"""
    recs = {}
    e040 = _find_eval("040_lenses_gridv3"); e038 = _find_eval("038_handoff_gridv3"); e041 = _find_eval("041_endpoint_motion_gridv3")
    fps = {}
    for d in e038.iterdir():
        p = d / "rows.jsonl"
        if p.exists():
            for r in _load_jsonl(p): fps[(d.name, r["item_id"], r["seed"])] = r.get("fps")
    per_gen = {}
    for p in list(EVALS.glob("028_grid_v3_paper_arms*/*/per_gen.jsonl")) + list(EVALS.glob("030_external_zs_authornative*/*/per_gen.jsonl")):
        for r in _load_jsonl(p): per_gen[(p.parent.name, r["item_id"], r["seed"])] = r
    ep = {}
    if e041 is not None:
        for d in e041.iterdir():
            p = d / "rows.jsonl"
            if p.exists():
                for r in _load_jsonl(p): ep[(d.name, r["item_id"], r["seed"])] = r
    tasks = []
    for label, base in OWN + EXT:
        for vrel in GENS[base]:
            vdir = REPO_ROOT / vrel
            grid = {r["item_id"]: r for r in _load_jsonl(vdir / "grid.jsonl")}
            videos = sorted((vdir / "videos").glob("*.mp4"))
            harm = _parse_stem(videos[0].stem)[0].split("__")[1]     # harness arm = 2nd field of the item id (frozen stamp)
            lens = {}
            lp = e040 / harm / "rows.jsonl"
            if lp.exists():
                for r in _load_jsonl(lp): lens[(r["item_id"], r["seed"])] = r
            for v in videos:
                item_id, seed = _parse_stem(v.stem); g = grid.get(item_id)
                if not g: continue
                k = (harm, item_id, seed); ln = lens.get((item_id, seed), {}); pg = per_gen.get(k, {}); f = fps.get(k) or float("nan")
                # seam z exactly as eval 038 / the main tables: prefix seam for one-sided rows, the worse of both seams for two-sided
                pz, sz = pg.get("prefix_seam_z"), pg.get("suffix_seam_z")
                if g.get("sided") == "two":
                    z = (max(pz, sz) if (pz is not None and sz is not None and math.isfinite(pz) and math.isfinite(sz)) else None)
                else:
                    z = pz if (pz is not None and math.isfinite(pz)) else None
                rec = dict(label=label, base=base, video=str(v), tier=g.get("ref_novelty"), sided=g.get("sided", "one"), cell=g.get("cell"),
                           endpoint=g["endpoint"], reference=g.get("reference"), gtype=("external" if base in ("vap", "vfxmaster", "refvfx") else ("ED" if "ed81" in vdir.name else "HF")), fps=f,
                           motfid=ln.get("det_motion_fidelity"), smooth_native=ln.get("motion_smoothness"),
                           dyn_pxs=(ln["dynamic_degree_mean_mag"] * f) if (ln.get("dynamic_degree_mean_mag") is not None and math.isfinite(f)) else None,
                           seam_z=z, seam_free=(100.0 if z is not None and z <= 3 else (0.0 if z is not None else None)),
                           ep_A=ep.get(k, {}).get("A_agree"), ep_B=ep.get(k, {}).get("B_agree"))
                recs[k] = rec; tasks.append((str(v), f))
    if with_smooth:
        cache = OUT / "smooth8.json"
        sm = json.loads(cache.read_text()) if cache.exists() else {}
        todo = [t for t in tasks if t[0] not in sm]
        if todo:
            with Pool(16, initializer=_init) as pool:
                sm.update(dict(pool.imap_unordered(_smooth8, todo, chunksize=16)))
            cache.write_text(json.dumps(sm))
        for k, rec in recs.items(): rec["smooth8"] = sm.get(rec["video"])
    return recs


def agg(vals, how="mean"):
    v = [x for x in vals if isinstance(x, (int, float)) and x is not None and math.isfinite(x)]
    if not v: return (None, 0)
    return ((float(np.median(v)) if how == "median" else float(np.mean(v))), len(v))


def cell_vals(rs):
    out = []
    for key, _, _, _ in COLS:
        if key == "seam_med": out.append(agg([r.get("seam_z") for r in rs], "median"))
        else: out.append(agg([r.get(key) for r in rs]))
    return out


def block(cells, labels, n_all, md=False):
    best = []
    for ci, (_, _, d, _) in enumerate(COLS):
        vals = [cells[l][ci][0] for l in labels if cells[l][ci][0] is not None]
        best.append((max(vals) if d == "max" else min(vals)) if (vals and d != "none") else None)
    out = []
    for l in labels:
        parts = []
        for ci, (_, _, d, fmt) in enumerate(COLS):
            m, n = cells[l][ci]
            if m is None: parts.append("--"); continue
            s = f"{m:.{fmt}f}"
            if best[ci] is not None and abs(m - best[ci]) <= 1e-9: s = ("**" + s + "**") if md else (r"\textbf{" + s + "}")
            if n != n_all[l]: s += (f" (n={n})" if md else r"{\tiny\,($n{=}" + str(n) + "$)}")
            parts.append(s)
        out.append((l, parts, n_all[l]))
    return out


NOTE = (r"\emph{Endpoint Motion A/B}: dense RAFT flow of the given window, output vs. given clip, agreement $=1-\sum\|\Delta F\|/\sum(\|F_\mathrm{out}\|+\|F_\mathrm{given}\|)$ over moving pixels "
        r"(1 = identical motion; NaN when the given window is static; one-frame conditioning has no motion: --). "
        r"\emph{Motion fid.\ (vs ref)}: Yatim et al.\ motion fidelity, CoTracker tracklet velocity-direction correlation between the output and the reference, unpaired; NaN where nothing moves (n per cell). "
        r"\emph{Motion smooth.\ (8\,fps)}: mean cosine between CLIP-B/32 embeddings of frames $\approx$125\,ms apart (stride $=\mathrm{round}(\mathrm{fps}/8)$: 3 for our 24\,fps clips, 1 for the prior works at 6.5--9.7\,fps), "
        r"so arms are compared at matched temporal spacing; a video that does not move scores 1. "
        r"\emph{Dynamic degree}: mean RAFT flow magnitude per step at 256\,px, times fps, in px/s -- how much moves, descriptive only (no bold). "
        r"\emph{Seam-free \%}: share of generations whose hand-off step has a temporal-LPIPS robust $z \le 3$ against the video's own steps; \emph{Seam $z$ (median)}: the median of that $z$ (lower is better). "
        r"$\uparrow$ higher is better, $\downarrow$ lower is better; bold: best per column and block.")


def render(recs: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hdr = r"arm & " + " & ".join(f"{h} " + (r"$\uparrow$" if d == "max" else (r"$\downarrow$" if d == "min" else "")) for _, h, d, _ in COLS) + r" & $n$ \\"
    md_hdr = "| arm | " + " | ".join(f"{h.replace(chr(92)+',', ' ').replace(chr(92), '')} " + ("↑" if d == "max" else ("↓" if d == "min" else "")) for _, h, d, _ in COLS) + " | n |"
    md = ["# Motion tables (grid v3 preview)", "", f"_Generated {date.today().isoformat()} by `scripts/motion_tables.py`._", ""]
    # Table 1
    L = [r"\begin{table}[H]", r"\centering", r"\caption{\textbf{Motion metrics, own arms by tier (neutral prompt, grid v3; HF-121f and ED-81f rows pooled per tier).} " + NOTE + "}",
         r"\label{tab:motion_own}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\linewidth}{!}{\begin{tabular}{l ccccccc c}", r"\toprule", hdr, r"\midrule"]
    md += ["## Table 1 -- own arms by tier", md_hdr, "|---" * (len(COLS) + 2) + "|"]
    for tier, tname in (("seen", "Seen"), ("unseen", "Unseen"), ("zero_shot", "Zero-shot")):
        cells, n_all = {}, {}
        for label, _ in OWN:
            rs = [r for r in recs.values() if r["label"] == label and r["tier"] == tier]
            cells[label] = cell_vals(rs); n_all[label] = len(rs)
        L.append(r"\multicolumn{" + str(len(COLS) + 2) + r"}{l}{\textit{" + tname + r"}} \\")
        md.append(f"| **{tname}** |" + " |" * (len(COLS) + 1))
        for l, parts, n in block(cells, [l for l, _ in OWN], n_all): L.append(TEX[l] + " & " + " & ".join(parts) + f" & {n} " + r"\\")
        for l, parts, n in block(cells, [l for l, _ in OWN], n_all, md=True): md.append(f"| {l} | " + " | ".join(parts) + f" | {n} |")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (out_dir / "tab_M1.tex").write_text("\n".join(L) + "\n")
    # Table 2: shared one-sided zero-shot set
    arms = EXT + OWN[1:]
    per = {}
    for label, _ in arms:
        per[label] = {(r["cell"], r["endpoint"], r["reference"], k[2]): r for k, r in recs.items() if r["label"] == label and r["tier"] == "zero_shot" and r["sided"] == "one"}
    shared = None
    for label, _ in arms:
        ks = set(per[label]); shared = ks if shared is None else shared & ks
    cells = {l: cell_vals([per[l][k] for k in shared]) for l, _ in arms}; n_all = {l: len(shared) for l, _ in arms}
    L = [r"\begin{table}[H]", r"\centering", r"\caption{\textbf{Motion metrics, prior works vs.\ ours on the shared one-sided zero-shot set} ($n{=}" + str(len(shared)) + r"$ identical rows per arm; externals at their author-native prompt, ours neutral). " + NOTE + "}",
         r"\label{tab:motion_prior}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\linewidth}{!}{\begin{tabular}{l ccccccc c}", r"\toprule", hdr, r"\midrule"]
    md += ["", f"## Table 2 -- prior works, shared one-sided zero-shot set (n = {len(shared)})", md_hdr, "|---" * (len(COLS) + 2) + "|"]
    for l, parts, n in block(cells, [l for l, _ in arms], n_all): L.append(TEX[l] + " & " + " & ".join(parts) + f" & {n} " + r"\\")
    for l, parts, n in block(cells, [l for l, _ in arms], n_all, md=True): md.append(f"| {l} | " + " | ".join(parts) + f" | {n} |")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (out_dir / "tab_M2.tex").write_text("\n".join(L) + "\n")
    # native-spacing smoothness for the record
    md += ["", "### Motion smoothness at native spacing (as the prior works report it; frame-rate confounded)"]
    for l, _ in arms:
        m, n = agg([per[l][k].get('smooth_native') for k in shared])
        md.append(f"- {l}: " + (f"{m:.3f} (n={n})" if m is not None else "--"))
    (out_dir / "TABLES.md").write_text("\n".join(md) + "\n")
    tex = (r"% GENERATED by scripts/motion_tables.py" "\n" r"\documentclass{article}" "\n" r"\usepackage{natbib}" "\n" r"\input{preamble}" "\n"
           r"\usepackage{geometry}" "\n" r"\geometry{landscape, margin=1.5cm}" "\n" r"\begin{document}" "\n" r"\begin{center}" "\n"
           r"{\Large\bfseries \segue{} -- motion metrics (grid v3 preview)}\\[3pt]" "\n"
           r"\small Generated " + date.today().isoformat() + r" from store evals 038/040/028/030 (+041 when present) over the feature store. A preview for owner review, not the paper." "\n"
           r"\end{center}" "\n" r"\vspace{0.5em}" "\n" r"\input{tab_M1}" "\n" r"\vspace{1em}" "\n" r"\input{tab_M2}" "\n" r"\end{document}" "\n")
    (out_dir / "simple.tex").write_text(tex)
    paper = REPO_ROOT / "papers_drafts" / "ctt_iclr2027"
    env = dict(os.environ); env["PATH"] = "/taiga/illinois/eng/cs/jrehg/users/emirkisa/texlive/bin/aarch64-linux:" + env.get("PATH", "")
    env["TEXINPUTS"] = f".:{paper}:{paper}//:"; env["BIBINPUTS"] = env["TEXINPUTS"]; env["BSTINPUTS"] = env["TEXINPUTS"]
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-file-line-error", "simple.tex"], cwd=out_dir, env=env, capture_output=True, text=True, timeout=600)
    subprocess.run(["latexmk", "-c", "simple.tex"], cwd=out_dir, env=env, capture_output=True, text=True)
    print("[tables] build", "OK" if r.returncode == 0 else "FAILED", "->", out_dir / "simple.pdf")
    if r.returncode != 0: print("\n".join(r.stdout.splitlines()[-25:]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(OUT)); ap.add_argument("--cache", default=str(OUT / "records.json"))
    ap.add_argument("--no-smooth", action="store_true", help="skip the CLIP re-read (use cached records)")
    args = ap.parse_args(argv)
    cache = Path(args.cache)
    sm_cache = OUT / "smooth8.json"
    if cache.exists() and not sm_cache.exists():
        old = json.loads(cache.read_text())
        sm_cache.write_text(json.dumps({v["video"]: v.get("smooth8") for v in old.values() if v.get("smooth8") is not None}))
    if args.no_smooth and cache.exists():
        recs = {tuple(json.loads(k)): v for k, v in json.loads(cache.read_text()).items()}
    else:
        recs = collect(with_smooth=True)
        cache.parent.mkdir(parents=True, exist_ok=True); cache.write_text(json.dumps({json.dumps(list(k)): v for k, v in recs.items()}))
    print(f"[collect] {len(recs)} generations")
    render(recs, Path(args.out_dir)); return 0


if __name__ == "__main__":
    sys.exit(main())
