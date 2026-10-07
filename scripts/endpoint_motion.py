#!/usr/bin/env python
"""Endpoint MOTION preservation from dense RAFT flow of the given windows (feature-store namespace
``raft_flow_win@r256``): does the output move like the given clip inside the given window?

Per generation and side (A = given start clip, B = given end clip on two-sided rows):
  F_given  = flow of the given clip's window  (start9 frames 0..8 -> 8 steps; end9 frames 1..8 -> 7 steps)
  F_out    = flow of the output's pinned window (frames 0..8 -> 8 steps; last 8 frames -> 7 steps)
  M        = pixels (per step) where either clip moves: |F| >= MOVE_PX
  agree    = 1 - sum_M |F_out - F_given| / sum_M (|F_out| + |F_given|)   in [0, 1]; 1 = identical motion
  epe      = mean_M |F_out - F_given|  (pixels of the 256-px frame)
  NaN when the given window has no moving pixel (nothing to preserve).
Only rows whose conditioning is a CLIP (HF-121f rows, 9 given frames) are scored; ED rows and the
prior works get one frame (no motion) and are absent by construction (tables show '--').

Outputs (store eval, numbered, never edited in place):
  store/evals/<EVAL_ID>/<harness_arm>/rows.jsonl + meta.yaml + one store/INDEX.md line.
--tables renders the two standard tables (own arms by tier; prior works on the shared set) to
  papers_drafts/_preview/motion_gridv3/{tab_M1,tab_M2}.tex + simple.pdf.
"""
from __future__ import annotations
import argparse, json, math, os, socket, subprocess, sys
from datetime import date
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src")); sys.path.insert(0, str(REPO_ROOT / "scripts"))
from diffusion.feature_store import FeatureStore  # noqa: E402
from store_eval_common import grid_type as _grid_type, cond_clips as _cond_clips, windows as _windows  # noqa: E402

NS = "raft_flow_win@r256"
MOVE_PX = 0.5
CONDS_DIR = REPO_ROOT / "eval_ladder" / "conds"
POP = REPO_ROOT / "misc/2026-09-17_feature_store/population_flowwin.json"
EXTERNAL_ARMS = ("refvfx", "vap", "vfxmaster")
DEFINITION = [
    "F_given = raft_flow_win@r256 of the given clip: start9 -> flow_start (frames 0..8, 8 steps); end9 -> flow_end (frames 1..8, 7 steps).",
    "F_out = raft_flow_win@r256 of the output: flow_start (frames 0..8) for side A; flow_end (last 8 frames) for side B (two-sided rows only).",
    "If the given clip and the output decode to different HxW, F_given is resized to the output's HxW and its vectors scaled accordingly.",
    f"M = per-step pixels where max(|F_out|, |F_given|) >= {MOVE_PX} px. agree = 1 - sum_M |F_out - F_given| / sum_M (|F_out| + |F_given|); epe = mean_M |F_out - F_given| px; cos = |F_out||F_given|-weighted cosine over M.",
    "NaN (with given_moving_frac) when the given window has no moving pixel. Only HF-121f rows (9-frame clip given) are scored.",
    "Not measurable (row field not_measurable, no A_/B_ values) when the given window of that side is a single frame: store_eval_common.windows(gtype, sided) "
    "n_pre < 2 (side A) / n_suf < 2 (side B) — the ED rows and the frame-conditioned externals (refVFX, Wan FLF2V), even when a raft_flow_win feature exists for the gen (2026-09-23).",
    "TEG baseline with CLIP endpoints (VACE16, 2026-09-20): the given clips are the 16-fps resamples conds_16fps/<endpoint>_start6 (5 steps) / _end4 (3 steps); "
    "side A compares the first 5 steps, side B the LAST 3 steps of the output's pinned end window (both windows aligned at the seam they share with the given clip).",
]
WHY = ("Endpoint identity (DINO) shows the given frames are kept; this is the motion counterpart: the given clip's motion must be "
       "reproduced inside the pinned window. Dense per-step flow is the aligned-video standard (flow end-point error); the earlier "
       "track-based probe compared one net displacement over 400 grid points and was jitter-limited.")
CAVEAT = ("Whole-frame motion (subject + background + camera), no subject mask yet; flow at 256 px short side; the output window is "
          "pinned by conditioning so values near 1 are expected for every arm (a conditioning check, not a ranking).")


def _read_meta(path: Path) -> dict:
    out = {}
    for ln in path.read_text().splitlines():
        if ln[:1].isspace() or ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        out[k.strip()] = v.split("#", 1)[0].strip().strip("'\"")
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _parse_stem(stem: str) -> tuple[str, int]:
    item, s = stem.rsplit("__s", 1)
    return item, int(s)


def _resize_flow(F: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """[S,H,W,2] -> [S,h,w,2] with vector components scaled by the size ratio."""
    import cv2
    S, H, W, _ = F.shape
    h, w = hw
    if (H, W) == (h, w):
        return F
    out = np.empty((S, h, w, 2), dtype=np.float32)
    for s in range(S):
        r = cv2.resize(F[s].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        r[..., 0] *= w / W
        r[..., 1] *= h / H
        out[s] = r
    return out


def window_agreement(F_out: np.ndarray, F_given: np.ndarray, align: str = "start") -> dict:
    """align='start' (side A): the first S steps of both; align='end' (side B): the last S steps of both — on HF rows
    both windows have equal length so the choice is moot; a shorter given clip (VACE16 end4) keeps the steps at the seam."""
    F_out = F_out.astype(np.float32)
    F_given = _resize_flow(F_given.astype(np.float32), F_out.shape[1:3])
    S = min(F_out.shape[0], F_given.shape[0])
    F_out, F_given = (F_out[:S], F_given[:S]) if align == "start" else (F_out[-S:], F_given[-S:])
    mo = np.linalg.norm(F_out, axis=-1)
    mg = np.linalg.norm(F_given, axis=-1)
    given_moving = float((mg >= MOVE_PX).mean())
    M = np.maximum(mo, mg) >= MOVE_PX
    if not (mg >= MOVE_PX).any():
        return dict(agree=float("nan"), epe=float("nan"), cos=float("nan"),
                    given_moving_frac=given_moving, given_mean_px=float(mg.mean()), out_mean_px=float(mo.mean()), n_steps=int(S))
    diff = np.linalg.norm(F_out - F_given, axis=-1)[M]
    den = (mo + mg)[M]
    agree = 1.0 - float(diff.sum() / den.sum())
    w = (mo * mg)[M]
    cosv = (F_out * F_given).sum(-1)[M] / (mo[M] * mg[M] + 1e-9)
    cos = float((w * cosv).sum() / w.sum()) if w.sum() > 0 else float("nan")
    return dict(agree=agree, epe=float(diff.mean()), cos=cos, given_moving_frac=given_moving,
                given_mean_px=float(mg.mean()), out_mean_px=float(mo.mean()), n_steps=int(S))


def score_variant(fs: FeatureStore, vdir: Path) -> dict:
    videos = sorted((vdir / "videos").glob("*.mp4"))
    harness_arm = _parse_stem(videos[0].stem)[0].split("__")[1]     # frozen stamp inside the item id (same as evals 038-040)
    grid = {}
    for ln in (vdir / "grid.jsonl").read_text().splitlines():
        if ln.strip():
            r = json.loads(ln); grid[r["item_id"]] = r
    rows, cache = [], {}
    gtype = _grid_type(vdir)

    def cond(p: Path):
        if p not in cache:
            cache[p] = fs.get(p, NS) if (p.exists() and fs.has(p, NS)) else None
        return cache[p]

    for v in videos:
        item_id, seed = _parse_stem(v.stem)
        g = grid.get(item_id)
        row = dict(item_id=item_id, seed=seed, arm=harness_arm, endpoint=g.get("endpoint") if g else None,
                   sided=(g.get("sided", "one") if g else "one"), tier=(g.get("ref_novelty") if g else None),
                   cell=(g.get("cell") if g else None), reference=(g.get("reference") if g else None), missing=[])
        if g is None:
            row["missing"].append("grid_row"); rows.append(row); continue
        n_pre, n_suf = _windows(gtype, g.get("sided", "one"))
        row["not_measurable"] = ([] if n_pre >= 2 else ["A"]) + ([] if (g.get("sided") != "two" or n_suf >= 2) else ["B"])
        if n_pre < 2 and (g.get("sided") != "two" or n_suf < 2):     # one given frame per side: no given motion to preserve
            rows.append(row); continue
        go = fs.get(v, NS) if fs.has(v, NS) else None
        if go is None:
            row["missing"].append(f"{NS}:gen")
        pA, pB = _cond_clips(gtype, g["endpoint"])
        if n_pre >= 2:
            cA = cond(pA)
            if cA is None:
                row["missing"].append(f"{NS}:condA")
            if go is not None and cA is not None:
                row.update({f"A_{k}": val for k, val in window_agreement(go["flow_start"], cA["flow_start"], "start").items()})
        if g.get("sided") == "two" and n_suf >= 2:
            cB = cond(pB)
            if cB is None:
                row["missing"].append(f"{NS}:condB")
            elif go is not None:
                row.update({f"B_{k}": val for k, val in window_agreement(go["flow_end"], cB["flow_end"], "end").items()})
        rows.append(row)
    fin = lambda k: sum(1 for r in rows if isinstance(r.get(k), float) and math.isfinite(r[k]))
    mean = lambda k: (float(np.mean([r[k] for r in rows if isinstance(r.get(k), float) and math.isfinite(r[k])])) if fin(k) else None)
    cov = dict(n=len(rows), A_defined=fin("A_agree"), B_defined=fin("B_agree"), missing=sum(1 for r in rows if r["missing"]),
               A_agree_mean=mean("A_agree"), A_epe_mean=mean("A_epe"), B_agree_mean=mean("B_agree"), B_epe_mean=mean("B_epe"))
    return dict(harness_arm=harness_arm, gen=str(vdir.relative_to(REPO_ROOT)), rows=rows, coverage=cov)


def _atomic_write(p: Path, text: str) -> None:
    tmp = p.with_suffix(p.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(text); os.replace(tmp, p)


def write_eval(eval_id: str, results: list[dict], created: str, no_index: bool) -> Path:
    draft = not eval_id[:3].isdigit()
    ed = REPO_ROOT / "store" / "evals" / ("_draft" if draft else "") / eval_id
    ed.mkdir(parents=True, exist_ok=True)
    for res in results:
        d = ed / res["harness_arm"]; d.mkdir(exist_ok=True)
        _atomic_write(d / "rows.jsonl", "".join(json.dumps(r) + "\n" for r in res["rows"]))
    L = [f"id: {eval_id}", f"seq: {int(eval_id.split('_', 1)[0]) if not draft else 'null  # DRAFT'}", f"shelf: {'evals/_draft' if draft else 'evals'}", f"created: '{created}'",
         f"machine: dai (login CPU, numpy over stored features; flow extracted on ghx4, see feature sidecars)",
         f"instrument: scripts/endpoint_motion.py @ {_git_sha()}", f"namespace: {NS}", f"move_px: {MOVE_PX}", "definition:"]
    L += [f"  - {json.dumps(d)}" for d in DEFINITION]
    L += [f"why: {json.dumps(WHY)}", f"caveat: {json.dumps(CAVEAT)}", "arms_scored:"]
    for res in results:
        c = res["coverage"]
        L += [f"  {res['harness_arm']}:", f"    gen: {res['gen']}", f"    rows: {c['n']}",
              "    coverage: {" + ", ".join(f"{k}: {v}" for k, v in c.items() if k != "n") + "}"]
    _atomic_write(ed / "meta.yaml", "\n".join(L) + "\n")
    if not no_index and not draft:
        index = REPO_ROOT / "store" / "INDEX.md"; text = index.read_text()
        if f"`{eval_id}`" not in text:
            n_rows = sum(r["coverage"]["n"] for r in results)
            line = (f"{int(eval_id.split('_', 1)[0])}. `{eval_id}` — endpoint MOTION preservation on grid v3 ({len(results)} own-arm HF variants, "
                    f"{n_rows} gens): dense per-step RAFT flow of the given window (namespace {NS}) of the output vs the given clip; "
                    f"agree = 1 - sum|dF| / sum(|F_out|+|F_given|) over moving pixels, epe in px; side A (start clip) and B (end clip, two-sided). "
                    f"NaN where the given window is static. CPU/login, numpy over stored features; rows.jsonl are store artifacts (not committed); "
                    f"definition/why/caveat in meta.yaml. scripts/endpoint_motion.py.")
            lines = text.splitlines()
            ev = next(i for i, ln in enumerate(lines) if ln.strip() == "## evals")
            nxt = next((i for i in range(ev + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
            ins = nxt
            while ins > ev + 1 and not lines[ins - 1].strip():
                ins -= 1
            lines.insert(ins, line); _atomic_write(index, "\n".join(lines) + "\n")
    return ed


# ---------------------------------------------------------------- tables (standard two-table format)
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
COLS = [("A_agree", "Endpoint Motion A"), ("B_agree", "Endpoint Motion B")]
NOTE = (r"Does the output move like the given clip inside the given window? Dense RAFT flow of every step of the window, output vs. given clip, "
        r"over the pixels that move in either: agreement $=1-\sum\|F_\mathrm{out}-F_\mathrm{given}\|/\sum(\|F_\mathrm{out}\|+\|F_\mathrm{given}\|)$, "
        r"1 = identical motion, 0 = static or opposite; mean over generations, higher is better ($\uparrow$). A: the 8 steps of the given 9-frame start clip; "
        r"B: the 7 steps of the given end clip (two-sided rows only). A given window with no moving pixel has nothing to preserve and is NaN; a per-cell $n$ "
        r"counts defined rows, the last column all rows. One-frame conditioning (ED rows, all prior works) carries no motion: --. Whole-frame motion, no subject mask. "
        r"Bold: best per column and block.")


def _grid_rows(vrel: str) -> list[dict]:
    vdir = REPO_ROOT / vrel
    out = []
    for ln in (vdir / "grid.jsonl").read_text().splitlines():
        if ln.strip():
            r = json.loads(ln); out.append(r)
    return out


def _gen_keys(vrel: str) -> list[tuple]:
    """(cell, endpoint, reference, seed) per generated mp4 of a variant, joined with its grid row."""
    vdir = REPO_ROOT / vrel
    grid = {r["item_id"]: r for r in _grid_rows(vrel)}
    keys = []
    for v in sorted((vdir / "videos").glob("*.mp4")):
        item_id, seed = _parse_stem(v.stem)
        g = grid.get(item_id)
        if g:
            keys.append(((g.get("cell"), g["endpoint"], g.get("reference"), seed), g, item_id))
    return keys


def render_tables(eval_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = {}   # (base, item_id, seed) -> row
    for d in eval_dir.iterdir():
        p = d / "rows.jsonl"
        if p.exists():
            for ln in p.read_text().splitlines():
                if ln.strip():
                    r = json.loads(ln); scored[(d.name, r["item_id"], r["seed"])] = r

    def agg(vals):
        v = [x for x in vals if isinstance(x, float) and math.isfinite(x)]
        return (float(np.mean(v)), len(v)) if v else (None, 0)

    def block(cells, labels, n_all):
        best = [max((cells[l][ci][0] for l in labels if cells[l][ci][0] is not None), default=None) for ci in range(len(COLS))]
        out = []
        for l in labels:
            parts = []
            for ci in range(len(COLS)):
                m, n = cells[l][ci]
                if m is None: parts.append("--"); continue
                s = f"{m:.3f}"
                if best[ci] is not None and abs(m - best[ci]) <= 1e-9: s = r"\textbf{" + s + "}"
                if n != n_all[l]: s += r"{\tiny\,($n{=}" + str(n) + "$)}"
                parts.append(s)
            out.append(TEX[l] + " & " + " & ".join(parts) + f" & {n_all[l]} " + r"\\")
        return out

    HDR = r"arm & " + " & ".join(f"{c} $\\uparrow$" for _, c in COLS) + r" & $n$ \\"
    # Table 1: own arms by tier (HF + ED pooled; ED rows count in n only)
    L = [r"\begin{table}[H]", r"\centering",
         r"\caption{\textbf{Endpoint motion preservation, own arms by tier (neutral prompt, grid v3; HF-121f and ED-81f rows pooled per tier).} " + NOTE + "}",
         r"\label{tab:endpoint_motion_own}", r"\small", r"\begin{tabular}{l cc c}", r"\toprule", HDR, r"\midrule"]
    for tier, tname in (("seen", "Seen"), ("unseen", "Unseen"), ("zero_shot", "Zero-shot")):
        cells, n_all = {}, {}
        for label, base in OWN:
            vals = {k: [] for k, _ in COLS}; n = 0
            for vrel in GENS[base]:
                for key, g, item_id in _gen_keys(vrel):
                    harm = item_id.split("__")[1]
                    if g.get("ref_novelty") != tier: continue
                    n += 1
                    r = scored.get((harm, item_id, key[3]))
                    for k, _ in COLS: vals[k].append(r.get(k) if r else None)
            cells[label] = [agg(vals[k]) for k, _ in COLS]; n_all[label] = n
        L.append(r"\multicolumn{4}{l}{\textit{" + tname + r"}} \\")
        L += block(cells, [l for l, _ in OWN], n_all); L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out_dir / "tab_M1.tex").write_text("\n".join(L) + "\n")
    # Table 2: prior works vs ours on the shared one-sided zero-shot set
    arms = EXT + OWN[1:]
    per_arm = {}
    for label, base in arms:
        rows = {}
        for vrel in GENS[base]:
            for key, g, item_id in _gen_keys(vrel):
                harm = item_id.split("__")[1]
                if g.get("ref_novelty") == "zero_shot" and g.get("sided", "one") == "one":
                    rows[key] = scored.get((harm, item_id, key[3]))
        per_arm[label] = rows
    shared = None
    for label, _ in arms:
        ks = set(per_arm[label]); shared = ks if shared is None else shared & ks
    cells = {l: [agg([(per_arm[l][k] or {}).get(c) for k in shared]) for c, _ in COLS] for l, _ in arms}
    n_all = {l: len(shared) for l, _ in arms}
    L = [r"\begin{table}[H]", r"\centering",
         r"\caption{\textbf{Endpoint motion preservation, prior works vs.\ ours on the shared one-sided zero-shot set} ($n{=}" + str(len(shared)) + r"$ identical rows per arm). " + NOTE +
         r" The prior works receive one start frame (--); our arms receive the 9-frame clip on the HF rows of this set and one frame on its ED rows, so Endpoint Motion A is defined on the moving HF rows only.}",
         r"\label{tab:endpoint_motion_prior}", r"\small", r"\begin{tabular}{l cc c}", r"\toprule", HDR, r"\midrule"]
    L += block(cells, [l for l, _ in arms], n_all)
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out_dir / "tab_M2.tex").write_text("\n".join(L) + "\n")
    tex = (r"% GENERATED by scripts/endpoint_motion.py --tables" "\n" r"\documentclass{article}" "\n" r"\usepackage{natbib}" "\n" r"\input{preamble}" "\n"
           r"\begin{document}" "\n" r"\begin{center}" "\n" r"{\Large\bfseries \segue{} -- endpoint motion preservation (grid v3 preview)}\\[3pt]" "\n"
           r"\small Generated " + date.today().isoformat() + r" from store eval \texttt{" + eval_dir.name.replace("_", r"\_") + r"} (dense RAFT flow of the given windows, namespace \texttt{raft\_flow\_win@r256})." "\n"
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
    ap.add_argument("--eval-id", default=None)
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--population", default=str(POP))
    ap.add_argument("--no-index", action="store_true")
    ap.add_argument("--tables", action="store_true", help="render the two standard tables from the eval (no rescoring)")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "papers_drafts/_preview/motion_gridv3"))
    args = ap.parse_args(argv)
    eval_id = args.eval_id or f"endpoint_motion_gridv3__dai__{args.date}"   # DRAFT (unnumbered) until finalized
    draft = not eval_id[:3].isdigit()
    ed = REPO_ROOT / "store" / "evals" / ("_draft" if draft else "") / eval_id
    if not args.tables:
        fs = FeatureStore(REPO_ROOT)
        pop = json.loads(Path(args.population).read_text())
        results = []
        for vrel in pop["gen_variants"]:
            res = score_variant(fs, REPO_ROOT / vrel)
            c = res["coverage"]
            print(f"[score] {res['harness_arm']:<36} n={c['n']} A={c['A_defined']} B={c['B_defined']} missing={c['missing']} "
                  f"A_agree={c['A_agree_mean']} A_epe={c['A_epe_mean']} B_agree={c['B_agree_mean']}", flush=True)
            results.append(res)
        write_eval(eval_id, results, args.date, args.no_index)
        print(f"[eval] wrote {ed}")
    render_tables(ed, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
