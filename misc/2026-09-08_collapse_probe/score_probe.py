#!/usr/bin/env python3
"""score_probe.py — score R1/R2/R3 with the certified null-family instrument (CPU).

Uses the EXISTING instrument misc/2026-09-08_collapse_remeasure/instrument.py
(load_matrix + measure) and the DINOv2 model-loading path from that campaign's covariates.py.

Per clip (prefix=9, suffix=8):
  * DR_med = median gap-normalised residual to the a->b endpoint line
    (R1 is one-sided: b = final generated frame, two_sided=False, suffix consumed 0)
Per R1 output additionally:
  * realized_dino = 1 - cos( DINOv2-base CLS(frame 8), CLS(last frame) )  — the achieved scene change

Outputs (results/):
  per_clip.csv   one row per generated clip
  paired.csv     one row per (prompt_id, seed): dDR_R2_minus_R1, dDR_R3_minus_R1, dDR_R3_minus_R2,
                 tier, realized_dino
  TABLES.md      medians/IQR per run x tier; paired medians + sign counts + Wilcoxon
  fig_probe.png  left: DR median+IQR per run split by tier (dots per clip);
                 right: dDR(R3-R1) vs realized_dino, tiers coloured

Partial-result safe: missing clips are skipped (counted); a delta needing an absent member is NaN.

    python score_probe.py [PROMPTS_JSONL]            # score real outputs -> results/
    python score_probe.py --smoke                    # stand-in mp4s, results/_smoke/, nothing committed
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

import _probe_common as C

REMEASURE = C.REPO_ROOT / "misc" / "2026-09-08_collapse_remeasure"
sys.path.insert(0, str(REMEASURE))
from instrument import load_matrix, measure  # noqa: E402

import cv2  # noqa: E402
cv2.setNumThreads(1)

PREFIX, SUFFIX = 9, 8


# ------------------------------------------------------------------ discovery
def discover(prompts: list[dict], roots: dict[str, Path]) -> list[dict]:
    """Map every generated clip we can find to (run, prompt_id, seed, tier, endpoint, variant, path)."""
    recs = []
    pmap = {p["prompt_id"]: p for p in prompts}

    # R1: reg/r1.jsonl has no seed -> glob per item_id
    for p in prompts:
        item = C.r1_item_id(p)
        for f in sorted(glob.glob(str(roots["r1"] / C.ARM / f"{item}__s*.mp4"))):
            seed = int(Path(f).stem.rsplit("__s", 1)[1])
            recs.append(dict(run="R1", prompt_id=p["prompt_id"], seed=seed, tier=p["tier"],
                             endpoint=p["endpoint"], variant="full", path=f))

    # R2 / R3: per-seed registries carry seed + item_id
    for run, root in (("R2", roots["r2"]), ("R3", roots["r3"])):
        for regf in sorted(C.REG.glob(f"{run.lower()}_s*.jsonl")):
            for line in regf.read_text().splitlines():
                if not line.strip():
                    continue
                r = __import__("json").loads(line)
                pid = r["prompt_id"]
                if pid not in pmap:
                    continue
                seed = r["seed"]
                f = C.out_mp4(run.lower(), r["item_id"], seed, out_root=root)
                if f.exists():
                    recs.append(dict(run=run, prompt_id=pid, seed=seed, tier=r["tier"],
                                     endpoint=r["real_endpoint"], variant=r["probe_prompt_variant"],
                                     path=str(f)))
    return recs


# ------------------------------------------------------------------ measurement
def score_clip(path: str, two_sided: bool) -> dict:
    M = load_matrix(path)
    if M is None:
        return dict(decode_ok=False)
    out, _, _ = measure(M, PREFIX, SUFFIX, two_sided)
    out["decode_ok"] = True
    return out


def read_two_frames_rgb(path: str, idxs=(PREFIX - 1, -1)):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return None
    T = len(frames)
    out = {}
    for i in idxs:
        j = i if i >= 0 else T + i
        if 0 <= j < T:
            out[i] = cv2.cvtColor(frames[j], cv2.COLOR_BGR2RGB)
    return out


def make_dino_embed():
    """DINOv2-base CLS embedder — same load path as covariates.py (transformers 5)."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").eval()

    @torch.no_grad()
    def embed(frames_rgb):
        ims = [Image.fromarray(f) for f in frames_rgb]
        z = dino(**proc(images=ims, return_tensors="pt")).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(z, dim=-1).numpy()

    return embed


# ------------------------------------------------------------------ tables / figure
def iqr(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan, 0)
    return (float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75)), len(x))


def write_tables(per_clip: pd.DataFrame, paired: pd.DataFrame, out: Path) -> None:
    from scipy.stats import wilcoxon
    lines = ["# Collapse-to-the-endpoint-line probe — TABLES", ""]
    lines.append(f"_n clips scored: {len(per_clip)} "
                 f"({(per_clip['run']=='R1').sum()} R1 / {(per_clip['run']=='R2').sum()} R2 / "
                 f"{(per_clip['run']=='R3').sum()} R3)_")
    lines.append("")

    # DR per run x tier
    lines += ["## DR_med by run x tier", "", "| run | tier | median | IQR (25-75) | n |", "|---|---|---|---|---|"]
    for run in ("R1", "R2", "R3"):
        for tier in ("high", "low"):
            sub = per_clip[(per_clip.run == run) & (per_clip.tier == tier)]["DR_med"]
            med, q1, q3, n = iqr(sub)
            lines.append(f"| {run} | {tier} | {med:.4f} | [{q1:.4f}, {q3:.4f}] | {n} |")
    lines.append("")

    # paired deltas
    lines += ["## Paired DR deltas (per prompt x seed)", "",
              "| delta | tier | median | n_pos | n_neg | n_zero | n | Wilcoxon p |",
              "|---|---|---|---|---|---|---|---|"]
    for col in ("dDR_R2_minus_R1", "dDR_R3_minus_R1", "dDR_R3_minus_R2"):
        for tier in ("high", "low", "all"):
            sub = paired if tier == "all" else paired[paired.tier == tier]
            d = np.asarray(sub[col], float)
            d = d[np.isfinite(d)]
            if len(d) == 0:
                lines.append(f"| {col} | {tier} | nan | 0 | 0 | 0 | 0 | nan |")
                continue
            med = float(np.median(d))
            npos, nneg, nzero = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
            try:
                p = float(wilcoxon(d).pvalue) if np.any(d != 0) else float("nan")
            except ValueError:
                p = float("nan")
            lines.append(f"| {col} | {tier} | {med:+.4f} | {npos} | {nneg} | {nzero} | {len(d)} | {p:.3g} |")
    lines.append("")

    # realized change
    lines += ["## Realized scene change (R1 DINOv2 CLS distance) by tier", "",
              "| tier | median | IQR (25-75) | n |", "|---|---|---|---|"]
    for tier in ("high", "low", "all"):
        sub = paired if tier == "all" else paired[paired.tier == tier]
        med, q1, q3, n = iqr(sub["realized_dino"])
        lines.append(f"| {tier} | {med:.4f} | [{q1:.4f}, {q3:.4f}] | {n} |")
    lines.append("")
    out.write_text("\n".join(lines) + "\n")


def make_fig(per_clip: pd.DataFrame, paired: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    tcol = {"high": "#c0392b", "low": "#2471a3"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

    # left: DR per run x tier
    runs = ["R1", "R2", "R3"]
    for ti, tier in enumerate(("high", "low")):
        xs, meds, los, his = [], [], [], []
        for ri, run in enumerate(runs):
            sub = per_clip[(per_clip.run == run) & (per_clip.tier == tier)]["DR_med"].values
            x = ri + (ti - 0.5) * 0.28
            if len(sub):
                axL.scatter(np.full(len(sub), x) + np.random.uniform(-0.04, 0.04, len(sub)),
                            sub, s=10, alpha=0.35, color=tcol[tier], zorder=1)
                med, q1, q3, _ = iqr(sub)
                xs.append(x); meds.append(med); los.append(med - q1); his.append(q3 - med)
        if xs:
            axL.errorbar(xs, meds, yerr=[los, his], fmt="o", color=tcol[tier], capsize=4,
                         ms=8, lw=2, label=f"{tier}", zorder=3)
    axL.set_xticks(range(len(runs))); axL.set_xticklabels(runs)
    axL.set_ylabel("DR_med (residual to endpoint line)")
    axL.set_title("Endpoint-line residual per run")
    axL.legend(title="tier"); axL.grid(alpha=0.25)

    # right: dDR(R3-R1) vs realized change
    for tier in ("high", "low"):
        sub = paired[paired.tier == tier]
        axR.scatter(sub["realized_dino"], sub["dDR_R3_minus_R1"], s=22, alpha=0.7,
                    color=tcol[tier], label=tier)
    axR.axhline(0, color="k", lw=0.8, ls="--")
    axR.set_xlabel("realized scene change  (R1 DINOv2 CLS distance)")
    axR.set_ylabel("dDR (R3 - R1)")
    axR.set_title("Text-off shift vs realized change")
    axR.legend(title="tier"); axR.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ driver
def run(prompts, roots, results_dir: Path, do_dino=True) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    recs = discover(prompts, roots)
    print(f"[score] discovered {len(recs)} clips "
          f"(R1={sum(r['run']=='R1' for r in recs)}, R2={sum(r['run']=='R2' for r in recs)}, "
          f"R3={sum(r['run']=='R3' for r in recs)})")
    if not recs:
        print("[score] no outputs found — nothing to score")
        return

    # per-clip measurement
    rows = []
    for rec in recs:
        m = score_clip(rec["path"], two_sided=(rec["run"] != "R1"))
        row = dict(rec)
        row.pop("path", None)
        row["path"] = os.path.relpath(rec["path"], C.REPO_ROOT)
        for k in ("decode_ok", "DR_med", "DR_mean", "online_frac", "M", "R", "S", "tau_med",
                  "gap", "gap_rel", "PR", "n_interior", "T"):
            row[k] = m.get(k, np.nan)
        rows.append(row)
    per_clip = pd.DataFrame(rows)

    # realized DINO change on R1 outputs
    realized = {}
    r1recs = [r for r in recs if r["run"] == "R1"]
    if do_dino and r1recs:
        print(f"[score] DINOv2 realized-change on {len(r1recs)} R1 clips ...", flush=True)
        embed = make_dino_embed()
        for i, rec in enumerate(r1recs):
            fr = read_two_frames_rgb(rec["path"])
            if fr and (PREFIX - 1) in fr and -1 in fr:
                z = embed([fr[PREFIX - 1], fr[-1]])
                realized[(rec["prompt_id"], rec["seed"])] = float(1.0 - (z[0] * z[1]).sum())
            if (i + 1) % 20 == 0:
                print(f"   ..{i+1}/{len(r1recs)}", flush=True)
    per_clip["realized_dino"] = [realized.get((r.prompt_id, r.seed), np.nan)
                                 for r in per_clip.itertuples()]
    per_clip.to_csv(results_dir / "per_clip.csv", index=False)

    # paired
    def dr(run, pid, seed):
        s = per_clip[(per_clip.run == run) & (per_clip.prompt_id == pid) & (per_clip.seed == seed)]
        return float(s["DR_med"].iloc[0]) if len(s) else np.nan

    keys = sorted({(r.prompt_id, r.seed, r.tier) for r in per_clip.itertuples()})
    prows = []
    for pid, seed, tier in keys:
        d1, d2, d3 = dr("R1", pid, seed), dr("R2", pid, seed), dr("R3", pid, seed)
        prows.append(dict(
            prompt_id=pid, seed=seed, tier=tier,
            DR_R1=d1, DR_R2=d2, DR_R3=d3,
            dDR_R2_minus_R1=d2 - d1, dDR_R3_minus_R1=d3 - d1, dDR_R3_minus_R2=d3 - d2,
            realized_dino=realized.get((pid, seed), np.nan)))
    paired = pd.DataFrame(prows)
    paired.to_csv(results_dir / "paired.csv", index=False)

    write_tables(per_clip, paired, results_dir / "TABLES.md")
    try:
        make_fig(per_clip, paired, results_dir / "fig_probe.png")
    except Exception as e:  # figure is non-critical
        print(f"[score] figure skipped: {e}")

    print(f"[score] wrote per_clip.csv ({len(per_clip)}), paired.csv ({len(paired)}), "
          f"TABLES.md, fig_probe.png -> {results_dir}")


def smoke(prompts) -> None:
    """Exercise the whole pipeline with stand-in mp4s. Writes results/_smoke/, commits nothing real."""
    standins = sorted(glob.glob(str(
        C.REPO_ROOT / "store/gens/005_base_cond/04_neutral_v3__dai/videos/*__s42.mp4")))[:2]
    if len(standins) < 2:
        raise SystemExit("[smoke] need >=2 existing mp4s as stand-ins")
    print(f"[smoke] STAND-IN mp4s (NOT probe outputs): {[os.path.basename(s) for s in standins]}")
    ex = prompts[:2]
    # fabricate a run map: each example prompt gets R1/R2/R3 pointing at the two stand-ins
    recs = []
    for k, p in enumerate(ex):
        for rr in ("R1", "R2", "R3"):
            recs.append(dict(run=rr, prompt_id=p["prompt_id"], seed=42, tier=p["tier"],
                             endpoint=p["endpoint"], variant="full", path=standins[k % 2]))
    # monkeypatch discover for this run
    global discover
    _orig = discover
    discover = lambda *a, **k: recs  # noqa: E731
    try:
        run(ex, {"r1": C.OUT / "r1", "r2": C.OUT / "r2", "r3": C.OUT / "r3"},
            C.HERE / "results" / "_smoke", do_dino=True)
    finally:
        discover = _orig
    print("[smoke] pipeline ran end-to-end; outputs under results/_smoke/ are a SMOKE TEST — "
          "not real probe results.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", nargs="?", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-dino", action="store_true", help="skip the DINOv2 realized-change pass")
    args = ap.parse_args()
    prompts = C.load_prompts(Path(args.prompts) if args.prompts else C.default_prompts())
    if args.smoke:
        smoke(prompts)
        return
    run(prompts, {"r1": C.OUT / "r1", "r2": C.OUT / "r2", "r3": C.OUT / "r3"},
        C.HERE / "results", do_dino=not args.no_dino)


if __name__ == "__main__":
    main()
