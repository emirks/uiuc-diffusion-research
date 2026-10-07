#!/usr/bin/env python
"""grid v3 — does TEXT<->VIDEO alignment of a class's effect clause with its clips predict the DCG-vs-prompt gap?

Companion to gap_predictors.py (video-only descriptors). Hypothesis: the effect-prompt gap is about how well TEXT
describes the effect, which no motion/appearance descriptor sees. Descriptor: CLIP ViT-B/32 cosine similarity between the
class's effect clause (as spliced into the effect prompts) and (a) the clip's core frames, (b) endpoint frames; per class
the mean over its corpus clips. Also clause length in words. Correlated (Spearman) with gap_eff / gap_neu / base_eff /
delta_prompt over the 76 grid-v3 classes. Exploratory, in-sample.
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval_ladder")); sys.path.insert(0, str(REPO / "scripts/grid_v3"))
STD = REPO / "data/processed/transitions_std121"
CSV_IN = REPO / "misc/2026-09-07_eval_grid_v2/eval/gap_predictors.csv"
CSV_OUT = REPO / "misc/2026-09-07_eval_grid_v2/eval/gap_predictors_clip.csv"


def clauses() -> dict:
    """reference clip stem -> its effect clause (misc/refvfx_baseline/reference_effects.json, the source the effect prompts
    were spliced from); class = clip stem minus the trailing _N (Higgsfield) or 'ed.<Effect>' (EffectData)."""
    d = json.loads((REPO / "misc/refvfx_baseline/reference_effects.json").read_text())
    out = {}
    for stem, clause in d.items():
        cls = "ed." + stem.split(".")[1] if stem.startswith("ed.") else stem.rsplit("_", 1)[0]
        out.setdefault(cls, []).append((stem, clause))
    return out


def frames_of(path: Path, idx):
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]; s.thread_count = 1
        fr = [f for f in c.decode(s)]
    T = len(fr); pick = sorted({min(T - 1, max(0, int(round(i * (T - 1))))) for i in idx})
    return [fr[i].to_image() for i in pick]


def main():
    from transformers import CLIPModel, CLIPProcessor
    from scipy.stats import spearmanr
    dev = "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval(); proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    lv = {r["class"]: r for r in csv.DictReader(open(CSV_IN))}
    cls = clauses(); corpus = json.loads((STD / "corpus_manifest.json").read_text())["clips"]
    by_class = collections.defaultdict(list)
    for k in corpus:
        c, clip = k.split("/"); by_class[c].append(clip)
    rows = []
    for c, r in sorted(lv.items()):
        pairs = cls.get(c) or []
        if not pairs:
            print("no clause for", c); continue
        core_s, end_s, words = [], [], []
        for stem, clause in pairs:
            path = STD / c / f"{stem}.mp4"
            if not path.exists():
                continue
            with torch.no_grad():
                t = proc(text=[clause], return_tensors="pt", padding=True, truncation=True).to(dev)
                te = model.text_projection(model.text_model(**t).pooler_output); te = te / te.norm(dim=-1, keepdim=True)
            fr = frames_of(path, [0.0, 0.35, 0.5, 0.65, 1.0])
            with torch.no_grad():
                im = proc(images=fr, return_tensors="pt").to(dev); ie = model.visual_projection(model.vision_model(**im).pooler_output); ie = ie / ie.norm(dim=-1, keepdim=True)
            sims = (ie @ te.T).squeeze(1).numpy()
            core_s.append(float(sims[1:4].mean())); end_s.append(float(sims[[0, 4]].mean())); words.append(len(clause.split()))
        if not core_s:
            print("no clips for", c); continue
        row = {"class": c, "family": r["family"], "n_refs": len(core_s), "clause_words": float(np.mean(words)),
               "clip_core_sim": float(np.mean(core_s)), "clip_endpoint_sim": float(np.mean(end_s)), "clip_core_minus_end": float(np.mean(core_s) - np.mean(end_s)),
               **{k: float(r[k]) for k in ("gap_eff", "gap_neu", "base_eff", "delta_prompt", "ceiling")}}
        rows.append(row); print(f"  {c:34s} refs {len(core_s)} core {row['clip_core_sim']:.3f} end {row['clip_endpoint_sim']:.3f} words {row['clause_words']:4.1f} gap_eff {row['gap_eff']:+6.1f}", flush=True)
    with open(CSV_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    targets = ["gap_eff", "gap_neu", "base_eff", "delta_prompt"]
    for fam_sel, label in (("all", "ALL"), ("HF", "Higgsfield"), ("ED", "EffectData")):
        sub = [x for x in rows if fam_sel == "all" or x["family"] == fam_sel]
        print(f"\n### {label} (n={len(sub)}) — Spearman ρ")
        print("| descriptor | " + " | ".join(f"ρ vs {t}" for t in targets) + " |"); print("|---|" + "---|" * len(targets))
        for d in ("clip_core_sim", "clip_endpoint_sim", "clip_core_minus_end", "clause_words"):
            x = np.array([s[d] for s in sub], float); cells = []
            for t in targets:
                y = np.array([s[t] for s in sub], float); rho, p = spearmanr(x, y); cells.append(f"{rho:+.2f}{'*' if p < 0.05 else ''}")
            print(f"| {d} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
