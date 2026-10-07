#!/usr/bin/env python
"""Text consistency (VBench `overall_consistency`) as a store eval (DRAFT until finalized).

Video side: the stored ViCLIP embedding of each generation (namespace viclip@l14-f8, VBench recipe: 8 middle-sampled
frames, ViT-L/14 InternVid-10M-FLT). Text side: the same model's text tower on the prompt, L2-normalised. Score = cosine.
Two prompt pairings per generation:
  text_own     the prompt the generation was actually conditioned on (neutral / effect / author-native), LoRA trigger
               token " sksz." removed -- this is VBench's definition (prompt vs. its own video)
  text_effect  the effect description of the same grid row (the effect-variant prompt of our arms for that row, trigger
               removed) -- "does the output show the intended transition", comparable across arms whatever they were
               prompted with; NaN when no effect row exists for that (cell, endpoint, reference)
Rows -> store/evals/_draft/<EVAL_ID>/<harness_arm>/rows.jsonl (+ meta.yaml). CPU: the text tower runs on the login node.
"""
from __future__ import annotations
import argparse, json, math, os, re, sys
from datetime import date
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src")); sys.path.insert(0, str(REPO_ROOT / "scripts"))
from store_eval_common import POP_GRIDV3, parse_stem, harness_arm_of, grid_of, write_eval  # noqa: E402

NS = "viclip@l14-f8"
TRIGGER = re.compile(r"\s*\bsksz\b\.?", re.IGNORECASE)


def clean_prompt(p: str | None) -> str | None:
    if not p:
        return None
    return re.sub(r"\s{2,}", " ", TRIGGER.sub("", p)).strip()


def row_key(g: dict, seed: int) -> tuple:
    return (g.get("cell"), g.get("endpoint"), g.get("reference"), seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-id", default=None); ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--population", default=str(POP_GRIDV3)); ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    eval_id = args.eval_id or f"viclip_text_gridv3__dai__{args.date}"   # DRAFT (unnumbered) until finalized
    from diffusion.feature_store import FeatureStore
    from diffusion.feature_extractors import REGISTRY
    fs = FeatureStore(REPO_ROOT)
    pop = json.loads(Path(args.population).read_text())

    # 1) effect description per grid row (from our effect variants; identical rows across own arms)
    effect_of: dict[tuple, str] = {}
    for vrel in pop["gen_variants"]:
        if "effect" not in Path(vrel).name:
            continue
        for g in grid_of(REPO_ROOT / vrel).values():
            for seed in (42, 43):
                effect_of.setdefault(row_key(g, seed), clean_prompt(g.get("prompt")))
    # 2) collect (video, own prompt, effect prompt) per generation
    per_variant = []
    prompts = set()
    for vrel in pop["gen_variants"]:
        vdir = REPO_ROOT / vrel; grid = grid_of(vdir); items = []
        for v in sorted((vdir / "videos").glob("*.mp4")):
            item_id, seed = parse_stem(v.stem); g = grid.get(item_id)
            if not g:
                continue
            own = clean_prompt(g.get("prompt")); eff = effect_of.get(row_key(g, seed))
            items.append((v, item_id, seed, own, eff))
            for t in (own, eff):
                if t: prompts.add(t)
        per_variant.append((vrel, items))
    # 3) text embeddings (unique prompts)
    prompts = sorted(prompts)
    import hashlib, torch
    torch.set_num_threads(8)
    cache = REPO_ROOT / "store" / "evals" / "_draft" / "viclip_text_cache.npz"   # prompt-hash -> [768]; reused across runs
    temb = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True); temb = dict(zip(z["keys"].tolist(), z["embs"]))
    todo = [p for p in prompts if hashlib.sha1(p.encode()).hexdigest() not in temb]
    print(f"[text] {len(prompts)} unique prompts, {len(todo)} to encode on {args.device}", flush=True)
    if todo:
        ex = REGISTRY[NS](args.device)
        for i in range(0, len(todo), 64):
            chunk = todo[i:i + 64]; E = ex.encode_text(chunk)
            for p_, e in zip(chunk, E): temb[hashlib.sha1(p_.encode()).hexdigest()] = e
            keys = list(temb); np.savez(cache, keys=np.array(keys), embs=np.stack([temb[k] for k in keys]))
            print(f"[text] {min(i + 64, len(todo))}/{len(todo)}", flush=True)
        del ex
    temb = {p_: temb[hashlib.sha1(p_.encode()).hexdigest()] for p_ in prompts}
    # 4) score
    results = []
    for vrel, items in per_variant:
        rows = []
        for v, item_id, seed, own, eff in items:
            r = dict(item_id=item_id, seed=seed, arm=harness_arm_of(item_id), prompt_own=own, prompt_effect=eff, missing=[])
            if not fs.has(v, NS):
                r["missing"].append(f"{NS}:gen"); rows.append(r); continue
            f = fs.get(v, NS)["feat"].astype(np.float32)
            r["text_own"] = float(np.dot(f, temb[own])) if own else None
            r["text_effect"] = float(np.dot(f, temb[eff])) if eff else None
            rows.append(r)
        fin = lambda k: [r[k] for r in rows if isinstance(r.get(k), float) and math.isfinite(r[k])]
        cov = dict(n=len(rows), own_defined=len(fin("text_own")), effect_defined=len(fin("text_effect")), missing=sum(1 for r in rows if r["missing"]),
                   text_own_mean=(round(float(np.mean(fin("text_own"))), 4) if fin("text_own") else None),
                   text_effect_mean=(round(float(np.mean(fin("text_effect"))), 4) if fin("text_effect") else None))
        results.append(dict(harness_arm=rows[0]["arm"], gen=vrel, rows=rows, coverage=cov))
        print(f"[score] {rows[0]['arm']:<36} {cov}", flush=True)
    ed = write_eval(eval_id, results, created=args.date, instrument="scripts/viclip_text.py",
                    definition=[ln.strip() for ln in __doc__.splitlines()[2:11] if ln.strip()],
                    why="VBench overall consistency: which system's output is most consistent with its text; for a reference-driven method a LOW value on the neutral prompt is the intended isolation from text.",
                    caveat="ViCLIP is trained on InternVid captions: coarse video-text similarity in a narrow range (~0.1-0.35). The prior works run on their author-native prompt, our arms on the neutral (or effect) prompt; text_effect scores every arm against the same effect description of the row.",
                    extra={"namespace": NS, "text_tower": "ViCLIP ViT-L/14 InternVid-10M-FLT (same checkpoint)", "trigger_removed": "sksz"})
    print(f"[eval] wrote {ed}"); return 0


if __name__ == "__main__":
    sys.exit(main())
