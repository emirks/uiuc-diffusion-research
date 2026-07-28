#!/usr/bin/env python
"""A12 condition 4 (ADVISORY): CLIP text-image diagonal argmax inside each pack.

For every pack, description i must be more similar to clip i's nine anchor frames than to any
of the other K-1 clips IN THE SAME PACK.  This is the direct test of whether a packed response
attached each description to the right snippet -- independent of the auditor, and free.

It also adjudicates the one pack (P06) whose response echoed all ten ids intact but with two
ADJACENT items transposed: CLIP is asked which of the two competing assignments (key-by-echoed
-id vs key-by-array-position) the pixels actually support.

Bar: diagonal argmax >= 98% within each pack.  Exceedance => REVIEW, not fail.
INTERPRETER: $LAB/envs/diffusion/bin/python  (torch + transformers + cv2; no sklearn needed)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

#: The pool embeddings (`content_pool_emb_union.npy`) use exactly this checkpoint, so the
#: similarity geometry here is the same one the pool's diversity gates were measured in.
#: Loaded from a MERGED local snapshot: the cached HF revision that carries the configs and
#: tokenizer only ships `pytorch_model.bin`, and transformers 4.57 refuses `torch.load` on
#: torch 2.5 (CVE-2025-32434), while the revision that ships `model.safetensors` carries no
#: config.  The merged directory symlinks the configs from the first and the weights from the
#: second, so the load is offline and pinned rather than depending on a network fetch.
MODEL = "/projects/illinois/eng/cs/jrehg/users/emirkisa/cache/clip_vitb32_merged"


def frames(path: str, max_n: int = 9) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    out = []
    while len(out) < max_n:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    D = Path(a.dir)
    rows = json.loads((D / "packed_rows.json").read_text())
    plan = json.loads((D / "pack_plan.json").read_text())
    idx = json.loads((Path(__file__).resolve().parents[3]
                      / "data/processed/caption_strips/strips_index.json").read_text())

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL, use_safetensors=True).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)
    print(f"[clip] {MODEL} on {dev}")

    byrow = {(r["clip_id"], r["role"]): r for r in rows}
    packs = {p["pack_id"]: p for p in plan["packs"]}
    img_cache: dict[tuple, np.ndarray] = {}

    @torch.no_grad()
    def img_emb(clip_id: str, role: str) -> np.ndarray:
        k = (clip_id, role)
        if k not in img_cache:
            fr = frames(idx[clip_id][f"{role}_video"])
            b = proc(images=fr, return_tensors="pt").to(dev)
            e = model.get_image_features(**b)
            e = e / e.norm(dim=-1, keepdim=True)
            v = e.mean(0)
            img_cache[k] = (v / v.norm()).cpu().numpy()
        return img_cache[k]

    @torch.no_grad()
    def txt_emb(texts: list[str]) -> np.ndarray:
        b = proc(text=texts, return_tensors="pt", padding=True, truncation=True,
                 max_length=77).to(dev)
        e = model.get_text_features(**b)
        return (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()

    per_pack, hits, tot = [], 0, 0
    for pid, p in packs.items():
        items = [byrow[(i["clip_id"], i["role"])] for i in p["items"]]
        if any(not r["description"] for r in items):
            continue
        T = txt_emb([r["description"] for r in items])
        I = np.stack([img_emb(r["clip_id"], r["role"]) for r in items])
        S = T @ I.T                                   # rows = descriptions, cols = clips
        am = S.argmax(1)
        ok = int((am == np.arange(len(items))).sum())
        hits += ok
        tot += len(items)
        per_pack.append({"pack_id": pid, "k": len(items), "diagonal_hits": ok,
                         "pct": round(100.0 * ok / len(items), 1),
                         "misattributed": [{"code": items[i]["code"],
                                            "clip": items[i]["clip_id"],
                                            "best_match_clip": items[am[i]]["clip_id"],
                                            "sim_self": round(float(S[i, i]), 4),
                                            "sim_best": round(float(S[i, am[i]]), 4)}
                                           for i in range(len(items)) if am[i] != i]})
    pct = 100.0 * hits / tot if tot else 0.0

    # ---- adjudicate the transposed pack ----------------------------------
    adj = None
    recs = [json.loads(x) for x in (D / "raw_generation_responses.jsonl").open()]
    for rec in recs:
        codes = [i["code"] for i in rec["items"]]
        got = [str(o["id"]) for o in (rec.get("parsed") or []) if isinstance(o, dict)]
        if got == codes or sorted(got) != sorted(codes):
            continue
        texts = [o["description"] for o in rec["parsed"]]
        T = txt_emb(texts)
        # by echoed id: description at array position j belongs to the item whose code == got[j]
        pos_of = {c: j for j, c in enumerate(codes)}
        I = np.stack([img_emb(i["clip_id"], i["role"]) for i in rec["items"]])
        s_id = float(np.mean([T[j] @ I[pos_of[got[j]]] for j in range(len(got))]))
        s_pos = float(np.mean([T[j] @ I[j] for j in range(len(got))]))
        detail = []
        for j in range(len(got)):
            if got[j] != codes[j]:
                detail.append({"array_pos": j, "requested_code": codes[j],
                               "echoed_code": got[j],
                               "sim_to_echoed_id_clip": round(float(T[j] @ I[pos_of[got[j]]]), 4),
                               "sim_to_position_clip": round(float(T[j] @ I[j]), 4)})
        adj = {"pack_id": rec["pack_id"],
               "mean_sim_key_by_echoed_id": round(s_id, 4),
               "mean_sim_key_by_array_position": round(s_pos, 4),
               "winner": "echoed_id" if s_id > s_pos else "array_position",
               "transposed_items": detail}

    R = {"model": MODEL, "bar": "diagonal argmax >= 98% (ADVISORY)",
         "n_items": tot, "diagonal_hits": hits, "diagonal_pct": round(pct, 2),
         "verdict": "PASS" if pct >= 98.0 else "REVIEW",
         "transposition_adjudication": adj, "per_pack": per_pack}
    if a.out:
        Path(a.out).write_text(json.dumps(R, indent=1))
    print(f"[clip] diagonal argmax {hits}/{tot} = {pct:.2f}%   bar >=98%  -> {R['verdict']}")
    for q in per_pack:
        if q["misattributed"]:
            print(f"  {q['pack_id']} {q['pct']}%: " + "; ".join(
                f"{m['code']} ({m['clip']}) matched {m['best_match_clip']} "
                f"[self {m['sim_self']} vs best {m['sim_best']}]" for m in q["misattributed"]))
    if adj:
        print(f"\n[clip] TRANSPOSITION ADJUDICATION for {adj['pack_id']}:")
        print(f"  keying by ECHOED ID       mean sim {adj['mean_sim_key_by_echoed_id']}")
        print(f"  keying by ARRAY POSITION  mean sim {adj['mean_sim_key_by_array_position']}")
        print(f"  => the pixels support: {adj['winner']}")
        for d in adj["transposed_items"]:
            print(f"     pos {d['array_pos']}: requested {d['requested_code']} / echoed "
                  f"{d['echoed_code']} | sim(echoed-id clip)={d['sim_to_echoed_id_clip']} "
                  f"sim(position clip)={d['sim_to_position_clip']}")


if __name__ == "__main__":
    main()
