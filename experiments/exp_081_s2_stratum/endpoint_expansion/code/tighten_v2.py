#!/usr/bin/env python
"""ctt_v2 endpoint EXPANSION — acceptance stage: floor, tightening, similarity guards, review.

Runs after build_v2.py has standardised the survivors to the std121 contract. Three stages:

    --stage guards    apply the aesthetic floor + the round-1 tightening policy + both
                      similarity guards; emit REVIEW_QUEUE.json
    --stage sheets    build filmstrip contact sheets so every surviving candidate can be
                      inspected ONE BY ONE (the owner's explicit condition on expansion)
    --stage finalize  consume the operator's visual verdicts and write bank_tightened_v2.json

ADVISOR RULINGS IMPLEMENTED (fable-advisor, ctt_v2 S2/S3, expansion reversal + floor ruling):

  * AESTHETIC FLOOR = 0.4. The 0.4-0.6 band is where the yield is and where "aesthetic score
    != endpoint fitness" holds. The 0.0-0.4 band is discarded: it buys ~35 candidates (~6% of
    the expansion) while widening the quality tail of targets that make up ~67% of the training
    mix, and an IC-LoRA can absorb the quality distribution of its targets.
  * num_scenes == 1 is ABSOLUTE (ruled, not discretionary). A multi-scene source contains a cut
    by construction; admitting one puts a hard discontinuity inside a 121-frame endpoint stream,
    which is the D1 failure class. The 300 multi-scene candidates are never collected.
  * GUARD 1  CLIP cos <= 0.90 against every existing bank member AND every already-accepted new
    member (explicit dedup threshold).
  * GUARD 2  CLIP cos <= 0.85 against each of the 20 RESERVED clips. Violators are DROPPED
    ENTIRELY, not routed to training — the reserve's unseen-ness is load-bearing for the eval
    grid and must survive expansion.
  * New endpoints join the TRAINING pool only. The frozen 207/20 split is not reopened.
  * The round-1 bank is READ-ONLY: this writes an ADDITIVE `bank_tightened_v2.json` with new
    clip ids under a separate tree. No existing row is mutated.
  * Report funnel + visual-review accept rates BROKEN DOWN BY AESTHETIC BAND (0.4-0.5,
    0.5-0.6, >=0.6) so the floor is priced on data for any future round.
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = "/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research"
sys.path.insert(0, os.path.join(REPO, "experiments/exp_081_s2_stratum"))
from letterbox import is_letterboxed, REJECT_FRAC  # noqa: E402
V2 = os.path.join(REPO, "data/processed/ctt_v2_strata/endpoints_v2")
WORK = os.path.join(V2, "_work")
OLD_BANK = os.path.join(REPO, "data/processed/synth_endpoints")
SPLIT = os.path.join(REPO, "data/processed/ctt_v2_strata/ENDPOINT_SPLIT.json")

AES_FLOOR = 0.4          # advisor ruling
DUP_COS = 0.90           # guard 1
RESERVED_COS = 0.85      # guard 2 — violators dropped entirely
# GUARD 0 — letterbox. 20 of the 227 round-1 bank clips carry baked-in black mattes (up to 34%
# of frame height); the advisor excluded them unconditionally and required the same bar here:
# "grandfathering old clips to a lower bar than new ones is indefensible".

# round-1 tightening policy, copied verbatim from bank_tightened.json::policy
TIGHTEN = {"area_min": 0.15, "score_min": 0.7,
           "keep_labels": ["airplane", "bear", "bicycle", "bird", "boat", "bus", "car", "cat",
                           "cow", "dog", "elephant", "giraffe", "horse", "motorcycle", "person",
                           "sheep", "train", "truck", "zebra"]}


def band(a: float) -> str:
    if a >= 0.6:
        return ">=0.6"
    return "0.5-0.6" if a >= 0.5 else "0.4-0.5"


def load_new():
    rows = [json.loads(l) for l in open(os.path.join(V2, "manifest.jsonl"))]
    cands = {c["orig_id"]: c for c in
             (json.loads(l) for l in open(os.path.join(WORK, "candidates_v2.jsonl")))}
    for r in rows:
        r["aesthetic"] = float(cands[r["orig_id"]]["meta"].get("aesthetic") or 0.0)
        r["category"] = cands[r["orig_id"]]["meta"].get("category")
    return rows


def stage_guards() -> None:
    rows = load_new()
    split = json.load(open(SPLIT))
    reserved = split["reserved_eval_only"]

    old_E = np.load(os.path.join(OLD_BANK, "embeddings.npy")).astype(np.float64)
    old_ids = json.load(open(os.path.join(OLD_BANK, "embed_ids.json")))
    old_E /= np.linalg.norm(old_E, axis=1, keepdims=True)
    orow = {c: i for i, c in enumerate(old_ids)}
    R = np.stack([old_E[orow[c]] for c in reserved])

    funnel = {"in_manifest": len(rows)}
    stages = []
    kept = []
    accepted_emb: list[np.ndarray] = []

    # deterministic order: strongest candidates first, so the peer-dedup guard keeps the best
    # of any near-duplicate cluster rather than whichever happened to be read first
    rows.sort(key=lambda r: (-r["aesthetic"], r["clip_id"]))

    for r in rows:
        rec = {"clip_id": r["clip_id"], "aesthetic": r["aesthetic"], "band": band(r["aesthetic"]),
               "category": r["category"], "label": (r.get("subject") or {}).get("label")}
        if r["aesthetic"] < AES_FLOOR:
            rec["outcome"] = "reject_aesthetic_floor"
            stages.append(rec)
            continue
        lb_bad, lb_m = is_letterboxed(os.path.join(V2, r["mp4"]))
        rec["letterbox_max"] = lb_m["max"]
        if lb_bad:                                    # GUARD 0
            rec["outcome"] = "reject_letterbox"
            stages.append(rec)
            continue
        s = r.get("subject") or {}
        area = float(s.get("w_frac", 0)) * float(s.get("h_frac", 0))
        rec["area"] = round(area, 4)
        rec["score"] = round(float(s.get("score", 0)), 4)
        if not (s.get("present") and area >= TIGHTEN["area_min"]
                and float(s.get("score", 0)) >= TIGHTEN["score_min"]
                and s.get("label") in TIGHTEN["keep_labels"]):
            rec["outcome"] = "reject_tighten"
            stages.append(rec)
            continue

        e = np.asarray(r["embedding"], dtype=np.float64)
        e /= np.linalg.norm(e)
        c_res = float((R @ e).max())
        rec["max_cos_vs_reserved"] = round(c_res, 4)
        if c_res > RESERVED_COS:                      # GUARD 2 — dropped entirely
            rec["outcome"] = "reject_guard2_reserved"
            stages.append(rec)
            continue
        c_old = float((old_E @ e).max())
        c_new = float(max((np.asarray(a) @ e for a in accepted_emb), default=0.0))
        rec["max_cos_vs_bank"] = round(c_old, 4)
        rec["max_cos_vs_accepted"] = round(c_new, 4)
        if max(c_old, c_new) > DUP_COS:               # GUARD 1
            rec["outcome"] = "reject_guard1_dup"
            stages.append(rec)
            continue

        rec["outcome"] = "pass_to_visual_review"
        accepted_emb.append(e)
        kept.append(r)
        stages.append(rec)

    by_outcome: dict = {}
    for rec in stages:
        by_outcome.setdefault(rec["outcome"], 0)
        by_outcome[rec["outcome"]] += 1
    by_band: dict = {}
    for rec in stages:
        if rec["aesthetic"] < AES_FLOOR:
            continue
        b = rec["band"]
        d = by_band.setdefault(b, {"in": 0, "passed_guards": 0})
        d["in"] += 1
        d["passed_guards"] += int(rec["outcome"] == "pass_to_visual_review")

    funnel.update(by_outcome)
    out = {"created": "2026-07-25", "stage": "guards",
           "authority": "fable-advisor expansion reversal + floor=0.4 ruling",
           "policy": {"aesthetic_floor": AES_FLOOR, "dup_cos": DUP_COS,
                      "reserved_cos": RESERVED_COS, "tighten": TIGHTEN,
                      "letterbox_reject_frac": REJECT_FRAC,
                      "num_scenes": "== 1, absolute (ruled)"},
           "funnel": funnel, "by_aesthetic_band": by_band,
           "n_for_review": len(kept),
           "per_clip": stages,
           "review_queue": [r["clip_id"] for r in kept]}
    json.dump(out, open(os.path.join(WORK, "REVIEW_QUEUE.json"), "w"), indent=1)
    print(f"[guards] funnel: {json.dumps(funnel)}")
    print(f"[guards] by aesthetic band: {json.dumps(by_band)}")
    print(f"[guards] {len(kept)} candidates -> visual review")


def stage_sheets(per_sheet: int = 10) -> None:
    """Filmstrip contact sheets — one row per candidate, so motion and cuts are visible."""
    import cv2
    import PIL.Image
    import PIL.ImageDraw

    q = json.load(open(os.path.join(WORK, "REVIEW_QUEUE.json")))
    ids = q["review_queue"]
    info = {r["clip_id"]: r for r in q["per_clip"]}
    sheets_dir = os.path.join(V2, "review_sheets")
    os.makedirs(sheets_dir, exist_ok=True)

    IDX = [0, 13, 26, 40, 53, 67, 80, 93, 107, 120]      # spans the full 121-frame clip
    TW, TH = 128, 171                                     # 3:4 portrait thumbs
    LABEL_H = 18

    sheets = []
    for s0 in range(0, len(ids), per_sheet):
        chunk = ids[s0: s0 + per_sheet]
        rows_img = []
        for cid in chunk:
            cap = cv2.VideoCapture(os.path.join(V2, "clips", f"{cid}.mp4"))
            frames = []
            for i in IDX:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, f = cap.read()
                frames.append(cv2.resize(f, (TW, TH)) if ok and f is not None
                              else np.zeros((TH, TW, 3), np.uint8))
            cap.release()
            strip = np.concatenate(frames, axis=1)[:, :, ::-1]
            img = PIL.Image.new("RGB", (strip.shape[1], TH + LABEL_H), "white")
            img.paste(PIL.Image.fromarray(strip), (0, LABEL_H))
            m = info[cid]
            PIL.ImageDraw.Draw(img).text(
                (3, 4), f"{cid}   aes={m['aesthetic']:.2f} {m.get('label')} "
                        f"area={m.get('area')} cosBank={m.get('max_cos_vs_bank')}",
                fill="black")
            rows_img.append(np.asarray(img))
        sheet = np.concatenate(rows_img, axis=0)
        n = len(sheets)
        path = os.path.join(sheets_dir, f"sheet_{n:02d}.png")
        PIL.Image.fromarray(sheet).save(path)
        sheets.append({"sheet": path, "clip_ids": chunk})
        print(f"[sheets] {path}  ({len(chunk)} candidates)")
    json.dump({"n_sheets": len(sheets), "per_sheet": per_sheet, "sheets": sheets},
              open(os.path.join(WORK, "REVIEW_SHEETS.json"), "w"), indent=1)
    print(f"[sheets] {len(sheets)} sheets covering {len(ids)} candidates -> {sheets_dir}")


def stage_finalize(verdict_path: str) -> None:
    """Consume the operator's per-clip visual verdicts and emit the additive v2 bank."""
    q = json.load(open(os.path.join(WORK, "REVIEW_QUEUE.json")))
    verdicts = json.load(open(verdict_path))          # {clip_id: "accept" | "<reject reason>"}
    rows = {r["clip_id"]: r for r in load_new()}
    info = {r["clip_id"]: r for r in q["per_clip"]}

    missing = [c for c in q["review_queue"] if c not in verdicts]
    assert not missing, (f"{len(missing)} candidates have no visual verdict — every clip must be "
                         f"seen one by one: {missing[:8]}")

    accepted = [c for c in q["review_queue"] if verdicts[c] == "accept"]
    rejects: dict = {}
    for c in q["review_queue"]:
        if verdicts[c] != "accept":
            rejects.setdefault(verdicts[c], []).append(c)

    by_band: dict = {}
    for c in q["review_queue"]:
        b = info[c]["band"]
        d = by_band.setdefault(b, {"reviewed": 0, "accepted": 0})
        d["reviewed"] += 1
        d["accepted"] += int(verdicts[c] == "accept")
    for b, d in by_band.items():
        d["visual_accept_rate"] = round(d["accepted"] / max(d["reviewed"], 1), 3)

    clips = [{"clip_id": c, "mp4": rows[c]["mp4"], "source": "vcbench_v2",
              "orig_id": rows[c]["orig_id"], "license": rows[c]["license"],
              "aesthetic": info[c]["aesthetic"], "label": info[c]["label"],
              "max_cos_vs_bank": info[c]["max_cos_vs_bank"],
              "max_cos_vs_reserved": info[c]["max_cos_vs_reserved"]}
             for c in accepted]

    out = {"created": "2026-07-25", "version": "v2 (ADDITIVE — round-1 bank untouched)",
           "authority": "fable-advisor expansion reversal, floor=0.4",
           "base_bank": "data/processed/synth_endpoints/bank_tightened.json (227, READ-ONLY)",
           "clips_root": os.path.join(V2, "clips"),
           "policy": q["policy"],
           "guards_funnel": q["funnel"],
           "visual_review": {"reviewed": len(q["review_queue"]), "accepted": len(accepted),
                             "reject_reasons": {k: len(v) for k, v in sorted(rejects.items())},
                             "rejects": rejects},
           "by_aesthetic_band": by_band,
           "n_new": len(accepted),
           "joins": "TRAINING pool only — the frozen 207/20 split is not reopened",
           "clips": clips}
    json.dump(out, open(os.path.join(V2, "bank_tightened_v2.json"), "w"), indent=1)
    print(f"[finalize] visual review: {len(accepted)}/{len(q['review_queue'])} accepted")
    print(f"[finalize] reject reasons: {json.dumps({k: len(v) for k, v in sorted(rejects.items())})}")
    print(f"[finalize] by band: {json.dumps(by_band)}")
    print(f"[finalize] -> {os.path.join(V2, 'bank_tightened_v2.json')}  (+{len(accepted)} endpoints)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["guards", "sheets", "finalize"])
    ap.add_argument("--verdicts", help="json {clip_id: accept|<reason>} for --stage finalize")
    ap.add_argument("--per-sheet", type=int, default=10)
    a = ap.parse_args()
    if a.stage == "guards":
        stage_guards()
    elif a.stage == "sheets":
        stage_sheets(a.per_sheet)
    else:
        if not a.verdicts:
            sys.exit("--verdicts is required for --stage finalize")
        stage_finalize(a.verdicts)
