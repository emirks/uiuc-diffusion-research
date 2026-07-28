#!/usr/bin/env python
"""BUILD pass for the HumanVid endpoint bank.

candidates.jsonl + detections.jsonl -> QC cascade (thresholds verbatim from the blessed
bank) -> TIGHTENED policy pre-filter (area>=0.15, score>=0.7 — bank_tightened parity)
-> CLIP farthest-point diversity cap (--cap N, default 1500) -> window standardize to
std121 (480x640, 121f, 24fps) -> cut gate -> DEDUP (against the existing 227 bank first —
vcbench is also Pexels-sourced — then within) -> encode survivors.

Writes: clips/<id>.mp4, manifest.jsonl, embeddings.npy(+ids), license_ledger.json,
        _work/qc_log.jsonl, _work/build_report.json, bank_sample_sheet.png
Usage: build.py [--dry] [--cap N]
"""
import collections
import json
import os
import random
import sys

import cv2
import numpy as np

from hv_common import (
    BANK, CLIPS, CUT_THRESH, CLIP_COS_DUP, FIT_TOL, OLD_BANK, PHASH_THRESH,
    SAL_COMPACT_MAX, SUBJ_MIN_AREA, TEXT_CAP_REJECT, TEXT_CORNER_REJECT,
    TEXT_COVER_REJECT, TIGHT_AREA, TIGHT_SCORE, W, H, F, WORK, center_window,
    encode_std, hamming, phash, resize_crop_subject, text_stats_from_boxes,
    window_indices,
)

os.makedirs(CLIPS, exist_ok=True)


def text_reject(ts):
    if ts["text_cap_max"] > TEXT_CAP_REJECT:
        return True, f"caption_text={ts['text_cap_max']:.4f}"
    if ts["watermark_persist"] and ts["text_corner_max"] > TEXT_CORNER_REJECT:
        return True, f"watermark_corner={ts['text_corner_max']:.4f}"
    if ts["text_cover_max"] > TEXT_COVER_REJECT:
        return True, f"text_cover={ts['text_cover_max']:.4f}"
    return False, ""


def subject_of(d):
    s = d.get("subject", {})
    if s.get("present"):
        area = s.get("w_frac", 0.3) * s.get("h_frac", 0.3)
        return True, s["cx"], s["cy"], s.get("w_frac", 0.3), area, s
    sal = d.get("saliency")
    if sal and sal.get("compact_area", 1.0) < SAL_COMPACT_MAX:
        return True, sal["cx"], sal["cy"], 0.3, None, {"present": True, "via": "saliency",
                                                       "label": "salient_object", **sal}
    return False, 0.5, 0.5, 0.3, 0.0, {"present": False, "via": s.get("via", "none")}


def standardize(path, w0, w1, cx, cy):
    """Decode window -> crop -> (crops, cut_flag, (sig, cut_s, cut_e), ok)."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 121:
        cap.release(); return None, True, None, False
    want = window_indices(w0, min(w1, n - 1))
    wanted = set(int(i) for i in want)
    got = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, w0)
    for i in range(w0, min(w1, n - 1) + 1):
        ok, fr = cap.read()
        if not ok:
            break
        if i in wanted:
            got[i] = resize_crop_subject(fr, cx, cy)
    cap.release()
    crops = [got.get(int(i)) for i in want]
    if all(c is None for c in crops):
        return None, True, None, False
    last = next(c for c in crops if c is not None)
    for k in range(F):          # fill rare decode gaps with nearest
        if crops[k] is None:
            crops[k] = last
        last = crops[k]

    def win_cut(idxs):
        g = [cv2.cvtColor(cv2.resize(crops[i], (120, 160)), cv2.COLOR_BGR2GRAY) for i in idxs]
        diffs = [np.mean(np.abs(a.astype(int) - b.astype(int))) / 255.0 for a, b in zip(g[:-1], g[1:])]
        return max(diffs) if diffs else 0.0

    cut_s = win_cut(list(range(0, 9)))
    cut_e = win_cut(list(range(112, 121)))
    cut = (cut_s > CUT_THRESH) or (cut_e > CUT_THRESH)
    sig = np.concatenate([phash(cv2.cvtColor(crops[0], cv2.COLOR_BGR2GRAY)),
                          phash(cv2.cvtColor(crops[120], cv2.COLOR_BGR2GRAY))])
    return crops, cut, (sig, round(cut_s, 3), round(cut_e, 3)), True


def old_bank_refs():
    """(sigs, embs, ids) of the existing 227-clip bank for cross-source dedup."""
    sigs, ids = [], []
    man = os.path.join(OLD_BANK, "manifest.jsonl")
    if not os.path.exists(man):
        return [], None, []
    for line in open(man):
        r = json.loads(line)
        p = os.path.join(OLD_BANK, r["mp4"])
        cap = cv2.VideoCapture(p)
        ok0, f0 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
        ok1, f1 = cap.read()
        cap.release()
        if not (ok0 and ok1):
            continue
        sigs.append(np.concatenate([phash(cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)),
                                    phash(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY))]))
        ids.append(r["clip_id"])
    embp = os.path.join(OLD_BANK, "embeddings.npy")
    embs = np.load(embp) if os.path.exists(embp) else None
    return sigs, embs, ids


def fps_diversity_cap(rows, cap):
    """Greedy farthest-point selection on CLIP embeddings."""
    if len(rows) <= cap:
        return rows
    embs = np.array([r[2]["embed"] for r in rows], dtype=np.float32)
    picked = [0]
    dmin = 1.0 - embs @ embs[0]
    while len(picked) < cap:
        i = int(np.argmax(dmin))
        picked.append(i)
        dmin = np.minimum(dmin, 1.0 - embs @ embs[i])
    keep = set(picked)
    return [r for j, r in enumerate(rows) if j in keep]


def main():
    dry = "--dry" in sys.argv
    cap_n = int(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else 1500

    cands = {c["orig_ref"]: c for c in (json.loads(l) for l in open(os.path.join(WORK, "candidates.jsonl")))}
    dets = {}
    for l in open(os.path.join(WORK, "detections.jsonl")):
        d = json.loads(l); dets[d["orig_ref"]] = d

    counts = collections.Counter()
    qc_log = []
    passed = []
    for ref, c in cands.items():
        d = dets.get(ref)
        rec = {"orig_ref": ref, "orig_id": c["orig_id"], "split": c["split"]}
        if d is None or not d.get("ok"):
            counts["reject_unreadable"] += 1
            rec.update(outcome="reject_unreadable", reason=(d or {}).get("reason", "no_detection"))
            qc_log.append(rec); continue
        ts = text_stats_from_boxes(d.get("east_boxes", {}), d["w"], d["h"])
        tr, treason = text_reject(ts)
        if tr:
            counts["reject_text"] += 1
            rec.update(outcome="reject_text", reason=treason); qc_log.append(rec); continue
        present, cx, cy, w_frac, area, sdesc = subject_of(d)
        if not present or (area is not None and area < SUBJ_MIN_AREA):
            counts["reject_subjectless"] += 1
            rec.update(outcome="reject_subjectless",
                       reason=f"via={sdesc.get('via')},area={area}"); qc_log.append(rec); continue
        s = max(W / d["w"], H / d["h"])
        if w_frac * d["w"] * s > W * FIT_TOL:
            counts["reject_fit"] += 1
            rec.update(outcome="reject_fit", reason=f"scaled_w>{W*FIT_TOL:.0f}")
            qc_log.append(rec); continue
        counts["pass_loose"] += 1
        # tightened-policy pre-filter (bank_tightened parity) BEFORE the expensive stages
        subj = d.get("subject", {})
        if not (subj.get("present") and area is not None
                and area >= TIGHT_AREA and subj.get("score", 0) >= TIGHT_SCORE):
            counts["not_tightened"] += 1
            rec.update(outcome="pass_loose_only", reason=f"area={area},score={subj.get('score')}")
            qc_log.append(rec); continue
        rec.update(outcome="tightened", cx=cx, cy=cy, subject=sdesc, **ts)
        passed.append((c, d, {**rec, "embed": d.get("embed")}))

    print(f"[cascade] in={len(cands)} unreadable={counts['reject_unreadable']} "
          f"text={counts['reject_text']} subjectless={counts['reject_subjectless']} "
          f"fit={counts['reject_fit']} -> loose={counts['pass_loose']} "
          f"tightened={len(passed)}", flush=True)

    no_emb = [p for p in passed if not p[2].get("embed")]
    if no_emb:
        counts["no_embed"] += len(no_emb)
        passed = [p for p in passed if p[2].get("embed")]
        print(f"[warn] {len(no_emb)} tightened rows lack a CLIP embed -> dropped pre-cap", flush=True)

    passed = fps_diversity_cap(passed, cap_n)
    print(f"[diversity] capped to {len(passed)} (cap={cap_n})", flush=True)
    if dry:
        with open(os.path.join(WORK, "qc_log_dry.jsonl"), "w") as f:
            for r in qc_log:
                f.write(json.dumps(r) + "\n")
        return

    ob_sigs, ob_embs, ob_ids = old_bank_refs()
    print(f"[dedup-ref] existing bank: {len(ob_sigs)} sigs, "
          f"{0 if ob_embs is None else len(ob_embs)} embeds", flush=True)

    kept = []
    for c, d, rec in passed:
        w0, w1 = d.get("window") or center_window(d["n_frames"], d.get("fps") or 24.0)
        try:
            crops, cut, sig_t, ok = standardize(c["path"], int(w0), int(w1), rec["cx"], rec["cy"])
        except Exception as e:
            counts["reject_encode_err"] += 1
            qc_log.append({**{k: rec[k] for k in ("orig_ref", "orig_id", "split")},
                           "outcome": "reject_encode_err", "reason": str(e)[:120]})
            continue
        if not ok:
            counts["reject_unreadable"] += 1; continue
        if cut:
            counts["reject_cut"] += 1
            qc_log.append({**{k: rec[k] for k in ("orig_ref", "orig_id", "split")},
                           "outcome": "reject_cut", "reason": f"cut={sig_t[1]},{sig_t[2]}"})
            continue
        sig = sig_t[0]
        emb = np.array(rec["embed"], dtype=np.float32) if rec.get("embed") else None
        dup = None
        for j, os_ in enumerate(ob_sigs):        # vs existing bank first
            if hamming(sig, os_) <= PHASH_THRESH:
                dup = f"oldbank:{ob_ids[j]}"; break
        if dup is None and emb is not None and ob_embs is not None:
            cos = ob_embs @ emb
            jj = int(np.argmax(cos))
            if float(cos[jj]) >= CLIP_COS_DUP:
                dup = f"oldbank_cos:{ob_ids[jj]}"
        if dup is None:
            for kc, kr, ksig, kemb in kept:      # within-humanvid
                hd = hamming(sig, ksig)
                if hd <= PHASH_THRESH or (
                        emb is not None and kemb is not None
                        and hd <= PHASH_THRESH * 2 and float(emb @ kemb) >= CLIP_COS_DUP):
                    dup = kr["clip_id"]; break
        if dup:
            counts["reject_dedup"] += 1
            qc_log.append({**{k: rec[k] for k in ("orig_ref", "orig_id", "split")},
                           "outcome": "reject_dedup", "dup_of": dup})
            continue
        clip_id = f"humanvid_{c['orig_id']}"
        out = os.path.join(CLIPS, clip_id + ".mp4")
        try:
            encode_std(crops, out)
        except Exception as e:
            counts["reject_encode_err"] += 1
            qc_log.append({**{k: rec[k] for k in ("orig_ref", "orig_id", "split")},
                           "outcome": "reject_encode_err", "reason": str(e)[:120]})
            continue
        rec.update(clip_id=clip_id, cut_s=sig_t[1], cut_e=sig_t[2], window=[int(w0), int(w1)])
        kept.append((c, rec, sig, emb))
        if len(kept) % 50 == 0:
            print(f"  encoded {len(kept)}...", flush=True)

    emb_rows, emb_ids = [], []
    with open(os.path.join(BANK, "manifest.jsonl"), "w") as mf:
        for c, rec, sig, emb in kept:
            d = dets[c["orig_ref"]]
            mf.write(json.dumps({
                "clip_id": rec["clip_id"], "mp4": f"clips/{rec['clip_id']}.mp4",
                "source": "humanvid", "orig_id": c["orig_id"], "orig_ref": c["orig_ref"],
                "url": c["url"], "license": c["license"], "split": c["split"],
                "orig_resolution": f"{d['w']}x{d['h']}", "orig_fps": d.get("fps"),
                "orig_num_frames": d.get("n_frames"), "window": rec["window"],
                "subject": rec["subject"], "crop_centroid_norm": [rec["cx"], rec["cy"]],
                "qc": {"text_cover_max": rec.get("text_cover_max"),
                       "cut_start_score": rec.get("cut_s"), "cut_end_score": rec.get("cut_e")},
                "std": {"width": W, "height": H, "frames": F, "fps": 24},
                "embed_model": "openai/clip-vit-base-patch32",
                "embed_row": len(emb_rows),
            }) + "\n")
            emb_rows.append(emb.tolist())
            emb_ids.append(rec["clip_id"])
    if emb_rows:
        np.save(os.path.join(BANK, "embeddings.npy"), np.array(emb_rows, dtype=np.float32))
        json.dump(emb_ids, open(os.path.join(BANK, "embed_ids.json"), "w"))

    with open(os.path.join(WORK, "qc_log.jsonl"), "w") as f:
        for r in qc_log:
            f.write(json.dumps(r) + "\n")

    json.dump({
        "humanvid": {"accepted": len(kept), "candidates": len(cands),
                     "license": "Pexels via HumanVid URL lists",
                     "note": ("Pexels ToS restricts ML use (notes/dataset/humanvid_real.md); "
                              "OWNER CLEARED USE 2026-07-27 (recorded in "
                              "$LAB/misc/ctt_v2/DOSSIER.md §11).")},
    }, open(os.path.join(BANK, "license_ledger.json"), "w"), indent=2)
    json.dump({"counts": dict(counts), "accepted": len(kept), "cap": cap_n,
               "splits": dict(collections.Counter(r["split"] for _, r, _, _ in kept))},
              open(os.path.join(WORK, "build_report.json"), "w"), indent=2)
    print(f"[build] cut={counts['reject_cut']} dedup={counts['reject_dedup']} "
          f"err={counts['reject_encode_err']} ACCEPTED={len(kept)}", flush=True)
    make_sheet(kept)


def make_sheet(kept, n=24, seed=42):
    from PIL import Image, ImageDraw
    pick = kept if len(kept) <= n else random.Random(seed).sample(kept, n)
    tw, th, pad, caph, cols = 150, 200, 6, 18, 6
    rows = (len(pick) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 2 * pad), rows * (th + caph + 2 * pad)), (18, 18, 20))
    dr = ImageDraw.Draw(sheet)
    for i, (c, rec, _, _) in enumerate(pick):
        cap = cv2.VideoCapture(os.path.join(CLIPS, rec["clip_id"] + ".mp4"))
        strip = Image.new("RGB", (tw, th), (0, 0, 0))
        for j, fi in enumerate([0, 60, 120]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if ok:
                fr = cv2.cvtColor(cv2.resize(fr, (tw, th // 3)), cv2.COLOR_BGR2RGB)
                strip.paste(Image.fromarray(fr), (0, j * (th // 3)))
        cap.release()
        x = (i % cols) * (tw + 2 * pad) + pad
        y = (i // cols) * (th + caph + 2 * pad) + pad
        sheet.paste(strip, (x, y))
        dr.text((x, y + th + 3), rec["clip_id"][:26], fill=(210, 210, 210))
    out = os.path.join(BANK, "bank_sample_sheet.png")
    sheet.save(out)
    print(f"[sheet] {len(pick)} clips -> {out}")


if __name__ == "__main__":
    main()
