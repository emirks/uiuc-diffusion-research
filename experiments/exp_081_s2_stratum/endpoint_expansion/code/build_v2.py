#!/usr/bin/env python
"""BUILD pass for the synth endpoint bank.

Consumes _work/candidates.jsonl + _work/detections.jsonl and applies the QC
accept/reject cascade, then standardizes accepted clips to the std121 contract
(480x640, 121f, 24fps, libx264 crf14 -- matches transitions_std121 & exp_062 cond).

QC stages (reject attributed to the FIRST failing stage):
  1 text     : EAST text/caption/watermark coverage over the sampled frames
  2 subject  : COCO subject present (torchvision) OR compact-salient fallback
  3 fit       : subject fits a 480x640 portrait crop after resize-cover
  4 cut       : frame-diff cut inside the start9 [0,9) or end9 [112,121) window
  5 dedup     : pHash(start9-frame0 + end9-frame120) near-dup across ALL sources

Writes: clips/<clip_id>.mp4, manifest.jsonl, qc_log.jsonl, license_ledger.json,
        embeddings.npy (+ embed_ids.json), bank_sample_sheet.png

Usage:  build.py [--dry]      (--dry: cascade + counts only, no encode/sheet)
"""
import os, sys, json, re, random, subprocess, collections
import numpy as np
import cv2

REPO = "/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research"
BANK = os.path.join(REPO, "data/processed/ctt_v2_strata/endpoints_v2")
WORK = os.path.join(BANK, "_work")
CLIPS = os.path.join(BANK, "clips")
FFMPEG = "/u/emirkisa/.local/bin/ffmpeg"
os.makedirs(CLIPS, exist_ok=True)

W, H, F, FPS = 480, 640, 121, 24

# ---- thresholds (logged in report) ----
EAST_CONF = 0.75            # min EAST box confidence counted as real text (FP filter)
TEXT_COVER_REJECT = 0.030   # max total high-conf text cover fraction
TEXT_CAP_REJECT = 0.015     # caption-band (top/bottom) high-conf text cover
TEXT_CORNER_REJECT = 0.010  # corner/watermark-zone high-conf text cover (with persistence)
WM_COVER_MIN = 0.008        # persistent watermark only rejects above this cover
SUBJ_MIN_AREA = 0.05        # dominant subject bbox area frac; below => scenic/subjectless
SAL_COMPACT_MAX = 0.12      # saliency: 50% mass within <12% area => compact object
FIT_TOL = 1.75             # reject only if scaled subject width > 1.75 * 480 (keeps centered wide subjects)
CUT_THRESH = 0.20           # mean abs luma diff (0..1) inside an endpoint window
PHASH_THRESH = 12           # /128 bits hamming for near-dup
CLIP_COS_DUP = 0.985        # CLIP cosine near-dup (paired with a looser pHash)
SRC_PRIORITY = {"davis": 0, "vcbench": 1, "openvid": 2}


def frame_indices(n):
    return np.round(np.linspace(0, n - 1, F)).astype(int)


def resize_crop_subject(frame, cx, cy):
    h, w = frame.shape[:2]
    s = max(W / w, H / h)
    nw, nh = round(w * s), round(h * s)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4
    r = cv2.resize(frame, (nw, nh), interpolation=interp)
    x0 = int(round(cx * nw - W / 2)); y0 = int(round(cy * nh - H / 2))
    x0 = max(0, min(x0, nw - W)); y0 = max(0, min(y0, nh - H))
    return r[y0:y0 + H, x0:x0 + W]


def phash(gray):
    img = cv2.resize(gray, (32, 32)).astype(np.float32)
    d = cv2.dct(img); low = d[:8, :8].flatten()
    med = np.median(low[1:])
    return (low > med)


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def text_from_boxes(eb, conf=EAST_CONF):
    """Recompute text coverage using only high-confidence EAST boxes (FP filter)."""
    W, H = eb.get("w", 0), eb.get("h", 0)
    if not W or not H:
        return None
    area = float(W * H)
    cover_max = cap_max = corner_max = 0.0
    centers_by = {}
    for nm in ["start", "mid", "end"]:
        cov = capc = corc = 0.0; centers = []
        for x0, y0, x1, y1, sc in eb.get(nm, []):
            if sc < conf:
                continue
            a = max(0.0, x1 - x0) * max(0.0, y1 - y0); cov += a
            cx = (x0 + x1) / 2 / W; cy = (y0 + y1) / 2 / H; centers.append((cx, cy))
            if cy > 0.78 or cy < 0.12:
                capc += a
            if (cx < 0.25 or cx > 0.75) and (cy < 0.18 or cy > 0.82):
                corc += a
        cover_max = max(cover_max, cov / area); cap_max = max(cap_max, capc / area)
        corner_max = max(corner_max, corc / area); centers_by[nm] = centers
    wm = False
    allc = [(nm, cc) for nm in centers_by for cc in centers_by[nm]]
    for i in range(len(allc)):
        for j in range(i + 1, len(allc)):
            if allc[i][0] == allc[j][0]:
                continue
            if abs(allc[i][1][0] - allc[j][1][0]) < 0.05 and abs(allc[i][1][1] - allc[j][1][1]) < 0.05:
                wm = True
    return {"text_cover_max": round(cover_max, 5), "text_cap_max": round(cap_max, 5),
            "text_corner_max": round(corner_max, 5), "watermark_persist": wm}


def text_reject(d):
    if d.get("text_cap_max", 0) > TEXT_CAP_REJECT:
        return True, f"caption_text={d['text_cap_max']:.4f}"
    if d.get("watermark_persist") and d.get("text_corner_max", 0) > TEXT_CORNER_REJECT:
        return True, f"watermark_corner={d['text_corner_max']:.4f}"
    if d.get("text_cover_max", 0) > TEXT_COVER_REJECT:
        return True, f"text_cover={d['text_cover_max']:.4f}"
    return False, ""


def subject_of(d):
    """Return (present, cx, cy, w_frac, area, descriptor). area=None for saliency."""
    s = d.get("subject", {})
    if s.get("present"):
        area = s.get("w_frac", 0.3) * s.get("h_frac", 0.3)
        return True, s["cx"], s["cy"], s.get("w_frac", 0.3), area, s
    sal = d.get("saliency")
    if sal and sal.get("compact_area", 1.0) < SAL_COMPACT_MAX:
        return True, sal["cx"], sal["cy"], 0.3, None, {"present": True, "via": "saliency",
                                                       "label": "salient_object",
                                                       "cx": sal["cx"], "cy": sal["cy"],
                                                       "compact_area": sal["compact_area"]}
    return False, 0.5, 0.5, 0.3, 0.0, {"present": False, "via": s.get("via", "none")}


def fit_reject(w_frac, src_w, src_h):
    s = max(W / src_w, H / src_h)
    nw = src_w * s
    scaled_subj_w = w_frac * nw
    return scaled_subj_w > W * FIT_TOL, round(scaled_subj_w, 1)


def standardize_and_qc(path, cx, cy, out_path):
    """Decode -> subject-crop -> return (cropped_frames_list, cut, sig, ok)."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 9:
        cap.release(); return None, True, None, False
    want = frame_indices(n)
    wanted = {int(i): pos for pos, i in enumerate(want)}
    crops = [None] * F
    read = 0
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        if i in wanted:
            c = resize_crop_subject(fr, cx, cy)
            # a source index can map to multiple std positions (short clips)
            for pos, wi in enumerate(want):
                if int(wi) == i:
                    crops[pos] = c
            read += 1
    cap.release()
    if any(c is None for c in crops):
        # fill any gaps (rare) with nearest
        last = None
        for k in range(F):
            if crops[k] is None:
                crops[k] = last if last is not None else crops[next(j for j in range(F) if crops[j] is not None)]
            last = crops[k]
    # cut check on endpoint windows
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


def encode(crops, out_path):
    tmp = out_path + ".tmp.mp4"
    p = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", out_path + ".tmp.mp4"],
        stdin=subprocess.PIPE)
    for c in crops:
        p.stdin.write(np.ascontiguousarray(c[:, :, ::-1]).tobytes())
    p.stdin.close()
    assert p.wait() == 0, "ffmpeg encode failed"
    os.rename(tmp, out_path)


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80]


def main():
    dry = "--dry" in sys.argv
    cands = {c["orig_ref"]: c for c in (json.loads(l) for l in open(os.path.join(WORK, "candidates_v2.jsonl")))}
    dets = {}
    for l in open(os.path.join(WORK, "detections_v2.jsonl")):
        d = json.loads(l); dets[d["orig_ref"]] = d
    # override text stats with high-confidence-box recomputation where available
    ebpath = os.path.join(WORK, "east_boxes.jsonl")
    n_override = 0
    if os.path.exists(ebpath):
        for l in open(ebpath):
            eb = json.loads(l)
            d = dets.get(eb["orig_ref"])
            if d is None or not eb.get("ok"):
                continue
            ts = text_from_boxes(eb)
            if ts:
                d.update(ts); n_override += 1
    print(f"[build] text stats from high-conf EAST boxes for {n_override} candidates (conf>={EAST_CONF})")

    counts = collections.Counter()
    per_source_in = collections.Counter()
    qc_log = []
    accepted = []   # dicts to encode/dedup

    for ref, c in cands.items():
        per_source_in[c["source"]] += 1
        d = dets.get(ref)
        rec = {"orig_ref": ref, "orig_id": c["orig_id"], "source": c["source"]}
        if d is None or not d.get("ok"):
            counts["reject_unreadable"] += 1
            rec["outcome"] = "reject_unreadable"; rec["reason"] = (d or {}).get("reason", "no_detection")
            qc_log.append(rec); continue
        # stage 1 text
        tr, treason = text_reject(d)
        if tr:
            counts["reject_text"] += 1
            rec["outcome"] = "reject_text"; rec["reason"] = treason
            qc_log.append(rec); continue
        # stage 2 subject (present + big enough to be a distinct foreground)
        present, cx, cy, w_frac, area, sdesc = subject_of(d)
        if not present:
            counts["reject_subjectless"] += 1
            rec["outcome"] = "reject_subjectless"; rec["reason"] = f"via={sdesc.get('via')}"
            qc_log.append(rec); continue
        if area is not None and area < SUBJ_MIN_AREA:
            counts["reject_subjectless"] += 1
            rec["outcome"] = "reject_subjectless"; rec["reason"] = f"subj_area={area:.3f}<{SUBJ_MIN_AREA}"
            qc_log.append(rec); continue
        # stage 3 fit
        fr, scaled_w = fit_reject(w_frac, d["w"], d["h"])
        if fr:
            counts["reject_fit"] += 1
            rec["outcome"] = "reject_fit"; rec["reason"] = f"scaled_subj_w={scaled_w}>{W*FIT_TOL:.0f}"
            qc_log.append(rec); continue
        rec.update({"outcome": "pending", "subject": sdesc, "cx": cx, "cy": cy,
                    "w_frac": w_frac, "src_w": d["w"], "src_h": d["h"],
                    "text_cover_max": d.get("text_cover_max"), "embed": d.get("embed")})
        accepted.append((c, d, rec))

    print(f"[cascade] in={dict(per_source_in)}  "
          f"unreadable={counts['reject_unreadable']} text={counts['reject_text']} "
          f"subjectless={counts['reject_subjectless']} fit={counts['reject_fit']}  "
          f"-> pre-cut/dedup accepted={len(accepted)}")

    if dry:
        # still run cut on a small sample for a rate estimate? no -- dry is cascade only
        with open(os.path.join(WORK, "qc_log_dry_v2.jsonl"), "w") as f:
            for r in qc_log:
                f.write(json.dumps(r) + "\n")
        return

    # stage 4 cut + encode survivors
    survivors = []
    for c, d, rec in accepted:
        clip_id = f"{c['source']}_{sanitize(c['orig_id'])}"
        out = os.path.join(CLIPS, clip_id + ".mp4")
        try:
            crops, cut, sig_t, ok = standardize_and_qc(c["path"], rec["cx"], rec["cy"], out)
        except Exception as e:
            counts["reject_encode_err"] += 1
            rec["outcome"] = "reject_encode_err"; rec["reason"] = str(e)[:120]
            qc_log.append(rec); continue
        if not ok:
            counts["reject_unreadable"] += 1
            rec["outcome"] = "reject_unreadable"; rec["reason"] = "decode_short"
            qc_log.append(rec); continue
        if cut:
            counts["reject_cut"] += 1
            rec["outcome"] = "reject_cut"; rec["reason"] = f"cut_s={sig_t[1]},cut_e={sig_t[2]}"
            qc_log.append(rec); continue
        try:
            encode(crops, out)
        except Exception as e:
            counts["reject_encode_err"] += 1
            rec["outcome"] = "reject_encode_err"; rec["reason"] = str(e)[:120]
            qc_log.append(rec); continue
        rec["clip_id"] = clip_id; rec["_sig"] = sig_t[0]
        rec["cut_s"] = sig_t[1]; rec["cut_e"] = sig_t[2]
        survivors.append((c, d, rec))
        if len(survivors) % 25 == 0:
            print(f"  encoded {len(survivors)} survivors...", flush=True)

    # stage 5 dedup (keep by source priority then resolution)
    def qual(item):
        c, d, rec = item
        return (SRC_PRIORITY.get(c["source"], 9), -(d["w"] * d["h"]))
    survivors.sort(key=qual)
    kept = []
    for c, d, rec in survivors:
        emb = np.array(rec["embed"], dtype=np.float32) if rec.get("embed") else None
        dup = False
        for kc, kd, kr in kept:
            hd = hamming(rec["_sig"], kr["_sig"])
            if hd <= PHASH_THRESH:
                dup = True; rec["dup_of"] = kr["clip_id"]; rec["dup_hamming"] = hd; break
            if emb is not None and kr.get("_emb") is not None and hd <= PHASH_THRESH * 2:
                cos = float(emb @ kr["_emb"])
                if cos >= CLIP_COS_DUP:
                    dup = True; rec["dup_of"] = kr["clip_id"]; rec["dup_cos"] = round(cos, 4); break
        if dup:
            counts["reject_dedup"] += 1
            rec["outcome"] = "reject_dedup"
            qc_log.append({k: v for k, v in rec.items() if not k.startswith("_")})
            continue
        rec["_emb"] = emb
        kept.append((c, d, rec))

    # delete encoded clips that got deduped out
    keep_ids = {rec["clip_id"] for _, _, rec in kept}
    for c, d, rec in survivors:
        if rec.get("clip_id") and rec["clip_id"] not in keep_ids:
            fp = os.path.join(CLIPS, rec["clip_id"] + ".mp4")
            if os.path.exists(fp):
                os.remove(fp)

    # write manifest + embeddings + ledger
    emb_rows = []
    emb_ids = []
    with open(os.path.join(BANK, "manifest.jsonl"), "w") as mf:
        for c, d, rec in kept:
            emb = rec.get("embed")
            row = {
                "clip_id": rec["clip_id"],
                "mp4": f"clips/{rec['clip_id']}.mp4",
                "source": c["source"],
                "orig_id": c["orig_id"],
                "orig_ref": c["orig_ref"],
                "license": c["license"],
                "orig_resolution": f"{d['w']}x{d['h']}",
                "orig_fps": d.get("fps"),
                "orig_num_frames": d.get("n_frames"),
                "subject": rec["subject"],
                "crop_centroid_norm": [rec["cx"], rec["cy"]],
                "qc": {"text_cover_max": rec.get("text_cover_max"),
                       "cut_start_score": rec.get("cut_s"), "cut_end_score": rec.get("cut_e"),
                       "cut_tool": "frame-diff(mean-abs-luma, endpoint windows)",
                       "text_tool": "EAST(cv2.dnn)", "subject_tool": "torchvision_fasterrcnn_mnv3_fpn_coco",
                       "dedup_tool": "pHash(start0+end120, 128bit)+CLIP-cosine"},
                "std": {"width": W, "height": H, "frames": F, "fps": FPS},
                "embed_model": "openai/clip-vit-base-patch32", "embed_dim": len(emb) if emb else 0,
                "embed_row": len(emb_rows),
                "embedding": emb,
            }
            mf.write(json.dumps(row) + "\n")
            emb_rows.append(emb); emb_ids.append(rec["clip_id"])
    if emb_rows:
        np.save(os.path.join(BANK, "embeddings.npy"), np.array(emb_rows, dtype=np.float32))
        json.dump(emb_ids, open(os.path.join(BANK, "embed_ids.json"), "w"))

    with open(os.path.join(WORK, "qc_log_v2.jsonl"), "w") as f:
        for r in qc_log:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")
        for c, d, rec in kept:
            f.write(json.dumps({"orig_ref": c["orig_ref"], "source": c["source"],
                                "clip_id": rec["clip_id"], "outcome": "accepted"}) + "\n")

    # license ledger
    comp = collections.Counter(c["source"] for c, _, _ in kept)
    ledger = {}
    lic_by = {c["source"]: c["license"] for c in cands.values()}
    for src in ["vcbench", "davis", "openvid"]:
        ledger[src] = {"accepted": comp.get(src, 0), "candidates": per_source_in.get(src, 0),
                       "license": lic_by.get(src, "?")}
    ledger["_provenance_note"] = ("vcbench=Pexels via VC-Bench; davis=DAVIS-2017 research "
                                  "(per-seq sources in data/raw/DAVIS/SOURCES.md); "
                                  "openvid=OpenVid-1M CC-BY-4.0 (underlying Panda-70M/YouTube). "
                                  "Add Pexels/Pixabay later via the same candidate schema + QC.")
    json.dump(ledger, open(os.path.join(BANK, "license_ledger.json"), "w"), indent=2)

    print(f"[build] cut={counts['reject_cut']} dedup={counts['reject_dedup']} "
          f"encode_err={counts['reject_encode_err']}  ACCEPTED={len(kept)}  {dict(comp)}")

    # contact sheet on a random 24 of the final bank
    make_sheet(kept)
    # persist counts
    json.dump({"per_source_in": dict(per_source_in), "counts": dict(counts),
               "accepted": len(kept), "composition": dict(comp),
               "thresholds": {"TEXT_COVER_REJECT": TEXT_COVER_REJECT, "TEXT_CAP_REJECT": TEXT_CAP_REJECT,
                              "WM_COVER_MIN": WM_COVER_MIN, "SAL_COMPACT_MAX": SAL_COMPACT_MAX,
                              "FIT_TOL": FIT_TOL, "CUT_THRESH": CUT_THRESH,
                              "PHASH_THRESH": PHASH_THRESH, "CLIP_COS_DUP": CLIP_COS_DUP}},
              open(os.path.join(WORK, "build_report.json"), "w"), indent=2)


def make_sheet(kept, n=24, seed=42):
    from PIL import Image, ImageDraw
    rng = random.Random(seed)
    pick = kept if len(kept) <= n else rng.sample(kept, n)
    tw, th = 150, 200
    pad, caph = 6, 18
    cols = 6
    rows = (len(pick) + cols - 1) // cols
    cellw = tw + 2 * pad
    cellh = th + caph + 2 * pad
    sheet = Image.new("RGB", (cols * cellw, rows * cellh), (18, 18, 20))
    dr = ImageDraw.Draw(sheet)
    for i, (c, d, rec) in enumerate(pick):
        cap = cv2.VideoCapture(os.path.join(CLIPS, rec["clip_id"] + ".mp4"))
        frs = {}
        for fi in [0, 60, 120]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, fr = cap.read()
            frs[fi] = fr if ok else np.zeros((H, W, 3), np.uint8)
        cap.release()
        strip = Image.new("RGB", (tw, th), (0, 0, 0))
        for j, fi in enumerate([0, 60, 120]):
            fr = cv2.cvtColor(cv2.resize(frs[fi], (tw, th // 3)), cv2.COLOR_BGR2RGB)
            strip.paste(Image.fromarray(fr), (0, j * (th // 3)))
        cxc = (i % cols) * cellw + pad
        cyc = (i // cols) * cellh + pad
        sheet.paste(strip, (cxc, cyc))
        cap = f"{c['source'][:2]}:{c['orig_id']}"[:26]
        dr.text((cxc, cyc + th + 3), cap, fill=(210, 210, 210))
    out = os.path.join(BANK, "bank_sample_sheet.png")
    sheet.save(out)
    print(f"[sheet] {len(pick)} clips -> {out}")


if __name__ == "__main__":
    main()
