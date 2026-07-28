#!/usr/bin/env python
"""DETECT pass (expensive, cached, idempotent) for the HumanVid endpoint bank.

Per candidate: decode 3 frames near the start/middle/end of the CENTER WINDOW
(hv_common.center_window) and run EAST (raw boxes stored), Faster R-CNN MobileNetV3
subject detection, spectral-residual saliency fallback, CLIP ViT-B/32 embedding.
Appends to _work/detections.jsonl; skips refs already present.

Detector stack identical to the blessed bank's detect.py (provenance in hv_common).
"""
import json
import os
import sys
import time

import cv2
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

from hv_common import DSX_WORK, WORK, center_window, east_detect_raw  # noqa: E402

CAND = os.path.join(WORK, "candidates.jsonl")
OUT = os.path.join(WORK, "detections.jsonl")
EAST_PB = os.path.join(WORK, "frozen_east_text_detection.pb")
CLIP_LOCAL = os.path.join(DSX_WORK, "clip_local")  # read-only fallback
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read_frames(path, idxs):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    for idx in idxs:
        idx = max(0, min(idx, max(0, n - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        frames.append(f if ok and f is not None else None)
    cap.release()
    return frames, n, fps, w, h


def load_detector():
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
    )
    w = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    return fasterrcnn_mobilenet_v3_large_fpn(weights=w).eval().to(DEVICE), w.meta["categories"]


@torch.no_grad()
def detect_subject(model, cats, frames, W0, H0):
    valid = [(k, f) for k, f in enumerate(frames) if f is not None]
    if not valid:
        return {"present": False, "via": "none"}
    tens = [torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(DEVICE)
            for _, f in valid]
    outs = model(tens)
    bests = []
    for out in outs:
        b = out["boxes"].cpu().numpy(); s = out["scores"].cpu().numpy(); l = out["labels"].cpu().numpy()
        best = None
        for i in range(len(b)):
            if s[i] < 0.5:
                continue
            x0, y0, x1, y1 = b[i]
            af = ((x1 - x0) * (y1 - y0)) / float(W0 * H0)
            if af < 0.02:
                continue
            if best is None or af > best[0]:
                best = (af, (x0, y0, x1, y1), float(s[i]), int(l[i]))
        bests.append(best)
    good = [x for x in bests if x is not None]
    if not good:
        return {"present": False, "via": "detector_empty"}
    dom = max(good, key=lambda x: x[0])
    cxs = [((g[1][0] + g[1][2]) / 2.0) / W0 for g in good]
    cys = [((g[1][1] + g[1][3]) / 2.0) / H0 for g in good]
    x0, y0, x1, y1 = dom[1]
    return {"present": True, "via": "detector",
            "label": cats[dom[3]] if dom[3] < len(cats) else str(dom[3]),
            "score": round(dom[2], 3),
            "cx": round(float(np.median(cxs)), 4), "cy": round(float(np.median(cys)), 4),
            "w_frac": round(float((x1 - x0) / W0), 4), "h_frac": round(float((y1 - y0) / H0), 4)}


def saliency_sr(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (128, 128)).astype(np.float32)
    f = np.fft.fft2(small)
    logamp = np.log(np.abs(f) + 1e-8); phase = np.angle(f)
    sr = logamp - cv2.blur(logamp, (3, 3))
    recon = np.abs(np.fft.ifft2(np.exp(sr + 1j * phase))) ** 2
    recon = cv2.GaussianBlur(recon, (0, 0), 3)
    recon = (recon - recon.min()) / (np.ptp(recon) + 1e-8)
    flat = np.sort(recon.ravel())[::-1]
    csum = np.cumsum(flat); csum /= csum[-1]
    k = int(np.searchsorted(csum, 0.5))
    ys, xs = np.nonzero(recon > np.percentile(recon, 90))
    return {"compact_area": round((k + 1) / flat.size, 4),
            "cx": round(float(xs.mean()) / 128 if len(xs) else 0.5, 4),
            "cy": round(float(ys.mean()) / 128 if len(ys) else 0.5, 4)}


def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    src = CLIP_LOCAL if os.path.isdir(CLIP_LOCAL) else "openai/clip-vit-base-patch32"
    m = CLIPModel.from_pretrained(src, local_files_only=True).eval().to(DEVICE)
    p = CLIPProcessor.from_pretrained(src, local_files_only=True)
    return m, p


@torch.no_grad()
def clip_embed(model, proc, bgr):
    from PIL import Image
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    inp = proc(images=img, return_tensors="pt").to(DEVICE)
    feat = model.get_image_features(**inp)[0]
    feat = feat / feat.norm()
    return [round(float(v), 5) for v in feat.cpu().numpy()]


def main():
    import random
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    cands = [json.loads(l) for l in open(CAND)]
    random.Random(0).shuffle(cands)   # deterministic order -> raising --limit later RESUMES
    if limit:
        cands = cands[:limit]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                done.add(json.loads(l)["orig_ref"])
            except Exception:
                pass
    print(f"[detect] {len(cands)} candidates, {len(done)} done, device={DEVICE}", flush=True)

    east = cv2.dnn.readNet(EAST_PB)
    model, cats = load_detector()
    clipm, clipp = load_clip()

    fout = open(OUT, "a")
    t0 = time.time(); k = 0
    for c in cands:
        if c["orig_ref"] in done:
            continue
        k += 1
        rec = {"orig_ref": c["orig_ref"], "orig_id": c["orig_id"], "source": c["source"]}
        try:
            cap = cv2.VideoCapture(c["path"])
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            cap.release()
            if n < 121:
                rec.update(ok=False, reason="too_short_or_unreadable", n_frames=n)
                fout.write(json.dumps(rec) + "\n"); fout.flush(); continue
            w0, w1 = center_window(n, fps)
            wlen = w1 - w0 + 1
            i_s = w0 + int(round(4 / 120.0 * (wlen - 1)))
            i_m = w0 + wlen // 2
            i_e = w0 + int(round(116 / 120.0 * (wlen - 1)))
            frames, n, fps, W0, H0 = read_frames(c["path"], [i_s, i_m, i_e])
            if all(f is None for f in frames) or W0 == 0:
                rec.update(ok=False, reason="decode_failed")
                fout.write(json.dumps(rec) + "\n"); fout.flush(); continue
            rec.update(ok=True, n_frames=n, fps=round(fps, 2), w=W0, h=H0,
                       window=[int(w0), int(w1)])
            boxes_by = {}
            for nm, f in zip(["start", "mid", "end"], frames):
                if f is not None:
                    boxes_by[nm] = east_detect_raw(east, f)
            rec["east_boxes"] = boxes_by
            rec["subject"] = detect_subject(model, cats, frames, W0, H0)
            mid = frames[1] if frames[1] is not None else next(f for f in frames if f is not None)
            if not rec["subject"].get("present"):
                rec["saliency"] = saliency_sr(mid)
            rec["embed"] = clip_embed(clipm, clipp, mid)
        except Exception as e:
            rec.update(ok=False, reason=f"exception: {type(e).__name__}: {str(e)[:120]}")
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        if k % 100 == 0:
            print(f"  {k} processed, {round(time.time() - t0)}s", flush=True)
    fout.close()
    print(f"[detect] done, {k} new in {round(time.time() - t0)}s", flush=True)


if __name__ == "__main__":
    main()
