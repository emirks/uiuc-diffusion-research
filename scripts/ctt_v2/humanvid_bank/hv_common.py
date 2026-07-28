"""Shared helpers for the HumanVid endpoint-bank screening pipeline (ctt_v2).

Contract parity: thresholds, detectors and the std121 output contract are IDENTICAL to
the blessed synth_endpoints bank build (data/processed/synth_endpoints/_work/{detect,build}.py,
read-only provenance). HumanVid-specific deltas, by design:
  * clips are long (median 13.8 s) -> we cut a CENTER window of ~121/24 s at source fps and
    resample 121 frames inside it (tempo ~=native), instead of whole-clip linspace.
  * EAST raw boxes are stored in the detect pass so build can recompute high-confidence
    text stats directly (merges the old east_repass step).
"""
import os
import subprocess

import cv2
import numpy as np

REPO = "/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research"
RAW_HV = os.path.join(REPO, "data/raw/humanvid_real")
HV_MANIFEST = os.path.join(REPO, "data/manifests/humanvid_real/clips.jsonl.gz")
DSX_WORK = os.path.join(REPO, "data/processed/synth_endpoints/_work")  # read-only (weights)
OLD_BANK = os.path.join(REPO, "data/processed/synth_endpoints")        # read-only (dedup ref)

# Bank lives in the MAIN tree's data/ (gitignored), next to synth_endpoints and the
# raw humanvid_real — data never lives inside a worktree.
BANK = os.path.join(REPO, "data/processed/humanvid_bank")
WORK = os.path.join(BANK, "_work")
CLIPS = os.path.join(BANK, "clips")
FFMPEG = "/u/emirkisa/.local/bin/ffmpeg"

W, H, F, FPS = 480, 640, 121, 24

# ---- thresholds: verbatim from the blessed bank build ----
EAST_CONF = 0.75
TEXT_COVER_REJECT = 0.030
TEXT_CAP_REJECT = 0.015
TEXT_CORNER_REJECT = 0.010
SUBJ_MIN_AREA = 0.05
SAL_COMPACT_MAX = 0.12
FIT_TOL = 1.75
CUT_THRESH = 0.20
PHASH_THRESH = 12
CLIP_COS_DUP = 0.985
# ---- tightened policy (bank_tightened parity), applied pre-encode ----
TIGHT_AREA = 0.15
TIGHT_SCORE = 0.7


def center_window(n_frames: int, src_fps: float):
    """Inclusive [w0, w1] source-frame window covering ~F/FPS seconds at src fps."""
    need = int(round(F / float(FPS) * (src_fps or FPS)))
    need = min(max(need, F), n_frames)
    w0 = (n_frames - need) // 2
    return w0, w0 + need - 1


def window_indices(w0: int, w1: int):
    return np.round(np.linspace(w0, w1, F)).astype(int)


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


def east_detect_raw(net, bgr, conf=0.5, inp=320):
    """EAST forward -> [[x0,y0,x1,y1,score],...] in source pixels (low conf floor;
    the build pass re-filters at EAST_CONF)."""
    H0, W0 = bgr.shape[:2]
    rW = W0 / float(inp); rH = H0 / float(inp)
    blob = cv2.dnn.blobFromImage(bgr, 1.0, (inp, inp), (123.68, 116.78, 103.94), True, False)
    net.setInput(blob)
    scores, geom = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
    s0 = scores[0, 0]
    mask = s0 >= conf
    ys, xs = np.nonzero(mask)
    rects, confs = [], []
    if len(ys):
        d0 = geom[0, 0][mask]; d1 = geom[0, 1][mask]; d2 = geom[0, 2][mask]
        d3 = geom[0, 3][mask]; ang = geom[0, 4][mask]
        offX = xs * 4.0; offY = ys * 4.0
        cos = np.cos(ang); sin = np.sin(ang)
        hh = d0 + d2; ww = d1 + d3
        eX = offX + cos * d1 + sin * d2
        eY = offY - sin * d1 + cos * d2
        sX = eX - ww; sY = eY - hh
        rects = np.stack([sX, sY, ww, hh], 1).astype(int).tolist()
        confs = s0[mask].astype(float).tolist()
    boxes = []
    if rects:
        idxs = cv2.dnn.NMSBoxes(rects, confs, conf, 0.4)
        for i in np.array(idxs).flatten():
            x, y, ww, hh = rects[i]
            sX, sY, eX, eY = x * rW, y * rH, (x + ww) * rW, (y + hh) * rH
            sX = max(0, min(sX, W0)); eX = max(0, min(eX, W0))
            sY = max(0, min(sY, H0)); eY = max(0, min(eY, H0))
            boxes.append([round(sX, 1), round(sY, 1), round(eX, 1), round(eY, 1), round(confs[i], 3)])
    return boxes


def text_stats_from_boxes(boxes_by_frame: dict, W0: int, H0: int, conf=EAST_CONF):
    """High-confidence text coverage stats + watermark persistence (build-pass filter)."""
    area = float(W0 * H0)
    cover_max = cap_max = corner_max = 0.0
    centers_by = {}
    for nm, boxes in boxes_by_frame.items():
        cov = capc = corc = 0.0; centers = []
        for x0, y0, x1, y1, sc in boxes:
            if sc < conf:
                continue
            a = max(0.0, x1 - x0) * max(0.0, y1 - y0); cov += a
            cx = (x0 + x1) / 2 / W0; cy = (y0 + y1) / 2 / H0; centers.append((cx, cy))
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
            if allc[i][0] != allc[j][0] and \
               abs(allc[i][1][0] - allc[j][1][0]) < 0.05 and abs(allc[i][1][1] - allc[j][1][1]) < 0.05:
                wm = True
    return {"text_cover_max": round(cover_max, 5), "text_cap_max": round(cap_max, 5),
            "text_corner_max": round(corner_max, 5), "watermark_persist": wm}


def encode_std(crops, out_path):
    tmp = out_path + ".tmp.mp4"
    p = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)
    for c in crops:
        p.stdin.write(np.ascontiguousarray(c[:, :, ::-1]).tobytes())
    p.stdin.close()
    assert p.wait() == 0, "ffmpeg encode failed"
    os.rename(tmp, out_path)
