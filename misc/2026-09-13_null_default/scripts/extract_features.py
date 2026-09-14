"""A3 / Phase B - DINO-CLS, TRANS (44-ch), and VAE latents for every manifest clip and its
landmarks -> $LAB/cache/null_default/features/<clip_id>[__LERP|__CUT50|__FREEZE|__LATLERP].npz.

Runs on GPU (Phase B, --device cuda) and on CPU for the smoke (--device cpu, DINO+TRANS only;
VAE needs the GPU). Resumable: existing npz files are skipped; --shard i/n and --only-missing.

npz keys: dino (T,768) fp16 L2-normed CLS ; vae (T_lat,C,H,W) fp16 ; trans (T_lat,H,W,44) fp32 ;
plus a_idx,b_idx,T. Pixel landmarks (LERP/CUT50/FREEZE) carry dino+trans(+vae); LATLERP is
vae-only (latent interpolation of the two anchor latents).

Decode = cv2 single-threaded (campaign convention); S-SWEEP frames are downscaled to 768x512
before every extractor. VAE uses encode_conditioning.load_vae + encode (the repo's encoder);
the frame tensor is built from the same cv2 frames as DINO/TRANS (so a clip's main and landmark
latents share one decoder), rather than preprocess()'s PyAV reader which cannot do the sweep's
193 frames, the downscale, or the synthetic landmark frames.

    python misc/2026-09-13_null_default/scripts/extract_features.py --device cpu \
        --extractors dino,trans --clips <id1>,<id2> --verbose        # the A4 smoke
    python .../extract_features.py --device cuda --shard 0/4 --only-missing   # Phase B shard
"""
from __future__ import annotations

import argparse
import os
import sys

# bounded BLAS/OMP threads: the login node's per-process thread limit is small (CPU smoke), and
# on the GPU node the extractors run on-device anyway.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

ARMA = C.REPO / "misc" / "2026-08-24_flow_signal_conditioning" / "armA"
sys.path.insert(0, str(ARMA))
EVAL_LADDER = C.REPO / "eval_ladder"
sys.path.insert(0, str(EVAL_LADDER))

VAE_MODEL = C.LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"
DINO_MODEL = "facebook/dinov2-base"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ============================================================ DINO CLS (SPEC 2.2)
class DinoCLS:
    def __init__(self, device, dtype):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.model = AutoModel.from_pretrained(DINO_MODEL, torch_dtype=dtype).to(device).eval()

    def embed(self, frames, batch=32):
        """(T,H,W,3) in [0,255] -> (T,768) float32 L2-normed CLS. Whole-frame resize to 224, no crop."""
        import cv2
        torch = self.torch
        T = len(frames)
        pre = np.empty((T, 224, 224, 3), np.float32)
        for i, f in enumerate(frames):
            g = cv2.resize(np.asarray(f, np.float32), (224, 224), interpolation=cv2.INTER_AREA) / 255.0
            pre[i] = (g - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(pre).permute(0, 3, 1, 2).contiguous()
        outs = []
        with torch.no_grad():
            for s in range(0, T, batch):
                xb = x[s:s + batch].to(self.device, self.dtype)
                cls = self.model(pixel_values=xb).last_hidden_state[:, 0].float()
                cls = torch.nn.functional.normalize(cls, dim=-1)
                outs.append(cls.cpu().numpy())
        return np.concatenate(outs, 0).astype(np.float32)


# ============================================================ TRANS (armA compute_clip)
class Trans:
    def __init__(self, device, dtype_str):
        import armA_extract as A
        self.A = A
        self.device = device
        z = np.load(A.PCA_PATH)
        self.pca_mean, self.pca_comp = z["mean"], z["comp"]
        self.dino = A.Dino(device, dtype=dtype_str)

    def field(self, frames):
        """(T,H,W,3) in [0,255] -> (T_lat,H,W,44) float32."""
        A = self.A
        T = frames.shape[0]
        T_lat, centers, B, uniq = A.select_frames(T)
        H = int(round(frames.shape[1] / 32))
        W = int(round(frames.shape[2] / 32))
        fr = frames
        if (frames.shape[1], frames.shape[2]) != (32 * H, 32 * W):
            fr, _ = A._fit_to_grid(np.asarray(frames), 32 * H, 32 * W)
        Hp, Wp = 2 * H, 2 * W
        sub = np.asarray(fr)[uniq]
        P_raw, rs = self.dino.raw(sub, Hp, Wp)
        A._resized_rgb_cache = {i: rs[i] for i in range(len(uniq))}
        field = A.compute_clip(P_raw, centers, B, uniq, self.pca_mean, self.pca_comp, H, W, self.device)
        return field.astype(np.float32)


# ============================================================ VAE (repo encoder)
class Vae:
    def __init__(self, device, dtype):
        import encode_conditioning as ec
        import torch
        self.ec = ec
        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.vae = ec.load_vae(str(VAE_MODEL), device=device, dtype=dtype)

    def latents(self, frames):
        """(T,H,W,3) in [0,255] -> (T_lat,C,H,W) float32."""
        torch = self.torch
        v = torch.from_numpy(np.asarray(frames, np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()
        lat = self.ec.encode(v, self.vae, self.device, self.dtype)   # (1,C,T_lat,H,W)
        return lat[0].permute(1, 0, 2, 3).float().cpu().numpy()


# ============================================================ per-clip driver
def save_npz(path, a_idx, b_idx, T, dino=None, vae=None, trans=None):
    d = dict(a_idx=int(a_idx), b_idx=int(b_idx), T=int(T))
    if dino is not None:
        d["dino"] = dino.astype(np.float16)
    if vae is not None:
        d["vae"] = vae.astype(np.float16)
    if trans is not None:
        d["trans"] = trans.astype(np.float32)
    np.savez_compressed(path, **d)


def process(row, models, ext, device, only_missing, verbose):
    stratum, group = row["stratum"], row["group"]
    a_idx, b_idx, T = int(row["a_idx"]), int(row["b_idx"]), int(row["T"])
    ds = C.sweep_downscale_wh() if stratum == "S-SWEEP" else None
    C.FEATURES.mkdir(parents=True, exist_ok=True)
    made = []

    def run_extractors(frames):
        out = {}
        if "dino" in ext:
            out["dino"] = models["dino"].embed(frames)
        if "trans" in ext:
            out["trans"] = models["trans"].field(frames)
        if "vae" in ext:
            out["vae"] = models["vae"].latents(frames)
        return out

    def emit(clip_id, res, need_keys):
        p = C.FEATURES / f"{clip_id}.npz"
        if only_missing and p.exists():
            return
        save_npz(p, a_idx, b_idx, T, dino=res.get("dino"), vae=res.get("vae"), trans=res.get("trans"))
        made.append((clip_id, {k: tuple(v.shape) for k, v in res.items()}))

    # ---- main clip
    main_p = C.FEATURES / f"{row['clip_id']}.npz"
    frames = None
    need_main = not (only_missing and main_p.exists())
    if need_main:
        frames = C.decode_rgb(C.REPO / row["path"], downscale_wh=ds)
        res_main = run_extractors(frames)
        emit(row["clip_id"], res_main, ext)
    else:
        res_main = None

    # ---- landmarks
    if C.is_landmark_source(stratum, group):
        if frames is None:
            frames = C.decode_rgb(C.REPO / row["path"], downscale_wh=ds)
        lms = C.build_pixel_landmarks(frames, a_idx, b_idx)
        for kind, clip in lms.items():
            res = run_extractors(clip)
            emit(f"{row['clip_id']}__{kind}", res, ext)
        # LATLERP (vae only): interpolate the two anchor latents of the main clip
        if "vae" in ext:
            latp = C.FEATURES / f"{row['clip_id']}__LATLERP.npz"
            if not (only_missing and latp.exists()):
                main_vae = None
                if res_main and "vae" in res_main:
                    main_vae = res_main["vae"]
                elif main_p.exists():
                    z = np.load(main_p)
                    if "vae" in z.files:
                        main_vae = z["vae"].astype(np.float32)
                if main_vae is not None:
                    lat = C.build_latlerp_latents(main_vae, a_idx, b_idx)
                    save_npz(latp, a_idx, b_idx, T, vae=lat)
                    made.append((f"{row['clip_id']}__LATLERP", {"vae": tuple(lat.shape)}))
    if verbose:
        for cid, shapes in made:
            print(f"    [{cid}] {shapes}", flush=True)
    return made


# ============================================================ main
def build_models(ext, device, gpu):
    import torch
    dtype = torch.float16 if gpu else torch.float32
    models = {}
    if "dino" in ext:
        models["dino"] = DinoCLS(device, dtype)
    if "trans" in ext:
        models["trans"] = Trans(device, "float16" if gpu else "float32")
    if "vae" in ext:
        models["vae"] = Vae(device, torch.bfloat16 if gpu else torch.float32)
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only-missing", action="store_true")
    ap.add_argument("--extractors", default="dino,trans,vae")
    ap.add_argument("--clips", default="", help="comma-separated clip_ids (smoke)")
    ap.add_argument("--limit-clips", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(C.LAB / "cache/huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import cv2
    cv2.setNumThreads(1)

    ext = set(x.strip() for x in args.extractors.split(",") if x.strip())
    gpu = args.device.startswith("cuda")
    man = C.load_manifest()
    if args.clips:
        wanted = set(args.clips.split(","))
        man = man[man.clip_id.isin(wanted)]
    si, sn = (int(x) for x in args.shard.split("/"))
    man = man.iloc[si::sn]
    if args.limit_clips:
        man = man.iloc[: args.limit_clips]

    print(f"[extract] device={args.device} shard={args.shard} extractors={sorted(ext)} "
          f"clips={len(man)}", flush=True)
    models = build_models(ext, args.device, gpu)

    import time
    t0 = time.time()
    n_files = 0
    fail_log = C.CAMP / "logs" / "extract_failures.txt"
    n_fail = 0
    for i, (_, row) in enumerate(man.iterrows()):
        try:
            made = process(row, models, ext, args.device, args.only_missing, args.verbose)
        except RuntimeError as e:          # persistent decode failure: record, continue the shard
            n_fail += 1
            msg = f"{row['clip_id']}\t{row['path']}\t{e}"
            print(f"[extract] FAILED {msg}", flush=True)
            fail_log.parent.mkdir(parents=True, exist_ok=True)
            with open(fail_log, "a") as fh:
                fh.write(msg + "\n")
            continue
        n_files += len(made)
        if (i + 1) % 25 == 0:
            print(f"  ..{i+1}/{len(man)} clips, {n_files} npz, {time.time()-t0:.0f}s", flush=True)
    print(f"[extract] done {len(man)} clips -> {n_files} npz in {time.time()-t0:.0f}s "
          f"({C.FEATURES}); failed clips: {n_fail}", flush=True)
    if n_fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
