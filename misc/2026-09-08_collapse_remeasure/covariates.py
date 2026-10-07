"""Per-endpoint-pair covariates for the collapse analysis (CPU).

For every endpoint id in the scoped grids (base_cond v2 / v3 / v3ed81 grid.jsonl) compute, on the pair of
conditioning frames (a = start anchor frame, b = end anchor frame):
  dino_dist   : 1 - cos( DINOv2-base CLS(a), CLS(b) )          semantic scene change (paper's feature space)
  clip_dist   : 1 - cos( CLIP ViT-B/32 image embeds )          cross-check
  pixel_gap_rel : ||b-a|| / mean(||a||,||b||) at 128x128 blurred (instrument units)
plus, when the endpoint's real transition clip exists (data/processed/transitions_std121/<class>/<ep>.mp4):
  gt_DR_med, gt_M, gt_motion_prefix, gt_motion_suffix : the REAL transition between the same endpoints,
  i.e. the per-pair ceiling of the confinement statistic.
Frame source priority: GT clip (frames 8 and T-8) > base_cond two-sided generation, seed 42 (frames 8 / T-8,
these are the conditioned frames after VAE round-trip) > none (one-sided-only endpoints: NaN).
Output: endpoint_covariates.csv
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from instrument import load_matrix, measure

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(REPO)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "16")))
cv2.setNumThreads(1)

STD = "data/processed/transitions_std121"
GRIDS = ["store/gens/005_base_cond/02_neutral__dai", "store/gens/005_base_cond/04_neutral_v3__dai",
         "store/gens/005_base_cond/05_neutral_v3ed81__dai"]
PREFIX, SUFFIX = 9, 8


def gt_path(ep):
    classes = sorted([d for d in os.listdir(STD) if os.path.isdir(f"{STD}/{d}")], key=len, reverse=True)
    for c in classes:
        if ep.startswith(c + "_") and ep[len(c) + 1:].isdigit():
            p = f"{STD}/{c}/{ep}.mp4"
            return p if os.path.exists(p) else None
    return None


def read_frames(path, idxs):
    cap = cv2.VideoCapture(path)
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = {}
    want = {(i if i >= 0 else T + i) for i in idxs}
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if k in want:
            out[k] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        k += 1
    cap.release()
    return T, out


def small(frame_rgb):
    f = cv2.resize(frame_rgb, (128, 128), interpolation=cv2.INTER_AREA)
    f = cv2.GaussianBlur(f, (0, 0), 1.0).astype(np.float32) / 255.0
    return f.reshape(-1)


def main():
    # ---- endpoints in scope ----
    eps = {}
    for gd in GRIDS:
        for l in open(f"{gd}/grid.jsonl"):
            d = json.loads(l)
            e = d["endpoint"]
            r = eps.setdefault(e, dict(endpoint=e, endpoint_source=d.get("endpoint_source"),
                                       endpoint_class=d.get("endpoint_class"), grids=set(), two_sided_rows=0))
            r["grids"].add(os.path.basename(gd))
            if d.get("sided") == "two":
                r["two_sided_rows"] += 1
                r.setdefault("gen_two", f"{gd}/videos/{d['item_id']}__s42.mp4")
    print(f"[cov] {len(eps)} unique endpoints", flush=True)

    # ---- models ----
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").eval()
    cproc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()

    @torch.no_grad()
    def embed(frames_rgb):
        ims = [Image.fromarray(f) for f in frames_rgb]
        dz = dino(**dproc(images=ims, return_tensors="pt")).last_hidden_state[:, 0]
        # transformers 5: get_image_features returns an output object; go through the tower + projection explicitly
        vis = clip.vision_model(pixel_values=cproc(images=ims, return_tensors="pt")["pixel_values"])
        cz = clip.visual_projection(vis.pooler_output)
        return torch.nn.functional.normalize(dz, dim=-1).numpy(), torch.nn.functional.normalize(cz, dim=-1).numpy()

    rows = []
    for i, (e, r) in enumerate(sorted(eps.items())):
        rec = dict(endpoint=e, endpoint_source=r["endpoint_source"], endpoint_class=r["endpoint_class"],
                   grids="|".join(sorted(r["grids"])), two_sided_rows=r["two_sided_rows"], frames_source="none",
                   dino_dist=np.nan, clip_dist=np.nan, pixel_gap_rel=np.nan,
                   gt_DR_med=np.nan, gt_M=np.nan, gt_motion_prefix=np.nan, gt_motion_suffix=np.nan)
        gp = gt_path(e)
        a = b = None
        if gp:
            T, fr = read_frames(gp, [PREFIX - 1, -SUFFIX])
            if (PREFIX - 1) in fr and (T - SUFFIX) in fr:
                a, b = fr[PREFIX - 1], fr[T - SUFFIX]
                rec["frames_source"] = "gt_clip"
                Mg = load_matrix(gp)
                if Mg is not None:
                    g, _, _ = measure(Mg, PREFIX, SUFFIX, True)
                    rec.update(gt_DR_med=g.get("DR_med"), gt_M=g.get("M"),
                               gt_motion_prefix=g.get("motion_prefix"), gt_motion_suffix=g.get("motion_suffix"))
        if a is None and r.get("gen_two") and os.path.exists(r["gen_two"]):
            T, fr = read_frames(r["gen_two"], [PREFIX - 1, -SUFFIX])
            if (PREFIX - 1) in fr and (T - SUFFIX) in fr:
                a, b = fr[PREFIX - 1], fr[T - SUFFIX]
                rec["frames_source"] = "gen_cond"
        if a is not None:
            dz, cz = embed([a, b])
            rec["dino_dist"] = float(1.0 - (dz[0] * dz[1]).sum())
            rec["clip_dist"] = float(1.0 - (cz[0] * cz[1]).sum())
            sa, sb = small(a), small(b)
            rec["pixel_gap_rel"] = float(np.linalg.norm(sb - sa) / (0.5 * (np.linalg.norm(sa) + np.linalg.norm(sb)) + 1e-8))
        rows.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  ..{i+1}/{len(eps)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{HERE}/endpoint_covariates.csv", index=False)
    print(df["frames_source"].value_counts().to_string())
    print(df[["dino_dist", "clip_dist", "pixel_gap_rel", "gt_DR_med", "gt_M"]].describe().to_string())


if __name__ == "__main__":
    main()
