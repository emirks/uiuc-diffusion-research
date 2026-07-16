"""GPU pass — DINOv2-large PATCH-token Gram (texture / second-moment) per clip.

Fable's B channel: global CLS is an attention-weighted MEAN of tokens (first moment,
"what content"); it discards the SECOND moment — how local features co-occur = texture.
The unresolved m1a misses are texture confusions (shadow/smoke/gas, wireframe/polygon)
with near-identical global CLS. This extracts the per-clip uncentered second moment of
L2-normalized patch tokens over the SAME base sided-core frames (masks reused verbatim).

Per clip i: Y_i = all L2-normed patch tokens over core frames [M_i, 1024];
Gram G_i = (1/M_i) Y_iᵀ Y_i; Frobenius-normalized Ĝ_i saved flattened.
D_tex[i,j] = 1 - <Ĝ_i,Ĝ_j>_F is then a CPU matmul.

Writes ONLY to $WB_CACHE/search. Certified shared cache untouched.
"""

from __future__ import annotations

import time

import numpy as np

from .. import paths

LARGE_MODEL = "facebook/dinov2-large"
OUT = paths.WB_CACHE / "search" / "texture_grams.npz"
SUBSTRATE = paths.WB_CACHE / "search" / "substrate.npz"


def log(m: str) -> None:
    print(f"[tex {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    import torch
    from transformers import AutoImageProcessor, AutoModel
    from ...video_io import load_frames

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("REFUSING: GPU work on CPU. Submit through Slurm.")
        return 1

    corpus = paths.load_corpus()
    keys = paths.corpus_keys(corpus)
    sub = np.load(SUBSTRATE, allow_pickle=True)
    assert list(sub["keys"]) == list(keys), "substrate key order mismatch"
    masks = sub["mask_sided"]                              # [223, 121] bool, base-space

    proc = AutoImageProcessor.from_pretrained(LARGE_MODEL)
    model = AutoModel.from_pretrained(LARGE_MODEL, torch_dtype=torch.float16).to(device).eval()
    log(f"{LARGE_MODEL} on {device}; texture Grams for {len(keys)} clips")

    grams, t0 = [], time.time()
    P_seen = set()
    for idx, key in enumerate(keys):
        frames, _ = load_frames(paths.clip_path(key), short_side=paths.FEATURE_SHORT_SIDE)
        core_idx = np.flatnonzero(masks[idx])
        if len(core_idx) == 0:
            core_idx = np.arange(len(frames))             # coverage guard (never fires)
        G = torch.zeros((1024, 1024), dtype=torch.float32, device=device)
        M = 0
        with torch.no_grad():
            for s in range(0, len(core_idx), 16):
                batch = [frames[t] for t in core_idx[s:s + 16]]
                inp = proc(images=batch, return_tensors="pt").to(device)
                inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
                patch = model(**inp).last_hidden_state[:, 1:].float()   # [b, P, 1024]
                P_seen.add(patch.shape[1])
                patch = torch.nn.functional.normalize(patch, dim=-1)    # L2 per token
                Z = patch.reshape(-1, 1024)                             # [b*P, 1024]
                G += Z.T @ Z
                M += Z.shape[0]
        G = (G / max(M, 1)).cpu().numpy()
        G = G / (np.linalg.norm(G) + 1e-12)               # Frobenius-normalize
        grams.append(G.astype(np.float32).ravel())        # flatten [1048576]
        if (idx + 1) % 25 == 0:
            log(f"  {idx + 1}/{len(keys)} ({time.time()-t0:.0f}s)  P={sorted(P_seen)}")

    del model
    torch.cuda.empty_cache()
    G = np.stack(grams)                                    # [223, 1048576]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, keys=np.array(keys), grams=G, model=LARGE_MODEL, patches=sorted(P_seen))
    log(f"WROTE {OUT}  grams {G.shape}  patches/frame {sorted(P_seen)} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
