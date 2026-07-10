"""Per-frame DINOv2 embeddings with a content-keyed disk cache.

Cache keys hash (abspath, mtime, size, model, short_side) so a re-run — or a
requeued Slurm job — skips completed videos; synthetic frame arrays (lerp
controls) pass an explicit key instead.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np
import torch

from .video_io import load_frames

DEFAULT_MODEL = "facebook/dinov2-base"


class DinoExtractor:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        from transformers import AutoImageProcessor, AutoModel

        self.model_name = model_name
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()

    @torch.no_grad()
    def extract(self, frames: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """uint8 [T,H,W,3] -> L2-normalized CLS features float32 [T, D]."""
        feats = []
        for i in range(0, len(frames), batch_size):
            batch = list(frames[i:i + batch_size])
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
            cls = self.model(**inputs).last_hidden_state[:, 0].float()
            feats.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        return np.concatenate(feats)

    def free(self) -> None:
        del self.model
        torch.cuda.empty_cache()


def file_key(path: pathlib.Path, *parts: str) -> str:
    st = pathlib.Path(path).stat()
    raw = "|".join([str(pathlib.Path(path).resolve()), str(st.st_mtime_ns), str(st.st_size), *parts])
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def video_features(path: pathlib.Path, cache_dir: pathlib.Path, extractor: DinoExtractor,
                   short_side: int = 256) -> tuple[np.ndarray, float]:
    """Cached per-frame features for a video file. Returns (feats [T,D], fps)."""
    cache = pathlib.Path(cache_dir) / f"dino_{file_key(path, extractor.model_name, str(short_side))}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["feats"], float(z["fps"])
    frames, fps = load_frames(path, short_side=short_side)
    feats = extractor.extract(frames)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, feats=feats, fps=fps, src=str(path))
    return feats, fps


def array_features(frames: np.ndarray, key: str, cache_dir: pathlib.Path,
                   extractor: DinoExtractor) -> np.ndarray:
    """Cached features for an in-memory frame array (synthetic controls)."""
    cache = pathlib.Path(cache_dir) / f"dino_arr_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"
    if cache.exists():
        return np.load(cache)["feats"]
    feats = extractor.extract(frames)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, feats=feats, src=key)
    return feats
