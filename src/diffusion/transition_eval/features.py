"""Per-frame DINOv2 embeddings with a content-keyed disk cache.

Cache keys hash (abspath, mtime, size, model, short_side) so a re-run — or a
requeued Slurm job — skips completed videos; synthetic frame arrays (lerp
controls) pass an explicit key instead.
"""

from __future__ import annotations

import hashlib
import os
import zipfile
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


def savez_atomic(path: pathlib.Path, **arrays) -> None:
    """np.savez_compressed through a private temp file + os.replace: a concurrent reader never sees a
    half-written archive (grid v3 amendment 2026-09-07 — 16 score shards extracting the same NEW corpus
    clip raced on the shared cache and readers hit `EOFError: No data left in file`)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def load_npz_or_none(path: pathlib.Path):
    """np.load that treats a truncated/corrupt archive as a MISS (unlinks it) instead of a crash."""
    path = pathlib.Path(path)
    if not path.exists():
        return None
    try:
        z = np.load(path)
        _ = z.files
        return z
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as e:  # noqa: F841 — any unreadable archive
        try:
            path.unlink()
        except OSError:
            pass
        return None


def video_features(path: pathlib.Path, cache_dir: pathlib.Path, extractor: DinoExtractor,
                   short_side: int = 256) -> tuple[np.ndarray, float]:
    """Cached per-frame features for a video file. Returns (feats [T,D], fps)."""
    cache = pathlib.Path(cache_dir) / f"dino_{file_key(path, extractor.model_name, str(short_side))}.npz"
    z = load_npz_or_none(cache)
    if z is not None:
        return z["feats"], float(z["fps"])
    frames, fps = load_frames(path, short_side=short_side)
    feats = extractor.extract(frames)
    savez_atomic(cache, feats=feats, fps=fps, src=str(path))
    return feats, fps


def feature_cache_path(key: str, cache_dir: pathlib.Path) -> pathlib.Path:
    """Cache location for array_features under this key."""
    return pathlib.Path(cache_dir) / f"dino_arr_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"


def array_features(frames: np.ndarray | None, key: str, cache_dir: pathlib.Path,
                   extractor: DinoExtractor) -> np.ndarray:
    """Cached features for an in-memory frame array (synthetic controls)."""
    cache = feature_cache_path(key, cache_dir)
    z = load_npz_or_none(cache)
    if z is not None:
        return z["feats"]
    if frames is None:
        raise RuntimeError(f"feature cache miss for {key} but no frames were decoded")
    feats = extractor.extract(frames)
    savez_atomic(cache, feats=feats, src=key)
    return feats
