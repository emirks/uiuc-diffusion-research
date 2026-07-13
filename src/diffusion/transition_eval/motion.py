"""M2 — Motion Fidelity (Yatim et al., Space-Time Diffusion Features).

Tracklets from CoTracker3 (grid queries), per-step velocity directions,
normalized cross-correlation between all tracklet pairs across the two videos,
bidirectional mean-of-max. Robust to spatial misalignment between unpaired
videos by construction; velocities are resampled to a fixed number of steps so
122- and 242-frame clips compare.

`motion_fidelity()` is pure numpy (unit-testable); only `Tracker` needs GPU.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np
import torch


def track_cache_path(final_key: str, cache_dir: pathlib.Path) -> pathlib.Path:
    """Cache location for cached_track under the fully-composed key
    (i.e. including the ':tracks:<CACHE_TAG>' suffix)."""
    return pathlib.Path(cache_dir) / f"tracks_{hashlib.sha1(final_key.encode()).hexdigest()[:16]}.npz"


class Tracker:
    """CoTracker3 offline wrapper with a per-video disk cache (normalized coords).

    Grid points are queried at frame 0 AND the middle frame (with backward
    tracking): in transition videos the effect medium doesn't exist at frame 0,
    so a frame-0-only grid samples scene content that the effect occludes and
    misses the effect's own motion — the signal this metric is for."""

    CACHE_TAG = "v2"  # bump when tracking protocol changes

    def __init__(self, device: str = "cuda", grid_size: int = 20, max_side: int = 384):
        self.device = device
        self.grid_size = grid_size
        self.max_side = max_side
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device).eval()

    @torch.no_grad()
    def track(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """uint8 [T,H,W,3] -> (tracks [T,N,2] in [0,1] coords, visibility [T,N])."""
        t, h, w = frames.shape[:3]
        s = self.max_side / max(h, w)
        video = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        if s < 1.0:
            video = torch.nn.functional.interpolate(
                video, size=(round(h * s), round(w * s)), mode="bilinear", align_corners=False)
        _, _, rh, rw = video.shape
        video = video[None].to(self.device)
        all_tracks, all_vis = [], []
        for qf in (0, t // 2):
            tracks, vis = self.model(video, grid_size=self.grid_size,
                                     grid_query_frame=qf, backward_tracking=qf > 0)
            all_tracks.append(tracks[0].cpu().numpy())
            all_vis.append(vis[0].float().cpu().numpy())
        tracks = np.concatenate(all_tracks, axis=1) / np.array([rw, rh], dtype=np.float32)
        return tracks.astype(np.float32), np.concatenate(all_vis, axis=1).astype(np.float32)

    def cached_track(self, frames: np.ndarray | None, key: str, cache_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
        key = f"{key}:{self.CACHE_TAG}"
        cache = track_cache_path(key, cache_dir)
        if cache.exists():
            z = np.load(cache)
            return z["tracks"], z["vis"]
        if frames is None:
            raise RuntimeError(f"track cache miss for {key} but no frames were decoded")
        tracks, vis = self.track(frames)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, tracks=tracks, vis=vis, src=key)
        return tracks, vis

    def free(self) -> None:
        del self.model
        torch.cuda.empty_cache()


def _velocity_directions(tracks: np.ndarray, vis: np.ndarray, n_steps: int,
                         min_vis: float, speed_floor: float,
                         min_moving_frac: float) -> np.ndarray:
    """[T,N,2] tracks -> [M, n_steps, 2] unit velocity directions.

    Velocities are scaled to per-NORMALIZED-duration units (frame-fraction per
    full video) so one speed floor serves 121- and 242-frame clips alike.
    Steps where a point is occluded contribute zero direction instead of
    hallucinated-track noise — in transition videos most points get engulfed
    by the effect, so a whole-tracklet visibility cut would empty the set."""
    keep = vis.mean(axis=0) >= min_vis  # coarse gate only
    tr, vs = tracks[:, keep, :], vis[:, keep]
    if tr.shape[1] == 0:
        return np.zeros((0, n_steps, 2), dtype=np.float32)
    T = len(tr)
    # box-smooth tracks before differencing: sub-pixel tracking jitter on
    # static points otherwise passes the speed floor in per-duration units
    win = max(3, T // 32)
    kern = np.ones(win, dtype=np.float32) / win
    pad = np.pad(tr, ((win // 2, win - 1 - win // 2), (0, 0), (0, 0)), mode="edge")
    tr = np.apply_along_axis(lambda x: np.convolve(x, kern, mode="valid"), 0, pad)
    v = np.diff(tr, axis=0) * (T - 1)          # [T-1, N, 2], per unit duration
    vv = 0.5 * (vs[:-1] + vs[1:])              # velocity-step visibility
    src = np.linspace(0.0, 1.0, v.shape[0])
    dst = np.linspace(0.0, 1.0, n_steps)
    vr = np.stack([np.stack([np.interp(dst, src, v[:, i, d]) for d in (0, 1)], axis=-1)
                   for i in range(v.shape[1])])                       # [N, n_steps, 2]
    vis_r = np.stack([np.interp(dst, src, vv[:, i]) for i in range(vv.shape[1])])
    speed = np.linalg.norm(vr, axis=-1)
    active = (speed > speed_floor) & (vis_r > 0.5)
    keep2 = active.mean(axis=1) >= min_moving_frac
    vr, speed, active = vr[keep2], speed[keep2], active[keep2]
    dirs = np.where(active[..., None], vr / (speed[..., None] + 1e-9), 0.0)
    return dirs.astype(np.float32)


def motion_fidelity(tracks1: np.ndarray, vis1: np.ndarray,
                    tracks2: np.ndarray, vis2: np.ndarray,
                    n_steps: int = 64, min_vis: float = 0.2,
                    speed_floor: float = 0.1, min_moving_frac: float = 0.05) -> float:
    """Bidirectional mean over tracklets of the max velocity-direction
    correlation with any tracklet of the other video. NaN if either video has
    no moving tracklets."""
    D1 = _velocity_directions(tracks1, vis1, n_steps, min_vis, speed_floor, min_moving_frac)
    D2 = _velocity_directions(tracks2, vis2, n_steps, min_vis, speed_floor, min_moving_frac)
    if len(D1) == 0 or len(D2) == 0:
        return float("nan")
    C = np.einsum("itc,jtc->ij", D1, D2) / n_steps
    return float(0.5 * (C.max(axis=1).mean() + C.max(axis=0).mean()))
