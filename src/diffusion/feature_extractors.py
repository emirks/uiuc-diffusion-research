"""feature_extractors — one extractor per feature namespace (store/FEATURES.md).

Each extractor loads its backbone ONCE (lazily, inside ``__init__``) and exposes
``extract(video_path) -> dict[str, np.ndarray]`` returning exactly the arrays the
namespace pins. ``REGISTRY`` maps namespace -> factory(device) so the CLI
(``scripts/store_features.py extract``) can build the right extractor by name.

Three namespaces REUSE the certified transition-eval code verbatim (import and
call, never reimplement):
  - ``dino_cls@dinov2b-r256``  -> ``transition_eval.features.DinoExtractor.extract``
  - ``cotracker3@g20-m384-v2`` -> ``transition_eval.motion.Tracker`` (grid 20,
    max side 384, query frames 0+mid, backward tracking — ``Tracker.track``)
  - ``lpips_t@alex-r256``      -> ``transition_eval.endpoints.temporal_lpips``
    via ``LpipsScorer("alex")``

Four competitor-lens namespaces are PORTED faithfully from
``misc/2026-08-13_baseline_metric_table/their_metrics/{clip_sim,videoprism_sim,
dynamic_degree}.py`` (per SCORERS.json). They are code-complete but were NOT
GPU-verified in P3a (no-GPU round); the operator verifies them at extraction
time (P4). Nothing here runs at import.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

DECODE_SHORT_SIDE = 256


def _load_frames(video: pathlib.Path | str, short_side: int = DECODE_SHORT_SIDE) -> np.ndarray:
    """The certified harness decoder (uint8 [T,H,W,3], shortest side resized)."""
    from diffusion.transition_eval.video_io import load_frames
    frames, _fps = load_frames(pathlib.Path(video), short_side=short_side)
    return frames


# --- verbatim transition-eval reuse ------------------------------------------
class DinoClsExtractor:
    NS = "dino_cls@dinov2b-r256"

    def __init__(self, device: str = "cuda"):
        from diffusion.transition_eval.features import DEFAULT_MODEL, DinoExtractor
        self._ext = DinoExtractor(model_name=DEFAULT_MODEL, device=device)

    def extract(self, video) -> dict[str, np.ndarray]:
        # DinoExtractor.extract: uint8 [T,H,W,3] -> L2-normalized CLS f32 [T,768]
        return {"feats": self._ext.extract(_load_frames(video))}


class CoTracker3Extractor:
    NS = "cotracker3@g20-m384-v2"

    def __init__(self, device: str = "cuda"):
        from diffusion.transition_eval.motion import Tracker
        self._trk = Tracker(device=device, grid_size=20, max_side=384)

    def extract(self, video) -> dict[str, np.ndarray]:
        # Tracker.track = the SAME protocol as cached_track: grid 20, max side
        # 384, query at frame 0 and the middle frame, backward tracking on the
        # mid query. Emits tracks f32 [T,N,2] and vis f32 [T,N].
        tracks, vis = self._trk.track(_load_frames(video))
        return {"tracks": tracks, "vis": vis}


class TemporalLpipsExtractor:
    NS = "lpips_t@alex-r256"

    def __init__(self, device: str = "cuda"):
        from diffusion.transition_eval.endpoints import LpipsScorer
        self._scorer = LpipsScorer(device=device, net="alex")

    def extract(self, video) -> dict[str, np.ndarray]:
        from diffusion.transition_eval.endpoints import temporal_lpips
        return {"d": temporal_lpips(_load_frames(video), self._scorer)}


# --- competitor lenses (ported from their_metrics/*, per SCORERS.json) --------
class ClipFrameExtractor:
    """Per-frame CLIP image embeddings (projected, L2-normalized), the
    ``clip_sim.ClipEmbedder`` port. ``clip_b32@r256`` -> ViT-B/32 (512-d),
    ``clip_l14@r224`` -> ViT-L/14 (768-d)."""

    def __init__(self, ns: str, model_id: str, device: str = "cuda"):
        from transformers import CLIPImageProcessor, CLIPModel
        self.NS = ns
        self.model_id = model_id
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.proc = CLIPImageProcessor.from_pretrained(model_id)
        self.device = device

    def extract(self, video, batch_size: int = 64) -> dict[str, np.ndarray]:
        frames = _load_frames(video)
        out = []
        with torch.no_grad():
            for i in range(0, len(frames), batch_size):
                batch = list(frames[i:i + batch_size])
                inp = self.proc(images=batch, return_tensors="pt").to(self.device)
                r = self.model.get_image_features(**inp)
                emb = r.pooler_output if hasattr(r, "pooler_output") else r
                emb = torch.nn.functional.normalize(emb.float(), dim=-1)
                out.append(emb.cpu().numpy())
        return {"feats": np.concatenate(out).astype(np.float32)}


class VideoPrismExtractor:
    """VideoPrism per-temporal-token embeddings (spatial mean-pool, L2), the
    ``videoprism_sim.VideoPrismEmbedder`` port -> ``feats`` f32 [16,768]."""

    NS = "videoprism@f16r288"
    MODEL_ID = "MHRDYN7/videoprism-base-f16r288"
    REVISION = "c5bb17adeb575aaf1d46b56c5b9dfa1b00465c80"

    def __init__(self, device: str = "cuda"):
        from transformers import AutoVideoProcessor, VideoPrismVisionModel
        self.device = device
        self.model = VideoPrismVisionModel.from_pretrained(
            self.MODEL_ID, revision=self.REVISION, torch_dtype=torch.float32).to(device).eval()
        self.proc = AutoVideoProcessor.from_pretrained(self.MODEL_ID, revision=self.REVISION)
        self.num_frames = int(getattr(self.model.config, "num_frames", 16))
        if hasattr(self.proc, "do_sample_frames"):
            self.proc.do_sample_frames = False

    @staticmethod
    def _subsample(frames: np.ndarray, n: int) -> np.ndarray:
        T = len(frames)
        if T == n:
            return frames
        idx = np.linspace(0, T - 1, n).round().astype(int)
        return frames[idx]

    def extract(self, video) -> dict[str, np.ndarray]:
        frames = _load_frames(video)
        f16 = self._subsample(frames, self.num_frames)
        with torch.no_grad():
            inp = self.proc(videos=[list(f16)], return_tensors="pt").to(self.device)
            lhs = self.model(**inp).last_hidden_state          # [1, nf*np, D]
            nf, D = self.num_frames, lhs.shape[-1]
            npatch = lhs.shape[1] // nf
            perframe = lhs.view(1, nf, npatch, D).mean(2)[0]    # [nf, D]
            perframe = torch.nn.functional.normalize(perframe.float(), dim=-1)
        return {"feats": perframe.cpu().numpy().astype(np.float32)}


class RaftMagExtractor:
    """Per-step mean RAFT flow magnitude (pixels), the ``dynamic_degree.RaftFlow``
    port -> ``mag`` f32 [T-1]. Frames forced to dims divisible by 8, longest side
    capped at 1024."""

    NS = "raft_mag@r256"
    MAX_SIZE = 1024

    def __init__(self, device: str = "cuda"):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        self.device = device
        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights).to(device).eval()
        self.transforms = weights.transforms()

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        import cv2
        h, w = frame.shape[:2]
        scale = min(self.MAX_SIZE / max(h, w), 1.0)
        nh = max(8, (int(round(h * scale)) // 8) * 8)
        nw = max(8, (int(round(w * scale)) // 8) * 8)
        if (nh, nw) == (h, w):
            return frame
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    def extract(self, video) -> dict[str, np.ndarray]:
        import torchvision.transforms.functional as TF
        frames = _load_frames(video)
        mags = []
        with torch.no_grad():
            for i in range(len(frames) - 1):
                f1 = TF.to_tensor(self._resize(frames[i])).unsqueeze(0)
                f2 = TF.to_tensor(self._resize(frames[i + 1])).unsqueeze(0)
                f1t, f2t = self.transforms(f1, f2)
                flow = self.model(f1t.to(self.device), f2t.to(self.device))[-1]
                mag = torch.linalg.vector_norm(flow, dim=1)   # [1,H,W]
                mags.append(float(mag.mean()))
        return {"mag": np.asarray(mags, dtype=np.float32)}


# --- namespace -> factory(device) -> extractor -------------------------------
REGISTRY: dict[str, "callable"] = {
    "dino_cls@dinov2b-r256": lambda device="cuda": DinoClsExtractor(device),
    "cotracker3@g20-m384-v2": lambda device="cuda": CoTracker3Extractor(device),
    "lpips_t@alex-r256": lambda device="cuda": TemporalLpipsExtractor(device),
    "clip_b32@r256": lambda device="cuda": ClipFrameExtractor(
        "clip_b32@r256", "openai/clip-vit-base-patch32", device),
    "clip_l14@r224": lambda device="cuda": ClipFrameExtractor(
        "clip_l14@r224", "openai/clip-vit-large-patch14", device),
    "videoprism@f16r288": lambda device="cuda": VideoPrismExtractor(device),
    "raft_mag@r256": lambda device="cuda": RaftMagExtractor(device),
}
