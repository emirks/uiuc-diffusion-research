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


class RaftFlowWinExtractor(RaftMagExtractor):
    """Dense RAFT flow FIELDS for the given windows only (endpoint-motion metrics):
    ``flow_start`` f16 [8,H,W,2] = steps of frames 0..8 (the 9-frame given start clip /
    the output's pinned start window); ``flow_end`` f16 [7,H,W,2] = steps of the LAST 8
    frames (end9 frames 1..8 / the output's pinned end window). Same decode (short side
    256), resize (dims divisible by 8) and RAFT-large pin as ``raft_mag@r256``; flow in
    pixels of the resized frame, stored (y,x)-> [..., (dx, dy)]."""

    NS = "raft_flow_win@r256"
    START_FRAMES = 9
    END_FRAMES = 8

    def _flow(self, f_a: np.ndarray, f_b: np.ndarray) -> np.ndarray:
        import torchvision.transforms.functional as TF
        f1 = TF.to_tensor(self._resize(f_a)).unsqueeze(0)
        f2 = TF.to_tensor(self._resize(f_b)).unsqueeze(0)
        f1t, f2t = self.transforms(f1, f2)
        flow = self.model(f1t.to(self.device), f2t.to(self.device))[-1]   # [1,2,H,W]
        return flow[0].permute(1, 2, 0).float().cpu().numpy().astype(np.float16)

    def extract(self, video) -> dict[str, np.ndarray]:
        frames = _load_frames(video)
        T = len(frames)
        if T < 2:
            raise ValueError(f"{video}: {T} frames < 2")
        # a given clip shorter than the window (the 16-fps 6/4-frame VACE endpoints, 2026-09-20) IS its window:
        # flow_start = all its steps, flow_end = the steps of its last min(8, T) frames; T >= 9 unchanged
        n_start, n_end = min(self.START_FRAMES, T), min(self.END_FRAMES, T)
        with torch.no_grad():
            start = np.stack([self._flow(frames[i], frames[i + 1])
                              for i in range(n_start - 1)])
            e0 = T - n_end
            end = np.stack([self._flow(frames[i], frames[i + 1])
                            for i in range(e0, T - 1)])
        return {"flow_start": start, "flow_end": end}


class FlowU32Extractor(RaftMagExtractor):
    """Whole-video optical-flow signature (metrics v5, eval 049): RAFT-large flow over T=32
    uniform steps (33 uniformly sampled frames), each step's dense flow field reduced to a 24x32
    grid of the frame-diagonal fraction. SAME decode (short side 256), ``_resize`` (dims /8),
    transforms and RAFT-large pin as ``raft_mag@r256``. ``flow`` f16 [32,24,32,2] (last axis
    ``(dx, dy)`` like ``raft_flow_win``; pixels of the resized frame divided by that frame's
    diagonal, so a fraction of the frame diagonal per step); ``idx`` i32 [33] (the sampled frame
    indices). idx = np.round(np.linspace(0, T-1, 33)); T >= 33 for every video of the gridv3
    population, so idx is strictly increasing there. If T < 33 the linspace round REPEATS indices
    (a slower rate); the repeats are recorded in the stored ``idx`` array (the sidecar schema is
    fixed by FeatureStore.put). ~100 KB/video."""

    NS = "flow_u32@raft-r256-g24x32"
    N_STEPS = 32
    GRID = (24, 32)

    def extract(self, video) -> dict[str, np.ndarray]:
        import torch.nn.functional as F
        import torchvision.transforms.functional as TF
        frames = _load_frames(video)
        T = len(frames)
        idx = np.round(np.linspace(0, T - 1, self.N_STEPS + 1)).astype(np.int32)   # [33]
        steps = []
        with torch.no_grad():
            for k in range(self.N_STEPS):
                f1 = TF.to_tensor(self._resize(frames[idx[k]])).unsqueeze(0)
                f2 = TF.to_tensor(self._resize(frames[idx[k + 1]])).unsqueeze(0)
                f1t, f2t = self.transforms(f1, f2)
                flow = self.model(f1t.to(self.device), f2t.to(self.device))[-1]     # [1,2,h,w]
                h, w = flow.shape[-2], flow.shape[-1]
                diag = float((h * h + w * w) ** 0.5)                                # resized-frame diagonal (px)
                f = flow[0] / diag                                                  # [2,h,w] frame-diagonal fraction
                g = F.adaptive_avg_pool2d(f, self.GRID)                             # [2,24,32]
                steps.append(g.permute(1, 2, 0).float().cpu().numpy())             # [24,32,2] (dx,dy)
        return {"flow": np.stack(steps).astype(np.float16),                         # [32,24,32,2]
                "idx": idx}


class ActionSwinExtractor:
    """Video Swin-B action-class signature (metrics v5, eval 049): Kinetics-400 logits / probs of
    32 uniformly sampled frames (the model's native clip length). Weights
    ``Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1`` (``swin3d_b_22k-7c6ae6fa.pth``, 81.6 top-1 K400),
    loaded from TORCH_HOME (the compute nodes run HF_HUB_OFFLINE=1 and must not download). Frames
    decoded short side 256 (a no-op resize under the shipped ``VideoClassification`` transform, which
    expects [..., T, C, H, W] float in [0,1], resizes short side 256, center-crops 224, applies
    ImageNet mean/std and permutes to [C,T,H,W]); batch of 1, no_grad. ``prob`` f32 [400] (softmax of
    the logits), ``logit`` f32 [400], ``idx`` i32 [32]. idx = np.round(np.linspace(0, T-1, 32)); T >= 32
    for every video here, so idx is strictly increasing; if T < 32 the round repeats indices (recorded
    in the stored ``idx`` array)."""

    NS = "action@swin3db-k400-u32"
    N_FRAMES = 32

    def __init__(self, device: str = "cuda"):
        from torchvision.models.video import Swin3D_B_Weights, swin3d_b
        self.device = device
        weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
        self.model = swin3d_b(weights=weights).to(device).eval()
        self.transforms = weights.transforms()

    def extract(self, video) -> dict[str, np.ndarray]:
        frames = _load_frames(video)
        T = len(frames)
        idx = np.round(np.linspace(0, T - 1, self.N_FRAMES)).astype(np.int32)       # [32]
        sel = np.ascontiguousarray(frames[idx])                                      # [32,H,W,3] uint8
        t = torch.from_numpy(sel).permute(0, 3, 1, 2).float().div(255.0)            # [T,C,H,W] in [0,1]
        with torch.no_grad():
            inp = self.transforms(t.unsqueeze(0)).to(self.device)                    # [1,3,32,224,224]
            logits = self.model(inp)[0].float()                                      # [400]
            prob = torch.softmax(logits, dim=-1)
        return {"prob": prob.cpu().numpy().astype(np.float32),
                "logit": logits.cpu().numpy().astype(np.float32),
                "idx": idx}


class TrackDescExtractor:
    """Track descriptors DERIVED from the stored CoTracker3 tracks (metrics v5).
    Reads ``cotracker3@g20-m384-v2`` (tracks/vis) from the feature store and emits
      dirs = _velocity_directions(tracks, vis, 64, 0.2, 0.1, 0.05).astype(f32)   [M,64,2]
      px   = step_features(tracks, vis).astype(f32)                              [31,18]
    exactly the ``dirs_for`` / ``px_for`` recipes of
    misc/2026-09-02_temporal_dynamics_metric/score_v3_mf.py (the PX channel's
    whole-field descriptor from run_motion_descriptors.step_features). CPU-only:
    the ``device`` argument is accepted and ignored (no backbone). Raises when the
    source tracks are absent."""

    NS = "trackdesc@cotracker3-s64-v1"
    SOURCE_NS = "cotracker3@g20-m384-v2"
    N_STEPS = 64

    def __init__(self, device: str = "cuda"):
        import pathlib as _pl
        import sys as _sys
        from diffusion.feature_store import FeatureStore
        from diffusion.transition_eval.motion import _velocity_directions
        # step_features lives in the campaign script; import it the way blend_grid does.
        repo = _pl.Path(__file__).resolve().parents[2]
        misc = str(repo / "misc" / "2026-09-02_temporal_dynamics_metric")
        if misc not in _sys.path:
            _sys.path.insert(0, misc)
        from run_motion_descriptors import step_features
        self._fs = FeatureStore(repo)
        self._veldir = _velocity_directions
        self._stepf = step_features

    def extract(self, video) -> dict[str, np.ndarray]:
        if not self._fs.has(video, self.SOURCE_NS):
            raise FileNotFoundError(
                f"{video}: source namespace {self.SOURCE_NS} absent; trackdesc "
                f"is derived from the stored cotracker3 tracks")
        z = self._fs.get(video, self.SOURCE_NS)
        dirs = self._veldir(z["tracks"], z["vis"], self.N_STEPS, 0.2, 0.1, 0.05).astype(np.float32)
        px = self._stepf(z["tracks"], z["vis"]).astype(np.float32)
        return {"dirs": dirs, "px": px}


class ViClipExtractor:
    """VBench `overall_consistency` video side: ViCLIP ViT-L/14 (InternVid-10M-FLT) video embedding of 8 frames
    sampled the VBench way (`sample="middle"`: the middle frame of 8 equal segments), preprocessed exactly as
    VBench's `clip_transform(224)` (bicubic resize of the short side to 224 without antialias, center crop,
    CLIP mean/std) on frames decoded at NATIVE resolution. `feat [768] f32`, L2-normalised; `frame_idx [8] i32`.
    Text side (`encode_text`) is the same model's text tower, used by the scoring script (prompt -> [768])."""

    NS = "viclip@l14-f8"
    N_FRAMES = 8
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, device: str = "cuda"):
        from diffusion.third_party.viclip.viclip import ViCLIP
        from diffusion.third_party.viclip.simple_tokenizer import SimpleTokenizer
        self.device = device
        self.tokenizer = SimpleTokenizer()
        self.model = ViCLIP(tokenizer=self.tokenizer).to(device).eval()

    @staticmethod
    def frame_indices(vlen: int, n: int = 8) -> list[int]:
        intervals = np.linspace(start=0, stop=vlen, num=min(n, vlen) + 1).astype(int)
        idx = [(a + (b - 1)) // 2 for a, b in zip(intervals[:-1], intervals[1:])]
        while len(idx) < n:
            idx.append(idx[-1])
        return idx

    def _transform(self, frames: np.ndarray) -> torch.Tensor:
        import torchvision.transforms.functional as TF
        from torchvision.transforms import InterpolationMode
        t = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)          # [T,3,H,W] uint8
        t = TF.resize(t, 224, interpolation=InterpolationMode.BICUBIC, antialias=False)   # short side -> 224
        t = TF.center_crop(t, 224).float().div(255.0)
        return TF.normalize(t, self.MEAN, self.STD)

    def extract(self, video) -> dict[str, np.ndarray]:
        frames = _load_frames(video, short_side=None)                                     # native resolution
        idx = self.frame_indices(len(frames), self.N_FRAMES)
        x = self._transform(frames[idx]).unsqueeze(0).to(self.device)                    # [1,T,3,224,224]
        with torch.no_grad():
            feat = self.model.encode_vision(x, test=True).float()
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return {"feat": feat[0].cpu().numpy().astype(np.float32), "frame_idx": np.asarray(idx, dtype=np.int32)}

    def encode_text(self, texts: list[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for t in texts:
                f = self.model.encode_text(t).float()
                out.append((f / f.norm(dim=-1, keepdim=True))[0].cpu().numpy())
        return np.stack(out).astype(np.float32)


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
    "raft_flow_win@r256": lambda device="cuda": RaftFlowWinExtractor(device),
    "viclip@l14-f8": lambda device="cuda": ViClipExtractor(device),
    "trackdesc@cotracker3-s64-v1": lambda device="cuda": TrackDescExtractor(device),
    "flow_u32@raft-r256-g24x32": lambda device="cuda": FlowU32Extractor(device),
    "action@swin3db-k400-u32": lambda device="cuda": ActionSwinExtractor(device),
}
