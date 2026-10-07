"""Batch-scoring driver for the 5 competitor metrics — roster-scale, shardable.

For each generated .mp4 it derives the reference clip and input start frame from
the filename (item_id embeds `__ref_<reference>`), loads every model ONCE and
shares it across metrics, and writes ONE JSON row per gen:

    {item_id, arm, group, variant, seed, endpoint, reference, ref_class,
     gen, reference_video, input_frame, decode_short_side, impl_sha, host,
     elapsed_sec,
     det_motion_fidelity,
     videoprism_sim_ref, videoprism_sim_ref_global,
     videoprism_sim_input, videoprism_sim_input_global,
     clip_sim_ref, clip_sim_input,
     motion_smoothness,
     dynamic_degree_mean_mag, dynamic_degree_bit,
     warnings: [...]}

Filename contract (store contract v2):
    .../store/gens/<group>/<variant>/videos/<PREFIX>__<endpoint>__ref_<reference>__<seedtok>.mp4
  * endpoint  = the `__`-field immediately before `__ref_`  (clip names use single _)
  * reference = the first `__`-field after `__ref_`
  * seed      = digits of the trailing s<NN> / seed<NN> field
  * ref_class = eval_ladder/prompts.clip_class(reference)   (authoritative, split_v1.2.json)
  * reference_video = data/processed/transitions_std121/<ref_class>/<reference>.mp4
  * input_frame     = misc/refvfx_baseline/frames/<endpoint>__first.png

Sharding: gens are sorted then sliced [shard::num_shards]. shard defaults to
SLURM_ARRAY_TASK_ID, num_shards to SLURM_ARRAY_TASK_COUNT (both overridable).
Per-gen skip-if-exists makes reruns cheap. CoTracker tracks are disk-cached by
absolute path, so the (few) reference clips are tracked once and reused across
every gen and every shard.

impl_sha = sha256 over the scorer .py files (first 16 hex) — asserts identical
implementation across scoring batches; stamped into every row + the run header.

NOTE (fixture finding, disclose downstream): the reference-fidelity embedding
sims (clip_sim_ref, videoprism_sim_ref) discriminate the true reference only
weakly — VideoPrism especially saturates in a narrow cone. The *_input and
*_global columns separate cleanly. Faithful refVFX/VAP-style impl, not a bug.
"""
from __future__ import annotations

import argparse
import glob as globmod
import hashlib
import json
import math
import os
import pathlib
import re
import socket
import sys
import time

import numpy as np

import common
from common import to_frames, load_image

HERE = pathlib.Path(__file__).resolve().parent
DR = HERE.parents[2]
STD = DR / "data/processed/transitions_std121"
FRAMES = DR / "misc/refvfx_baseline/frames"

# authoritative clip -> class
sys.path.insert(0, str(DR / "eval_ladder"))
import prompts  # noqa: E402

IMPL_FILES = ["common.py", "clip_sim.py", "motion_smoothness.py",
              "videoprism_sim.py", "dynamic_degree.py", "det_motion_fidelity.py",
              "score_batch.py"]


def impl_sha() -> str:
    h = hashlib.sha256()
    for f in sorted(IMPL_FILES):
        h.update((HERE / f).read_bytes())
    return h.hexdigest()[:16]


_SEED_RE = re.compile(r"(?:s|seed)(\d+)$")


def parse_item(gen_path: pathlib.Path) -> dict:
    stem = gen_path.stem
    variant = gen_path.parent.parent.name          # .../<variant>/videos/<file>
    group = gen_path.parent.parent.parent.name     # .../<group>/<variant>/videos
    if "__ref_" not in stem:
        raise ValueError(f"no __ref_ in item_id {stem!r}")
    left, right = stem.split("__ref_", 1)
    endpoint = left.split("__")[-1]
    parts = right.split("__")
    reference = parts[0]
    seed = None
    for tok in parts[1:]:
        m = _SEED_RE.match(tok)
        if m:
            seed = int(m.group(1))
    return dict(item_id=stem, arm=f"{group}/{variant}", group=group,
                variant=variant, endpoint=endpoint, reference=reference, seed=seed)


def _finite(x):
    """NaN/inf -> None (JSON-safe). NaN motion_fidelity = 'no moving tracklets' (valid)."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def collect_gens(args) -> list[pathlib.Path]:
    paths: list[str] = []
    if args.glob:
        paths += globmod.glob(args.glob)
    if args.manifest:
        for line in pathlib.Path(args.manifest).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                paths.append(json.loads(line)["gen"])
            else:
                paths.append(line)
    uniq = sorted(set(str(pathlib.Path(p).resolve()) for p in paths))
    return [pathlib.Path(p) for p in uniq]


# --- feature-store namespaces (store/FEATURES.md) ----------------------------
CLIP_NS = "clip_b32@r256"
VP_NS = "videoprism@f16r288"
RAFT_NS = "raft_mag@r256"
TRACK_NS = "cotracker3@g20-m384-v2"


class StoreLenses:
    """Per-video lens features through the contract-v2 feature store, keyed BY
    VIDEO PATH. ``diffusion.feature_extractors.REGISTRY`` is the SINGLE source of
    the preprocessing, so the arrays are identical by construction with the eval
    harness and any other store consumer. A store hit is returned as-is; a miss
    is extracted by the pinned REGISTRY extractor and written back atomically.
    ``.track_cache`` is retired — tracks live in the ``cotracker3@g20-m384-v2``
    namespace. The input FRAME (a png, not a video) is embedded fresh, reusing
    the SAME REGISTRY model instances so frame and video embeddings compare."""

    def __init__(self, store_root, device: str):
        from diffusion.feature_store import FeatureStore
        from diffusion.feature_extractors import REGISTRY
        self.store = FeatureStore(store_root)
        self.device = device
        self._REG = REGISTRY
        self._ext: dict = {}

    def _ext_for(self, ns: str):
        if ns not in self._ext:
            print(f"[store] loading extractor {ns} ...", flush=True)
            self._ext[ns] = self._REG[ns](self.device)
        return self._ext[ns]

    def has(self, path, ns: str) -> bool:
        return self.store.has(pathlib.Path(path), ns)

    def video(self, path, ns: str) -> dict:
        """Arrays for (video, ns): a store hit, else REGISTRY extract + write-back."""
        v = pathlib.Path(path)
        if self.store.has(v, ns):
            return self.store.get(v, ns)
        arrays = self._ext_for(ns).extract(str(v))
        self.store.put(v, ns, arrays,
                       {"origin": "extracted", "host": socket.gethostname(), "code_sha": ""})
        return arrays

    def clip_frame(self, frame: np.ndarray) -> np.ndarray:
        """CLIP-B/32 embedding of one frame [H,W,3] -> [1,512], via the
        ClipFrameExtractor model (mirrors clip_sim.ClipEmbedder.embed)."""
        import torch
        ext = self._ext_for(CLIP_NS)
        with torch.no_grad():
            inp = ext.proc(images=[frame], return_tensors="pt").to(ext.device)
            r = ext.model.get_image_features(**inp)
            emb = r.pooler_output if hasattr(r, "pooler_output") else r
            emb = torch.nn.functional.normalize(emb.float(), dim=-1)
        return emb.cpu().numpy().astype(np.float32)

    def vp_frame(self, frame: np.ndarray) -> np.ndarray:
        """VideoPrism embedding of one frame replicated to num_frames -> [16,768],
        via the VideoPrismExtractor model (mirrors VideoPrismEmbedder.embed)."""
        import torch
        ext = self._ext_for(VP_NS)
        rep = np.repeat(np.asarray(frame)[None], ext.num_frames, axis=0)
        with torch.no_grad():
            inp = ext.proc(videos=[list(rep)], return_tensors="pt").to(ext.device)
            lhs = ext.model(**inp).last_hidden_state
            nf, D = ext.num_frames, lhs.shape[-1]
            npatch = lhs.shape[1] // nf
            pf = lhs.view(1, nf, npatch, D).mean(2)[0]
            pf = torch.nn.functional.normalize(pf.float(), dim=-1)
        return pf.cpu().numpy().astype(np.float32)


class Models:
    """Load every backbone once; hold shared decode + track caches.

    With ``store`` set (default), the per-video lens arrays (CLIP-B/32,
    VideoPrism, RAFT magnitudes, CoTracker3 tracks) are read from / written to
    the feature store via :class:`StoreLenses` (REGISTRY extractors loaded lazily,
    once each, reused for the fresh input-frame embeds). Without it, the legacy
    in-process embedders are loaded and every video is embedded every run."""

    def __init__(self, device: str, track_cache: pathlib.Path, store=None):
        self.device = device
        self.store = store
        self._frames: dict[str, np.ndarray] = {}   # decode cache (refs/frames repeat)
        self._imgs: dict[str, np.ndarray] = {}
        if store is not None:
            self.clip = self.vp = self.raft = self.tracker = None
            self.track_cache = None
            return
        from clip_sim import ClipEmbedder
        from videoprism_sim import VideoPrismEmbedder
        from dynamic_degree import RaftFlow
        from diffusion.transition_eval.motion import Tracker
        print("[models] CLIP ...", flush=True)
        self.clip = ClipEmbedder(device=device)
        print("[models] VideoPrism ...", flush=True)
        self.vp = VideoPrismEmbedder(device=device)
        print("[models] RAFT ...", flush=True)
        self.raft = RaftFlow(device=device)
        print("[models] CoTracker3 ...", flush=True)
        self.tracker = Tracker(device=device)
        self.track_cache = track_cache
        self.track_cache.mkdir(parents=True, exist_ok=True)

    def frames(self, path: str, cache: bool) -> np.ndarray:
        if cache and path in self._frames:
            return self._frames[path]
        f = to_frames(path)
        if cache:
            self._frames[path] = f
        return f

    def image(self, path: str) -> np.ndarray:
        if path not in self._imgs:
            self._imgs[path] = load_image(path)
        return self._imgs[path]

    def tracks(self, path: str, frames: np.ndarray):
        # keyed by absolute path -> reference clips tracked once, reused everywhere
        return self.tracker.cached_track(frames, path, self.track_cache)


def score_one(gen: pathlib.Path, M: Models, sha: str) -> dict:
    from clip_sim import CLIP_ID  # noqa
    from common import set_similarity
    from diffusion.transition_eval.motion import motion_fidelity

    t0 = time.time()
    meta = parse_item(gen)
    ref_class = prompts.clip_class(meta["reference"])
    ref = STD / ref_class / f"{meta['reference']}.mp4"
    frame = FRAMES / f"{meta['endpoint']}__first.png"
    warnings = []

    has_ref = ref.exists()
    has_frame = frame.exists()
    if not has_ref:
        warnings.append(f"missing reference_video {ref}")
    if not has_frame:
        warnings.append(f"missing input_frame {frame}")
    frame_img = M.image(str(frame)) if has_frame else None

    # ---- acquire the per-video lens arrays (store or legacy embedders) -------
    # store path: read/write clip_b32/videoprism/raft_mag/cotracker3 BY VIDEO
    # PATH through the feature store, filling misses with the REGISTRY extractors
    # (single preprocessing source). The input FRAME is embedded fresh either way.
    if M.store is not None:
        g_clip = M.store.video(str(gen), CLIP_NS)["feats"]
        r_clip = M.store.video(str(ref), CLIP_NS)["feats"] if has_ref else None
        g_vp = M.store.video(str(gen), VP_NS)["feats"]
        r_vp = M.store.video(str(ref), VP_NS)["feats"] if has_ref else None
        g_mags = M.store.video(str(gen), RAFT_NS)["mag"]
        if has_ref:
            _tg = M.store.video(str(gen), TRACK_NS); tg, vg = _tg["tracks"], _tg["vis"]
            _tr = M.store.video(str(ref), TRACK_NS); tr, vr = _tr["tracks"], _tr["vis"]
        f_clip = M.store.clip_frame(frame_img) if has_frame else None
        i_vp = M.store.vp_frame(frame_img) if has_frame else None
    else:
        gen_frames = M.frames(str(gen), cache=False)
        ref_frames = M.frames(str(ref), cache=True) if has_ref else None
        g_clip = M.clip.embed(gen_frames)
        r_clip = M.clip.embed(ref_frames) if has_ref else None
        g_vp = M.vp.embed(gen_frames)
        r_vp = M.vp.embed(ref_frames) if has_ref else None
        g_mags = (M.raft.mean_magnitudes(gen_frames) if len(gen_frames) >= 2
                  else np.asarray([], dtype=np.float32))
        if has_ref:
            tg, vg = M.tracks(str(gen), gen_frames)
            tr, vr = M.tracks(str(ref), ref_frames)
        f_clip = M.clip.embed(frame_img[None]) if has_frame else None
        i_vp = M.vp.embed(np.repeat(frame_img[None], M.vp.num_frames, axis=0)) if has_frame else None

    row = dict(**meta, ref_class=ref_class, gen=str(gen),
               reference_video=str(ref), input_frame=str(frame),
               decode_short_side=common.DECODE_SHORT_SIDE, impl_sha=sha)

    # ---- CLIP: clip_sim (ref + input) + motion_smoothness -------------------
    row["clip_sim_ref"] = _finite(set_similarity(g_clip, r_clip)) if has_ref else None
    row["clip_sim_input"] = _finite(set_similarity(g_clip, f_clip)) if has_frame else None
    if len(g_clip) >= 2:
        row["motion_smoothness"] = _finite(float((g_clip[:-1] * g_clip[1:]).sum(axis=1).mean()))
    else:
        row["motion_smoothness"] = None

    # ---- VideoPrism: ref + input (per-frame mean-of-max + global) -----------
    def vp_global(pf):
        v = pf.mean(axis=0)
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)
    gg = vp_global(g_vp)
    if has_ref:
        row["videoprism_sim_ref"] = _finite(set_similarity(g_vp, r_vp))
        row["videoprism_sim_ref_global"] = _finite(float(gg @ vp_global(r_vp)))
    else:
        row["videoprism_sim_ref"] = row["videoprism_sim_ref_global"] = None
    if has_frame:
        row["videoprism_sim_input"] = _finite(set_similarity(g_vp, i_vp))
        row["videoprism_sim_input_global"] = _finite(float(gg @ vp_global(i_vp)))
    else:
        row["videoprism_sim_input"] = row["videoprism_sim_input_global"] = None

    # ---- Dynamic degree (gen only) -----------------------------------------
    if len(g_clip) >= 2:
        mm = float(g_mags.mean())
        row["dynamic_degree_mean_mag"] = _finite(mm)
        from dynamic_degree import MOTION_THRESHOLD
        row["dynamic_degree_bit"] = float(mm >= MOTION_THRESHOLD)
    else:
        row["dynamic_degree_mean_mag"] = row["dynamic_degree_bit"] = None

    # ---- DeT motion fidelity (gen vs ref); tracks from the store ------------
    if has_ref:
        row["det_motion_fidelity"] = _finite(float(motion_fidelity(tg, vg, tr, vr)))
    else:
        row["det_motion_fidelity"] = None

    row["warnings"] = warnings
    row["host"] = socket.gethostname()
    row["elapsed_sec"] = round(time.time() - t0, 2)
    return row


def main():
    ap = argparse.ArgumentParser(description="Batch competitor-metric scorer (shardable)")
    ap.add_argument("--glob", default=None, help="glob of gen .mp4 paths")
    ap.add_argument("--manifest", default=None, help="file: one gen path per line, or .jsonl {gen:...}")
    ap.add_argument("--out-dir", default=None, help="per-gen JSON rows go here (required unless --dry-run)")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num-shards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--limit", type=int, default=0, help="cap N gens (validation)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--track-cache", default=str(HERE / ".track_cache"),
                    help="legacy path-keyed track cache (only used with --no-store)")
    ap.add_argument("--store", action=argparse.BooleanOptionalAction, default=True,
                    help="read/write per-video lens features (clip_b32, videoprism, "
                         "raft_mag, cotracker3) through the feature store keyed by "
                         "video path; REGISTRY extractors fill misses (default on). "
                         "--no-store restores the legacy in-process embedders.")
    ap.add_argument("--store-root", default=None,
                    help="feature store root (default: the repo root)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve store namespaces for each gen+ref and print "
                         "hit/would-extract per namespace; loads no models, scores nothing")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    sha = impl_sha()
    gens = collect_gens(a)
    if not gens:
        raise SystemExit("no gens matched --glob/--manifest")
    shard_gens = gens[a.shard::a.num_shards] if a.num_shards > 1 else gens
    if a.limit:
        shard_gens = shard_gens[:a.limit]
    store_root = pathlib.Path(a.store_root).resolve() if a.store_root else DR

    if a.dry_run:
        from diffusion.feature_store import FeatureStore
        fs = FeatureStore(store_root)
        print(f"[dry-run] impl_sha={sha} store_root={store_root} gens={len(shard_gens)}", flush=True)
        for gen in shard_gens:
            meta = parse_item(gen)
            ref = STD / prompts.clip_class(meta["reference"]) / f"{meta['reference']}.mp4"
            for label, vid in (("gen", gen), ("ref", ref)):
                if not pathlib.Path(vid).exists():
                    print(f"[dry] {label:3s} MISSING {vid}", flush=True)
                    continue
                st = {ns: ("hit" if fs.has(pathlib.Path(vid), ns) else "extract")
                      for ns in (CLIP_NS, VP_NS, RAFT_NS, TRACK_NS)}
                print(f"[dry] {label:3s} {pathlib.Path(vid).name}: "
                      + "  ".join(f"{ns.split('@')[0]}={s}" for ns, s in st.items()), flush=True)
        return

    if not a.out_dir:
        raise SystemExit("--out-dir is required unless --dry-run")
    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[batch] impl_sha={sha} total_gens={len(gens)} shard={a.shard}/{a.num_shards} "
          f"this_shard={len(shard_gens)} out={out_dir} store={'on' if a.store else 'off'}", flush=True)

    store = StoreLenses(store_root, a.device) if a.store else None
    M = Models(a.device, pathlib.Path(a.track_cache), store=store)
    done = fail = skip = 0
    for i, gen in enumerate(shard_gens):
        item = gen.stem
        of = out_dir / f"{item}.json"
        if of.exists() and not a.overwrite:
            skip += 1
            continue
        try:
            row = score_one(gen, M, sha)
            tmp = of.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(row))
            tmp.replace(of)
            done += 1
            print(f"[{i+1}/{len(shard_gens)}] {item}  detMF={row['det_motion_fidelity']} "
                  f"clip_ref={row['clip_sim_ref']} vp_input={row['videoprism_sim_input']} "
                  f"smooth={row['motion_smoothness']} dyn={row['dynamic_degree_mean_mag']} "
                  f"({row['elapsed_sec']}s)", flush=True)
        except Exception as e:
            fail += 1
            print(f"[{i+1}/{len(shard_gens)}] FAIL {item}: {type(e).__name__}: {e}", flush=True)
    print(f"[batch] done={done} skip={skip} fail={fail} impl_sha={sha}", flush=True)


if __name__ == "__main__":
    main()
