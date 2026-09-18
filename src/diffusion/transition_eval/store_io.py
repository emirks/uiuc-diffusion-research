"""store_io — the scoring path's bridge to the FeatureStore (SPEC §9; I/O only).

The certified transition-eval numerics are untouched; this module only changes
WHERE the three cached feature namespaces live. It replaces the hashed-filename
disk caches (``dino_arr_*.npz`` / ``tracks_*.npz`` / ``lpips_*.npz`` under a
``--cache-dir``) with the contract-v2 feature store (``store/FEATURES.md``,
``diffusion.feature_store``), keyed BY VIDEO PATH:

    dino_cls@dinov2b-r256   feats [T,768] f32   (was ``dino_arr_{sha}.npz``)
    cotracker3@g20-m384-v2  tracks,vis          (was ``tracks_{sha}.npz``)
    lpips_t@alex-r256       d [T-1] f32         (was the ``:tlpips:`` ``lpips_{sha}.npz``)

Real videos (gens, corpus / pool references, condition clips) persist by path
through :class:`~diffusion.feature_store.FeatureStore`. Synthetic controls
(lerp / static-hold — frames synthesized at scoring, no file on disk) have no
path of their own; they persist NEXT TO the gen's features as
``<ns>.ctl-<name>.npz`` + ``.json`` (``origin: "control:<name>"``), using the
store's own atomic (``.tmp-<pid>`` + rename) and sidecar-last conventions and
carrying the gen video's identity so ``FeatureStore.fsck`` reads them cleanly.

The dropped endpoint-LPIPS pair cache (``…:endp:…`` scalars, owner decision
P3b) has no home here — endpoint fidelity is a pairwise metric on 9 frames,
recomputed every run. Nothing in this module computes a metric: the
extractor / tracker / scorer calls stay in ``features.py`` / ``motion.py`` /
``endpoints.py``. numpy + stdlib + ``feature_store`` at import (no torch).
"""

from __future__ import annotations

import datetime
import os
import pathlib
import socket

import numpy as np

from ..feature_store import FeatureStore, NS_ARRAYS

# The three namespaces the scoring path reads / writes (store/FEATURES.md).
DINO_NS = "dino_cls@dinov2b-r256"
TRACK_NS = "cotracker3@g20-m384-v2"
LPIPS_NS = "lpips_t@alex-r256"
NAMESPACES = (DINO_NS, TRACK_NS, LPIPS_NS)


class RealVideo:
    """Identity of a real video file (gen, corpus / pool reference, cond clip).
    Its features persist by path via the FeatureStore path rule."""

    __slots__ = ("path",)

    def __init__(self, path: pathlib.Path | str):
        self.path = pathlib.Path(path)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"RealVideo({self.path})"


class Control:
    """Identity of a synthesized control (lerp / static-hold): no file of its
    own, so its features persist next to the GEN's features as
    ``<ns>.ctl-<name>``. ``name`` is the bare control name (``lerp`` / ``hold``)."""

    __slots__ = ("gen", "name")

    def __init__(self, gen_video: pathlib.Path | str, name: str):
        self.gen = pathlib.Path(gen_video)
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Control({self.gen}, {self.name!r})"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _primary_array(ns: str, files) -> str | None:
    return next((a for a in NS_ARRAYS.get(ns, ()) if a in files),
                files[0] if files else None)


class HarnessStore:
    """Adapter over :class:`FeatureStore` for the scoring path. Reads / writes
    the three eval namespaces by *identity* — a real video path, or a control
    persisted next to its gen. Real-video writes go through ``FeatureStore.put``
    (self-describing sidecar, atomic, sidecar-last); control writes mirror the
    same conventions in :meth:`_put_control`."""

    def __init__(self, store: FeatureStore | pathlib.Path | str, code_sha: str = ""):
        self.store = store if isinstance(store, FeatureStore) else FeatureStore(store)
        self.code_sha = code_sha

    @property
    def root(self) -> pathlib.Path:
        return self.store.root

    # -- presence / read / write, routed by identity --------------------------
    def has(self, identity, ns: str) -> bool:
        if isinstance(identity, Control):
            npz, side = self._control_paths(identity, ns)
            return npz.exists() and side.exists()
        return self.store.has(identity.path, ns)

    def get(self, identity, ns: str) -> dict[str, np.ndarray] | None:
        """Arrays for (identity, ns), or ``None`` on a miss."""
        if isinstance(identity, Control):
            npz, side = self._control_paths(identity, ns)
            if not (npz.exists() and side.exists()):
                return None
            z = np.load(npz)
            return {k: z[k] for k in z.files}
        if not self.store.has(identity.path, ns):
            return None
        return self.store.get(identity.path, ns)

    def put(self, identity, ns: str, arrays: dict[str, np.ndarray]) -> pathlib.Path:
        if isinstance(identity, Control):
            return self._put_control(identity, ns, arrays)
        return self.store.put(
            identity.path, ns, arrays,
            {"origin": "extracted", "code_sha": self.code_sha,
             "host": socket.gethostname()})

    def key_str(self, identity) -> str:
        """A stable human-readable bundle key (persistence no longer uses it)."""
        if isinstance(identity, Control):
            return f"control:{identity.name}:{identity.gen.stem}"
        return str(identity.path)

    # -- control file layout + atomic write (mirrors FeatureStore.put) --------
    def _control_paths(self, identity: "Control", ns: str) -> tuple[pathlib.Path, pathlib.Path]:
        base = self.store.path(identity.gen, ns)          # <feat_dir>/<ns>.npz
        d = base.parent
        return d / f"{ns}.ctl-{identity.name}.npz", d / f"{ns}.ctl-{identity.name}.json"

    def _gen_video_identity(self, gen: pathlib.Path, ns: str) -> dict:
        """The gen video's identity fields (sha/size/mtime) reused for the
        control sidecar so ``FeatureStore.fsck`` (which stats the item dir's
        gen video) reads a control as fresh. Prefer the gen's own sidecar
        (already migrated) over a fresh stat; never re-hashes the video."""
        try:
            if self.store.has(gen, ns):
                m = self.store.read_meta(gen, ns)
                return {"video_sha256": m.get("video_sha256"),
                        "video_size": m.get("video_size"),
                        "video_mtime_ns": m.get("video_mtime_ns")}
        except Exception:
            pass
        st = gen.stat()
        return {"video_sha256": None, "video_size": st.st_size,
                "video_mtime_ns": st.st_mtime_ns}

    def _put_control(self, identity: "Control", ns: str,
                     arrays: dict[str, np.ndarray]) -> pathlib.Path:
        import json
        npz, side = self._control_paths(identity, ns)
        d = npz.parent
        d.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()

        tmp_npz = d / f"{npz.name}.tmp-{pid}"
        if tmp_npz.exists():
            tmp_npz.unlink()
        with open(tmp_npz, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp_npz, npz)                          # atomic move into place

        z = np.load(npz)
        prim = _primary_array(ns, z.files)
        rel_gen = str(identity.gen.resolve().relative_to(self.store.root))
        vid = self._gen_video_identity(identity.gen, ns)
        sc = {
            "ns": ns,
            "video": rel_gen,                             # the gen this control derives from
            "video_sha256": vid["video_sha256"],
            "video_size": vid["video_size"],
            "video_mtime_ns": vid["video_mtime_ns"],
            "host": socket.gethostname(),                 # synthesized here, this run
            "control": identity.name,
            "code_sha": self.code_sha,
            "created": _now_iso(),
            "origin": f"control:{identity.name}",
            "shape": list(z[prim].shape) if prim is not None else [],
            "dtype": str(z[prim].dtype) if prim is not None else "",
            "bytes": npz.stat().st_size,
        }
        tmp_json = d / f"{side.name}.tmp-{pid}"
        tmp_json.write_text(json.dumps(sc, indent=2))
        os.replace(tmp_json, side)                        # sidecar written LAST
        return npz
