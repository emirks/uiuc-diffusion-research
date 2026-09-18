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

import pathlib
import socket

import numpy as np

from ..feature_store import FeatureStore

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


class HarnessStore:
    """Adapter over :class:`FeatureStore` for the scoring path. Reads / writes
    the three eval namespaces by *identity* — a real video path, or a control
    persisted next to its gen. Both routes go through the library: a real video
    is ``store.put(path, ns, …)``; a control is ``store.put(gen, ns, …,
    variant=name)``, which writes ``<ns>.ctl-<name>`` with the gen's identity in
    the sidecar (self-describing, atomic, sidecar-last). No I/O is reimplemented
    here — the store owns the control-file layout."""

    def __init__(self, store: FeatureStore | pathlib.Path | str, code_sha: str = ""):
        self.store = store if isinstance(store, FeatureStore) else FeatureStore(store)
        self.code_sha = code_sha

    @property
    def root(self) -> pathlib.Path:
        return self.store.root

    # -- presence / read / write, routed by identity --------------------------
    def has(self, identity, ns: str) -> bool:
        if isinstance(identity, Control):
            return self.store.has(identity.gen, ns, variant=identity.name)
        return self.store.has(identity.path, ns)

    def get(self, identity, ns: str) -> dict[str, np.ndarray] | None:
        """Arrays for (identity, ns), or ``None`` on a miss."""
        if isinstance(identity, Control):
            if not self.store.has(identity.gen, ns, variant=identity.name):
                return None
            return self.store.get(identity.gen, ns, variant=identity.name)
        if not self.store.has(identity.path, ns):
            return None
        return self.store.get(identity.path, ns)

    def put(self, identity, ns: str, arrays: dict[str, np.ndarray]) -> pathlib.Path:
        if isinstance(identity, Control):
            # the gen's identity carried in the control sidecar; reuse the gen's
            # already-known sha (its sidecar, else SHA256SUMS) — never re-hash.
            return self.store.put(
                identity.gen, ns, arrays,
                {"origin": f"control:{identity.name}", "code_sha": self.code_sha,
                 "host": socket.gethostname()},
                variant=identity.name,
                video_sha256=self._gen_sha(identity.gen, ns))
        return self.store.put(
            identity.path, ns, arrays,
            {"origin": "extracted", "code_sha": self.code_sha,
             "host": socket.gethostname()},
            video_sha256=self.store.sha_from_sums(identity.path))

    def key_str(self, identity) -> str:
        """A stable human-readable bundle key (persistence no longer uses it)."""
        if isinstance(identity, Control):
            return f"control:{identity.name}:{identity.gen.stem}"
        return str(identity.path)

    def _gen_sha(self, gen: pathlib.Path, ns: str) -> str | None:
        """The gen video's sha, reused for the control sidecar so ``fsck``
        (stats the item folder's gen mp4) reads a control as fresh WITHOUT
        re-hashing. Prefer the gen's own sidecar, then ``SHA256SUMS``; ``None``
        lets the store hash the gen (only when neither exists)."""
        try:
            if self.store.has(gen, ns):
                s = self.store.read_meta(gen, ns).get("video_sha256")
                if s:
                    return s
        except Exception:
            pass
        return self.store.sha_from_sums(gen)
