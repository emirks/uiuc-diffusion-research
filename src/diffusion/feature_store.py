"""feature_store — colocated per-video feature storage (contract v2, clause 10).

One rule, two cases (see store/FEATURES.md):
  - store gens: ``features/`` sits NEXT TO ``videos/`` —
    ``<variant>/videos/<item>.mp4`` <-> ``<variant>/features/<item>/<ns>.npz``.
  - clips not under a ``videos/`` folder (corpus, endpoint clips): ``features/``
    sits INSIDE the clip folder — ``<class>/features/<clip>/<ns>.npz``.

Each (video, namespace) is TWO files: ``<ns>.npz`` (the arrays, exactly as the
extractor emits them — a migrated file is a HARD LINK of the legacy cache file,
same inode) and ``<ns>.json`` (the meta sidecar). Writes are atomic
(``<file>.tmp-<pid>`` -> rename) and the ``.json`` is written LAST, so a ``.npz``
without its ``.json`` is an interrupted write that ``fsck`` reports.

Library only (repo rule): numpy + stdlib at import; no CLI, no backbones. The
CLI lives in ``scripts/store_features.py`` and extractors in
``src/diffusion/feature_extractors.py``.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import socket
from pathlib import Path

import numpy as np

# --- namespace registry (order = the coverage-matrix column order) -----------
# The pins themselves live in store/FEATURES.md; here we only need each
# namespace's array names (the first is the "primary" array whose shape/dtype
# the sidecar records).
NS_ARRAYS: dict[str, tuple[str, ...]] = {
    "dino_cls@dinov2b-r256": ("feats",),
    "cotracker3@g20-m384-v2": ("tracks", "vis"),
    "lpips_t@alex-r256": ("d",),
    "clip_b32@r256": ("feats",),
    "videoprism@f16r288": ("feats",),
    "raft_mag@r256": ("mag",),
    "clip_l14@r224": ("feats",),
}
NAMESPACES: tuple[str, ...] = tuple(NS_ARRAYS)

# --- legacy-cache key + filename recipes (must match transition_eval exactly) -
# transition_eval/features.py::file_key(path, "facebook/dinov2-base", "256").
DINO_MODEL = "facebook/dinov2-base"
SHORT_SIDE = 256
# only these three namespaces were ever persisted to the legacy caches.
MIGRATABLE: tuple[str, ...] = (
    "dino_cls@dinov2b-r256", "cotracker3@g20-m384-v2", "lpips_t@alex-r256")


def _sha16(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def legacy_key(video: Path | str) -> str:
    """The transition_eval stat-based bundle key for a real video file:
    ``sha1(resolved_path | st_mtime_ns | st_size | model | short_side)[:16]``."""
    video = Path(video)
    st = video.stat()
    raw = "|".join([str(video.resolve()), str(st.st_mtime_ns), str(st.st_size),
                    DINO_MODEL, str(SHORT_SIDE)])
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def legacy_filenames(ns: str, key: str) -> list[str]:
    """Candidate legacy cache filenames for a namespace, in preference order.

    - ``dino_cls@dinov2b-r256``: ``dino_arr_{sha1(key)[:16]}.npz`` (array-features
      form, arrays ``feats`` [, ``src``]), then the older ``dino_{key}.npz``
      (arrays ``feats``, ``fps``, ``src``).
    - ``cotracker3@g20-m384-v2``: ``tracks_{sha1(key + ':tracks:v2')[:16]}.npz``.
    - ``lpips_t@alex-r256``: ``lpips_{sha1(key + ':tlpips:alex-v1')[:16]}.npz``.
    """
    if ns == "dino_cls@dinov2b-r256":
        return [f"dino_arr_{_sha16(key)}.npz", f"dino_{key}.npz"]
    if ns == "cotracker3@g20-m384-v2":
        return [f"tracks_{_sha16(key + ':tracks:v2')}.npz"]
    if ns == "lpips_t@alex-r256":
        return [f"lpips_{_sha16(key + ':tlpips:alex-v1')}.npz"]
    return []


# --- small stdlib helpers -----------------------------------------------------
def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path | str, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _canon(rec: dict) -> str:
    """Serialization used to compare manifest lines to sidecars (order-free)."""
    return json.dumps(rec, sort_keys=True)


class FeatureStore:
    """Path-is-identity feature store rooted at the repo root."""

    def __init__(self, repo_root: Path | str):
        self.root = Path(repo_root).resolve()

    # -- path rule (the one rule from store/FEATURES.md) ----------------------
    def _feat_dir(self, video: Path | str) -> Path:
        video = Path(video)
        base = video.parent.parent if video.parent.name == "videos" else video.parent
        return base / "features" / video.stem

    @staticmethod
    def _fname(ns: str, variant: str | None) -> str:
        """The on-disk basename for a (ns, variant): ``<ns>`` for a real
        namespace, ``<ns>.ctl-<variant>`` for a synthetic control."""
        return f"{ns}.ctl-{variant}" if variant is not None else ns

    def path(self, video: Path | str, ns: str, *, variant: str | None = None) -> Path:
        return self._feat_dir(video) / f"{self._fname(ns, variant)}.npz"

    def sidecar(self, video: Path | str, ns: str, *, variant: str | None = None) -> Path:
        return self._feat_dir(video) / f"{self._fname(ns, variant)}.json"

    @staticmethod
    def _feat_root(video_dir: Path | str) -> Path:
        video_dir = Path(video_dir)
        base = video_dir.parent if video_dir.name == "videos" else video_dir
        return base / "features"

    def _relvideo(self, video: Path | str) -> str:
        return str(Path(video).resolve().relative_to(self.root))

    # -- presence / read ------------------------------------------------------
    # ``variant`` selects a synthetic control (``<ns>.ctl-<variant>``) instead
    # of the real namespace file; ``None`` is the ordinary namespace.
    def has(self, video: Path | str, ns: str, *, variant: str | None = None) -> bool:
        """True only when BOTH files exist (a lone .npz is an interrupted write)."""
        return (self.path(video, ns, variant=variant).exists()
                and self.sidecar(video, ns, variant=variant).exists())

    def get(self, video: Path | str, ns: str, *, variant: str | None = None) -> dict[str, np.ndarray]:
        z = np.load(self.path(video, ns, variant=variant))
        return {k: z[k] for k in z.files}

    def read_meta(self, video: Path | str, ns: str, *, variant: str | None = None) -> dict:
        return json.loads(self.sidecar(video, ns, variant=variant).read_text())

    def sha_from_sums(self, video: Path | str) -> str | None:
        """The sha256 of ``video`` from the ``SHA256SUMS`` next to it (the gen's
        ``videos/SHA256SUMS`` or the clip folder's), or ``None`` when the file or
        the line is absent. Lets ``put`` skip re-hashing (P4 throughput)."""
        video = Path(video)
        sums = video.parent / "SHA256SUMS"
        if not sums.exists():
            return None
        for ln in sums.read_text().splitlines():
            if "  " in ln:
                sha, name = ln.split("  ", 1)
                if name == video.name:
                    return sha
        return None

    # -- write (atomic; sidecar last) -----------------------------------------
    def put(self, video: Path | str, ns: str, arrays: dict | None, meta: dict,
            *, variant: str | None = None, link_from: Path | str | None = None,
            video_sha256: str | None = None) -> Path:
        """Write ``<ns>.npz`` + ``<ns>.json`` atomically.

        ``variant`` (a control name) writes ``<ns>.ctl-<variant>`` instead of the
        bare namespace — a synthetic control (lerp / static-hold) derived from
        ``video`` (the gen it was synthesized against); the sidecar then carries
        the gen's identity and a ``control`` field, and ``origin`` is expected to
        be ``control:<variant>``. ``link_from`` hard-links an existing (legacy)
        npz instead of writing ``arrays`` — same inode, zero extra bytes. The
        sidecar is written LAST. ``meta`` supplies the caller-known fields
        (``host``, ``code_sha``, ``origin`` and optionally ``created``); the
        mechanical fields (ns, relative video path, video size/mtime,
        shape/dtype/bytes) are filled here from the files themselves.
        ``video_sha256`` (or ``meta["video_sha256"]``), when given, skips
        re-hashing the video — pass a precomputed sha (e.g. from
        ``sha_from_sums``) for throughput.
        """
        video = Path(video)
        name = self._fname(ns, variant)
        npz = self.path(video, ns, variant=variant)
        side = self.sidecar(video, ns, variant=variant)
        d = npz.parent
        d.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()

        tmp_npz = d / f"{name}.npz.tmp-{pid}"
        if tmp_npz.exists():
            tmp_npz.unlink()
        if link_from is not None:
            os.link(Path(link_from), tmp_npz)          # hard link (same inode)
        else:
            if arrays is None:
                raise ValueError("put() needs arrays when link_from is None")
            with open(tmp_npz, "wb") as f:
                np.savez_compressed(f, **arrays)
        os.replace(tmp_npz, npz)                        # atomic move into place

        st_v = video.stat()
        z = np.load(npz)
        prim = next((a for a in NS_ARRAYS.get(ns, ()) if a in z.files),
                    z.files[0] if z.files else None)
        origin = meta.get("origin", "extracted")
        migrated = origin.startswith("migrated")
        sha = video_sha256 or meta.get("video_sha256") or sha256_file(video)
        # `host` names the machine that EXTRACTED the arrays (the store's host
        # rule guards cross-machine feature drift). Legacy caches carry no host,
        # so a migrated file's extraction host is UNKNOWN -> null; the migrating
        # node is recorded separately as `migrated_by_host`.
        sc = {
            "ns": ns,
            "video": self._relvideo(video),
            "video_sha256": sha,
            "video_size": st_v.st_size,
            "video_mtime_ns": st_v.st_mtime_ns,
            "host": None if migrated else (meta.get("host") or socket.gethostname()),
        }
        if migrated:
            sc["migrated_by_host"] = meta.get("migrated_by_host") or socket.gethostname()
        if variant is not None:
            sc["control"] = variant                     # a synthetic control file
        sc.update({
            "code_sha": meta.get("code_sha", ""),
            "created": meta.get("created") or _now_iso(),
            "origin": origin,
            "shape": list(z[prim].shape) if prim is not None else [],
            "dtype": str(z[prim].dtype) if prim is not None else "",
            "bytes": npz.stat().st_size,
        })

        tmp_json = d / f"{name}.json.tmp-{pid}"
        tmp_json.write_text(json.dumps(sc, indent=2))
        os.replace(tmp_json, side)                      # sidecar written LAST
        return npz

    # -- coverage / fsck / manifest -------------------------------------------
    def iter_videos(self, video_dir: Path | str) -> list[Path]:
        return sorted(Path(video_dir).glob("*.mp4"))

    def coverage(self, video_dir: Path | str, videos=None) -> dict[str, dict]:
        """Per-namespace ``{have, of, hosts}`` over the videos in ``video_dir``
        (or over ``videos``, an explicit list, when a population restricts the
        run). The per-namespace ``have`` counts only real-namespace files;
        synthetic controls (``<ns>.ctl-<name>``) are reported separately under
        the ``controls`` key — ``{have, of, names}`` — so they never inflate a
        namespace cell."""
        vids = [Path(v) for v in videos] if videos is not None else self.iter_videos(video_dir)
        of = len(vids)
        cov: dict[str, dict] = {}
        for ns in NAMESPACES:
            have, legacy, hosts = 0, 0, set()
            for v in vids:
                if self.has(v, ns):          # real-namespace file only (no .ctl-)
                    have += 1
                    try:
                        h = self.read_meta(v, ns).get("host")
                    except Exception:
                        h = None
                    if h:
                        hosts.add(h)          # known extraction host
                    else:
                        legacy += 1           # migrated: extraction host unknown
            cov[ns] = {"have": have, "of": of, "legacy": legacy,
                       "hosts": sorted(hosts)}
        # controls: derived <ns>.ctl-<name> files living in the item folders,
        # first-class but counted apart from the namespaces they hang off.
        ctl_have, ctl_names = 0, set()
        for v in vids:
            fd = self._feat_dir(v)
            if not fd.exists():
                continue
            for npz in fd.glob("*.ctl-*.npz"):
                if npz.with_suffix(".json").exists():
                    ctl_have += 1
                    ctl_names.add(npz.stem.split(".ctl-", 1)[1])
        cov["controls"] = {"have": ctl_have, "of": of, "names": sorted(ctl_names)}
        return cov

    def fsck(self, video_dir: Path | str, rehash: bool = False, videos=None) -> dict:
        """Validate the features under ``video_dir``.

        Report keys: ``orphan_npz`` (npz with no sidecar = interrupted write),
        ``orphan_json`` (sidecar with no npz), ``no_video`` (feature folder with
        no matching mp4), ``stale`` (video size/mtime differ, or sha differs when
        ``rehash``), ``mixed_hosts`` ({ns: [hosts]} when a namespace spans >1
        host), ``manifest_drift`` (bool). ``ok`` is False on any orphan/stale/
        mixed-host (manifest drift alone is fixable and does not flip ``ok``).
        ``videos`` (a population subset) restricts the checked items to those
        clips; ``no_video`` is still judged against every mp4 in the dir.
        """
        video_dir = Path(video_dir)
        feat_root = self._feat_root(video_dir)
        rep = {"video_dir": str(video_dir), "orphan_npz": [], "orphan_json": [],
               "no_video": [], "stale": [], "mixed_hosts": {}, "legacy_ns": [],
               "manifest_drift": False, "n_videos": 0, "n_features": 0,
               "n_controls": 0}
        all_vids = {v.stem: v for v in self.iter_videos(video_dir)}
        vids = ({Path(v).stem: Path(v) for v in videos} if videos is not None
                else all_vids)
        rep["n_videos"] = len(vids)
        if not feat_root.exists():
            rep["ok"] = True
            return rep

        ns_hosts: dict[str, set] = {}
        sidecars: list[dict] = []
        for item_dir in sorted(p for p in feat_root.iterdir() if p.is_dir()):
            stem = item_dir.name
            if stem not in all_vids:
                rep["no_video"].append(stem)
            if videos is not None and stem not in vids:
                continue
            npzs = {p.stem for p in item_dir.glob("*.npz")}   # stem drops ".npz"
            jsons = {p.stem for p in item_dir.glob("*.json")}
            # ``name`` is the file basename minus ".npz"; a synthetic control
            # is ``<ns>.ctl-<variant>`` (the ".ctl-" infix never occurs in a
            # real namespace tag). Controls are first-class — pair + sidecar +
            # stale-checked exactly like namespaces (they carry the gen video's
            # identity, so the item-folder mp4 stat validates them) — but they
            # do NOT feed the per-namespace host tracking.
            for name in sorted(npzs - jsons):
                rep["orphan_npz"].append(f"{stem}/{name}.npz")
            for name in sorted(jsons - npzs):
                rep["orphan_json"].append(f"{stem}/{name}.json")
            for name in sorted(npzs & jsons):
                sc = json.loads((item_dir / f"{name}.json").read_text())
                sidecars.append(sc)
                if ".ctl-" in name:
                    rep["n_controls"] += 1
                else:
                    rep["n_features"] += 1
                    ns_hosts.setdefault(name, set()).add(sc.get("host"))
                v = vids.get(stem)
                if v is not None:
                    st = v.stat()
                    if (st.st_size != sc.get("video_size")
                            or st.st_mtime_ns != sc.get("video_mtime_ns")):
                        rep["stale"].append(f"{stem}/{name} (stat drift)")
                    elif rehash and sha256_file(v) != sc.get("video_sha256"):
                        rep["stale"].append(f"{stem}/{name} (sha mismatch)")

        # only KNOWN extraction hosts count for the mixed-host guard; a
        # namespace whose present features are all migrated (host unknown) is
        # "legacy", reported informationally rather than as a mixed-host warning.
        rep["mixed_hosts"] = {ns: sorted(h for h in hs if h)
                              for ns, hs in ns_hosts.items()
                              if len({h for h in hs if h}) > 1}
        rep["legacy_ns"] = sorted(ns for ns, hs in ns_hosts.items()
                                  if not {h for h in hs if h})

        manifest = feat_root / "manifest.jsonl"
        if manifest.exists():
            have = {ln for ln in manifest.read_text().splitlines() if ln.strip()}
            want = {_canon(sc) for sc in sidecars}
            rep["manifest_drift"] = ({_canon(json.loads(ln)) for ln in have} != want)
        else:
            rep["manifest_drift"] = bool(sidecars)

        rep["ok"] = not (rep["orphan_npz"] or rep["orphan_json"]
                         or rep["stale"] or rep["mixed_hosts"])
        return rep

    def rebuild_manifest(self, video_dir: Path | str) -> int:
        """Rewrite ``features/manifest.jsonl`` = the sidecars, one per line."""
        feat_root = self._feat_root(video_dir)
        recs: list[dict] = []
        if feat_root.exists():
            for side in feat_root.glob("*/*.json"):
                recs.append(json.loads(side.read_text()))
        recs.sort(key=lambda r: (r.get("video", ""), r.get("ns", "")))
        feat_root.mkdir(parents=True, exist_ok=True)
        tmp = feat_root / f"manifest.jsonl.tmp-{os.getpid()}"
        tmp.write_text("".join(_canon(r) + "\n" for r in recs))
        os.replace(tmp, feat_root / "manifest.jsonl")
        return len(recs)

    # -- gen meta.yaml features: block (text-level edit) ----------------------
    def write_meta_block(self, variant_dir: Path | str) -> dict:
        """Refresh (or insert) the ``features:`` block in a gen ``meta.yaml``,
        leaving every other key and comment untouched. Only namespaces with
        ``have > 0`` are listed. Returns the coverage that was written."""
        variant_dir = Path(variant_dir)
        meta_p = variant_dir / "meta.yaml"
        cov = self.coverage(variant_dir / "videos")
        listed = [ns for ns in NAMESPACES if cov[ns]["have"] > 0]
        block = ["features:"]
        for ns in listed:
            c = cov[ns]
            hosts = "[" + ", ".join(c["hosts"]) + "]"
            extra = f", legacy: {c['legacy']}" if c.get("legacy") else ""
            block.append(f"  {ns}: {{have: {c['have']}, of: {c['of']}, "
                         f"hosts: {hosts}{extra}}}")
        block_text = "\n".join(block)

        lines = meta_p.read_text().splitlines()
        out, i, replaced = [], 0, False
        while i < len(lines):
            if lines[i].startswith("features:"):
                out.append(block_text)
                replaced = True
                i += 1
                while i < len(lines) and (lines[i][:1] in (" ", "\t")):
                    i += 1                                   # drop old block body
                continue
            out.append(lines[i])
            i += 1
        if not replaced:
            if out and out[-1].strip():
                out.append(block_text)
            else:
                out.append(block_text)
        meta_p.write_text("\n".join(out) + "\n")
        return cov
