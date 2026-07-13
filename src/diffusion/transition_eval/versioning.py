"""Versioning & provenance stamping for the transition eval harness (SPEC.md §7/§9/§10).

Every result artifact carries a stamp answering three questions:

    WHAT measured it — package version, git commit, SPEC.md hash, package dirty state
    WITH WHAT        — pinned measurement dependencies + observed environment
    ON WHAT          — corpus manifest hash

A stamp is CERTIFIED only when (a) the harness package's working tree is clean,
(b) HEAD carries the exact annotated tag ``eval/v<version>``, and (c) the
version is not a draft. Anything else is stamped UNCERTIFIED with the reasons
listed, and every report rendered from such a run must surface that. Numbers
under different stamps are never comparable; the only bridge is rescoring old
items under one stamp (features are cached, so this is cheap).

Pure stdlib — no torch, no numpy. Safe on login nodes and inside any env.

CLI (two equivalent invocations):
    python -m diffusion.transition_eval.versioning ...   # inside the diffusion env
    python src/diffusion/transition_eval/versioning.py ...  # any python ≥3.9, no deps
        (the parent `diffusion` package imports torch at import time, so the -m
        form needs the env; this file itself is standalone by design)
    --corpus PATH  hash a corpus_manifest.json into the stamp
    --check        exit 1 unless the stamp is certified (used by the §10 flow)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import subprocess

PKG_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[2]
SPEC_PATH = PKG_DIR / "SPEC.md"
VERSION_PATH = PKG_DIR / "VERSION"
TAG_PREFIX = "eval/v"

# Paths whose modification invalidates the instrument (dirty check scope).
# Deliberately NOT the whole repo: experiment scaffolds and notes may change
# freely without re-certification; the measuring device may not.
INSTRUMENT_PATHS = (
    "src/diffusion/transition_eval",
    "tests/test_transition_eval.py",
    "tests/test_transition_eval_v3.py",
    "tests/test_certify_v3.py",
    "tests/test_versioning.py",
)

# Declared measurement-dependency pins (SPEC §7). `None` = OPEN, to be frozen
# at v3.0 lock; a None pin is itself an uncertified_reason. Observed versions
# come from env_fingerprint() — declared and observed are reported side by side.
# All checkpoints are staged under $LAB/cache/{huggingface,torch}/ (2026-07-06),
# so these pins stay runnable even if upstream hubs move.
PINS = {
    "dino_model": "facebook/dinov2-base",
    "dino_revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",  # staged HF snapshot
    "cotracker_hub": "facebookresearch/co-tracker:cotracker3_offline",
    # torch.hub ships gitless snapshots — pinned by content instead of commit:
    "cotracker_code_pysha": "868059fa2619b4ab",  # sha256-of-sorted-*.py-sha256s, staged hub dir
    "cotracker_ckpt_sha256": "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834",  # scaled_offline.pth
    "lpips_net": "alex",
    "judge_model": "gemini-3-flash-preview",  # model_version also recorded per response
    "feature_short_side": 256,
    "core_threshold": 0.5,
    "cross_high_threshold": 0.85,
    "tau_copy": 0.88,                 # DRAFT — recalibrated in certification
    "core_fallback_min_frames": 8,    # DRAFT — δ-expansion params set at lock
}

_ENV_PACKAGES = ("torch", "numpy", "transformers", "lpips", "google-genai", "opencv-python")


def _git(*args: str) -> str | None:
    """Run git in the repo containing this package; None on any failure."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def version() -> str:
    return VERSION_PATH.read_text().strip() if VERSION_PATH.exists() else "0.0.0-unversioned"


def is_draft(ver: str | None = None) -> bool:
    return "draft" in (ver or version())


def sha256_file(path: pathlib.Path) -> str | None:
    p = pathlib.Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def spec_sha() -> str | None:
    return sha256_file(SPEC_PATH)


def corpus_sha(manifest_path: str | pathlib.Path) -> str | None:
    """Content hash of a corpus manifest, canonicalized so formatting/key-order
    changes don't masquerade as corpus changes. Non-JSON files hash raw."""
    p = pathlib.Path(manifest_path)
    if not p.exists():
        return None
    raw = p.read_bytes()
    try:
        canonical = json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return hashlib.sha256(raw).hexdigest()


def git_state() -> dict:
    """Commit / branch / exact tag / instrument-scoped dirty state."""
    commit = _git("rev-parse", "HEAD")
    # Path-only queries — no status-letter parsing to corrupt (porcelain output
    # interacts badly with _git's strip()): tracked changes vs HEAD + untracked.
    changed = _git("diff", "--name-only", "HEAD", "--", *INSTRUMENT_PATHS) or ""
    untracked = _git("ls-files", "--others", "--exclude-standard", "--",
                     *INSTRUMENT_PATHS) or ""
    dirty_files = sorted({*changed.splitlines(), *untracked.splitlines()} - {""})
    tag = _git("describe", "--exact-match", "--tags", "HEAD")
    return {
        "commit": commit,
        "commit_short": commit[:12] if commit else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "exact_tag": tag,
        "package_dirty": bool(dirty_files),
        "dirty_files": dirty_files[:20],
    }


def env_fingerprint() -> dict:
    """Observed environment. Uses importlib.metadata only (no heavy imports)."""
    from importlib import metadata

    pkgs = {}
    for name in _ENV_PACKAGES:
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": pkgs}


def certification(state: dict, ver: str) -> tuple[bool, list[str]]:
    """A run is certified iff clean tree + exact eval/v<version> tag + non-draft
    version + no OPEN pins. Reasons are user-facing — keep them precise."""
    reasons = []
    if is_draft(ver):
        reasons.append(f"version {ver} is a draft")
    if state["package_dirty"]:
        reasons.append(f"instrument files modified since HEAD: {state['dirty_files']}")
    if state["commit"] is None:
        reasons.append("not a git checkout — provenance unavailable")
    expected = f"{TAG_PREFIX}{ver}"
    if state["exact_tag"] != expected:
        reasons.append(f"HEAD tag is {state['exact_tag']!r}, expected {expected!r}")
    open_pins = [k for k, v in PINS.items() if v is None]
    if open_pins:
        reasons.append(f"OPEN dependency pins: {open_pins}")
    return (not reasons), reasons


def stamp(corpus_manifest: str | pathlib.Path | None = None) -> dict:
    """The provenance stamp. Embed under results['provenance'] and in every
    items.jsonl row (scorers do this; nothing ships without it)."""
    ver = version()
    state = git_state()
    ok, reasons = certification(state, ver)
    return {
        "harness": f"transition-eval/{ver}",
        "version": ver,
        "certified": ok,
        "uncertified_reasons": reasons,
        "git": state,
        "spec_sha256": spec_sha(),
        "corpus_sha256": corpus_sha(corpus_manifest) if corpus_manifest else None,
        "pins": PINS,
        "env": env_fingerprint(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", help="corpus_manifest.json to hash into the stamp")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 unless the stamp is certified")
    args = ap.parse_args()
    s = stamp(args.corpus)
    print(json.dumps(s, indent=2))
    if args.check and not s["certified"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
