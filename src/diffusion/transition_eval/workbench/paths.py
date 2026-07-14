"""Absolute paths for the workbench (OPERATIONS.md §2).

Cache keys are sha1(abspath|mtime_ns|size|model|short_side)-derived, so the
corpus MUST be read through $CORPUS_ROOT — copying videos or reading them from
another checkout silently misses the 952 MB warm cache and forces a full GPU
re-extraction. The class dirs under $CORPUS_ROOT are symlinks into the
data-owning checkout, so Path.resolve() canonicalizes to the same target the
certification run keyed against, from any worktree.

READ-ONLY (never write, never touch mtimes): EV, CORPUS_ROOT, SHARED_CACHE,
BASELINE_DIR, RECORD_DIR. Writable: WB_CACHE, WB_OUT.
"""

from __future__ import annotations

import json
import pathlib

LAB = pathlib.Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")

WB = LAB / "diffusion-research/.claude/worktrees/metric-workbench"
EV = LAB / "diffusion-research/.claude/worktrees/eval-v3-spec"

CORPUS_ROOT = EV / "data/processed/transitions_std121"          # RO — video paths
SHARED_CACHE = EV / "outputs/eval/cache"                        # RO — warm dino_arr_*
BASELINE_DIR = EV / "outputs/eval/certification/3.0.0-draft.8"  # RO — analysis/ + exam/
RECORD_DIR = EV / "outputs/eval/certification/3.0.0"            # RO — regrade record

WB_CACHE = WB / "outputs/eval/workbench_cache"                  # RW — flow_*/null_*/zca
WB_OUT = WB / "outputs/eval/workbench"                          # RW — per-rung outputs

CORPUS_MANIFEST = CORPUS_ROOT / "corpus_manifest.json"
NPZ = BASELINE_DIR / "analysis/distance_matrices.npz"
ANALYSIS_JSON = BASELINE_DIR / "analysis/analysis.json"

# Pins the warm cache was keyed against (versioning.PINS; asserted in baselines).
DINO_MODEL = "facebook/dinov2-base"
FEATURE_SHORT_SIDE = 256


def load_corpus() -> dict:
    return json.loads(CORPUS_MANIFEST.read_text())


def corpus_keys(corpus: dict) -> list[str]:
    """The row order every distance matrix must share (exam.run_exam's order)."""
    return sorted(corpus["clips"])


def clip_path(key: str) -> pathlib.Path:
    """Video path for a corpus key — always through CORPUS_ROOT (cache keying)."""
    return CORPUS_ROOT / key


def labels_of(corpus: dict, keys: list[str]) -> list[str]:
    return [corpus["clips"][k]["class"] for k in keys]


def sidedness_of(corpus: dict, keys: list[str]) -> list[str]:
    return [corpus["classes"][corpus["clips"][k]["class"]]["sidedness"] for k in keys]
