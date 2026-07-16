"""bar8 rebuild-parity PROVENANCE check (advisor step 1).

Establishes, on a single compute node with the warm cache:
  (a) on-node self-reproducibility: two cert-path rebuilds (n_jobs=8) bit-identical?
  (b) code-path equivalence: build-script path (n_jobs=16) == cert path (n_jobs=8)?
  (c) flip-counts vs committed reference_v4.npz for the lattice channels App/Dyn.

Mirrors scripts/build_reference_v4.py and run_certification.py exactly
(v4_headline_matrices -> build_reference_from_parts). No kernel edits.
Run from the worktree repo root with PYTHONPATH=src.
"""
from __future__ import annotations
import pathlib, platform, subprocess, sys
import numpy as np

from diffusion.transition_eval import reference_stats as RS
from diffusion.transition_eval import versioning
from diffusion.transition_eval.certify import exam as cert_exam
from diffusion.transition_eval.manifests_v3 import load_corpus_manifest
from diffusion.transition_eval.pipeline import process_video_file

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(REPO_ROOT / "src"))
CORPUS = "data/processed/transitions_std121/corpus_manifest.json"
CACHE = "outputs/eval/cache"

VALUE_ARRAYS = ["mu", "pop_P1", "pop_P2", "pop_V1", "pop_V1e", "pop_Z", "pop_P",
                "pop_R", "r_obj", "k_csls", "rgrid", "s3_app_weight"]
LATTICE_ARRAYS = ["pop_App", "pop_Dyn"]


def env_banner():
    print("=" * 70)
    print("NODE:", platform.node())
    try:
        cpu = subprocess.check_output(
            "grep -m1 'model name' /proc/cpuinfo", shell=True).decode().strip()
        print("CPU:", cpu)
    except Exception as e:
        print("CPU: (unknown)", e)
    print("python:", platform.python_version(), "| numpy:", np.__version__)
    try:
        import scipy
        print("scipy:", scipy.__version__)
    except Exception:
        pass
    print("--- numpy BLAS ---")
    try:
        np.show_config()
    except Exception as e:
        print("(show_config failed)", e)
    print("=" * 70, flush=True)


def build_bundles():
    corpus = load_corpus_manifest(CORPUS)
    corpus_sha = versioning.corpus_sha(CORPUS)
    keys = sorted(corpus["clips"])
    labels = [corpus["clips"][k]["class"] for k in keys]
    sidedness = [corpus["classes"][l]["sidedness"] for l in labels]
    root = REPO_ROOT / corpus["corpus_root"]
    cache_dir = REPO_ROOT / CACHE
    import torch
    from diffusion.transition_eval.features import DinoExtractor
    from diffusion.transition_eval.motion import Tracker
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[prov] device={device}; {len(keys)} clips; corpus_sha={corpus_sha[:16]}", flush=True)
    extractor = DinoExtractor(versioning.PINS["dino_model"], device=device)
    tracker = Tracker(device=device)
    bundles = []
    for i, k in enumerate(keys):
        b, _ = process_video_file(root / k, cache_dir, extractor, tracker,
                                  short_side=versioning.PINS["feature_short_side"],
                                  need_frames=False)
        bundles.append({"feats": b.feats, "tracks": b.tracks, "vis": b.vis,
                        "profile": b.profile})
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(keys)}", flush=True)
    extractor.free(); tracker.free()
    return bundles, keys, sidedness, corpus_sha


def rebuild(bundles, sidedness, keys, corpus_sha, n_jobs):
    hl = cert_exam.v4_headline_matrices(bundles, sidedness, n_jobs=n_jobs)
    return RS.build_reference_from_parts(hl["channels"], hl["views"], hl["object_D"],
                                         keys, corpus_sha)


def max_abs_delta(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def bit_identical(refA, refB):
    keys = sorted(set(refA) | set(refB))
    diffs = {}
    for k in keys:
        if k in ("keys", "corpus_sha"):
            continue
        a, b = refA.get(k), refB.get(k)
        if a is None or b is None:
            diffs[k] = "missing"; continue
        a = np.asarray(a); b = np.asarray(b)
        if a.shape != b.shape:
            diffs[k] = f"shape {a.shape}!={b.shape}"; continue
        if not np.array_equal(a, b):
            diffs[k] = f"max|Δ|={max_abs_delta(a, b):.3e}"
    return diffs


def flip_report(fresh_pop, committed_pop, name):
    N = len(fresh_pop)
    assert len(committed_pop) == N, f"{name}: length mismatch"
    uf = np.rint(np.asarray(fresh_pop, dtype=np.float64) * 2 * N)
    uc = np.rint(np.asarray(committed_pop, dtype=np.float64) * 2 * N)
    lat_fresh = float(np.max(np.abs(np.asarray(fresh_pop) * 2 * N - uf)))
    lat_comm = float(np.max(np.abs(np.asarray(committed_pop) * 2 * N - uc)))
    du = np.abs(uf - uc)
    n_diff = int(np.count_nonzero(du > 0))
    max_du = int(du.max()) if N else 0
    print(f"  [{name}] N={N}  value max|Δ|={max_abs_delta(fresh_pop, committed_pop):.4e}")
    print(f"    lattice sanity (max|pop*2N - round|): fresh={lat_fresh:.3e} committed={lat_comm:.3e}")
    print(f"    FLIP-COUNT (entries with different lattice unit) = {n_diff} / {N}  ({100*n_diff/N:.4f}%)")
    print(f"    max |Δ lattice steps| = {max_du}", flush=True)
    return {"N": N, "n_diff": n_diff, "max_du": max_du,
            "lat_fresh": lat_fresh, "lat_comm": lat_comm}


def main():
    env_banner()
    bundles, keys, sidedness, corpus_sha = build_bundles()

    print("\n[prov] rebuild #1 (cert path, n_jobs=8) ...", flush=True)
    R8a = rebuild(bundles, sidedness, keys, corpus_sha, 8)
    print("[prov] rebuild #2 (cert path, n_jobs=8) ...", flush=True)
    R8b = rebuild(bundles, sidedness, keys, corpus_sha, 8)
    print("[prov] rebuild #3 (build-script path, n_jobs=16) ...", flush=True)
    R16 = rebuild(bundles, sidedness, keys, corpus_sha, 16)

    committed = RS.load_reference(expect_corpus_sha=corpus_sha)
    sha = versioning.sha256_file(REPO_ROOT / "src/diffusion/transition_eval/reference_v4.npz")
    print(f"\n[prov] committed artifact sha256 = {sha}")
    print(f"[prov] pin reference_v4_sha256    = {versioning.PINS['reference_v4_sha256']}")
    print(f"[prov] sha match = {sha == versioning.PINS['reference_v4_sha256']}", flush=True)

    print("\n=== (a) ON-NODE SELF-REPRODUCIBILITY: R8a vs R8b (must be bit-identical) ===")
    d_self = bit_identical(R8a, R8b)
    print("  bit-identical" if not d_self else f"  DIFFERENCES: {d_self}")

    print("\n=== (b) CODE-PATH / n_jobs: R8a vs R16 (must be bit-identical) ===")
    d_path = bit_identical(R8a, R16)
    print("  bit-identical" if not d_path else f"  DIFFERENCES: {d_path}")

    print("\n=== R8a vs COMMITTED: per-array max|Δ| ===")
    for k in VALUE_ARRAYS:
        if k in R8a and k in committed:
            print(f"  {k}: {max_abs_delta(R8a[k], committed[k]):.4e}")
    print("\n=== LATTICE CHANNELS flip-count (R8a vs COMMITTED) ===")
    flips = {}
    for k in LATTICE_ARRAYS:
        flips[k] = flip_report(R8a[k], committed[k], k)

    print("\n=== SUMMARY ===")
    print(f"self_repro_bit_identical = {not d_self}")
    print(f"code_path_bit_identical  = {not d_path}")
    for k in LATTICE_ARRAYS:
        print(f"{k}: flips={flips[k]['n_diff']}  max_step={flips[k]['max_du']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
