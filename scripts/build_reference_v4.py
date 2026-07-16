"""Build the v4 corpus-reference artifact (reference_v4.npz) — a pinned
instrument constant (SPEC §4/§7). One-time build from the pinned corpus via the
committed builder; commit the artifact and set versioning.PINS['reference_v4_sha256'].

    PYTHONPATH=src python scripts/build_reference_v4.py --corpus <corpus_manifest>

The artifact holds mu, the 9 ECDF populations (P1/P2/V1/V1e/App/Dyn/Z/P/R), the
CSLS neighborhood means r_obj[223], and k/rgrid/weight constants — everything a
single new (gen, ref) pair needs to be ranked against the corpus at score time.

Determinism: the build reuses the certified warm cache. The seven value-space
populations rebuild within reference.value_tol (1e-6) across environments; the two
ECDF-COMPOSED rank-lattice populations (pop_App, pop_Dyn) are graded in integer rank
units under the two-class rebuild-parity criterion (bars.yaml `reference:`, SPEC §6.4),
because a scalar float tolerance below their 1/(2N) quantum is unsatisfiable under any
cross-environment rebuild (float32-reduction order varies by CPU/BLAS). Run on the SAME
warm cache score.py uses; never seed CPU LPIPS.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from diffusion.transition_eval import reference_stats as RS
from diffusion.transition_eval import versioning
from diffusion.transition_eval.certify import exam as cert_exam
from diffusion.transition_eval.manifests_v3 import load_corpus_manifest
from diffusion.transition_eval.pipeline import process_video_file
from diffusion.transition_eval.s_structure import core_mask_v3

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cache-dir", default="outputs/eval/cache")
    ap.add_argument("--out", default=str(RS.REFERENCE_PATH))
    args = ap.parse_args()

    corpus = load_corpus_manifest(args.corpus)
    corpus_sha = versioning.corpus_sha(args.corpus)
    keys = sorted(corpus["clips"])
    labels = [corpus["clips"][k]["class"] for k in keys]
    sidedness = [corpus["classes"][l]["sidedness"] for l in labels]
    root = REPO_ROOT / corpus["corpus_root"]
    cache_dir = REPO_ROOT / args.cache_dir

    import torch
    from diffusion.transition_eval.features import DinoExtractor
    from diffusion.transition_eval.motion import Tracker
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[build_ref] device={device}; {len(keys)} corpus clips", flush=True)
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

    # reuse the exam's headline builder so the artifact and the exam share ONE
    # build path (the exam's rebuild-parity check then compares them at cert time)
    hl = cert_exam.v4_headline_matrices(bundles, sidedness, n_jobs=16)
    ref = RS.build_reference_from_parts(hl["channels"], hl["views"], hl["object_D"],
                                        keys, corpus_sha)
    out = pathlib.Path(args.out)
    sha = RS.save_reference(ref, out)
    print(f"\n[build_ref] wrote {out}", flush=True)
    print(f"[build_ref] corpus_sha256 = {corpus_sha}", flush=True)
    print(f"[build_ref] artifact sha256 = {sha}", flush=True)
    print(f"[build_ref] SET versioning.PINS['reference_v4_sha256'] = \"{sha}\"", flush=True)
    # sanity: retrieval headlines on the built matrices
    for nm, D in (("S3", hl["S3"]), ("D_ZPR", hl["D_ZPR"]), ("CSLS", hl["CSLS"])):
        print(f"  [{nm}] finite pop = {len(RS.population(D))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
