"""score — lifecycle phase 3 (SPEC §4/§8): the ONE v3 scorer. Replaces and
retires every experiment-dir run_score*.py fork.

Per item: S structure (sidedness-aware core) -> M1a appearance-to-reference,
M1b/M1c camera/object motion vs reference, M2a copy / M2b intrusion /
[M2c memorization if a training manifest is given], M3a endpoint fidelity,
M3b seams. Controls (lerp for two-sided, static-hold for one-sided /
prefix-only) are synthesized and scored through the identical pipeline as
arms. All raw — no floor/ceiling normalization exists in v3. Paired twin
deltas are the inferential unit; every row and results.json embeds
versioning.stamp() and renders UNCERTIFIED until the tag exists.

STATUS: end-to-end wiring complete; first GPU execution happens during the
v3 certification run (its warm/cold reruns double as the stability check).

Usage:
    python -m diffusion.transition_eval.score \
        --manifest eval_manifest.json --corpus corpus_manifest.json \
        [--training training_manifest.json] [--suite suite.json] --label L
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from . import versioning
from .endpoints import LpipsScorer, endpoint_fidelity, seam_scores, temporal_lpips
from .features import DinoExtractor
from .controls import make_lerp, make_static_hold
from .m1_transfer import appearance_ref, camera_match, camera_trajectory, object_match
from .m2_integrity import (TAU_COPY, copy_score, intrusion_margin,
                           memorization_score, mid_mask)
from .manifests_v3 import (completeness, derive_tier, load_corpus_manifest,
                           load_eval_manifest, load_training_manifest,
                           sidedness_of, tags_of)
from .motion import Tracker
from .pipeline import process_video, process_video_file
from .s_structure import core_mask_v3, structure_flags
from .video_io import load_frames

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SHORT_SIDE = versioning.PINS["feature_short_side"]


def _ref_bundle_cache(corpus: dict, cache_dir, extractor, tracker):
    """Process every corpus clip once: bundles + per-class core-feature pools
    (M2b needs ALL classes, not just the item's)."""
    root = REPO_ROOT / corpus["corpus_root"]
    bundles, pools = {}, {}
    for key in sorted(corpus["clips"]):
        cls = corpus["clips"][key]["class"]
        b, _ = process_video_file(root / key, cache_dir, extractor, tracker,
                                  short_side=SHORT_SIDE)
        mask, _meta = core_mask_v3(b.profile, corpus["classes"][cls]["sidedness"])
        bundles[key] = (b, mask)
        pools.setdefault(cls, []).append(b.feats[mask])
    pools = {c: np.concatenate(fs) for c, fs in pools.items()}
    return bundles, pools


def score_item(item, sidedness, gen_bundle, gen_frames, ref_bundle, ref_core,
               pools, lpips_scorer, extractor, training_pools=None):
    n_pre = item.condition_prefix.num_frames if item.condition_prefix else 9
    n_suf = item.condition_suffix.num_frames if item.condition_suffix else 0
    T = len(gen_frames)
    gcore, gmeta = core_mask_v3(gen_bundle.profile, sidedness)
    gmid = mid_mask(T, n_pre, n_suf)
    ref_cam = camera_trajectory(ref_bundle.tracks, ref_bundle.vis)
    gen_cam = camera_trajectory(gen_bundle.tracks, gen_bundle.vis)

    row = {
        "item_id": item.item_id, "arm": item.arm, "style": item.style,
        "twin_of": item.twin_of, "sidedness": sidedness,
        **structure_flags(gen_bundle.profile, gmeta),
        **{f"scalar_{k}": v for k, v in gen_bundle.scalars.items() if k != "hold"},
        # M1
        "app_ref": appearance_ref(gen_bundle.feats, gcore, ref_bundle.feats, ref_core),
        **camera_match(gen_cam, ref_cam),
        "obj_match": object_match(gen_bundle.tracks, gen_bundle.vis,
                                  ref_bundle.tracks, ref_bundle.vis, gen_cam, ref_cam),
        # M2
        **copy_score(gen_bundle.feats, gmid, ref_bundle.feats, ref_core, TAU_COPY),
        **intrusion_margin(gen_bundle.feats, gcore, pools, item.style),
    }
    if training_pools:
        row.update(memorization_score(gen_bundle.feats, gmid, training_pools))
    # M3
    if item.condition_prefix:
        pre, _ = load_frames(pathlib.Path(item.condition_prefix.video), short_side=SHORT_SIDE)
        row.update(endpoint_fidelity(gen_frames, gen_bundle.feats, pre[:n_pre],
                                     lambda f: extractor.extract(f), lpips_scorer, "prefix"))
    if item.condition_suffix:
        suf, _ = load_frames(pathlib.Path(item.condition_suffix.video), short_side=SHORT_SIDE)
        row.update(endpoint_fidelity(gen_frames, gen_bundle.feats, suf[-n_suf:],
                                     lambda f: extractor.extract(f), lpips_scorer, "suffix"))
    d = temporal_lpips(gen_frames, lpips_scorer)
    row.update(seam_scores(d, n_pre, max(n_suf, 1)))
    return row


def control_frames(item, sidedness, gen_len, short_side=SHORT_SIDE):
    """The degenerate-solution control for this item's contract (SPEC §4)."""
    pre, _ = load_frames(pathlib.Path(item.condition_prefix.video), short_side=short_side)
    pre = pre[: item.condition_prefix.num_frames]
    suf = None
    if item.condition_suffix:
        s, _ = load_frames(pathlib.Path(item.condition_suffix.video), short_side=short_side)
        suf = s[-item.condition_suffix.num_frames:]
    if sidedness == "twosided" and suf is not None:
        return make_lerp(pre, suf, gen_len), "control_lerp"
    return make_static_hold(pre, suf, gen_len), "control_hold"


def paired_table(rows):
    """Twin deltas — the inferential unit. metric: ic − base per twin pair."""
    by_id = {r["item_id"]: r for r in rows}
    out = []
    for r in rows:
        if r.get("twin_of") and r["twin_of"] in by_id:
            ic = by_id[r["twin_of"]]
            out.append({"pair": f"{r['twin_of']} vs {r['item_id']}",
                        **{f"d_{m}": (None if ic.get(m) is None or r.get(m) is None
                                      or not (np.isfinite(ic[m]) and np.isfinite(r[m]))
                                      else float(ic[m] - r[m]))
                           for m in ("app_ref", "copy_max", "margin", "max_seam_z",
                                     "prefix_dino", "suffix_dino")}})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--training")
    ap.add_argument("--suite", help="suite.json for completeness verification")
    ap.add_argument("--label", default="score_v3")
    ap.add_argument("--out-root", default="outputs/eval/v3")
    args = ap.parse_args()

    stamp = versioning.stamp(args.corpus)
    if not stamp["certified"]:
        print("[UNCERTIFIED] " + "; ".join(stamp["uncertified_reasons"]))

    corpus = load_corpus_manifest(args.corpus)
    training = load_training_manifest(args.training) if args.training else None
    items = load_eval_manifest(args.manifest)

    out_dir = REPO_ROOT / args.out_root / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = REPO_ROOT / "outputs/eval/cache"

    device = "cuda"
    extractor = DinoExtractor(versioning.PINS["dino_model"], device=device)
    tracker = Tracker(device=device)
    lpips_scorer = LpipsScorer(device=device)

    ref_bundles, pools = _ref_bundle_cache(corpus, cache_dir, extractor, tracker)
    training_pools = None
    if training:
        training_pools = {k: ref_bundles[k][0].feats
                          for k in training["_clipset"] if k in ref_bundles}

    rows = []
    for it in items:
        side = sidedness_of(it.style, corpus)
        ref_key = "/".join(pathlib.Path(it.reference_video).parts[-2:])
        if ref_key not in ref_bundles:
            raise ValueError(f"{it.item_id}: reference {ref_key} not in corpus manifest")
        rb, rcore = ref_bundles[ref_key]
        gb, gframes = process_video_file(pathlib.Path(it.generated_video),
                                         cache_dir, extractor, tracker,
                                         short_side=SHORT_SIDE)
        row = score_item(it, side, gb, gframes, rb, rcore, pools,
                         lpips_scorer, extractor, training_pools)
        row["tier"] = derive_tier(it, corpus, training)
        row["tags"] = tags_of(it.style, corpus)
        row["provenance"] = {"harness": stamp["harness"], "certified": stamp["certified"]}
        rows.append(row)

        if it.condition_prefix:  # control arm through the identical pipeline
            cframes, cname = control_frames(it, side, len(gframes))
            cb = process_video(cframes, gb.key + f":{cname}", cache_dir,
                               extractor, tracker)
            crow = score_item(it, side, cb, cframes, rb, rcore, pools,
                              lpips_scorer, extractor, None)
            crow.update({"item_id": f"{cname}__{it.item_id}", "arm": cname,
                         "twin_of": None, "provenance": row["provenance"]})
            rows.append(crow)

    extractor.free(); tracker.free(); lpips_scorer.free()

    with open(out_dir / "items.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    results = {"provenance": stamp,
               "completeness": (completeness(json.loads(pathlib.Path(args.suite).read_text())["items"],
                                             {r["item_id"] for r in rows})
                                if args.suite else None),
               "paired": paired_table(rows),
               "n_items": len(rows)}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[done] {len(rows)} rows -> {out_dir}"
          + ("" if stamp["certified"] else "  [UNCERTIFIED]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
