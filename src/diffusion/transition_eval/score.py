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
from .endpoints import (LPIPS_CACHE_TAG, LpipsScorer, cached_temporal_lpips,
                        endpoint_fidelity, lpips_cache_path, seam_scores)
from .features import DinoExtractor, file_key
from .controls import make_lerp, make_static_hold
from .m1_transfer import (appearance_ref, appearance_s3, camera_match,
                          camera_trajectory, camera_zpr, object_csls,
                          object_match_from_profiles, residual_direction_profile)
from .reference_stats import load_reference
from .m2_integrity import (TAU_COPY, copy_score, intrusion_margin,
                           memorization_score, mid_mask)
from .manifests_v3 import (completeness, derive_tier, load_corpus_manifest,
                           load_eval_manifest, load_training_manifest,
                           sidedness_of, tags_of)
from .motion import Tracker
from .pipeline import process_video, process_video_file
from .s_structure import core_mask_v3, structure_flags
from .video_io import load_frames

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]  # src/diffusion/transition_eval/ -> repo
SHORT_SIDE = versioning.PINS["feature_short_side"]


def _ref_bundle_cache(corpus: dict, cache_dir, extractor, tracker):
    """Process every corpus clip once: bundles + per-class core-feature pools
    (M2b needs ALL classes, not just the item's)."""
    root = REPO_ROOT / corpus["corpus_root"]
    bundles, pools = {}, {}
    for key in sorted(corpus["clips"]):
        cls = corpus["clips"][key]["class"]
        b, _ = process_video_file(root / key, cache_dir, extractor, tracker,
                                  short_side=SHORT_SIDE, need_frames=False)
        mask, _meta = core_mask_v3(b.profile, corpus["classes"][cls]["sidedness"])
        bundles[key] = (b, mask)
        pools.setdefault(cls, []).append(b.feats[mask])
    pools = {c: np.concatenate(fs) for c, fs in pools.items()}
    return bundles, pools


def _corpus_v4_pack(corpus: dict, bundles: dict) -> dict:
    """Per-corpus-clip v4 precomputations, once per run: camera fits + residual-
    direction profiles in sorted key order (the reference artifact's row order).
    M1c's CSLS neighborhood term needs every item's similarities to all 223
    corpus clips; profiles make that 223 cheap correlations per item."""
    keys = sorted(corpus["clips"])
    cams = {k: camera_trajectory(bundles[k][0].tracks, bundles[k][0].vis)
            for k in keys}
    profiles = [residual_direction_profile(bundles[k][0].tracks,
                                           bundles[k][0].vis, cams[k])
                for k in keys]
    return {"keys": keys, "cams": cams, "profiles": profiles}


def _endpoint_key(gen_key: str, side: str, cond, short_side: int) -> str:
    """Cache key for one endpoint_fidelity call: generated-video identity +
    condition-clip identity (stat-based) + slice length + LPIPS tag."""
    ck = file_key(pathlib.Path(cond.video), str(short_side))
    return f"{gen_key}:endp:{side}:{ck}:{cond.num_frames}:{LPIPS_CACHE_TAG}"


def _cached_endpoint(item, side, n, gen_bundle, gen_frames, extractor,
                     lpips_scorer, cache_dir, short_side=SHORT_SIDE):
    """endpoint_fidelity through the LPIPS cache; a hit also skips decoding the
    condition clip. cache_dir=None disables caching (always compute fresh)."""
    cond = item.condition_prefix if side == "prefix" else item.condition_suffix
    p = (lpips_cache_path(_endpoint_key(gen_bundle.key, side, cond, short_side),
                          cache_dir) if cache_dir is not None else None)
    if p is not None and p.exists():
        z = np.load(p)
        return {k: float(z[k]) for k in z.files}
    if gen_frames is None:
        raise RuntimeError(f"endpoint cache miss for {item.item_id}:{side} "
                           "but no frames were decoded")
    frames, _ = load_frames(pathlib.Path(cond.video), short_side=short_side)
    sl = frames[:n] if side == "prefix" else frames[-n:]
    out = endpoint_fidelity(gen_frames, gen_bundle.feats, sl,
                            lambda f: extractor.extract(f), lpips_scorer, side)
    if p is not None:
        np.savez_compressed(p, **out)
    return out


def lpips_warm(item, gen_key: str, cache_dir: pathlib.Path | None,
               short_side: int = SHORT_SIDE) -> bool:
    """True iff every LPIPS quantity score_item needs for this item is cached —
    only then may the generated video's decode be skipped."""
    if cache_dir is None:
        return False
    if not lpips_cache_path(f"{gen_key}:tlpips:{LPIPS_CACHE_TAG}", cache_dir).exists():
        return False
    for side in ("prefix", "suffix"):
        cond = getattr(item, f"condition_{side}")
        if cond and not lpips_cache_path(
                _endpoint_key(gen_key, side, cond, short_side), cache_dir).exists():
            return False
    return True


def score_item(item, sidedness, gen_bundle, gen_frames, ref_bundle, ref_core,
               pools, lpips_scorer, extractor, ref_key, v4pack,
               training_pools=None, lpips_cache_dir=None):
    n_pre = item.condition_prefix.num_frames if item.condition_prefix else 9
    n_suf = item.condition_suffix.num_frames if item.condition_suffix else 0
    T = len(gen_bundle.feats)
    gcore, gmeta = core_mask_v3(gen_bundle.profile, sidedness)
    gmid = mid_mask(T, n_pre, n_suf)
    ref_cam = v4pack["cams"][ref_key]
    gen_cam = camera_trajectory(gen_bundle.tracks, gen_bundle.vis)

    # v4 M1c: the gen clip's residual profile vs the full reference corpus
    gen_res = residual_direction_profile(gen_bundle.tracks, gen_bundle.vis, gen_cam)
    sims_corpus = np.array([object_match_from_profiles(gen_res, pj)
                            for pj in v4pack["profiles"]])
    ref_idx = v4pack["keys"].index(ref_key)
    prof_g, prof_r = gen_bundle.profile, ref_bundle.profile

    row = {
        "item_id": item.item_id, "arm": item.arm, "style": item.style,
        "twin_of": item.twin_of, "sidedness": sidedness,
        **structure_flags(gen_bundle.profile, gmeta),
        **{f"scalar_{k}": v for k, v in gen_bundle.scalars.items() if k != "hold"},
        # M1 — v4 headline metrics (SPEC §3; corpus-relative, reference_v4-ranked)
        **appearance_s3(gen_bundle.feats, gcore, prof_g["n_prefix"], prof_g["n_suffix"],
                        ref_bundle.feats, ref_core, prof_r["n_prefix"], prof_r["n_suffix"],
                        v4pack["ref_stats"]),
        **camera_zpr(gen_bundle.tracks, gen_bundle.vis, gen_cam,
                     ref_bundle.tracks, ref_bundle.vis, ref_cam, v4pack["ref_stats"]),
        **object_csls(float(sims_corpus[ref_idx]), sims_corpus, ref_idx,
                      v4pack["ref_stats"]),
        # M1 — v3 analysis/bridge fields (raw substrate statistics, ungated)
        "app_ref_v3": appearance_ref(gen_bundle.feats, gcore, ref_bundle.feats, ref_core),
        **camera_match(gen_cam, ref_cam),
        "obj_match": float(sims_corpus[ref_idx]),
        # M2
        **copy_score(gen_bundle.feats, gmid, ref_bundle.feats, ref_core, TAU_COPY),
        **intrusion_margin(gen_bundle.feats, gcore, pools, item.style),
    }
    if training_pools:
        row.update(memorization_score(gen_bundle.feats, gmid, training_pools))
    # M3
    if item.condition_prefix:
        row.update(_cached_endpoint(item, "prefix", n_pre, gen_bundle, gen_frames,
                                    extractor, lpips_scorer, lpips_cache_dir))
    if item.condition_suffix:
        row.update(_cached_endpoint(item, "suffix", n_suf, gen_bundle, gen_frames,
                                    extractor, lpips_scorer, lpips_cache_dir))
    d = cached_temporal_lpips(gen_frames, gen_bundle.key, lpips_cache_dir, lpips_scorer)
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
                           for m in ("app_ref", "cam_zpr", "obj_csls", "copy_max",
                                     "margin", "max_seam_z",
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
    ap.add_argument("--controls", choices=("auto", "off"), default="auto",
                    help="auto: synthesize the degenerate control arm per item "
                         "(SPEC §4); off: skip (probe suites that carry no "
                         "floor claim, e.g. splices/swaps/hard-cuts)")
    ap.add_argument("--cache-dir", default="outputs/eval/cache",
                    help="feature/track cache (stability's cold-anchor rerun "
                         "points this at a fresh directory)")
    ap.add_argument("--lpips-cache", choices=("on", "off"), default="on",
                    help="cache temporal/endpoint LPIPS in --cache-dir keyed by "
                         "stat-based video identity (numeric no-op: a miss "
                         "computes exactly what uncached code computed); off "
                         "never reads or writes it — certification's warm rerun "
                         "uses off so bar 8 keeps recomputing LPIPS end-to-end")
    args = ap.parse_args()

    stamp = versioning.stamp(args.corpus)
    if not stamp["certified"]:
        print("[UNCERTIFIED] " + "; ".join(stamp["uncertified_reasons"]))

    corpus = load_corpus_manifest(args.corpus)
    training = load_training_manifest(args.training) if args.training else None
    items = load_eval_manifest(args.manifest)

    out_dir = REPO_ROOT / args.out_root / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = REPO_ROOT / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = DinoExtractor(versioning.PINS["dino_model"], device=device)
    tracker = Tracker(device=device)
    lpips_scorer = LpipsScorer(device=device)

    lpips_cache_dir = cache_dir if args.lpips_cache == "on" else None
    ref_bundles, pools = _ref_bundle_cache(corpus, cache_dir, extractor, tracker)
    # v4: the frozen reference artifact (pinned instrument constant) + per-run
    # corpus precomputations. Corpus mismatch refuses loudly (SPEC §4/§7).
    ref_stats = load_reference(expect_corpus_sha=stamp["corpus_sha256"])
    v4pack = _corpus_v4_pack(corpus, ref_bundles)
    assert [str(k) for k in ref_stats["keys"]] == v4pack["keys"], \
        "reference_v4 artifact key order != corpus manifest key order"
    v4pack["ref_stats"] = ref_stats
    training_pools = None
    if training:
        training_pools = {k: ref_bundles[k][0].feats
                          for k in training["_clipset"] if k in ref_bundles}

    rows, n_errors = [], 0
    for it in items:
        # per-item isolation: one bad item yields an ERROR ROW, never a dead
        # stage (draft.7 lesson — a single crash killed three bars' data).
        # Error rows carry no metric keys, so graders count them as documented
        # misses; certification counts them into bar 8's no-crash clause.
        try:
            side = sidedness_of(it.style, corpus)
            ref_key = "/".join(pathlib.Path(it.reference_video).parts[-2:])
            if ref_key not in ref_bundles:
                raise ValueError(f"reference {ref_key} not in corpus manifest")
            rb, rcore = ref_bundles[ref_key]
            gpath = pathlib.Path(it.generated_video)
            gkey = file_key(gpath, extractor.model_name, str(SHORT_SIDE))
            # decode only if some consumer needs pixels: feature/track cache
            # misses decode inside process_video_file regardless; LPIPS misses
            # are pre-checked here (numeric no-op — a miss always decodes)
            gb, gframes = process_video_file(
                gpath, cache_dir, extractor, tracker, short_side=SHORT_SIDE,
                need_frames=not lpips_warm(it, gkey, lpips_cache_dir))
            row = score_item(it, side, gb, gframes, rb, rcore, pools,
                             lpips_scorer, extractor, ref_key, v4pack,
                             training_pools, lpips_cache_dir=lpips_cache_dir)
            row["tier"] = derive_tier(it, corpus, training)
            row["tags"] = tags_of(it.style, corpus)
            row["provenance"] = {"harness": stamp["harness"], "certified": stamp["certified"]}
            rows.append(row)

            if args.controls == "auto" and it.condition_prefix:  # control arm through the identical pipeline
                cframes, cname = control_frames(it, side, len(gb.feats))
                cb = process_video(cframes, gb.key + f":{cname}", cache_dir,
                                   extractor, tracker)
                crow = score_item(it, side, cb, cframes, rb, rcore, pools,
                                  lpips_scorer, extractor, ref_key, v4pack,
                                  None, lpips_cache_dir=lpips_cache_dir)
                crow.update({"item_id": f"{cname}__{it.item_id}", "arm": cname,
                             "twin_of": None, "provenance": row["provenance"]})
                rows.append(crow)
        except Exception as e:  # noqa: BLE001 — isolation is the point; the row is loud
            n_errors += 1
            rows.append({"item_id": it.item_id, "arm": it.arm, "style": it.style,
                         "twin_of": it.twin_of,
                         "error": f"{type(e).__name__}: {e}",
                         "provenance": {"harness": stamp["harness"],
                                        "certified": stamp["certified"]}})
            print(f"[error-row] {it.item_id}: {type(e).__name__}: {e}", flush=True)

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
    print(f"[done] {len(rows)} rows ({n_errors} error rows) -> {out_dir}"
          + ("" if stamp["certified"] else "  [UNCERTIFIED]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
