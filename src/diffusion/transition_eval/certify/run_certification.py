"""Certification driver — SPEC §6.5 as code: freeze happened in its own commit,
then this runs A -> B -> C -> D mechanically (no human decision between freeze
and record) and writes certifications/v<version>.md + artifacts.

    PYTHONPATH=src python -m diffusion.transition_eval.certify.run_certification \
        --corpus data/processed/transitions_std121/corpus_manifest.json \
        --main-root /path/to/checkout-that-owns-the-archives

Requires: bars.yaml frozen; cwd = repo root (probe manifests carry repo-relative
paths); corpus clips resolvable under <repo>/<corpus_root> (worktrees symlink
the class dirs); GPU for uncached featurization (falls back to CPU, slowly).
sigma_seed (O6) needs model inference and is recorded PENDING — it gates the
first model report, not the tag (SPEC §6.4).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import yaml

from .. import versioning
from ..manifests_v3 import load_corpus_manifest
from ..m1_transfer import camera_trajectory
from ..pipeline import process_video_file
from . import blockc, exam, probes
from .stability import compare_runs

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]  # .../certify/ -> repo
BARS_PATH = pathlib.Path(__file__).parent / "bars.yaml"

CLAIMS = ("the instrument separates the styles it has seen (per-class trust map "
          "attached), refuses known-degenerate and copied inputs, is direction-"
          "sensitive on motion, runs deterministically end-to-end on real "
          "generations, and its blind spots are enumerated.")


def log(msg: str) -> None:
    print(f"[certify] {msg}", flush=True)


def run_score(manifest: pathlib.Path, label: str, out_root: pathlib.Path,
              controls: str, cache_dir: str | None = None) -> pathlib.Path:
    """score.py through its real CLI — certification exercises the shipped
    entrypoint, not a private shim. Returns the items.jsonl path."""
    cmd = [sys.executable, "-m", "diffusion.transition_eval.score",
           "--manifest", str(manifest),
           "--corpus", str(CORPUS_PATH),
           "--label", label, "--out-root", str(out_root),
           "--controls", controls]
    if cache_dir:
        cmd += ["--cache-dir", cache_dir]
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    log(f"score: {label} ({controls=})")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)
    return REPO_ROOT / out_root / label / "items.jsonl"


def load_rows(items_jsonl: pathlib.Path) -> dict[str, dict]:
    return {r["item_id"]: r for r in
            (json.loads(l) for l in items_jsonl.read_text().splitlines() if l.strip())}


def anchor_ids(pairs_by_class: dict, corpus: dict, c_items: list[dict],
               bars: dict) -> list[str]:
    """Frozen anchor RULE (bars.stability.anchors.rule): deterministic picks,
    no cherry-picking surface."""
    def first(pred):
        return next((c for c in sorted(pairs_by_class)
                     if pred(corpus["classes"][c])), None)
    ts = first(lambda v: v["sidedness"] == "twosided" and v["n_clips"] >= 4)
    os_ = first(lambda v: v["sidedness"] == "onesided" and v["n_clips"] >= 4)
    cam = first(lambda v: "camera" in v.get("tags", []) and v["n_clips"] >= 4)
    e57_base = sorted(i["item_id"] for i in c_items if i["arm"].startswith("base"))
    e57_ic = sorted(i["item_id"] for i in c_items if i["arm"].startswith("ic"))
    ids = [f"sib__{ts}", f"sib__{os_}", f"sib__{cam}",
           f"control_lerp__sib__{ts}", e57_base[0], e57_ic[0]]
    assert len(set(ids)) == bars["stability"]["bar8"]["anchors"]["n"], ids
    return ids


def main() -> int:
    global CORPUS_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--main-root", required=True,
                    help="checkout that owns outputs/ archives + experiment condition clips")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bars = yaml.safe_load(BARS_PATH.read_text())
    if not bars.get("frozen"):
        raise RuntimeError("bars.yaml is not frozen — SPEC §6.5 forbids running")
    if pathlib.Path.cwd().resolve() != REPO_ROOT.resolve():
        raise RuntimeError(f"run from the repo root {REPO_ROOT} (relative probe paths)")

    CORPUS_PATH = pathlib.Path(args.corpus).resolve()
    corpus = load_corpus_manifest(CORPUS_PATH)
    ver = versioning.version()
    out = pathlib.Path(args.out or REPO_ROOT / "outputs/eval/certification" / ver)
    out.mkdir(parents=True, exist_ok=True)
    main_root = pathlib.Path(args.main_root)
    stamp = versioning.stamp(CORPUS_PATH)
    (out / "stamp.json").write_text(json.dumps(stamp, indent=1))

    probe_root = corpus["corpus_root"]
    first_clip = sorted(corpus["clips"])[0]
    if not (REPO_ROOT / probe_root / first_clip).exists():
        raise RuntimeError(f"corpus clips not resolvable at {REPO_ROOT / probe_root} "
                           "— symlink the class dirs from the data-owning checkout")

    # ---- corpus bundles (cached pipeline; GPU only on cache miss) -------------------
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}; processing {len(corpus['clips'])} corpus clips")
    from ..endpoints import LpipsScorer  # noqa: F401  (imported for env parity)
    from ..features import DinoExtractor
    from ..motion import Tracker
    extractor = DinoExtractor(versioning.PINS["dino_model"], device=device)
    tracker = Tracker(device=device)
    cache_dir = REPO_ROOT / "outputs/eval/cache"

    keys = sorted(corpus["clips"])
    bundles_by_key: dict[str, dict] = {}
    for i, key in enumerate(keys):
        b, _ = process_video_file(REPO_ROOT / probe_root / key, cache_dir,
                                  extractor, tracker,
                                  short_side=versioning.PINS["feature_short_side"])
        bundles_by_key[key] = b
        if (i + 1) % 40 == 0:
            log(f"  corpus {i + 1}/{len(keys)}")
    labels = [corpus["clips"][k]["class"] for k in keys]
    sidedness = [corpus["classes"][l]["sidedness"] for l in labels]
    bundles = [bundles_by_key[k] for k in keys]

    # ---- Block A ---------------------------------------------------------------------
    log("Block A: exam (R1 + R2 + adoption)")
    exam_res = exam.run_exam(bundles, labels, sidedness, corpus, bars, out / "exam")
    mask_w = exam_res["mask_adoption"]["winner"]
    log(f"  mask winner={mask_w}; motion winner={exam_res['motion_adoption']['winner']}; "
        f"bar1 pass={exam_res['bar1']['pass']}")

    # ---- Block B build ----------------------------------------------------------------
    log("Block B: probe construction")
    pairs = probes.sibling_pairs(bundles_by_key, corpus)
    classes = sorted(pairs)
    audit = probes.content_invariance_audit(bundles_by_key, corpus, pairs)
    (out / "content_invariance.json").write_text(json.dumps(audit, indent=1))

    probe_dir = out / "probe_videos"
    man_dir = out / "manifests"
    man_dir.mkdir(exist_ok=True)
    sib_man = probes.build_sibling_manifest(pairs, corpus)
    other_man = (probes.build_splice_manifests(pairs, corpus, bundles_by_key,
                                               probe_dir, REPO_ROOT, bars)
                 + probes.build_swap_manifest(pairs, corpus)
                 + probes.build_hardcut_manifests(pairs, corpus, probe_dir, REPO_ROOT))
    (man_dir / "siblings.json").write_text(json.dumps(sib_man, indent=1))
    (man_dir / "probes.json").write_text(json.dumps(other_man, indent=1))

    # reversal: enumerate (corpus-only), build + track reversed refs in-driver
    cams = {k: camera_trajectory(b["tracks"], b["vis"]) for k, b in bundles_by_key.items()}
    rev_pairs = probes.enumerate_reversal_pairs(pairs, corpus, cams, bars)
    log(f"  reversal-sensitive pairs: {len(rev_pairs)}")
    rev_cams = {}
    for p in rev_pairs:
        vp = probes.build_reversed_video(REPO_ROOT / probe_root / p["ref"],
                                         probe_dir / f"rev__{p['class']}.mp4")
        rb, _ = process_video_file(vp, cache_dir, extractor, tracker,
                                   short_side=versioning.PINS["feature_short_side"])
        rev_cams[p["ref"]] = camera_trajectory(rb["tracks"], rb["vis"])
    extractor.free(); tracker.free()

    # ---- Block B score + grade ---------------------------------------------------------
    crashed = []
    try:
        sib_items = run_score(man_dir / "siblings.json", "cert_siblings", out, "auto")
    except subprocess.CalledProcessError as e:
        crashed.append(f"siblings scoring: {e}"); sib_items = None
    try:
        probe_items = run_score(man_dir / "probes.json", "cert_probes", out, "off")
    except subprocess.CalledProcessError as e:
        crashed.append(f"probe scoring: {e}"); probe_items = None

    rows = {}
    if sib_items:
        rows.update(load_rows(sib_items))
    if probe_items:
        rows.update(load_rows(probe_items))
    g_sib = probes.grade_siblings(rows, classes, bars)
    g_ctl = probes.grade_controls(rows, classes, bars)
    g_spl = probes.grade_splices(rows, classes, bars)
    g_rev = probes.grade_reversal(rev_pairs, cams, rev_cams, bars)
    g_m3 = probes.grade_m3_panel(rows, classes, bars)

    # ---- Block C -----------------------------------------------------------------------
    log("Block C: archives")
    c_items, c_excluded = [], []
    v2_paths = {
        "exp_056": main_root / "outputs/eval/exp_056/quads/run_0002",
        "exp_057": main_root / "outputs/eval/exp_057/quads/run_0001",
        "exp_058": main_root / "outputs/eval/exp_058/quads/run_0001",
    }
    e57_items = []
    for name, run_dir in v2_paths.items():
        items, excl = blockc.convert_v2_manifest(run_dir / "manifest_scoring.json",
                                                 corpus, main_root)
        for it in items:
            it["item_id"] = f"{name}__{it['item_id']}"
        c_items += items
        c_excluded += [{**e, "run": name} for e in excl]
        if name == "exp_057":
            e57_items = items
    (man_dir / "blockc.json").write_text(json.dumps(c_items, indent=1))
    log(f"  archives: {len(c_items)} convertible, {len(c_excluded)} excluded (loud)")
    try:
        c_jsonl = run_score(man_dir / "blockc.json", "cert_blockc", out, "off")
        c_rows = load_rows(c_jsonl)
    except subprocess.CalledProcessError as e:
        crashed.append(f"blockC scoring: {e}"); c_rows = {}

    twin_ids = [f"exp_057__{t}" for t in
                blockc.copy_twin_ids(v2_paths["exp_057"] / "manifest_scoring.json")]
    g_twin = blockc.grade_copy_twins(c_rows, twin_ids)
    bridge = {name: blockc.bridge_v2_v3(run_dir / "items.jsonl",
                                        {k.split("__", 1)[1]: v for k, v in c_rows.items()
                                         if k.startswith(name + "__")})
              for name, run_dir in v2_paths.items()}
    dists = blockc.arm_distributions(list(c_rows.values()))

    # ---- Block D -----------------------------------------------------------------------
    log("Block D: stability")
    warm_cmp = cold_cmp = None
    if sib_items:
        try:
            sib2 = run_score(man_dir / "siblings.json", "cert_siblings_rerun", out, "auto")
            warm_cmp = compare_runs(sib_items, sib2,
                                    tolerance=bars["stability"]["bar8"]["warm_max_abs_delta"])
        except subprocess.CalledProcessError as e:
            crashed.append(f"warm rerun: {e}")
        anchors = anchor_ids(pairs, corpus, e57_items, bars)
        anchor_man = [it for it in (sib_man + c_items) if it["item_id"] in anchors]
        (man_dir / "anchors.json").write_text(json.dumps(anchor_man, indent=1))
        try:
            cold_jsonl = run_score(man_dir / "anchors.json", "cert_anchors_cold", out,
                                   "auto", cache_dir=str(out / "cold_cache"))
            base = {i: r for i, r in {**rows, **c_rows}.items() if i in anchors}
            base_path = out / "anchors_warm.jsonl"
            base_path.write_text("\n".join(json.dumps(r) for r in base.values()))
            cold_cmp = compare_runs(base_path, cold_jsonl,
                                    tolerance=bars["stability"]["bar8"]["anchors"]["reproduction_tolerance"])
        except subprocess.CalledProcessError as e:
            crashed.append(f"cold anchors: {e}")

    floors = {}
    for side in ("twosided", "onesided"):
        vals = [rows[k]["app_ref"] for k in rows
                if rows[k]["arm"].startswith("control")
                and rows[k].get("sidedness") == side
                and np.isfinite(rows[k].get("app_ref", np.nan))]
        floors[side] = {"mean": float(np.mean(vals)), "n": len(vals)} if vals else None

    # ---- verdicts + record ---------------------------------------------------------------
    bar8 = {"no_crash": not crashed, "crashes": crashed,
            "warm": warm_cmp, "cold_anchors": cold_cmp,
            "pass": bool(not crashed and warm_cmp and warm_cmp["pass"]
                         and cold_cmp and cold_cmp["pass"])}
    verdicts = {
        "bar1_m1a_floor": exam_res["bar1"]["pass"],
        "bar2_siblings": g_sib["pass"],
        "bar3_controls": g_ctl["pass"],
        "bar4_splices": g_spl["pass"],
        "bar5_reversal": g_rev["pass"],
        "bar6_m3_panel": g_m3["pass"],
        "bar7_copy_twins": g_twin["pass"],
        "bar8_integration_determinism": bar8["pass"],
    }
    overall = all(verdicts.values())
    calibration = {
        "tau_copy": {"initial": bars["probes"]["copy_splices"]["tau_copy_initial"],
                     "recalibrated": g_spl.get("tau_recalibrated"),
                     "rule": "midpoint(splice_min, honest_max); tested by bar7"},
        "core_fallback": {"min_frames": 8, "delta": 0.05,
                          "note": "frozen constants; fallback output is always flagged"},
        "sidedness_floors": floors,
        "reversal_sensitive_set": rev_pairs,
        "sigma_seed": {"status": "PENDING", "gates": "first model report, not the tag",
                       "protocol": bars["stability"]["sigma_seed"]},
    }
    record = {"version": ver, "overall_pass": overall, "verdicts": verdicts,
              "stamp": stamp, "bars_sha256": versioning.sha256_file(BARS_PATH),
              "exam": {k: v for k, v in exam_res.items() if k != "trust_map"},
              "grades": {"siblings": g_sib, "controls": g_ctl, "splices": g_spl,
                         "reversal": g_rev, "m3_panel": g_m3, "copy_twins": g_twin,
                         "bar8": bar8},
              "content_invariance": {**audit, "per_class": "see content_invariance.json",
                                     "alarm_level": bars["record"]["content_invariance"]["alarm_level"],
                                     "gating": False},
              "blockc": {"n_scored": len(c_rows), "excluded": c_excluded,
                         "bridge": bridge, "arm_distributions": dists},
              "calibration": calibration, "claims": CLAIMS}
    (out / "record.json").write_text(json.dumps(record, indent=1, default=str))

    md = [f"# Certification record — transition-eval/{ver}",
          "",
          f"**Overall: {'PASS' if overall else 'FAIL'}** · bars sha256 "
          f"`{record['bars_sha256'][:16]}…` · corpus sha256 "
          f"`{(stamp['corpus_sha256'] or '')[:16]}…` · commit `{stamp['git']['commit_short']}`",
          "", "| bar | verdict |", "|---|---|"]
    md += [f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in verdicts.items()]
    md += ["",
           f"**What certification claims, exactly:** *{CLAIMS}* It does not claim: "
           "that metrics track human judgment (M4 exempt until O9); that pools/masks "
           "behave identically on generated-domain frames (untestable without labels); "
           "M2c validity (first real training manifest); M1b absolute validity "
           "(injected-trajectory test = post-lock appendix).",
           "",
           f"- mask winner: **{mask_w}** · motion winner: "
           f"**{exam_res['motion_adoption']['winner']}** · O7 Huber triggered: "
           f"{exam_res['o7_conditional']['huber_triggered']}",
           f"- content-invariance pooled partial corr: {audit['pooled_partial_corr']} "
           f"(alarm {bars['record']['content_invariance']['alarm_level']}, non-gating)",
           f"- tau_copy recalibrated: {g_spl.get('tau_recalibrated')} "
           f"(gap {g_spl.get('gap')})",
           f"- sigma_seed: PENDING (gates first model report, not this tag)",
           f"- Block C: {len(c_rows)} archived items scored; "
           f"{len(c_excluded)} excluded loudly (see record.json)",
           "", "Artifacts: exam/, content_invariance.json, manifests/, "
               "probe_videos/, cert_*/items.jsonl, record.json — all under "
               f"`{out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}`."]
    rec_dir = pathlib.Path(__file__).parents[1] / "certifications"
    rec_dir.mkdir(exist_ok=True)
    (rec_dir / f"v{ver}.md").write_text("\n".join(md))
    log(f"record -> certifications/v{ver}.md  overall={'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
