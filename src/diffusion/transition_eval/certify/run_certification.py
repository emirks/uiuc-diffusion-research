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
          "attached), does so with content-controlled discrimination that exceeds "
          "an explicit content baseline (bar 9), refuses known-degenerate and "
          "copied inputs, is direction-sensitive on motion, runs deterministically "
          "end-to-end on real generations, and its blind spots are enumerated.")

# v4.0.0 additional non-claims (advisor V1/F6, SPEC §6.5): the corpus-relative
# reference populations were fitted on real-corpus pairs — behavior on
# generated-domain inputs is flagged (saturation), not certified.
V4_NONCLAIM = ("reference populations (S3 ECDFs, D_ZPR ECDFs, CSLS neighborhoods) "
               "were fitted on real-corpus pairs; behavior on generated-domain "
               "inputs outside their support is flagged (saturation), not certified. "
               "M1c ships a SCOPED causal stamp: object-motion signal is real but "
               "faint, and its margin over the strongest content proxy sits at the "
               "0.10 practical bar (excess over DINO 0.156; over the color proxy "
               "0.099 — DINO-only gates, both ship in the datasheet, advisor V1/F1a).")


def log(msg: str) -> None:
    print(f"[certify] {msg}", flush=True)


class ScoreError(RuntimeError):
    """A score.py subprocess exited nonzero (tail of its log attached)."""


def start_score(manifest: pathlib.Path, label: str, out_root: pathlib.Path,
                controls: str, cache_dir: str | None = None,
                lpips_cache: str | None = None) -> dict:
    """Launch score.py through its real CLI — certification exercises the
    shipped entrypoint, not a private shim. Independent manifests run
    concurrently (disjoint item sets -> disjoint cache writes; per-item math
    is untouched, so outputs are identical to sequential runs — and bar 8's
    warm/cold comparisons verify that at run time). Output goes to
    <out_root>/<label>.score.log; wait_score() collects the items.jsonl."""
    cmd = [sys.executable, "-m", "diffusion.transition_eval.score",
           "--manifest", str(manifest),
           "--corpus", str(CORPUS_PATH),
           "--label", label, "--out-root", str(out_root),
           "--controls", controls]
    if cache_dir:
        cmd += ["--cache-dir", cache_dir]
    if lpips_cache:
        cmd += ["--lpips-cache", lpips_cache]
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    logf = open(REPO_ROOT / out_root / f"{label}.score.log", "w")
    log(f"score: {label} ({controls=}) started")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env,
                            stdout=logf, stderr=subprocess.STDOUT)
    return {"proc": proc, "logf": logf, "label": label, "out_root": out_root}


def wait_score(h: dict) -> pathlib.Path:
    rc = h["proc"].wait()
    h["logf"].close()
    if rc != 0:
        tail = " | ".join((REPO_ROOT / h["out_root"] / f"{h['label']}.score.log")
                          .read_text().splitlines()[-3:])
        raise ScoreError(f"{h['label']} exit {rc}: {tail}")
    log(f"score: {h['label']} done")
    return REPO_ROOT / h["out_root"] / h["label"] / "items.jsonl"


def load_rows(items_jsonl: pathlib.Path) -> dict[str, dict]:
    return {r["item_id"]: r for r in
            (json.loads(l) for l in items_jsonl.read_text().splitlines() if l.strip())}


def anchor_ids(pairs_by_class: dict, corpus: dict, c_items: list[dict],
               bars: dict) -> list[str]:
    """Frozen anchor RULE (bars.stability.anchors.rule): deterministic picks,
    no cherry-picking surface. Strata are filled in order (two-sided,
    one-sided, camera), each taking the first lexicographic n>=4 class NOT
    already picked — as written in draft.7 the rule produced 5 ids
    (air_bending led two strata) and its own n=6 assertion refused; the dedup
    makes the frozen rule executable without changing any pick semantics."""
    taken: set[str] = set()

    def first(pred):
        c = next((c for c in sorted(pairs_by_class)
                  if c not in taken and pred(corpus["classes"][c])), None)
        if c is not None:
            taken.add(c)
        return c
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
                                  short_side=versioning.PINS["feature_short_side"],
                                  need_frames=False)
        bundles_by_key[key] = b
        if (i + 1) % 40 == 0:
            log(f"  corpus {i + 1}/{len(keys)}")
    labels = [corpus["clips"][k]["class"] for k in keys]
    sidedness = [corpus["classes"][l]["sidedness"] for l in labels]
    bundles = [bundles_by_key[k] for k in keys]

    # ---- Block A (v4: direct replacement — headline metrics ARE S3/D_ZPR/CSLS) -------
    log("Block A: exam v4 (R1 S3/D_ZPR/CSLS + R2 M2b + bar9 causal gate)")
    clip_paths = [REPO_ROOT / probe_root / k for k in keys]
    exam_res = exam.run_exam_v4(bundles, labels, sidedness, corpus, bars, out / "exam",
                                clip_paths, analysis_dir=out / "analysis")
    log(f"  bar1(S3 d>={exam_res['bar1']['d_min']}) pass={exam_res['bar1']['pass']} "
        f"(d={exam_res['bar1']['d']:.3f}); bar9(causal gate) pass={exam_res['bar9']['pass']}")
    for nm, v in exam_res["bar9"]["metrics_causal_PASS"].items():
        log(f"    bar9 {nm}: causal_PASS={v}")
    for nm, v in exam_res["bar9"]["controls_causal_PASS"].items():
        log(f"    bar9 control {nm}: causal_PASS={v} (must be False)")

    # ---- reference-artifact rebuild parity (SPEC §4/§7) -----------------------------
    # The committed reference_v4 artifact (a pinned instrument constant, sha in
    # versioning.PINS) is what score.py ranks every generated pair against. Rebuild
    # it from THIS corpus (reusing the exam's already-computed matrices) and assert
    # it reproduces the committed artifact within tolerance — else the artifact is
    # stale/corrupt and Block B/C would score against wrong reference populations.
    from .. import reference_stats as RS
    parts = exam_res.pop("_reference_parts")
    fresh_ref = RS.build_reference_from_parts(parts["channels"], parts["views"],
                                              parts["object_D"], keys,
                                              stamp["corpus_sha256"])
    committed_ref = None
    try:
        committed_ref = RS.load_reference(expect_corpus_sha=stamp["corpus_sha256"])
        rebuild_parity = RS.compare_reference(fresh_ref, committed_ref,
                                              tol=bars["reference"]["rebuild_tol"])
    except Exception as e:  # noqa: BLE001 — a missing/mismatched artifact is a hard fail
        rebuild_parity = {"pass": False, "mismatch": [f"{type(e).__name__}: {e}"],
                          "per_array": {}}
    rebuild_parity["artifact_sha256"] = versioning.sha256_file(
        REPO_ROOT / "src/diffusion/transition_eval/reference_v4.npz")
    log(f"  reference rebuild-parity: pass={rebuild_parity['pass']} "
        f"(sha {str(rebuild_parity['artifact_sha256'])[:12]}…"
        + ("" if rebuild_parity["pass"] else f"; mismatch {rebuild_parity['mismatch']}") + ")")

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
    rev_bundles = {}   # v4 (Q1): reversed-ref tracks/vis for the D_ZPR reversal field
    for p in rev_pairs:
        vp = probes.build_reversed_video(REPO_ROOT / probe_root / p["ref"],
                                         probe_dir / f"rev__{p['class']}.mp4")
        rb, _ = process_video_file(vp, cache_dir, extractor, tracker,
                                   short_side=versioning.PINS["feature_short_side"],
                                   need_frames=False)
        rev_cams[p["ref"]] = camera_trajectory(rb["tracks"], rb["vis"])
        rev_bundles[p["ref"]] = {"tracks": rb["tracks"], "vis": rb["vis"],
                                 "cam": rev_cams[p["ref"]]}
    extractor.free(); tracker.free()

    # ---- Block C conversion (pure CPU) — before scoring, so all three
    # independent manifests can score concurrently ---------------------------------------
    log("Block C: archives (conversion)")
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

    # ---- Blocks B+C scoring (three independent manifests, concurrent) + grade ----------
    # Every step from here on funnels failures into `crashed` so the record is
    # ALWAYS written (draft.7's driver died at the anchor rule and the record
    # had to be assembled post-hoc — never again).
    crashed = []

    def safe(name, fn, default):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — recorded loudly, run continues
            crashed.append(f"{name}: {type(e).__name__}: {e}")
            return default

    GRADER_CRASH = {"pass": False, "reason": "grader crashed (see bar8.crashes)"}
    h_sib = start_score(man_dir / "siblings.json", "cert_siblings", out, "auto")
    h_prb = start_score(man_dir / "probes.json", "cert_probes", out, "off")
    h_blc = start_score(man_dir / "blockc.json", "cert_blockc", out, "off")
    try:
        sib_items = wait_score(h_sib)
    except ScoreError as e:
        crashed.append(f"siblings scoring: {e}"); sib_items = None
    try:
        probe_items = wait_score(h_prb)
    except ScoreError as e:
        crashed.append(f"probe scoring: {e}"); probe_items = None
    try:
        c_rows = load_rows(wait_score(h_blc))
    except ScoreError as e:
        crashed.append(f"blockC scoring: {e}"); c_rows = {}

    rows = {}
    if sib_items:
        rows.update(load_rows(sib_items))
    if probe_items:
        rows.update(load_rows(probe_items))
    min_n = bars["probes"]["siblings"]["bar2"]["eligibility_min_n"]
    eligible = [c for c in classes if corpus["classes"][c]["n_clips"] >= min_n]
    inelig = {c: corpus["classes"][c]["n_clips"] for c in classes if c not in eligible}
    g_sib = safe("grade_sibling_floor",
                 lambda: probes.grade_sibling_floor(rows, eligible, bars, inelig),
                 GRADER_CRASH)

    # bar 2 leave-own-clip-out robustness clause (v4, advisor Q2): a PASS must not
    # depend on the sibling's in-sample ECDF leakage. Re-score each eligible
    # sibling through the deployed m1a_pair kernel against M1a populations with the
    # sibling's own row/col masked, and require sibling_LOO app_ref > its control's
    # app_ref (the control is synthetic — out-of-sample already — so it is unchanged).
    def _bar2_loo():
        per = {}
        for cls in eligible:
            sib_key, ref_key = [str(k) for k in pairs[cls]["bar_pair"]]
            ctrl = next((rows[k] for k in rows if k.endswith(f"__sib__{cls}")
                         and rows[k]["arm"].startswith("control")), None)
            if ctrl is None or not np.isfinite(ctrl.get("app_ref", np.nan)):
                per[cls] = {"pass": False, "reason": "missing/NaN control"}
                continue
            loo = exam.bar2_loo_app_ref(parts["channels"], bundles, sidedness, keys,
                                        sib_key, ref_key)
            per[cls] = {"pass": bool(loo > ctrl["app_ref"]),
                        "sib_loo_app_ref": float(loo),
                        "control_app_ref": float(ctrl["app_ref"]),
                        "loo_margin": float(loo - ctrl["app_ref"])}
        n_pass = sum(1 for v in per.values() if v["pass"])
        return {"pass": bool(eligible and n_pass == len(eligible)),
                "n_pass": n_pass, "n_eligible": len(eligible),
                "rule": "sibling LOO app_ref > control app_ref, all eligible classes; "
                        "M1a ECDF populations mask the sibling's own row/col (mu full-corpus)",
                "per_class": per}
    g_sib_loo = safe("grade_bar2_loo", _bar2_loo, GRADER_CRASH)
    g_spl = safe("grade_splices", lambda: probes.grade_splices(rows, classes, bars), GRADER_CRASH)
    g_rev = safe("grade_reversal", lambda: probes.grade_reversal(rev_pairs, cams, rev_cams, bars), GRADER_CRASH)
    g_m3 = safe("grade_m3_panel", lambda: probes.grade_m3_panel(rows, classes, bars), GRADER_CRASH)

    # D_ZPR reversal deltas (v4, advisor Q1) — MANDATORY NON-GATING record field.
    # bar 5 gates on the verbatim v3 cam_corr form (above); this additionally
    # records the shipped M1b metric's per-pair behavior under reversal through the
    # deployed m1b_pair (ecdf_lookup) path — the reversed ref is non-corpus, exactly
    # the out-of-sample path a real generation takes. delta > 0 ⟺ reversed ref is a
    # WORSE (larger-distance) D_ZPR match, i.e. the shipped metric is direction-
    # sensitive. A promotion candidate for the next version's bars, not gated here
    # (untested strict inequality; F8 forbids pre-testing; a spurious FAIL would burn
    # a fail-forward re-run over likely ECDF-quantization noise).
    def _dzpr_reversal():
        from ..m1_transfer import camera_zpr
        out_rows = []
        for p in rev_pairs:
            g, r = p["gen"], p["ref"]
            gb, rb0 = bundles_by_key[g], bundles_by_key[r]
            un = camera_zpr(gb["tracks"], gb["vis"], cams[g],
                            rb0["tracks"], rb0["vis"], cams[r], committed_ref)["cam_zpr"]
            rv = camera_zpr(gb["tracks"], gb["vis"], cams[g],
                            rev_bundles[r]["tracks"], rev_bundles[r]["vis"],
                            rev_bundles[r]["cam"], committed_ref)["cam_zpr"]
            delta = (rv - un) if (np.isfinite(un) and np.isfinite(rv)) else float("nan")
            out_rows.append({"class": p["class"], "dzpr_unreversed": float(un),
                             "dzpr_reversed": float(rv), "delta_reversed_minus_unrev": float(delta)})
        fin = [x["delta_reversed_minus_unrev"] for x in out_rows
               if np.isfinite(x["delta_reversed_minus_unrev"])]
        return {"gating": False, "per_pair": out_rows, "n": len(fin),
                "n_direction_sensitive": int(sum(1 for d in fin if d > 0)),
                "note": "delta>0 => reversed ref is a worse D_ZPR match (direction-sensitive)"}
    dzpr_reversal = safe("dzpr_reversal_field", _dzpr_reversal,
                         {"gating": False, "per_pair": "not computed — crashed"})

    twin_ids = [f"exp_057__{t}" for t in
                blockc.copy_twin_ids(v2_paths["exp_057"] / "manifest_scoring.json")]
    g_twin = safe("grade_copy_twins", lambda: blockc.grade_copy_twins(c_rows, twin_ids),
                  GRADER_CRASH)
    bridge = safe("bridge_v2_v3", lambda: {
        name: blockc.bridge_v2_v3(run_dir / "items.jsonl",
                                  {k.split("__", 1)[1]: v for k, v in c_rows.items()
                                   if k.startswith(name + "__")})
        for name, run_dir in v2_paths.items()}, "not computed — crashed")
    dists = safe("arm_distributions",
                 lambda: blockc.arm_distributions(list(c_rows.values())),
                 "not computed — crashed")

    # ---- Block D (warm rerun + cold anchors, concurrent) --------------------------------
    # The rerun scores with --lpips-cache off: bar 8's warm comparison keeps
    # recomputing LPIPS end-to-end, so a stale/corrupt LPIPS cache entry from
    # the first pass shows up as a warm delta instead of passing silently.
    log("Block D: stability")
    warm_cmp = cold_cmp = None
    if sib_items:
        h_rerun = h_cold = None
        try:
            h_rerun = start_score(man_dir / "siblings.json", "cert_siblings_rerun",
                                  out, "auto", lpips_cache="off")
        except Exception as e:  # noqa: BLE001 — the record must still be written
            crashed.append(f"warm rerun: {type(e).__name__}: {e}")
        try:
            anchors = anchor_ids(pairs, corpus, e57_items, bars)
            anchor_man = [it for it in (sib_man + c_items) if it["item_id"] in anchors]
            (man_dir / "anchors.json").write_text(json.dumps(anchor_man, indent=1))
            h_cold = start_score(man_dir / "anchors.json", "cert_anchors_cold", out,
                                 "auto", cache_dir=str(out / "cold_cache"))
        except Exception as e:  # noqa: BLE001
            crashed.append(f"cold anchors: {type(e).__name__}: {e}")
        if h_rerun:
            try:
                sib2 = wait_score(h_rerun)
                warm_cmp = compare_runs(sib_items, sib2,
                                        tolerance=bars["stability"]["bar8"]["warm_max_abs_delta"])
            except Exception as e:  # noqa: BLE001
                crashed.append(f"warm rerun: {type(e).__name__}: {e}")
        if h_cold:
            try:
                cold_jsonl = wait_score(h_cold)
                base = {i: r for i, r in {**rows, **c_rows}.items() if i in anchors}
                base_path = out / "anchors_warm.jsonl"
                base_path.write_text("\n".join(json.dumps(r) for r in base.values()))
                cold_cmp = compare_runs(base_path, cold_jsonl,
                                        tolerance=bars["stability"]["bar8"]["anchors"]["reproduction_tolerance"])
            except Exception as e:  # noqa: BLE001
                crashed.append(f"cold anchors: {type(e).__name__}: {e}")

    floors = {}
    for side in ("twosided", "onesided"):
        vals = [rows[k]["app_ref"] for k in rows
                if rows[k]["arm"].startswith("control")
                and rows[k].get("sidedness") == side
                and np.isfinite(rows[k].get("app_ref", np.nan))]
        floors[side] = {"mean": float(np.mean(vals)), "n": len(vals)} if vals else None

    # ---- verdicts + record ---------------------------------------------------------------
    # error rows count against no_crash: per-item isolation keeps the DATA of
    # the other bars alive, but an item that cannot score is still an
    # instrument failure and must gate exactly as a crash did.
    error_rows = {i: r["error"] for i, r in {**rows, **c_rows}.items() if r.get("error")}
    bar8 = {"no_crash": bool(not crashed and not error_rows),
            "crashes": crashed, "error_rows": error_rows,
            "warm": warm_cmp, "cold_anchors": cold_cmp,
            "reference_rebuild_parity": rebuild_parity,   # v4: committed artifact matches rebuild
            "pass": bool(not crashed and not error_rows
                         and warm_cmp and warm_cmp["pass"]
                         and cold_cmp and cold_cmp["pass"]
                         and rebuild_parity["pass"])}
    verdicts = {
        "bar1_m1a_floor": exam_res["bar1"]["pass"],
        # bar 2 = the deployed sibling>control floor AND the v4 leave-own-clip-out
        # robustness clause (Q2): a PASS cannot ride on in-sample ECDF leakage.
        "bar2_sibling_floor": bool(g_sib["pass"] and g_sib_loo["pass"]),
        "bar4_splices": g_spl["pass"],
        "bar5_reversal": g_rev["pass"],
        "bar6_m3_panel": g_m3["pass"],
        "bar7_copy_twins": g_twin["pass"],
        "bar8_integration_determinism": bar8["pass"],
        "bar9_causal_gate": exam_res["bar9"]["pass"],   # v4.0.0 (SPEC §6.2)
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
              "grades": {"sibling_floor": g_sib, "sibling_floor_loo": g_sib_loo,
                         "splices": g_spl, "reversal": g_rev, "m3_panel": g_m3,
                         "copy_twins": g_twin, "bar8": bar8,
                         "bar9_causal_gate": exam_res["bar9"]},
              "content_invariance": {**audit, "per_class": "see content_invariance.json",
                                     "alarm_level": bars["record"]["content_invariance"]["alarm_level"],
                                     "gating": False},
              "blockc": {"n_scored": len(c_rows), "excluded": c_excluded,
                         "bridge": bridge, "arm_distributions": dists},
              "calibration": calibration, "claims": CLAIMS,
              # mandatory NON-GATING record fields (advisor V1/consult-2):
              "non_gating_fields": {
                  # F1a: excess over the strongest content proxy (max(DINO,color)),
                  # per headline metric — the record ships both baselines.
                  "causal_excess_maxproxy": {nm: v.get("causal_excess_maxproxy")
                                             for nm, v in exam_res["bar9"]["headline"].items()},
                  # Q1: shipped-M1b (D_ZPR) behavior under reversal, deployed path.
                  "dzpr_reversal": dzpr_reversal,
                  # Q3 (measured by the advisor on the frozen artifact; disclosed):
                  "ecdf_tie_bound": "lookup-vs-compose tie discrepancy <= max_tie_run/n "
                                    "= 5/24753 ~= 2e-4 on [0,1] (pop_App/pop_Dyn max run 5, "
                                    "no mass point; other 7 populations tie-free)",
                  "path_separation": "no bar compares the corpus-matrix path and the "
                                     "per-item lookup path on opposite sides: bar1/bar9 use "
                                     "the corpus matrices only; bar2 (both clauses), bars "
                                     "4/6/7 compare score.py lookup-path rows to score.py "
                                     "lookup-path rows; bar5 uses cam_corr (neither path).",
              }}
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
           f"(injected-trajectory test = post-lock appendix). v4 additionally: {V4_NONCLAIM}",
           "",
           f"- headline metrics (direct replacement, advisor V1/F3a): M1a=**S3** "
           f"(d {exam_res['bar1']['d']:.2f}), M1b=**D_ZPR**, M1c=**CSLS** (scoped "
           f"stamp — object-motion signal real but faint, margin over the strongest "
           f"content proxy at the 0.10 bar); incumbents rescored descriptively (bridge)",
           f"- bar9 causal-excess gate: pass={exam_res['bar9']['pass']} "
           f"(headliners {exam_res['bar9']['metrics_causal_PASS']}; negative controls "
           f"{exam_res['bar9']['controls_causal_PASS']} — all must be False)",
           f"- content-invariance pooled partial corr: {audit['pooled_partial_corr']} "
           f"(alarm {bars['record']['content_invariance']['alarm_level']}, non-gating)",
           f"- tau_copy recalibrated: {g_spl.get('tau_recalibrated')} "
           f"(gap {g_spl.get('gap')})",
           f"- sigma_seed: PENDING (gates first model report, not this tag)",
           f"- Block C: {len(c_rows)} archived items scored; "
           f"{len(c_excluded)} excluded loudly (see record.json)"]

    def _pct(v):
        return "—" if v is None else f"{v:.3f}"
    r1 = exam_res["r1"]
    md += ["", "## Exam detail (representation, non-gating)", "",
           "| metric | acc (1-NN) | Cohen's d | chance |", "|---|---|---|---|"]
    md += [f"| {m} | {_pct(r['accuracy_1nn'])} | {r['separation_cohens_d']:.2f} "
           f"| {_pct(r['chance'])} |" for m, r in r1.items()]
    md += ["", f"R2 pool accuracy: {_pct(exam_res['r2']['accuracy'])} "
               f"over {exam_res['r2']['n_graded']} graded."]
    if exam_res.get("by_tag"):
        mnames = list(r1)
        md += ["", "### R1 accuracy by tag group", "",
               "| group | n | " + " | ".join(mnames) + " |",
               "|---|---|" + "---|" * len(mnames)]
        for row in exam_res["by_tag"]["coarse"] + exam_res["by_tag"]["patterns"]:
            md += [f"| {row['group']} | {row['n']} | "
                   + " | ".join(_pct(row.get(m)) for m in mnames) + " |"]
    md += ["", "Artifacts: exam/, analysis/ (confusion, per-clip margins, class "
               "distances), figures/*.png, results_explorer.html, "
               "content_invariance.json, manifests/, probe_videos/, "
               "cert_*/items.jsonl, record.json — all under "
               f"`{out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}`."]
    rec_dir = pathlib.Path(__file__).parents[1] / "certifications"
    rec_dir.mkdir(exist_ok=True)
    (rec_dir / f"v{ver}.md").write_text("\n".join(md))
    log(f"record -> certifications/v{ver}.md  overall={'PASS' if overall else 'FAIL'}")

    # ---- representation (figures + explorer) — never gates the record ------------------
    log("representation: figures + results explorer")
    try:
        from . import explorer, figures
        figs = figures.save_all(out)
        explorer.build(out)
        log(f"  {len(figs)} figures -> figures/, results_explorer.html written")
    except Exception as e:  # noqa: BLE001 — record is already on disk
        log(f"  representation failed (non-gating, record unaffected): "
            f"{type(e).__name__}: {e}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
