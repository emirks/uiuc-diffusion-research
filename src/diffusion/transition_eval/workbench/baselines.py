"""Step-0 freeze verification (OPERATIONS.md §4, items 1/2/5).

Regenerates every pinned incumbent number in RUNBOOK §B from the frozen
certification artifacts, asserts equality with the pinned table, verifies the
npz sha256 and the row-order alignment, and backfills the within-stratum
recalls that RUNBOOK A4 requires (analysis.json predates the by-tag machinery,
so the 0.62/0.44 figures cited from memory in §3.5 have no persisted source).

Everything here runs the DEPLOYED code — report.retrieval_eval (the frozen exam
kernel) and certify.diagnostics (per_clip_rows / clip_tags / tag_accuracy) — on
the frozen matrices. Nothing is reimplemented; a candidate is judged by the same
functions that judge the incumbent here.

Outputs $WB_OUT/step0/baselines.json. A mismatch with §B is a STOP.

    PYTHONPATH=$WB/src python -m diffusion.transition_eval.workbench.baselines
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from ..certify import diagnostics
from ..report import retrieval_eval
from . import paths

# RUNBOOK §B, pinned 2026-07-14 before any candidate ran. The workbench refuses
# to proceed if the artifacts no longer reproduce these.
NPZ_SHA256 = "f96934c65fdc95f9a4709e5673ba39b00f3c257aba191c8c8a14889ceb31483b"
PINNED = {
    #                     acc        d          coverage  misretrieved
    "m1a__v3_sided":   (0.672646, 1.522006, 1.0000, 73),
    "m1a__v2_envelope": (0.578475, 1.319097, 1.0000, 94),
    "m1a__all_frames": (0.538117, 1.271417, 1.0000, 103),
    "m_incumbent":     (0.062780, 0.368167, 1.0000, 209),
    "m1b_camera":      (0.268519, 0.519962, 0.9686, 165),
    "m1c_object":      (0.076577, 0.247601, 0.9955, 206),
}
TOL = 5e-7          # the pinned table is written to 6 decimals
COVERAGE_TOL = 5e-5  # coverage is pinned to 4 decimals

INCUMBENT_APPEARANCE = "m1a__v3_sided"
INCUMBENT_CAMERA = "m1b_camera"
INCUMBENT_OBJECT = "m1c_object"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def misretrieved(rows: list[dict]) -> int:
    """Clips whose 1-NN is not same-class. An UNCOVERED row (no finite distance
    -> pred None) is misretrieved, not excluded: definedness never buys accuracy
    (RUNBOOK §1.5). This is the count the §4.1 kill rule and §7 adoption rule
    are stated against."""
    return sum(1 for r in rows if r["pred"] != r["label"])


def evaluate(D: np.ndarray, keys: list[str], labels: list[str]) -> dict:
    """The frozen kernel + deployed per-clip rows for one distance matrix."""
    r = retrieval_eval(D, labels)
    rows = diagnostics.per_clip_rows(D, keys, labels)
    mis = misretrieved(rows)
    # cross-check: the kernel's (accuracy, coverage) must imply the same count
    n = len(labels)
    n_valid = int(round(r["coverage"] * n))
    n_correct = int(round(r["accuracy_1nn"] * n_valid))
    assert n - n_correct == mis, (
        f"kernel accuracy/coverage imply {n - n_correct} misretrieved but the "
        f"deployed per-clip rows say {mis}")
    return {
        "accuracy_1nn": r["accuracy_1nn"],
        "separation_cohens_d": r["separation_cohens_d"],
        "coverage": r["coverage"],
        "misretrieved": mis,
        "n_clips": n,
        "accuracy_wilson95": list(r["accuracy_wilson95"]),
        "chance": r["chance"],
        "within_mean": r["within_mean"],
        "cross_mean": r["cross_mean"],
        "per_class_recall": r["per_class_recall"],
        "rows": rows,
    }


def stratum_recalls(D: np.ndarray, keys: list[str], labels: list[str],
                    per_class_recall: dict, stratum: set[str],
                    eligible: set[str]) -> dict:
    """Every defensible reading of "within-stratum recall" (RUNBOOK §3.5/§3.6),
    computed for one metric on one stratum. All five are RECORDED; gates.yaml
    pins which one gates, and the pinned one is applied identically to the
    incumbent here and to every candidate.

    The stratum is defined by the MANIFEST tags (corpus['classes'][c]['tags']) —
    the source the deployed GATING code uses (exam.adopt_motion). The deployed
    DESCRIPTIVE table (diagnostics.tag_accuracy) instead parses the source dir
    name, which drops class `monstrosity` (dir `onesided_object-monstrosity`:
    clip_tags splits on "_", so the hyphenated tag never matches). monstrosity
    is n=3 and therefore ineligible, so the two sources cannot disagree on any
    eligible-scoped statistic; the divergence is recorded, not patched.
    """
    lab = np.array(labels)
    idx = np.flatnonzero(np.isin(lab, sorted(stratum)))
    rows = diagnostics.per_clip_rows(D, keys, labels)
    graded = [rows[i] for i in idx if rows[i]["pred"] is not None]
    hits_all = [rows[i]["pred"] == rows[i]["label"] for i in idx]

    sub_labels = [labels[i] for i in idx]
    sub = retrieval_eval(D[np.ix_(idx, idx)], sub_labels)

    in_stratum = sorted(stratum)
    in_both = sorted(stratum & eligible)
    macro_all = float(np.nanmean([per_class_recall.get(c, np.nan) for c in in_stratum]))
    macro_elig = float(np.nanmean([per_class_recall.get(c, np.nan) for c in in_both]))
    return {
        "n_clips": len(idx),
        "n_classes": len(in_stratum),
        "n_classes_eligible": len(in_both),
        # (i) clip-pooled; an uncovered clip counts as a miss (the `misretrieved` convention)
        "clip_pooled_uncovered_as_miss": float(np.mean(hits_all)),
        # (ii) clip-pooled; uncovered clips dropped from the denominator
        #      — the deployed descriptive convention (diagnostics.tag_accuracy)
        "clip_pooled_uncovered_dropped": float(
            np.mean([r["pred"] == r["label"] for r in graded])) if graded else float("nan"),
        # (iii) macro mean of per-class recall over every class in the stratum
        "macro_per_class_all": macro_all,
        # (iv) macro mean of per-class recall over stratum INTERSECT n>=4-eligible
        #      — the certified exam's own statistic (o7 camera_stratum_mean_recall)
        "macro_per_class_eligible": macro_elig,
        # (v) restricted-pool: retrieval among the stratum's clips only (easier task)
        "restricted_pool_accuracy": sub["accuracy_1nn"],
        "restricted_pool_coverage": sub["coverage"],
    }


def check_pinned(name: str, got: dict) -> list[str]:
    acc, d, cov, mis = PINNED[name]
    errs = []
    if abs(got["accuracy_1nn"] - acc) > TOL:
        errs.append(f"{name}: accuracy {got['accuracy_1nn']:.6f} != pinned {acc:.6f}")
    if abs(got["separation_cohens_d"] - d) > TOL:
        errs.append(f"{name}: cohens_d {got['separation_cohens_d']:.6f} != pinned {d:.6f}")
    if abs(got["coverage"] - cov) > COVERAGE_TOL:
        errs.append(f"{name}: coverage {got['coverage']:.4f} != pinned {cov:.4f}")
    if got["misretrieved"] != mis:
        errs.append(f"{name}: misretrieved {got['misretrieved']} != pinned {mis}")
    return errs


def main() -> int:
    corpus = paths.load_corpus()
    keys = paths.corpus_keys(corpus)
    labels = paths.labels_of(corpus, keys)
    sidedness = paths.sidedness_of(corpus, keys)

    print(f"[step0] corpus {len(keys)} clips / {len(set(labels))} classes")

    # --- 1. npz integrity -----------------------------------------------------
    got_sha = sha256(paths.NPZ)
    print(f"[step0] distance_matrices.npz sha256 {got_sha}")
    if got_sha != NPZ_SHA256:
        print(f"STOP: npz sha256 mismatch — pinned {NPZ_SHA256}", file=sys.stderr)
        return 1

    z = np.load(paths.NPZ)

    # --- 2. key alignment (row order candidates must share) -------------------
    npz_keys = [str(k) for k in z["keys"]]
    if npz_keys != keys:
        print(f"STOP: npz key order != sorted corpus keys "
              f"({len(npz_keys)} vs {len(keys)})", file=sys.stderr)
        return 1
    print("[step0] key alignment OK — npz rows == sorted(corpus['clips'])")

    # --- 3. regenerate RUNBOOK §B --------------------------------------------
    metrics, errors = {}, []
    for name in PINNED:
        m = evaluate(z[name], keys, labels)
        metrics[name] = m
        errors += check_pinned(name, m)
        print(f"[step0] {name:18s} acc {m['accuracy_1nn']:.6f}  d {m['separation_cohens_d']:.6f}  "
              f"cov {m['coverage']:.4f}  mis {m['misretrieved']}/{m['n_clips']}")
    if errors:
        print("STOP: regenerated numbers disagree with RUNBOOK §B:", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        return 1
    print("[step0] all 6 metrics reproduce RUNBOOK §B exactly")

    # --- 4. stratum backfill (RUNBOOK A4) -------------------------------------
    # Deployed convention (certify.diagnostics.tag_accuracy): 1-NN runs over the
    # FULL 223-clip corpus; a stratum's recall is the accuracy of the clips
    # carrying that tag. The retrieval POOL is never restricted — restricting it
    # would be an easier task and non-comparable to the pinned pooled numbers.
    clips = [{"key": k, "class": l, "sidedness": s,
              "tags": diagnostics.clip_tags(corpus["clips"][k]["source"])}
             for k, l, s in zip(keys, labels, sidedness)]
    by_tag = diagnostics.tag_accuracy({n: m["rows"] for n, m in metrics.items()}, clips)
    for row in by_tag["coarse"]:
        print(f"[step0] stratum {row['group']:10s} n={row['n']:3d}  "
              f"m1b {row[INCUMBENT_CAMERA]}  m1c {row[INCUMBENT_OBJECT]}  "
              f"m1a {row[INCUMBENT_APPEARANCE]}")

    # --- 5. corpus facts (frozen, outcome-independent) ------------------------
    class_n = {c: corpus["classes"][c]["n_clips"] for c in corpus["classes"]}
    eligible = {c for c, n in class_n.items() if n >= 4}
    strata = {
        "camera": {c for c, v in corpus["classes"].items() if "camera" in v.get("tags", [])},
        "object": {c for c, v in corpus["classes"].items() if "object" in v.get("tags", [])},
        "style": {c for c, v in corpus["classes"].items() if "style" in v.get("tags", [])},
    }
    stratum_table = {
        name: {s: stratum_recalls(z[name], keys, labels, m["per_class_recall"],
                                  strata[s], eligible)
               for s in strata}
        for name, m in metrics.items()
    }
    for name, s in (("m1b_camera", "camera"), ("m1c_object", "object")):
        t = stratum_table[name][s]
        print(f"[step0] {name} on the {s} stratum (n={t['n_clips']} clips, "
              f"{t['n_classes_eligible']}/{t['n_classes']} classes eligible):")
        print(f"          clip-pooled (uncovered=miss)    {t['clip_pooled_uncovered_as_miss']:.5f}")
        print(f"          clip-pooled (uncovered dropped) {t['clip_pooled_uncovered_dropped']:.5f}")
        print(f"          macro per-class (all)           {t['macro_per_class_all']:.5f}")
        print(f"          macro per-class (eligible)      {t['macro_per_class_eligible']:.5f}")
        print(f"          restricted pool                 {t['restricted_pool_accuracy']:.5f}")

    # Integrity tie-back: definition (iv) on the camera stratum IS the statistic
    # the certified exam persisted (o7_conditional.camera_stratum_mean_recall).
    # If our stratum code cannot reproduce the certified number, it is the wrong
    # code and no candidate may be judged by it.
    exam_json = json.loads((paths.BASELINE_DIR / "exam/exam.json").read_text())
    certified_cam = exam_json["o7_conditional"]["camera_stratum_mean_recall"]
    ours = stratum_table["m1b_camera"]["camera"]["macro_per_class_eligible"]
    if abs(ours - certified_cam) > 1e-12:
        print(f"STOP: our camera-stratum recall {ours!r} != the certified exam's "
              f"recorded {certified_cam!r}", file=sys.stderr)
        return 1
    print(f"[step0] stratum code ties back to the certified exam: "
          f"camera_stratum_mean_recall {ours!r} == exam.json")

    facts = {
        "class_n": class_n,
        "eligible_n_ge_4": sorted(c for c, n in class_n.items() if n >= 4),
        "onesided_classes": sorted(c for c, v in corpus["classes"].items()
                                   if v["sidedness"] == "onesided"),
        "camera_classes": sorted(c for c, v in corpus["classes"].items()
                                 if "camera" in v.get("tags", [])),
        "object_classes": sorted(c for c, v in corpus["classes"].items()
                                 if "object" in v.get("tags", [])),
        "style_classes": sorted(c for c, v in corpus["classes"].items()
                                if "style" in v.get("tags", [])),
    }
    print(f"[step0] eligible (n>=4): {len(facts['eligible_n_ge_4'])}/39 classes; "
          f"camera {len(facts['camera_classes'])}, object {len(facts['object_classes'])}")

    out_dir = paths.WB_OUT / "step0"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_from": {
            "npz": str(paths.NPZ), "npz_sha256": got_sha,
            "analysis_json": str(paths.ANALYSIS_JSON),
            "record": str(paths.RECORD_DIR / "record.json"),
            "kernel": "diffusion.transition_eval.report.retrieval_eval (frozen)",
            "stratum_code": "diffusion.transition_eval.certify.diagnostics.tag_accuracy "
                            "(deployed; full-corpus retrieval pool)",
        },
        "pinned_table_runbook_B": {k: dict(zip(
            ("accuracy_1nn", "separation_cohens_d", "coverage", "misretrieved"), v))
            for k, v in PINNED.items()},
        "regenerated": {n: {k: v for k, v in m.items() if k != "rows"}
                        for n, m in metrics.items()},
        "by_tag": by_tag,
        "stratum_recalls": stratum_table,
        "stratum_definition_note": (
            "RUNBOOK §3.5 cites 0.62/0.44 from memory; A4 requires backfill from the "
            "frozen artifact. NO definition reproduces 0.62/0.44 (see stratum_recalls) — "
            "the memory figures are a drafting error of the same class as A1's 71/221, and "
            "A4 pre-registered that the backfilled values, not the memory figures, are the "
            "numbers to beat. Definition macro_per_class_eligible reproduces the certified "
            "exam's own o7_conditional.camera_stratum_mean_recall bit-for-bit; gates.yaml "
            "pins which definition gates, and it is applied identically to incumbent and "
            "candidate."),
        "certified_exam_tieback": {
            "o7_camera_stratum_mean_recall": certified_cam,
            "reproduced_by": "stratum_recalls.m1b_camera.camera.macro_per_class_eligible",
        },
        "corpus_facts": facts,
        "incumbents": {"appearance": INCUMBENT_APPEARANCE,
                       "camera": INCUMBENT_CAMERA, "object": INCUMBENT_OBJECT},
    }
    (out_dir / "baselines.json").write_text(json.dumps(payload, indent=1, default=str))
    print(f"[step0] wrote {out_dir / 'baselines.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
