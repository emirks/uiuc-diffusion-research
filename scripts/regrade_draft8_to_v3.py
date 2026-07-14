"""One-time REGRADE of the draft.8 certification run under the 3.0.0 bars.

Owner directive (2026-07-14 inspection): apply the closed-list bar revision
(bar 1 -> d-only; bars 2+3 -> one merged bar) and DO NOT redo already-done
calculations. The two changed bars are pure grading-rule changes over data
the draft.8 run (job 9465002, commit 31dd07e, zero error rows) already
produced under frozen pins; the graders for bars 4-8 are byte-identical in
this revision, so their verdicts and grade payloads carry over verbatim from
the draft.8 record. This script re-runs ONLY the changed graders — imported
from the deployed certify package, never reimplemented — and assembles the
3.0.0 record with full regrade provenance.

    PYTHONPATH=src python scripts/regrade_draft8_to_v3.py

Honesty notes baked into the record it writes:
- Both bar edits are OUTCOME-AWARE (decided with the draft.8 numbers on the
  table). They re-register for future runs; for this record, the surviving
  clauses were also part of the pre-registered draft.8 bars and passed there.
- nature_bloom (n=2), draft.8's only miss on the old bars 2 and 3, leaves
  the merged bar's denominator under the n>=4 eligibility rule.
"""

from __future__ import annotations

import json
import pathlib
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diffusion.transition_eval import versioning                     # noqa: E402
from diffusion.transition_eval.certify import probes                 # noqa: E402
from diffusion.transition_eval.manifests_v3 import load_corpus_manifest  # noqa: E402

SRC_CERT = REPO_ROOT / "outputs/eval/certification/3.0.0-draft.8"
CORPUS = REPO_ROOT / "data/processed/transitions_std121/corpus_manifest.json"
BARS_PATH = REPO_ROOT / "src/diffusion/transition_eval/certify/bars.yaml"

BAR1_SENTENCE = (
    "Bar 1's accuracy conjunct (LOO 1-NN accuracy >= 0.80) was deleted by "
    "owner decision at the draft.8 inspection WITH THE OUTCOME KNOWN: the "
    "observed accuracy was 0.673, against a floor calibrated on the 47-clip / "
    "11-style v2 corpus (chance 0.213) and never re-derived for the 223-clip "
    "/ 39-class exam (chance 0.067). This deletion is outcome-aware and does "
    "not count as pre-registration for the data graded here; the surviving "
    "conjunct, Cohen's d >= 1.5, was pre-registered in draft.7/draft.8 and "
    "passed (d = 1.522) before the change was made. Accuracy remains a "
    "reported, non-gating exam statistic (0.673, ten times chance)."
)

MERGE_SENTENCE = (
    "Draft.8's bars 2 and 3 gated the same inequality (sibling app_ref > "
    "control app_ref) from opposite sides; they are merged into one bar 2 — "
    "per n>=4-eligible class, sibling > control AND M2a silent on the "
    "sibling, ALL eligible classes must pass (the old 35/37 and 37/39 count "
    "floors were arbitrary headroom; eligibility reuses the exam's n>=4 "
    "trust convention). DISCLOSED PLAINLY: this rule is outcome-aware — "
    "nature_bloom (n=2), draft.8's only floor inversion and only bar-2 miss, "
    "leaves the denominator under it; a 2-clip class yields exactly one "
    "sibling pair and no distributional basis for a hard claim, and the "
    "residual risk is documented in the nature_bloom note below. "
    "core_degenerate is removed from the certification bar path entirely "
    "(no conjunct, no silent logging); the flag stays live in S, the exam's "
    "mask-adoption criterion, and Block C descriptive rates."
)

NATURE_BLOOM_NOTE = (
    "nature_bloom residual risk (carried from the draft.8 record): its two "
    "clips are maximally content-diverse (exam R1 recall 0 at clip level, "
    "R2 pool recall 1.0); its lerp control scored ABOVE its sibling (0.596 "
    "vs 0.420) — the one draft.8 floor inversion, the poster child of the "
    "content-invariance alarm (pooled partial corr 0.82, non-gating). Under "
    "the 3.0.0 eligibility rule it is not graded; it remains scored, "
    "reported, and permanently untrusted in the trust map (n<4)."
)


def main() -> int:
    bars = yaml.safe_load(BARS_PATH.read_text())
    if not bars.get("frozen"):
        raise RuntimeError("bars.yaml is not frozen — regrade refused (SPEC §6.5)")
    ver = versioning.version()
    if "draft" in ver:
        raise RuntimeError(f"VERSION {ver} is a draft — this script writes the 3.0.0 record")

    old = json.loads((SRC_CERT / "record.json").read_text())
    corpus = load_corpus_manifest(CORPUS)
    rows = {r["item_id"]: r for r in
            map(json.loads, (SRC_CERT / "cert_siblings/items.jsonl").read_text().splitlines())}

    # --- changed bar 1: d-only, over the draft.8 exam data (winner unchanged) ------
    b1_old = old["exam"]["bar1"]
    bar1 = {"acc": b1_old["acc"], "d": b1_old["d"],
            "d_min": bars["exam"]["bar1_m1a_floor"]["d_min"]}
    bar1["pass"] = bool(bar1["d"] >= bar1["d_min"])

    # --- changed bar 2 (merged): deployed grader over the draft.8 score rows -------
    classes = sorted({v["class"] for v in corpus["clips"].values()
                      if corpus["classes"][v["class"]]["n_clips"] >= 2})
    min_n = bars["probes"]["siblings"]["bar2"]["eligibility_min_n"]
    eligible = [c for c in classes if corpus["classes"][c]["n_clips"] >= min_n]
    inelig = {c: corpus["classes"][c]["n_clips"] for c in classes if c not in eligible}
    g_sib = probes.grade_sibling_floor(rows, eligible, bars, inelig)

    # --- unchanged bars: verbatim carry (grader code byte-identical) ---------------
    carried = {k: old["grades"][k] for k in
               ("splices", "reversal", "m3_panel", "copy_twins", "bar8")}
    exam = {**old["exam"], "bar1": bar1}

    verdicts = {
        "bar1_m1a_floor": bar1["pass"],
        "bar2_sibling_floor": g_sib["pass"],
        "bar4_splices": old["verdicts"]["bar4_splices"],
        "bar5_reversal": old["verdicts"]["bar5_reversal"],
        "bar6_m3_panel": old["verdicts"]["bar6_m3_panel"],
        "bar7_copy_twins": old["verdicts"]["bar7_copy_twins"],
        "bar8_integration_determinism": old["verdicts"]["bar8_integration_determinism"],
    }
    overall = all(verdicts.values())

    out = REPO_ROOT / "outputs/eval/certification" / ver
    out.mkdir(parents=True, exist_ok=True)
    stamp = versioning.stamp(CORPUS)
    record = {
        "version": ver, "overall_pass": overall, "verdicts": verdicts,
        "regrade_of": {
            "run": "3.0.0-draft.8 (job 9465002, complete A-D, zero error rows)",
            "run_stamp": old["stamp"],
            "run_bars_sha256": old["bars_sha256"],
            "owner_directive": "closed-list bar revision; do not redo "
                               "already-done calculations — regrade, not re-run",
            "changed": ["bar1_m1a_floor (d-only)",
                        "bar2_sibling_floor (merged draft.8 bars 2+3)"],
            "carried_verbatim": list(carried),
            "disclosures": [BAR1_SENTENCE, MERGE_SENTENCE, NATURE_BLOOM_NOTE],
            "post_run_commits": "perf/representation commits after 31dd07e are "
                                "numeric no-ops verified bitwise on real score "
                                "rows (CHANGELOG 2026-07-14); no metric formula "
                                "changed between the run and this regrade",
        },
        "stamp": stamp, "bars_sha256": versioning.sha256_file(BARS_PATH),
        "exam": exam,
        "grades": {"sibling_floor": g_sib, **carried},
        "content_invariance": old["content_invariance"],
        "blockc": old["blockc"], "calibration": old["calibration"],
        "claims": old["claims"],
    }
    (out / "record.json").write_text(json.dumps(record, indent=1, default=str))

    md = [
        f"# Certification record — transition-eval/{ver}",
        "",
        f"**Overall: {'PASS' if overall else 'FAIL'}** · bars sha256 "
        f"`{record['bars_sha256'][:16]}…` · corpus sha256 "
        f"`{(stamp['corpus_sha256'] or '')[:16]}…` · regrade commit "
        f"`{stamp['git']['commit_short']}`",
        "",
        "## Provenance: regrade, not re-run",
        "",
        "All scores in this record were produced by the **3.0.0-draft.8 "
        "certification run** (job 9465002, commit `31dd07e`, clean tree, "
        "1h38m, every planned A–D item scored, zero error rows, zero grader "
        "crashes). The 3.0.0 revision changes only two grading rules; per "
        "owner directive no computation was repeated. The changed bars were "
        "re-graded by the deployed `certify` code over the run's committed "
        "artifacts; bars 4–8 grade payloads carry over verbatim (their "
        "grader code is byte-identical in this revision). Instrument commits "
        "between the run and this regrade are disclosed numeric no-ops, "
        "verified bitwise on real score rows (CHANGELOG 2026-07-14).",
        "",
        "## Owner-decided bar revision (closed list) — disclosures, verbatim",
        "",
        f"1. {BAR1_SENTENCE}",
        "",
        f"2. {MERGE_SENTENCE}",
        "",
        f"3. {NATURE_BLOOM_NOTE}",
        "",
        "| bar | verdict | data |",
        "|---|---|---|",
        f"| bar1_m1a_floor (d ≥ {bar1['d_min']}) | "
        f"{'PASS' if bar1['pass'] else 'FAIL'} | d {bar1['d']:.3f}; "
        f"acc {bar1['acc']:.3f} reported descriptively |",
        f"| bar2_sibling_floor (merged 2+3) | "
        f"{'PASS' if g_sib['pass'] else 'FAIL'} | {g_sib['n_pass']}/"
        f"{g_sib['n_eligible']} eligible (n≥4), all must pass; "
        f"{len(inelig)} classes n<4 scored-not-graded |",
    ]
    md += [f"| {k} | {'PASS' if v else 'FAIL'} | carried verbatim from draft.8 |"
           for k, v in verdicts.items() if k.startswith(("bar4", "bar5", "bar6",
                                                         "bar7", "bar8"))]
    md += [
        "",
        f"Classes outside the bar-2 denominator (n<4): "
        + ", ".join(f"{c} (n={n})" for c, n in sorted(inelig.items())) + ".",
        "",
        f"**What certification claims, exactly:** *{record['claims']}* It does "
        "not claim: that metrics track human judgment (M4 exempt until O9); "
        "that pools/masks behave identically on generated-domain frames "
        "(untestable without labels); M2c validity (first real training "
        "manifest); M1b absolute validity (injected-trajectory test = "
        "post-lock appendix).",
        "",
        "Draft.8 data highlights (unchanged by this revision): splices 74/74 "
        "at τ 0.88, gap 0.112, τ_copy recalibrated 0.858; reversal 12W/3L "
        "p=0.0176; swaps 37/37, hard-cuts 37/37; copy twins 11/11; warm rerun "
        "bitwise 0.0, cold anchors worst Δ 3.9e-4; content-invariance pooled "
        "partial corr 0.82 (alarm 0.4, non-gating); σ_seed PENDING (gates the "
        "first model report, not this tag).",
        "",
        "Artifacts: this record regrades `outputs/eval/certification/"
        "3.0.0-draft.8/` (exam/, analysis/, cert_*/items.jsonl, figures/, "
        "results_explorer.html — the full representation of the underlying "
        "data); the regraded record.json sits in `outputs/eval/certification/"
        f"{ver}/`. Reproduce with `PYTHONPATH=src python "
        "scripts/regrade_draft8_to_v3.py`.",
        "",
        "On PASS this record authorizes the annotated tag `eval/v3.0.0`; the "
        "stamp above was taken before the tag exists and says UNCERTIFIED for "
        "exactly that reason (the tag is created on the commit that carries "
        "this record).",
    ]
    rec_dir = REPO_ROOT / "src/diffusion/transition_eval/certifications"
    (rec_dir / f"v{ver}.md").write_text("\n".join(md))
    print(f"[regrade] overall={'PASS' if overall else 'FAIL'} "
          f"bar1 d={bar1['d']:.3f} (pass={bar1['pass']}) · "
          f"bar2 {g_sib['n_pass']}/{g_sib['n_eligible']} (pass={g_sib['pass']})")
    print(f"[regrade] record -> certifications/v{ver}.md + {out / 'record.json'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
