#!/usr/bin/env python
"""Evaluate A12's six pre-registered packing-pilot conditions.  No API calls; free.

INTERPRETER: needs scikit-learn -> $LAB/envs/nichescout/bin/python

Conditions (A12 §3, bars pinned BEFORE the pilot ran):
  1. Full 12-gate battery on the packed store (8a <= 0.73, 8b <= 0.60)  -> gate_battery.py
  2. Packed-vs-unpacked function-word probe on the 100 paired items: <= 0.60 HARD;
     0.60-0.73 REVIEW; > 0.73 packing rejected.
  3. Cross-contamination HARD: (a) within-pack derangement flagged >= 99% with correct
     positional attribution; (b) matched-audit first-pass flag rate <= 10%; (c) ID echo
     intact on 100% of items.
  4. Contamination ADVISORY: CLIP diagonal argmax >= 98% (pack_clip_check.py) and
     within-pack vs cross-pack lexical-overlap ratio <= 1.15.
  5. First-pass rate >= 97% on the PROMPT-CONTROLLABLE scope: leak + format + Tier-1 +
     audio/style only.  `inaccurate` is generator perception and out of scope per A8.
  6. Audit packing gated separately by 3(a).

Usage
-----
  $LAB/envs/nichescout/bin/python pack_analysis.py --dir <pilot dir> [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from caption_common import (  # noqa: E402
    audio_hits, function_word_tokens, jaccard, markup_hits,
)
from gate_battery import classifier_probe  # noqa: E402

ROUND2 = HERE / "pilot_m3/round2"


# --------------------------------------------------------------------------
def verdict_of(v: dict) -> bool:
    """Did the auditor flag this item at all?"""
    return v.get("leak") == "YES" or v.get("inaccurate") == "YES"


def load(d: Path, name: str) -> list[dict]:
    p = d / name
    return [json.loads(x) for x in p.open()] if p.exists() else []


def ngram_set(text: str, n: int) -> set:
    w = [t for t in text.lower().replace(",", " ").split() if t]
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def ngram_jaccard(a: str, b: str, n: int) -> float:
    sa, sb = ngram_set(a, n), ngram_set(b, n)
    return len(sa & sb) / len(sa | sb) if (sa and sb) else 0.0


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    D = Path(a.dir)
    rows = json.loads((D / "packed_rows.json").read_text())
    r2 = json.loads((ROUND2 / "records.json").read_text())
    r2desc = {(v["clip_id"], v["role"]): v["description"]
              for v in r2.values() if v.get("description")}
    R: dict = {"n_items": len(rows), "bars": {
        "cond2_hard": 0.60, "cond2_reject": 0.73, "cond3a_min_pct": 99.0,
        "cond3b_max_pct": 10.0, "cond3c_pct": 100.0, "cond4_clip_min_pct": 98.0,
        "cond4_lex_ratio_max": 1.15, "cond5_min_pct": 97.0}}

    # ---- matched audits (unpacked, A8's validated shape) -------------------
    matched = load(D, "raw_matched_audit_responses.jsonl")
    mv = {(x["clip_id"], x["role"]): (x.get("verdict") or {}) for x in matched}
    for r in rows:
        r["audit"] = mv.get((r["clip_id"], r["role"]), {})

    # ======================= CONDITION 5 ==================================
    # A8's scope ruling: prompt-controllable failures only.  `inaccurate` is excluded
    # (perceptual attribute errors are not prompt-fixable) and gets its own governance.
    pc_fail, insc = [], 0
    for r in rows:
        f = []
        if not r["description"]:
            f.append("no_text")
        else:
            if r["format_violations"]:
                f.append("format:" + ",".join(r["format_violations"]))
            if r["tier1"]:
                f.append("tier1")
            if audio_hits(r["description"]):
                f.append("audio")
            if markup_hits(r["description"]):
                f.append("markup/style")
            if r["audit"].get("leak") == "YES":
                f.append("leak")
        if r["audit"].get("inaccurate") == "YES":
            insc += 1
        r["pc_fail"] = f
        if f:
            pc_fail.append(r)
    n = len(rows)
    fp = 100.0 * (n - len(pc_fail)) / n
    R["cond5_first_pass_prompt_controllable"] = {
        "n": n, "failures": len(pc_fail), "rate_pct": round(fp, 2),
        "bar": ">= 97%", "verdict": "PASS" if fp >= 97.0 else "FAIL",
        "breakdown": {k: sum(1 for r in pc_fail if any(x.startswith(k) for x in r["pc_fail"]))
                      for k in ("no_text", "format", "tier1", "audio", "markup/style", "leak")},
        "inaccurate_out_of_scope": {"n": insc, "rate_pct": round(100.0 * insc / n, 2),
                                    "governance": "<=8% REVIEW; 0 unresolved in final store HARD",
                                    "round2_baseline_pct": 5.75}}

    # ======================= CONDITION 3 ==================================
    # (c) ID echo
    intact = sum(r["id_echo_intact"] for r in rows)
    order = sum(r["order_preserved"] for r in rows)
    R["cond3c_id_echo"] = {
        "n": n, "id_echo_intact": intact, "intact_pct": round(100.0 * intact / n, 2),
        "order_preserved": order, "order_preserved_pct": round(100.0 * order / n, 2),
        "bar": "intact on 100% of items",
        "verdict": "PASS" if intact == n else "FAIL",
        "note": ("every code was echoed exactly once (0 missing / 0 extra / 0 corrupted), so the "
                 "echo is intact; but 1 pack of 20 returned two ADJACENT items transposed, so "
                 "ARRAY POSITION is not a safe key -- rows are keyed by the echoed id. Whether "
                 "the model's own id or its position tells the truth is decided by cond4 CLIP.")}
    # (b) matched-audit first-pass flag rate
    flags = [(r["clip_id"], r["role"]) for r in rows if verdict_of(r["audit"])]
    fr = 100.0 * len(flags) / n
    R["cond3b_matched_flag_rate"] = {
        "n": n, "flagged": len(flags), "rate_pct": round(fr, 2), "bar": "<= 10%",
        "round2_baseline_pct": 5.75, "verdict": "PASS" if fr <= 10.0 else "FAIL",
        "auditor_errors": sum(1 for r in rows if not r["audit"])}

    # (a) within-pack derangement, PACKED audits, keyed by echoed id
    pk = load(D, "raw_packed_audit_responses.jsonl")
    per_pack, tot = [], {"DERANGED": [0, 0], "MATCHED": [0, 0]}
    echo_bad = 0
    for x in pk:
        parsed = x.get("parsed") or []
        by = {}
        for o in parsed:
            if isinstance(o, dict) and o.get("id") is not None:
                by.setdefault(str(o["id"]), []).append(o)
        truth_d, flag_d = set(), set()
        for it in x["items"]:
            cand = by.get(it["code"], [])
            if len(cand) != 1:
                echo_bad += 1
                continue
            v = cand[0]
            f = verdict_of(v)
            tot[it["truth"]][0] += 1
            tot[it["truth"]][1] += int(f)
            if it["truth"] == "DERANGED":
                truth_d.add(it["code"])
            if f:
                flag_d.add(it["code"])
        per_pack.append({"pack_id": x["pack_id"], "exact_attribution": flag_d == truth_d,
                         "n_deranged": len(truth_d), "n_flagged": len(flag_d),
                         "missed": sorted(truth_d - flag_d),
                         "false_flags": sorted(flag_d - truth_d)})
    dn, df = tot["DERANGED"]
    mn, mf = tot["MATCHED"]
    dpct = 100.0 * df / dn if dn else 0.0
    exact = sum(1 for p in per_pack if p["exact_attribution"])
    R["cond3a_packed_derangement"] = {
        "n_deranged": dn, "deranged_flagged": df, "deranged_flag_pct": round(dpct, 2),
        "bar_deranged": ">= 99%",
        "n_matched_in_packed": mn, "matched_flagged_in_packed": mf,
        "matched_flag_pct_in_packed": round(100.0 * mf / mn, 2) if mn else None,
        "packs_with_exact_positional_attribution": f"{exact}/{len(per_pack)}",
        "exact_attribution_pct": round(100.0 * exact / len(per_pack), 2) if per_pack else None,
        "id_echo_failures_in_packed_audits": echo_bad,
        "verdict": "PASS" if (dpct >= 99.0 and exact == len(per_pack)) else "FAIL",
        "per_pack": per_pack}
    R["cond3_overall"] = "PASS" if all(
        R[k]["verdict"] == "PASS" for k in
        ("cond3a_packed_derangement", "cond3b_matched_flag_rate", "cond3c_id_echo")) else "FAIL"

    # ======================= CONDITION 2 ==================================
    # Paired: same (clip, role), same model, same prompt -- ONLY packing differs.
    acc = {(r["clip_id"], r["role"]): r["description"] for r in rows
           if r["description"] and not r["pc_fail"] and not verdict_of(r["audit"])}
    paired = [k for k in acc if k in r2desc and
              any(r["arm"] == "paired" for r in rows if (r["clip_id"], r["role"]) == k)]
    pk_txt = [acc[k] for k in paired]
    un_txt = [r2desc[k] for k in paired]
    probe = classifier_probe(un_txt, pk_txt, analyzer=function_word_tokens,
                             report_features=True)
    b = probe["mean_balanced_accuracy"]
    verdict = ("PASS" if b <= 0.60 else
               "REVIEW-ESCALATE" if b <= 0.73 else "FAIL-PACKING-REJECTED")
    # NULL at the SAME n and the same everything-but-clip-identity: round-2 unpacked
    # descriptions of paired items vs round-2 unpacked descriptions of other items.
    others = [v for k, v in r2desc.items() if k not in set(paired)]
    null_same_shape = classifier_probe(un_txt, others[:len(un_txt)],
                                       analyzer=function_word_tokens)
    fresh = [acc[k] for k in acc if k not in set(paired)]
    null_packed = classifier_probe(pk_txt, fresh, analyzer=function_word_tokens)
    R["cond2_packed_vs_unpacked_probe"] = {
        "n_paired": len(paired), "balanced_accuracy": round(b, 4),
        "std": round(probe["std_balanced_accuracy"], 4),
        "bar": "<= 0.60 HARD; 0.60-0.73 REVIEW; > 0.73 packing rejected",
        "verdict": verdict,
        "controls": {
            "NULL_same_shape_unpacked_vs_unpacked_other_clips":
                round(null_same_shape["mean_balanced_accuracy"], 4),
            "NULL_packed_paired_vs_packed_fresh":
                round(null_packed["mean_balanced_accuracy"], 4),
            "reference_round2_stratum_internal_8b": 0.5518,
            "reference_measured_NULL": 0.506},
        "top_features_toward_PACKED": probe["top_features_toward_NEW"][:10],
        "top_features_toward_UNPACKED": probe["top_features_toward_CORPUS"][:10]}

    # ======================= CONDITION 4 (lexical half) ===================
    # Packs are role- AND bank-homogeneous, so cross-pack pairs are restricted to the SAME
    # (role, bank) cell -- otherwise the ratio would measure role/bank difference, not echo.
    bypack: dict[str, list] = {}
    meta: dict[str, tuple] = {}
    for r in rows:
        if r["description"] and not r["pc_fail"]:
            bypack.setdefault(r["pack_id"], []).append(r["description"])
            meta[r["pack_id"]] = (r["role"], r["bank"])
    lex = {}
    for label, fn in (("token_jaccard", lambda x, y: jaccard(x, y)),
                      ("bigram_jaccard", lambda x, y: ngram_jaccard(x, y, 2)),
                      ("trigram_jaccard", lambda x, y: ngram_jaccard(x, y, 3))):
        win = [fn(x, y) for v in bypack.values() for x, y in combinations(v, 2)]
        cro = []
        pids = sorted(bypack)
        for i, p1 in enumerate(pids):
            for p2 in pids[i + 1:]:
                if meta[p1] != meta[p2]:
                    continue                      # control for role x bank
                cro += [fn(x, y) for x in bypack[p1] for y in bypack[p2]]
        w, c = st.mean(win), st.mean(cro)
        lex[label] = {"within_pack_mean": round(w, 5), "cross_pack_mean": round(c, 5),
                      "n_within_pairs": len(win), "n_cross_pairs": len(cro),
                      "ratio": round(w / c, 4) if c else None,
                      "verdict": ("PASS" if c and w / c <= 1.15 else "REVIEW")}
    R["cond4_lexical_overlap"] = {
        "bar": "within/cross ratio <= 1.15 (ADVISORY; exceedance => REVIEW, not fail)",
        "primary": "token_jaccard", **lex}

    # ======================= c_desc =======================================
    def toks(fn):
        return sum(x["raw_response"]["usageMetadata"]["totalTokenCount"]
                   for x in load(D, fn) if x.get("raw_response"))
    g, mm, pp = toks("raw_generation_responses.jsonl"), \
        toks("raw_matched_audit_responses.jsonl"), toks("raw_packed_audit_responses.jsonl")
    R["c_desc_measured"] = {
        "n_descriptions": n,
        "packed_generation_tok_per_desc": round(g / n, 1),
        "unpacked_audit_tok_per_desc": round(mm / n, 1),
        "packed_audit_tok_per_desc": round(pp / n, 1),
        "c_desc_packed_gen_plus_unpacked_audit": round((g + mm) / n, 1),
        "c_desc_packed_gen_plus_packed_audit": round((g + pp) / n, 1),
        "round2_unpacked_baseline": 682.0,
        "round2_generation_only": 472.2, "round2_audit_only": 209.4,
        "saving_vs_round2_pct": round(100 * (1 - ((g + mm) / n) / 682.0), 1),
        "note": "A12 was right that the audit barely amortises: the description text itself "
                "is per-item, so packed audit is only ~7% cheaper than unpacked."}

    if a.out:
        Path(a.out).write_text(json.dumps(R, indent=1))

    # ---- report ----------------------------------------------------------
    def line(k, val, bar, v):
        print(f"  {k:<46} {str(val):<22} {bar:<34} {v}")
    print("\n=== A12 PACKING PILOT — PRE-REGISTERED CONDITIONS ===")
    print(f"  {'condition':<46} {'measured':<22} {'bar':<34} verdict")
    c2 = R["cond2_packed_vs_unpacked_probe"]
    line("2 packed-vs-unpacked function-word probe", c2["balanced_accuracy"],
         "<=0.60 HARD (>0.73 reject)", c2["verdict"])
    c3a = R["cond3a_packed_derangement"]
    line("3a packed derangement flagged", f'{c3a["deranged_flag_pct"]}% ({c3a["deranged_flagged"]}/{c3a["n_deranged"]})',
         ">=99%", c3a["verdict"])
    line("3a  exact positional attribution",
         c3a["packs_with_exact_positional_attribution"], "all packs", c3a["verdict"])
    c3b = R["cond3b_matched_flag_rate"]
    line("3b matched-audit first-pass flag rate", f'{c3b["rate_pct"]}%', "<=10%", c3b["verdict"])
    c3c = R["cond3c_id_echo"]
    line("3c ID echo intact", f'{c3c["intact_pct"]}% ({c3c["id_echo_intact"]}/{n})',
         "100%", c3c["verdict"])
    line("3c  (order preserved, diagnostic)", f'{c3c["order_preserved_pct"]}%',
         "not a bar", "-")
    for k in ("token_jaccard", "bigram_jaccard", "trigram_jaccard"):
        d = R["cond4_lexical_overlap"][k]
        line(f"4 lexical overlap ratio [{k}]",
             f'{d["ratio"]} ({d["within_pack_mean"]}/{d["cross_pack_mean"]})',
             "<=1.15 ADVISORY", d["verdict"])
    c5 = R["cond5_first_pass_prompt_controllable"]
    line("5 first-pass (prompt-controllable scope)", f'{c5["rate_pct"]}%', ">=97%", c5["verdict"])
    print(f"\n  cond2 controls: NULL(same shape, other clips) = "
          f'{c2["controls"]["NULL_same_shape_unpacked_vs_unpacked_other_clips"]} | '
          f'NULL(packed paired vs packed fresh) = {c2["controls"]["NULL_packed_paired_vs_packed_fresh"]} '
          f'| n_paired={c2["n_paired"]}')
    print(f'  cond3b breakdown: leak/inaccurate flags {c3b["flagged"]}/{n}; '
          f'inaccurate alone {R["cond5_first_pass_prompt_controllable"]["inaccurate_out_of_scope"]["n"]}')
    print(f'  cond5 failure breakdown: {c5["breakdown"]}')
    print(f'  packed audit: matched-side false-flag rate {c3a["matched_flag_pct_in_packed"]}% '
          f'on {c3a["n_matched_in_packed"]} in-pack matched items')
    cd = R["c_desc_measured"]
    print(f'\n  c_desc: packed gen {cd["packed_generation_tok_per_desc"]} + unpacked audit '
          f'{cd["unpacked_audit_tok_per_desc"]} = {cd["c_desc_packed_gen_plus_unpacked_audit"]} '
          f'tok/desc  ({cd["saving_vs_round2_pct"]}% below the 682 round-2 baseline)')
    print(f'  c_desc all-packed = {cd["c_desc_packed_gen_plus_packed_audit"]} tok/desc')


if __name__ == "__main__":
    main()
