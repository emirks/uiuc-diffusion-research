#!/usr/bin/env python
"""CTT v2 caption distributional gate battery -- advisor A4 (round 8) Q4.

Runs A4's twelve pre-registered gates against the bars pinned in
`misc/ctt_v2_final/DOSSIER.md` §4 (pinned 2026-07-27, BEFORE any new caption
existed -- that ordering is what makes them pre-registered).

Gate #8 -- the function-word-only classifier probe -- is the load-bearing one:
it is the content-controlled version of "can text style identify the stratum",
and it subsumes #3/#10/#11.

INTERPRETER: needs scikit-learn, which is NOT in envs/diffusion.  Use
  /projects/illinois/eng/cs/jrehg/users/emirkisa/envs/nichescout/bin/python
(sklearn 1.7.2, numpy, requests).

Usage
-----
  $PY gate_battery.py --store <dir-with-records.json> --out <report.json> \
      [--llm-participial]     # gate #5 auditor-LLM classification (needs GEMINI_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caption_common import (  # noqa: E402
    all_word_tokens,
    audio_hits,
    colour_count,
    function_word_tokens,
    has_camera_phrase,
    has_ing_participle,
    jaccard,
    load_corpus_descriptions,
    markup_hits,
    opens_with_determiner,
    speech_action_hits,
    word_count,
    PUNCT_FEATURES,
)

# --------------------------------------------------------------------------
# Pinned bars -- DOSSIER §4
# --------------------------------------------------------------------------
BARS = {
    "p50_range": (29, 36),
    "p10_range": (16, 26),
    "p90_range": (34, 44),
    "det_A_min": 86.4,
    "det_B_min": 86.9,
    "B_lowercase_min": 100.0,
    "B_ing_min": 80.6,
    "B_participial_llm_delta_pp": 10.0,
    "audio_max": 0,
    "dup_rate_max_pct": 2.0,
    "gate8_bacc_max": 0.65,          # ORIGINAL -- superseded, kept for 8c
    "gate8a_max": 0.73,             # re-pinned drift guard (round-9 ruling)
    "gate8b_max": 0.60,             # re-pinned load-bearing: stratum-internal blindness
    "gate9_auc_investigate": 0.80,
    "colour_band": (1.579, 4.737),
    "camera_corpus_pct": 3.51,
    "camera_delta_pp": 10.0,
}
CORPUS = {
    "p10": 21, "p25": 29, "p50": 33, "p75": 36, "p90": 39,
    "det_A_pct": 96.4, "det_B_pct": 96.9,
    "B_lowercase_pct": 100.0, "B_ing_pct": 90.6,
    "colour_mean_pinned": 3.158, "camera_pct": 3.51, "audio": 0, "dups": 0,
}


def pcts(vals):
    v = np.asarray(vals, dtype=float)
    return {q: float(np.percentile(v, q)) for q in (10, 25, 50, 75, 90)}


# --------------------------------------------------------------------------
# Gate #8 / #9 -- classifier probes
# --------------------------------------------------------------------------
def _numeric_features(texts):
    rows = []
    for t in texts:
        low = t
        rows.append([low.count(p) for p in PUNCT_FEATURES] + [word_count(t), len(t)])
    return np.asarray(rows, dtype=float)


def classifier_probe(corpus_texts, new_texts, analyzer, seeds=(0, 1, 2, 3, 4),
                     n_folds=5, use_numeric=True, report_features=False):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from scipy.sparse import hstack, csr_matrix

    n = min(len(corpus_texts), len(new_texts))
    baccs, aucs = [], []
    feat_weight = Counter()

    for seed in seeds:
        rng = np.random.RandomState(seed)
        c_idx = rng.choice(len(corpus_texts), n, replace=False)
        n_idx = rng.choice(len(new_texts), n, replace=False)
        X = [corpus_texts[i] for i in c_idx] + [new_texts[i] for i in n_idx]
        y = np.array([0] * n + [1] * n)
        num = _numeric_features(X)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            vec = TfidfVectorizer(analyzer=analyzer, min_df=1, sublinear_tf=True)
            Xtr = vec.fit_transform([X[i] for i in tr])
            Xte = vec.transform([X[i] for i in te])
            if use_numeric:
                sc = StandardScaler().fit(num[tr])
                Xtr = hstack([Xtr, csr_matrix(sc.transform(num[tr]))]).tocsr()
                Xte = hstack([Xte, csr_matrix(sc.transform(num[te]))]).tocsr()
            clf = LogisticRegression(max_iter=4000, C=1.0)
            clf.fit(Xtr, y[tr])
            pred = clf.predict(Xte)
            baccs.append(balanced_accuracy_score(y[te], pred))
            try:
                aucs.append(roc_auc_score(y[te], clf.decision_function(Xte)))
            except ValueError:
                pass
            if report_features:
                names = list(vec.get_feature_names_out()) + (
                    [f"PUNCT[{p}]" for p in PUNCT_FEATURES] + ["n_words", "n_chars"]
                    if use_numeric else []
                )
                for nm, w in zip(names, clf.coef_[0]):
                    feat_weight[nm] += w / (len(seeds) * n_folds)

    out = {
        "mean_balanced_accuracy": float(np.mean(baccs)),
        "std_balanced_accuracy": float(np.std(baccs)),
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "n_per_class": n, "n_fits": len(baccs),
    }
    if report_features:
        top_new = sorted(feat_weight.items(), key=lambda kv: -kv[1])[:20]
        top_corpus = sorted(feat_weight.items(), key=lambda kv: kv[1])[:20]
        out["top_features_toward_NEW"] = [[k, round(v, 4)] for k, v in top_new]
        out["top_features_toward_CORPUS"] = [[k, round(v, 4)] for k, v in top_corpus]
    return out


# --------------------------------------------------------------------------
# Gate #5 -- auditor-LLM participial-NP classification
# --------------------------------------------------------------------------
PARTICIPIAL_MODEL = "gemini-3.5-flash"
PARTICIPIAL_Q = (
    "Is the following text a NOUN PHRASE whose action is expressed with -ing participles "
    "(e.g. \"a woman in a gray coat sipping coffee beside a window\"), rather than a "
    "complete finite sentence (e.g. \"A woman sips coffee.\")? "
    'Answer JSON {"participial_np": "YES"/"NO"}.\n\nText: '
)


def llm_participial_rate(texts, workers=100):
    import requests
    from concurrent.futures import ThreadPoolExecutor
    key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{PARTICIPIAL_MODEL}:generateContent"

    def one(t):
        body = {
            "contents": [{"role": "user", "parts": [{"text": PARTICIPIAL_Q + t}]}],
            "generationConfig": {
                "temperature": 0.0, "maxOutputTokens": 60,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {"participial_np": {"type": "STRING", "enum": ["YES", "NO"]}},
                    "required": ["participial_np"],
                },
                "thinkingConfig": {"thinkingLevel": "minimal"},
            },
        }
        for _ in range(4):
            r = requests.post(url, headers={"x-goog-api-key": key,
                                            "Content-Type": "application/json"},
                              json=body, timeout=120)
            if r.status_code == 200:
                try:
                    j = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                    return json.loads(j).get("participial_np")
                except Exception:
                    return None
            if r.status_code == 429:
                return "HTTP429"
        return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        verdicts = list(ex.map(one, texts))
    yes = sum(1 for v in verdicts if v == "YES")
    ok = sum(1 for v in verdicts if v in ("YES", "NO"))
    return {"yes": yes, "n_classified": ok, "n": len(texts),
            "rate_pct": 100.0 * yes / ok if ok else None,
            "errors": sum(1 for v in verdicts if v not in ("YES", "NO"))}


# --------------------------------------------------------------------------
# Battery
# --------------------------------------------------------------------------
def describe_set(texts):
    if not texts:
        return {}
    w = [word_count(t) for t in texts]
    p = pcts(w)
    dups = len(texts) - len(set(texts))
    nd = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if jaccard(texts[i], texts[j]) > 0.8:
                nd += 1
    npairs = len(texts) * (len(texts) - 1) // 2
    return {
        "n": len(texts),
        "words_p10": round(p[10], 1), "words_p25": round(p[25], 1),
        "words_p50": round(p[50], 1), "words_p75": round(p[75], 1),
        "words_p90": round(p[90], 1),
        "words_min": min(w), "words_max": max(w), "words_mean": round(float(np.mean(w)), 2),
        "determiner_pct": round(100 * sum(opens_with_determiner(t) for t in texts) / len(texts), 1),
        "uppercase_initial_pct": round(100 * sum(t[:1].isupper() for t in texts) / len(texts), 1),
        "lowercase_initial_pct": round(100 * sum(t[:1].islower() for t in texts) / len(texts), 1),
        "ing_participle_pct": round(100 * sum(has_ing_participle(t) for t in texts) / len(texts), 1),
        "colour_mean": round(float(np.mean([colour_count(t) for t in texts])), 3),
        "colour_ge1_pct": round(100 * sum(colour_count(t) >= 1 for t in texts) / len(texts), 1),
        "camera_pct": round(100 * sum(has_camera_phrase(t) for t in texts) / len(texts), 2),
        "audio_hits": sum(1 for t in texts if audio_hits(t)),
        "speech_action_hits": sum(1 for t in texts if speech_action_hits(t)),
        "markup_hits": sum(1 for t in texts if markup_hits(t)),
        "exact_dup_count": dups,
        "exact_dup_pct": round(100 * dups / len(texts), 2),
        "near_dup_pairs_jaccard_gt_0.8": nd,
        "near_dup_pair_pct": round(100 * nd / npairs, 3) if npairs else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="dir containing records.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm-participial", action="store_true")
    a = ap.parse_args()

    recs = json.loads((Path(a.store) / "records.json").read_text())
    accepted = [r for r in recs.values() if r.get("description")]
    cA, cB = load_corpus_descriptions()
    corpus_all = cA + cB

    # ---- strata: bank x role, bank, pooled --------------------------------
    groups = {}
    for r in accepted:
        groups.setdefault(f"{r['bank']}|{r['role']}", []).append(r["description"])
    for r in accepted:
        groups.setdefault(f"{r['bank']}|ALL", []).append(r["description"])
    groups["NEW|ALL"] = [r["description"] for r in accepted]
    groups["NEW|A"] = [r["description"] for r in accepted if r["role"] == "A"]
    groups["NEW|B"] = [r["description"] for r in accepted if r["role"] == "B"]

    report = {
        "bars": BARS, "corpus_pinned": CORPUS,
        "corpus_measured": {
            "ALL": describe_set(corpus_all),
            "A": describe_set(cA),
            "B": describe_set(cB),
        },
        "strata": {k: describe_set(v) for k, v in sorted(groups.items())},
    }

    # ---- HARD/REVIEW/FLAG verdicts on the pooled new set ------------------
    new_all = groups["NEW|ALL"]
    new_A = groups["NEW|A"]
    new_B = groups["NEW|B"]
    s = report["strata"]["NEW|ALL"]
    gates = []

    lo, hi = BARS["p50_range"]
    gates.append(dict(gate=1, name="word-count p50", value=s["words_p50"],
                      corpus=CORPUS["p50"], bar=f"p50 in [{lo}, {hi}]",
                      verdict="PASS" if lo <= s["words_p50"] <= hi else "FAIL", type="HARD"))

    l10, h10 = BARS["p10_range"]; l90, h90 = BARS["p90_range"]
    ok2 = (l10 <= s["words_p10"] <= h10) and (l90 <= s["words_p90"] <= h90)
    gates.append(dict(gate=2, name="word-count p10 / p90",
                      value=f'{s["words_p10"]} / {s["words_p90"]}',
                      corpus=f'{CORPUS["p10"]} / {CORPUS["p90"]}',
                      bar=f"p10 in [{l10},{h10}] and p90 in [{l90},{h90}]",
                      verdict="PASS" if ok2 else "FAIL", type="HARD"))

    dA = report["strata"]["NEW|A"]["determiner_pct"]
    dB = report["strata"]["NEW|B"]["determiner_pct"]
    ok3 = dA >= BARS["det_A_min"] and dB >= BARS["det_B_min"]
    gates.append(dict(gate=3, name="opens with determiner (A / B)",
                      value=f"{dA}% / {dB}%",
                      corpus=f'{CORPUS["det_A_pct"]}% / {CORPUS["det_B_pct"]}%',
                      bar=f'>= {BARS["det_A_min"]}% / >= {BARS["det_B_min"]}%',
                      verdict="PASS" if ok3 else "FAIL", type="HARD"))

    lcB = report["strata"]["NEW|B"]["lowercase_initial_pct"]
    ucA = report["strata"]["NEW|A"]["uppercase_initial_pct"]
    gates.append(dict(gate=4, name="B-role lowercase-initial (and A-role uppercase-initial)",
                      value=f"B {lcB}% / A {ucA}%", corpus="B 100% / A 100%",
                      bar="both 100%",
                      verdict="PASS" if (lcB == 100.0 and ucA == 100.0) else "FAIL",
                      type="HARD"))

    ingB = report["strata"]["NEW|B"]["ing_participle_pct"]
    g5 = dict(gate=5, name="B-role participial-NP", value=f"{ingB}% (-ing regex)",
              corpus=f'{CORPUS["B_ing_pct"]}% (-ing regex)',
              bar=f'>= {BARS["B_ing_min"]}% (pinned regex form)',
              verdict="PASS" if ingB >= BARS["B_ing_min"] else "FAIL", type="HARD")
    if a.llm_participial:
        cal = llm_participial_rate(cB)
        new = llm_participial_rate(new_B)
        g5["llm_corpus_calibration"] = cal
        g5["llm_new"] = new
        if cal["rate_pct"] is not None and new["rate_pct"] is not None:
            bar_llm = cal["rate_pct"] - BARS["B_participial_llm_delta_pp"]
            g5["llm_bar_pct"] = round(bar_llm, 1)
            g5["llm_verdict"] = "PASS" if new["rate_pct"] >= bar_llm else "FAIL"
            if g5["llm_verdict"] == "FAIL":
                g5["verdict"] = "FAIL"
    gates.append(g5)

    audio_n = s["audio_hits"]; markup_n = s["markup_hits"]
    gates.append(dict(gate=6, name="audio words / Style: prefix / markdown / quotes",
                      value=f"audio {audio_n}, markup {markup_n} "
                            f'(visible-speech actions, not gated: {s["speech_action_hits"]})',
                      corpus="0 / 0", bar="0",
                      verdict="PASS" if (audio_n == 0 and markup_n == 0) else "FAIL",
                      type="HARD"))

    worst_dup = max((report["strata"][k]["exact_dup_pct"]
                     for k in report["strata"] if "|" in k and k.endswith(("|A", "|B"))),
                    default=0.0)
    gates.append(dict(gate=7, name="exact-duplicate rate within stratum",
                      value=f"{worst_dup}% (worst bank x role cell)", corpus="0%",
                      bar="< 2%", verdict="PASS" if worst_dup < 2.0 else "FAIL", type="HARD"))

    # ---- gate 8: RE-PINNED by the round-9 advisor into 8a / 8b / 8c ----------
    # 8c records the original absolute <=0.65 bar as FAILED, never quietly replaced.
    g8 = classifier_probe(corpus_all, new_all, analyzer=function_word_tokens,
                          report_features=True)
    b8 = g8["mean_balanced_accuracy"]
    gates.append(dict(gate="8a", name="corpus-vs-new function-word probe (DRIFT GUARD)",
                      value=round(b8, 4), corpus="0.50 = chance",
                      bar=f'<= {BARS["gate8a_max"]} (above this cannot be the known '
                          "fingerprint => bug: mixed prompts / wrong store / contamination)",
                      verdict="PASS" if b8 <= BARS["gate8a_max"] else "FAIL",
                      type="HARD", detail=g8))

    # 8b: stratum-internal style blindness -- the load-bearing replacement.
    bank_groups = {}
    for r in accepted:
        bank_groups.setdefault(r["bank"], []).append(r["description"])
    banks = sorted(bank_groups)
    if len(banks) >= 2:
        g8b = classifier_probe(bank_groups[banks[0]], bank_groups[banks[1]],
                               analyzer=function_word_tokens, report_features=True)
        b8b = g8b["mean_balanced_accuracy"]
        gates.append(dict(gate="8b", name=f"stratum-internal blindness: "
                                         f"{banks[0]} vs {banks[1]} (LOAD-BEARING)",
                          value=round(b8b, 4), corpus="0.506 = measured NULL",
                          bar=f'<= {BARS["gate8b_max"]}',
                          verdict="PASS" if b8b <= BARS["gate8b_max"] else "FAIL",
                          type="HARD", detail=g8b))
    else:
        gates.append(dict(gate="8b", name="stratum-internal blindness",
                          value=None, corpus="0.506 = measured NULL",
                          bar=f'<= {BARS["gate8b_max"]}',
                          verdict="SKIPPED (need >=2 banks in the store)", type="HARD"))

    gates.append(dict(gate="8c", name="ORIGINAL absolute bar (superseded, recorded not replaced)",
                      value=round(b8, 4), corpus="0.50 = chance",
                      bar="<= 0.65  [SUPERSEDED by 8a/8b -- round-9 ruling]",
                      verdict="FAIL (recorded; the bar was in an unreachable reference frame: "
                              "it demanded a different model generation land within 0.008 of the "
                              "corpus's own internal A-vs-B register distance 0.6419, while a "
                              "prompt delta inside one model opens 0.7233)",
                      type="RECORD"))

    g9 = classifier_probe(corpus_all, new_all, analyzer=all_word_tokens,
                          use_numeric=True, report_features=True)
    gates.append(dict(gate=9, name="full-vocabulary classifier",
                      value=round(g9["mean_auc"], 4) if g9["mean_auc"] else None,
                      corpus="0.50 = chance", bar="AUC >= 0.80 triggers feature investigation",
                      verdict="INVESTIGATE" if (g9["mean_auc"] or 0) >= 0.80 else "PASS",
                      type="REVIEW", detail=g9))

    clo, chi = BARS["colour_band"]
    gates.append(dict(gate=10, name="colour-term density",
                      value=s["colour_mean"], corpus=CORPUS["colour_mean_pinned"],
                      bar=f"in [{clo}, {chi}]",
                      verdict="PASS" if clo <= s["colour_mean"] <= chi else "FAIL",
                      type="REVIEW"))

    camd = abs(s["camera_pct"] - CORPUS["camera_pct"])
    gates.append(dict(gate=11, name="camera-phrase rate", value=f'{s["camera_pct"]}%',
                      corpus=f'{CORPUS["camera_pct"]}%', bar="within +/-10 pp",
                      verdict="PASS" if camd <= 10.0 else "FLAG", type="FLAG"))

    gates.append(dict(gate=12, name="near-duplicate rate (token Jaccard > 0.8)",
                      value=f'{s["near_dup_pairs_jaccard_gt_0.8"]} pairs '
                            f'({s["near_dup_pair_pct"]}% of pairs)',
                      corpus=f'{report["corpus_measured"]["ALL"]["near_dup_pairs_jaccard_gt_0.8"]} pairs',
                      bar="report only", verdict="REPORT", type="FLAG"))

    report["gates"] = gates
    report["summary"] = {
        "hard_fail": [g["gate"] for g in gates
                      if g["type"] == "HARD" and str(g["verdict"]).startswith("FAIL")],
        "gate8a_corpus_vs_new": round(b8, 4),
        "gate8b_stratum_internal": next((g["value"] for g in gates if g["gate"] == "8b"), None),
        "gate9_auc": round(g9["mean_auc"], 4) if g9["mean_auc"] else None,
    }
    Path(a.out).write_text(json.dumps(report, indent=1))

    # human-readable table
    print(f'{"#":<3}{"gate":<52}{"corpus":<18}{"new":<26}{"bar":<44}{"verdict"}')
    for gt in gates:
        print(f'{gt["gate"]:<3}{gt["name"][:50]:<52}{str(gt["corpus"])[:16]:<18}'
              f'{str(gt["value"])[:24]:<26}{str(gt["bar"])[:42]:<44}{gt["verdict"]}')
    print()
    print("GATE 8a corpus-vs-new  :", round(b8, 4),
          "+/-", round(g8["std_balanced_accuracy"], 4),
          f'({g8["n_fits"]} fits, {g8["n_per_class"]} per class)   bar <= {BARS["gate8a_max"]}')
    print("GATE 8b stratum-internal:",
          next((g["value"] for g in gates if g["gate"] == "8b"), None),
          f'  bar <= {BARS["gate8b_max"]}  (LOAD-BEARING)')
    print("HARD FAILURES:", report["summary"]["hard_fail"] or "none")


if __name__ == "__main__":
    main()
