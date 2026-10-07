#!/usr/bin/env python3
"""Analyse SEGUE human-study ratings, protocol v5 (stdlib only).

Protocol v5 (misc/2026-09-22_vlm_judge_ab/DESIGN_v5.md, "Human page mirror"): per pair
two A/B answers (transition, overall) and three flags per video (no_effect, endpoint,
leak). The gate is applied per rating from that rater's own no-effect flags (done by
serve.py and stored as segue_score): SEGUE has an effect and the opponent none -> 1;
the reverse -> 0; neither -> 0.5; both -> the A/B answer stands.

Reports
  - per opponent x question (plus per task and overall): SEGUE win rate over pairs by
    majority of the raters' gated scores (mean > .5 -> 1, < .5 -> 0, = .5 -> 0.5),
    W/T/L, Wilson 95% CI over pairs, exact two-sided sign test on decided pairs,
    per-rating win rate
  - how often the gate fired, per opponent
  - the v5 table: per task block, one row per system (opponents + SEGUE), win columns
    = SEGUE's win rate against that row's system, rates over that system's distinct
    clips (per clip: majority of the raters' flags; endpoint miss = endpoint OR
    no_effect flag, as a still or a cut cannot keep the endpoints; leak not gated)
  - inter-rater agreement (mean pairwise, pairs with >= 2 raters) for the two
    questions and the three flags
  - attention checks (never in win rates): the COPY must carry the endpoint and leak
    boxes and the REAL must not; 2 items x 2 = 4 scored answers, pass >= 3.

    python3 scripts/user_study/analyze.py [--csv out.csv] [--drop-failed-sessions]
    python3 scripts/user_study/analyze.py --ratings exported.jsonl --pairs pairs.json

Lines of another protocol (the v4 four-question ratings in state/) are skipped.
v4 version of this script: misc/2026-09-22_user_study/v4_backup_2026-09-25/analyze.py.
"""
import argparse
import collections
import csv
import json
import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STUDY = os.path.join(REPO, "misc", "2026-09-22_user_study")
PAIRS_FILE = os.path.join(STUDY, "pairs.json")
RATINGS_FILE = os.path.join(STUDY, "state_v5", "ratings.jsonl")
PROTOCOL = "v5"
QUESTIONS = ("transition", "overall")
FLAGS = ("no_effect", "endpoint", "leak")
ATTENTION_SCORED = ("endpoint", "leak")  # flags scored on attention items
EXPECTED_ATTENTION = 2   # attention items per full session (-> 4 scored answers)
ATTENTION_PASS = 3       # >= this many of the scored answers correct = pass
TASK_NAME = {"two": "TEG (both endpoints)", "one": "Transfer (start given)"}
Z = 1.959964  # 95%


def wilson(wins, n):
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sign_test(w, l):
    """exact two-sided sign test on decided pairs (ties excluded)."""
    n = w + l
    if n == 0:
        return float("nan")
    k = min(w, l)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def majority(scores):
    """scores in {0, 0.5, 1} (SEGUE's gated score per rater) -> 1.0 / 0.5 / 0.0."""
    m = sum(scores) / len(scores)
    return 1.0 if m > 0.5 else (0.0 if m < 0.5 else 0.5)


def flag_majority(bools):
    t = sum(1 for b in bools if b)
    f = len(bools) - t
    return 1.0 if t > f else (0.0 if f > t else 0.5)


def pairwise_agreement(lists):
    """mean pairwise agreement over the lists with >= 2 entries -> (n_lists, mean)."""
    agrs = []
    for vs in lists:
        k = len(vs)
        if k < 2:
            continue
        conc = sum(1 for i in range(k) for j in range(i + 1, k) if vs[i] == vs[j])
        agrs.append(conc / (k * (k - 1) / 2))
    return len(agrs), (sum(agrs) / len(agrs) if agrs else float("nan"))


def pct(x):
    return "--" if x != x else f"{x*100:5.1f}%"


def build_votes(real_recs):
    """votes[pid][q] = list of (rater_id, gated score)."""
    votes = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in real_recs:
        for q in QUESTIONS:
            votes[r["pair_id"]][q].append((r["rater_id"], float(r["segue_score"][q])))
    return votes


def group_stats(votes, pids, q):
    pairs_rated = [p for p in pids if p in votes]
    w = t = l = 0
    maj_sum = 0.0
    n_rat, win_rat = 0, 0.0
    for p in pairs_rated:
        sc = [s for _, s in votes[p][q]]
        m = majority(sc)
        maj_sum += m
        w += (m == 1.0)
        t += (m == 0.5)
        l += (m == 0.0)
        n_rat += len(sc)
        win_rat += sum(sc)
    n = len(pairs_rated)
    return {"n_pairs": n, "n_ratings": n_rat, "w": w, "t": t, "l": l,
            "win_pairs": (maj_sum / n) if n else float("nan"),
            "ci": wilson(maj_sum, n) if n else (float("nan"), float("nan")),
            "p": sign_test(w, l),
            "per_rating": (win_rat / n_rat) if n_rat else float("nan")}


def fmt(s):
    if s["n_pairs"] == 0:
        return "-- | -- | -- | -- | -- | -- | --"
    lo, hi = s["ci"]
    return (f"{pct(s['win_pairs'])} | {s['w']}/{s['t']}/{s['l']} | [{lo*100:4.1f},{hi*100:4.1f}] | "
            f"{s['p']:.3g} | {s['n_pairs']} | {s['n_ratings']} | {pct(s['per_rating'])}")


def win_rate_report(real_recs, pairs, title):
    """Print the win-rate, gate, v5-table and agreement sections for REAL ratings."""
    votes = build_votes(real_recs)

    # key the per-opponent breakdown on (opponent, task): refVFX is used in both
    # tasks, and those are different comparisons.
    by_opp = collections.defaultdict(list)
    by_task = collections.defaultdict(list)
    for pid, p in pairs.items():
        by_opp[(p["opponent"], p["task"])].append(pid)
        by_task[p["task"]].append(pid)
    all_pids = list(pairs)

    print(f"# SEGUE human study, protocol v5 — win rates ({title})\n")
    print("Scores are after the per-rater no-effect gate. Win rate = mean over pairs of the "
          "raters' majority (ties = 0.5). CI = Wilson 95% over pairs; p = exact two-sided sign "
          "test on decided pairs; per-rating = mean gated score over individual ratings.\n")
    header = "| scope | question | SEGUE win% | W/T/L | 95% CI | p | pairs | ratings | per-rating% |"
    sep = "|---|---|---|---|---|---|---|---|---|"

    print("## By opponent\n")
    print(header); print(sep)
    for (opp, task) in sorted(by_opp):
        for q in QUESTIONS:
            print(f"| {opp} (task {task}) | {q} | {fmt(group_stats(votes, by_opp[(opp, task)], q))} |")
    print()

    print("## By task\n")
    print(header); print(sep)
    for task in ("two", "one"):
        for q in QUESTIONS:
            print(f"| task {task} | {q} | {fmt(group_stats(votes, by_task[task], q))} |")
    print()

    print("## Overall\n")
    print(header); print(sep)
    for q in QUESTIONS:
        print(f"| all | {q} | {fmt(group_stats(votes, all_pids, q))} |")
    print()

    # --- gate ---------------------------------------------------------------
    print("## Gate (per rating, from that rater's own no-effect boxes)\n")
    print("| opponent (task) | ratings | both effect (A/B stands) | SEGUE only (SEGUE wins) "
          "| opponent only (SEGUE loses) | neither (tie) |")
    print("|---|---|---|---|---|---|")
    gates = collections.defaultdict(collections.Counter)
    for r in real_recs:
        gates[(r["opponent"], r["task"])][r["gate"]] += 1
    for key in sorted(by_opp):
        c = gates.get(key, collections.Counter())
        n = sum(c.values())
        print(f"| {key[0]} (task {key[1]}) | {n} | {c['both_effect']} | {c['segue_only_effect']} "
              f"| {c['opponent_only_effect']} | {c['neither_effect']} |")
    print()

    # --- v5 table: flag rates per system over distinct clips -----------------
    # obs[(task, system, clip)][flag] = list of bools (one per rating that showed the clip)
    obs = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in real_recs:
        p = pairs[r["pair_id"]]
        for system, clip, fl in (("SEGUE", p["segue_clip"], r["flags_segue"]),
                                 (p["opponent"], p["opponent_clip"], r["flags_opponent"])):
            d = obs[(p["task"], system, clip)]
            d["no_effect"].append(bool(fl["no_effect"]))
            d["endpoint_miss"].append(bool(fl["endpoint"] or fl["no_effect"]))
            d["leak"].append(bool(fl["leak"]))
    clips_total = collections.defaultdict(set)
    for p in pairs.values():
        clips_total[(p["task"], "SEGUE")].add(p["segue_clip"])
        clips_total[(p["task"], p["opponent"])].add(p["opponent_clip"])

    def rate(task, system, key):
        vals = [flag_majority(d[key]) for (t, s, _), d in obs.items()
                if t == task and s == system and d[key]]
        return (sum(vals) / len(vals)) if vals else float("nan"), len(vals)

    print("## v5 table (DESIGN_v5 shape)\n")
    print("Win columns = SEGUE's gated win rate against that row's system (W/T/L over pairs). "
          "Rates over that system's distinct clips in the block; per clip the raters' majority "
          "(ties 0.5). Endpoint miss counts a no-effect box as a miss. Leak is not gated.\n")
    print("| block | system | Transition win % vs SEGUE (W/T/L) | Overall win % vs SEGUE (W/T/L) "
          "| No effect % | Endpoint miss % | Leak % | clips rated |")
    print("|---|---|---|---|---|---|---|---|")
    for task in ("two", "one"):
        opps = sorted({p["opponent"] for p in pairs.values() if p["task"] == task})
        for system in opps + ["SEGUE"]:
            if system == "SEGUE":
                wcols = ["--", "--"]
            else:
                wcols = []
                for q in QUESTIONS:
                    s = group_stats(votes, by_opp[(system, task)], q)
                    wcols.append("--" if not s["n_pairs"] else
                                 f"{pct(s['win_pairs'])} ({s['w']}/{s['t']}/{s['l']})")
            ne, n1 = rate(task, system, "no_effect")
            em, _ = rate(task, system, "endpoint_miss")
            lk, _ = rate(task, system, "leak")
            print(f"| {TASK_NAME[task]} | {system} | {wcols[0]} | {wcols[1]} | {pct(ne)} | "
                  f"{pct(em)} | {pct(lk)} | {n1}/{len(clips_total[(task, system)])} |")
    print()

    # --- agreement -------------------------------------------------------------
    print("## Inter-rater agreement (mean pairwise, pairs with >=2 raters)\n")
    print("| item | pairs>=2 | mean pairwise agreement |")
    print("|---|---|---|")
    for q in QUESTIONS:
        n, m = pairwise_agreement([[s for _, s in votes[p][q]] for p in votes])
        print(f"| {q} (gated score) | {n} | {pct(m)} |")
    flag_lists = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in real_recs:
        for who, fl in (("segue", r["flags_segue"]), ("opponent", r["flags_opponent"])):
            for f in FLAGS:
                flag_lists[f][(r["pair_id"], who)].append(bool(fl[f]))
    for f in FLAGS:
        n, m = pairwise_agreement(list(flag_lists[f].values()))
        print(f"| flag {f} (per video) | {n} | {pct(m)} |")
    print()
    return votes


def attention_report(attn_recs):
    """Per-session attention pass/fail; returns the set of FAILED session ids."""
    by_sess = collections.OrderedDict()
    for r in attn_recs:
        by_sess.setdefault(r["session_id"], []).append(r)

    print("## Attention checks\n")
    print(f"Each session hides {EXPECTED_ATTENTION} attention items (REAL transition vs a COPY of "
          "the reference); each scores 2 boxes (endpoint, leak): correct when the COPY's box is "
          "ticked and the REAL's is not = 4 scored answers per session.")
    print(f"A fully-checked session PASSES with >= {ATTENTION_PASS} of the 4 correct. "
          "Attention ratings never enter win rates.\n")

    if not attn_recs:
        print("_no attention ratings yet._\n")
        return set()

    print("| rater | session | attn items | scored | correct | result |")
    print("|---|---|---|---|---|---|")
    failed, passed, partial = set(), set(), set()
    for sid, recs in by_sess.items():
        nick = recs[0].get("nickname") or recs[0]["rater_id"][:8]
        n_items = len(recs)
        scored = len(ATTENTION_SCORED) * n_items
        n_correct = sum(1 for r in recs for q in ATTENTION_SCORED if r["correct"].get(q))
        if n_items >= EXPECTED_ATTENTION:
            result = "pass" if n_correct >= ATTENTION_PASS else "FAIL"
            (passed if result == "pass" else failed).add(sid)
        else:
            result = "partial"
            partial.add(sid)
        print(f"| {nick} | {sid[:8]}… | {n_items} | {scored} | {n_correct} | {result} |")
    print()

    n_full = len(passed) + len(failed)
    rate = (len(passed) / n_full) if n_full else float("nan")
    print(f"- fully-checked sessions: **{n_full}**  ·  passed: **{len(passed)}**  ·  "
          f"failed: **{len(failed)}**  ·  partial (excluded): {len(partial)}")
    if n_full:
        print(f"- overall session pass rate: **{rate*100:.1f}%**")

    per_rater = collections.OrderedDict()
    for sid, recs in by_sess.items():
        rid = recs[0]["rater_id"]
        nick = recs[0].get("nickname") or rid[:8]
        d = per_rater.setdefault(rid, {"nick": nick, "sess": 0, "pass": 0, "fail": 0, "partial": 0})
        d["sess"] += 1
        d["pass"] += (sid in passed)
        d["fail"] += (sid in failed)
        d["partial"] += (sid in partial)
    print("\n### Per rater\n")
    print("| rater | sessions checked | passed | failed | partial |")
    print("|---|---|---|---|---|")
    for rid, d in per_rater.items():
        print(f"| {d['nick']} | {d['sess']} | {d['pass']} | {d['fail']} | {d['partial']} |")
    print()
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="write per-pair majority table here (attention excluded)")
    ap.add_argument("--drop-failed-sessions", action="store_true",
                    help="also print win rates with the ratings from failed attention "
                         "sessions removed (both tables are reported; default keeps all)")
    ap.add_argument("--ratings", default=RATINGS_FILE,
                    help="ratings.jsonl to read (default = the repo v5 state file; "
                         "point at an exported file for a hosted study)")
    ap.add_argument("--pairs", default=PAIRS_FILE, help="pairs.json (default = the repo copy)")
    args = ap.parse_args()

    with open(args.pairs) as f:
        pairs = {p["pair_id"]: p for p in json.load(f)}

    recs, skipped = [], 0
    if os.path.exists(args.ratings):
        with open(args.ratings) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("protocol") != PROTOCOL:
                    skipped += 1
                    continue
                recs.append(r)
    if skipped:
        print(f"_skipped {skipped} rating line(s) of another protocol._\n")
    if not recs:
        print("no v5 ratings yet.")
        return

    # a rater may go back ("Previous") and change answers: the server appends a revision line; keep the LAST
    # line per (session, pair) so every session counts each pair once
    last = {}
    for r in recs:
        last[(r["session_id"], r["pair_id"])] = r
    n_rev = len(recs) - len(last)
    recs = list(last.values())
    if n_rev:
        print(f"_{n_rev} revised rating line(s) replaced by their latest version._\n")
    real_recs = [r for r in recs if not r.get("attention")]
    attn_recs = [r for r in recs if r.get("attention")]

    votes = win_rate_report(real_recs, pairs, "all sessions")
    failed_sessions = attention_report(attn_recs)

    if args.drop_failed_sessions:
        kept = [r for r in real_recs if r["session_id"] not in failed_sessions]
        n_drop = len(real_recs) - len(kept)
        print(f"\n---\n\n_Dropping {len(failed_sessions)} failed attention session(s) "
              f"= {n_drop} real rating(s)._\n")
        win_rate_report(kept, pairs, "failed sessions dropped")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            # votes_A = raters whose gated score favours SEGUE, votes_B = the opponent,
            # votes_tie = gated ties (neither video had an effect for that rater)
            w.writerow(["pair_id", "task", "opponent", "q",
                        "votes_A", "votes_B", "segue_wins", "votes_tie"])
            for pid in sorted(votes):
                p = pairs[pid]
                for q in QUESTIONS:
                    sc = [s for _, s in votes[pid][q]]
                    w.writerow([pid, p["task"], p["opponent"], q,
                                sum(1 for s in sc if s == 1.0), sum(1 for s in sc if s == 0.0),
                                majority(sc), sum(1 for s in sc if s == 0.5)])
        print(f"[csv] wrote {args.csv}")


if __name__ == "__main__":
    main()
