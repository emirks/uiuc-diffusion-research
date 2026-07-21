"""Paired fix-vs-null verdict on suffix_lpips — exp_073 conditioning-bleed fix.

Re-keyed from misc/advised_method_impl/verify/compare_arm.py (which keyed on `margin`). This
campaign's fix targets the SUFFIX endpoint, so the PRIMARY metric is `suffix_lpips`.

CONSULT-2 hardenings:
  1. KEYED JOIN on (item_id, seed); exact key-set equality asserted — a missing key is a HARD
     ERROR, never a silent drop (the eval-ladder C4/C8 blind-join failure mode).
  2. DIRECTION: for lower-is-better metrics (lpips, seam_z) Δ = metric(base) - metric(fix);
     for higher-is-better (dino, margin) Δ = metric(fix) - metric(base). Positive Δ ALWAYS
     favors fix. Stated once, here.
  3. Per-item delta = MEDIAN over the (up to 3) gen seeds; item = (item_id). Exact two-sided
     binomial SIGN test on per-item deltas is the SOLE verdict key.
  4. Wilcoxon signed-rank + median Δ reported as SECONDARY descriptors (never drive the tier).
Also: per-tier (A/B/C or R2/R3) descriptive breakdown; guards (prefix_*, margin, near_copy).

Usage:
  python compare_suffix.py <fix_items_glob> <base_items_glob> [--base-label nullA]
Each items.jsonl row must carry: item_id (seed-independent grid item), seed, arm, tier,
sidedness, and the metric fields. Rows are filtered to sidedness=='two_sided' for the primary
(pass --all to keep every row, e.g. the one-sided control).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import statistics as st

LOWER_BETTER = {"suffix_lpips", "prefix_lpips", "suffix_seam_z", "prefix_seam_z", "max_seam_z",
                "near_copy", "copy_max"}
HIGHER_BETTER = {"suffix_dino", "prefix_dino", "margin", "app_ref"}
PRIMARY = "suffix_lpips"
SECONDARY = ["suffix_dino", "suffix_seam_z", "max_seam_z"]
GUARDS = ["prefix_lpips", "prefix_dino", "margin", "near_copy", "copy_max"]


def load(pat: str) -> dict:
    """Return {(item_id, seed): row}. Hard error on duplicate keys."""
    rows = {}
    files = glob.glob(pat)
    if not files:
        raise SystemExit(f"[error] no items.jsonl matched: {pat}")
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["item_id"], int(r["seed"]))
            if key in rows:
                raise SystemExit(f"[error] duplicate key {key} in {f}")
            rows[key] = r
    return rows


def sign_test(deltas: list[float]) -> tuple[int, int, float]:
    """Exact two-sided binomial sign test. Zeros dropped. Returns (pos, n_nonzero, p)."""
    nz = [d for d in deltas if d != 0]
    pos = sum(1 for d in nz if d > 0)
    n = len(nz)
    if n == 0:
        return 0, 0, 1.0
    k = max(pos, n - pos)
    p = 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return pos, n, min(p, 1.0)


def wilcoxon(deltas: list[float]) -> float:
    """Wilcoxon signed-rank two-sided p (normal approx w/ continuity+tie correction). SECONDARY."""
    nz = [d for d in deltas if d != 0]
    n = len(nz)
    if n < 1:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1  # average rank (1-based)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    mean = n * (n + 1) / 4
    # tie correction
    tie = collections.Counter(abs(d) for d in nz)
    tc = sum(t ** 3 - t for t in tie.values())
    var = n * (n + 1) * (2 * n + 1) / 24 - tc / 48
    if var <= 0:
        return 1.0
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(var)
    return math.erfc(z / math.sqrt(2))


def signed_delta(fix_row: dict, base_row: dict, metric: str) -> float | None:
    a, b = fix_row.get(metric), base_row.get(metric)
    if a is None or b is None:
        return None
    if metric in HIGHER_BETTER:
        return float(a) - float(b)      # fix - base ; positive favors fix
    return float(b) - float(a)          # base - fix ; positive favors fix (lower-is-better)


def per_item_deltas(fix: dict, base: dict, metric: str) -> tuple[dict, list]:
    """Median-over-seeds delta per item_id. Asserts exact (item_id,seed) key-set equality."""
    if set(fix) != set(base):
        only_f = sorted(set(fix) - set(base))[:8]
        only_b = sorted(set(base) - set(fix))[:8]
        raise SystemExit(f"[error] key-set mismatch: {len(set(fix)^set(base))} differ. "
                         f"fix-only(e.g.) {only_f}  base-only(e.g.) {only_b}")
    by_item = collections.defaultdict(list)
    for (iid, seed) in fix:
        d = signed_delta(fix[(iid, seed)], base[(iid, seed)], metric)
        if d is not None:
            by_item[iid].append(d)
    item_delta = {iid: st.median(ds) for iid, ds in by_item.items() if ds}
    return item_delta, list(item_delta.values())


def report_metric(fix, base, metric, base_label, tier_of):
    item_delta, deltas = per_item_deltas(fix, base, metric)
    pos, n, p = sign_test(deltas)
    med = st.median(deltas) if deltas else float("nan")
    wp = wilcoxon(deltas)
    p75 = st.quantiles([abs(d) for d in deltas], n=4)[-1] if len(deltas) >= 2 else (
        abs(deltas[0]) if deltas else float("nan"))
    print(f"  [{metric}] items={len(deltas)}  median Δ={med:+.4f} (+favors fix)  "
          f"sign {pos}/{n} p={p:.3f}  wilcoxon p={wp:.3f}  P75|Δ|={p75:.4f}")
    # per-tier descriptive
    byt = collections.defaultdict(list)
    for iid, d in item_delta.items():
        byt[tier_of.get(iid, "?")].append(d)
    for t in sorted(byt):
        dd = byt[t]
        tp, tn, tpv = sign_test(dd)
        print(f"      tier {t}: n={len(dd)} medianΔ={st.median(dd):+.4f} sign {tp}/{tn} p={tpv:.3f}")
    return dict(metric=metric, n=len(deltas), median=med, pos=pos, nnz=n, sign_p=p,
                wilcoxon_p=wp, p75_abs=p75)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fix_glob")
    ap.add_argument("base_glob")
    ap.add_argument("--base-label", default="nullA")
    ap.add_argument("--all", action="store_true", help="keep all rows (control); default = two_sided only")
    args = ap.parse_args()

    fix, base = load(args.fix_glob), load(args.base_glob)
    if not args.all:
        fix = {k: r for k, r in fix.items() if r.get("sidedness") == "two_sided"}
        base = {k: r for k, r in base.items() if r.get("sidedness") == "two_sided"}
    tier_of = {iid: r.get("tier", "?") for (iid, _), r in {**base, **fix}.items()}
    print(f"=== fix vs {args.base_label}  (keys: fix={len(fix)} base={len(base)}; "
          f"{'ALL' if args.all else 'two_sided only'}) ===")
    print(f"\n-- PRIMARY (verdict key = exact sign test on {PRIMARY}) --")
    prim = report_metric(fix, base, PRIMARY, args.base_label, tier_of)
    print(f"\n-- SECONDARY (descriptive; do NOT drive verdict) --")
    for m in SECONDARY:
        report_metric(fix, base, m, args.base_label, tier_of)
    print(f"\n-- GUARDS --")
    for m in GUARDS:
        try:
            report_metric(fix, base, m, args.base_label, tier_of)
        except Exception as e:
            print(f"  [{m}] skipped: {e}")
    print(f"\n=== PRIMARY summary: {PRIMARY} median Δ={prim['median']:+.4f}, "
          f"sign {prim['pos']}/{prim['nnz']} p={prim['sign_p']:.3f} (verdict key) ===")


if __name__ == "__main__":
    main()
