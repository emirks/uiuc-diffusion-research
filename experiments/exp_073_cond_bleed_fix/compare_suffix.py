"""Paired fix-vs-null verdict on suffix_lpips — exp_073 conditioning-bleed fix.

Re-keyed from misc/advised_method_impl/verify/compare_arm.py (which keyed on `margin`). This
campaign's fix targets the SUFFIX endpoint, so the PRIMARY metric is `suffix_lpips`.

CONSULT-2 hardenings:
  1. KEYED JOIN on item_id (which encodes seed: ...__s<seed>[__ckpt<n>]); exact key-set
     equality asserted — a missing key is a HARD ERROR, never a silent drop (the eval-ladder
     C4/C8 blind-join failure mode).
  2. DIRECTION: for lower-is-better metrics (lpips, seam_z) Δ = metric(base) - metric(fix);
     for higher-is-better (dino, margin) Δ = metric(fix) - metric(base). Positive Δ ALWAYS
     favors fix. Stated once, here.
  3. Per-item delta = MEDIAN over the gen seeds; item = seed-stripped base id. Exact two-sided
     binomial SIGN test on per-item deltas is the SOLE verdict key.
  4. Wilcoxon signed-rank + median Δ reported as SECONDARY descriptors (never drive the tier).
Also: per-tier (rung) descriptive breakdown; guards (prefix_*, margin, near_copy).

Each items.jsonl row carries item_id (with __s<seed>), arm, and the metric fields. Fix and
null arms share identical item_ids (arm-independent). The caller points fix_glob / base_glob at
the SAME item set (e.g. two-sided primary set, or the n=24 control set).

Usage: python compare_suffix.py <fix_items_glob> <base_items_glob> [--base-label nullA]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import re
import statistics as st

LOWER_BETTER = {"suffix_lpips", "prefix_lpips", "suffix_seam_z", "prefix_seam_z", "max_seam_z",
                "near_copy", "copy_max"}
HIGHER_BETTER = {"suffix_dino", "prefix_dino", "margin", "app_ref"}
PRIMARY = "suffix_lpips"
SECONDARY = ["suffix_dino", "suffix_seam_z", "max_seam_z"]
GUARDS = ["prefix_lpips", "prefix_dino", "margin", "near_copy", "copy_max"]

_SEED = re.compile(r"__s\d+")
_TIER = {"R4A": "A", "R4B": "B", "R5": "C"}


def base_item(item_id: str) -> str:
    return _SEED.sub("", item_id)


def tier_of(item_id: str) -> str:
    rung = item_id.split("__", 1)[0]
    return _TIER.get(rung, rung)


def load(pat: str) -> dict:
    """Return {item_id: row}. Hard error on duplicate keys / no files."""
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
            iid = r["item_id"]
            if iid in rows:
                raise SystemExit(f"[error] duplicate item_id {iid} in {f}")
            rows[iid] = r
    return rows


def sign_test(deltas):
    nz = [d for d in deltas if d != 0]
    pos = sum(1 for d in nz if d > 0)
    n = len(nz)
    if n == 0:
        return 0, 0, 1.0
    k = max(pos, n - pos)
    p = 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return pos, n, min(p, 1.0)


def wilcoxon(deltas):
    """Two-sided Wilcoxon signed-rank p (normal approx, tie+continuity corrected). SECONDARY."""
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
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    mean = n * (n + 1) / 4
    tie = collections.Counter(abs(d) for d in nz)
    tc = sum(t ** 3 - t for t in tie.values())
    var = n * (n + 1) * (2 * n + 1) / 24 - tc / 48
    if var <= 0:
        return 1.0
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(var)
    return math.erfc(z / math.sqrt(2))


def signed_delta(fix_row, base_row, metric):
    a, b = fix_row.get(metric), base_row.get(metric)
    if a is None or b is None:
        return None
    if metric in HIGHER_BETTER:
        return float(a) - float(b)      # fix - base ; positive favors fix
    return float(b) - float(a)          # base - fix ; positive favors fix (lower-is-better)


def per_item_deltas(fix, base, metric):
    """Median-over-seeds Δ per seed-stripped item. Asserts exact item_id key-set equality."""
    if set(fix) != set(base):
        of = sorted(set(fix) - set(base))[:8]
        ob = sorted(set(base) - set(fix))[:8]
        raise SystemExit(f"[error] key-set mismatch: {len(set(fix) ^ set(base))} differ. "
                         f"fix-only(e.g.) {of}  base-only(e.g.) {ob}")
    by_item = collections.defaultdict(list)
    for iid in fix:
        d = signed_delta(fix[iid], base[iid], metric)
        if d is not None:
            by_item[base_item(iid)].append(d)
    item_delta = {b: st.median(ds) for b, ds in by_item.items() if ds}
    return item_delta


def p75(vals):
    a = sorted(abs(v) for v in vals)
    if len(a) >= 2:
        return st.quantiles(a, n=4)[-1]
    return a[0] if a else float("nan")


def report_metric(fix, base, metric, is_primary=False):
    item_delta = per_item_deltas(fix, base, metric)
    deltas = list(item_delta.values())
    if not deltas:
        print(f"  [{metric}] no data")
        return None
    pos, n, p = sign_test(deltas)
    med = st.median(deltas)
    wp = wilcoxon(deltas)
    print(f"  [{metric}] items={len(deltas)}  medianΔ={med:+.4f} (+favors fix)  "
          f"sign {pos}/{n} p={p:.3f}  wilcoxon p={wp:.3f}  P75|Δ|={p75(deltas):.4f}"
          + ("   <-- VERDICT KEY" if is_primary else ""))
    byt = collections.defaultdict(list)
    for b, d in item_delta.items():
        byt[tier_of(b)].append(d)
    for t in sorted(byt):
        dd = byt[t]
        tp, tn, tpv = sign_test(dd)
        print(f"      tier {t}: n={len(dd)} medianΔ={st.median(dd):+.4f} sign {tp}/{tn} p={tpv:.3f}")
    return dict(metric=metric, n=len(deltas), median=med, pos=pos, nnz=n, sign_p=p,
                wilcoxon_p=wp, p75_abs=p75(deltas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fix_glob")
    ap.add_argument("base_glob")
    ap.add_argument("--base-label", default="nullA")
    args = ap.parse_args()

    fix, base = load(args.fix_glob), load(args.base_glob)
    print(f"=== fix vs {args.base_label}  (item_ids: fix={len(fix)} base={len(base)}) ===")
    print(f"\n-- PRIMARY (verdict key = exact sign test on {PRIMARY}) --")
    prim = report_metric(fix, base, PRIMARY, is_primary=True)
    print("\n-- SECONDARY (descriptive; do NOT drive verdict) --")
    for m in SECONDARY:
        report_metric(fix, base, m)
    print("\n-- GUARDS --")
    for m in GUARDS:
        report_metric(fix, base, m)
    if prim:
        print(f"\n=== PRIMARY summary: {PRIMARY} medianΔ={prim['median']:+.4f}, "
              f"sign {prim['pos']}/{prim['nnz']} p={prim['sign_p']:.3f} (verdict key) ===")


if __name__ == "__main__":
    main()
