"""exp_066 — aggregate certified scores into the tier table + contrasts C1-C11.

SPEC §4 discipline: no composite score; paired deltas as the inferential unit;
trust map consumed (†-classes never back a claim); near_copy REFLAGGED at
tau_copy=0.858 (amendment-1; rows embed draft 0.88); sigma_seed MDE attached
to every delta; certified-stamp verified on every consumed label.

Usage:
  python3 aggregate_contrasts.py [--allow-uncertified] [--out-dir OUT]
Outputs: OUT/contrasts.md, OUT/contrasts.json, OUT/tier_table.md
"""

import argparse
import json
import math
import pathlib
import re
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parents[2]
EVAL = REPO / "outputs/eval/ladder_v3"
TRUST = REPO / "outputs/eval/certification/3.0.0-draft.8/exam/trust_map.json"
TAU_COPY = 0.858

# amendment-1 MDEs at n=10 (adapter-arm sigma; MDE(n) = 1.96*sigma*sqrt(2/n))
MDE10 = {"app_ref": 0.024, "margin": 0.037, "copy_max": 0.022,
         "cam_dtw": 0.076, "obj_match": 0.008, "max_seam_z": 0.27}
SIGMA = {k: v / (1.96 * math.sqrt(2 / 10)) for k, v in MDE10.items()}
CHANNELS = ["app_ref", "margin", "obj_match", "cam_dtw", "copy_max", "max_seam_z"]
# reading direction: is a POSITIVE delta better, worse, or diagnostic-only?
DIR = {"app_ref": "higher=better", "margin": "higher=better",
       "obj_match": "higher=better", "cam_dtw": "LOWER=better",
       "copy_max": "diagnostic (near-copy at >=0.858)",
       "max_seam_z": "LOWER=better"}
# trust-map exam key per channel (None = no exam stratum -> always shown plain)
TRUST_KEY = {"app_ref": "m1a", "margin": "m1a", "obj_match": "m1c",
             "cam_dtw": "m1b", "copy_max": None, "max_seam_z": None}
C5_CLASSES = ["shadow", "portal", "polygon", "wireframe", "animalization",
              "color_rain", "gas_transformation", "illustration_scene"]
B8 = C5_CLASSES  # same 8 recipients for C9 confirmatory


def mde(metric, n):
    s = SIGMA.get(metric)
    return 1.96 * s * math.sqrt(2 / max(n, 1)) if s and n else None


def load_rows(allow_uncert):
    rows, bad_labels = [], []
    for d in sorted(EVAL.iterdir()):
        rj = d / "results.json"
        ij = d / "items.jsonl"
        if not (rj.exists() and ij.exists()):
            continue
        prov = json.loads(rj.read_text()).get("provenance", {})
        if not prov.get("certified"):
            bad_labels.append(d.name)
            if not allow_uncert:
                continue
        for line in ij.open():
            r = json.loads(line)
            if r.get("error"):
                continue
            r["_label"] = d.name
            r["near_copy"] = (r.get("copy_max") or 0) >= TAU_COPY  # reflag
            rows.append(r)
    return rows, bad_labels


def parse_id(r):
    """item_id -> (clip, seed, ckpt). arm/style come from the row itself."""
    iid = re.sub(r"__recheck$", "", r["item_id"])
    m = re.search(r"__s(\d+)(?:__ckpt(\d+))?$", iid)
    seed, ckpt = int(m.group(1)), (int(m.group(2)) if m.group(2) else None)
    body = iid[: m.start()]
    clip = body.split("__")[-1]
    return clip, seed, ckpt


def index(rows):
    """(armkey, style, clip, seed) -> row. armkey folds ckpt into the arm."""
    ix = {}
    for r in rows:
        clip, seed, ckpt = parse_id(r)
        ix[(r["arm"], r["style"], clip, seed)] = r
    return ix


def paired(ix, arm_a, arm_b, classes=None):
    """items present in BOTH arms -> list of (style, clip, seed, rowA, rowB)."""
    out = []
    for (arm, style, clip, seed), ra in ix.items():
        if arm != arm_a or (classes and style not in classes):
            continue
        rb = ix.get((arm_b, style, clip, seed))
        if rb is not None:
            out.append((style, clip, seed, ra, rb))
    return out


def delta_stats(pairs, trust):
    """per-channel: item-level mean deltas (A-B), class sign counts, trust cut."""
    res = {}
    for ch in CHANNELS:
        per_item = defaultdict(list)          # (style, clip) -> [delta per seed]
        for style, clip, seed, ra, rb in pairs:
            a, b = ra.get(ch), rb.get(ch)
            if a is None or b is None:
                continue
            if isinstance(a, float) and isinstance(b, float) and \
               not (math.isnan(a) or math.isnan(b)):
                per_item[(style, clip)].append(a - b)
        items = {k: sum(v) / len(v) for k, v in per_item.items() if v}
        if not items:
            res[ch] = None
            continue
        tkey = TRUST_KEY[ch]
        def trusted(style):
            return True if tkey is None else bool(
                trust.get(style, {}).get(tkey, False))
        t_items = {k: v for k, v in items.items() if trusted(k[0])}
        cls_mean = defaultdict(list)
        for (style, clip), v in t_items.items():
            cls_mean[style].append(v)
        cls_delta = {s: sum(v) / len(v) for s, v in cls_mean.items()}
        pos = sum(1 for v in cls_delta.values() if v > 0)
        n_cls = len(cls_delta)
        vals = list(t_items.values())
        mean_d = sum(vals) / len(vals)
        res[ch] = {
            "n_items": len(items), "n_items_trusted": len(vals),
            "n_classes_trusted": n_cls, "mean_delta": mean_d,
            "class_deltas": cls_delta, "sign_pos": pos,
            "sign_p_two_sided": binom_two_sided(pos, n_cls),
            "mde_at_n": mde(ch, len(vals)),
            "exceeds_mde": (abs(mean_d) >= mde(ch, len(vals))
                            if mde(ch, len(vals)) else None),
            "dropped_untrusted_classes": sorted(
                {k[0] for k in items} - {k[0] for k in t_items}),
        }
    return res


def binom_two_sided(k, n):
    if n == 0:
        return None
    def pmf(i):
        return math.comb(n, i) / 2 ** n
    p_obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= p_obs + 1e-12))


ARMS_PE = ("r1", "r1k", "r1k_ext")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-uncertified", action="store_true")
    ap.add_argument("--out-dir", default=str(EVAL / "_contrasts"))
    args = ap.parse_args()

    trust = json.loads(TRUST.read_text())
    rows, bad = load_rows(args.allow_uncertified)
    ix = index(rows)
    # fold base-PE union into one pseudo-arm for joins vs adapters
    for (arm, style, clip, seed), r in list(ix.items()):
        if arm in ARMS_PE:
            ix.setdefault(("basePE", style, clip, seed), r)

    arms_present = sorted({k[0] for k in ix})
    contrasts = {}

    def add(name, arm_a, arm_b, classes=None, note=""):
        pairs = paired(ix, arm_a, arm_b, classes)
        if pairs:
            contrasts[name] = {"a": arm_a, "b": arm_b, "note": note,
                               "n_pairs_rowlevel": len(pairs),
                               "channels": delta_stats(pairs, trust)}
        else:
            contrasts[name] = {"a": arm_a, "b": arm_b, "note": note,
                               "n_pairs_rowlevel": 0, "channels": None}

    add("C1_r1_minus_r0", "r1", "r0", note="base PE effect; 50 paired items")
    add("C3_r2_minus_r3_ckpt2000", "r2_ckpt2000", "r3_ckpt2000",
        note="overfit gap at 2000 (class-level; items differ -> join is empty; "
             "see class_means in tier table)")
    add("C4_r3_minus_basePE", "r3_ckpt2000", "basePE",
        note="specialist value over keyed base, same items")
    add("C5_ic3b_minus_r3", "ic3_b", "r3_ckpt2000", classes=C5_CLASSES,
        note="PRIMARY; same items, sign over 8 classes")
    add("C6_ic3c_minus_basePE", "ic3_c", "basePE",
        note="zero-shot vs base; descriptive n=4 classes")
    add("C7_r3_minus_ic3c", "r3_ckpt2000", "ic3_c",
        note="specialist vs zero-shot generalist; descriptive n=3")
    add("C8_ic3b_minus_basePE", "ic3_b", "basePE", note="generalist value")
    add("C9_r3x_minus_ic3x_B8", "r3x", "ic3_x", classes=B8,
        note="confirmatory one-sided R3X>R4X over 8 recipients")
    add("C9ext_r3x_minus_ic3x", "r3x", "ic3_x",
        classes=["hero_flight", "shadow_smoke", "super_fast_run"],
        note="labeled exploratory extension")
    add("C10_ic3_minus_ic2_r4band", "ic3_a", "ic2_r4",
        note="alignment value on shared held-in items (joined subset)")
    add("C10b_ic3b_minus_ic2r4", "ic3_b", "ic2_r4",
        note="alignment value where ic2's 'unseen' items were actually trained")
    add("C10c_ic3c_minus_ic2r5", "ic3_c", "ic2_r5", note="zero-shot arm compare")

    # C11 + C3: class-level arm means (items differ across arms)
    class_means = defaultdict(lambda: defaultdict(dict))
    per = defaultdict(list)
    for (arm, style, clip, seed), r in ix.items():
        if arm == "basePE":
            continue
        for ch in CHANNELS:
            v = r.get(ch)
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                per[(arm, style, ch)].append(v)
    for (arm, style, ch), vals in per.items():
        class_means[arm][style][ch] = sum(vals) / len(vals)
    copy_rates = defaultdict(lambda: [0, 0])
    for (arm, style, clip, seed), r in ix.items():
        if arm == "basePE":
            continue
        copy_rates[arm][0] += 1 if r["near_copy"] else 0
        copy_rates[arm][1] += 1

    out = {
        "labels_uncertified_skipped" if not args.allow_uncertified
        else "labels_uncertified_INCLUDED": bad,
        "arms_present": arms_present,
        "n_rows": len(rows),
        "tau_copy": TAU_COPY,
        "contrasts": contrasts,
        "class_means": {a: dict(v) for a, v in class_means.items()},
        "near_copy_rate_by_arm": {a: {"rate": c[0] / c[1], "n": c[1]}
                                  for a, c in copy_rates.items()},
    }
    od = pathlib.Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "contrasts.json").write_text(json.dumps(out, indent=1, default=str))

    # markdown
    md = ["# Ladder v3 — contrasts (certified)\n",
          f"rows={len(rows)} arms={arms_present}",
          f"uncertified labels {'INCLUDED (DEV!)' if args.allow_uncertified else 'skipped'}: {bad}\n"]
    for name, c in contrasts.items():
        md.append(f"## {name}  ({c['a']} − {c['b']})  — {c['note']}")
        if not c["channels"]:
            md.append("_no joined pairs yet_\n")
            continue
        md.append("| channel (direction) | Δ mean | n items | classes+ / n (p 2s) | MDE(n) | ≥MDE | dropped† |")
        md.append("|---|---|---|---|---|---|---|")
        for ch, s in c["channels"].items():
            if not s:
                md.append(f"| {ch} | — | 0 | | | | |")
                continue
            m = s["mde_at_n"]
            md.append(
                f"| {ch} ({DIR[ch]}) | {s['mean_delta']:+.4f} | {s['n_items_trusted']} |"
                f" {s['sign_pos']}/{s['n_classes_trusted']}"
                f" (p={s['sign_p_two_sided']:.3f}) |"
                f" {('%.4f' % m) if m else '—'} |"
                f" {'YES' if s['exceeds_mde'] else ('no' if s['exceeds_mde'] is not None else '—')} |"
                f" {','.join(s['dropped_untrusted_classes']) or '—'} |")
        md.append("")
    md.append("## near-copy rate by arm (tau=0.858)")
    for a, v in sorted(out["near_copy_rate_by_arm"].items()):
        md.append(f"- {a}: {v['rate']:.1%} of {v['n']}")
    (od / "contrasts.md").write_text("\n".join(md) + "\n")
    print(f"[done] {len(rows)} rows -> {od}/contrasts.md (+json)")
    if bad:
        print(f"[note] uncertified labels: {bad}")


if __name__ == "__main__":
    main()
