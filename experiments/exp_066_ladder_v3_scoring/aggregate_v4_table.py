"""exp_066 — v4 cross-comparison table (v4.0.0 instrument, NOT re-certified).

Owner directive 2026-07-17: score the full ladder under v4.0.0 as a
cross-comparison to the certified v3 headline. The reference_v4 artifact was
rebuilt for the corrected 222-clip corpus (owner-authorized); no v4
re-certification ran, so every stamp here reads uncertified BY DESIGN and this
table never replaces the v3 headline.

Instrument notes vs the v3 aggregator (aggregate_contrasts.py):
- v4 headline channels: app_ref (rank composite, HIGHER better), cam_zpr and
  obj_csls (rank distances, LOWER better). No sigma_seed was measured under
  the v4 normalization -> those channels get sign tests only, no MDE gate.
- Carried-over channels (margin, copy_max, max_seam_z) and the bridge field
  app_ref_v3 (byte-identical computation to v3's app_ref) keep amendment-1
  MDEs.
- Trust map: v3 exam (draft.8) applied at metric-family level (m1a/m1b/m1c) —
  an approximation, disclosed here; v4 shipped no exam.
- Lane precedence: ladder_v4h (H100, warm shared cache) over ladder_v4
  (mixed-pool insurance); mixed rows disclosed via _label "@mix".
- *_saturated flags (raw value outside fitted reference support, SPEC §6.5)
  reported per arm; saturated cells are flagged, not certified.

Usage: python3 aggregate_v4_table.py [--out-dir OUT]
Outputs: OUT/contrasts_v4.md, OUT/contrasts_v4.json, OUT/tier_table_v4.md
"""

import argparse
import json
import math
import pathlib
import re
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parents[2]
V4H = REPO / "outputs/eval/ladder_v4h"
V4M = REPO / "outputs/eval/ladder_v4"
V3 = REPO / "outputs/eval/ladder_v3"
TRUST = REPO / "outputs/eval/certification/3.0.0-draft.8/exam/trust_map.json"
TAU_COPY = 0.858

# amendment-1 MDEs apply ONLY to channels whose definition is unchanged in v4
MDE10 = {"app_ref_v3": 0.024, "margin": 0.037, "copy_max": 0.022,
         "max_seam_z": 0.27}
SIGMA = {k: v / (1.96 * math.sqrt(2 / 10)) for k, v in MDE10.items()}
CHANNELS = ["app_ref", "app_ref_v3", "margin", "obj_csls", "cam_zpr",
            "copy_max", "max_seam_z"]
DIR = {"app_ref": "higher=better (v4 rank composite)",
       "app_ref_v3": "higher=better (v3 bridge)",
       "margin": "higher=better",
       "obj_csls": "LOWER=better (v4 rank)", "cam_zpr": "LOWER=better (v4 rank)",
       "copy_max": "diagnostic (near-copy at >=0.858)",
       "max_seam_z": "LOWER=better"}
TRUST_KEY = {"app_ref": "m1a", "app_ref_v3": "m1a", "margin": "m1a",
             "cam_zpr": "m1b", "obj_csls": "m1c",
             "copy_max": None, "max_seam_z": None}
SAT_FLAG = {"app_ref": "app_saturated", "cam_zpr": "cam_zpr_saturated"}
C5_CLASSES = ["shadow", "portal", "polygon", "wireframe", "animalization",
              "color_rain", "gas_transformation", "illustration_scene"]
B8 = C5_CLASSES


def mde(metric, n):
    s = SIGMA.get(metric)
    return 1.96 * s * math.sqrt(2 / max(n, 1)) if s and n else None


def load_dir(root, tag, have):
    rows = []
    if not root.is_dir():
        return rows
    for d in sorted(root.iterdir()):
        rj, ij = d / "results.json", d / "items.jsonl"
        if d.name.startswith("_") or d.name in have or \
           not (rj.exists() and ij.exists()):
            continue
        for line in ij.open():
            r = json.loads(line)
            if r.get("error"):
                continue
            r["_label"] = d.name + tag
            r["near_copy"] = (r.get("copy_max") or 0) >= TAU_COPY
            rows.append(r)
        have.add(d.name)
    return rows


def parse_id(r):
    iid = re.sub(r"__recheck$", "", r["item_id"])
    m = re.search(r"__s(\d+)(?:__ckpt(\d+))?$", iid)
    seed, ckpt = int(m.group(1)), (int(m.group(2)) if m.group(2) else None)
    clip = iid[: m.start()].split("__")[-1]
    return clip, seed, ckpt


def index(rows):
    ix = {}
    for r in rows:
        clip, seed, ckpt = parse_id(r)
        ix[(r["arm"], r["style"], clip, seed)] = r
    return ix


def paired(ix, arm_a, arm_b, classes=None):
    out = []
    for (arm, style, clip, seed), ra in ix.items():
        if arm != arm_a or (classes and style not in classes):
            continue
        rb = ix.get((arm_b, style, clip, seed))
        if rb is not None:
            out.append((style, clip, seed, ra, rb))
    return out


def binom_two_sided(k, n):
    if n == 0:
        return None
    def pmf(i):
        return math.comb(n, i) / 2 ** n
    p_obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= p_obs + 1e-12))


def delta_stats(pairs, trust):
    res = {}
    for ch in CHANNELS:
        per_item = defaultdict(list)
        for style, clip, seed, ra, rb in pairs:
            a, b = ra.get(ch), rb.get(ch)
            if a is None or b is None:
                continue
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and \
               not (math.isnan(float(a)) or math.isnan(float(b))):
                per_item[(style, clip)].append(float(a) - float(b))
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
        if not vals:
            res[ch] = None
            continue
        mean_d = sum(vals) / len(vals)
        m = mde(ch, len(vals))
        res[ch] = {
            "n_items": len(items), "n_items_trusted": len(vals),
            "n_classes_trusted": n_cls, "mean_delta": mean_d,
            "class_deltas": cls_delta, "sign_pos": pos,
            "sign_p_two_sided": binom_two_sided(pos, n_cls),
            "mde_at_n": m,
            "exceeds_mde": (abs(mean_d) >= m) if m else None,
            "dropped_untrusted_classes": sorted(
                {k[0] for k in items} - {k[0] for k in t_items}),
        }
    return res


ARMS_PE = ("r1", "r1k", "r1k_ext")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(V4H / "_contrasts"))
    args = ap.parse_args()

    trust = json.loads(TRUST.read_text())
    have = set()
    rows = load_dir(V4H, "", have) + load_dir(V4M, "@mix", have)
    lanes = sorted({r["_label"] for r in rows})
    ix = index(rows)
    for (arm, style, clip, seed), r in list(ix.items()):
        if arm in ARMS_PE:
            ix.setdefault(("basePE", style, clip, seed), r)

    arms_present = sorted({k[0] for k in ix})
    contrasts = {}

    def add(name, arm_a, arm_b, classes=None, note=""):
        pairs = paired(ix, arm_a, arm_b, classes)
        contrasts[name] = {"a": arm_a, "b": arm_b, "note": note,
                           "n_pairs_rowlevel": len(pairs),
                           "channels": delta_stats(pairs, trust) if pairs
                           else None}

    add("C1_r1_minus_r0", "r1", "r0", note="base PE effect; 50 paired items")
    add("C3_r2_minus_r3_ckpt2000", "r2_ckpt2000", "r3_ckpt2000",
        note="overfit gap at 2000 (items differ -> join empty; see tier table)")
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

    class_means = defaultdict(lambda: defaultdict(dict))
    per = defaultdict(list)
    sat = defaultdict(lambda: [0, 0])
    for (arm, style, clip, seed), r in ix.items():
        if arm == "basePE":
            continue
        for ch in CHANNELS:
            v = r.get(ch)
            if isinstance(v, (int, float)) and not (
                    isinstance(v, float) and math.isnan(v)):
                per[(arm, style, ch)].append(v)
        for f in SAT_FLAG.values():
            sat[arm][0] += 1 if r.get(f) else 0
            sat[arm][1] += 1
    for (arm, style, ch), vals in per.items():
        class_means[arm][style][ch] = sum(vals) / len(vals)
    copy_rates = defaultdict(lambda: [0, 0])
    for (arm, style, clip, seed), r in ix.items():
        if arm == "basePE":
            continue
        copy_rates[arm][0] += 1 if r["near_copy"] else 0
        copy_rates[arm][1] += 1

    # bridge check: v4's app_ref_v3 must reproduce v3's certified app_ref
    bridge = {}
    if V3.is_dir():
        v3ix = {}
        for d in sorted(V3.iterdir()):
            ij = d / "items.jsonl"
            if d.name.startswith("_") or not ij.exists():
                continue
            for line in ij.open():
                r = json.loads(line)
                if r.get("error") or r.get("app_ref") is None:
                    continue
                clip, seed, ckpt = parse_id(r)
                v3ix[(r["arm"], r["style"], clip, seed)] = r["app_ref"]
        diffs = defaultdict(list)
        for key, r in ix.items():
            if key[0] == "basePE" or r.get("app_ref_v3") is None:
                continue
            v3v = v3ix.get(key)
            if v3v is not None and not math.isnan(float(r["app_ref_v3"])):
                diffs[r["_label"]].append(abs(float(r["app_ref_v3"]) - v3v))
        bridge = {lab: {"n": len(v), "mean_abs": sum(v) / len(v),
                        "max_abs": max(v)} for lab, v in diffs.items() if v}

    out = {
        "instrument": "v4.0.0 — reference_v4 rebuilt for corrected 222-clip "
                      "corpus per owner directive 2026-07-17; NOT re-certified",
        "lane_precedence": "ladder_v4h (H100) > ladder_v4 (@mix)",
        "labels_present": lanes,
        "arms_present": arms_present, "n_rows": len(rows),
        "tau_copy": TAU_COPY, "contrasts": contrasts,
        "class_means": {a: dict(v) for a, v in class_means.items()},
        "near_copy_rate_by_arm": {a: {"rate": c[0] / c[1], "n": c[1]}
                                  for a, c in copy_rates.items()},
        "saturation_rate_by_arm": {a: {"rate": c[0] / c[1], "n_flags": c[1]}
                                   for a, c in sat.items() if c[1]},
        "bridge_app_ref_v3_vs_v3": bridge,
    }
    od = pathlib.Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "contrasts_v4.json").write_text(json.dumps(out, indent=1,
                                                     default=str))

    md = ["# Ladder v4 cross-comparison — contrasts (v4.0.0, NOT re-certified)\n",
          out["instrument"], "",
          f"rows={len(rows)} labels={lanes}",
          "MDE gates only on carried-over channels (app_ref_v3/margin/"
          "copy_max/max_seam_z); v4-normalized channels: sign tests only.",
          "Trust: v3 exam applied at family level (approximation; v4 shipped "
          "no exam).\n"]
    for name, c in contrasts.items():
        md.append(f"## {name}  ({c['a']} − {c['b']})  — {c['note']}")
        if not c["channels"]:
            md.append("_no joined pairs yet_\n")
            continue
        md.append("| channel (direction) | Δ mean | n items | classes+ / n "
                  "(p 2s) | MDE(n) | ≥MDE | dropped† |")
        md.append("|---|---|---|---|---|---|---|")
        for ch, s in c["channels"].items():
            if not s:
                md.append(f"| {ch} | — | 0 | | | | |")
                continue
            m = s["mde_at_n"]
            md.append(
                f"| {ch} ({DIR[ch]}) | {s['mean_delta']:+.4f} |"
                f" {s['n_items_trusted']} |"
                f" {s['sign_pos']}/{s['n_classes_trusted']}"
                f" (p={s['sign_p_two_sided']:.3f}) |"
                f" {('%.4f' % m) if m else 'n/a (v4)'} |"
                f" {'YES' if s['exceeds_mde'] else ('no' if s['exceeds_mde'] is not None else '—')} |"
                f" {','.join(s['dropped_untrusted_classes']) or '—'} |")
        md.append("")
    md.append("## near-copy rate by arm (tau=0.858)")
    for a, v in sorted(out["near_copy_rate_by_arm"].items()):
        md.append(f"- {a}: {v['rate']:.1%} of {v['n']}")
    md.append("\n## saturation flags by arm (SPEC §6.5 — flagged, not certified)")
    for a, v in sorted(out["saturation_rate_by_arm"].items()):
        md.append(f"- {a}: {v['rate']:.1%} of {v['n_flags']} flag-checks")
    md.append("\n## bridge check — v4 app_ref_v3 vs certified v3 app_ref")
    if bridge:
        md.append("| label | n common | mean |Δ| | max |Δ| |")
        md.append("|---|---|---|---|")
        for lab, b in sorted(bridge.items()):
            md.append(f"| {lab} | {b['n']} | {b['mean_abs']:.5f} |"
                      f" {b['max_abs']:.5f} |")
    else:
        md.append("_no overlapping labels scored yet_")
    (od / "contrasts_v4.md").write_text("\n".join(md) + "\n")

    tt = ["# Ladder v4 — tier table (trusted-class channel means; "
          "v4.0.0, NOT re-certified)\n",
          "| model·tier (arm) | n cls | " + " | ".join(
              f"{ch} ({DIR[ch].split()[0].split('=')[0]})"
              for ch in CHANNELS) + " | near-copy |",
          "|---|---|" + "---|" * (len(CHANNELS) + 1)]
    ARM_LABEL = {"r0": "base·P", "r1": "base·PE", "r1k": "base·PE-keyed",
                 "r1k_ext": "base·PE-ext", "r2_ckpt250": "spec·SEEN@250",
                 "r2_ckpt2000": "spec·SEEN@2000",
                 "r3_ckpt250": "spec·UNSEEN@250",
                 "r3_ckpt2000": "spec·UNSEEN@2000", "r3x": "spec·FOREIGN",
                 "ic3_a": "ic3·A held-in", "ic3_b": "ic3·B unseen",
                 "ic3_c": "ic3·C zero-shot", "ic3_x": "ic3·X foreign",
                 "ic2_r4": "ic2·R4 (frozen)", "ic2_r5": "ic2·R5 (frozen)",
                 "control_hold": "CONTROL hold", "control_lerp": "CONTROL lerp"}
    for arm in sorted(class_means, key=lambda a: list(ARM_LABEL).index(a)
                      if a in ARM_LABEL else 99):
        cm = class_means[arm]
        cells = []
        for ch in CHANNELS:
            tk = TRUST_KEY[ch]
            vals = [v[ch] for s, v in cm.items() if ch in v and
                    (tk is None or trust.get(s, {}).get(tk, False))]
            cells.append(f"{sum(vals)/len(vals):.3f} (n={len(vals)})"
                         if vals else "†")
        nc = out["near_copy_rate_by_arm"].get(arm)
        tt.append(f"| {ARM_LABEL.get(arm, arm)} | {len(cm)} | " +
                  " | ".join(cells) +
                  (f" | {nc['rate']:.0%}" if nc else " | —"))
    (od / "tier_table_v4.md").write_text("\n".join(tt) + "\n")
    print(f"[done] {len(rows)} rows, labels={lanes} -> {od}")


if __name__ == "__main__":
    main()
