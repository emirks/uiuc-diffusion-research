#!/usr/bin/env python3
"""Freeze the SEGUE human-study pair selection (zero-shot tier, blind).

Reads no scores. Joins SEGUE against its prior-work opponents on the row key
(endpoint, reference), picks 30 rows per task with a seeded greedy coverage
rule, and writes the frozen selection under misc/2026-09-22_user_study/.

Run with the aarch64 media python (has imageio_ffmpeg for probing):
    $LAB/envs-aarch64/ltx2/bin/python scripts/user_study/select_pairs.py

Outputs (all under misc/2026-09-22_user_study/, never served):
    pairs.json          210 pairs (30x4 TEG + 30x3 transfer)
    rows.json           the 60 picked rows (for later VLM-judge reuse)
    selection_report.md candidates, picks, histograms, probe summary
    probes.jsonl        per-clip probe record (frames/fps/duration/size)
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

import imageio_ffmpeg as iio

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "misc", "2026-09-22_user_study")
SEED = 2026
GEN_SEED = 42  # every clip is the s42 generation

# --- systems (label = KEY ONLY, never served) --------------------------------
SEGUE_HF = "store/gens/032_dualforce_dcg_w6/03_neutral_v3__dai"
SEGUE_ED = "store/gens/032_dualforce_dcg_w6/04_neutral_v3ed81__dai"
# per-task SEGUE arm (HF-reference grid, EffectData-reference grid); owner 2026-09-25 pm:
# TEG (two) = w=6 neutral, transfer (one) = w=3 effect prompt. Override with --segue-two / --segue-one.
SEGUE_ARMS = {
    "two": (SEGUE_HF, SEGUE_ED),
    "one": ("store/gens/031_dualforce_dcg_w3/05_effect_v3__dai",
            "store/gens/031_dualforce_dcg_w3/06_effect_v3ed81__dai"),
}

TASK_OPP = {
    "two": {  # transition effect generation, both endpoints given
        "Base LTX-2":   "store/gens/005_base_cond/06_effect_v3__dai",
        "VACE":         "store/gens/043_wan_vace/01_effect_v3__cc",
        "refVFX":       "store/gens/003_refvfx/04_effect_v3__cc",
        "Wan2.1 FLF2V": "store/gens/042_wan_flf2v/01_effect_v3__cc",
    },
    "one": {  # visual effect transfer, start endpoint given
        "VAP":       "store/gens/011_vap/05_author_native__dai",
        "VFXMaster": "store/gens/012_vfxmaster/05_author_native__dai",
        "refVFX":    "store/gens/003_refvfx/03_author_native__dai",
    },
}

N_PICK = 30


def apath(rel):
    return os.path.join(REPO, rel)


def load_grid(rel):
    rows = []
    with open(apath(os.path.join(rel, "grid.jsonl"))) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def key_map(rows, sided):
    """(endpoint, reference) -> row, for zero-shot rows of this sidedness.

    Some grids list every row line twice (byte-identical); collapse those.
    Two *different* item_ids on one key would be a real conflict -> abort.
    """
    m = {}
    for r in rows:
        if r.get("sided") != sided:
            continue
        if r.get("ref_novelty") != "zero_shot":
            continue
        k = (r["endpoint"], r["reference"])
        if k in m:
            if m[k]["item_id"] != r["item_id"]:
                sys.exit(f"conflicting rows for key {k}: "
                         f"{m[k]['item_id']} vs {r['item_id']}")
            continue  # exact duplicate line
        m[k] = r
    return m


def enumerate_candidates(task):
    """Candidate rows for a task (importable; reused by build_picker.py).

    Reads no scores. Loads the SEGUE grids (03 HF, 04 ED81) and every opponent
    grid of the task, builds the (endpoint, reference) key maps, and returns
    the candidate keys = intersection over SEGUE-union and every opponent.
    Identical to the enumeration select_pairs.main did inline.

    Returns dict:
        seg_hf_m, seg_ed_m : key -> SEGUE row (grid 03 / grid 04)
        opp_maps           : {label: {key -> opponent row}}
        candidates         : sorted list of (endpoint, reference) keys
    """
    opp = TASK_OPP[task]
    seg_hf_dir, seg_ed_dir = SEGUE_ARMS[task]
    seg_hf_m = key_map(load_grid(seg_hf_dir), task)
    seg_ed_m = key_map(load_grid(seg_ed_dir), task)
    seg_keys = set(seg_hf_m) | set(seg_ed_m)
    opp_maps = {lbl: key_map(load_grid(d), task) for lbl, d in opp.items()}
    cand = set(seg_keys)
    for m in opp_maps.values():
        cand &= set(m)
    return {"seg_hf_m": seg_hf_m, "seg_ed_m": seg_ed_m,
            "opp_maps": opp_maps, "candidates": sorted(cand)}


def clip_path(gen_dir, item_id):
    return apath(os.path.join(gen_dir, "videos", item_id + "__s42.mp4"))


def reference_path(seg_row, opp_maps, key):
    """Reference video shared by every system of a row.

    Prefer the reference_video field carried by the external grids; assert the
    opponents that carry it agree. Fall back to the canonical transitions path.
    """
    refs = set()
    for m in opp_maps.values():
        rv = m[key].get("reference_video")
        if rv:
            refs.add(rv)
    if len(refs) > 1:
        sys.exit(f"opponents disagree on reference_video for {key}: {refs}")
    if refs:
        p = next(iter(refs))
    else:
        p = apath(os.path.join(
            "data", "processed", "transitions_std121",
            seg_row["gt_pool_class"], seg_row["reference"] + ".mp4"))
    if not os.path.exists(p):
        sys.exit(f"reference video missing for {key}: {p}")
    return p


def greedy_pick(candidates, seg_of, rng, n=N_PICK, preselected=()):
    """Pick rows spreading over class, then endpoint, then reference.

    Each fill step chooses the unpicked candidate minimising
    (class_count, endpoint_count, reference_count, stable_random).

    ``preselected`` keys are taken as-is: they seed the coverage counters (so
    the rule keeps spreading around them) but are never re-picked. Filling stops
    at ``n`` total rows (preselected + fills); if preselected already reaches or
    exceeds ``n`` nothing is filled and every preselected row is kept.

    Returns (picked, filled): picked = list(preselected) + filled (in that
    order); filled = only the keys the rule added. With the defaults
    (preselected=(), n=N_PICK) this is the original blind greedy selection and
    consumes the rng identically.
    """
    # stable per-candidate random tiebreak, assigned in a deterministic order
    rand = {}
    for k in sorted(candidates):
        rand[k] = rng.random()
    cls_c = collections.Counter()
    ep_c = collections.Counter()
    ref_c = collections.Counter()
    remaining = set(candidates)
    for k in preselected:
        r = seg_of(k)
        cls_c[r["gt_pool_class"]] += 1
        ep_c[k[0]] += 1
        ref_c[k[1]] += 1
        remaining.discard(k)
    filled = []
    target = min(n, len(candidates))
    while len(preselected) + len(filled) < target and remaining:
        def keyfn(k):
            r = seg_of(k)
            return (cls_c[r["gt_pool_class"]], ep_c[k[0]], ref_c[k[1]], rand[k])
        best = min(remaining, key=keyfn)
        remaining.discard(best)
        r = seg_of(best)
        cls_c[r["gt_pool_class"]] += 1
        ep_c[best[0]] += 1
        ref_c[best[1]] += 1
        filled.append(best)
    return list(preselected) + filled, filled


def load_picks(path):
    """Read an owner picks.json: {"two": [[endpoint, reference], ...], "one": [...]}

    Extra top-level fields the picker writes (``outputs_revealed`` and, if true,
    ``outputs_revealed_at``) are accepted and returned; any other field is
    ignored. Returns (by_task, outputs_revealed, outputs_revealed_at).
    """
    with open(path) as f:
        picks = json.load(f)
    out = {}
    for task in ("two", "one"):
        out[task] = [tuple(item) for item in picks.get(task, [])]
    revealed = bool(picks.get("outputs_revealed", False))
    revealed_at = picks.get("outputs_revealed_at")
    return out, revealed, revealed_at


def apply_picks(task, picks, cand, seg_of, rng):
    """Owner picks for a task, validated against the candidates, then filled.

    Unknown key (not a candidate of this task) -> abort listing it. Duplicate
    picks are collapsed (first occurrence kept). Fewer than N_PICK picks -> the
    seeded greedy rule fills the remainder from the unpicked candidates; more
    than N_PICK -> all picks kept (no truncation).

    Returns (picked, owner, filled).
    """
    cand_set = set(cand)
    owner = []
    seen = set()
    unknown = []
    for item in picks.get(task, []):
        k = tuple(item)
        if k not in cand_set:
            unknown.append(k)
        elif k not in seen:
            owner.append(k)
            seen.add(k)
    if unknown:
        sys.exit(f"[picks] task \"{task}\": {len(unknown)} picked rows are not "
                 f"candidates: {unknown}")
    picked, filled = greedy_pick(cand, seg_of, rng, n=N_PICK, preselected=owner)
    return picked, owner, filled


def probe(path):
    g = iio.read_frames(path)
    meta = next(g)
    g.close()
    n, secs = iio.count_frames_and_secs(path)
    size = list(meta.get("size"))
    return {"frames": n, "fps": meta.get("fps"),
            "duration_s": round(secs, 3), "size": size}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--picks", help="owner picks.json "
                    "({\"two\": [[endpoint, reference], ...], \"one\": [...]}); "
                    "picked rows are kept as-is and the seeded greedy rule only "
                    "fills each task up to 30 (more than 30 picks are all kept)")
    ap.add_argument("--human-opponents", default="",
                    help="restrict the PAIRS (not the candidate rows) to these opponent labels, "
                    "as task=lbl1,lbl2;task=... e.g. \"two=Base LTX-2,refVFX\". Rows are still the "
                    "intersection over ALL opponents; unlisted opponents get no human pairs.")
    ap.add_argument("--segue-two", default=None, help="SEGUE arm for task two as hf_dir,ed_dir (store/gens/...)")
    ap.add_argument("--segue-one", default=None, help="SEGUE arm for task one as hf_dir,ed_dir (store/gens/...)")
    args = ap.parse_args()
    for t, v in (("two", args.segue_two), ("one", args.segue_one)):
        if v:
            hf, ed = [x.strip() for x in v.split(",")]
            SEGUE_ARMS[t] = (hf, ed)
    human_opp = {}
    for chunk in filter(None, args.human_opponents.split(";")):
        t, lbls = chunk.split("=", 1)
        human_opp[t.strip()] = [x.strip() for x in lbls.split(",") if x.strip()]
        bad = set(human_opp[t.strip()]) - set(TASK_OPP[t.strip()])
        if bad:
            sys.exit(f"--human-opponents: unknown label(s) for task {t!r}: {sorted(bad)}")
    picks = revealed = revealed_at = None
    if args.picks:
        picks, revealed, revealed_at = load_picks(args.picks)

    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)

    report = []
    report.append("# SEGUE human study — pair selection report\n")
    report.append(f"seed={SEED} · generation seed={GEN_SEED} · "
                  "zero-shot tier only · reads no scores\n")
    if picks is not None:
        line = f"owner picks made with outputs revealed: {'yes' if revealed else 'no'}"
        if revealed and revealed_at:
            line += f" (first revealed {revealed_at})"
        report.append(line + "\n")
    if human_opp:
        report.append("human-study opponents restricted (owner, 2026-09-22): " +
                      "; ".join(f"task {t}: {', '.join(v)}" for t, v in human_opp.items()) +
                      " — unlisted opponents keep their rows in the VLM judge only\n")

    all_rows = []
    all_pairs = []
    probe_records = []
    probe_cache = {}

    def do_probe(path, kind):
        if path not in probe_cache:
            rec = probe(path)
            if kind == "video" and rec["size"] != [480, 640]:
                sys.exit(f"unexpected resolution {rec['size']} for {path}")
            probe_cache[path] = rec
            probe_records.append({"path": path, "kind": kind, **rec})
        return probe_cache[path]

    picked_counts = {}
    for task in ("two", "one"):
        opp = TASK_OPP[task]
        ce = enumerate_candidates(task)
        seg_hf_m, seg_ed_m = ce["seg_hf_m"], ce["seg_ed_m"]
        opp_maps, cand = ce["opp_maps"], ce["candidates"]
        seg_keys = set(seg_hf_m) | set(seg_ed_m)

        def seg_of(k, _h=seg_hf_m, _e=seg_ed_m):
            return _h.get(k) or _e[k]

        def seg_dir(k, _h=seg_hf_m, _arms=SEGUE_ARMS[task]):
            return _arms[0] if k in _h else _arms[1]

        report.append(f"\n## Task \"{task}\"  ({'both endpoints given' if task=='two' else 'start endpoint given'})\n")
        report.append(f"- SEGUE arm: `{SEGUE_ARMS[task][0]}` / `{SEGUE_ARMS[task][1]}`")
        report.append(f"- SEGUE zero-shot rows: hf={len(seg_hf_m)} ed={len(seg_ed_m)} union={len(seg_keys)}")
        for lbl, m in opp_maps.items():
            report.append(f"- opponent `{lbl}`: {len(m)} zero-shot rows")
        report.append(f"- candidates (intersection over SEGUE + all opponents): **{len(cand)}**")

        if picks is not None:
            picked, owner_keys, filled_keys = apply_picks(task, picks, cand,
                                                          seg_of, rng)
            owner_set = set(owner_keys)
            report.append(f"- picked: **{len(picked)}** rows "
                          f"(owner {len(owner_keys)} · filled by rule "
                          f"{len(filled_keys)})\n")
            if filled_keys:
                print(f"[picks] task \"{task}\": {len(owner_keys)} owner picks, "
                      f"filled {len(filled_keys)} by rule:")
                for k in filled_keys:
                    print(f"[picks]   fill  {k[0]}  ref={k[1]}")
            else:
                print(f"[picks] task \"{task}\": {len(owner_keys)} owner picks, "
                      "no fill needed")
        else:
            picked, filled_keys = greedy_pick(cand, seg_of, rng)
            owner_set = set()
            report.append(f"- picked: **{len(picked)}** rows\n")
        picked_counts[task] = len(picked)

        # histograms over the picked rows
        cls_h = collections.Counter(seg_of(k)["gt_pool_class"] for k in picked)
        ep_h = collections.Counter(k[0] for k in picked)
        ref_h = collections.Counter(k[1] for k in picked)
        hf_n = sum(1 for k in picked if k in seg_hf_m)
        ed_n = len(picked) - hf_n
        report.append(f"- HF-reference rows (in grid 03): {hf_n} · "
                      f"ED-reference rows (grid 04): {ed_n}  *(informational)*")
        report.append(f"- distinct classes among picks: {len(cls_h)} · "
                      f"distinct endpoints: {len(ep_h)} · distinct references: {len(ref_h)}")
        report.append("\nClass histogram (picked rows):\n")
        report.append("| gt_pool_class | picked |")
        report.append("|---|---|")
        for cls, n in sorted(cls_h.items(), key=lambda x: (-x[1], x[0])):
            report.append(f"| {cls} | {n} |")

        report.append("\nPicked rows (source = owner pick / rule fill):\n")
        report.append("| endpoint | reference | gt_pool_class | source |")
        report.append("|---|---|---|---|")
        for k in picked:
            src = "owner" if k in owner_set else "rule"
            report.append(f"| {k[0]} | {k[1]} | {seg_of(k)['gt_pool_class']} | {src} |")

        for k in picked:
            sr = seg_of(k)
            ep, ref = k
            seg_clip = clip_path(seg_dir(k), sr["item_id"])
            do_probe(seg_clip, "video")
            ref_clip = reference_path(sr, opp_maps, k)
            do_probe(ref_clip, "video")
            start_still = apath(os.path.join("eval_ladder", "conds", ep + "_start9.mp4"))
            do_probe(start_still, "video")
            end_still = None
            if task == "two":
                end_still = apath(os.path.join("eval_ladder", "conds", ep + "_end9.mp4"))
                do_probe(end_still, "video")

            all_rows.append({
                "task": task, "endpoint": ep, "reference": ref,
                "gt_pool_class": sr["gt_pool_class"], "cell": sr["cell"],
                "content": sr["content"], "seed": GEN_SEED,
                "segue_grid": seg_dir(k), "segue_item_id": sr["item_id"],
                "segue_clip": seg_clip, "reference_clip": ref_clip,
                "start_still": start_still, "end_still": end_still,
                "is_ed_reference": k not in seg_hf_m,
            })

            for lbl, m in opp_maps.items():
                if task in human_opp and lbl not in human_opp[task]:
                    continue
                orow = m[k]
                opp_clip = clip_path(opp[lbl], orow["item_id"])
                do_probe(opp_clip, "video")
                all_pairs.append({
                    "task": task, "endpoint": ep, "reference": ref,
                    "gt_pool_class": sr["gt_pool_class"], "cell": sr["cell"],
                    "content": sr["content"], "seed": GEN_SEED,
                    "opponent": lbl,
                    "segue_clip": seg_clip, "opponent_clip": opp_clip,
                    "reference_clip": ref_clip,
                    "start_still": start_still, "end_still": end_still,
                })

    # shuffle then assign order-free ids
    rng.shuffle(all_pairs)
    width = max(3, len(str(len(all_pairs) - 1)))
    for i, p in enumerate(all_pairs):
        p_id = "p" + str(i).zfill(width)
        all_pairs[i] = {"pair_id": p_id, **p}

    with open(os.path.join(OUT, "pairs.json"), "w") as f:
        json.dump(all_pairs, f, indent=1)
    with open(os.path.join(OUT, "rows.json"), "w") as f:
        json.dump(all_rows, f, indent=1)
    with open(os.path.join(OUT, "probes.jsonl"), "w") as f:
        for rec in probe_records:
            f.write(json.dumps(rec) + "\n")

    # probe summary for the report
    n_video = sum(1 for r in probe_records if r["kind"] == "video")
    fps_h = collections.Counter(round(r["fps"], 2) for r in probe_records if r["kind"] == "video")
    bad = [r for r in probe_records if r["kind"] == "video" and r["size"] != [480, 640]]
    report.append("\n## Probe summary\n")
    report.append(f"- distinct source clips probed: **{len(probe_records)}** (all 480x640: {not bad})")
    report.append(f"- fps histogram (video sources): "
                  + ", ".join(f"{k}fps x{v}" for k, v in sorted(fps_h.items())))
    report.append("- per-clip records in `probes.jsonl`")

    report.append("\n## Totals\n")
    report.append(f"- rows: **{len(all_rows)}**  ·  pairs: **{len(all_pairs)}**")
    report.append(f"- distinct source media files: **{len(probe_records)}**")

    with open(os.path.join(OUT, "selection_report.md"), "w") as f:
        f.write("\n".join(report) + "\n")

    print(f"[select] rows={len(all_rows)} pairs={len(all_pairs)} "
          f"distinct_media={len(probe_records)}")
    print(f"[select] wrote {OUT}/pairs.json, rows.json, probes.jsonl, selection_report.md")
    # hard assertions (adaptive: owner picks may keep >30 rows in a task)
    exp_pairs = sum(picked_counts[t] * len(human_opp.get(t, TASK_OPP[t]))
                    for t in ("two", "one"))
    assert len(all_rows) == picked_counts["two"] + picked_counts["one"], len(all_rows)
    assert len(all_pairs) == exp_pairs, (len(all_pairs), exp_pairs)
    if picks is None:
        assert len(all_rows) == 60 and len(all_pairs) == 210
    print(f"[select] OK: {len(all_rows)} rows, {len(all_pairs)} pairs "
          f"(two={picked_counts['two']} one={picked_counts['one']}), "
          "all clips exist and probe 480x640")


if __name__ == "__main__":
    main()
