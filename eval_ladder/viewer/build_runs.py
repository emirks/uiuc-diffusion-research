"""IC-LoRA trainings — build the multi-run results viewer.

    python eval_ladder/viewer/build_runs.py [--out outputs/reports/iclora_runs/index.html]

ONE page that holds EVERY IC-LoRA training run, with a chip per run. Adding the next training is
one entry in RUNS below — nothing else changes.

Provenance: forked from `eval_ladder/viewer/build.py` (+ `template.html`), which builds the ladder2
results viewer. That page is the published record of the ladder2 campaign and its numbers bind to
pre-registered claims, so it is NOT touched and NOT extended — this is a sibling. The fork is
deliberate: a shared template would couple that page's evolution to this one's.

WHAT IS DIFFERENT FROM build.py
  * a run is a first-class dimension. `build.py` has one hardcoded `generalist` tier; here each
    entry in RUNS gets its own tier column, and the chips multi-select over them. Both runs are on
    by default, so the DEFAULT VIEW IS THE COMPARISON.
  * scores carry their INSTRUMENT. `report_full.SCORES` is a module-level constant pointing at
    `outputs/eval/ladder2` and it ignores $LADDER_SCORES — so importing it and calling load_scored()
    silently reads the stale-artifact score set. That is the exact cross-instrument trap this
    campaign was bitten by, so here every score set is loaded explicitly, by path, and every
    generation is tagged with which artifact scored it. Nothing is implicit.
  * every join is asserted with counts. A silent no-op that exits 0 cost this campaign four
    debugging windows; see `check()`.

THE CARD MODEL is inherited verbatim and is still two levels with no view-time joins:

    card = one INPUT   (donor_class + endpoint + sidedness; ic_gen and ctt_v2 share `input_key`
                        field-for-field, so both runs' answers land in the SAME card natively)
    gen  = one arm's answer to that input, already averaged pool-refs -> seeds
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
REPO_ROOT = LADDER.parents[0]
LAB = REPO_ROOT.parent
sys.path.insert(0, str(LADDER))

import encode_conditioning as ec  # noqa: E402
import prompts  # noqa: E402
import report_full as rf  # noqa: E402
import run_eval  # noqa: E402

STD = REPO_ROOT / "data/processed/transitions_std121"
SEEDS = (42, 43)

# --------------------------------------------------------------------------------- the runs
#: One entry per IC-LoRA training. `gen_dir` is EXPLICIT (repo-relative) rather than derived from
#: $LADDER_GEN_STEP — that env var is what made the scorer look in `ctt_v2/` instead of
#: `ctt_v2__ck10000/` and silently score nothing. A checkpoint is part of the run's identity, so
#: a second checkpoint of the same adapter is simply a second entry.
RUNS = [
    {"id": "ic_gen", "arm": "ic_gen", "checkpoint": None,
     "label": "IC-LoRA generalist", "sub": "ladder2 · the incumbent",
     "gen_dir": "outputs/videos/ladder2/ic_gen", "registry": None},
    {"id": "ctt_v2", "arm": "ctt_v2", "checkpoint": 10000,
     "label": "CTT v2", "sub": "step 10,000 · rank 128 · one-way ref attention",
     "gen_dir": "outputs/videos/ladder2/ctt_v2__ck10000",
     "registry": "eval_ladder/registry_ctt_v2.jsonl"},
]
RUN_BY_ARM = {r["arm"]: r for r in RUNS}
RUN_TIER = {r["arm"]: f"run_{r['id']}" for r in RUNS}

# --------------------------------------------------------------------------- the score sets
#: Ordered by preference. A generation takes its numbers from the FIRST set that scored it, and
#: carries that set's id in `instr` so the page can badge it. `primary` is the comparison-valid
#: instrument: ic_gen and ctt_v2 were both rescored under it on eps, which is the only reason a
#: cross-run comparison is meaningful at all.
SCORE_SETS = [
    {"id": "rebuilt222", "primary": True,
     "path": "outputs/eval/ctt_v2_compare",
     "short": "rebuilt reference_v4 · 222-clip corpus dc2e139a",
     "label": "reference_v4 rebuilt on the pinned 222-clip corpus (dc2e139a); both adapters "
              "rescored together on eps, 2026-07-30"},
    {"id": "stale223", "primary": False,
     "path": "outputs/eval/ladder2",
     "short": "as-published ladder2 · 223-clip reference_v4",
     "label": "the as-published ladder2 scores, computed under the superseded 223-clip "
              "reference_v4 (aa28c6d5) before live_concert_2 was quarantined"},
]

#: Owner instruction 2026-07-30: the specialist generations must be visible here, as they are in
#: the ladder2 results viewer. Specialists were never rescored on eps (no GPU budget, and the owner
#: forbade new scoring), so their numbers exist ONLY under the superseded artifact.
#:
#: The advisor's two-class instrument policy (after the cross-build error was measured): a
#: stale-scored tier appears WITH its numbers, badged, and may enter the aggregate panel as its own
#: column — never merged into a run's column. The justification is on the page: the same 304 ic_gen
#: videos scored under BOTH artifacts differ by 0.09pp mean / 0.31pp max at CELL level, which is the
#: regime aggregates live in. Per-GENERATION the tail reaches 8.3pp, which is why no per-card
#: specialist-vs-run delta is ever drawn — that comparison happens only in the aggregate.
SECONDARY_IN_STATS = True

#: metric -> (label, direction, decimals, group). Groups become the collapsible tables.
METRICS = [
    ("app_ref", "M1a app_ref", "up", 3, "ref"),
    ("cam_zpr", "M1b cam_zpr", "down", 3, "ref"),
    ("obj_csls", "M1c obj_csls", "down", 3, "ref"),
    ("copy_max", "M2a copy_max", "down", 3, "ref"),
    ("cam_dtw", "cam_dtw", "up", 3, "ref"),
    ("cam_corr", "cam_corr", "up", 3, "ref"),
    ("obj_match", "obj_match", "up", 3, "ref"),
    ("app_ref_v3", "app_ref_v3", "up", 3, "ref"),
    ("cross", "cross", "info", 3, "ref"),
    ("margin", "M2b margin", "up", 3, "gen"),
    ("app_target", "app_target", "up", 3, "gen"),
    ("prefix_dino", "M3a prefix_dino", "up", 3, "gen"),
    ("prefix_lpips", "M3a prefix_lpips", "down", 4, "gen"),
    ("max_seam_z", "M3b max_seam_z", "down", 2, "gen"),
    ("scalar_depth", "depth", "info", 3, "gen"),
    ("scalar_depart", "depart", "info", 3, "gen"),
    ("scalar_arrive", "arrive", "info", 3, "gen"),
    ("scalar_core_frac", "core_frac", "info", 3, "gen"),
]
FLAGS = [("near_copy", "near_copy"), ("cross_high", "cross_high"),
         ("app_saturated", "app_sat"), ("core_degenerate", "core_degen"),
         ("intruder", "intruder")]

NOVELTY_ORDER = ["seen", "unseen", "zero_shot"]
CONTENT_ORDER = ["same", "cross", "foreign"]
NOVELTY_LABEL = {"seen": "seen<br><span>held-in training sample</span>",
                 "unseen": "unseen<br><span>held-in test sample</span>",
                 "zero_shot": "zero-shot<br><span>held-out sample</span>"}
CONTENT_LABEL = {"same": "same<br><span>test sample from reference's class</span>",
                 "cross": "cross<br><span>test sample from other class</span>",
                 "foreign": "foreign<br><span>DAVIS endpoints</span>"}

#: Context tiers bracket the runs and do NOT participate in the run chips — they are the invariant
#: yardstick. `copier` keeps the owner's 2026-07-23 ruling in its label: a no-adapter model handed a
#: reference is never a baseline, it is a copier.
#:
#: 🔴 THERE IS NO PROMPT+ENDPOINT BASELINE TIER ON THIS PAGE, and that is a ruling, not an omission.
#: `base_prompt`/`base_cond` were never scored under EITHER artifact (their generation lane was
#: stopped by owner call; base_cond has zero mp4 on disk). The two candidate substitutes are both
#: roster-confounded: the eps no-reference base rows are all `base:SP-*` cells (56.6%, and zero of
#: them land on a scored ctt_v2 card), and exp_072's base·PE is a separate 48-generation roster
#: (75.1%). Those differ by ~19pp from roster composition ALONE — larger than the entire
#: ctt_v2−ic_gen effect this page exists to show. A horizontal line from either would manufacture
#: the exact false comparison the page is built to avoid. The gap is stated on the page instead.
#:
#: `text_floor` is likewise absent, and for a structural reason rather than a policy one: all 12 of
#: its rows carry `endpoint: None` — it is a per-DONOR-CLASS floor (prompt with no anchors at all),
#: not a per-card row, so it cannot join a (donor, endpoint, sided) card. It joined 0/139 and was
#: dropped on the advisor's pre-stated condition. Its numbers live in the ladder2 results viewer.
CONTEXT_TIERS_BEFORE = ["specialist"]
CONTEXT_TIERS_AFTER = ["copier"]
TIER_LABEL = {
    "specialist": ["② SPECIALIST", "transition baked into the weights"],
    "copier": ["⚠ BASE + DEMO", "NOT a baseline — it copies the demo"],
}
#: tiers whose numbers come from the superseded artifact (no eps rescore exists)
BADGED_TIERS = {"specialist"}


def tier_of(r: dict) -> str | None:
    arm = r["arm"]
    if arm in RUN_TIER:
        return RUN_TIER[arm]
    if arm in ("base_prompt", "base_cond", "text_floor"):
        return None                         # see the two notes above
    if arm == "base":
        return "copier" if r.get("reference") else None
    if arm.startswith("spec_"):
        return "specialist"
    return None


def novelty_view(r: dict) -> str:
    if r["arm"].startswith("spec_"):
        return "seen" if r["cell"] == "SP-fit" else "unseen"
    return r["ref_novelty"]


def rel(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT))


def clip_video(clip: str) -> str | None:
    if prompts.is_davis(clip):
        return None
    p = STD / prompts.clip_class(clip) / f"{clip}.mp4"
    return rel(p) if p.exists() else None


def video_paths(row: dict) -> dict[str, str]:
    """Where a row's mp4 live. Baseline rows share one canonical video per (endpoint, sided) via
    video_key; a run's videos come from ITS OWN explicit gen_dir."""
    vk = row.get("video_key")
    if vk:
        d, name = vk.split("/", 1)
        base = REPO_ROOT / "outputs/videos/ladder2" / d
    else:
        run = RUN_BY_ARM.get(row["arm"])
        base = (REPO_ROOT / run["gen_dir"]) if run else REPO_ROOT / "outputs/videos/ladder2" / row["arm"]
        name = row["item_id"]
    out = {}
    for s in SEEDS:
        p = base / f"{name}__s{s}.mp4"
        if p.exists():
            out[str(s)] = rel(p)
    return out


# ------------------------------------------------------------------------------- score loading
def load_all_scores() -> tuple[dict, dict]:
    """item_id -> metrics (from the most-preferred set that scored it), and item_id -> set id.

    Each set is loaded by EXPLICIT path. `report_full.SCORES` is a module constant that ignores
    $LADDER_SCORES; assigning it here is the whole reason this function can be trusted."""
    ceil = run_eval.ceilings()
    registry = {r["item_id"]: r for r in run_eval.load_registry()}
    metrics: dict[str, dict] = {}
    instr: dict[str, str] = {}
    per_set_counts: dict[str, collections.Counter] = {}

    for ss in SCORE_SETS:
        path = (REPO_ROOT / ss["path"]).resolve()
        if not path.is_dir():
            raise SystemExit(f"score set '{ss['id']}' missing: {path}")
        run_eval.SCORES = path
        rf.SCORES = path                       # module-level and hardcoded — must be overridden
        assert rf.SCORES == path, "report_full.SCORES did not take the override"
        scored = rf.load_scored()

        acc: dict[str, dict[str, list]] = {}
        counts = collections.Counter()
        for (item, _seed), rows in scored.items():
            r = registry.get(item)
            if r is None or item in metrics:   # a more-preferred set already supplied it
                continue
            d = acc.setdefault(item, collections.defaultdict(list))
            c = rf.collapse(rows)
            for k, v in c.items():
                d[k].append(v)
            cls = r["gt_pool_class"]
            if cls in ceil and "app_ref" in c:
                d["pct"].append(c["app_ref"] / ceil[cls])
        for item, d in acc.items():
            metrics[item] = {m: rf.mean_or_nan(v) for m, v in d.items()}
            instr[item] = ss["id"]
            counts[registry[item]["arm"]] += 1
        per_set_counts[ss["id"]] = counts

    print("\n[scores] which instrument supplied each arm (first set wins):")
    for ss in SCORE_SETS:
        c = per_set_counts[ss["id"]]
        if not c:
            print(f"   {ss['id']:11s} —")
            continue
        print(f"   {ss['id']:11s} " + "  ".join(f"{a}={n}" for a, n in sorted(c.items())))
    return metrics, instr


#: the two degenerate references every transition must beat, as scored pseudo-arms
FLOORS = {"control_lerp": ("crossfade", "a linear dissolve between this card's own endpoints"),
          "control_hold": ("freeze", "hold the first frame — no motion at all")}
#: shown under the floor line so nobody equates these with POOL_YARDSTICK's headline
FLOOR_NOTE = ("Recomputed from this campaign's own ladder2 control rows, where each control is "
              "built from the card's own endpoints. Not comparable to POOL_YARDSTICK's 48% / 22%, "
              "which are exp_072's separately-constructed control arms (that lane re-aggregates "
              "to 43.6% / 18.9%). Roster is not the difference — the lane is. See RUN_RECORD §20.")


def control_floors(registry: dict) -> dict:
    """Recompute the crossfade / freeze floors FROM THIS PAGE'S OWN LANE.

    POOL_YARDSTICK.md's headline 48% / 22% are NOT quoted, and the reason is stronger than a
    roster difference — it is a DIFFERENT LANE. Those figures come from exp_072
    (outputs/eval/exp_072_pool_v4/), which builds its own control_lerp/control_hold arms over its
    own pairing; re-aggregating that lane reproduces them (43.6% / 18.9%). ladder2's control_lerp
    is instead constructed per-card from that card's own endpoints, and gives 30.3% / 17.5% here.
    Both are correct for their own lane; they are different quantities and must never be quoted
    against each other. Checked before concluding: roster does NOT explain the gap — the same eps
    control rows on the SP-* roster give 31.7% / 21.3%, and no roster × score-set combination
    reaches 48% (max 32.6%). See RUN_RECORD.md §20.

    A control is derived from the INPUT clips, so the same control content is scored once against a
    treatment row and again against its base twin — deduped here on (kind, item, seed) via the
    accumulator key, with `base:`-prefixed cells dropped so a floor is never counted twice."""
    ceil = run_eval.ceilings()
    path = (REPO_ROOT / SCORE_SETS[0]["path"]).resolve()      # floors are primary-instrument only
    acc: dict[str, dict[tuple, list]] = {k: {} for k in FLOORS}
    for f in sorted(path.glob("*/items.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            kind = r.get("arm", "")
            if kind not in FLOORS or "app_ref" not in r:
                continue
            head, _, _ref = r["item_id"].split("__", 1)[1].rpartition("__ref_")
            item, _, seed = head.rpartition("__s")
            t = registry.get(item)
            if t is None or t["cell"].startswith("base:") or t["gt_pool_class"] not in ceil:
                continue
            acc[kind].setdefault((item, seed, t["cell"]), []).append(
                r["app_ref"] / ceil[t["gt_pool_class"]])
    out = {}
    for kind, (label, blurb) in FLOORS.items():
        per: dict[str, list] = collections.defaultdict(list)
        for (_i, _s, cell), vs in acc[kind].items():
            per[cell].append(st.mean(vs))
        allv = [v for vs in per.values() for v in vs]
        if not allv:
            continue
        out[kind] = {
            "label": label, "blurb": blurb,
            "global": round(st.mean(allv) * 100, 2), "n": len(allv),
            # n>=8 per the advisor's rule: below that a per-cell floor is noise dressed as a line
            "cells": {c: {"pct": round(st.mean(v) * 100, 2), "n": len(v)}
                      for c, v in sorted(per.items()) if len(v) >= 8},
        }
    return out


def instrument_delta(registry: dict) -> dict:
    """ic_gen is scored in BOTH sets — the SAME video files. Their difference IS the cost of
    showing a stale-instrument number next to a rebuilt-instrument one, so the page states it
    instead of asserting the badge is harmless."""
    ceil = run_eval.ceilings()
    per: dict[str, dict] = {}
    for ss in SCORE_SETS:
        path = (REPO_ROOT / ss["path"]).resolve()
        run_eval.SCORES = path
        rf.SCORES = path
        out = {}
        for (item, seed), rows in rf.load_scored().items():
            r = registry.get(item)
            if r is None or r["arm"] != "ic_gen":
                continue
            c = rf.collapse(rows)
            cls = r["gt_pool_class"]
            if cls in ceil and "app_ref" in c:
                out[(item, seed)] = c["app_ref"] / ceil[cls]
        per[ss["id"]] = out
    a, b = per[SCORE_SETS[1]["id"]], per[SCORE_SETS[0]["id"]]     # stale, rebuilt
    shared = [k for k in a if k in b]
    if not shared:
        return {}
    by_cell: dict[str, list] = collections.defaultdict(list)
    for k in shared:
        by_cell[registry[k[0]]["cell"]].append((b[k] - a[k]) * 100)
    cells = {c: st.mean(v) for c, v in by_cell.items()}
    pg = [(b[k] - a[k]) * 100 for k in shared]
    return {"n": len(shared),
            "cell_mean_abs": round(st.mean([abs(v) for v in cells.values()]), 3),
            "cell_max_abs": round(max(abs(v) for v in cells.values()), 3),
            "gen_mean_abs": round(st.mean([abs(v) for v in pg]), 3),
            "gen_max_abs": round(max(abs(v) for v in pg), 3),
            "gen_over_1pp": sum(1 for v in pg if abs(v) > 1)}


# ------------------------------------------------------------------------------------- build
def build() -> dict:
    run_eval.EXTRA_REGISTRY = None
    extras = [r["registry"] for r in RUNS if r["registry"]]
    if extras:
        # load_registry() concatenates ONE extra file; more than one would need a loop upstream.
        assert len(extras) == 1, f"only one extra registry is wired: {extras}"
        run_eval.EXTRA_REGISTRY = REPO_ROOT / extras[0]
        assert run_eval.EXTRA_REGISTRY.exists(), f"missing {run_eval.EXTRA_REGISTRY}"

    rows = run_eval.load_registry()
    registry = {r["item_id"]: r for r in rows}
    ceil = run_eval.ceilings()
    metrics, instr = load_all_scores()
    idelta = instrument_delta(registry)
    floors = control_floors(registry)

    def gen_entry(r: dict) -> dict | None:
        vids = video_paths(r)
        m = metrics.get(r["item_id"])
        if m is None and not vids:
            return None
        m = m or {}
        cond = ("none" if r.get("conditioning") == "none" or not r["endpoint"]
                else "prefix+suffix" if r["sided"] == "two" else "prefix")
        iid = instr.get(r["item_id"])
        primary = SCORE_SETS[0]["id"]
        e = {
            "id": r["item_id"], "arm": r["arm"], "cell": r["cell"], "videos": vids,
            "novelty": novelty_view(r), "content": r["content"], "donor": r["donor_class"],
            "pct_type": r["pct_type"], "cond": cond, "ref": r.get("reference"),
            "mismatched_ref": bool(r.get("mismatched_reference")),
            "ceil": ceil.get(r["gt_pool_class"]), "tier": tier_of(r),
            "scored": bool(m), "instr": iid,
            # a gen only feeds the recomputed statistics if its instrument is the primary one —
            # this is what keeps every aggregate on the page single-instrument
            "stat": bool(m) and (iid == primary or SECONDARY_IN_STATS),
        }
        e["m"] = {k: (None if m.get(k) is None or m.get(k) != m.get(k) else round(m[k], 6))
                  for k, _l, _d, _dp, _g in METRICS}
        e["f"] = {k: (None if m.get(k) is None or m.get(k) != m.get(k) else round(m[k], 4))
                  for k, _l in FLAGS}
        e["pct"] = (None if m.get("pct") is None or m.get("pct") != m.get("pct")
                    else round(m["pct"], 6))
        return e

    all_tiers = CONTEXT_TIERS_BEFORE + [RUN_TIER[r["arm"]] for r in RUNS] + CONTEXT_TIERS_AFTER
    carded = set(all_tiers)

    def new_card(r: dict) -> dict:
        ep, sided = r["endpoint"], r["sided"]
        paths = ec.cond_paths(ep, sided)
        return {
            "key": f"{r['donor_class']}|{ep}|{sided}",
            "donor": r["donor_class"], "endpoint": ep, "sided": sided, "prompt": r["prompt"],
            "endpoint_class": r.get("endpoint_class"), "endpoint_split": r.get("endpoint_split"),
            "prefix_video": rel(paths["prefix"]),
            "suffix_video": rel(paths["suffix"]) if sided == "two" else None,
            "endpoint_video": clip_video(ep),
            "slots": {t: [] for t in all_tiers},
        }

    cards: dict[str, dict] = {}
    for r in rows:
        if tier_of(r) not in carded:
            continue
        g = gen_entry(r)
        if g is None:
            continue
        key = f"{r['donor_class']}|{r['endpoint']}|{r['sided']}"
        card = cards.get(key) or cards.setdefault(key, new_card(r))
        if r.get("reference"):
            g["ref_class"] = prompts.clip_class(r["reference"])
            g["ref_video"] = clip_video(r["reference"])
        card["slots"][g["tier"]].append(g)

    # a card exists only if at least one RUN answers it — this page is about the trainings
    run_tiers = [RUN_TIER[r["arm"]] for r in RUNS]
    cards = {k: c for k, c in cards.items() if any(c["slots"][t] for t in run_tiers)}

    # INPUTS band owns every input; output boxes only INDICATE what they received
    for card in cards.values():
        refs: dict[str, dict] = {}
        for slot in run_tiers + ["copier"]:
            for g in card["slots"][slot]:
                if not g.get("ref"):
                    continue
                e = refs.setdefault(g["ref"], {
                    "clip": g["ref"], "cls": g.get("ref_class"), "video": g.get("ref_video"),
                    "mismatched": g["mismatched_ref"], "tiers": []})
                if slot not in e["tiers"]:
                    e["tiers"].append(slot)
        card["refs"] = sorted(refs.values(), key=lambda e: e["clip"])

    # per-card head-to-head: the LAST run minus the FIRST, in pool-% points
    a_t, b_t = run_tiers[0], run_tiers[-1]
    for card in cards.values():
        a = [g["pct"] for g in card["slots"][a_t] if g.get("pct") is not None]
        b = [g["pct"] for g in card["slots"][b_t] if g.get("pct") is not None]
        card["delta"] = round((st.mean(b) - st.mean(a)) * 100, 2) if a and b else None

    nrank = {n: i for i, n in enumerate(reversed(NOVELTY_ORDER))}
    crank = {c: i for i, c in enumerate(reversed(CONTENT_ORDER))}

    def card_sort(c):
        t = [g for tt in run_tiers for g in c["slots"][tt]]
        return (min((nrank.get(g["novelty"], 9) for g in t), default=9),
                min((crank.get(g["content"], 9) for g in t), default=9), c["key"])
    ordered = sorted(cards.values(), key=card_sort)

    treatments = [g for c in ordered for t in run_tiers for g in c["slots"][t]]
    matrix = collections.Counter((g["novelty"], g["content"]) for g in treatments)

    n_spec = sum(1 for c in ordered if c["slots"]["specialist"])
    n_vid = sum(len(g["videos"]) for c in ordered for s in c["slots"].values() for g in s)

    tier_label = dict(TIER_LABEL)
    for i, r in enumerate(RUNS):
        tier_label[RUN_TIER[r["arm"]]] = [f"{'④⑤⑥⑦'[i]} {r['label'].upper()}", r["sub"]]

    return {
        "meta": {
            "title": "IC-LoRA trainings — results",
            "design_version": (LADDER / "VERSION").read_text().strip(),
            "instrument": "transition_eval 4.0.0 (m1a_S3)",
            "score_sets": SCORE_SETS,
            "primary_set": SCORE_SETS[0]["id"],
            "instrument_delta": idelta,
            "floors": floors, "floor_note": FLOOR_NOTE,
            "badged_tiers": sorted(BADGED_TIERS),
            "no_baseline_note":
                "No same-instrument, same-roster prompt+endpoint baseline exists for this roster. "
                "base_prompt / base_cond were never scored under either artifact (lane stopped "
                "2026-07-23); the eps no-reference base rows belong to the specialist roster and "
                "none land on a card here; exp_072's base·PE is a different 48-generation roster. "
                "Those two candidates differ by ~19pp on roster composition alone — more than the "
                "effect this page shows — so neither is drawn. Producing a real one requires new "
                "scoring, which is currently barred.",
            "registry_rows": len(rows), "generations": n_vid, "cards": len(ordered),
            "seeds": list(SEEDS), "px_prefix": ec.PX_PREFIX,
            "suffix_gen_frames": ec.SUFFIX_GEN_FRAMES, "frames": 121,
            "tiers": all_tiers, "run_tiers": run_tiers,
            "context_before": CONTEXT_TIERS_BEFORE, "context_after": CONTEXT_TIERS_AFTER,
            "spec_cards": n_spec,
            "record": "misc/ctt_v2_training/RUN_RECORD.md §19",
            "absent_tiers": [
                ["prompt + endpoint baseline",
                 "never scored under either artifact (lane stopped 2026-07-23); the two candidate "
                 "substitutes are different rosters and differ from each other by ~19pp"],
                ["text floor (prompt only)",
                 "all 12 rows have no endpoint — a per-donor-class floor, not a per-card row, so "
                 "it joins 0 of these cards. It is in the ladder2 results viewer."],
            ],
        },
        "runs": [{"id": r["id"], "arm": r["arm"], "tier": RUN_TIER[r["arm"]],
                  "label": r["label"], "sub": r["sub"], "checkpoint": r["checkpoint"],
                  "gens": sum(1 for g in treatments if g["arm"] == r["arm"])} for r in RUNS],
        "tier_label": tier_label,
        "metrics": [{"k": k, "label": l, "dir": d, "dp": dp, "group": g}
                    for k, l, d, dp, g in METRICS],
        "flags": [{"k": k, "label": l} for k, l in FLAGS],
        "novelty_order": NOVELTY_ORDER, "content_order": CONTENT_ORDER,
        "novelty_label": NOVELTY_LABEL, "content_label": CONTENT_LABEL,
        "matrix": {f"{n}|{c}": matrix.get((n, c), 0) for n in NOVELTY_ORDER for c in CONTENT_ORDER},
        "ceilings": {k: round(v, 6) for k, v in ceil.items()},
        "verdict": VERDICT,
        "cards": ordered,
    }


#: RUN_RECORD.md §19, the advisor's own three-register wording. On the page verbatim: a page that
#: showed only the favourable pool-% table would misrepresent this run, and one that led with
#: "GATE: FAIL" would misrepresent it the other way. Both registers, at once.
VERDICT = {
    "headline": "a null-to-modest result — not a failure of the adapter, not a success of the method",
    "registers": [
        {"k": "GATE", "v": "FAIL as written",
         "d": "clause (a) at the run-local fallback τ=0.2134. <b>Arm-invariant</b> (base 86.5/97.5, "
              "ic_gen 59.6/67.5, ctt_v2 67.3/77.5) and diagnosed as a fallback-calibration defect, "
              "not a property of the checkpoint. At the certified τ=0.858 <b>ctt_v2's copy rate is "
              "0.0%</b> and its maximum copy_max is 0.648 — nothing approaches the threshold. "
              "Clause (c) passed; clause (b) skipped by rule (info: +0.211)."},
        {"k": "PRIMARY CLAIM", "v": "NOT MET, direction consistent",
         "d": "+0.054 on G-unseen-cross against a committed +0.10, with all four cross/foreign "
              "cells positive at half to a fifth of the bar. Same-ontology no-regression holds. "
              "G-ref-control regressed −0.071 and is reported as a cost."},
        {"k": "MECHANISM", "v": "UNRESOLVED",
         "d": "The leak index is unmoved (G-unseen-cross +0.392 → +0.378). The training improved "
              "cross-ontology margin <b>without</b> reducing the reference-appearance leakage the "
              "campaign targeted."},
    ],
    "guards": [
        "“all arms fail the copy gate” must <b>not</b> be read as “the models copy” — at the "
        "calibrated τ=0.858 nobody copies, and the inline canary independently showed the adapter "
        "stopped reproducing demo content by step 1,000 (output-vs-demo MAD 0.08 → 0.42, flat to 10k).",
        "the +0.05 margins must <b>not</b> be sold as the predicted effect; they are half the "
        "committed bar.",
    ],
}


def check(data: dict) -> None:
    """Loud joins. Every failure mode this campaign actually hit exits 0 while producing nothing,
    so the build refuses to emit a page it cannot vouch for."""
    m, runs, cards = data["meta"], data["runs"], data["cards"]
    print("\n[join] run × card × generation")
    print(f"   {'run':10s} {'gens':>6s} {'cards':>7s} {'scored':>7s} {'in-stats':>9s}")
    bad = []
    for r in runs:
        t = r["tier"]
        gs = [g for c in cards for g in c["slots"][t]]
        nc = sum(1 for c in cards if c["slots"][t])
        ns = sum(1 for g in gs if g["scored"])
        nst = sum(1 for g in gs if g["stat"])
        print(f"   {r['id']:10s} {len(gs):6d} {nc:7d} {ns:7d} {nst:9d}")
        if not gs:
            bad.append(f"run '{r['id']}' produced ZERO generations")
        if ns != len(gs):
            bad.append(f"run '{r['id']}': {len(gs) - ns} of {len(gs)} generations are unscored")
    # the 1:1 grid is the reason a card can hold two runs at once — verify it, don't assume it
    sets = {r["tier"]: {c["key"] for c in cards if c["slots"][r["tier"]]} for r in runs}
    keys = list(sets)
    for i in range(len(keys) - 1):
        a, b = sets[keys[i]], sets[keys[i + 1]]
        if a != b:
            bad.append(f"cards for {keys[i]} and {keys[i+1]} are not 1:1 "
                       f"(+{len(b - a)} / −{len(a - b)})")
    for t in m["context_before"] + m["context_after"]:
        gs = [g for c in cards for g in c["slots"][t]]
        nc = sum(1 for c in cards if c["slots"][t])
        badge = "  ← badged (stale223)" if t in m["badged_tiers"] else ""
        print(f"   {t:10s} {len(gs):6d} {nc:7d} {sum(1 for g in gs if g['scored']):7d} "
              f"{sum(1 for g in gs if g['stat']):9d}{badge}")
        # the owner asked to SEE the specialists; an empty specialist join is the failure that
        # would look like a working page. Say the count out loud either way.
        if t == "specialist" and nc == 0:
            bad.append("specialist tier joined ZERO cards — the owner asked to see these")
    if not cards:
        bad.append("ZERO cards")
    # every column that carries numbers must declare which artifact produced them, or a future run
    # silently inherits 'eps' without saying so
    undeclared = {g["arm"] for c in cards for s in c["slots"].values() for g in s
                  if g["scored"] and not g.get("instr")}
    if undeclared:
        bad.append(f"arms with numbers but no instrument tag: {sorted(undeclared)}")
    if bad:
        for b in bad:
            print(f"   ✗ {b}")
        raise SystemExit("[join] refusing to emit — fix the wiring above")
    nspec = sum(1 for c in cards if c["slots"]["specialist"])
    print(f"   ✓ every run scored, card sets 1:1, every scored arm declares its instrument")
    print(f"   ✓ specialists join {nspec}/{len(cards)} cards "
          f"({len(cards) - nspec} without — zero-shot donors have no specialist by design)")
    f = m.get("floors") or {}
    for k, v in f.items():
        print(f"   floor {v['label']:10s} {v['global']:5.1f}%  n={v['n']:4d}  "
              f"per-cell where n≥8: {len(v['cells'])} cells")


def emit(data: dict, out: Path) -> None:
    depth = len(out.parent.relative_to(REPO_ROOT).parts)
    data["meta"]["rel"] = "../" * depth
    tpl = (HERE / "template_runs.html").read_text()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/reports/iclora_runs/index.html")
    args = ap.parse_args()
    data = build()
    check(data)
    out = REPO_ROOT / args.out
    emit(data, out)
    m = data["meta"]
    print(f"\n[viewer] {m['cards']} cards · {m['generations']} videos · "
          f"{len(data['runs'])} runs -> {args.out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"[viewer] serve:  python3 scripts/viewers/viewerctl.py serve   (port 8017)")


if __name__ == "__main__":
    main()
