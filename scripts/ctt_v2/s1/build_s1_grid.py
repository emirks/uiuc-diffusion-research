#!/usr/bin/env python
"""CTT v2 / S1 -- pin the specialist-counterfactual grid (A5 RULING 3, A1b Q1).

S1 = for each of the 11 trained specialist c2v LoRAs, generate transitions over endpoints
drawn from the audited union content pool, so the SAME content appears under DIFFERENT
transition manners.  With S4 out (Ruling 2), this is the mix's only non-S0 anti-copy signal
in the real-VFX visual domain.

Grid (A1b Q1d, adopted wholesale by Ruling 3):
    9 one-sided specialists  x 30 endpoints       = 270 clips
    2 two-sided  specialists x 60 endpoint pairs  = 120 clips
                                            total = 390 clips
    a designated 10-endpoint PROBE SET shared by all 11  = 110 of the 390
      -> the same-content x different-manner diagonal
    endpoints unique within a specialist; outside the probe set, disjoint ACROSS
      specialists too, so shared content is exactly the probe set and nothing else
    ~50/50 synth / humanvid per specialist, no specialist bank-pure (Q1c)
    two-sided rows draw A and B from DIFFERENT banks (Q1c)

Everything below is a pure function of (the union pool, the frozen eval split, seed 42).

HARD ASSERT (A3-F5b, restated by Ruling 3): S1 endpoints must be disjoint from
{eval endpoints, zs audited endpoints, held-in test clips}.  Clip classes are resolved via
`eval_ladder/prompts.py:clip_class()` -- NEVER by string-splitting a clip name.

The assert is run at SOURCE-SEQUENCE level, not just clip-id level.  A pure id comparison
passes vacuously and misses the real leak: the eval FOREIGN lane's endpoints are DAVIS
pseudo-clips with composite ids (`davis_bear_elephant`) built from DAVIS sequences that the
union pool also carries under their plain ids (`davis_bear`, `davis_elephant`).  Same
footage, different name.  Those are excluded here.

Usage
-----
  PY=/projects/illinois/eng/cs/jrehg/users/emirkisa/envs/diffusion/bin/python
  $PY scripts/ctt_v2/s1/build_s1_grid.py \
      --tau outputs/ctt_v2/s1/tau_endpoint.json \
      --out /projects/illinois/eng/cs/jrehg/users/emirkisa/misc/ctt_v2_final/S1_GRID.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "eval_ladder"))
import prompts  # noqa: E402  -- the AUTHORITY on clip -> class

POOL = REPO_ROOT / "data/processed/ctt_v2_strata/CONTENT_POOL_union.json"
REGISTRY = REPO_ROOT / "eval_ladder/registry.jsonl"
ARMS = REPO_ROOT / "eval_ladder/arms.yaml"
DAVIS_YAML = REPO_ROOT / "eval_ladder/davis.yaml"
SPLIT = REPO_ROOT / "data/processed/transitions_std121/split_v1.2.json"
STD = REPO_ROOT / "data/processed/transitions_std121"

#: campaign-private S1 output root (never the shared ladder2 video tree)
OUT_ROOT = "outputs/videos/ctt_v2_s1"
#: conditioning windows for pool endpoints, cut by the generator (idempotent)
CONDS_DIR = "data/processed/ctt_v2_strata/s1_conds"

SEED = 42
#: one generation seed for every S1 row -- S1 clips are training data, one clip per row
GEN_SEED = 42

#: sidedness of each specialist, VERIFIED from registry.jsonl (DOSSIER 1.11); 9 of 11 cannot
#: take a suffix anchor, so their episode is A -> effect(A), exactly S0's one-sided semantics
SPECIALISTS = {
    "spec_animalization": "one",
    "spec_color_rain": "one",
    "spec_gas_transformation": "one",
    "spec_hero_flight": "two",
    "spec_illustration_scene": "one",
    "spec_polygon": "one",
    "spec_portal": "one",
    "spec_shadow": "one",
    "spec_shadow_smoke": "two",
    "spec_super_fast_run": "one",
    "spec_wireframe": "one",
}
N_PROBE = 10            #: shared probe endpoints (A1b Q1d)
N_ONE_SIDED = 30        #: endpoints per one-sided specialist
N_TWO_SIDED = 60        #: endpoint PAIRS per two-sided specialist
PILOT_PER_SPEC = 3      #: A1b Q1e -- 33-clip pilot, 3 per specialist


# ------------------------------------------------------------------ frozen inputs
def davis_source_sequences() -> set[str]:
    """The raw DAVIS sequence names the eval FOREIGN roster is built from.

    davis.yaml pseudo-clips carry composite ids; the underlying `sequence:` fields are what
    the union pool also holds (as `davis_<sequence>`), so this is the join key that matters.
    """
    cfg = yaml.safe_load(DAVIS_YAML.read_text())
    seqs = {e["sequence"] for e in (cfg.get("one_sided") or {}).values()}
    for e in (cfg.get("two_sided") or {}).values():
        seqs |= {e["prefix"]["sequence"], e["suffix"]["sequence"]}
    return seqs


def eval_sets() -> dict:
    """The three sets S1 endpoints must avoid, plus the DAVIS source-sequence join."""
    rows = [json.loads(l) for l in REGISTRY.read_text().splitlines() if l.strip()]
    eval_endpoints = sorted({r["endpoint"] for r in rows if r.get("endpoint")})

    split = json.loads(SPLIT.read_text())
    quarantined = set(split["quarantined"])
    held_out = set(split["generalist_holdout"])
    audited = prompts.audited_clips()

    # "the 42 test clips" of Ruling 3 = the WHOLE test band of split v1.2 across all 39
    # classes (34 held-in + 8 in held-out classes), not the held-in subset. Verified: 42.
    zs_audited, test_clips, heldin_test, all_corpus = set(), set(), set(), set()
    for cls, entry in split["classes"].items():
        train = [c for c in entry["train"] if c not in quarantined]
        test = [c for c in entry["test"] if c not in quarantined]
        for clip in train + test:
            # class resolved through the authority, never by splitting the name
            assert prompts.clip_class(clip) == cls, f"class disagreement for {clip}"
        all_corpus |= set(train) | set(test)
        test_clips |= set(test)
        if cls in held_out:
            zs_audited |= {c for c in train + test if c in audited}
        else:
            heldin_test |= set(test)
    assert len(test_clips) == 42, f"expected the 42 test clips, got {len(test_clips)}"

    return {
        "eval_endpoints": eval_endpoints,
        "zs_audited_endpoints": sorted(zs_audited),
        "test_clips_42": sorted(test_clips),
        "heldin_test_subset_of_the_42": sorted(heldin_test),
        "all_corpus_clips": sorted(all_corpus),
        "davis_source_sequences": sorted(davis_source_sequences()),
    }


def load_pool() -> tuple[list[dict], list[list]]:
    pool = json.loads(POOL.read_text())
    return pool["training"], pool["near_dup_pairs_pinned_to_training"]


# ------------------------------------------------------------------ eligibility + assert
def build_eligible(pool_training: list[dict], sets: dict, near_dups: list[list]) -> tuple[list[dict], dict]:
    """Eligible S1 endpoints = pool training clips minus every leak/duplicate hazard."""
    banned_eval = set(sets["eval_endpoints"])
    banned_zs = set(sets["zs_audited_endpoints"])
    banned_test = set(sets["test_clips_42"])
    banned_davis = {f"davis_{s}" for s in sets["davis_source_sequences"]}
    banned_dup = {c for pair in near_dups for c in pair[:2]}

    excluded, eligible = {}, []
    for rec in pool_training:
        cid = rec["clip_id"]
        why = []
        if cid in banned_eval:
            why.append("eval_endpoint")
        if cid in banned_zs:
            why.append("zs_audited")
        if cid in banned_test:
            why.append("test_clip_42")
        if cid in set(sets["all_corpus_clips"]):
            why.append("corpus_clip")
        if cid in banned_davis:
            why.append("davis_eval_source_sequence")
        if cid in banned_dup:
            why.append("near_duplicate_pair")
        if why:
            excluded[cid] = why
        else:
            eligible.append(rec)

    report = {
        "pool_training": len(pool_training),
        "eligible": len(eligible),
        "excluded_total": len(excluded),
        "excluded_by_reason": {
            r: sorted(c for c, w in excluded.items() if r in w)
            for r in ("eval_endpoint", "zs_audited", "test_clip_42", "corpus_clip",
                      "davis_eval_source_sequence", "near_duplicate_pair")
        },
    }
    return eligible, report


def run_hard_assert(chosen: set[str], sets: dict) -> dict:
    """Ruling 3 / A3-F5b. Returns the evidence; raises if any intersection is non-empty."""
    banned_davis = {f"davis_{s}" for s in sets["davis_source_sequences"]}
    result = {
        "S1_endpoints": len(chosen),
        "vs_eval_endpoints": sorted(chosen & set(sets["eval_endpoints"])),
        "vs_zs_audited_endpoints": sorted(chosen & set(sets["zs_audited_endpoints"])),
        "vs_test_clips_42": sorted(chosen & set(sets["test_clips_42"])),
        "vs_davis_eval_source_sequences": sorted(chosen & banned_davis),
        "vs_all_corpus_clips": sorted(chosen & set(sets["all_corpus_clips"])),
        "sizes": {k: len(v) for k, v in sets.items()},
    }
    result["PASS"] = not any(result[k] for k in
                             ("vs_eval_endpoints", "vs_zs_audited_endpoints",
                              "vs_test_clips_42", "vs_davis_eval_source_sequences",
                              "vs_all_corpus_clips"))
    if not result["PASS"]:
        raise AssertionError(f"S1 endpoint intersection assert FAILED: {json.dumps(result, indent=1)}")
    return result


# ------------------------------------------------------------------ the draw
def out_name(arm: str, key: str, seed: int) -> str:
    """Readable per-clip filename, hashed only if the natural name would be absurd."""
    stem = f"{arm}__{key}"
    if len(stem) > 150:
        stem = f"{arm}__{hashlib.sha1(key.encode()).hexdigest()[:16]}"
    return f"{stem}__s{seed}.mp4"


TOKEN = "sksz"


def render_s1_prompt(a_desc: str, b_desc: str | None, sided: str) -> str:
    """The S0/corpus caption grammar (prompts.render_prompt), applied to the S1 endpoints.

        one-sided ->  "{A-role description}. sksz."
        two-sided ->  "{A-role description}. sksz. {B-role description}."

    prompts.render_prompt() itself reads the CORPUS caption store, which has no entry for a
    pool endpoint, so the grammar is reproduced here rather than reused -- identical shape,
    identical trimming, and the Ruling-9 invariant (` sksz.` exactly once, no outcome marker)
    asserted on every string.
    """
    trim = prompts._trim  # noqa: SLF001 - the campaign's one trimming rule, deliberately shared
    out = (f"{trim(a_desc)}. {TOKEN}." if sided == "one"
           else f"{trim(a_desc)}. {TOKEN}. {trim(b_desc)}.")
    assert out.count(f" {TOKEN}.") == 1, f"token appears {out.count(f' {TOKEN}.')}x: {out}"
    assert prompts.MARKER not in out, f"outcome marker leaked into an S1 prompt: {out}"
    return out


def build_grid(eligible: list[dict], sets: dict, store: dict) -> dict:
    by_bank = {"synth": [], "humanvid": []}
    for rec in eligible:
        by_bank[rec["bank"]].append(rec)
    for bank in by_bank:
        by_bank[bank].sort(key=lambda r: r["clip_id"])          # deterministic base order

    rng = random.Random(SEED)
    pools = {b: rng.sample(v, len(v)) for b, v in by_bank.items()}   # one shuffle, seed 42
    cursor = {"synth": 0, "humanvid": 0}
    used: set[str] = set()

    def take(bank: str, role: str | None = None) -> dict:
        """Next clip of `bank` in the seed-42 order; if `role` is given, next one that already
        has that role's description (see the probe-set note in main())."""
        while True:
            rec = pools[bank][cursor[bank]]
            cursor[bank] += 1
            if rec["clip_id"] in used:
                continue
            if role is not None and not (store.get(rec["clip_id"], {}) or {}).get(role):
                continue
            used.add(rec["clip_id"])
            return rec

    # -- the shared 10-endpoint probe set: 5 synth / 5 humanvid, INTERLEAVED so that any
    #    prefix of it (in particular the pilot's first 3) is bank-stratified by construction.
    #    Probe endpoints are restricted to clips that ALREADY carry the description they need,
    #    because the probe set must be renderable into a prompt TODAY (the Gemini project ran
    #    out of prepayment credits on 2026-07-28, so no new description can be generated).
    #    The restriction is applied to the probe set only; it is not a quality filter -- the
    #    described subset is itself a seed-42 per-(bank x role) sample from the M3 pilot.
    probe = [take("synth" if i % 2 == 0 else "humanvid", role="A") for i in range(N_PROBE)]
    #: B-side partner for each probe endpoint, opposite bank (Q1c), FIXED across both
    #: two-sided specialists so the two-sided diagonal is a same-pair x different-manner test
    probe_partner = [take("humanvid" if p["bank"] == "synth" else "synth", role="B")
                     for p in probe]

    rows, per_spec = [], {}
    for arm, sided in SPECIALISTS.items():
        n = N_ONE_SIDED if sided == "one" else N_TWO_SIDED
        entries = []
        # probe block first: same content, every specialist
        for i in range(N_PROBE):
            entries.append({"a": probe[i], "b": probe_partner[i] if sided == "two" else None,
                            "probe_index": i})
        # unique block: disjoint across specialists (the cursors never rewind)
        for _ in range(n - N_PROBE):
            if sided == "one":
                # alternate banks so each specialist lands ~50/50 and is never bank-pure
                bank = "synth" if len([e for e in entries if e["a"]["bank"] == "synth"]) * 2 < len(entries) else "humanvid"
                entries.append({"a": take(bank), "b": None, "probe_index": None})
            else:
                a = take("synth" if len(entries) % 2 == 0 else "humanvid")
                entries.append({"a": a, "b": take("humanvid" if a["bank"] == "synth" else "synth"),
                                "probe_index": None})

        for j, e in enumerate(entries):
            a, b = e["a"], e["b"]
            key = a["clip_id"] if b is None else f"{a['clip_id']}__{b['clip_id']}"
            a_desc = (store.get(a["clip_id"], {}) or {}).get("A")
            b_desc = None if b is None else (store.get(b["clip_id"], {}) or {}).get("B")
            renderable = bool(a_desc) and (b is None or bool(b_desc))
            rows.append({
                "row_id": f"S1__{arm}__{key}"[:200],
                "arm": arm,
                "specialist_class": arm[len("spec_"):],
                "sided": sided,
                "endpoint_a": a["clip_id"],
                "endpoint_a_bank": a["bank"],
                "endpoint_a_mp4": a["mp4"],
                "endpoint_b": None if b is None else b["clip_id"],
                "endpoint_b_bank": None if b is None else b["bank"],
                "endpoint_b_mp4": None if b is None else b["mp4"],
                "probe_index": e["probe_index"],
                "is_probe": e["probe_index"] is not None,
                "in_pilot": e["probe_index"] is not None and e["probe_index"] < PILOT_PER_SPEC,
                "slot": j,
                "seed": GEN_SEED,
                # Rendered where the per-(clip, role) description already exists. Probe rows
                # (and therefore every pilot row) are renderable by construction; the rest fill
                # in once the caption store is complete -- see prompt_status below.
                "prompt": render_s1_prompt(a_desc, b_desc, sided) if renderable else None,
                "prompt_status": "rendered" if renderable else "pending_caption_store",
                "out_path": f"{OUT_ROOT}/{arm}/{out_name(arm, key, GEN_SEED)}",
            })
        banks = [e["a"]["bank"] for e in entries] + [e["b"]["bank"] for e in entries if e["b"]]
        per_spec[arm] = {
            "sided": sided, "rows": len(entries),
            "synth": banks.count("synth"), "humanvid": banks.count("humanvid"),
            "bank_pure": len(set(banks)) == 1,
        }

    return {"rows": rows, "per_specialist": per_spec, "probe": probe,
            "probe_partner": probe_partner, "cursor": cursor}


# ------------------------------------------------------------------ gate control arm
def control_arm(sets: dict) -> list[dict]:
    """Real corpus clips of the same 11 classes -- the Gemini gate's instrument validation.

    Ruling 3(i) requires the blind 11-way class-ID batch gate to carry a control arm of REAL
    footage: if the judge cannot identify real corpus clips of these classes either, the
    instrument is broken and the generated batch's score means nothing.  Drawn from the
    TRAIN band (never an eval endpoint), matched n = PILOT_PER_SPEC per class, seed 42.
    """
    split = json.loads(SPLIT.read_text())
    quarantined = set(split["quarantined"])
    rng = random.Random(SEED)
    out = []
    for arm in SPECIALISTS:
        cls = arm[len("spec_"):]
        band = sorted(c for c in split["classes"][cls]["train"] if c not in quarantined)
        for clip in band:
            assert prompts.clip_class(clip) == cls, f"class disagreement for {clip}"
        pick = rng.sample(band, min(PILOT_PER_SPEC, len(band)))
        for clip in sorted(pick):
            out.append({"arm": arm, "class": cls, "clip": clip,
                        "mp4": str((STD / cls / f"{clip}.mp4").relative_to(REPO_ROOT))})
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", default="outputs/ctt_v2/s1/tau_endpoint.json")
    ap.add_argument("--descriptions", nargs="*", default=[
        "scripts/ctt_v2/captions/pilot_m3/round1/descriptions.json",
        "scripts/ctt_v2/captions/pilot_m3/round2/descriptions.json",
    ], help="per-(clip, role) description stores, later files win")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    store: dict[str, dict] = {}
    for rel in args.descriptions:
        path = REPO_ROOT / rel
        if not path.exists():
            print(f"[s1] description store missing, skipped: {rel}")
            continue
        for clip, roles in json.loads(path.read_text()).items():
            for role in ("A", "B"):
                if roles.get(role):
                    store.setdefault(clip, {})[role] = roles[role]

    sets = eval_sets()
    pool_training, near_dups = load_pool()
    eligible, elig_report = build_eligible(pool_training, sets, near_dups)
    grid = build_grid(eligible, sets, store)

    chosen = {r["endpoint_a"] for r in grid["rows"]} | {r["endpoint_b"] for r in grid["rows"] if r["endpoint_b"]}
    assert_result = run_hard_assert(chosen, sets)

    tau = json.loads((REPO_ROOT / args.tau).read_text())
    pilot = [r for r in grid["rows"] if r["in_pilot"]]
    assert len(pilot) == PILOT_PER_SPEC * len(SPECIALISTS) == 33, f"pilot is {len(pilot)} rows"
    assert len(grid["rows"]) == 390, f"grid is {len(grid['rows'])} rows, expected 390"
    assert len({r["out_path"] for r in grid["rows"]}) == 390, "output paths collide"
    assert not any(v["bank_pure"] for v in grid["per_specialist"].values()), "a specialist is bank-pure"
    probe_ids = [p["clip_id"] for p in grid["probe"]]
    for arm in SPECIALISTS:
        got = [r["endpoint_a"] for r in grid["rows"] if r["arm"] == arm]
        assert len(got) == len(set(got)), f"{arm} repeats an endpoint"
        assert got[:N_PROBE] == probe_ids, f"{arm} does not open on the shared probe set"
    non_probe = [r["endpoint_a"] for r in grid["rows"] if not r["is_probe"]]
    assert len(non_probe) == len(set(non_probe)), "non-probe endpoints are shared across specialists"
    assert all(r["prompt"] for r in pilot), "a pilot row has no prompt -- it cannot be generated"

    doc = {
        "created": "2026-07-28",
        "authority": ("A5 SYNTHESIS RULING 3 (grid/counts/endpoints/pilot/sidedness = A1b Q1, "
                      "quality gates swapped to Gemini + mechanical; NO DINOv2 anywhere in "
                      "data selection)"),
        "seed": SEED,
        "generation_seed": GEN_SEED,
        "deterministic": "pure function of (CONTENT_POOL_union.json, split_v1.2, davis.yaml, seed 42)",
        "inputs": {
            "content_pool": str(POOL.relative_to(REPO_ROOT)),
            "registry": str(REGISTRY.relative_to(REPO_ROOT)),
            "split": str(SPLIT.relative_to(REPO_ROOT)),
            "davis_roster": str(DAVIS_YAML.relative_to(REPO_ROOT)),
            "arms": str(ARMS.relative_to(REPO_ROOT)),
            "adapter_template": "outputs/training/ladder2/{arm}/checkpoints/lora_weights_step_02000.safetensors",
        },
        "counts": {"total_clips": len(grid["rows"]),
                   "one_sided_specialists": 9, "one_sided_clips": 9 * N_ONE_SIDED,
                   "two_sided_specialists": 2, "two_sided_clips": 2 * N_TWO_SIDED,
                   "probe_endpoints": N_PROBE, "probe_clips": N_PROBE * len(SPECIALISTS),
                   "pilot_clips": len(pilot),
                   "distinct_endpoint_clips": len(chosen)},
        "eligibility": elig_report,
        "HARD_ASSERT_endpoint_disjointness": assert_result,
        "tau_endpoint": {
            "value": tau["TAU_ENDPOINT"],
            "basis": tau["tau_basis"],
            "definition": tau["definition"],
            "space": tau["space"],
            "all_checkpoints": tau["all_checkpoints"],
            "final_checkpoint_step2000": tau["final_checkpoint_step2000"],
            "artifact": args.tau,
        },
        "mechanical_rejects": {
            "authority": "A5 Ruling 3(ii) -- per-clip HARD rejects, mechanical ONLY",
            "decode_corruption": "ffmpeg decode fails, or frame count != 121, or geometry != 480x640",
            "frozen_or_black": ("mean abs inter-frame delta over frames 9..120 < 1.0/255 (frozen), "
                                "or mean luma < 8/255 on >= 10% of frames (black)"),
            "endpoint_identity": (f"prefix rel-L2(gen[0:9], anchor) > tau = {tau['TAU_ENDPOINT']:.4f} "
                                  "-- same pixel-space measurement as the tau calibration"),
            "NOT_USED": "DINOv2 / any harness substrate -- banned from selection by Ruling 3(i)",
        },
        "batch_gate": {
            "authority": "A5 Ruling 3(i)",
            "instrument": "blind Gemini 11-way class identification",
            "model": "gemini-3.5-flash",
            "temperature": 0,
            "max_output_tokens": 2000,
            "note": ("only FLASH models work -- every *-pro-* returns HTTP 429 (DOSSIER 5.1); "
                     "thinking models return empty text at low output caps, hence >= 2000"),
            "bar_top1": 0.80,
            "chance": 1 / 11,
            "control_arm": {
                "what": "real corpus clips of the same 11 classes, same judge, same prompt",
                "why": "instrument validation -- a failing control invalidates the batch score",
                "n": PILOT_PER_SPEC * len(SPECIALISTS),
                "clips": control_arm(sets),
            },
            "acceptance_by_bank_audit": {"flag_if_difference_pp_gt": 15,
                                         "authority": "A1b Q1e / Ruling 3(iii)"},
        },
        "pre_registered_fallback": ("A5 Ruling 3: if the 33-clip pilot fails its gates, S1 DROPS "
                                    "entirely and the mix renormalizes to S0 15 / S2 85. The "
                                    "schedule does not slip."),
        "verdict_rule": {
            "PASS": ("batch top-1 >= 80% AND control arm top-1 >= 80% AND mechanical reject rate "
                     "<= 10% (<= 3 of 33)"),
            "FAIL_S1_DROPS": "batch top-1 < 80% with a passing control arm",
            "INSTRUMENT_INVALID": ("control arm top-1 < 80% -- the judge cannot read these classes "
                                   "off real footage either; re-adjudicate, do not blame S1"),
        },
        "prompts": {
            "grammar": ("one-sided: '{A-role description}. sksz.'   |   "
                        "two-sided: '{A-role description}. sksz. {B-role description}.'  "
                        "-- the corpus grammar of prompts.render_prompt(), Ruling 5"),
            "token": TOKEN,
            "sources": args.descriptions,
            "rendered": sum(1 for r in grid["rows"] if r["prompt"]),
            "pending_caption_store": sum(1 for r in grid["rows"] if not r["prompt"]),
            "OPERATIONAL_NOTE": (
                "The Gemini project's prepayment credits are DEPLETED as of 2026-07-28 "
                "(HTTP 429 'Your prepayment credits are depleted' on gemini-3.6-flash AND "
                "gemini-3.5-flash) -- this supersedes DOSSIER 1.4, which measured the key "
                "before the M3 caption pilot spent them. No new description can be generated "
                "until billing is topped up. Probe endpoints were therefore restricted to "
                "clips that already carry the role description they need, so the pilot is "
                "renderable today; the remaining 280 rows render when the store is complete. "
                "The blind Gemini batch gate (Ruling 3(i)) is blocked on the same credits, "
                "which is why generation is queued now and gated afterwards."),
            "provisional": ("pilot prompts come from the M3 pilot rounds (prompt-variant v2 "
                            "where available, else v1); the full 390 must re-render from the "
                            "final pinned caption store before the S1 root is assembled"),
        },
        "output_root": OUT_ROOT,
        "conds_dir": CONDS_DIR,
        "probe_set": [{"index": i, "clip_id": p["clip_id"], "bank": p["bank"],
                       "source": p["source"], "mp4": p["mp4"],
                       "two_sided_partner": grid["probe_partner"][i]["clip_id"],
                       "two_sided_partner_bank": grid["probe_partner"][i]["bank"],
                       "two_sided_partner_mp4": grid["probe_partner"][i]["mp4"],
                       "in_pilot": i < PILOT_PER_SPEC}
                      for i, p in enumerate(grid["probe"])],
        "specialists": grid["per_specialist"],
        "pilot_rows": [r["row_id"] for r in pilot],
        "rows": grid["rows"],
    }

    out = Path(args.out)
    out.write_text(json.dumps(doc, indent=1))
    print(f"[s1] eligible {elig_report['eligible']}/{elig_report['pool_training']} "
          f"(excluded {elig_report['excluded_total']})")
    for reason, clips in elig_report["excluded_by_reason"].items():
        if clips:
            print(f"[s1]   - {reason}: {len(clips)} {clips[:8]}")
    print(f"[s1] HARD ASSERT endpoint disjointness: "
          f"{'PASS' if assert_result['PASS'] else 'FAIL'} "
          f"({assert_result['S1_endpoints']} S1 endpoints vs "
          f"{assert_result['sizes']['eval_endpoints']} eval / "
          f"{assert_result['sizes']['zs_audited_endpoints']} zs / "
          f"{assert_result['sizes']['test_clips_42']} test / "
          f"{assert_result['sizes']['all_corpus_clips']} corpus)")
    print(f"[s1] tau_endpoint = {tau['TAU_ENDPOINT']:.4f}")
    print(f"[s1] {len(grid['rows'])} rows, {len(chosen)} distinct endpoint clips, "
          f"{len(pilot)} in pilot -> {out}")


if __name__ == "__main__":
    main()
