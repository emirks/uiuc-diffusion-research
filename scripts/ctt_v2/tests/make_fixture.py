"""CTT v2 — build a TEST fixture for the root machinery (no GPU, tiny tensors).

The real per-stratum latents are still in flight (S2a/S2b/S4 encodes running, S1 is a
33-clip pilot, the caption store is credit-blocked), so the assembler, the assert battery
and the dry-run epoch are exercised against a fixture that has the REAL STRUCTURE and
STUB PAYLOADS:

  * S0  — the REAL corpus stratum, real tensors, real captions (nothing is stubbed); it is
          consumed straight from `outputs/ctt_v2/inventories/S0.json`
  * S2a — the REAL op/clip/endpoint structure from
          `outputs/videos/ctt_v2_s2/full/meta/clips_shard*.jsonl` (7,990 clips / 799 ops)
  * S2b — the REAL 800-op plan from `experiments/exp_082_s2_humanvid/PLAN_S2_UNION.json`
          (op ids, shaders, content pairs)
  * S1  — the RULING-3 grid shape (390 clips: 9 one-sided arms x 30 + 2 two-sided x 60),
          grouped by ARM per A11 item 6 (so 11 groups and exactly 1,170 pairs at full size)
  * S4  — the REAL frozen selection from `data/processed/s4_refvfx/selection.json`
          (2,000 samples / 42 train triggers), grouped by trigger, one-sided

**TWO SHAPES.**  S4's stub latents carry `(5,14,26)` @ fps 16; every other stratum carries
`(16,20,15)` @ fps 24.  The metadata is real even though the payload is 3 KB instead of
1.2 MB, so masks, shape checks, token counts and the derived noise-schedule shifts are all
exercised for real — which is the whole point of A9 §5(iv) "extend the root asserts to two
shapes".

Captions are drawn from the 139 certified corpus captions, so they are leak-free by
construction and carry the real grammar — the caption assert runs on real strings.

Two role-scoped guarantees the fixture deliberately preserves so the asserts are never
vacuous:
  * a B-role occurrence of every `role_scoped_exclusions_for_caption_store` clip is forced
    into S2b if the plan offers one (the legal case: role B is the healthy anchor);
  * S2a keeps the 8 pre-registered inline-OOD ops in the INVENTORY, so `assemble_root.py`
    has something to exclude and assert A6 is not vacuous.

    python scripts/ctt_v2/tests/make_fixture.py --out <dir>            # default: small
    python scripts/ctt_v2/tests/make_fixture.py --out <dir> --preset full

EVERYTHING it writes lives under --out and is disposable.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import root_common as rc  # noqa: E402

REPO = rc.REPO_ROOT
S2A_META = "outputs/videos/ctt_v2_s2/full/meta/clips_shard*.jsonl"
S2B_PLAN = REPO / "experiments/exp_082_s2_humanvid/PLAN_S2_UNION.json"
S4_SELECTION = REPO / "data/processed/s4_refvfx/selection.json"

#: latent (F, H/32, W/32) + fps per stratum — the two shapes the root must hold.
#: 480x640x121 @24 -> (16,20,15) = 4,800 tokens ; 832x448x33 @16 -> (5,14,26) = 1,820.
STD_SHAPE, STD_FPS = (16, 20, 15), 24.0
S4_SHAPE, S4_FPS = (5, 14, 26), 16.0

#: sizes that keep the fixture small while landing the ruled mix inside +-0.5 pp against
#: S0's fixed 385 real pairs.  Deliberately tuned so the stub strata need a replica
#: multiplier of 2 while S0 needs 1 — otherwise the mix would be realised with one replica
#: each and the replica-duplication mechanism (the thing that makes the mix COUNTABLE)
#: would never be exercised.  `full` reproduces every stratum's real roster.
PRESETS = {
    "small": {"s2a_ops": 30, "s2b_ops": 30, "s2_clips_per_op": 10,
              "s1_arms": 2, "s1_clips_per_arm": 13, "s4_triggers": 2, "s4_clips": 43},
    "full": {"s2a_ops": None, "s2b_ops": None, "s2_clips_per_op": 10,
             "s1_arms": None, "s1_clips_per_arm": None, "s4_triggers": None, "s4_clips": None},
}


def stub_tensors(out: Path, stems: list[str], fhw, fps: float) -> dict:
    """Write tiny latent / cond_clean / conditions tensors with the REAL metadata keys."""
    import torch  # noqa: PLC0415

    dirs = {k: out / k for k in ("latents", "cond_clean", "conditions")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    f, h, w = fhw
    lat = {"latents": torch.zeros(4, 2, 2, 2, dtype=torch.bfloat16),
           "num_frames": f, "height": h, "width": w, "fps": float(fps)}
    cond = {"video_prompt_embeds": torch.zeros(8, 16, dtype=torch.bfloat16),
            "prompt_attention_mask": torch.ones(8, dtype=torch.int64),
            "audio_prompt_embeds": torch.zeros(8, 16, dtype=torch.bfloat16)}
    for stem in stems:
        for key, payload in (("latents", lat), ("cond_clean", lat), ("conditions", cond)):
            p = dirs[key] / f"{stem}.pt"
            if not p.exists():
                torch.save(payload, p)
    return {k: str(v) for k, v in dirs.items()}


def corpus_caption_pool() -> list[str]:
    """The 139 certified captions — real grammar, leak-free, ' sksz.' exactly once."""
    rows = rc.read_json(REPO / "eval_ladder/dataset/captions/dataset_captions.json")
    return [r["caption"] for r in rows]


def role_excluded_clips() -> dict:
    """{clip: {roles}} from the M3 adjudication — the fixture keeps their LEGAL roles."""
    role, _clip, _prov = rc.load_caption_store_exclusions()
    return role


# --------------------------------------------------------------------------------------
def build_s2a(out: Path, cfg: dict, caps: list[str]) -> Path:
    rows = []
    for f in sorted(glob.glob(str(REPO / S2A_META))):
        for line in open(f):
            if line.strip():
                rows.append(json.loads(line))
    ops = sorted({r["op_id"] for r in rows})
    if cfg["s2a_ops"]:
        ops = sorted(random.Random(0).sample(ops, min(cfg["s2a_ops"], len(ops))))
        # Keep ALL 8 pre-registered inline-OOD ops in the inventory.  Not a convenience:
        # A6 (per A11 item 1) requires every pre-registered op_id to RESOLVE in the S2a
        # inventory, precisely so a stale pre-registration that excludes nothing cannot
        # pass quietly.  A fixture that carried only some of them would fail A6 for the
        # right reason on the wrong grounds, and the 8-op exclusion would be untestable.
        prereg = rc.PREREG_INLINE_OOD
        if prereg.exists():
            have = {r["op_id"] for r in rows}
            keep = [o for o in rc.read_json(prereg)["op_ids"] if o in have]
            ops = sorted(set(ops) | set(keep))
    keep = set(ops)
    rows = [r for r in rows if r["op_id"] in keep]
    if cfg["s2_clips_per_op"]:
        per: dict[str, list] = {}
        for r in sorted(rows, key=lambda r: r["stem"]):
            per.setdefault(r["op_id"], []).append(r)
        rows = [r for v in per.values() for r in v[:cfg["s2_clips_per_op"]]]
    return _emit(out, "S2a", rows, caps, STD_SHAPE, STD_FPS)


def build_s2b(out: Path, cfg: dict, caps: list[str]) -> Path:
    plan = rc.read_json(S2B_PLAN)
    pairs = {p["pair_id"]: p for p in plan["pairs"]}
    rng = random.Random(2)
    ops = plan["ops"]
    if cfg["s2b_ops"]:
        ops = rng.sample(ops, min(cfg["s2b_ops"], len(ops)))
    # force a LEGAL (role-B) occurrence of every role-excluded clip into the fixture, so
    # assert A12 is exercised on a root that really contains the adjudicated clip
    want_b = {c for c, roles in role_excluded_clips().items() if "A" in roles}
    forced = {}
    for c in sorted(want_b):
        hit = next((pid for pid, p in sorted(pairs.items()) if p.get("B") == c), None)
        if hit is not None:
            forced[c] = hit
    n_per = cfg["s2_clips_per_op"] or 10
    rows = []
    for i, op in enumerate(sorted(ops, key=lambda o: o["op_id"])):
        cands = [c for c in op["candidates"] if c in pairs]
        if i == 0:
            cands = sorted(set(forced.values()) | set(cands[:n_per]))
        chosen = cands[:n_per] if len(cands) >= n_per else (cands * n_per)[:n_per]
        for j, pid in enumerate(chosen):
            rows.append({"stem": f"s2b_{i:04d}_c{j:02d}", "op_id": op["op_id"],
                         "shader": op["shader"], "A": pairs[pid]["A"], "B": pairs[pid]["B"]})
    return _emit(out, "S2b", rows, caps, STD_SHAPE, STD_FPS)


def build_s1(out: Path, cfg: dict, caps: list[str]) -> Path:
    """RULING 3 grid: 9 one-sided arms x 30 + 2 two-sided x 60 = 390 clips.

    The pairing group is the ARM, not a sub-chunk of it (A11 item 6: group=endpoint would
    pair same-content x different-op and break "reference = same operator, different
    content").  At full size that is 11 groups and 9*30*3 + 2*60*3 = 1,170 pairs — A1b's
    own count, which only reconciles under arm grouping.
    """
    pool = rc.read_json(rc.CONTENT_POOL)
    endpoints = sorted(r["clip_id"] for r in pool["training"])
    rng = random.Random(1)
    one = ["animalization", "color_rain", "gas_transformation", "illustration_scene", "polygon",
           "portal", "shadow", "super_fast_run", "wireframe"]
    two = ["hero_flight", "shadow_smoke"]
    plan = [(s, 30, "one") for s in one] + [(s, 60, "two") for s in two]
    if cfg["s1_arms"]:
        plan = plan[:cfg["s1_arms"] - 1] + plan[-1:]     # keep >=1 two-sided arm
        plan = plan[:cfg["s1_arms"]]
    rows = []
    for spec, n, sided in plan:
        if cfg["s1_clips_per_arm"]:
            n = min(n, cfg["s1_clips_per_arm"])
        for k in range(n):
            a = rng.choice(endpoints)
            b = rng.choice(endpoints) if sided == "two" else a
            rows.append({"stem": f"s1_{spec}_{k:03d}", "op_id": f"spec_{spec}",
                         "shader": None, "A": a, "B": b, "sided": sided, "class": spec})
    return _emit(out, "S1", rows, caps, STD_SHAPE, STD_FPS)


def build_s4(out: Path, cfg: dict, caps: list[str]) -> Path:
    """THE SECOND SHAPE.  Real frozen selection, grouped by trigger, 100% one-sided.

    Groups are the 42 train triggers; references are S4-native (ring within trigger), so
    reference and target share the ~2 s span — the deliberate RoPE decision of A9 §3.
    Endpoints are the refVFX sample ids: they are not corpus clips and not union-pool
    clips, so the endpoint-disjointness assert is exercised and lands empty honestly.
    """
    sel = rc.read_json(S4_SELECTION)
    triggers = list(sel["train_triggers"])
    if cfg["s4_triggers"]:
        triggers = sorted(random.Random(4).sample(triggers, min(cfg["s4_triggers"],
                                                               len(triggers))))
    keep = set(triggers)
    per: dict[str, list] = {}
    for r in sorted(sel["samples"], key=lambda r: r["k"]):
        if r["effect"] in keep:
            per.setdefault(r["effect"], []).append(r)
    rows = []
    for trig, samples in sorted(per.items()):
        if cfg["s4_clips"]:
            samples = samples[:cfg["s4_clips"]]
        for r in samples:
            rows.append({"stem": f"s4_{r['k']}", "op_id": trig.split()[0], "shader": None,
                         "A": f"refvfx_{r['k']}", "B": None, "sided": "one",
                         "class": None, "trigger": trig})
    return _emit(out, "S4", rows, caps, S4_SHAPE, S4_FPS)


# --------------------------------------------------------------------------------------
def _emit(out: Path, stratum: str, rows: list[dict], caps: list[str], fhw, fps: float,
          kind: str = "synthetic_op") -> Path:
    groups: dict[str, dict] = {}
    clips: dict[str, dict] = {}
    stems = [r["stem"] for r in rows]
    src = stub_tensors(out / stratum, stems, fhw, fps)
    rng = random.Random(sum(ord(c) for c in stratum))
    for r in rows:
        g = groups.setdefault(r["op_id"], {"class": r.get("class"), "shader": r.get("shader"),
                                           "sided": r.get("sided", "two"), "clips": []})
        g["clips"].append(r["stem"])
        eps = [e for e in (r.get("A"), r.get("B")) if e]
        clips[r["stem"]] = {
            "group": r["op_id"],
            "latents": f"{src['latents']}/{r['stem']}.pt",
            "cond_clean": f"{src['cond_clean']}/{r['stem']}.pt",
            "conditions": f"{src['conditions']}/{r['stem']}.pt",
            "caption": caps[rng.randrange(len(caps))],
            "endpoints": eps,
        }
    for g in groups.values():
        g["clips"] = sorted(g["clips"])
    inv = {"schema": rc.INVENTORY_SCHEMA, "stratum": stratum, "kind": kind,
           "endpoint_disjointness": True,
           "latent_shape": list(fhw), "fps": fps,
           "groups": dict(sorted(groups.items())), "clips": dict(sorted(clips.items())),
           "provenance": {
               "FIXTURE": "STUB TENSORS — test only, never trainable",
               "latent_shape": list(fhw),
               "tokens": rc.latent_tokens(fhw),
               "shift": round(rc.shift_for_tokens(rc.latent_tokens(fhw)), 6),
               "structure_source": {"S2a": S2A_META, "S2b": str(S2B_PLAN),
                                    "S1": "RULING 3 grid shape, grouped by arm (A11 item 6)",
                                    "S4": str(S4_SELECTION)}.get(stratum)}}
    n = sum(len(rc.ring_pairs(g["clips"])) for g in groups.values())
    inv["counts"] = {"groups": len(groups), "clips": len(clips), "pairs_if_unfiltered": n}
    path = out / f"{stratum}.json"
    rc.write_json(path, inv)
    print(f"[fixture] {stratum}: {len(groups)} groups / {len(clips)} clips / {n} pairs / "
          f"shape {list(fhw)} @ {fps} fps ({rc.latent_tokens(fhw)} tokens, "
          f"shift {rc.shift_for_tokens(rc.latent_tokens(fhw)):.4f}) -> {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="small", choices=sorted(PRESETS))
    ap.add_argument("--strata", default="S1,S2a,S2b,S4",
                    help="S0 is never stubbed — it is consumed from its real inventory")
    for k in ("s2a-ops", "s2b-ops", "s2-clips-per-op", "s1-arms", "s1-clips-per-arm",
              "s4-triggers", "s4-clips"):
        ap.add_argument(f"--{k}", type=int, default=None, help="override the preset")
    args = ap.parse_args()

    cfg = dict(PRESETS[args.preset])
    for k in list(cfg):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    caps = corpus_caption_pool()
    want = set(args.strata.split(","))
    builders = {"S1": build_s1, "S2a": build_s2a, "S2b": build_s2b, "S4": build_s4}
    for name, fn in builders.items():
        if name in want:
            fn(out, cfg, caps)
    rc.write_json(out / "FIXTURE.json", {"preset": args.preset, "config": cfg,
                                         "strata": sorted(want),
                                         "shapes": {"corpus_format": list(STD_SHAPE),
                                                    "S4": list(S4_SHAPE)},
                                         "WARNING": "STUB TENSORS — never trainable"})
    print(f"[fixture] -> {out}")


if __name__ == "__main__":
    main()
