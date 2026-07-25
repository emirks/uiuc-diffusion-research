"""ctt_v2 S3 — freeze the camera-operator roster and the (op x content-pair) grid.

ADVISOR RULINGS IMPLEMENTED (fable-advisor, ctt_v2 S2/S3, rounds 1-2, 2026-07-25):

  R4  300 exact camera-ops x 6 content pairs = 1,800 clips. S3's job is a second operator
      MODALITY (parallax/camera manner that 2D shaders cannot fake), not the main factorial
      load — S2 carries that. m=6 still yields 30 ordered (ref,target) combos per op.
  R2  The 6 clips per op ARE the pool — no dedicated reference render. 12 DISTINCT endpoint
      clips per op, so every (ref,target) draw inside a block is content-disjoint.
  R3  Contents come from the 207 TRAINING endpoints only, and S3 SHOULD share pairs with S2:
      "identical content under a 2D shader op and a 3D camera op is the strongest same-content-
      different-operator contrast in the whole corpus — it spans modalities." Soft preference,
      never at the cost of the disjointness/diversity rules.
  R5  HOLD OUT the `spiral` family entirely: spiral IS dolly o orbit (cameras.py:80-83) and both
      primitives stay in training, so it is the one family whose holdout tests COMPOSITIONAL
      generalisation of seen primitives rather than removed capability. Plus ~30 random exact
      ops from the 6 retained families, for the cheap "unseen op, seen family" tier.
  R-rider  Timing is part of exact-op identity and is FROZEN across all 6 clips of an op; the
      retry axis is the CONTENT PAIR, never the timing.

Writes PLAN_S3.json (the frozen grid) and HOLDOUT_S3.json (the never-trained registry).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(HERE))

from diffusion.exp_utils import load_config  # noqa: E402

from engine3d import cameras, ops3d  # noqa: E402


def s3_op_id(op: ops3d.Operator3D, onset: int, release: int) -> str:
    """Exact-op identity: every operator field plus the frozen timing.

    `seed` is excluded — it only drives the handheld noise phase, and two ops that differ solely
    in that phase are the same manner. Every other field is part of the operator.
    """
    d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in op.__dict__.items()}
    d.pop("seed", None)
    key = json.dumps(d, sort_keys=True) + f"|{onset}|{release}"
    return f"{op.path}_{hashlib.sha1(key.encode()).hexdigest()[:10]}"


def build_pair_pool(ids, S, idx, *, n_pairs, max_cos, banned, prefer, prefer_frac, rng):
    """Degree-balanced pool of content pairs, preferring pairs already used by S2.

    `prefer` is the set of frozenset({A,B}) pairs in the S2 grid. We take a `prefer_frac` share
    from that set first (cross-modality contrast on identical content), then fill the remainder
    with fresh degree-balanced draws so S3 still covers content S2 did not.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[frozenset] = set()
    deg = {c: 0 for c in ids}

    shared = [tuple(sorted(p)) for p in prefer if all(c in deg for c in p)]
    rng.shuffle(shared)
    for a, b in shared[: int(round(n_pairs * prefer_frac))]:
        seen.add(frozenset((a, b)))
        deg[a] += 1
        deg[b] += 1
        pairs.append((a, b))
    n_shared = len(pairs)

    guard = 0
    while len(pairs) < n_pairs:
        guard += 1
        if guard > n_pairs * 400:
            raise RuntimeError(f"pair pool stalled at {len(pairs)}/{n_pairs}")
        a = min(ids, key=lambda c: (deg[c], rng.random()))
        cand = [b for b in ids if b != a and frozenset((a, b)) not in seen
                and frozenset((a, b)) not in banned and S[idx[a], idx[b]] <= max_cos]
        if not cand:
            continue
        lo = min(deg[b] for b in cand)
        pick = rng.choice([b for b in cand if deg[b] <= lo + 1])
        seen.add(frozenset((a, pick)))
        deg[a] += 1
        deg[pick] += 1
        pairs.append((a, pick))
    return pairs, n_shared


def main() -> None:
    t0 = time.time()
    cfg = load_config(HERE / "config_s3.yaml")
    s3, tim = cfg["s3"], cfg["timing"]
    rng = random.Random(cfg["runtime"]["seed"] + 1)

    # Same resolved content pool as S2 — one manifest, absolute mp4 paths, one embedding
    # matrix, participation-ratio check carried with it (see exp_081/build_content_pool.py).
    pool = json.loads((REPO_ROOT / cfg["inputs"]["content_pool"]).read_text())
    train_ids = [e["clip_id"] for e in pool["training"]]
    mp4 = {e["clip_id"]: e["mp4"] for e in pool["training"] + pool["reserved"]}
    assert pool["participation_ratio_pass"], "content pool failed the bank-level gates"
    d = pool.get("diagnostics_non_gating", {})
    print(f"[plan] content pool: {len(train_ids)} training endpoints "
          f"(v1 {d.get('n_training_v1','?')} + v2 {d.get('n_training_v2','?')})")

    E = np.load(REPO_ROOT / "data/processed/ctt_v2_strata/content_pool_emb.npy").astype(np.float64)
    erow = {c: i for i, c in enumerate(pool["ids"])}
    V = np.stack([E[erow[c]] for c in train_ids])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    S = V @ V.T
    idx = {c: i for i, c in enumerate(train_ids)}
    banned = {frozenset(p[:2]) for p in pool["near_dup_pairs_pinned_to_training"]}
    split = {"reserved_eval_only": [e["clip_id"] for e in pool["reserved"]]}

    # S2's pairs, for the cross-modality preference (ruling 3)
    s2_path = REPO_ROOT / cfg["inputs"]["s2_plan"]
    prefer: set[frozenset] = set()
    if s2_path.exists():
        prefer = {frozenset((p["A"], p["B"]))
                  for p in json.loads(s2_path.read_text())["pairs"]}
        print(f"[plan] S2 grid found: {len(prefer)} pairs available for cross-modality overlap")

    pairs, n_shared = build_pair_pool(
        train_ids, S, idx, n_pairs=s3["n_content_pairs"], max_cos=s3["pair_max_cos"],
        banned=banned, prefer=prefer, prefer_frac=s3["s2_pair_overlap_target"], rng=rng)
    cdeg: dict[str, int] = {c: 0 for c in train_ids}
    for a, b in pairs:
        cdeg[a] += 1
        cdeg[b] += 1
    print(f"[plan] pair pool: {len(pairs)} pairs ({n_shared} shared with S2 = "
          f"{n_shared/len(pairs):.0%}) | clip degree min/med/max = "
          f"{min(cdeg.values())}/{int(np.median(list(cdeg.values())))}/{max(cdeg.values())}")

    # ---- operator roster: 300 training + 30 held-out, over the 6 RETAINED families ---------
    families = [f for f in sorted(cameras.PATHS) if f != s3["holdout_family"]]
    assert s3["holdout_family"] in cameras.PATHS and len(families) == 6, \
        f"expected 6 retained families, got {families}"
    print(f"[plan] families: {len(cameras.PATHS)} total - holdout '{s3['holdout_family']}' "
          f"=> {len(families)} trainable: {families}")

    n_total = s3["n_ops"] + s3["holdout_n_ops"]
    base, extra = divmod(n_total, len(families))
    quota = {f: base + (1 if i < extra else 0) for i, f in enumerate(families)}

    def draw_timing() -> tuple[int, int]:
        w0, w1 = tim["window"]
        span = w1 - w0
        return (int(round(w0 + rng.random() * tim["jitter_frac"] * span)),
                int(round(w1 - rng.random() * tim["jitter_frac"] * span)))

    ops, seen_ids = [], set()
    for fam in families:
        made = 0
        while made < quota[fam]:
            op = ops3d.sample_operator(rng)
            op.path = fam
            op.easing = rng.choice(ops3d.PATH_EASINGS)
            op.amplitude = op.amplitude * s3["amplitude_scale"]
            if fam not in ("dolly", "spiral"):
                op.dolly_zoom = 0.0          # dolly-zoom is undefined off the dolly axis
            onset, release = draw_timing()
            oid = s3_op_id(op, onset, release)
            if oid in seen_ids:
                continue
            seen_ids.add(oid)
            ops.append({
                "op_id": oid, "family": fam, "onset": onset, "release": release,
                "describe": op.describe(),
                "params": {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in op.__dict__.items()},
            })
            made += 1
    rng.shuffle(ops)
    holdout_ops = ops[: s3["holdout_n_ops"]]
    train_ops = ops[s3["holdout_n_ops"]:]
    assert len(train_ops) == s3["n_ops"], f"{len(train_ops)} training ops != {s3['n_ops']}"
    print(f"[plan] roster: {len(ops)} unique exact ops = {len(train_ops)} training "
          f"+ {len(holdout_ops)} held out")

    # ---- bipartite grid: 6 primary pairs per op + ordered spares --------------------------
    m, n_cand = s3["contents_per_op"], s3["candidates_per_op"]
    pkey = [f"{a}|{b}" for a, b in pairs]
    pdeg = {k: 0 for k in pkey}
    grid: dict[int, list[int]] = {}
    order = list(range(len(train_ops)))
    rng.shuffle(order)
    for oi in order:
        used: set[str] = set()
        chosen: list[int] = []
        for pi in sorted(range(len(pairs)), key=lambda i: (pdeg[pkey[i]], rng.random())):
            a, b = pairs[pi]
            if a in used or b in used:
                continue
            chosen.append(pi)
            used |= {a, b}
            if len(chosen) == m:
                break
        assert len(chosen) == m, f"op {oi}: only {len(chosen)} endpoint-disjoint pairs available"
        for pi in chosen:
            pdeg[pkey[pi]] += 1
        spares = [pi for pi in sorted(range(len(pairs)),
                                      key=lambda i: (pdeg[pkey[i]], rng.random()))
                  if pi not in set(chosen)][: n_cand - m]
        grid[oi] = chosen + spares

    dv = list(pdeg.values())
    print(f"[plan] grid: ops-per-pair min/median/max = "
          f"{min(dv)}/{int(np.median(dv))}/{max(dv)} (target ~{len(train_ops)*m//len(pairs)})")

    plan = {
        "created": "2026-07-25", "stratum": "S3",
        "authority": "fable-advisor rulings 2,3,4,5 + riders (ctt_v2 S2/S3, rounds 1-2)",
        "engine_source": "exp_080_depth3d_realstream_121 @ cecf231/fc58617/e47c7f1 "
                         "(engine3d/ copied byte-identical to this experiment)",
        "contract": {"num_frames": cfg["inference"]["num_frames"],
                     "height": cfg["inference"]["height"], "width": cfg["inference"]["width"],
                     "fps": cfg["inference"]["fps"],
                     "streams": "both source clips play in lockstep; pure phases byte-exact"},
        "design": {
            "n_ops": len(train_ops), "contents_per_op": m, "n_clips": len(train_ops) * m,
            "n_content_pairs": len(pairs),
            "pairs_shared_with_s2": n_shared,
            "target_ops_per_pair": round(len(train_ops) * m / len(pairs), 2),
            "exact_op_definition": "(family, amplitude, sign, pivot, fov, axis, turns, easing, "
                                   "blend+params, dissolve+params, fog, focus, handheld, "
                                   "motion_blur, depth range/gamma, ONSET/RELEASE) — all frozen "
                                   "across the op's 6 clips",
            "reference_pairing": "DYNAMIC at train time: ref != target from the SAME op_id's 6 "
                                 "clips (30 ordered combos). No dedicated reference render.",
            "retry_policy": f"content-pair SWAP only, never timing; {s3['max_pair_attempts']} "
                            f"attempts to fill {m} slots, else the op is DROPPED",
        },
        "holdout": {
            "family": s3["holdout_family"],
            "family_rationale": "spiral IS dolly o orbit (cameras.py:80-83); both primitives stay "
                                "in training, so holding spiral out tests COMPOSITIONAL "
                                "generalisation of seen primitives, not removed capability",
            "n_holdout_ops": len(holdout_ops),
            "holdout_op_rationale": "unseen exact op, seen family — the cheap tier, symmetric "
                                    "with S2's structure",
        },
        "content": {"pool": "207 training endpoints", "pair_max_cos": s3["pair_max_cos"],
                    "reserved_eval_only": split["reserved_eval_only"]},
        "gate": cfg["gate"], "seed": cfg["runtime"]["seed"],
        "pairs": [{"pair_id": i, "A": a, "B": b, "A_mp4": mp4[a], "B_mp4": mp4[b],
                   "cos": round(float(S[idx[a], idx[b]]), 4),
                   "shared_with_s2": bool(frozenset((a, b)) in prefer),
                   "ops_planned": pdeg[pkey[i]]} for i, (a, b) in enumerate(pairs)],
        "ops": [{**o, "candidates": grid[i]} for i, o in enumerate(train_ops)],
    }
    (HERE / "PLAN_S3.json").write_text(json.dumps(plan, indent=1))
    (HERE / "HOLDOUT_S3.json").write_text(json.dumps(
        {"stratum": "S3", "created": "2026-07-25", "authority": "fable-advisor ruling 5",
         "tier_1_family": {"family": s3["holdout_family"],
                           "rationale": plan["holdout"]["family_rationale"]},
         "tier_2_exact_ops": holdout_ops,
         "trainable_families": families,
         "reserved_eval_only_endpoints": split["reserved_eval_only"],
         "note": "NEVER rendered into the training grid. Eval renders use these ops on BOTH "
                 "seen contents and the 20 reserved endpoints, giving the factorised cells "
                 "(unseen op x seen content, seen op x unseen content, unseen x unseen)."},
        indent=1))
    print(f"[plan] FROZEN -> PLAN_S3.json ({len(train_ops)} ops x {m} = "
          f"{len(train_ops)*m} clips) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
