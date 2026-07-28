"""ctt_v2 S2 — freeze the operator roster and the sparse bipartite (op x content-pair) grid.

ADVISOR RULINGS IMPLEMENTED (fable-advisor, ctt_v2 S2/S3 campaign, rounds 1-2, 2026-07-25):

  R1  800 exact ops x 10 content pairs = 8,000 clips. An EXACT OP = (shader, uniforms, easing,
      onset/release, flip, swap) and ALL of it is frozen across the op's 10 clips — timing
      included. "Timing is part of the manner the reference must demonstrate; pace and placement
      are exactly the kind of thing we want transferred."
  R1a Ops drawn only from keep-set MINUS the held-out families (62 shaders), balanced.
  R1b The op's 10 pairs must be ENDPOINT-DISJOINT (20 distinct clips per op) and CLIP-diverse:
      reject candidate pairs with cos(A,B) > 0.85, never use a known near-dup pair as (A,B).
  R1c Incidence integrity is a GATE: every op ends with exactly 10 gate-passed clips, else the
      whole op is dropped and a replacement is resampled from the same shader.
  R2  The op's 10 clips ARE the pool — there is no 11th reference render. Reference and target
      are drawn from the same 10 at TRAIN time (dynamic pairing); endpoint-disjointness across
      the 10 makes any (ref, target) draw automatically content-disjoint.
  R3  Content pairs come from the 207 TRAINING endpoints only (the 20 reserved are eval-only).
  R-riders  Retry is by CONTENT-PAIR SWAP, never by redrawing timing. Budget 25 pair attempts to
      fill 10 slots (negative-binomial mean ~17 at p=0.59, sd ~3.4 => 25 ~ mean+2.4sd, so only
      ~1-2% of healthy ops hit the drop path spuriously). Replacement pairs are drawn with a
      usage-balancing preference so the ~333-pair / ~24-ops-per-pair structure stays regular.

Ops are PRE-VALIDATED here against gate-2 (endpoint identity at production resolution) on a
probe set of pairs, using a cached endpoint-frame bank, so a roster of 800 structurally sound
ops is frozen before a single 121-frame render is spent.

Writes PLAN_S2.json (the frozen grid) and HOLDOUT_S2.json (the never-trained registry).
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
sys.path.insert(0, str(REPO_ROOT / "experiments/exp_077_synth_stratum"))

from diffusion.exp_utils import load_config  # noqa: E402

import streams_real as sr  # noqa: E402
from engine import operators, shaders, videoio  # noqa: E402
from engine.glrunner import GLRunner  # noqa: E402

CONFIG_PATH = HERE / "config_s2.yaml"

# ---------------------------------------------------------------------------
# HELD OUT — advisor ruling 5. Never rendered into the training grid.
# Principle: "a held-out operator must have a same-mechanism cousin in training. Removing an
# entire mechanism genre tests extrapolation to unseen physics — failure there says nothing
# about manner-reading. The informative probe is unseen exact operator / unseen family, SEEN
# GENRE." Each entry is (held-out shader, cousins retained in training).
# `SimpleZoomOut` was deliberately NOT chosen: it supplied 2 of the 5 confirmed floor-flag BAD
# clips in the D2 audit, and eval probes should come from clean-rendering shaders.
HOLDOUT_S2 = {
    "wipeUp":            ["wipeDown", "wipeLeft", "wipeRight"],
    "VerticalOpen":      ["HorizontalOpen", "VerticalClose"],
    "fadegrayscale":     ["fade", "fadecolor"],
    "ripple":            ["WaterDrop", "Dreamy"],
    "scale-in":          ["SimpleZoom", "CrossZoom"],
    "randomsquares":     ["pixelize", "squareswire"],
    "burn0":             ["burn", "FilmBurn"],
    "directionalwarp":   ["crosswarp"],
    "DefocusBlur":       ["LinearBlur"],
    "parametric_glitch": ["GlitchDisplace", "StripDatamoshGlitch"],
}


def s2_op_id(shader: str, params: dict, easing: str, flip: str, swap: bool,
             timing: dict) -> str:
    """Identity of an EXACT OP under the S2 definition — timing INCLUDED.

    (shader, uniforms, easing, onset/release, flip, swap). Timing is part of the manner the
    reference demonstrates (pace and placement are transferable properties), so two draws that
    differ only in timing are different operators and must not share an id.
    """
    key = (f"{shader}|{sorted(params.items())}|{easing}|{flip}|{swap}"
           f"|{timing['onset']:.6f}|{timing['release']:.6f}")
    return f"{shader}_{hashlib.sha1(key.encode()).hexdigest()[:10]}"


def build_pair_pool(ids: list[str], S: np.ndarray, idx: dict, *, n_pairs: int,
                    max_cos: float, banned: set, rng: random.Random) -> list[tuple[str, str]]:
    """Degree-balanced pool of `n_pairs` content pairs over `ids`.

    Every clip should carry a similar number of pairs (207 clips / 333 pairs => ~3.2 each), so
    no endpoint dominates the stratum. Partners are chosen from the currently least-used clips
    that satisfy the diversity rule.
    """
    deg = {c: 0 for c in ids}
    seen: set[frozenset] = set()
    pairs: list[tuple[str, str]] = []
    guard = 0
    while len(pairs) < n_pairs:
        guard += 1
        if guard > n_pairs * 400:
            raise RuntimeError(f"pair pool stalled at {len(pairs)}/{n_pairs} — relax max_cos")
        a = min(ids, key=lambda c: (deg[c], rng.random()))
        cand = [b for b in ids
                if b != a and frozenset((a, b)) not in seen
                and frozenset((a, b)) not in banned
                and S[idx[a], idx[b]] <= max_cos]
        if not cand:
            continue
        lo = min(deg[b] for b in cand)
        pick = rng.choice([b for b in cand if deg[b] <= lo + 1])
        seen.add(frozenset((a, pick)))
        deg[a] += 1
        deg[pick] += 1
        pairs.append((a, pick))
    return pairs


def main() -> None:
    t_start = time.time()
    cfg = load_config(CONFIG_PATH)
    smp, s2 = cfg["sampling"], cfg["s2"]
    H, W = cfg["inference"]["height"], cfg["inference"]["width"]
    T, K = cfg["inference"]["num_frames"], cfg["inference"]["anchor_frames"]
    rng = random.Random(cfg["runtime"]["seed"])

    # ---- 1. content pool: the TRAINING endpoints (advisor ruling 3) -----------------------
    # CONTENT_POOL.json resolves the round-1 bank and the (optional) expansion bank into one
    # manifest with absolute mp4 paths and one aligned embedding matrix, and carries the
    # participation-ratio check. Re-running build_content_pool.py --with-v2 and then this
    # planner is the whole re-freeze after an expansion.
    pool = json.loads((REPO_ROOT / cfg["inputs"]["content_pool"]).read_text())
    train_ids = [e["clip_id"] for e in pool["training"]]
    mp4 = {e["clip_id"]: e["mp4"] for e in pool["training"] + pool["reserved"]}
    assert pool["participation_ratio_pass"], (
        "content pool failed the bank-level gates: "
        f"A {pool['gates']['after_trim']['gate_a_mean_cos']} "
        f"(<= {pool['gates']['after_trim']['gate_a_bar']}), "
        f"B {pool['gates']['after_trim']['gate_b_matched_pr']} "
        f"(>= {pool['gates']['after_trim']['gate_b_bar']}) — do not plan against this pool")
    d = pool.get("diagnostics_non_gating", {})
    g = pool["gates"]["after_trim"]
    print(f"[plan] content pool: {len(train_ids)} training endpoints "
          f"(v1 {d.get('n_training_v1','?')} + v2 {d.get('n_training_v2','?')}), "
          f"{pool['n_reserved']} reserved held out | gate A {g['gate_a_mean_cos']} "
          f"gate B {g['gate_b_matched_pr']} (n={g['gate_b_match_n']})")

    E = np.load(REPO_ROOT / "data/processed/ctt_v2_strata/content_pool_emb.npy").astype(np.float64)
    erow = {c: i for i, c in enumerate(pool["ids"])}
    V = np.stack([E[erow[c]] for c in train_ids])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    S = V @ V.T
    idx = {c: i for i, c in enumerate(train_ids)}
    banned = {frozenset(p[:2]) for p in pool["near_dup_pairs_pinned_to_training"]}
    split = {"reserved_eval_only": [e["clip_id"] for e in pool["reserved"]],
             "n_reserved": pool["n_reserved"]}

    pairs = build_pair_pool(train_ids, S, idx, n_pairs=s2["n_content_pairs"],
                            max_cos=s2["pair_max_cos"], banned=banned, rng=rng)
    pair_key = [f"{a}|{b}" for a, b in pairs]
    cdeg: dict[str, int] = {c: 0 for c in train_ids}
    for a, b in pairs:
        cdeg[a] += 1
        cdeg[b] += 1
    print(f"[plan] pair pool: {len(pairs)} pairs | clip degree min/med/max = "
          f"{min(cdeg.values())}/{int(np.median(list(cdeg.values())))}/{max(cdeg.values())} | "
          f"cos(A,B) max = {max(S[idx[a], idx[b]] for a, b in pairs):.3f}")

    # ---- 2. shader roster: keep-set MINUS holdout (advisor ruling 1a/5) -------------------
    policy = json.loads((REPO_ROOT / cfg["inputs"]["policy"]).read_text())
    keep = sorted(policy["keep_shaders"])
    missing = [s for s in HOLDOUT_S2 if s not in keep]
    assert not missing, f"held-out shader not in the keep set: {missing}"
    for h, cousins in HOLDOUT_S2.items():
        live = [c for c in cousins if c in keep and c not in HOLDOUT_S2]
        assert live, f"holdout {h} has no surviving cousin in training — violates ruling 5"
    trainable = [s for s in keep if s not in HOLDOUT_S2]
    print(f"[plan] shaders: keep {len(keep)} - holdout {len(HOLDOUT_S2)} = {len(trainable)} trainable")

    n_ops = s2["n_ops"]
    base, extra = divmod(n_ops, len(trainable))
    quota = {s: base + (1 if i < extra else 0) for i, s in enumerate(trainable)}
    assert sum(quota.values()) == n_ops

    # ---- 3. endpoint-frame cache (for cheap gate-2 pre-validation) -----------------------
    # INCREMENTAL: the training pool changes between freezes (reserve re-cuts, the letterbox
    # exclusion, the expansion), so the cache tops up rather than being all-or-nothing.
    cache_npz = REPO_ROOT / cfg["inputs"]["endpoint_frame_cache"]
    fa_cache, fb_cache = {}, {}
    if cache_npz.exists():
        z = np.load(cache_npz)
        have = {k[2:] for k in z.files if k.startswith("a_")}
        for c in train_ids:
            if c in have:
                fa_cache[c], fb_cache[c] = z[f"a_{c}"], z[f"b_{c}"]
        # keep entries for clips not currently in training too — a later freeze may want them
        for c in have - set(train_ids):
            fa_cache[c], fb_cache[c] = z[f"a_{c}"], z[f"b_{c}"]
    missing = [c for c in train_ids if c not in fa_cache]
    print(f"[plan] endpoint-frame cache: {len(train_ids) - len(missing)} hit, {len(missing)} to add")
    if missing:
        t0 = time.time()
        for n, c in enumerate(missing, 1):
            arr = videoio.read_clip(Path(mp4[c]))     # absolute path from the content pool
            assert arr.shape == (T, H, W, 3), f"{c}: unexpected shape {arr.shape}"
            fa_cache[c] = arr[K - 1].copy()      # last pure-A frame  (index 8)
            fb_cache[c] = arr[T - K].copy()      # first pure-B frame (index 112)
            if n % 50 == 0:
                print(f"[plan]   cached {n}/{len(missing)} ({time.time() - t0:.0f}s)", flush=True)
        cache_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_npz, **{f"a_{c}": v for c, v in fa_cache.items()},
                            **{f"b_{c}": v for c, v in fb_cache.items()})
        print(f"[plan] endpoint-frame cache: added {len(missing)} in {time.time()-t0:.0f}s "
              f"({len(fa_cache)} total on disk)")

    # ---- 4. operator roster, gate-2 pre-validated ----------------------------------------
    runner = GLRunner(W, H)
    print(f"[plan] GL: {runner.renderer_name()}")
    raw = shaders.load_bank(Path(cfg["model"]["shader_bank"]))
    gated, _ = operators.validate_bank(runner, raw, tol=smp["endpoint_tol"])
    bank_sh = {k: v for k, v in gated.items() if k in set(trainable)}
    assert set(bank_sh) == set(trainable), \
        f"gate-1 lost trainable shaders: {sorted(set(trainable) - set(bank_sh))}"
    easings = [e for e in sr.d2_easings() if e in policy["keep_easings"]]
    assert sorted(easings) == sorted(policy["keep_easings"]), "easing set drifted from the policy"

    n_probe, tol = s2["probe_pairs"], smp["endpoint_tol_operator"]
    ops, rejected = [], {"gate2": 0, "tries": 0, "dup_id": 0}
    seen_ids: set[str] = set()
    for shader in trainable:
        made = 0
        while made < quota[shader]:
            op = sr.make_operator(bank_sh, rng, shader, easings=easings,
                                  p_flip=smp["p_flip"], p_swap=smp["p_swap"],
                                  p_vary=smp["p_vary"], param_filter=None)
            rejected["tries"] += 1
            probe = rng.sample(pairs, n_probe)
            res = [operators.check_operator(runner, bank_sh, op, fa_cache[a], fb_cache[b])
                   for a, b in probe]
            if not all(max(d0, d1) <= tol for d0, d1 in res):
                rejected["gate2"] += 1
                if rejected["tries"] > n_ops * s2["max_op_draws_factor"]:
                    raise RuntimeError("operator pre-validation is rejecting almost everything")
                continue
            timing = sr.sample_timing(rng, T, K, frac=smp["timing_frac"])
            i0, j0 = sr.phase_indices(T, timing["onset"], timing["release"])
            # exp_077's _op_id hashes (shader, params, easing, flip, swap) only — it predates
            # timing being part of operator identity. Under the S2 definition an EXACT OP is
            # (shader, uniforms, easing, ONSET/RELEASE, flip, swap), so the id must cover timing
            # too; without it, parameterless shaders (fade, dissolve, swap, ...) and any two
            # draws that happen to land on the same uniforms collapse onto one id. Measured on
            # the first roster build: 800 ops -> only 769 unique ids, 31 collisions.
            op_id = s2_op_id(shader, op.params, op.easing, op.flip, op.swap, timing)
            if op_id in seen_ids:
                rejected["dup_id"] += 1
                continue
            seen_ids.add(op_id)
            ops.append({
                "op_id": op_id, "shader": shader, "params": op.params, "easing": op.easing,
                "flip": op.flip, "swap": op.swap, "extension": "none", "aux_kind": None,
                "timing": {k: timing[k] for k in ("onset", "release", "duration", "u1", "u2")},
                "phase": {"i0": i0, "j0": j0},
                "probe_gate2_max": round(max(max(d) for d in res), 4),
            })
            made += 1
        print(f"[plan]   {shader:<26} {made:2d} ops", flush=True)

    assert len({o["op_id"] for o in ops}) == len(ops) == n_ops, \
        f"op roster is not {n_ops} unique ops (got {len(ops)}, unique {len({o['op_id'] for o in ops})})"
    print(f"[plan] roster: {n_ops} ops pre-validated | gate-2 rejected {rejected['gate2']} "
          f"of {rejected['tries']} draws ({rejected['gate2']/rejected['tries']:.1%}) | "
          f"duplicate-id redraws {rejected['dup_id']}")

    # ---- 5. bipartite grid: 10 primary pairs per op + ordered spares ---------------------
    # Greedy, usage-balanced: an op takes the least-used pairs that keep its endpoints disjoint.
    m, n_cand = s2["contents_per_op"], s2["candidates_per_op"]
    pdeg = {k: 0 for k in pair_key}
    order = list(range(n_ops))
    rng.shuffle(order)
    grid: dict[int, list[int]] = {}
    for oi in order:
        used_clips: set[str] = set()
        chosen: list[int] = []
        for pi in sorted(range(len(pairs)), key=lambda i: (pdeg[pair_key[i]], rng.random())):
            a, b = pairs[pi]
            if a in used_clips or b in used_clips:
                continue
            chosen.append(pi)
            used_clips |= {a, b}
            if len(chosen) == m:
                break
        assert len(chosen) == m, f"op {oi}: only {len(chosen)} endpoint-disjoint pairs available"
        for pi in chosen:
            pdeg[pair_key[pi]] += 1
        # ordered spares for the render-time pair-swap retry (usage-balanced, disjointness is
        # re-checked by the renderer against whatever it has actually ACCEPTED)
        spares = [pi for pi in sorted(range(len(pairs)),
                                      key=lambda i: (pdeg[pair_key[i]], rng.random()))
                  if pi not in set(chosen)][: n_cand - m]
        grid[oi] = chosen + spares

    deg_vals = list(pdeg.values())
    print(f"[plan] grid: ops-per-pair min/median/max = "
          f"{min(deg_vals)}/{int(np.median(deg_vals))}/{max(deg_vals)} (target ~{n_ops*m//len(pairs)})")

    # ---- 6. freeze ------------------------------------------------------------------------
    plan = {
        "created": "2026-07-25",
        "authority": "fable-advisor rulings 1,1a,1b,1c,2,3,5 + riders (ctt_v2 S2/S3, rounds 1-2)",
        "stratum": "S2",
        "contract": {
            "num_frames": T, "height": H, "width": W, "fps": cfg["inference"]["fps"],
            "anchor_frames": K, "extension": "none", "aux_kind": None,
            "streams": "both source clips play in lockstep t=0..120; pure phases verbatim",
        },
        "design": {
            "n_ops": n_ops, "contents_per_op": m, "n_clips": n_ops * m,
            "n_content_pairs": len(pairs), "target_ops_per_pair": round(n_ops * m / len(pairs), 2),
            "exact_op_definition": "(shader, uniforms, easing, onset/release, flip, swap) — ALL "
                                   "frozen across the op's 10 clips, timing included",
            "reference_pairing": "DYNAMIC at train time: ref != target drawn from the SAME op_id's "
                                 "10 clips. No dedicated reference render exists. Endpoint-"
                                 "disjointness across the 10 makes every (ref,target) draw "
                                 "content-disjoint by construction.",
            "retry_policy": "content-pair SWAP only; timing is NEVER redrawn (it is part of op "
                            f"identity). Budget {s2['max_pair_attempts']} pair attempts to fill "
                            f"{m} slots, else the op is DROPPED and logged.",
        },
        "holdout": {
            "shaders": HOLDOUT_S2,
            "note": "eval only, never trained. Each holdout retains a same-genre cousin in "
                    "training so the probe tests unseen-operator generalisation, not removed "
                    "capability. The ffmpeg-xfade family is the separate cross-engine probe.",
        },
        "content": {
            "pool": "207 training endpoints (data/processed/ctt_v2_strata/ENDPOINT_SPLIT.json)",
            "reserved_eval_only": split["reserved_eval_only"],
            "pair_max_cos": s2["pair_max_cos"],
            "clip_degree": {"min": min(cdeg.values()), "max": max(cdeg.values())},
        },
        "gate": cfg["gate"],
        "seed": cfg["runtime"]["seed"],
        "pairs": [{"pair_id": i, "A": a, "B": b, "A_mp4": mp4[a], "B_mp4": mp4[b],
                   "cos": round(float(S[idx[a], idx[b]]), 4), "ops_planned": pdeg[pair_key[i]]}
                  for i, (a, b) in enumerate(pairs)],
        "ops": [{**o, "candidates": grid[i]} for i, o in enumerate(ops)],
    }
    (HERE / "PLAN_S2.json").write_text(json.dumps(plan, indent=1))
    (HERE / "HOLDOUT_S2.json").write_text(json.dumps(
        {"stratum": "S2", "created": "2026-07-25",
         "authority": "fable-advisor ruling 5",
         "principle": "every held-out shader keeps a same-genre cousin in training",
         "holdout_shaders": HOLDOUT_S2,
         "cross_engine_probe": "ffmpeg-xfade family (not in the gl-transitions bank)",
         "trainable_shaders": trainable,
         "reserved_eval_only_endpoints": split["reserved_eval_only"]}, indent=1))
    print(f"[plan] FROZEN -> PLAN_S2.json ({n_ops} ops x {m} = {n_ops*m} clips) "
          f"in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
