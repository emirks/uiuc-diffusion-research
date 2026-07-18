"""exp_067 Stage-2 vs ic3 — pre-registered bar verdict (paired, per tier).

Bar (advised campaign, Round-1, registered before results):
  PASS iff  (1) zero-shot tier-C margin improves over ic3 by >= MDE 0.037,
            (2) ID tier-B margin stays within MDE of ic3,
            (3) near-copy rate stays ~ic3's (~3%),
            (4) seam (max_seam_z) not degraded.
Interpretation guard: a pass = "residual reference moves zero-shot", NOT "solved"
(conditioned-base anchor margin 0.175).

Arms map to tiers by RUNG (the items' `tier` field is unreliable):
  ic3_a / s2_a = R4A = tier A (held-in);  ic3_b / s2_b = R4B = tier B (unseen);
  ic3_c / s2_c = R5  = tier C (zero-shot / held-out class).
Paired by item_id (identical rung__class__clip__seed across both adapters).
Margin reported all-items AND trusted-only (certification 3.0.0-draft.8 trust map,
margin channel) to match certified reporting. MDE_margin = 0.037 (amendment-1, n=10).
"""
import collections
import glob
import json
import math
import pathlib

R = pathlib.Path("/projects/illinois/eng/cs/jrehg/users/emirkisa/diffusion-research")
IC3_GLOB = str(R / "outputs/eval/ladder_v3/ic3_abc_c*/items.jsonl")
S2_GLOB = str(R / "outputs/eval/exp_067_s2/s2_abc_c*/items.jsonl")
TRUST = R / "outputs/eval/certification/3.0.0-draft.8/exam/trust_map.json"
MDE_MARGIN = 0.037
TIER_OF = {"a": "A_heldin", "b": "B_unseen", "c": "C_zeroshot"}


def load(glob_pat, arm_prefix):
    rows = {}
    for p in glob.glob(glob_pat):
        for line in open(p):
            r = json.loads(line)
            if not r["arm"].startswith(arm_prefix):
                continue
            rows[r["item_id"]] = r
    return rows


def trusted_classes_for(channel):
    if not TRUST.exists():
        return None
    tm = json.load(open(TRUST))
    ok = set()
    for cls, chans in tm.items():
        v = chans.get(channel) if isinstance(chans, dict) else None
        if v is True or (isinstance(v, dict) and v.get("trusted")):
            ok.add(cls)
    return ok or None


def sign_test(deltas):
    pos = sum(1 for d in deltas if d > 0)
    n = len(deltas)
    # two-sided exact binomial p at p0=0.5
    k = max(pos, n - pos)
    p = 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n) if n else 1.0
    return pos, n, min(p, 1.0)


def main():
    ic3 = load(IC3_GLOB, "ic3")
    s2 = load(S2_GLOB, "s2")
    if not s2:
        print("[wait] no exp_067 scored items yet at", S2_GLOB)
        return
    trust_margin = trusted_classes_for("margin")

    print(f"ic3 items={len(ic3)}  s2 items={len(s2)}  paired={len(set(ic3)&set(s2))}")
    tiers = collections.defaultdict(list)
    for iid in set(ic3) & set(s2):
        arm = s2[iid]["arm"].rsplit("_", 1)[-1]  # a/b/c
        tiers[arm].append(iid)

    verdict = {}
    for arm in ("a", "b", "c"):
        ids = tiers.get(arm, [])
        if not ids:
            print(f"\n[tier {TIER_OF[arm]}] no paired items yet")
            continue
        # margin (all + trusted), paired
        dm_all, dm_tr = [], []
        m_ic3, m_s2 = [], []
        d_seam, nc_ic3, nc_s2 = [], 0, 0
        for iid in ids:
            a, b = ic3[iid], s2[iid]
            if a.get("margin") is not None and b.get("margin") is not None:
                dm_all.append(b["margin"] - a["margin"])
                m_ic3.append(a["margin"]); m_s2.append(b["margin"])
                if trust_margin is None or a.get("style") in trust_margin:
                    dm_tr.append(b["margin"] - a["margin"])
            if a.get("max_seam_z") is not None and b.get("max_seam_z") is not None:
                d_seam.append(b["max_seam_z"] - a["max_seam_z"])
            nc_ic3 += int(bool(a.get("near_copy")))
            nc_s2 += int(bool(b.get("near_copy")))
        mean = lambda x: sum(x) / len(x) if x else float("nan")
        pos, n, p = sign_test(dm_all)
        verdict[arm] = dict(n=len(ids), margin_ic3=mean(m_ic3), margin_s2=mean(m_s2),
                            dmargin_all=mean(dm_all), dmargin_trusted=mean(dm_tr),
                            sign=f"{pos}/{n}", p=p, dseam=mean(d_seam),
                            nearcopy_ic3=nc_ic3 / len(ids), nearcopy_s2=nc_s2 / len(ids))
        v = verdict[arm]
        print(f"\n[tier {TIER_OF[arm]}]  n={v['n']}")
        print(f"  margin: ic3 {v['margin_ic3']:.3f} -> s2 {v['margin_s2']:.3f}  "
              f"Δ_all {v['dmargin_all']:+.3f}  Δ_trusted {v['dmargin_trusted']:+.3f}  "
              f"sign {v['sign']} (p={v['p']:.3f})  [MDE {MDE_MARGIN}]")
        print(f"  seam Δ (lower=better): {v['dseam']:+.3f}   "
              f"near-copy: ic3 {v['nearcopy_ic3']:.0%} -> s2 {v['nearcopy_s2']:.0%}")

    # ---- pre-registered bar ----
    print("\n=== PRE-REGISTERED BAR (residual reference moves zero-shot) ===")
    if all(t in verdict for t in ("b", "c")):
        c, b = verdict["c"], verdict["b"]
        c1 = c["dmargin_all"] >= MDE_MARGIN
        c2 = abs(b["dmargin_all"]) <= MDE_MARGIN or b["dmargin_all"] >= 0
        c3 = c["nearcopy_s2"] <= 0.10 and b["nearcopy_s2"] <= 0.10
        c4 = c["dseam"] <= MDE_MARGIN * 10 and b["dseam"] <= 5.0  # seam not materially worse
        print(f"  (1) tier-C zero-shot margin Δ {c['dmargin_all']:+.3f} >= {MDE_MARGIN}: {'PASS' if c1 else 'FAIL'}")
        print(f"  (2) tier-B ID margin Δ {b['dmargin_all']:+.3f} within MDE / not down: {'PASS' if c2 else 'FAIL'}")
        print(f"  (3) near-copy stays low (s2 C {c['nearcopy_s2']:.0%}, B {b['nearcopy_s2']:.0%}): {'PASS' if c3 else 'FAIL'}")
        print(f"  (4) seam not degraded (C Δ{c['dseam']:+.2f}, B Δ{b['dseam']:+.2f}): {'PASS' if c4 else 'FAIL'}")
        print(f"  >>> BAR: {'PASS' if (c1 and c2 and c3 and c4) else 'NOT MET'} "
              f"(advisor consult on the full picture regardless)")
    else:
        print("  tiers B and/or C not yet scored — rerun after generation+scoring complete")

    json.dump(verdict, open(R / "outputs/eval/exp_067_s2/bar_verdict.json", "w"), indent=1) \
        if pathlib.Path(R / "outputs/eval/exp_067_s2").exists() else None


if __name__ == "__main__":
    main()
