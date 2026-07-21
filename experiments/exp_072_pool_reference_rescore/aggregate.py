"""exp_072 — aggregate pool-reference rows into the common-yardstick table.

Per (arm, class): pool score = mean over refs per item, then mean over items;
reported as % of the class GT ceiling (same-class off-diagonal mean of the
certified m1a__v3_sided matrix). Run with the diffusion env python (numpy).
"""
import collections
import glob
import json
import pathlib
import statistics as st

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
NPZ = REPO / ".claude/worktrees/eval-v4-cert/outputs/eval/certification/4.0.0-draft.1/analysis/distance_matrices.npz"
MATRIX = "m1a_S3"  # v4 kernel — owner default 2026-07-20; rows scored by the v4 instrument
TIER = {"r1": "base·PE", "r2_ckpt2000": "spec SEEN(A)", "r3_ckpt2000": "spec UNSEEN(B)",
        "ic3_a": "ic3 held-in(A)", "ic3_b": "ic3 unseen(B)", "ic3_c": "ic3 zero-shot(C)"}
ORDER = ["r1", "r2_ckpt2000", "r3_ckpt2000", "ic3_a", "ic3_b", "ic3_c"]


def ceilings():
    z = np.load(NPZ, allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    S = 1.0 - z[MATRIX]
    cls = [k.split("/")[0] for k in keys]
    byc = collections.defaultdict(list)
    for i, c in enumerate(cls):
        byc[c].append(i)
    return {c: S[np.ix_(i, i)][~np.eye(len(i), dtype=bool)].mean()
            for c, i in ((c, np.array(v)) for c, v in byc.items()) if len(i) >= 2}


def main():
    ceil = ceilings()
    per_item = collections.defaultdict(list)  # (arm, style, source_item) -> app_ref over refs
    for f in glob.glob(str(REPO / "outputs/eval/exp_072_pool_v4/pool_c*/items.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["arm"].startswith("control") or r.get("app_ref") is None:
                continue
            src = r["item_id"].split("__ref_")[0]
            per_item[(r["arm"], r["style"], src)].append(r["app_ref"])
    per_cls = collections.defaultdict(list)  # (arm, style) -> item pool-means
    for (arm, style, _src), vals in per_item.items():
        per_cls[(arm, style)].append(st.mean(vals))
    classes = sorted({s for (_a, s) in per_cls})
    print(f"{'class':18s}" + "".join(f"{TIER[a]:>17s}" for a in ORDER) + f"{'ceiling':>9s}")
    for c in classes:
        cells = []
        for a in ORDER:
            v = per_cls.get((a, c))
            if not v or c not in ceil:
                cells.append(f"{'—':>17s}")
            else:
                cells.append(f"{st.mean(v):>7.3f} ({st.mean(v)/ceil[c]:>4.0%}) ")
        print(f"{c:18s}" + "".join(cells) + (f"{ceil[c]:>9.3f}" if c in ceil else f"{'—':>9s}"))
    print("\noverall achieved-% (mean over classes with data):")
    for a in ORDER:
        fr = [st.mean(v) / ceil[c] for (arm, c), v in per_cls.items() if arm == a and c in ceil]
        n_i = sum(len(v) for (arm, _c), v in per_cls.items() if arm == a)
        if fr:
            print(f"  {TIER[a]:16s} {st.mean(fr):>5.0%}   (classes={len(fr)}, items={n_i})")


if __name__ == "__main__":
    main()
