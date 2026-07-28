"""Draw the BAD-rate audit sample for an exp_083 run.

The sample is drawn from a fixed seed BEFORE anything is looked at, so it cannot
be cherry-picked. It writes `audit.json` with the sampled stems and an empty
`bad` map for the operator to fill in while eyeballing the filmstrips, then
`summarize.py <run_dir> audit.json` folds the verdict into PILOT_RESULT.json.

Usage:  python sample_audit.py <run_dir> [n=30] [seed=83]
"""

from __future__ import annotations

import json
import pathlib
import random
import sys


def main() -> None:
    run_dir = pathlib.Path(sys.argv[1]).resolve()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 83
    man = json.load(open(run_dir / "manifest.json"))
    stems = sorted(m["stem"] for m in man)
    sample = sorted(random.Random(seed).sample(stems, min(n, len(stems))))
    out = run_dir / "audit.json"
    json.dump({
        "protocol": f"uniform random sample of {len(sample)} of {len(stems)} clips, "
                    f"seed {seed}, drawn before any clip was viewed; each judged from "
                    f"its 9-tile filmstrip (both anchors, both seams, 5 middle frames). "
                    f"BAD = would not ship as a training tuple.",
        "seed": seed, "n_total": len(stems),
        "sampled": sample, "bad": {}, "reason_class": {},
    }, open(out, "w"), indent=1)
    print(f"[audit] {out} — {len(sample)} clips")
    for s in sample:
        print(" ", s)


if __name__ == "__main__":
    main()
