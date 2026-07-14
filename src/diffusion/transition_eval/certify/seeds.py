"""sigma_seed measurement (SPEC §4/§6, OPEN O6): generation-seed variance,
measured ONCE per model family at certification, consumed as the minimum
detectable effect (MDE) every routine 1-seed suite reports against.

Protocol (bars.yaml `seeds`): a fixed probe subset (stratified across
sidedness x taxonomy), each item generated with n_seeds seeds by the model
under certification, all scored normally. Here: per-metric between-seed std
pooled across probe items -> sigma_seed.json.

MDE for a paired comparison over n items: 1.96 * sigma_seed * sqrt(2 / n).
score.py's report stamps each delta as `> MDE` or `within noise` — the
escalation rule (rerun that comparison at 3 seeds) lives in SPEC §4.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

METRICS = ("app_ref", "margin", "copy_max", "cam_dtw", "obj_match", "max_seam_z")


def sigma_seed(items_jsonl: pathlib.Path, out_path: pathlib.Path | None = None) -> dict:
    """items.jsonl rows must carry `probe_group` (same item, different seed).
    Returns per-metric pooled between-seed std + the per-n MDE table."""
    rows = [json.loads(l) for l in pathlib.Path(items_jsonl).read_text().splitlines() if l.strip()]
    groups: dict[str, list[dict]] = {}
    for r in rows:
        g = r.get("probe_group")
        if g:
            groups.setdefault(g, []).append(r)
    if not groups:
        raise ValueError("no probe_group fields — not a seed-probe scoring run")

    out = {"n_groups": len(groups),
           "seeds_per_group": {g: len(v) for g, v in groups.items()},
           "sigma": {}, "mde": {}}
    for m in METRICS:
        stds = []
        for g, v in groups.items():
            vals = [r.get(m) for r in v]
            vals = [x for x in vals if x is not None and np.isfinite(x)]
            if len(vals) >= 3:
                stds.append(np.std(vals, ddof=1))
        if stds:
            s = float(np.sqrt(np.mean(np.square(stds))))   # pooled
            out["sigma"][m] = s
            out["mde"][m] = {str(n): round(1.96 * s * np.sqrt(2.0 / n), 4)
                             for n in (5, 10, 20, 40)}
    if out_path:
        pathlib.Path(out_path).write_text(json.dumps(out, indent=2))
    return out
