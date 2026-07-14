"""Block C — REALISM (SPEC §6.3): the full v3 pipeline meets the ~150 archived
exp_056-058 generations before it ever issues a model claim.

Hard content: bar 7 only — the 11 human-verified copy-regime base twins from
exp_057 (leak 0.97-0.995, visibly replaying the reference's scenes; notes/exp/
exp_057 §4a) must flag under v3 (near_copy OR max_seam_z > 3). This is
tau_copy's re-rendered test: tau is SET by Block B's splices and TESTED here;
a miss means a new draft version, never a tau nudge.

Everything else is descriptive — per-arm distributions, flag rates, margins,
and the per-item v2<->v3 bridge (O10 head start). NO bars on model behavior:
a generation that never departs endpoint A SHOULD flag core_degenerate.

Conversion: v2 `manifest_scoring.json` rows are nearly v3-shaped; the v3
`reference_video` is recovered from the v2 `notes` field ("reference=<stem>")
against the corpus manifest. Unconvertible items are EXCLUDED LOUDLY and
enumerated in the certification record — never silently dropped.
"""

from __future__ import annotations

import json
import pathlib
import re

import numpy as np

_REF_RE = re.compile(r"reference=([A-Za-z0-9_]+)")


def _stem_index(corpus: dict) -> dict[str, str]:
    idx = {}
    for key in corpus["clips"]:
        stem = pathlib.Path(key).stem
        if stem in idx:
            raise ValueError(f"corpus clip stem collision: {stem}")
        idx[stem] = key
    return idx


def convert_v2_manifest(manifest_path: pathlib.Path, corpus: dict,
                        main_root: pathlib.Path) -> tuple[list[dict], list[dict]]:
    """v2 scoring manifest -> v3 eval-manifest rows (+ loud exclusions).
    Paths are made absolute against the checkout that owns the archives."""
    rows = json.loads(pathlib.Path(manifest_path).read_text())
    stems = _stem_index(corpus)
    root = pathlib.Path(main_root)
    items, excluded = [], []
    for r in rows:
        m = _REF_RE.search(r.get("notes", ""))
        ref_key = stems.get(m.group(1)) if m else None
        gen = root / r["generated_video"]
        problems = []
        if ref_key is None:
            problems.append(f"reference not recoverable from notes: {r.get('notes')!r}")
        if not gen.exists():
            problems.append(f"generated video missing: {gen}")
        if problems:
            excluded.append({"item_id": r.get("item_id"), "problems": problems})
            continue
        item = {
            "item_id": r["item_id"],
            "generated_video": str(gen),
            "reference_video": str(root / corpus["corpus_root"] / ref_key),
            "style": r["style"],
            "n_endpoints": r.get("n_endpoints", 2),
            "condition_prefix": ({"video": str(root / r["condition_prefix"]["video"]),
                                  "num_frames": r["condition_prefix"]["num_frames"]}
                                 if r.get("condition_prefix") else None),
            "condition_suffix": ({"video": str(root / r["condition_suffix"]["video"]),
                                  "num_frames": r["condition_suffix"]["num_frames"]}
                                 if r.get("condition_suffix") else None),
            "arm": r.get("arm", ""),
            "twin_of": None,   # descriptive block: no capability claims, no twins needed
            "notes": f"blockC archive; {r.get('notes', '')}",
        }
        items.append(item)
    return items, excluded


def copy_twin_ids(exp057_manifest: pathlib.Path) -> list[str]:
    """The 11 human-verified copy-regime items = exp_057's base-arm items
    (notes/exp/exp_057 §4a: 'All 11 base twins are in the copy regime')."""
    rows = json.loads(pathlib.Path(exp057_manifest).read_text())
    ids = sorted(r["item_id"] for r in rows if r.get("arm", "").startswith("base"))
    if len(ids) != 11:
        raise ValueError(f"expected 11 exp_057 base twins, found {len(ids)}: {ids}")
    return ids


def grade_copy_twins(rows_by_id: dict[str, dict], twin_ids: list[str]) -> dict:
    """Bar 7: 11/11 flagged (near_copy OR max_seam_z > 3)."""
    per = {}
    for tid in twin_ids:
        r = rows_by_id.get(tid)
        if r is None:
            per[tid] = {"pass": False, "reason": "not scored"}
            continue
        flagged = bool(r.get("near_copy")) or bool(
            np.isfinite(r.get("max_seam_z", np.nan)) and r["max_seam_z"] > 3.0)
        per[tid] = {"pass": flagged, "near_copy": r.get("near_copy"),
                    "copy_max": r.get("copy_max"), "max_seam_z": r.get("max_seam_z")}
    n_pass = sum(1 for v in per.values() if v["pass"])
    return {"pass": bool(n_pass == len(twin_ids)), "n_pass": n_pass,
            "n_twins": len(twin_ids), "per_twin": per}


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 5:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


BRIDGE_PAIRS = (            # (v2 field, v3 field) — same construct, new estimator
    ("appearance_best", "app_ref"),
    ("leak_max_sim_target", "copy_max"),
    ("motion_fidelity_mean", "obj_match"),
    ("max_seam_z", "max_seam_z"),
    ("prefix_dino", "prefix_dino"),
)


def bridge_v2_v3(v2_items: pathlib.Path, v3_rows: dict[str, dict]) -> dict:
    """Descriptive continuity table: per-item join on item_id, Spearman per
    construct pair. NOT a bar — v2 was retired uncertified; this transfers
    intuition, not authority."""
    v2 = {r["item_id"]: r for r in
          (json.loads(l) for l in pathlib.Path(v2_items).read_text().splitlines() if l.strip())}
    shared = sorted(set(v2) & set(v3_rows))
    out = {"n_shared": len(shared), "pairs": {}}
    for f2, f3 in BRIDGE_PAIRS:
        xs, ys = [], []
        for i in shared:
            a, b = v2[i].get(f2), v3_rows[i].get(f3)
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                xs.append(float(a)); ys.append(float(b))
        out["pairs"][f"{f2}->{f3}"] = {"n": len(xs), "spearman": _spearman(xs, ys)}
    return out


DIST_METRICS = ("app_ref", "margin", "copy_max", "cam_corr", "obj_match",
                "prefix_dino", "max_seam_z")


def arm_distributions(rows: list[dict]) -> dict:
    """Descriptive: mean/std/n per arm x metric + flag rates (saturation and
    degeneracy are read by eyes in the record, not gated)."""
    arms: dict[str, list[dict]] = {}
    for r in rows:
        arms.setdefault(r.get("arm", "?"), []).append(r)
    out = {}
    for arm, rs in sorted(arms.items()):
        entry = {"n": len(rs), "flags": {
            "core_degenerate": float(np.mean([bool(r.get("core_degenerate")) for r in rs])),
            "near_copy": float(np.mean([bool(r.get("near_copy")) for r in rs])),
            "cross_high": float(np.mean([bool(r.get("cross_high")) for r in rs])),
        }}
        for m in DIST_METRICS:
            vals = [r[m] for r in rs if r.get(m) is not None and np.isfinite(r[m])]
            entry[m] = ({"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                         "n": len(vals)} if vals else None)
        out[arm] = entry
    return out
