"""eval_ladder v2.1.0 — derive the two CLEAN baselines for every treatment task.

Owner call 2026-07-23: base+reference is a COPIER, not a baseline (it reproduces the demo).
The honest baselines for a specialist/generalist task are:

    base_prompt   prompt only — the model is not given the endpoint
    base_cond     prompt + endpoint conditioning (sidedness-aware), NO reference

Two different units, deliberately:

  * SCORING is per task — (donor_class, endpoint, sided) — because the pool-%% must be taken
    against THAT task's donor pool. One registry row per missing tier per task.
  * VIDEO is per (endpoint, sided) — the baseline arms never see the donor, so their output is
    content-identical across donors. Rows share one canonical video through
    `video_key = "<dir>/<name>"`; run_gen/run_eval/viewer all resolve the same way, and
    run_gen's skip-if-exists makes the sharing free (194 new videos, not 708).

Reuse before generation:

  * a task whose (endpoint, sided, prompt) already has a NO-REFERENCE `base` row:
      - same donor  -> that row IS the prompt+endpoint baseline, scored against the right pool:
                       no base_cond row at all (the viewer slots the twin in directly);
      - other donor -> base_cond row that POINTS AT the twin's video (video_key = "base/<item>")
                       and only re-scores it against this task's donor pool. No generation.
  * only (endpoint, sided) units with no such twin get a new base_cond video (23 units).
  * base_prompt has no existing equivalent (text_floor covers 0 treatment endpoints — checked,
    not assumed) -> every unit generates (74 units).

    python eval_ladder/add_baselines.py            # dry run: counts + plan
    python eval_ladder/add_baselines.py --write    # REPLACE baseline rows in registry.jsonl

--write is idempotent and replace-only: lines whose arm is not a baseline arm are preserved
byte-for-byte (asserted == 444), old baseline rows are dropped, the fresh set is appended.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))
import prompts  # noqa: E402

REGISTRY = HERE / "registry.jsonl"
GENS = REPO_ROOT / "outputs/videos/ladder2"
BASELINE_ARMS = ("base_prompt", "base_cond")
SEEDS = (42, 43)
FROZEN_ROWS = 444          # the pre-registered v2.0.0 registry, never touched


def load_lines() -> list[str]:
    return [x for x in REGISTRY.read_text().splitlines() if x.strip()]


def _row(t: dict, arm: str, video_key: str) -> dict:
    """One scoring row. Everything scientific (prompt, %%-type, GT pool) comes from the task's
    representative treatment row `t` — same renderer, same pool, same firewall."""
    sided = t["sided"]
    cond = "none" if arm == "base_prompt" else ("two" if sided == "two" else "prefix")
    item_id = f"{arm}__{t['donor_class']}__{t['endpoint']}"
    return {
        "item_id": item_id,
        "mismatched_reference": False,
        "cell": "BL-prompt" if arm == "base_prompt" else "BL-cond",
        "priority": "P0",
        "arm": arm,
        "ref_novelty": "none",
        "content": t["content"],
        "donor_class": t["donor_class"],
        "endpoint": t["endpoint"],
        "endpoint_class": t["endpoint_class"],
        "endpoint_split": t["endpoint_split"],
        "sided": sided,
        "reference": None,
        "reference_split": None,
        "prompt": t["prompt"],
        "pct_type": t["pct_type"],
        "gt_pool_class": t["gt_pool_class"],
        "conditioning": cond,
        "input_key": item_id,          # baselines are never twinned; unique key
        "video_key": video_key,        # "<dir>/<name>": the shared canonical video
    }


def baseline_rows(rows: list[dict]) -> list[dict]:
    tasks: dict[tuple, dict] = {}
    for r in rows:
        if r["arm"].startswith("spec_") or r["arm"] == "ic_gen":
            tasks.setdefault((r["donor_class"], r["endpoint"], r["sided"]), r)

    # the clean no-reference base twins, keyed by the full input identity (donor-free)
    noref: dict[tuple, dict] = {}
    for r in rows:
        if r["arm"] == "base" and not r.get("reference"):
            noref[(r["endpoint"], r["sided"], r["prompt"])] = r

    out = []
    for (donor, endpoint, sided), t in sorted(tasks.items()):
        out.append(_row(t, "base_prompt", f"base_prompt/BLP__{endpoint}__{sided}"))
        tw = noref.get((endpoint, sided, t["prompt"]))
        if tw is None:
            out.append(_row(t, "base_cond", f"base_cond/BLC__{endpoint}__{sided}"))
        elif tw["donor_class"] != donor:
            out.append(_row(t, "base_cond", f"base/{tw['item_id']}"))
        # else: the task's own no-ref twin IS this baseline, already scored vs this donor pool

    # ---- invariants ----
    ids = [r["item_id"] for r in out]
    assert len(ids) == len(set(ids)), "duplicate baseline item_id"
    for r in out:
        assert (r["conditioning"] == "none") == (r["arm"] == "base_prompt")
        d, _, name = r["video_key"].partition("/")
        assert d in ("base_prompt", "base_cond", "base") and name, r["video_key"]
        if d == "base":  # reused video must actually exist, both seeds
            for s in SEEDS:
                p = GENS / d / f"{name}__s{s}.mp4"
                assert p.exists(), f"{r['item_id']}: reused video missing {p}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    lines = load_lines()
    rows = [json.loads(x) for x in lines]
    kept = [l for l, r in zip(lines, rows) if r["arm"] not in BASELINE_ARMS]
    assert len(kept) == FROZEN_ROWS, f"frozen registry drifted: {len(kept)} != {FROZEN_ROWS}"

    new = baseline_rows([json.loads(l) for l in kept])
    to_gen = {r["video_key"] for r in new if not r["video_key"].startswith("base/")}
    reused = sum(1 for r in new if r["video_key"].startswith("base/"))
    n_bp = sum(1 for r in new if r["arm"] == "base_prompt")
    n_bc = len(new) - n_bp
    print(f"[baselines] {len(new)} scoring rows: base_prompt {n_bp}, base_cond {n_bc} "
          f"({reused} reuse an existing base-twin video)")
    print(f"[baselines] canonical videos to generate: {len(to_gen)} x {len(SEEDS)} seeds "
          f"= {len(to_gen) * len(SEEDS)}")
    if args.write:
        dropped = len(lines) - len(kept)
        REGISTRY.write_text("\n".join(kept + [json.dumps(r) for r in new]) + "\n")
        print(f"[baselines] wrote {REGISTRY.name}: {len(kept)} frozen + {len(new)} baseline rows "
              f"(replaced {dropped} old)")
    else:
        print("[baselines] dry run — pass --write to replace baseline rows in the registry")


if __name__ == "__main__":
    main()
