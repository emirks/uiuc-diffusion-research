"""Shared helpers for the collapse-to-the-endpoint-line probe (CPU only).

One place for the row shape, the item-id scheme, the file locations and the prompt-rendering
proof, so build_r1 / splice_r1 / preflight / score all agree by construction.

Contract facts (verified against eval_ladder/run_gen.py, 2026-09-08):
  * build_sample() emits `ValidationSample(prompt=row["prompt"], conditions=conds)` — the row's
    `prompt` field is used VERBATIM, nothing is stripped or re-rendered at generation time.
    (The base_cond_* "prompt rules" live in build_registry.py, which builds the FROZEN registry;
    an --extra-registry row carries its own final prompt.)  => set row["prompt"] to the exact
    text we want and it renders exactly.
  * conditioning is a pure function of the row: `conditioning != "none"` attaches the prefix
    window conds/<endpoint>_start9.mp4 (9 px consumed); `sided == "two"` additionally attaches
    conds/<endpoint>_end9.mp4 (9 px cut, 8 consumed = SUFFIX_GEN_FRAMES).
  * out_path for a row WITHOUT `video_key` is <out-root>/<arm>/<item_id>__s<seed>.mp4.
  * `use_reference: false` keeps any reference off the model; we omit `reference` entirely
    (we never run run_eval on these, so the pool-identity field is not needed).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EVAL_LADDER = REPO_ROOT / "eval_ladder"
CONDS = EVAL_LADDER / "conds"

ARM = "base_cond_neutral"  # a base (no-adapter) arm; prompt is taken verbatim from the row
SEEDS = (42, 43, 44)

REAL_PROMPTS = HERE / "prompts" / "prompts.jsonl"
EXAMPLE_PROMPTS = HERE / "prompts" / "_example.jsonl"

REG = HERE / "reg"
OUT = HERE / "out"


# ---------------------------------------------------------------- prompt file
def default_prompts() -> Path:
    """Real prompts if present, else the 2-row example (dev)."""
    return REAL_PROMPTS if REAL_PROMPTS.exists() else EXAMPLE_PROMPTS


def load_prompts(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    ids = [r["prompt_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate prompt_id in {path}")
    for r in rows:
        for k in ("prompt_id", "endpoint", "tier", "full_prompt", "neutral_prompt"):
            if k not in r:
                raise ValueError(f"{r.get('prompt_id','?')}: missing field {k!r} in {path}")
    return rows


# ---------------------------------------------------------------- item ids / paths
def r1_item_id(p: dict) -> str:
    return f"probe__{p['prompt_id']}__{p['endpoint']}"


def r2_item_id(p: dict) -> str:
    return f"probe_r2__{p['prompt_id']}"


def r3_item_id(p: dict) -> str:
    return f"probe_r3__{p['prompt_id']}"


def probe_clip(prompt_id: str, seed: int) -> str:
    """The spliced clip id whose windows R2/R3 read (unique per prompt x seed)."""
    return f"probe_{prompt_id}_s{seed}"


def out_mp4(run: str, item_id: str, seed: int, out_root: Path | None = None) -> Path:
    """<out-root>/<arm>/<item_id>__s<seed>.mp4 — matches run_gen.out_path for a no-video_key row."""
    root = out_root or (OUT / run.lower())
    return root / ARM / f"{item_id}__s{seed}.mp4"


# ---------------------------------------------------------------- row builders
def build_r1_row(p: dict) -> dict:
    """One R1 registry row: start anchor only, full prompt, sided one."""
    return {
        "arm": ARM,
        "item_id": r1_item_id(p),
        "priority": "P0",
        "cell": "probe",
        "endpoint": p["endpoint"],
        "sided": "one",
        "conditioning": "prefix",       # != "none" => prefix window attached; 9 px consumed
        "prompt": p["full_prompt"],     # rendered VERBATIM by build_sample
        "use_reference": False,
        "no_twin": True,
        # ---- carried provenance / scoring keys (ignored by run_gen) ----
        "endpoint_class": p.get("endpoint_class"),
        "gt_pool_class": p.get("endpoint_class"),
        "endpoint_source": p.get("endpoint_source"),
        "prompt_id": p["prompt_id"],
        "tier": p["tier"],
        "probe_tier": p["tier"],
        "probe_prompt_variant": "full",
        "probe_run": "R1",
        "real_endpoint": p["endpoint"],
        "start_caption": p.get("start_caption"),
        "change_clause": p.get("change_clause"),
        "end_caption": p.get("end_caption"),
        "full_prompt": p["full_prompt"],
        "neutral_prompt": p["neutral_prompt"],
        "mechanism_family": p.get("mechanism_family"),
    }


def build_r2r3_row(p: dict, seed: int, run: str) -> dict:
    """R2 (full prompt) / R3 (neutral prompt): start + spliced-end anchors, sided two, same seed."""
    assert run in ("R2", "R3")
    clip = probe_clip(p["prompt_id"], seed)
    prompt = p["full_prompt"] if run == "R2" else p["neutral_prompt"]
    variant = "full" if run == "R2" else "neutral"
    item_id = r2_item_id(p) if run == "R2" else r3_item_id(p)
    return {
        "arm": ARM,
        "item_id": item_id,
        "priority": "P0",
        "cell": "probe",
        "endpoint": clip,               # windows resolved by NAME only -> probe_<id>_s<seed>_{start9,end9}
        "sided": "two",                 # prefix + suffix (suffix = 8 px consumed)
        "conditioning": "prefix",       # != "none"; suffix added because sided == two
        "prompt": prompt,
        "use_reference": False,
        "no_twin": True,
        "endpoint_class": p.get("endpoint_class"),
        "gt_pool_class": p.get("endpoint_class"),
        "endpoint_source": p.get("endpoint_source"),
        "prompt_id": p["prompt_id"],
        "seed": seed,
        "tier": p["tier"],
        "probe_tier": p["tier"],
        "probe_prompt_variant": variant,
        "probe_run": run,
        "real_endpoint": p["endpoint"],
        "start_caption": p.get("start_caption"),
        "change_clause": p.get("change_clause"),
        "end_caption": p.get("end_caption"),
        "full_prompt": p["full_prompt"],
        "neutral_prompt": p["neutral_prompt"],
        "mechanism_family": p.get("mechanism_family"),
    }


# ---------------------------------------------------------------- generator prompt-render proof
def render_via_generator(rows: list[dict]) -> dict[str, str]:
    """Return {item_id: rendered prompt} by calling the GENERATOR'S OWN build_sample().

    We import eval_ladder/run_gen.py and call its real build_sample, so the rendered prompt is
    exactly what generation will feed the model. build_sample only needs ltx_trainer.config
    (fast) + encode_conditioning + prompts; it never touches peft / model_loader /
    validation_runner, so we stub those three heavy top-level imports (peft alone is ~60 s on
    this login node) to keep the check CPU-cheap. No re-implementation of the prompt path.
    """
    import types

    for name, attrs in (
        ("peft", ["LoraConfig", "get_peft_model", "set_peft_model_state_dict"]),
        ("ltx_trainer.model_loader", ["load_transformer"]),
        ("ltx_trainer.progress", ["TrainingProgress"]),
        ("ltx_trainer.validation_runner", ["ValidationRunner"]),
    ):
        if name not in sys.modules:
            m = types.ModuleType(name)
            for a in attrs:
                setattr(m, a, object)
            sys.modules[name] = m

    sys.path.insert(0, str(EVAL_LADDER))
    import run_gen  # noqa: PLC0415

    bs = run_gen.build_sample
    bs.ref_downscale = 1
    bs.ref_attention = "bidirectional"
    bs.const_code_clip = None
    bs.signal_cfg = None
    out = {}
    for r in rows:
        out[r["item_id"]] = bs(r).prompt
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
