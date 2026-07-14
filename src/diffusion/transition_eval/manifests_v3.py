"""The three-manifest model (SPEC §2): eval manifest (per-generation run facts),
corpus manifest (per-clip truth), training manifest (per-adapter truth) — and
the derivations that keep facts stored exactly once.

Derived here, never stored in the eval manifest:
  sidedness, tags  — corpus lookup via the item's style
  tier A/B/C       — join of the item's clips against the training manifest
"""

from __future__ import annotations

import dataclasses
import json
import pathlib


@dataclasses.dataclass
class Condition:
    video: str
    num_frames: int


@dataclasses.dataclass
class EvalItemV3:
    """One generated video to score. Unknown keys reject loudly (SPEC §2)."""
    item_id: str
    generated_video: str
    reference_video: str            # v3: explicit — metrics are reference-centric
    style: str
    n_endpoints: int = 2
    condition_prefix: Condition | None = None
    condition_suffix: Condition | None = None
    arm: str = ""
    twin_of: str | None = None      # base<->ic pairing for paired stats
    notes: str = ""


def load_eval_manifest(path: str | pathlib.Path) -> list[EvalItemV3]:
    items = []
    for raw in json.loads(pathlib.Path(path).read_text()):
        for key in ("condition_prefix", "condition_suffix"):
            if raw.get(key):
                raw[key] = Condition(**raw[key])
        items.append(EvalItemV3(**raw))   # TypeError on unknown keys, by design
    ids = [i.item_id for i in items]
    dupes = {x for x in ids if ids.count(x) > 1}
    if dupes:
        raise ValueError(f"duplicate item_ids in eval manifest: {sorted(dupes)}")
    return items


def load_corpus_manifest(path: str | pathlib.Path) -> dict:
    m = json.loads(pathlib.Path(path).read_text())
    for req in ("classes", "clips", "std_contract"):
        if req not in m:
            raise ValueError(f"corpus manifest missing '{req}'")
    return m


def load_training_manifest(path: str | pathlib.Path) -> dict:
    """Minimal per-adapter schema: {adapter_id, base_model, clips: [corpus clip
    keys], pairs: [{target, reference, conditioning}]}. Only `clips` is required
    for tier derivation."""
    m = json.loads(pathlib.Path(path).read_text())
    if "clips" not in m:
        raise ValueError("training manifest missing 'clips'")
    m["_clipset"] = set(m["clips"])
    return m


# --- derivations ---------------------------------------------------------------

def sidedness_of(style: str, corpus: dict) -> str:
    cls = corpus["classes"].get(style)
    if cls is None or cls.get("sidedness") not in ("onesided", "twosided"):
        raise ValueError(f"style '{style}' has no sidedness in the corpus manifest")
    return cls["sidedness"]


def tags_of(style: str, corpus: dict) -> list[str]:
    return corpus["classes"].get(style, {}).get("tags", [])


def clip_key(video_path: str, corpus: dict) -> str | None:
    """Map an absolute/relative video path onto a corpus clip key (class/file)."""
    p = pathlib.Path(video_path)
    key = f"{p.parent.name}/{p.name}"
    return key if key in corpus["clips"] else None


def derive_tier(item: EvalItemV3, corpus: dict, training: dict | None) -> str | None:
    """A: the reference's class has zero clips in training.
    B: class trained, but none of THIS item's clips (reference + endpoint
       conditions) were — the honest unseen-clips cell.
    C: at least one of this item's clips was in training.
    None: no training manifest supplied (tier columns then read '—')."""
    if training is None:
        return None
    trained = training["_clipset"]
    cls = item.style
    class_trained = any(k.startswith(cls + "/") for k in trained)
    if not class_trained:
        return "A"
    item_clips = set()
    for vid in (item.reference_video,
                item.condition_prefix.video if item.condition_prefix else None,
                item.condition_suffix.video if item.condition_suffix else None):
        if vid:
            k = clip_key(vid, corpus)
            if k:
                item_clips.add(k)
    return "C" if item_clips & trained else "B"


# --- suite completeness ----------------------------------------------------------

def completeness(suite_items: list[dict], scored_ids: set[str]) -> dict:
    """score() must verify the plan was fulfilled — a partial suite is stamped
    partial, never silently scored as if complete (SPEC §4)."""
    wanted = {s["item_id"] for s in suite_items}
    missing = sorted(wanted - scored_ids)
    extra = sorted(scored_ids - wanted)
    return {"complete": not missing, "n_planned": len(wanted),
            "n_scored": len(scored_ids), "missing": missing, "extra": extra}
