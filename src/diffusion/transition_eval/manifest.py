"""Eval manifest — condition-count aware from day one.

An item is one generated video to score against one reference style. Items
with n_endpoints=1 (portal-style) drop b(t), suffix fidelity, and arrival
timing; the task contract is weaker there (outcome underdetermined) and the
report says so rather than hiding it.
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
class EvalItem:
    item_id: str
    generated_video: str
    style: str                      # reference style key, e.g. "shadow_smoke"
    n_endpoints: int = 2
    condition_prefix: Condition | None = None
    condition_suffix: Condition | None = None
    arm: str = ""                   # method/model label used for aggregation
    notes: str = ""


def load_manifest(path: pathlib.Path) -> list[EvalItem]:
    items = []
    for raw in json.loads(pathlib.Path(path).read_text()):
        for key in ("condition_prefix", "condition_suffix"):
            if raw.get(key):
                raw[key] = Condition(**raw[key])
        items.append(EvalItem(**raw))
    return items


def save_manifest(items: list[EvalItem], path: pathlib.Path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([dataclasses.asdict(i) for i in items], indent=2))


def scan_references(root: pathlib.Path, exclude: tuple[str, ...] = ("higgsfield",)) -> dict[str, list[pathlib.Path]]:
    """data/processed/transitions layout: one subdir per style, mp4s inside."""
    refs = {}
    for d in sorted(pathlib.Path(root).iterdir()):
        if d.is_dir() and d.name not in exclude:
            vids = sorted(d.glob("*.mp4"))
            if vids:
                refs[d.name] = vids
    return refs
