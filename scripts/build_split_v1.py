"""Frozen train/test split v1 for the transitions_std121 corpus (223 clips, 39 classes).

RULE (frozen, metadata-only — metric scores are NEVER used to choose):
  Per class with n_clips >= 8 -> 2 test clips; 4 <= n < 8 -> 1 test clip;
  n < 4 -> all-train, zero test (no per-class claims possible anyway).
  Test clips chosen by seeded RNG:
      random.Random(f"split_v1:{class_name}").sample(sorted(clip_ids), k)
  Everything else train.

PRE-REGISTERED REMEDIATION (near-duplicate audit, applied mechanically):
  A flagged test clip (M2a copy_max >= tau against any train clip of its
  class) is replaced by the next deterministic candidate from the SAME RNG
  stream (next `rng.sample(sorted(remaining), 1)` draw, where `remaining` =
  the class's clips minus current test picks minus all flagged clips); if all
  candidates in a class flag, the class goes all-train.
  Flags live in split_v1_flagged.json ({class: [clip_id, ...]}), written by
  scripts/audit_split_v1.py. Absent/empty file = no flags.

The split is exactly reproducible: corpus_manifest.json + the flagged file +
this script fully determine split_v1.json.

Usage (login node, stdlib only):
    python scripts/build_split_v1.py
"""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import random
from collections import defaultdict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
STD = REPO_ROOT / "data/processed/transitions_std121"
CORPUS = STD / "corpus_manifest.json"
FLAGGED = STD / "split_v1_flagged.json"
OUT = STD / "split_v1.json"

RULE_TEXT = (
    "per class with n_clips >= 8 -> 2 test clips; 4 <= n < 8 -> 1 test clip; "
    "n < 4 -> all-train, zero test (no per-class claims possible anyway). "
    'Test clips chosen by seeded RNG: python random.Random(f"split_v1:{class_name}")'
    ".sample(sorted(clip_ids), k). Everything else train. Remediation: a flagged "
    "test clip is replaced by the next deterministic candidate (same RNG stream, "
    "next sample); if all candidates in a class flag, the class goes all-train."
)


def k_test(n: int) -> int:
    if n >= 8:
        return 2
    if n >= 4:
        return 1
    return 0


def class_clips(corpus: dict) -> dict[str, dict[str, str]]:
    """class -> {clip_id (stem): repo-relative path}. clip_id = filename stem.
    Corpus manifest keys are relative to transitions_std121/."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for rel, meta in corpus["clips"].items():
        out[meta["class"]][pathlib.Path(rel).stem] = (
            f"data/processed/transitions_std121/{rel}")
    return out


def split_class(clip_ids: list[str], class_name: str,
                flagged: set[str]) -> tuple[list[str], list[str]]:
    """Return (test, train) for one class, applying the frozen rule +
    pre-registered remediation. Deterministic given inputs."""
    ids = sorted(clip_ids)
    k = k_test(len(ids))
    if k == 0:
        return [], ids
    rng = random.Random(f"split_v1:{class_name}")
    test = rng.sample(ids, k)          # the frozen initial draw
    final: list[str] = []
    used = set(test)                   # never re-draw a clip already picked
    for pick in test:
        cur: str | None = pick
        while cur is not None and cur in flagged:
            remaining = sorted(set(ids) - used - flagged)
            if not remaining:
                cur = None             # candidates exhausted
                break
            cur = rng.sample(remaining, 1)[0]   # next sample, same RNG stream
            used.add(cur)
        if cur is not None:
            final.append(cur)
    if not final:                      # all candidates flagged -> all-train
        return [], ids
    return final, [c for c in ids if c not in final]


def build_split(corpus: dict, flagged_by_class: dict[str, list[str]]) -> dict:
    by_class = class_clips(corpus)
    split = {}
    for cls in sorted(by_class):
        clips = by_class[cls]
        test, train = split_class(list(clips), cls, set(flagged_by_class.get(cls, [])))
        split[cls] = {
            "n_clips": len(clips),
            "train": train,
            "test": test,
            "paths": {cid: clips[cid] for cid in sorted(clips)},
        }
    return split


def main() -> None:
    corpus_bytes = CORPUS.read_bytes()
    corpus = json.loads(corpus_bytes)
    flagged = json.loads(FLAGGED.read_text())["flagged"] if FLAGGED.exists() else {}
    split = build_split(corpus, flagged)

    n_train = sum(len(v["train"]) for v in split.values())
    n_test = sum(len(v["test"]) for v in split.values())
    out = {
        "split": "v1",
        "provenance": {
            "rule": RULE_TEXT,
            "corpus_manifest": str(CORPUS.relative_to(REPO_ROOT)),
            "corpus_manifest_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
            "flagged_file": str(FLAGGED.relative_to(REPO_ROOT)) if FLAGGED.exists() else None,
            "flagged": flagged,
            "date": datetime.date.today().isoformat(),
            "script": "scripts/build_split_v1.py",
        },
        "n_classes": len(split),
        "n_train": n_train,
        "n_test": n_test,
        "test_fraction": round(n_test / (n_train + n_test), 4),
        "classes": split,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[done] split_v1: {n_train} train / {n_test} test "
          f"({out['test_fraction']:.1%} test) -> {OUT.relative_to(REPO_ROOT)}")
    for cls, v in split.items():
        tag = f" flagged={flagged[cls]}" if cls in flagged else ""
        print(f"  {cls:24s} n={v['n_clips']:2d} test={v['test']}{tag}")


if __name__ == "__main__":
    main()
