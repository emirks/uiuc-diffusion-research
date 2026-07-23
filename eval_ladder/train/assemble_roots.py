"""ladder2 — assemble the 12 preprocessed_data_roots (pure filesystem, no GPU).

The trainer pairs its data sources by IDENTICAL RELATIVE PATH and silently skips any sample
whose sources disagree (`datasets.py::_build_index_fast` only debug-logs the skip). A silent
drop is exactly the class of defect this campaign exists to kill, so every root is built with
one uniform layout and the file counts across its sources are hard-asserted equal.

    roots/spec_<class>/                       roots/ic_gen/
      latents/<class>/<clip>.pt                 latents/<class>/<tgt>__ref_<ref>.pt
      conditions/<class>/<clip>.pt              conditions/<class>/<tgt>__ref_<ref>.pt
      cond_clean_latents/… (two-sided only)     reference_latents/<class>/<tgt>__ref_<ref>.pt
                                                masks/<class>/<tgt>__ref_<ref>.pt
                                                cond_clean_latents/<class>/<tgt>__ref_<ref>.pt

Symlinks are per-file (never directory symlinks) so globbing is robust. Masks are written, not
linked: mask = f(conditioning) — [0,1]=1 always (prefix), [-1]=1 iff two-sided (suffix).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

import prompts  # noqa: E402

DATASET = HERE.parent / "dataset"
ROOTS = DATASET / "roots"
COND_CLEAN = DATASET / "cond_clean"
CONDITIONS = DATASET / "conditions"
INVENTORY = HERE / "inventory.json"


def link(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def assert_counts(root: Path, sources: list[str]) -> dict[str, int]:
    counts = {s: sum(1 for _ in (root / s).rglob("*.pt")) for s in sources}
    if len(set(counts.values())) != 1:
        raise AssertionError(f"{root.name}: source counts disagree {counts} — samples would be "
                             f"SILENTLY DROPPED by the trainer")
    return counts


def assemble_specialist(name: str, cls: str, clips: list[str], two_sided: bool, inv: dict) -> None:
    root = ROOTS / name
    for clip in clips:
        rel = f"{cls}/{clip}.pt"
        link(REPO_ROOT / inv["latents"][clip], root / "latents" / rel)
        link(CONDITIONS / rel, root / "conditions" / rel)
        if two_sided:
            link(COND_CLEAN / rel, root / "cond_clean_latents" / rel)
    sources = ["latents", "conditions"] + (["cond_clean_latents"] if two_sided else [])
    counts = assert_counts(root, sources)
    print(f"  {name:26s} {counts}")


def assemble_generalist(name: str, pairs: list[dict], inv: dict) -> None:
    import torch

    root = ROOTS / name
    mask_cache: dict[tuple, "torch.Tensor"] = {}
    for p in pairs:
        cls, tgt, ref = p["class"], p["target"], p["reference"]
        rel = f"{cls}/{tgt}__ref_{ref}.pt"
        link(REPO_ROOT / inv["latents"][tgt], root / "latents" / rel)
        link(CONDITIONS / f"{cls}/{tgt}.pt", root / "conditions" / rel)
        link(REPO_ROOT / inv["latents"][ref], root / "reference_latents" / rel)
        link(COND_CLEAN / f"{cls}/{tgt}.pt", root / "cond_clean_latents" / rel)

        tdata = torch.load(REPO_ROOT / inv["latents"][tgt], map_location="cpu", weights_only=True)
        f, h, w = int(tdata["num_frames"]), int(tdata["height"]), int(tdata["width"])
        key = (f, h, w, p["sidedness"])
        if key not in mask_cache:
            m = torch.zeros(f, h, w)
            m[:2] = 1.0                                    # prefix anchor (2 latent frames)
            if p["sidedness"] == "twosided":
                m[-1] = 1.0                                # suffix anchor (last latent frame)
            mask_cache[key] = m
        mdst = root / "masks" / rel
        mdst.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mask": mask_cache[key].clone()}, mdst)

    counts = assert_counts(root, ["latents", "conditions", "reference_latents", "masks",
                                  "cond_clean_latents"])
    n_two = sum(1 for p in pairs if p["sidedness"] == "twosided")
    print(f"  {name:26s} {counts}  ({n_two} two-sided / {len(pairs) - n_two} one-sided pairs)")


def main() -> None:
    inv = json.loads(INVENTORY.read_text())
    sided = prompts.sidedness()

    missing_cond = [c for c in inv["latents"]
                    if not (CONDITIONS / f"{prompts.clip_class(c)}/{c}.pt").exists()]
    if missing_cond:
        sys.exit(f"[assemble] {len(missing_cond)} text embeddings missing (run precompute "
                 f"--mode text first): {missing_cond[:8]}")
    missing_cc = [c for c in inv["latents"]
                  if not (COND_CLEAN / f"{prompts.clip_class(c)}/{c}.pt").exists()]
    if missing_cc:
        sys.exit(f"[assemble] {len(missing_cc)} cond_clean latents missing (run precompute "
                 f"--mode cond-clean first): {missing_cc[:8]}")

    print(f"[assemble] {len(inv['models'])} roots -> {ROOTS}")
    for name, meta in inv["models"].items():
        clips = inv["clips"][name]
        if meta["kind"] == "specialist":
            cls = meta["classes"][0]
            assemble_specialist(name, cls, clips, sided[cls] == "two", inv)
        else:
            assemble_generalist(name, inv["pairs"], inv)
    print("[assemble] all roots built; source counts verified equal")


if __name__ == "__main__":
    main()
