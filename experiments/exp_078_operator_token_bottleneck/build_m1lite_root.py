"""Build the M1-lite training root: same as B1 except reference_latents = COARSE (192-token) encodes.

M1-lite = plain IC-LoRA with a single DOWNSCALED real reference (no encoder, no bottleneck). The
target/mask/cond_clean/conditions are B1's (identical); only the reference is swapped from the
full-res 4800-token latent to the 96x128 coarse 192-token latent of the same reference clip.

    python experiments/exp_078_operator_token_bottleneck/build_m1lite_root.py [--check]
"""

import argparse
import glob
import os
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
REPO_ROOT = EXP.parents[1]
B1_ROOT = EXP / "dataset" / "roots" / "b1"
COARSE = EXP / "dataset" / "coarse_ref_latents"        # <class>/<clip>.pt
OUT = EXP / "dataset" / "roots" / "m1lite"

# sources that carry over from B1 verbatim (resolve via the B1 root's links)
PASSTHROUGH = ["latents", "masks", "cond_clean_latents", "conditions"]


def coarse_ref(reference: str) -> Path:
    hits = glob.glob(str(COARSE / f"*/{reference}.pt"))
    if not hits:
        raise FileNotFoundError(f"no coarse latent for reference {reference} (run encode_coarse_refs first)")
    return Path(hits[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    pairs = sorted(p.relative_to(B1_ROOT / "masks") for p in (B1_ROOT / "masks").rglob("*.pt"))
    print(f"[m1lite] {len(pairs)} pairs")

    plan: dict[str, dict[Path, Path]] = {s: {} for s in PASSTHROUGH + ["reference_latents"]}
    missing = []
    for rel in pairs:
        for s in PASSTHROUGH:
            link = B1_ROOT / s / rel
            tgt = Path(os.readlink(link)) if link.is_symlink() else link.resolve()
            if not tgt.exists():
                missing.append(f"{s}/{rel}")
            plan[s][rel] = tgt
        # reference_latents -> COARSE latent of the pair's reference clip
        reference = rel.name[: -len(".pt")].split("__ref_")[1]
        try:
            tgt = coarse_ref(reference)
        except FileNotFoundError as e:
            missing.append(str(e)); continue
        plan["reference_latents"][rel] = tgt

    if missing:
        print(f"[fatal] {len(missing)} unresolved, e.g. {missing[:3]}", file=sys.stderr)
        return 1

    # SEATBELT: equal relative-path sets across all sources (trainer silently drops mismatches).
    ref = set(plan["latents"])
    for s, m in plan.items():
        if set(m) != ref:
            print(f"[fatal] source '{s}' path set differs ({len(m)} vs {len(ref)})", file=sys.stderr)
            return 1
    print(f"[m1lite] seatbelt PASS — {len(plan)} sources share {len(ref)} paths")

    if args.check:
        print("[m1lite] --check: nothing written")
        return 0

    written = 0
    for s, m in plan.items():
        for rel, tgt in m.items():
            link = OUT / s / rel
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(tgt)
            written += 1
    unresolved = [p for p in OUT.rglob("*.pt") if not p.resolve().is_file()]
    if unresolved:
        print(f"[fatal] {len(unresolved)} links dangle, e.g. {unresolved[:3]}", file=sys.stderr)
        return 1
    print(f"[m1lite] wrote {written} links -> {OUT.relative_to(REPO_ROOT)}; all resolve")
    for s in plan:
        print(f"   {s:20s} {len(list((OUT / s).rglob('*.pt')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
