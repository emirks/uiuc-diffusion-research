"""CTT v2 — one dry-run epoch over the assembled root, CPU only (A5 RULING 9).

Reproduces `ltx_trainer.datasets.PreprocessedDataset._discover_samples` exactly — the same
`glob("**/*.pt")` over each configured source and the same "expected relative path must
exist in every other source" join — and then resolves and metadata-loads every sample.

**Zero skipped samples is a requirement, not a warning.**  The trainer only debug-logs a
skip (`datasets.py:202-228`), which is how a silently-truncated epoch reaches the claim
table.  Here any skip — join miss, dangling symlink, unreadable tensor, wrong keys,
disagreeing shapes — is promoted to a job failure: the script prints the count and the
first 10 offenders and exits non-zero.

**And "zero skipped" is TWO-SIDED.**  An absence assert passes trivially when the
instrument found nothing to inspect, so this one may only PASS if the population was
positively identified first: the epoch must resolve EXACTLY the sample count
`ROOT_MANIFEST.json` names (`--expect` overrides).  Under-count is the silent-truncation
case; over-count means the root holds samples nothing accounts for.  Both fail.  Without
that control, an empty or half-assembled root reports "0 skipped" and reads as healthy —
which is the worst failure direction available on a HARD gate.

Tensors are loaded once per DISTINCT resolved target (the mix is realised by symlink
duplication, so ~60 k samples share ~1.5 k physical tensors).  Every sample path is still
resolved and stat-ed individually; only the payload read is deduplicated.

**This script reads NO LOG, deliberately.**  The trainer logs through `RichHandler`, so its
numbers arrive wrapped in SGR colour codes and OSC-8 hyperlink escapes, and a regex over the
raw output silently never matches — which on this check would mean reporting zero skipped
samples on a root that is quietly dropping them: a false PASS on a HARD gate, the worst
failure direction available.  So "zero skipped" here is a COUNTED property of the filesystem
and of tensors this process opened itself, never a string match.  The one log-reading check
in the suite (`Fast index: N of N`) lives in `assert_root_shapes.py:B5`, which strips ANSI.

    python scripts/ctt_v2/dryrun_epoch.py --root <root>
    python scripts/ctt_v2/dryrun_epoch.py --root <root> --no-payload   # paths only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import root_common as rc  # noqa: E402

MAX_SHOW = 10

#: REF_root_format.md — the tensor payload contract
LATENT_KEYS = ("latents", "num_frames", "height", "width", "fps")
COND_KEYS = ("video_prompt_embeds", "prompt_attention_mask")
LATENT_DIRS = ("latents", "reference_latents", "cond_clean_latents")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--no-payload", action="store_true",
                    help="resolve paths only; skip the metadata read of each distinct tensor")
    ap.add_argument("--expect", type=int,
                    help="the sample count this root must have (default: the root manifest's "
                         "counts.total_samples). A mismatch is a FAILURE.")
    ap.add_argument("--report", help="where to write DRYRUN_REPORT.json (default: <root>/)")
    args = ap.parse_args()

    t0 = time.time()
    root = Path(args.root)
    offenders: list[str] = []

    # ---- POSITIVE-PRESENCE CONTROL (A11 sigma/S4 ruling, standing rule) ----------------
    # "zero skipped" is an ABSENCE assert, and an absence assert passes trivially when the
    # instrument found nothing to look at.  So it may only PASS if the population it was
    # supposed to inspect was positively identified first: the root manifest must exist and
    # name a sample count, and the epoch must resolve EXACTLY that many samples.  Without
    # this, an empty or half-assembled root reports "0 skipped" and reads as healthy.
    expected = args.expect
    expect_src = "--expect"
    if expected is None:
        mpath = root / "ROOT_MANIFEST.json"
        if not mpath.exists():
            print(f"[dryrun] FAIL: {mpath} is absent, so there is no independent count to "
                  f"check the epoch against. 'Zero skipped' cannot be trusted without a "
                  f"positive control on the population size — pass --expect to override.")
            return 2
        man = rc.read_json(mpath)
        expected = int(man["counts"]["total_samples"])
        expect_src = "ROOT_MANIFEST.json:counts.total_samples"
    print(f"[dryrun] positive control: expecting exactly {expected} samples ({expect_src})")

    # ---- pass 1: the trainer's own index build ----------------------------------------
    paths, sets = {}, {}
    for sub in rc.ROOT_DIRS:
        base = root / sub
        if not base.is_dir():
            print(f"[dryrun] FAIL: required source dir does not exist: {base}")
            return 2
        paths[sub] = sorted(base.glob("**/*.pt"))
        sets[sub] = {str(p.relative_to(base)) for p in paths[sub]}
    primary = "latents"
    n_primary = len(paths[primary])
    if not n_primary:
        print(f"[dryrun] FAIL: no data files under {root / primary}")
        return 2

    valid = []
    for p in paths[primary]:
        rel = str(p.relative_to(root / primary))
        miss = [sub for sub in rc.ROOT_DIRS if sub != primary and rel not in sets[sub]]
        if miss:
            offenders.append(f"JOIN-MISS {rel}: absent from {miss}")
        else:
            valid.append(rel)
    join_skipped = n_primary - len(valid)
    print(f"[dryrun] index: {len(valid)} joined / {n_primary} primary files "
          f"({join_skipped} join-skipped)")
    # a file present in masks/ but absent from latents/ is never enumerated by the trainer
    orphans = sorted(set().union(*(sets[s] for s in rc.ROOT_DIRS)) - sets[primary])
    for rel in orphans:
        offenders.append(f"ORPHAN {rel}: present in a non-primary source, absent from latents/")

    # ---- pass 2: resolve every sample path, metadata-load every distinct target --------
    cache: dict[str, object] = {}
    n_resolved = n_loaded = 0
    epoch_ok = 0
    torch = None
    if not args.no_payload:
        import torch  # noqa: PLC0415

    def meta(path: str, sub: str):
        nonlocal n_loaded
        if path in cache:
            return cache[path]
        d = torch.load(path, map_location="cpu", weights_only=True)
        n_loaded += 1
        if sub in LATENT_DIRS:
            missing = [k for k in LATENT_KEYS if k not in d]
            info = ("latent", None if missing else
                    (int(d["num_frames"]), int(d["height"]), int(d["width"])),
                    tuple(d["latents"].shape) if "latents" in d else None, missing)
        elif sub == "masks":
            info = ("mask", tuple(d["mask"].shape) if "mask" in d else None, None,
                    [] if "mask" in d else ["mask"])
        else:
            missing = [k for k in COND_KEYS if k not in d]
            info = ("cond", None, tuple(d["video_prompt_embeds"].shape)
                    if "video_prompt_embeds" in d else None, missing)
        cache[path] = info
        return info

    for rel in valid:
        bad = False
        infos = {}
        for sub in rc.ROOT_DIRS:
            p = root / sub / rel
            try:
                real = os.path.realpath(p)
                if not os.path.exists(real):
                    offenders.append(f"DANGLING {sub}/{rel} -> {real}")
                    bad = True
                    continue
                n_resolved += 1
            except OSError as exc:
                offenders.append(f"UNRESOLVABLE {sub}/{rel}: {exc}")
                bad = True
                continue
            if args.no_payload:
                continue
            try:
                infos[sub] = meta(real, sub)
            except Exception as exc:  # noqa: BLE001 — any read failure is a skip
                offenders.append(f"UNREADABLE {sub}/{rel}: {exc!r}")
                bad = True
        if bad:
            continue
        if not args.no_payload:
            for sub, info in infos.items():
                if info[3]:
                    offenders.append(f"BAD-KEYS {sub}/{rel}: missing {info[3]}")
                    bad = True
            shapes = {sub: infos[sub][1] for sub in LATENT_DIRS if sub in infos}
            if len(set(shapes.values())) > 1:
                offenders.append(f"SHAPE-DISAGREE {rel}: {shapes}")
                bad = True
            if "masks" in infos and shapes:
                fhw = next(iter(shapes.values()))
                if infos["masks"][1] != fhw:
                    offenders.append(f"MASK-SHAPE {rel}: mask {infos['masks'][1]} != "
                                     f"latent {fhw}")
                    bad = True
        if not bad:
            epoch_ok += 1

    skipped = n_primary - epoch_ok + len(orphans) if not args.no_payload else \
        n_primary - len(valid) + len(orphans)

    # the positive control, evaluated: the epoch must have seen exactly the population the
    # manifest names.  Under-count and over-count are both failures — an under-count is the
    # silent-truncation case, an over-count means the root holds samples nothing accounts for.
    counted = epoch_ok if not args.no_payload else len(valid)
    control_ok = counted == expected
    if not control_ok:
        offenders.insert(0, f"POPULATION-MISMATCH the epoch resolved {counted} usable samples "
                            f"but {expect_src} says {expected} — 'zero skipped' is not "
                            f"meaningful unless these agree")
    print(f"[dryrun] positive control: {'PASS' if control_ok else 'FAIL'} "
          f"({counted} usable == {expected} expected)")

    elapsed = time.time() - t0
    rep = {
        "root": str(root), "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "positive_control": {"expected": expected, "source": expect_src,
                             "counted": counted, "ok": control_ok},
        "n_primary_files": n_primary, "n_joined": len(valid), "n_epoch_ok": epoch_ok,
        "n_orphans": len(orphans), "n_paths_resolved": n_resolved,
        "n_distinct_tensors_loaded": n_loaded, "n_skipped": skipped,
        "payload_checked": not args.no_payload,
        "offenders_first_10": offenders[:MAX_SHOW], "n_offenders": len(offenders),
        "elapsed_s": round(elapsed, 2),
    }
    out = Path(args.report) if args.report else root / "DRYRUN_REPORT.json"
    try:
        rc.write_json(out, rep)
    except OSError as exc:
        print(f"[dryrun] WARNING: could not write {out}: {exc}")

    print(f"[dryrun] resolved {n_resolved} sample paths, loaded {n_loaded} distinct tensors, "
          f"{epoch_ok} samples usable in {elapsed:.1f}s")
    if skipped or offenders or not control_ok:
        print(f"[dryrun] FAIL: {skipped} SKIPPED sample(s), positive control "
              f"{'PASS' if control_ok else 'FAIL'} — promoted to a job failure "
              f"({len(offenders)} offender records)")
        for o in offenders[:MAX_SHOW]:
            print(f"        - {o}")
        if len(offenders) > MAX_SHOW:
            print(f"        ... and {len(offenders) - MAX_SHOW} more")
        return 1
    print(f"[dryrun] ZERO skipped samples over {epoch_ok} samples, and the population "
          f"matched the manifest exactly -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
