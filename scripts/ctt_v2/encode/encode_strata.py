"""CTT v2 — VAE-encode `latents/` + `cond_clean/` for every new dataset stratum.

Scope is deliberately HALF the root contract: this writes only the two PROMPT-AGNOSTIC
trees, so it is completely independent of the Gemini caption blocker.

    <ENC>/<stratum>/latents/<stem>.pt       full-video VAE encode
    <ENC>/<stratum>/cond_clean/<stem>.pt    isolation-encoded conditioning anchor

`masks/` are shape-derived and built by `assemble_root.py:ensure_mask()`; `conditions/`
are Gemma text embeds and need captions.  Neither belongs here.

Authority
---------
* `misc/ctt_v2_final/REF_root_format.md`  — the on-disk root contract
* `eval_ladder/encode_conditioning.py`    — the ONE definition of a conditioning window
  (`write_cond_clean`, `PX_PREFIX=9`, `PX_SUFFIX=9`, `STD_FRAMES=121`)
* `experiments/exp_077_synth_stratum/encode_d2full.py` — the proven encode call

The latent payload is produced by the trainer's own `scripts/process_videos.py`, never by a
hand-rolled encode, so the payload schema (`latents, num_frames, height, width, fps`) is
byte-identical to what the ic_gen root already carries.  `fps` is read from the mp4 itself,
which is why S4 lands at 16.0 and S2/S1 at 24.0 without any special-casing — the trainer
reads fps per sample and scales RoPE to seconds, so a wrong fps is a silent training defect.

Sidedness (drives cond_clean, per REF_root_format.md)
    two-sided -> last latent frame replaced by a STANDALONE encode of the trailing 9 pixels
    one-sided -> BITWISE COPY (the tree must stay shape-complete: the trainer joins the five
                 source trees by identical relative path and SILENTLY SKIPS mismatches)
    S2a/S2b   two   (a true A->B pair)
    S1        per specialist, read from `eval_ladder/registry.jsonl` (9 of 11 are one-sided;
              only `hero_flight` and `shadow_smoke` are two-sided)
    S4        one   (refVFX I2V_LoRA is A -> A-transformed)

🔴 S4 FORMAT NOTE — read before trusting the S4 shapes
    refVFX I2V_LoRA is natively 832x464 / 33f / 16fps.  33 frames is VAE-legal (33 % 8 == 1),
    16 fps is carried through faithfully, but **464 is NOT a multiple of the VAE's spatial
    factor 32** (464/32 = 14.5) and `process_videos.py:parse_resolution_buckets` rejects it
    outright.  S4 therefore CANNOT be encoded at literally-native height.  The bucket used
    here is `832x448x33`, which for a 832x464 source is a PURE 16-row CENTRE CROP with NO
    resampling (`_resize_and_crop` scales by width first; the width scale is exactly 1.0).
    Resulting latent grid: (128, 5, 14, 26).  Changing `S4_BUCKET` and re-running re-encodes
    S4 for ~0.3 L40S-h, so this is a cheap decision to revisit.

Sharding
--------
`NSHARDS` is a HARDCODED per-stratum constant in this file and is NEVER derived from
`SLURM_ARRAY_TASK_COUNT`: a partial resubmit with a different `--array` range would
otherwise silently re-partition the work and corrupt the run.  The roster each stratum is
sharded over is FROZEN to `<ENC>/<stratum>/ROSTER.json` at stage time for the same reason.

Everything is idempotent: outputs already on disk are skipped, every write goes through a
`.tmp` + `os.replace`, so a preempted/requeued task can never leave a truncated `.pt` that
the skip-if-exists logic would then trust.

Usage
-----
    # 1. CPU, once, on a login node: freeze rosters + stage clips (extracts S4 from the tar)
    python encode_strata.py stage

    # 2. GPU, job array (see job_encode.sbatch)
    python encode_strata.py encode --group s2  --shard $SLURM_ARRAY_TASK_ID
    python encode_strata.py encode --group aux --shard $SLURM_ARRAY_TASK_ID

    # 3. CPU: hard count assert + shape/fps spot check
    python encode_strata.py verify
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]                      # the worktree checkout
LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
MAIN = LAB / "diffusion-research"                # main tree: READ-ONLY for videos
TRAINER = LAB / "LTX-2-cond-bleed-fix/packages/ltx-trainer"
VENV_PY = LAB / "LTX-2-official/.venv/bin/python"
MODEL = LAB / "cache/huggingface/ltx2_models/ltx-2-19b-dev.safetensors"

sys.path.insert(0, str(REPO_ROOT / "eval_ladder"))   # encode_conditioning
#: same default `encode_conditioning._trainer_bits()` uses, so the script also works when the
#: caller forgot to export PYTHONPATH (the sbatch does export it).
sys.path.insert(0, os.environ.get("LTX_TRAINER_SRC", str(TRAINER / "src")))

#: bulk generated artifacts live in the main tree's (gitignored) outputs/, next to the
#: videos they are derived from — the worktree is source only and may be removed.
ENC = MAIN / "outputs/ctt_v2/encodes"

# --------------------------------------------------------------------------------------
# HARDCODED shard counts.  NEVER derive these from SLURM_ARRAY_TASK_COUNT.
# --------------------------------------------------------------------------------------
NSHARDS = {"S2a": 16, "S2b": 16, "S1": 1, "S4": 4}
GROUPS = {"s2": ["S2a", "S2b"], "aux": ["S1", "S4"]}

#: (width, height, frames) exactly as `process_videos.py --resolution-buckets WxHxF`
STD_BUCKET = (480, 640, 121)
S4_BUCKET = (832, 448, 33)          # see the S4 FORMAT NOTE above

STRATA = {
    "S2a": {"bucket": STD_BUCKET, "fps": 24.0, "sided": "two",
            "src": MAIN / "outputs/videos/ctt_v2_s2/full/videos", "mode": "flat"},
    "S2b": {"bucket": STD_BUCKET, "fps": 24.0, "sided": "two",
            "src": MAIN / "outputs/videos/ctt_v2_s2_humanvid/full/videos", "mode": "flat"},
    "S1":  {"bucket": STD_BUCKET, "fps": 24.0, "sided": "registry",
            "src": MAIN / "outputs/videos/ctt_v2_s1", "mode": "spec"},
    "S4":  {"bucket": S4_BUCKET, "fps": 16.0, "sided": "one",
            "src": REPO_ROOT / "data/processed/s4_refvfx/selection.json", "mode": "tar"},
}

S4_TARS = MAIN / "data/raw/refvfx/data/I2V_LoRA"
REGISTRY = REPO_ROOT / "eval_ladder/registry.jsonl"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sdir(stratum: str) -> Path:
    return ENC / stratum


def paths(stratum: str) -> tuple[Path, Path, Path, Path]:
    d = sdir(stratum)
    return d / "clips", d / "latents", d / "cond_clean", d / "ROSTER.json"


# --------------------------------------------------------------------------------- sidedness
def s1_sidedness() -> dict[str, str]:
    """spec arm -> {'one','two'}, read from the frozen eval registry (never a literal list)."""
    out: dict[str, str] = {}
    for line in REGISTRY.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        arm, sided = r.get("arm"), r.get("sided")
        if not (isinstance(arm, str) and arm.startswith("spec_") and sided in ("one", "two")):
            continue
        if out.setdefault(arm, sided) != sided:
            raise SystemExit(f"[encode] registry disagrees about sidedness of {arm}")
    if not out:
        raise SystemExit(f"[encode] no spec_* rows with a sidedness in {REGISTRY}")
    return out


def clip_sided(stratum: str, stem: str, s1_map: dict[str, str] | None = None) -> str:
    spec = STRATA[stratum]["sided"]
    if spec in ("one", "two"):
        return spec
    arm = stem.split("__", 1)[0]                 # spec_hero_flight__<endpoints>__s42
    m = s1_map if s1_map is not None else s1_sidedness()
    if arm not in m:
        raise SystemExit(f"[encode] {stratum}/{stem}: arm {arm!r} not in the registry")
    return m[arm]


# ------------------------------------------------------------------------------------- stage
def _discover(stratum: str) -> list[tuple[str, Path | dict]]:
    cfg = STRATA[stratum]
    if cfg["mode"] == "flat":
        return sorted((p.stem, p) for p in Path(cfg["src"]).glob("*.mp4"))
    if cfg["mode"] == "spec":
        return sorted((p.stem, p) for p in Path(cfg["src"]).glob("spec_*/*.mp4"))
    if cfg["mode"] == "tar":
        sel = json.loads(Path(cfg["src"]).read_text())
        return sorted((r["k"], r) for r in sel["samples"])
    raise SystemExit(f"unknown mode for {stratum}")


def stage(strata: list[str], force: bool = False) -> None:
    """Freeze the roster and materialise one mp4 per clip under <ENC>/<stratum>/clips/."""
    tars = sorted(S4_TARS.glob("shard-*.tar"))
    for stratum in strata:
        clips, lat, cc, roster_p = paths(stratum)
        for d in (clips, lat, cc):
            d.mkdir(parents=True, exist_ok=True)
        found = _discover(stratum)
        stems = [s for s, _ in found]
        if len(set(stems)) != len(stems):
            raise SystemExit(f"[stage] {stratum}: duplicate stems in the source")

        if roster_p.exists() and not force:
            roster = json.loads(roster_p.read_text())
            if roster["stems"] != stems:
                raise SystemExit(
                    f"[stage] {stratum}: FROZEN roster ({len(roster['stems'])} clips) disagrees "
                    f"with the source ({len(stems)} clips). Sharding is defined by the frozen "
                    f"roster; re-run with --force only if you intend to re-partition.")
        else:
            roster = {"stratum": stratum, "n": len(stems), "stems": stems,
                      "source": str(STRATA[stratum]["src"]),
                      "bucket_whf": list(STRATA[stratum]["bucket"]),
                      "fps": STRATA[stratum]["fps"],
                      "nshards": NSHARDS[stratum],
                      "frozen": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            tmp = roster_p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(roster, indent=1) + "\n")
            tmp.replace(roster_p)

        made = 0
        for stem, src in found:
            dst = clips / f"{stem}.mp4"
            if dst.exists() and dst.stat().st_size > 0:
                continue
            if isinstance(src, dict):                     # S4: slice bytes out of the shard
                off, size, ext = src["out"]
                if ext != "mp4":
                    raise SystemExit(f"[stage] S4/{stem}: unexpected member ext {ext!r}")
                tar = tars[src["shard"]]
                with open(tar, "rb") as fh:
                    fh.seek(off)
                    buf = fh.read(size)
                if len(buf) != size:
                    raise SystemExit(f"[stage] S4/{stem}: short read {len(buf)} != {size}")
                tmp = dst.with_suffix(".mp4.tmp")
                tmp.write_bytes(buf)
                tmp.replace(dst)
            else:
                if dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(os.path.realpath(src))
            made += 1
            if made % 500 == 0:
                log(f"[stage] {stratum}: {made} clips staged")
        log(f"[stage] {stratum}: roster {len(stems)} clips (nshards={NSHARDS[stratum]}), "
            f"{made} newly staged -> {clips}")


def roster(stratum: str) -> list[str]:
    _, _, _, roster_p = paths(stratum)
    if not roster_p.exists():
        raise SystemExit(f"[encode] {stratum}: no frozen roster ({roster_p}) — run `stage` first")
    return json.loads(roster_p.read_text())["stems"]


def shard_of(stratum: str, shard: int) -> list[str]:
    n = NSHARDS[stratum]
    return [s for i, s in enumerate(roster(stratum)) if i % n == shard] if shard < n else []


# ------------------------------------------------------------------------------------ encode
def run_process_videos(manifest: Path, out_dir: Path, bucket: tuple[int, int, int],
                       device: str) -> None:
    w, h, f = bucket
    cmd = [str(VENV_PY), "scripts/process_videos.py", str(manifest),
           "--resolution-buckets", f"{w}x{h}x{f}",
           "--output-dir", str(out_dir), "--model-path", str(MODEL),
           "--video-column", "video", "--device", device]
    log("[cmd] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(TRAINER), check=True,
                   env={**os.environ, "PYTHONPATH": str(TRAINER / "src")})


def encode_stratum(stratum: str, shard: int, device: str, limit: int = 0) -> dict:
    import torch

    import encode_conditioning as ec  # noqa: PLC0415

    cfg = STRATA[stratum]
    clips, lat_dir, cc_dir, _ = paths(stratum)
    w, h, f = cfg["bucket"]
    want = shard_of(stratum, shard)
    if limit:
        want = want[:limit]
    if not want:
        log(f"[encode] {stratum}: shard {shard} is empty (nshards={NSHARDS[stratum]}) — skip")
        return {"stratum": stratum, "shard": shard, "clips": 0}
    log(f"[encode] {stratum} shard {shard}/{NSHARDS[stratum]}: {len(want)} clips, "
        f"bucket {w}x{h}x{f}, expect fps={cfg['fps']}")

    # -- pre-flight: process_videos SILENTLY DROPS clips shorter than the bucket ------------
    from ltx_trainer.video_utils import get_video_frame_count  # noqa: PLC0415

    short = []
    for s in want:
        n = int(get_video_frame_count(clips / f"{s}.mp4"))
        if n < f:
            short.append((s, n))
    if short:
        raise SystemExit(f"[encode] {stratum}: {len(short)} clips < {f} frames "
                         f"-> SILENT SKIP risk: {short[:5]}")
    log(f"[encode] {stratum}: pre-flight OK, all {len(want)} clips >= {f} frames")

    # -- step 1: full-video latents ---------------------------------------------------------
    todo = [s for s in want if not (lat_dir / f"{s}.pt").exists()]
    log(f"[encode] {stratum}: {len(todo)} latents to encode ({len(want) - len(todo)} present)")
    if todo:
        man = clips / f"_manifest_{stratum}_shard{shard:02d}.json"
        man.write_text(json.dumps([{"video": f"{s}.mp4"} for s in todo], indent=1))
        run_process_videos(man, lat_dir, cfg["bucket"], device)
    miss = [s for s in want if not (lat_dir / f"{s}.pt").exists()]
    if miss:
        raise SystemExit(f"[encode] {stratum}: {len(miss)} latents missing after encode "
                         f"(SILENT DROP): {miss[:5]}")

    # -- step 2: cond_clean -----------------------------------------------------------------
    s1_map = s1_sidedness() if cfg["sided"] == "registry" else None
    sided = {s: clip_sided(stratum, s, s1_map) for s in want}
    todo_cc = [s for s in want if not (cc_dir / f"{s}.pt").exists()]
    n_two = sum(1 for s in todo_cc if sided[s] == "two")
    log(f"[encode] {stratum}: {len(todo_cc)} cond_clean to write "
        f"({n_two} corrected / {len(todo_cc) - n_two} bitwise copy)")

    vae = None
    rels, shapes, t0 = [], set(), time.time()
    for i, s in enumerate(todo_cc, 1):
        src = lat_dir / f"{s}.pt"
        # every clip's payload is checked, not a sample: a wrong fps is a SILENT training bug
        d = torch.load(src, map_location="cpu", weights_only=True)
        assert set(d) >= {"latents", "num_frames", "height", "width", "fps"}, \
            f"{stratum}/{s}: payload keys {sorted(d)}"
        assert abs(float(d["fps"]) - cfg["fps"]) < 1e-6, \
            f"{stratum}/{s}: fps={d['fps']} want {cfg['fps']}"
        assert (int(d["height"]), int(d["width"])) == (h // 32, w // 32), \
            f"{stratum}/{s}: latent HxW {d['height']}x{d['width']} want {h // 32}x{w // 32}"
        assert int(d["num_frames"]) == (f - 1) // 8 + 1, \
            f"{stratum}/{s}: latent frames {d['num_frames']} want {(f - 1) // 8 + 1}"
        shapes.add(tuple(d["latents"].shape))
        del d

        if sided[s] == "two" and vae is None:
            log(f"[encode] {stratum}: loading VAE for the two-sided suffix correction")
            vae = ec.load_vae(str(MODEL), device=device, dtype=torch.bfloat16)
        tmp = cc_dir / f"{s}.pt.tmp"
        r = ec.write_cond_clean(orig_pt=src, dst_pt=tmp, clip_mp4=clips / f"{s}.mp4",
                                correct_suffix=sided[s] == "two", vae=vae, device=device)
        tmp.replace(cc_dir / f"{s}.pt")
        if r["corrected"]:
            rels.append(r["suffix_rel_l2"])
        if i % 100 == 0 or i == len(todo_cc):
            log(f"[encode] {stratum}: cond_clean {i}/{len(todo_cc)} "
                f"({(time.time() - t0) / i:.2f}s/clip)")
    if vae is not None:
        del vae
        torch.cuda.empty_cache()

    miss_cc = [s for s in want if not (cc_dir / f"{s}.pt").exists()]
    if miss_cc:
        raise SystemExit(f"[encode] {stratum}: cond_clean missing: {miss_cc[:5]}")
    med = sorted(rels)[len(rels) // 2] if rels else None
    log(f"[encode] {stratum} shard {shard} DONE: {len(want)} latents + {len(want)} cond_clean; "
        f"latent shapes={sorted(shapes)}"
        + (f"; median suffix_rel_l2={med:.4f}" if med is not None else ""))
    return {"stratum": stratum, "shard": shard, "clips": len(want),
            "latents_new": len(todo), "cond_clean_new": len(todo_cc),
            "median_suffix_rel_l2": med, "shapes": sorted(str(s) for s in shapes)}


# ------------------------------------------------------------------------------------ verify
def verify(strata: list[str], n_sample: int = 3, seed: int = 42, count: bool = True) -> int:
    import torch

    rng = random.Random(seed)
    bad = 0
    for stratum in strata:
        clips, lat_dir, cc_dir, _ = paths(stratum)
        cfg = STRATA[stratum]
        w, h, f = cfg["bucket"]
        want = roster(stratum)
        got_l = {p.stem for p in lat_dir.glob("*.pt")}
        got_c = {p.stem for p in cc_dir.glob("*.pt")}
        exp = set(want)
        print(f"\n=== {stratum} ===")
        print(f"  roster {len(exp)} | latents {len(got_l)} | cond_clean {len(got_c)}")
        for name, got in (("latents", got_l), ("cond_clean", got_c)):
            if got == exp:
                print(f"  ✓ {name}: exactly {len(exp)} files, set-equal to the roster")
            elif count:
                bad += 1
                print(f"  ✗ {name}: MISSING {len(exp - got)} {sorted(exp - got)[:5]} | "
                      f"EXTRA {len(got - exp)} {sorted(got - exp)[:5]}")
            else:
                print(f"  · {name}: {len(got)}/{len(exp)} present "
                      f"(count assert skipped: --no-count)")

        s1_map = s1_sidedness() if cfg["sided"] == "registry" else None
        pool = sorted(exp & got_l & got_c)
        if not pool:
            continue
        for s in rng.sample(pool, min(n_sample, len(pool))):
            a = torch.load(lat_dir / f"{s}.pt", map_location="cpu", weights_only=True)
            b = torch.load(cc_dir / f"{s}.pt", map_location="cpu", weights_only=True)
            la, lb = a["latents"], b["latents"]
            sd = clip_sided(stratum, s, s1_map)
            same_head = torch.equal(la[:, :-1], lb[:, :-1])
            diff_tail = not torch.equal(la[:, -1:], lb[:, -1:])
            ok = (same_head and diff_tail == (sd == "two")
                  and abs(float(a["fps"]) - cfg["fps"]) < 1e-6
                  and (int(a["height"]), int(a["width"])) == (h // 32, w // 32)
                  and int(a["num_frames"]) == (f - 1) // 8 + 1)
            bad += not ok
            print(f"  {'✓' if ok else '✗'} {s:52s} shape={tuple(la.shape)} dtype={la.dtype} "
                  f"nf={a['num_frames']} h={a['height']} w={a['width']} fps={a['fps']} "
                  f"sided={sd} cond_clean[head==]={same_head} [tail!=]={diff_tail}")
    print(f"\n[verify] {'ALL CHECKS PASS' if bad == 0 else f'{bad} FAILURES'}")
    return 1 if bad else 0


# -------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("stage", help="freeze rosters + materialise clips (CPU)")
    ps.add_argument("--strata", default="all")
    ps.add_argument("--force", action="store_true", help="re-freeze a roster (RE-PARTITIONS)")

    pe = sub.add_parser("encode", help="encode one shard (GPU)")
    pe.add_argument("--group", choices=sorted(GROUPS), help="s2 = S2a+S2b, aux = S1+S4")
    pe.add_argument("--strata", default=None, help="explicit comma list (overrides --group)")
    pe.add_argument("--shard", type=int, required=True)
    pe.add_argument("--device", default="cuda")
    pe.add_argument("--limit", type=int, default=0, help="smoke test: first N clips of the shard")

    pv = sub.add_parser("verify", help="hard count assert + shape/fps/cond_clean spot check")
    pv.add_argument("--strata", default="all")
    pv.add_argument("--n-sample", type=int, default=3)
    pv.add_argument("--no-count", action="store_true",
                    help="smoke mode: check shapes/fps/cond_clean only, not the roster counts")

    args = ap.parse_args()

    def pick(spec: str | None) -> list[str]:
        if not spec or spec == "all":
            return list(STRATA)
        out = [s.strip() for s in spec.split(",") if s.strip()]
        for s in out:
            if s not in STRATA:
                raise SystemExit(f"unknown stratum {s!r}; known: {sorted(STRATA)}")
        return out

    if args.cmd == "stage":
        stage(pick(args.strata), force=args.force)
        return 0
    if args.cmd == "verify":
        return verify(pick(args.strata), n_sample=args.n_sample, count=not args.no_count)

    strata = pick(args.strata) if args.strata else GROUPS[args.group]
    log(f"[encode] host={os.uname().nodename} group={args.group} strata={strata} "
        f"shard={args.shard} NSHARDS={ {s: NSHARDS[s] for s in strata} }")
    reports = [encode_stratum(s, args.shard, args.device, args.limit) for s in strata]
    log("[encode] shard report: " + json.dumps(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
