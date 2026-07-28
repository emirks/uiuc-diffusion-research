"""CTT v2 — REGENERATE the conditioning masks for every latent geometry in the mix (A9 §3).

A9, verbatim: *"Masks regenerated at (5,20,15), **never reused**"*.  REF_mixed_length item 2
is the reason: `flexible.py:533` reshapes `batch[mask_dir]["mask"]` to `(B, seq_len)`, so a
mask whose element count does not equal `F_lat*H_lat*W_lat` of its own sample is either a
LOUD `RuntimeError` or — worse, if the numbers happen to coincide — a silently wrong
conditioning pattern.  Reusing a 121-frame mask for a 33-frame sample is exactly that bug.

🔴 (5,20,15) IS NOT A REAL S4 GEOMETRY — see DOSSIER §10.9.
    refVFX I2V_LoRA is natively 832x464; 464/32 = 14.5 is not VAE-legal, so the delivered
    encode is the minimal legal deviation 832x448x33 (a pure 16-row centre crop, no
    resampling) => latent (5,14,26) = 1,820 tokens, NOT (5,20,15) = 1,500.
    (20,15) is the 480x640 corpus grid; no bucket derived from 832x464 can produce it.
    This module therefore refuses to invent A9's number: it DISCOVERS each stratum's
    geometry from the encoded latents on disk and regenerates masks for what is actually
    there, while asserting that the impossible geometry is absent from the store (so nobody
    later "fixes" the encode to match a number that cannot exist).

What "regenerate, never reuse" means operationally, and what is checked
----------------------------------------------------------------------
1. The geometry is read from EVERY encoded latent of the stratum, not a sample; more than
    one distinct geometry inside one stratum is a hard failure.
2. `--force` UNLINKS an existing mask before writing.  Without it, an existing file with a
    mismatched payload is a hard failure rather than a silent reuse.
3. Every mask is verified after writing: `numel == F*H*W`, dtype float32, values in {0,1},
    `m[:P] == 1`, `m[P:-1] == 0`, and `m[-1] == (1 if two-sided else 0)`, where
    `P = root_common.prefix_latents(shape)` — 2 at 121f, **1 for S4** (frame-0 conditioning).
4. The mask bytes are proven **bit-identical to `assemble_root.ensure_mask()`** — the
    function the real assembly will call — by running that function into a scratch dir and
    comparing sha256.  `assemble_root.py` is owned by another agent and is imported
    READ-ONLY; this module never edits it.
5. A `MASKS_MANIFEST.json` records each mask's geometry, sidedness, sha256, and which
    stratum/latent files justified it.

    python scripts/ctt_v2/masks/regen_masks.py --strata S4 --force
    python scripts/ctt_v2/masks/regen_masks.py --verify-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CTT = HERE.parent                                   # scripts/ctt_v2
REPO_ROOT = HERE.parents[2]
LAB = Path("/projects/illinois/eng/cs/jrehg/users/emirkisa")
MAIN = LAB / "diffusion-research"

ENC = MAIN / "outputs/ctt_v2/encodes"
STORE = MAIN / "outputs/ctt_v2/masks/_mask_store"
MANIFEST = MAIN / "outputs/ctt_v2/masks/MASKS_MANIFEST.json"

#: sidedness per stratum.  S1 is per-specialist and is read from the frozen registry, never
#: a literal list (same rule `encode/encode_strata.py` follows).
SIDEDNESS = {"S2a": "two", "S2b": "two", "S4": "one", "S1": "registry"}

#: the geometry A9 §3 names for S4 and which cannot exist; asserted ABSENT from the store
IMPOSSIBLE_S4_LATENT = (5, 20, 15)

REGISTRY = REPO_ROOT / "eval_ladder/registry.jsonl"

if str(CTT) not in sys.path:
    sys.path.insert(0, str(CTT))
import root_common as rc  # noqa: E402  -- prefix width is a shape property, read from here


def log(msg: str) -> None:
    print(f"[masks] {msg}", flush=True)


# --------------------------------------------------------------------------------------
def _ensure_mask_from_assembler():
    """Import `assemble_root.ensure_mask` / `mask_store_path` READ-ONLY (another agent owns
    that file).  Loaded by path so importing it cannot depend on cwd."""
    spec = importlib.util.spec_from_file_location("_ctt_assemble_root", CTT / "assemble_root.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.ensure_mask, mod.mask_store_path


def store_name(f: int, h: int, w: int, sided: str) -> str:
    """DELEGATED to the assembler, never restated.

    The name encodes the prefix width (`p1`/`p2`), which became a shape property when S4
    moved to frame-0 conditioning.  If this function kept its own format the two sides would
    disagree silently: regen would validate `f5_h14_w26_onesided.pt` while assembly read
    `f5_h14_w26_p1_onesided.pt`, and the mask assembly actually used would never be checked.
    """
    _, mask_store_path = _ensure_mask_from_assembler()
    return mask_store_path(Path("/"), f, h, w, sided).name


def content_sha256(t) -> str:
    """Hash of the mask TENSOR, not the file.

    `torch.save` is not byte-deterministic — its zip container records an mtime, so two saves
    of an identical tensor seconds apart differ.  A file-sha equality check between two fresh
    writes therefore passes or fails on wall-clock alignment, which is why the bit-identity
    check below compares tensors.  This hash is over the raw buffer and IS reproducible.
    """
    import hashlib

    c = t.contiguous()
    return hashlib.sha256(
        f"{tuple(c.shape)}|{c.dtype}|".encode() + c.numpy().tobytes()).hexdigest()


def sha256_file(p: Path) -> str:
    import hashlib

    hh = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hh.update(chunk)
    return hh.hexdigest()


def s1_sidedness() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in REGISTRY.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        arm, sided = r.get("arm"), r.get("sided")
        if isinstance(arm, str) and arm.startswith("spec_") and sided in ("one", "two"):
            if out.setdefault(arm, sided) != sided:
                raise SystemExit(f"registry disagrees about sidedness of {arm}")
    if not out:
        raise SystemExit(f"no spec_* rows with a sidedness in {REGISTRY}")
    return out


# --------------------------------------------------------------------------------------
def discover_geometries(stratum: str, every: bool = True) -> dict:
    """(F,H,W) x sidedness actually present in `<ENC>/<stratum>/latents/`.

    Reads EVERY latent by default: the whole point of the mixed-format hazard is that one
    off-geometry file is enough to break a run, and a sample of 3 would not see it.
    """
    import torch

    lat = ENC / stratum / "latents"
    files = sorted(lat.glob("*.pt"))
    if not files:
        raise SystemExit(f"{stratum}: no encoded latents under {lat}")
    if not every:
        files = files[:8]
    s1_map = s1_sidedness() if SIDEDNESS[stratum] == "registry" else None

    combos: dict[tuple, dict] = {}
    fps_seen: set[float] = set()
    for p in files:
        d = torch.load(p, map_location="cpu", weights_only=True)
        f, h, w = int(d["num_frames"]), int(d["height"]), int(d["width"])
        shp = tuple(int(x) for x in d["latents"].shape)
        if shp[1:] != (f, h, w):
            raise SystemExit(f"{stratum}/{p.stem}: tensor shape {shp} contradicts metadata "
                             f"(num_frames,height,width)=({f},{h},{w})")
        fps_seen.add(float(d["fps"]))
        if s1_map is not None:
            arm = p.stem.split("__", 1)[0]
            if arm not in s1_map:
                raise SystemExit(f"{stratum}/{p.stem}: arm {arm!r} not in the registry")
            sided = s1_map[arm]
        else:
            sided = SIDEDNESS[stratum]
        key = (f, h, w, sided)
        e = combos.setdefault(key, {"n": 0, "examples": []})
        e["n"] += 1
        if len(e["examples"]) < 3:
            e["examples"].append(p.stem)
        del d

    geoms = sorted({k[:3] for k in combos})
    if len(geoms) != 1:
        raise SystemExit(f"{stratum}: {len(geoms)} distinct latent geometries in one stratum "
                         f"{geoms} — the encode is not homogeneous, refusing to guess")
    return {"stratum": stratum, "n_latents_read": len(files), "geometry_fhw": list(geoms[0]),
            "tokens": geoms[0][0] * geoms[0][1] * geoms[0][2],
            "fps_seen": sorted(fps_seen),
            "combos": {f"f{k[0]}_h{k[1]}_w{k[2]}_{k[3]}sided": v for k, v in sorted(combos.items())}}


def verify_mask(path: Path, f: int, h: int, w: int, sided: str) -> list[str]:
    import torch

    bad = []
    d = torch.load(path, map_location="cpu", weights_only=True)
    if set(d) != {"mask"}:
        bad.append(f"payload keys {sorted(d)} != ['mask']")
    m = d["mask"]
    if tuple(m.shape) != (f, h, w):
        bad.append(f"shape {tuple(m.shape)} != ({f},{h},{w})")
        return bad
    if m.numel() != f * h * w:
        bad.append(f"numel {m.numel()} != {f*h*w}")
    if m.dtype != torch.float32:
        bad.append(f"dtype {m.dtype} != float32")
    if not bool(((m == 0) | (m == 1)).all()):
        bad.append("values outside {0,1}")
    p = rc.prefix_latents((f, h, w))
    if not bool((m[:p] == 1).all()):
        bad.append(f"prefix anchor m[:{p}] is not all 1 ({p} latent frame(s))")
    if f > p + 1 and not bool((m[p:-1] == 0).all()):
        bad.append(f"interior m[{p}:-1] is not all 0")
    tail_should_be = 1.0 if sided == "two" else 0.0
    if not bool((m[-1] == tail_should_be).all()):
        bad.append(f"suffix anchor m[-1] != {tail_should_be} for sided={sided!r}")
    return bad


# --------------------------------------------------------------------------------------
def regenerate(strata: list[str], force: bool, verify_only: bool, every: bool) -> int:
    import torch

    ensure_mask, mask_store_path = _ensure_mask_from_assembler()
    STORE.mkdir(parents=True, exist_ok=True)
    rec = {"schema": "ctt_v2_masks_manifest/1",
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "authority": "A9 §3 — masks regenerated for the new shape, NEVER reused; "
                        "geometry DISCOVERED from the encodes (DOSSIER §10.9 corrects "
                        "A9's (5,20,15) to the real (5,14,26))",
           "store": str(STORE), "generator": "assemble_root.ensure_mask (imported read-only)",
           "strata": {}, "masks": {}, "failures": []}
    fails: list[str] = []

    for s in strata:
        geo = discover_geometries(s, every=every)
        rec["strata"][s] = geo
        f, h, w = geo["geometry_fhw"]
        log(f"{s}: {geo['n_latents_read']} latents read -> geometry (F,H,W)=({f},{h},{w}) "
            f"= {geo['tokens']} tokens, fps={geo['fps_seen']}")
        for combo, info in geo["combos"].items():
            sided = combo.rsplit("_", 1)[-1].replace("sided", "")
            name = store_name(f, h, w, sided)
            dst = STORE / name

            if dst.exists() and not verify_only:
                if force:
                    log(f"{s}: --force, unlinking {name} before regenerating (never reuse)")
                    dst.unlink()
                else:
                    pre = verify_mask(dst, f, h, w, sided)
                    if pre:
                        fails.append(f"{name}: exists but is WRONG and --force was not given "
                                     f"-> would have been REUSED: {pre}")
                        continue
                    log(f"{s}: {name} already correct (pass --force to rewrite)")
            if not dst.exists():
                if verify_only:
                    fails.append(f"{name}: absent (--verify-only)")
                    continue
                ensure_mask(dst, f, h, w, sided)
                log(f"{s}: WROTE {name}")

            bad = verify_mask(dst, f, h, w, sided)
            if bad:
                fails.append(f"{name}: {bad}")
                continue

            m = torch.load(dst, map_location="cpu", weights_only=True)["mask"]

            # -- identity with the assembler's own generator, into a scratch dir ------------
            # Compared as TENSORS: `torch.save`'s container is not byte-deterministic (see
            # `content_sha256`), so a file-sha check here would pass or fail on wall-clock
            # alignment rather than on whether the mask is right.
            tmpd = Path(tempfile.mkdtemp(prefix="ctt_mask_ref_"))
            try:
                ref = mask_store_path(tmpd, f, h, w, sided)
                ensure_mask(ref, f, h, w, sided)
                rm = torch.load(ref, map_location="cpu", weights_only=True)["mask"]
                same = (rm.shape == m.shape and rm.dtype == m.dtype
                        and bool(torch.equal(rm, m)))
                if not same:
                    fails.append(f"{name}: NOT identical to assemble_root.ensure_mask "
                                 f"output — assembly would produce a different mask")
            finally:
                shutil.rmtree(tmpd, ignore_errors=True)

            rec["masks"][name] = {
                "stratum": s, "geometry_fhw": [f, h, w], "sided": sided,
                "prefix_latents": rc.prefix_latents((f, h, w)),
                "tokens": f * h * w, "numel": int(m.numel()),
                "n_conditioned_tokens": int(m.sum().item()),
                "cond_fraction": float(m.mean().item()),
                "sha256": sha256_file(dst),
                "content_sha256": content_sha256(m),
                "identical_to_assemble_root_ensure_mask": same,
                "n_latents_justifying": info["n"], "examples": info["examples"],
            }
            if not same:
                continue
            log(f"{s}: {name} OK — numel {m.numel()} == {f}*{h}*{w}, "
                f"{int(m.sum())} conditioned ({float(m.mean()):.4f}), prefix "
                f"{rc.prefix_latents((f, h, w))} latent frame(s), "
                f"tensor-identical to assemble_root.ensure_mask")

    # -- the impossible geometry must be absent -----------------------------------------
    bad_glob = sorted(STORE.glob(f"f{IMPOSSIBLE_S4_LATENT[0]}_h{IMPOSSIBLE_S4_LATENT[1]}_"
                                 f"w{IMPOSSIBLE_S4_LATENT[2]}_*sided.pt"))
    if bad_glob:
        fails.append(f"the store contains A9's IMPOSSIBLE S4 geometry "
                     f"{IMPOSSIBLE_S4_LATENT}: {[p.name for p in bad_glob]} — no VAE-legal "
                     f"bucket from 832x464 yields it (DOSSIER §10.9)")
    else:
        log(f"assert: A9's impossible S4 geometry f{IMPOSSIBLE_S4_LATENT[0]}_"
            f"h{IMPOSSIBLE_S4_LATENT[1]}_w{IMPOSSIBLE_S4_LATENT[2]} is ABSENT from the store")

    # -- no two geometries may collide on token count ------------------------------------
    by_tok: dict[int, list[str]] = {}
    for name, m in rec["masks"].items():
        by_tok.setdefault(m["tokens"], []).append(name)
    for tok, names in sorted(by_tok.items()):
        geos = {tuple(rec["masks"][n]["geometry_fhw"]) for n in names}
        if len(geos) > 1:
            fails.append(f"{tok} tokens is produced by more than one geometry {sorted(geos)} "
                         f"— a reshape would SILENTLY succeed with the wrong mask")
    log(f"assert: token counts {sorted(by_tok)} each map to exactly one geometry — "
        f"a cross-format mask can only fail LOUDLY")

    rec["failures"] = fails
    rec["ok"] = not fails
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(rec, indent=1) + "\n")
    log(f"manifest -> {MANIFEST}")
    if fails:
        log(f"FAILED ({len(fails)}):")
        for f_ in fails:
            log(f"   - {f_}")
        return 1
    log(f"ALL MASK CHECKS PASS — {len(rec['masks'])} mask(s) regenerated/verified")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strata", default="S4", help="comma list, or 'all-encoded'")
    ap.add_argument("--force", action="store_true", help="unlink before writing (never reuse)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--sample-geometry", action="store_true",
                    help="read only 8 latents per stratum instead of every one (fast)")
    args = ap.parse_args()

    if args.strata == "all-encoded":
        strata = [d.name for d in sorted(ENC.iterdir())
                  if (d / "latents").is_dir() and any((d / "latents").glob("*.pt"))]
    else:
        strata = [s.strip() for s in args.strata.split(",") if s.strip()]
    for s in strata:
        if s not in SIDEDNESS:
            raise SystemExit(f"unknown stratum {s!r}; known {sorted(SIDEDNESS)}")
    log(f"strata={strata} force={args.force} verify_only={args.verify_only}")
    return regenerate(strata, args.force, args.verify_only, every=not args.sample_geometry)


if __name__ == "__main__":
    sys.exit(main())
