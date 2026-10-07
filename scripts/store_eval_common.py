"""Shared helpers for CPU store evals over stored features (numbered, append-only, meta.yaml + INDEX line)."""
from __future__ import annotations
import json, os, re, subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONDS_DIR = REPO_ROOT / "eval_ladder" / "conds"
POP_GRIDV3 = REPO_ROOT / "misc/2026-09-17_feature_store/population_gridv3.json"
# Frame-conditioned externals: one given frame per side (frame 0 of start9; on two-sided rows frame 8 of end9 = target frame 120).
EXTERNAL_ARMS = ("refvfx", "vap", "vfxmaster", "wan_flf2v")
# Clip-conditioned externals at 16 fps (TEG baselines, misc/2026-09-20_teg_baselines): 6 start + 4 end given frames, the
# 24-fps endpoints resampled to 16 fps (start9 idx 0,2,3,5,6,8 / end9 idx 4,5,7,8) and stored as conds_16fps/<endpoint>_{start6,end4}.mp4.
CLIP16_ARMS = ("wan_vace",)
CONDS16_DIR = REPO_ROOT / "misc/2026-09-20_teg_baselines/conds_16fps"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def parse_stem(stem: str) -> tuple[str, int]:
    item, s = stem.rsplit("__s", 1)
    return item, int(s)


def harness_arm_of(item_id: str) -> str:
    """The frozen harness-arm stamp is the 2nd '__' field of every grid-v3 item id (evals 038-040 use it)."""
    return item_id.split("__")[1]


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def grid_of(vdir: Path) -> dict[str, dict]:
    return {r["item_id"]: r for r in load_jsonl(vdir / "grid.jsonl")}


def arm_of(vdir: Path) -> str:
    """The store arm of a gens variant dir: store/gens/NNN_<arm>/KK_<variant>__<machine> -> <arm>."""
    return re.sub(r"^\d+_", "", Path(vdir).parent.name)


def grid_type(vdir: Path, arm_token: str = "") -> str:
    """``HF`` (121 f, 9-frame clip given) / ``ED`` (81 f, frame-0 anchor) / ``external`` (frame-conditioned prior works
    and the FLF2V baseline) / ``VACE16`` (the first-last CLIP baseline at 16 fps) — the tier that fixes the given windows."""
    arm = arm_of(vdir)
    if arm in CLIP16_ARMS or arm_token in CLIP16_ARMS:
        return "VACE16"
    if arm in EXTERNAL_ARMS or arm_token in EXTERNAL_ARMS:
        return "external"
    return "ED" if "ed81" in Path(vdir).name else "HF"


def windows(gtype: str, sided: str) -> tuple[int, int]:
    """(n_pre, n_suf): HF rows 9 given start frames (+8 end if two-sided); VACE16 rows 6 (+4 end if two-sided);
    ED rows and the frame-conditioned externals 1 frame (+1 end frame if two-sided — the TEG baselines)."""
    two = sided == "two"
    if gtype == "HF":
        return 9, (8 if two else 0)
    if gtype == "VACE16":
        return 6, (4 if two else 0)
    return 1, (1 if two else 0)


def cond_clips(gtype: str, endpoint: str) -> tuple[Path, Path]:
    """(start, end) condition clips whose frames ARE the given frames (same index arithmetic for every grid type):
    start9/end9 (24 fps) everywhere except VACE16, whose given frames are the 16-fps resamples start6/end4."""
    if gtype == "VACE16":
        return CONDS16_DIR / f"{endpoint}_start6.mp4", CONDS16_DIR / f"{endpoint}_end4.mp4"
    return CONDS_DIR / f"{endpoint}_start9.mp4", CONDS_DIR / f"{endpoint}_end9.mp4"


def atomic_write(p: Path, text: str) -> None:
    tmp = p.with_suffix(p.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, p)


def write_eval(eval_id: str, results: list[dict], *, created: str, instrument: str, definition: list[str], why: str,
               caveat: str, extra: dict | None = None, index_line: str | None = None) -> Path:
    draft = not eval_id[:3].isdigit()           # unnumbered id = DRAFT (owner rule 2026-09-18: register only when finalized)
    ed = REPO_ROOT / "store" / "evals" / ("_draft" if draft else "") / eval_id
    ed.mkdir(parents=True, exist_ok=True)
    for res in results:
        d = ed / res["harness_arm"]
        d.mkdir(exist_ok=True)
        atomic_write(d / "rows.jsonl", "".join(json.dumps(r) + "\n" for r in res["rows"]))
    L = [f"id: {eval_id}", f"seq: {int(eval_id.split('_', 1)[0]) if not draft else 'null  # DRAFT: unnumbered until finalized'}", f"shelf: {'evals/_draft' if draft else 'evals'}", f"created: '{created}'",
         "machine: dai (login CPU, numpy over stored features)", f"instrument: {instrument} @ {git_sha()}"]
    for k, v in (extra or {}).items():
        L.append(f"{k}: {json.dumps(v) if not isinstance(v, str) else v}")
    L.append("definition:")
    L += [f"  - {json.dumps(d)}" for d in definition]
    L += [f"why: {json.dumps(why)}", f"caveat: {json.dumps(caveat)}", "arms_scored:"]
    for res in results:
        c = res["coverage"]
        L += [f"  {res['harness_arm']}:", f"    gen: {res['gen']}", f"    rows: {c['n']}",
              "    coverage: {" + ", ".join(f"{k}: {v}" for k, v in c.items() if k != "n") + "}"]
    atomic_write(ed / "meta.yaml", "\n".join(L) + "\n")
    if index_line and not draft:
        index = REPO_ROOT / "store" / "INDEX.md"
        text = index.read_text()
        if f"`{eval_id}`" not in text:
            lines = text.splitlines()
            ev = next(i for i, ln in enumerate(lines) if ln.strip() == "## evals")
            nxt = next((i for i in range(ev + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
            ins = nxt
            while ins > ev + 1 and not lines[ins - 1].strip():
                ins -= 1
            lines.insert(ins, index_line)
            atomic_write(index, "\n".join(lines) + "\n")
    return ed
