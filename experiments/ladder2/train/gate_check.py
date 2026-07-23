"""ladder2 — the pilot gate, as a command instead of a vibe.

The fleet is deliberately held ~2-3 h behind a 2-model pilot because the leak-free training
prompt is the one genuinely new variable in this campaign: the outcome text is gone, so the
token has to carry the trigger by itself. If that fails systematically it fails for all 12
models, and the pilot is what makes that cost 2 hours instead of a full rerun.

Pre-committed observables (checked at step 500, extendable to 750 if ambiguous):

  1. cond_clean smoke assert logged      -> the isolation-encoded suffix anchor really is in use
  2. loss descending, no NaN             -> the run is healthy at all
  3. ID sample shows the class effect    -> the token became the trigger  <-- the new variable
  4. control sample shows no drift       -> the LoRA leaves unrelated generation alone

ABORT RULE: no in-distribution effect onset by step 1000 -> stop the fleet, take the
filmstrips to the advisor. Reverting to leaky prompts is not an available fix — it would
reintroduce exactly the defect this campaign exists to remove.

    python experiments/ladder2/train/gate_check.py spec_color_rain spec_shadow_smoke
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TRAIN_OUT = REPO_ROOT / "outputs/training/ladder2"
LOGS = REPO_ROOT / "outputs/logs/slurm"
STRIPS = HERE.parent / "assets/gate"
FFMPEG = os.path.expanduser("~/.local/bin/ffmpeg")

#: validation sample order emitted by make_configs.py
SAMPLE_ROLE = {1: "ID (trained class)", 2: "OOD (other class)", 3: "control (no token/cond)",
               4: "control (no token/cond)"}


def latest_log(model: str) -> Path | None:
    cands = sorted(LOGS.glob(f"ladder2_{model}-*.out"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def filmstrip(video: Path, out: Path, tiles: int = 8) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return True
    r = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-vf", f"select='not(mod(n\\,{max(1, 121 // tiles)}))',scale=120:-1,tile={tiles}x1",
         "-frames:v", "1", "-update", "1", str(out)], capture_output=True)
    return r.returncode == 0


def check(model: str) -> dict:
    out_dir = TRAIN_OUT / model
    result = {"model": model, "exists": out_dir.exists()}
    if not out_dir.exists():
        print(f"\n=== {model}: not started (no {out_dir.relative_to(REPO_ROOT)})")
        return result

    ckpts = sorted((out_dir / "checkpoints").glob("lora_weights_step_*.safetensors"))
    steps = [int(re.search(r"step_(\d+)", c.name).group(1)) for c in ckpts]
    result["last_step"] = max(steps) if steps else 0
    result["done"] = (out_dir / "DONE").exists()

    log = latest_log(model)
    smoke, losses, nans = None, [], 0
    if log:
        text = log.read_text(errors="ignore")
        m = re.search(r"\[smoke\][^\n]*", text)
        smoke = m.group(0) if m else None
        losses = [float(x) for x in re.findall(r"loss[=: ]+([0-9]*\.?[0-9]+)", text)][-400:]
        nans = len(re.findall(r"\bnan\b", text, flags=re.I))
    result.update(smoke=smoke, n_loss=len(losses), nans=nans)

    print(f"\n=== {model}  (step {result['last_step']}{' · DONE' if result['done'] else ''})")
    print(f"  1. smoke assert : {smoke or 'NOT FOUND IN LOG'}")
    if losses:
        head, tail = losses[: max(1, len(losses) // 5)], losses[-max(1, len(losses) // 5):]
        trend = sum(tail) / len(tail) - sum(head) / len(head)
        print(f"  2. loss         : first-fifth {sum(head)/len(head):.4f} -> last-fifth "
              f"{sum(tail)/len(tail):.4f}  ({'descending' if trend < 0 else 'NOT DESCENDING'})"
              f"{'  NaN SEEN' if nans else ''}")
    else:
        print("  2. loss         : no loss lines parsed yet")

    samples = sorted((out_dir / "samples").glob("step_*.mp4"))
    if not samples:
        print("  3/4. validation : no inline samples yet")
        return result
    by_step: dict[int, list[Path]] = {}
    for s in samples:
        by_step.setdefault(int(s.stem.split("_")[1]), []).append(s)
    step = max(by_step)
    print(f"  3/4. validation : step {step:05d} -> filmstrips in "
          f"{STRIPS.relative_to(REPO_ROOT)}/{model}/")
    for s in sorted(by_step[step]):
        idx = int(s.stem.rsplit("_", 1)[1])
        ok = filmstrip(s, STRIPS / model / f"{s.stem}.png")
        print(f"       sample {idx} {SAMPLE_ROLE.get(idx, '?'):26s} "
              f"{'strip ok' if ok else 'STRIP FAILED'}  {s.relative_to(REPO_ROOT)}")
    result["val_step"] = step
    return result


def main() -> None:
    models = sys.argv[1:] or ["spec_shadow_smoke", "spec_color_rain"]
    results = [check(m) for m in models]
    print("\n--- gate summary")
    for r in results:
        if not r.get("exists"):
            print(f"  {r['model']:24s} NOT STARTED")
            continue
        flags = []
        if not r.get("smoke"):
            flags.append("no smoke assert")
        if r.get("nans"):
            flags.append("NaN in log")
        if r.get("last_step", 0) < 500:
            flags.append("below step 500")
        print(f"  {r['model']:24s} step {r.get('last_step', 0):5d}  "
              f"{'; '.join(flags) if flags else 'observables 1-2 OK — now LOOK at the filmstrips'}")
    print("\nGate passes only if the ID filmstrip shows the class effect onsetting and the "
          "control shows none.\nAbort rule: no ID onset by step 1000 -> stop, do not revert to "
          "leaky prompts.")


if __name__ == "__main__":
    main()
