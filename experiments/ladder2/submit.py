"""ladder2 — the submitter. One place that decides WHERE work runs, so the queue policy is
visible instead of scattered across shell history.

Policy (advisor ruling):
  * `secondary` is the workhorse: preemptible, 4 h cap, huge. Every ladder2 job is
    resumable (training resumes from checkpoints; generation skips existing outputs), so
    preemption costs minutes, not runs. Short walltimes backfill sooner — ask for what the
    job needs, never the cap.
  * The lab HCESC partitions are shared with teammates: never more than `HCESC_CAP` ladder2
    jobs there at once, reserved for what genuinely does not fit `secondary` (the 5 h
    generalist) or for rescuing something that has been stuck.

    python experiments/ladder2/submit.py train --models spec_color_rain,spec_shadow_smoke
    python experiments/ladder2/submit.py train --fleet
    python experiments/ladder2/submit.py gen --arms spec_color_rain --priority P0 --chunks 4
    python experiments/ladder2/submit.py status
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

SECONDARY = ["--partition=secondary", "--account=campusclusterusers", "--gres=gpu:H100:1"]
HCESC = {
    "h100": ["--partition=HCESC-H100-normal", "--account=hcesc-h100", "--gres=gpu:H100:1"],
    "h200": ["--partition=HCESC-H200-normal", "--account=hcesc-h200", "--gres=gpu:H200:1"],
    "l40s": ["--partition=HCESC-L40S-normal", "--account=hcesc-l40s", "--gres=gpu:L40S:1"],
}
HCESC_CAP = 4  # politeness default: teammates share these nodes (--hcesc-cap to burst)

SPECIALIST_TIME = "02:30:00"   # 2000 steps ~= 1.5-2 h + inline validation
GENERALIST_TIME = "06:00:00"   # 5000 steps ~= 4 h 45 m measured -> HCESC only (secondary caps at 4 h)

MODEL_LOAD_MIN = 12            # measured: transformer + VAE + Gemma load before the first sample
MIN_PER_GEN = 3                # measured: 480x640x121 @ 30 steps on an H100, with headroom


def gen_walltime(rows_per_chunk: int) -> str:
    """Ask for what the chunk needs, not the queue cap — short jobs backfill sooner, and on a
    saturated cluster backfill is the only thing that actually starts a job."""
    minutes = MODEL_LOAD_MIN + MIN_PER_GEN * rows_per_chunk
    minutes = max(40, min(minutes, 230))          # stay under secondary's 4 h cap
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def sh(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    print("  " + " ".join(cmd[:3]) + " ... -> " + out)
    return out


def hcesc_in_flight() -> int:
    q = subprocess.run(["squeue", "-u", subprocess.run(["whoami"], capture_output=True, text=True)
                        .stdout.strip(), "-h", "-o", "%P"], capture_output=True, text=True).stdout
    return sum(1 for line in q.splitlines() if line.startswith("HCESC"))


def models() -> list[str]:
    return json.loads((HERE / "train/configs/index.json").read_text())


def submit_train(names: list[str], where: str, cap: int = HCESC_CAP) -> None:
    for name in names:
        cfg = HERE / f"train/configs/{name}.yaml"
        assert cfg.exists(), f"no config for {name} (run train/make_configs.py)"
        generalist = name == "ic_gen"
        place = where
        if place == "auto":
            place = "h200" if generalist else "secondary"
        if place != "secondary" and hcesc_in_flight() >= cap:
            print(f"[submit] HCESC cap {cap} reached — {name} goes to secondary instead")
            place = "secondary"
        if generalist and place == "secondary":
            sys.exit("[submit] ic_gen needs >4 h; it does not fit secondary. Free an HCESC slot.")
        target = SECONDARY if place == "secondary" else HCESC[place]
        sh(["sbatch", f"--job-name=ladder2_{name}", *target,
            f"--time={GENERALIST_TIME if generalist else SPECIALIST_TIME}",
            f"--export=ALL,MODEL={name}", str(HERE / "train/job_train.sbatch")])


def submit_gen(arms: list[str], seeds: list[str], priority: str | None, chunks: int,
               where: str, cells: str | None = None) -> None:
    rows = [json.loads(x) for x in (HERE / "registry.jsonl").read_text().splitlines() if x.strip()]
    for arm in arms:
        n = sum(1 for r in rows if r["arm"] == arm
                and (priority is None or r["priority"] in priority.split(","))
                and (cells is None or r["cell"] in cells.split(",")))
        if not n:
            print(f"[submit] {arm}: no rows for priority={priority} — skipped")
            continue
        k = min(chunks, n)
        target = SECONDARY if where == "secondary" else HCESC[where]
        per_chunk = -(-n // k)
        walltime = gen_walltime(per_chunk)
        # OWNER RULE: the DAVIS/foreign lane is additive, never load-bearing. Submit it with a
        # large `nice` so it can only ever consume slots nothing else wants — it must not delay
        # ic_gen, the base arm, or any specialist generation.
        foreign = (priority == "P2") or (cells is not None and "foreign" in cells)
        nice = ["--nice=5000"] if foreign else []
        for seed in seeds:
            sh(["sbatch", f"--job-name=ladder2_gen_{arm}_s{seed}", f"--array=0-{k - 1}", *target,
                *nice, f"--time={walltime}",
                f"--export=ALL,ARM={arm},SEED={seed},NCHUNKS={k}"
                + (f",PRIORITY={priority}" if priority else "")
                # pipe-separated: a comma here would be eaten by --export itself
                + (f",CELLS={cells.replace(',', '|')}" if cells else ""),
                str(HERE / "job_gen.sbatch")])
        print(f"[submit] {arm}: {n} rows x {len(seeds)} seeds over {k} chunks "
              f"(~{per_chunk}/chunk, walltime {walltime})")


def status() -> None:
    subprocess.run(["squeue", "-u", subprocess.run(["whoami"], capture_output=True, text=True)
                    .stdout.strip(), "-o", "%.12i %.26j %.20P %.8T %.10M %R"])
    out_train = REPO_ROOT / "outputs/training/ladder2"
    if out_train.exists():
        print("\ntraining progress:")
        for d in sorted(out_train.iterdir()):
            ckpts = sorted((d / "checkpoints").glob("lora_weights_step_*.safetensors")) \
                if (d / "checkpoints").exists() else []
            last = ckpts[-1].stem.split("_")[-1] if ckpts else "-"
            print(f"  {d.name:26s} {len(ckpts):3d} ckpts, last step {last}"
                  f"{'  DONE' if (d / 'DONE').exists() else ''}")
    gens = REPO_ROOT / "outputs/videos/ladder2"
    if gens.exists():
        print("\ngenerations:")
        for d in sorted(p for p in gens.iterdir() if p.is_dir() and not p.name.startswith("_")):
            print(f"  {d.name:26s} {len(list(d.glob('*.mp4'))):4d} videos")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--models", default=None)
    t.add_argument("--fleet", action="store_true", help="everything not already trained")
    t.add_argument("--where", default="auto", choices=["auto", "secondary", *HCESC])
    t.add_argument("--hcesc-cap", type=int, default=HCESC_CAP,
                   help="max concurrent ladder2 jobs on the shared lab partitions")
    g = sub.add_parser("gen")
    g.add_argument("--arms", required=True)
    g.add_argument("--seeds", default="42,43")
    g.add_argument("--priority", default=None)
    g.add_argument("--cells", default=None, help="comma-separated registry cells")
    g.add_argument("--chunks", type=int, default=4)
    g.add_argument("--where", default="secondary", choices=["secondary", *HCESC])
    sub.add_parser("status")
    args = ap.parse_args()

    if args.cmd == "train":
        if args.fleet:
            done = {d.name for d in (REPO_ROOT / "outputs/training/ladder2").glob("*/")
                    if (d / "DONE").exists()}
            names = [m for m in models() if m not in done]
        else:
            names = args.models.split(",")
        submit_train(names, args.where, args.hcesc_cap)
    elif args.cmd == "gen":
        submit_gen(args.arms.split(","), args.seeds.split(","), args.priority, args.chunks,
                   args.where, args.cells)
    else:
        status()


if __name__ == "__main__":
    main()
