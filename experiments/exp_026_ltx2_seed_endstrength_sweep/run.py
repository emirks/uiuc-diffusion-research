"""exp_026 — LTX-2 Seed × End-Clip Strength Sweep. PLACEHOLDER.

Not implemented. Fork exp_024/run.py when revisiting:
  - inner loop is cartesian over (seeds, end_clip_strengths)
  - positive prompt locked to ""
  - negative prompt locked to exp_025 winner (update config first)
  - output filename encodes seed and end_strength
"""
import sys


def main() -> None:
    raise NotImplementedError(
        "exp_026 is a placeholder. Implement by forking experiments/exp_024_ltx2_prompt_sweep/run.py."
    )


if __name__ == "__main__":
    sys.exit(main())
