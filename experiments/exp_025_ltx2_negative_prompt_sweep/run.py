"""exp_025 — LTX-2 Negative Prompt Sweep. PLACEHOLDER.

Not implemented. Fork exp_024/run.py when revisiting:
  - inner loop is over `inputs.negative_prompt_variants` (not `prompt_variants`)
  - positive prompt is locked to "" for every run
  - output filename encodes the negative variant key
"""
import sys


def main() -> None:
    raise NotImplementedError(
        "exp_025 is a placeholder. Implement by forking experiments/exp_024_ltx2_prompt_sweep/run.py."
    )


if __name__ == "__main__":
    sys.exit(main())
