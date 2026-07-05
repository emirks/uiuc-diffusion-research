#!/usr/bin/env python
"""exp_055 — build the EXAMPLE viewer bundle from the existing exp_053 results.

Thin preset wrapper around build_viewer.py: reads config.yaml for the source
paths and invokes the general builder. For a fresh eval run, call build_viewer.py
directly with --validation/--items/... (see its --help), or edit config.yaml.

    python run.py                 # builds outputs/eval/exp_053/viewer/
"""

import pathlib
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).parent
CONFIG_PATH = HERE / "config.yaml"

sys.path.insert(0, str(HERE))
import build_viewer  # noqa: E402


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    inp = cfg["inputs"]
    argv = [
        "--validation", inp["validation"],
        "--items", inp["items"],
        "--manifest", inp["manifest"],
        "--ceilings", inp["ceilings"],
        "--judge-summary", inp["judge_summary"],
        "--judge-results", inp["judge_results"],
        "--checks", inp["checks"],
        "--report", inp["report"],
        "--dedup", inp["dedup"],
        "--controls", inp["controls"],
        "--transitions-root", inp["transitions_root"],
        "--cache-dir", inp["cache_dir"],
        "--template", str(HERE / "viewer_template.html"),
        "--out", cfg["outputs"]["dir"],
        "--label", cfg["outputs"]["label"],
    ]
    for fd in inp.get("figures_dirs", []):
        argv += ["--figures-dir", fd]
    if inp.get("exclude"):
        argv += ["--exclude", *inp["exclude"]]
    sys.argv = ["build_viewer.py"] + argv
    build_viewer.main()


if __name__ == "__main__":
    main()
