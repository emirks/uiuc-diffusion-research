from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


pytest.importorskip("diffusers")
pytest.importorskip("ftfy")
pytest.importorskip("imageio")
pytest.importorskip("imageio_ffmpeg")


def load_exp003_module():
    run_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "exp_003_wan21_flf2v_baseline"
        / "run.py"
    )
    spec = importlib.util.spec_from_file_location("exp003_run", run_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {run_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_preprocess_and_export_video(tmp_path: Path) -> None:
    exp003 = load_exp003_module()

    # Simulate reading/writing start-end images from disk.
    start = Image.new("RGB", (1080, 1080), color=(120, 40, 20))
    end = Image.new("RGB", (1920, 1080), color=(20, 80, 140))
    start_path = tmp_path / "start.png"
    end_path = tmp_path / "end.png"
    start.save(start_path)
    end.save(end_path)

    start_img = Image.open(start_path).convert("RGB")
    end_img = Image.open(end_path).convert("RGB")

    target_w, target_h = 852, 480
    start_proc = exp003.preprocess_image_for_target(start_img, target_w=target_w, target_h=target_h)
    end_proc = exp003.preprocess_image_for_target(end_img, target_w=target_w, target_h=target_h)

    assert start_proc.size == (target_w, target_h)
    assert end_proc.size == (target_w, target_h)

    # Save processed images as an I/O check.
    start_proc_path = tmp_path / "start_processed.png"
    end_proc_path = tmp_path / "end_processed.png"
    start_proc.save(start_proc_path)
    end_proc.save(end_proc_path)
    assert start_proc_path.exists() and start_proc_path.stat().st_size > 0
    assert end_proc_path.exists() and end_proc_path.stat().st_size > 0

    # Create tiny dummy frames and export a video through diffusers util.
    frames = []
    for i in range(6):
        arr = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        arr[..., 0] = (20 * i) % 255
        arr[..., 1] = 80
        arr[..., 2] = 160
        frames.append(arr)

    out_video = tmp_path / "sample.mp4"
    exp003.export_to_video(frames, str(out_video), fps=8)
    assert out_video.exists() and out_video.stat().st_size > 0


def test_run_dir_increment_logic(tmp_path: Path) -> None:
    exp003 = load_exp003_module()

    (tmp_path / "run_0001").mkdir()
    (tmp_path / "run_0003").mkdir()
    run_id, run_dir = exp003.next_run_dir(tmp_path)

    assert run_id == "run_0004"
    assert run_dir.exists()


def test_preflight_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exp003 = load_exp003_module()

    # Keep preflight outputs in test temp space.
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    exp003.preflight_dependency_check()
    exp003.preflight_function_check()
