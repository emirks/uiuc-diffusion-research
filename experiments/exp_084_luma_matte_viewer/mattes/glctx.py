"""Construct exp_075's `GLRunner` on a node where EGL device 0 is unusable.

On an HCESC node requested WITHOUT `--gres`, glcontext's EGL backend picks
device 0 via EGL_EXT_platform_device — an NVIDIA card the job has no permission
to open — and `eglInitialize` returns 0x3001. `LIBGL_ALWAYS_SOFTWARE` does not
help ("not allowed to force software rendering when API explicitly selects a
hardware device"). Mesa also exposes a swrast EGL device, so walking the device
list until one initialises is the fix. exp_083 solved this inside its own
renderer; exp_075's `GLRunner` predates it and must not be edited, so the walk
lives here instead.

Costs nothing where device 0 already works (login nodes, plain CPU nodes),
because that is tried first.
"""

from __future__ import annotations

import os


def make_runner(cls, width: int, height: int):
    last = None
    for idx in [None, *range(16)]:
        if idx is None:
            os.environ.pop("GLCONTEXT_DEVICE_INDEX", None)
        else:
            os.environ["GLCONTEXT_DEVICE_INDEX"] = str(idx)
        try:
            r = cls(width, height)
            r.egl_device_index = idx
            return r
        except Exception as e:                                    # noqa: BLE001
            last = e
    raise RuntimeError(f"no usable EGL device (last error: {last})")
