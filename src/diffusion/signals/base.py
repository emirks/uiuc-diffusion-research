"""Abstract base class shared by all signal extractors.

Every extractor follows the same contract so the CLI and experiments
can compose them without knowing model internals:

    extractor = MyExtractor(device="cuda")
    result    = extractor(frames)           # lazy-loads model on first call
    viz       = extractor.visualize(frames, result)

``frames`` is always a list of HxWx3 uint8 NumPy arrays (RGB order).
``result`` is a dict whose keys are signal-specific (e.g. "depth", "flow", "masks").
Each value is a list of NumPy arrays — one entry per output frame.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


import numpy as np


class BaseExtractor(ABC):
    """Lazy-loading signal extractor base class.

    Subclasses must define:
      - ``name``       : unique slug used as output folder name (e.g. "depth_dav2")
      - ``input_type`` : "image" (per-frame) | "video" (needs temporal context)
      - ``_load_model``: instantiate self._model (called once, on first use)
      - ``extract``    : core inference; receives a list of RGB uint8 frames
      - ``visualize``  : renders result back onto frames for inspection
    """

    name: str
    input_type: str  # "image" | "video"

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._model: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Ensure model weights are loaded.  Safe to call multiple times."""
        if self._model is None:
            self._load_model()

    def __call__(self, frames: list[np.ndarray]) -> dict:
        """Load model (if needed) then run extraction."""
        self.load()
        return self.extract(frames)

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_model(self) -> None: ...

    @abstractmethod
    def extract(self, frames: list[np.ndarray]) -> dict:
        """Run inference.

        Parameters
        ----------
        frames:
            List of H×W×3 uint8 RGB arrays.

        Returns
        -------
        dict
            Signal-specific keys mapping to lists of NumPy arrays.
        """
        ...

    @abstractmethod
    def visualize(self, frames: list[np.ndarray], result: dict) -> list[np.ndarray]:
        """Render result back onto (copies of) the source frames.

        Returns a list of H×W×3 uint8 RGB arrays, one per output frame.
        """
        ...
