"""Per-channel running stack.

Keeps a per-pixel sum and per-pixel count so that frames whose aligned area
only partially covers the reference grid (dither, flexure, flip) still
contribute where they have data. The result is the plain running mean; a
rolling sigma-clip can be layered on later without changing the interface.
"""

from __future__ import annotations

import numpy as np


class ChannelStack:
    def __init__(self, name: str):
        self.name = name
        self._sum: np.ndarray | None = None
        self._count: np.ndarray | None = None
        self.frames = 0
        self.last_path: str = ""

    def add(self, aligned: np.ndarray, valid: np.ndarray | None = None) -> None:
        """Add one aligned frame. `valid` marks usable pixels (default: finite)."""
        if valid is None:
            valid = np.isfinite(aligned)
        contrib = np.where(valid, aligned, 0.0).astype(np.float64)

        if self._sum is None:
            self._sum = np.zeros(aligned.shape, dtype=np.float64)
            self._count = np.zeros(aligned.shape, dtype=np.int32)
        self._sum += contrib
        self._count += valid
        self.frames += 1

    @property
    def coverage(self) -> np.ndarray | None:
        """Boolean mask: pixels that received at least one frame."""
        if self._count is None:
            return None
        return self._count > 0

    @property
    def mean(self) -> np.ndarray | None:
        """Current running mean as float32, or None if nothing stacked yet."""
        if self._sum is None or self.frames == 0:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            out = self._sum / np.maximum(self._count, 1)
        return out.astype(np.float32)
