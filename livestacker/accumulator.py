"""Per-channel running stack with optional rolling sigma-clip rejection.

Per-pixel running statistics are kept with Welford's online algorithm (count,
mean, M2). Once a pixel has a few samples, an incoming value that sits more
than `kappa` robust-ish sigma from the running mean is dropped for that pixel
only — this is what rejects satellite/aircraft trails and cosmic-ray hits that
land in a single frame, which the per-frame hot-pixel pass (isolated pixels
only) cannot catch.

Rejection is online: it protects the stack from *future* outliers but cannot
retract an outlier that landed during the warm-up window. Per-pixel counts are
tracked so partially-overlapping (dithered / flipped) frames still contribute.
"""

from __future__ import annotations

import numpy as np

from .config import StackConfig


class ChannelStack:
    def __init__(self, name: str, cfg: StackConfig | None = None):
        self.name = name
        self.cfg = cfg or StackConfig()
        self._count: np.ndarray | None = None   # accepted samples per pixel
        self._mean: np.ndarray | None = None    # running mean (Welford)
        self._m2: np.ndarray | None = None      # running sum of sq. deviations
        self.frames = 0                          # frames offered to this channel
        self.rejected = 0                        # pixels dropped by sigma-clip
        self.last_path: str = ""

    def add(self, aligned: np.ndarray, valid: np.ndarray | None = None) -> None:
        """Add one aligned frame. `valid` marks usable pixels (default: finite)."""
        if valid is None:
            valid = np.isfinite(aligned)
        x = aligned.astype(np.float32, copy=False)

        if self._count is None:
            self._count = np.zeros(aligned.shape, dtype=np.int32)
            self._mean = np.zeros(aligned.shape, dtype=np.float32)
            self._m2 = np.zeros(aligned.shape, dtype=np.float32)

        accept = valid.copy()
        cfg = self.cfg
        if cfg.sigma_clip_enabled:
            ready = valid & (self._count >= cfg.sigma_clip_warmup)
            if ready.any():
                dev = np.abs(x - self._mean)
                with np.errstate(invalid="ignore", divide="ignore"):
                    local = np.sqrt(self._m2 / np.maximum(self._count - 1, 1))
                # Floor the per-pixel std at the frame's global sky noise, so a
                # pixel whose std is under-estimated from a handful of samples
                # doesn't reject normal scatter. Median over `ready` pixels is
                # robust to the small fraction that are stars.
                r = dev[ready]
                floor = 1.4826 * float(np.median(r)) if r.size else 0.0
                sigma = np.maximum(local, floor)
                reject = ready & (sigma > 0) & (dev > cfg.sigma_clip_kappa * sigma)
                n_rej = int(reject.sum())
                if n_rej:
                    self.rejected += n_rej
                    accept &= ~reject

        # Welford update, accepted pixels only
        idx = accept
        self._count[idx] += 1
        delta = x[idx] - self._mean[idx]
        self._mean[idx] += delta / self._count[idx]
        self._m2[idx] += delta * (x[idx] - self._mean[idx])
        self.frames += 1

    @property
    def coverage(self) -> np.ndarray | None:
        """Boolean mask: pixels that received at least one accepted sample."""
        if self._count is None:
            return None
        return self._count > 0

    @property
    def mean(self) -> np.ndarray | None:
        """Current running mean as float32, or None if nothing stacked yet."""
        if self._mean is None or self.frames == 0:
            return None
        return np.where(self._count > 0, self._mean, 0.0).astype(np.float32)
