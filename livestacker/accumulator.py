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

                # Sky read+shot noise: the running std of the dimmer pixels
                # (the bright fraction is stars / galaxy and must not define
                # "noise", especially in a crowded field).
                dim = self._mean <= np.percentile(self._mean[ready], 70.0)
                base = local[ready & dim]
                sky_noise = float(np.median(base)) if base.size else 0.0
                sigma = np.maximum(local, sky_noise)

                # Only clip pixels that are normally quiet. A trail / cosmic
                # lands on a flat sky pixel (running std ~ sky noise) and is a
                # huge outlier; a star / edge pixel swings every frame (seeing,
                # sub-pixel registration) so its running std sits well above
                # the sky — exclude it, or sigma-clip eats the PSF.
                if sky_noise > 0:
                    quiet = local < cfg.sigma_clip_quiet_ratio * sky_noise
                else:
                    quiet = np.ones_like(local, dtype=bool)
                reject = (ready & quiet & (sigma > 0)
                          & (dev > cfg.sigma_clip_kappa * sigma))
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

    def covered_mask(self, min_frac: float) -> np.ndarray | None:
        """Boolean mask: pixels covered by at least `min_frac` of this channel's
        frames (always at least one). Excludes the under-sampled dither fringe."""
        if self._count is None:
            return None
        need = max(1, int(round(min_frac * self.frames)))
        return self._count >= need

    @property
    def mean(self) -> np.ndarray | None:
        """Current running mean as float32, or None if nothing stacked yet."""
        if self._mean is None or self.frames == 0:
            return None
        return np.where(self._count > 0, self._mean, 0.0).astype(np.float32)
