"""Per-channel running stack with optional rolling sigma-clip rejection.

Per-pixel running statistics are kept with Welford's online algorithm (count,
mean, M2). Once a pixel has a few samples, an incoming value that sits more
than `kappa` robust-ish sigma from the running mean is dropped for that pixel
only — this is what rejects satellite/aircraft trails and cosmic-ray hits that
land in a single frame, which the per-frame hot-pixel pass (isolated pixels
only) cannot catch.

That online test needs a few samples before it can judge anything, so a trail in
one of the first subs would get in unchecked. For those, each pixel also
remembers its highest sample, and when the stack is read out that one sample is
taken back out wherever it stands far from all the others ("look-back"). It
removes one outlier per pixel, whenever it arrived.

Before either test, each sub's sky is brought to the stack's level: the sky
brightens and dims over a night, and left in, that drift looks like noise (which
blunts rejection) and leaves steps where dithered edges are covered by only some
of the subs. Per-pixel counts are tracked so partially-overlapping (dithered /
flipped) frames still contribute.
"""

from __future__ import annotations

import numpy as np

from .config import StackConfig

LOOKBACK_MIN_SAMPLES = 4      # a pixel needs this many before its top can be judged
# look-back judges the one *highest* sample of many, so it needs a stricter
# threshold than the online test to leave ordinary noise alone
LOOKBACK_EXTRA_KAPPA = 1.0


class ChannelStack:
    def __init__(self, name: str, cfg: StackConfig | None = None):
        self.name = name
        self.cfg = cfg or StackConfig()
        self._count: np.ndarray | None = None   # accepted samples per pixel
        self._seen: np.ndarray | None = None    # frames that landed on the pixel
        self._mean: np.ndarray | None = None    # running mean (Welford)
        self._m2: np.ndarray | None = None      # running sum of sq. deviations
        self._top: np.ndarray | None = None     # highest accepted sample (look-back)
        self.sky_offset = 0.0                    # taken off the latest sub to match
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
            self._seen = np.zeros(aligned.shape, dtype=np.int32)
            self._mean = np.zeros(aligned.shape, dtype=np.float32)
            self._m2 = np.zeros(aligned.shape, dtype=np.float32)
            self._top = np.full(aligned.shape, -np.inf, dtype=np.float32)
        elif self.cfg.sky_match_enabled:
            x = self._match_sky(x, valid)
        self._seen += valid

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
                # (estimated from a strided sample: it's a single global figure)
                s = (slice(None, None, 3), slice(None, None, 3))
                ready_s, mean_s, local_s = ready[s], self._mean[s], local[s]
                if ready_s.any():
                    dim = mean_s <= np.percentile(mean_s[ready_s], 70.0)
                    base = local_s[ready_s & dim]
                    sky_noise = float(np.median(base)) if base.size else 0.0
                else:
                    sky_noise = 0.0
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
        self._top[idx] = np.maximum(self._top[idx], x[idx])
        self.frames += 1

    def _match_sky(self, x: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """`x` shifted so its sky sits where the stack's does. Measured as the
        median difference *pixel for pixel* against the running mean, so it
        doesn't matter which part of a nebula each sub happens to cover (a
        sample of pixels is plenty for one number)."""
        s = (slice(None, None, 7), slice(None, None, 7))
        both = valid[s] & (self._count[s] > 0)
        if both.sum() < 100:
            return x
        self.sky_offset = float(np.median((x[s] - self._mean[s])[both]))
        return x - np.float32(self.sky_offset)

    @staticmethod
    def _without_top(n, mean, m2, top):
        """Mean and standard deviation of the samples other than the highest."""
        n = n.astype(np.float64)
        rest = (n * mean - top) / (n - 1)
        rest_m2 = np.maximum(m2 - (top - mean) * (top - rest), 0.0)
        return rest, np.sqrt(rest_m2 / (n - 2))

    def _look_back(self, out: np.ndarray) -> np.ndarray:
        """Take each pixel's highest sample back out of `out` where it stands far
        from all the others: a trail or cosmic ray that arrived before the online
        test could judge it."""
        cfg = self.cfg
        s = (slice(None, None, 3), slice(None, None, 3))
        enough = self._count[s] >= LOOKBACK_MIN_SAMPLES
        if not enough.any():
            return out
        n, mean = self._count[s][enough], self._mean[s][enough]
        _, spread = self._without_top(n, mean, self._m2[s][enough], self._top[s][enough])
        dim = mean <= np.percentile(mean, 70.0)           # sky, not stars
        noise = float(np.median(spread[dim])) if dim.any() else 0.0
        if noise <= 0:
            return out
        kappa = cfg.sigma_clip_kappa + LOOKBACK_EXTRA_KAPPA
        # only pixels whose top is well above their mean can qualify (the gap to
        # the mean of the rest is at most 4/3 of this): work on those alone
        with np.errstate(invalid="ignore"):
            maybe = np.nonzero((self._count >= LOOKBACK_MIN_SAMPLES)
                               & (self._top - self._mean > 0.7 * kappa * noise))
        if not maybe[0].size:
            return out
        top = self._top[maybe]
        rest, spread = self._without_top(self._count[maybe], self._mean[maybe],
                                         self._m2[maybe], top)
        # same rule as online: only normally-quiet pixels, never a star's flicker
        drop = ((spread < cfg.sigma_clip_quiet_ratio * noise)
                & (top - rest > kappa * np.maximum(spread, noise)))
        out[(maybe[0][drop], maybe[1][drop])] = rest[drop]
        return out

    @property
    def coverage(self) -> np.ndarray | None:
        """Boolean mask: pixels at least one frame landed on."""
        if self._seen is None:
            return None
        return self._seen > 0

    def covered_mask(self, min_frac: float) -> np.ndarray | None:
        """Boolean mask: pixels that at least `min_frac` of this channel's
        frames *landed on* — this counts frame footprint, not sigma-clip
        acceptance, so a noisy pixel whose samples keep getting rejected is
        still 'covered'. Excludes only the under-sampled dither fringe."""
        if self._seen is None:
            return None
        need = max(1, int(round(min_frac * self.frames)))
        return self._seen >= need

    @property
    def mean(self) -> np.ndarray | None:
        """The stack as float32 (running mean, with look-back rejection applied),
        or None if nothing stacked yet."""
        if self._mean is None or self.frames == 0:
            return None
        out = np.where(self._count > 0, self._mean, 0.0).astype(np.float32)
        return self._look_back(out) if self.cfg.sigma_clip_enabled else out
