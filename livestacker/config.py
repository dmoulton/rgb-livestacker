"""Runtime configuration objects.

`StackConfig` holds pipeline tuning that rarely changes during a session.
`StretchParams` holds the display controls the browser edits live; it is the
payload passed around between the web UI and the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace


# Canonical channel order for the combined RGB array.
RGB_FILTERS = ("R", "G", "B")


@dataclass(frozen=True)
class StackConfig:
    # Minimum frames in a channel before its background model is trusted.
    min_subs_for_background: int = 3
    # Gaussian sigma (px) for the blur-based background estimate. Large vs star
    # size, small vs gradient scale.
    background_sigma: float = 60.0
    # Enable the cheap blur-subtract background removal.
    background_enabled: bool = True
    # Automatic per-channel white balance (match bright-star levels).
    auto_white_balance: bool = True
    # Per-frame hot/cold pixel cosmetic correction (no calibration frames
    # needed; this is what stops un-dark-subtracted hot pixels from smearing
    # into "walking noise" as the mount dithers).
    hot_pixel_enabled: bool = True
    hot_pixel_sigma: float = 8.0
    # Longest preview edge pushed over the socket, in pixels. Full-res stays
    # available through the /snapshot endpoint.
    preview_max_px: int = 1600
    # Don't re-encode the preview more often than this (seconds).
    min_preview_interval: float = 0.5
    # Discard a frame whose valid (aligned) area is below this fraction.
    min_valid_fraction: float = 0.5


@dataclass
class StretchParams:
    """Linked stretch: one curve applied identically to R, G and B."""

    black: float = 0.0          # black point, 0..1 of the linear range
    midtones: float = 0.5       # MTF midtones balance, 0..1 (0.5 = linear)
    gamma: float = 1.0          # extra power curve, applied after MTF
    contrast: float = 1.0       # pivot-at-0.5 contrast multiplier
    auto: bool = True           # let the pipeline pick black/midtones each update

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "StretchParams":
        fields = {f: data[f] for f in cls().to_json() if f in data}
        return replace(cls(), **fields)
