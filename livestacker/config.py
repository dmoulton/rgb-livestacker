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
    # photutils Background2D: mesh box size (px) and the running-median filter
    # applied to the low-res background mesh.
    background_box: int = 128
    background_filter: int = 3
    # Shrink the common-coverage mask by this many px before estimating the
    # background, so the ragged dither border never leaks into the model.
    background_border_erode: int = 24
    # Trim this many px off the common-coverage edge before cropping the
    # preview, hiding the thin partial-coverage / amp-glow fringe.
    display_border_erode: int = 10
    # A pixel is only shown once it has been covered by at least this fraction
    # of a channel's frames. Without this, a strip that the dither pattern
    # pushed the field off early on keeps a frozen, poorly-sampled value while
    # the interior stacks deeper — showing up as a border that "grows" brighter
    # as the rest of the frame gets cleaner.
    coverage_min_frac: float = 0.6
    # Re-estimate a channel's background model only once its frame count has
    # advanced this far (the model changes slowly; the stretch does not).
    background_every_n_frames: int = 3
    # Bright-source rejection for the background estimate.
    source_mask_nsigma: float = 2.0
    source_mask_dilate: int = 12
    # Enable background / gradient removal.
    background_enabled: bool = True
    # Automatic per-channel white balance. The preferred method detects stars
    # and scales the channels so the median star is neutral (star colour tracks
    # filter/QE/extinction throughput, not the nebula's emission lines); it
    # falls back to bright-percentile matching when too few stars are found.
    auto_white_balance: bool = True
    wb_min_stars: int = 8          # need at least this many good stars
    wb_fwhm: float = 3.0          # DAOStarFinder PSF guess, px
    wb_nsigma: float = 5.0        # star detection threshold, sigma
    wb_max_stars: int = 300      # cap photometry cost
    # Per-frame hot/cold pixel cosmetic correction (no calibration frames
    # needed; this is what stops un-dark-subtracted hot pixels from smearing
    # into "walking noise" as the mount dithers).
    hot_pixel_enabled: bool = True
    hot_pixel_sigma: float = 8.0
    # Rolling per-pixel sigma-clip rejection in the running stack (rejects
    # satellite / aircraft trails and cosmics that land in one frame).
    sigma_clip_enabled: bool = True
    sigma_clip_kappa: float = 3.0
    # Samples a pixel needs before it is eligible for clipping.
    sigma_clip_warmup: int = 6
    # Only clip pixels whose running std is within this multiple of the sky
    # noise. A trail / cosmic lands on an otherwise-quiet sky pixel and stands
    # out; a star or edge pixel legitimately swings frame to frame (seeing,
    # sub-pixel registration) and must be left alone or the PSF gets eaten.
    sigma_clip_quiet_ratio: float = 2.5
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
    green: float = 0.0          # green-noise removal (SCNR), 0 = off .. 1 = full
    lum: float = 1.0            # luminance (L) blend when an L channel exists,
                                # 0 = RGB only .. 1 = full LRGB
    auto: bool = True           # let the pipeline pick black/midtones each update

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "StretchParams":
        fields = {f: data[f] for f in cls().to_json() if f in data}
        return replace(cls(), **fields)
