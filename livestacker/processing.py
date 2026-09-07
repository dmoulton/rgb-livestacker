"""Background removal, white balance, channel combination and the linked stretch.

Everything here operates on the *current* accumulated stacks. The linked
stretch is cheap enough to re-run on every slider move; the background model
is heavier, so it is cached per channel and refreshed only every few frames
(`BackgroundModeler`). Nothing here is baked back into the raw accumulation.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, detect_threshold

from .config import StackConfig, StretchParams

log = logging.getLogger("livestacker.processing")


# -- background / gradient removal ---------------------------------------

def source_mask(lum: np.ndarray, cfg: StackConfig) -> np.ndarray:
    """Boolean mask of bright sources (stars + nebula), so they don't pull the
    background estimate up. Built once from the combined luminance."""
    try:
        thresh = detect_threshold(lum, n_sigma=cfg.source_mask_nsigma)
        seg = detect_sources(lum, thresh, n_pixels=5)
    except Exception:
        return np.zeros(lum.shape, dtype=bool)
    if seg is None:
        return np.zeros(lum.shape, dtype=bool)
    return binary_dilation(seg.data > 0, iterations=cfg.source_mask_dilate)


def estimate_background(mean: np.ndarray, exclude: np.ndarray,
                        cfg: StackConfig) -> np.ndarray:
    """A smooth 2-D background surface (photutils Background2D), sigma-clipped
    per mesh cell and with `exclude` pixels (border + sources) left out."""
    box = max(16, min(cfg.background_box, *mean.shape))
    bkg = Background2D(
        mean, box_size=box, filter_size=cfg.background_filter,
        sigma_clip=SigmaClip(sigma=3.0), bkg_estimator=MedianBackground(),
        mask=exclude, exclude_percentile=100.0, fill_value=0.0,
    )
    return bkg.background.astype(np.float32)


class BackgroundModeler:
    """Per-channel background cache. Recomputes a channel's model only once its
    frame count has advanced `background_every_n_frames`."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[int, np.ndarray]] = {}

    def get(self, name: str, mean: np.ndarray, exclude: np.ndarray,
            cfg: StackConfig, frames: int) -> np.ndarray:
        hit = self._cache.get(name)
        if hit is not None and frames - hit[0] < cfg.background_every_n_frames \
                and hit[1].shape == mean.shape:
            return hit[1]
        bg = estimate_background(mean, exclude, cfg)
        self._cache[name] = (frames, bg)
        return bg


# -- white balance ------------------------------------------------------

def _white_balance_scales(channels: dict[str, np.ndarray],
                          valid: np.ndarray) -> dict[str, float]:
    """One scalar per channel so bright stars read roughly neutral. Backgrounds
    are already ~0 post-subtraction, so this matches a high (star-core)
    percentile measured over the solidly-covered region."""
    highs = {}
    for name, img in channels.items():
        highs[name] = max(float(np.percentile(img[valid], 99.5)), 1e-6)
    if not highs:
        return {}
    target = float(np.median(list(highs.values())))
    return {name: target / h for name, h in highs.items()}


# -- combine ----------------------------------------------------------

def combine_rgb(
    channels: dict[str, tuple[np.ndarray, np.ndarray, int]],
    cfg: StackConfig,
    modeler: BackgroundModeler | None = None,
) -> np.ndarray | None:
    """Build a linear (H, W, 3) RGB array from per-channel
    `(mean, coverage, frame_count)`.

    The result is cropped to the bounding box of the region every present
    channel covers, so the ragged dither border never appears. Missing
    channels are filled from the mean of the present ones.
    """
    present = {k: v for k, v in channels.items()
               if v is not None and v[0] is not None}
    if not present:
        return None
    if modeler is None:
        modeler = BackgroundModeler()

    common = None
    for _, (_, cover, _) in present.items():
        common = cover if common is None else (common & cover)
    if cfg.display_border_erode > 0:
        common = binary_erosion(common, iterations=cfg.display_border_erode)
    ys, xs = np.where(common)
    if ys.size == 0:
        return None
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    keep = common[sl]

    means = {name: mean[sl].astype(np.float32)
             for name, (mean, _, _) in present.items()}
    frames = {name: n for name, (_, _, n) in present.items()}

    do_bg = cfg.background_enabled and all(
        n >= cfg.min_subs_for_background for n in frames.values())

    exclude = None
    bg_valid = keep
    if do_bg:
        bg_valid = binary_erosion(keep, iterations=cfg.background_border_erode)
        if bg_valid.sum() < 0.05 * keep.sum():
            do_bg = False
    if do_bg:
        lum = np.mean(list(means.values()), axis=0)
        smask = source_mask(lum, cfg)
        exclude = ~bg_valid | smask
        sky_ref = bg_valid & ~smask

    proc: dict[str, np.ndarray] = {}
    for name, img in means.items():
        if do_bg:
            img = img - modeler.get(name, img, exclude, cfg, frames[name])
            img = img - np.median(img[sky_ref])
        proc[name] = np.clip(img, 0.0, None)

    if cfg.auto_white_balance:
        for name, s in _white_balance_scales(proc, keep).items():
            proc[name] = proc[name] * s

    fallback = np.mean(np.stack(list(proc.values())), axis=0)
    rgb = np.stack([proc.get(c, fallback) for c in ("R", "G", "B")], axis=-1)
    rgb *= keep[..., None]

    norm = float(np.percentile(rgb.max(axis=-1)[keep], 99.5))
    if norm > 0:
        rgb = rgb / norm
    return np.clip(rgb, 0.0, None).astype(np.float32)


# -- linked display stretch --------------------------------------------

def _mtf(x: np.ndarray, m: float) -> np.ndarray:
    """PixInsight-style midtones transfer function. m=0.5 is identity."""
    m = float(np.clip(m, 1e-4, 1 - 1e-4))
    if abs(m - 0.5) < 1e-6:
        return x
    return ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m)


def auto_stretch(rgb: np.ndarray, target_bg: float = 0.20) -> StretchParams:
    """Pick a black point and midtones from image statistics (linked, so from
    the combined luminance) aiming for a given background brightness."""
    lum = rgb.mean(axis=-1)
    lum = lum[lum > 0]  # ignore the masked border
    if lum.size == 0:
        return StretchParams()
    med = float(np.median(lum))
    mad = float(np.median(np.abs(lum - med))) or 1e-6
    black = float(np.clip(med - 2.8 * mad, 0.0, 0.98))

    shifted = max((med - black) / (1.0 - black), 1e-6)
    # solve _mtf(shifted, m) = target_bg for m
    m = (target_bg - 1.0) * shifted / (
        (2.0 * target_bg - 1.0) * shifted - target_bg
    )
    midtones = float(np.clip(m, 0.01, 0.99))
    return StretchParams(black=black, midtones=midtones, gamma=1.0,
                         contrast=1.0, auto=True)


def apply_stretch(rgb: np.ndarray, p: StretchParams) -> np.ndarray:
    """Linked black/midtones/gamma/contrast -> uint8 (H, W, 3)."""
    x = (rgb - p.black) / max(1.0 - p.black, 1e-6)
    x = np.clip(x, 0.0, 1.0)
    x = _mtf(x, p.midtones)
    if abs(p.gamma - 1.0) > 1e-3:
        x = np.power(x, 1.0 / max(p.gamma, 1e-3))
    if abs(p.contrast - 1.0) > 1e-3:
        x = (x - 0.5) * p.contrast + 0.5
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)
