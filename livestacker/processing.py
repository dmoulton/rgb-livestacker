"""Background removal, white balance, channel combination and the linked stretch.

Everything here operates on the *current* accumulated stacks. The linked
stretch is cheap enough to re-run on every slider move; the heavier scene
models (per-channel background surface, star-flux white balance) are cached
and refreshed only every few frames (`SceneModeler`). Nothing here is baked
back into the raw accumulation.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
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


# -- white balance ------------------------------------------------------

def _percentile_scales(channels: dict[str, np.ndarray],
                       valid: np.ndarray) -> dict[str, float]:
    """Fallback white balance: match a high (star-core) percentile per channel.
    Crude — the nebula's own bright pixels bias it — but robust with no stars."""
    highs = {}
    for name, img in channels.items():
        highs[name] = max(float(np.percentile(img[valid], 99.5)), 1e-6)
    if not highs:
        return {}
    target = float(np.median(list(highs.values())))
    return {name: target / h for name, h in highs.items()}


def star_flux_scales(channels: dict[str, np.ndarray], valid: np.ndarray,
                     cfg: StackConfig) -> dict[str, float] | None:
    """White balance from the ensemble of field stars: scale R/G/B so the
    *median* star has equal flux in all three. Star colour reflects the system
    throughput per filter (bandpass, QE, atmospheric extinction) — exactly what
    white balance should equalise — and is immune to the nebula's line
    emission. Returns None if fewer than `wb_min_stars` usable stars are found,
    so the caller can fall back.
    """
    if not all(c in channels for c in ("R", "G", "B")):
        return None

    lum = np.mean([channels[c] for c in ("R", "G", "B")], axis=0)
    _, med, std = sigma_clipped_stats(lum[valid], sigma=3.0)
    if not np.isfinite(std) or std <= 0:
        return None

    try:
        finder = DAOStarFinder(fwhm=cfg.wb_fwhm, threshold=cfg.wb_nsigma * std)
        tbl = finder(lum - med, mask=~valid)
    except Exception:
        return None
    if tbl is None or len(tbl) < cfg.wb_min_stars:
        return None

    def col(*names: str) -> np.ndarray:
        for n in names:                       # photutils renamed these in 3.0
            if n in tbl.colnames:
                return np.asarray(tbl[n], dtype=float)
        raise KeyError(names)

    # keep round, well-sampled, non-edge sources in a sane brightness band
    x = col("x_centroid", "xcentroid")
    y = col("y_centroid", "ycentroid")
    flux = col("flux")
    r1 = col("roundness1")
    r2 = col("roundness2")
    sharp = col("sharpness")
    h, w = lum.shape
    m = cfg.wb_fwhm * 3.0
    good = (
        (np.abs(r1) < 0.35) & (np.abs(r2) < 0.35)
        & (sharp > 0.3) & (sharp < 1.0)
        & (x > m) & (x < w - m) & (y > m) & (y < h - m)
    )
    if good.any():
        good &= flux < np.percentile(flux[good], 97)   # drop likely-saturated
        good &= flux > np.percentile(flux[good], 20)    # drop low-SNR
    if good.sum() < cfg.wb_min_stars:
        return None

    idx = np.where(good)[0]
    if idx.size > cfg.wb_max_stars:
        idx = idx[np.argsort(flux[idx])[-cfg.wb_max_stars:]]
    ap = CircularAperture(np.column_stack([x[idx], y[idx]]), r=2.5 * cfg.wb_fwhm)

    sums = {}
    for c in ("R", "G", "B"):
        phot = aperture_photometry(channels[c], ap)
        sums[c] = np.asarray(phot["aperture_sum"], dtype=float)

    ok = (sums["R"] > 0) & (sums["G"] > 0) & (sums["B"] > 0)
    if ok.sum() < cfg.wb_min_stars:
        return None
    lg = np.log(sums["G"][ok] / sums["R"][ok])
    lb = np.log(sums["B"][ok] / sums["R"][ok])

    def robust_median(a: np.ndarray) -> float:
        mid = np.median(a)
        mad = np.median(np.abs(a - mid)) or 1e-6
        keep = np.abs(a - mid) < 3.0 * 1.4826 * mad
        return float(np.median(a[keep])) if keep.any() else float(mid)

    s = {"R": 1.0, "G": float(np.exp(-robust_median(lg))),
         "B": float(np.exp(-robust_median(lb)))}
    g = (s["R"] * s["G"] * s["B"]) ** (1.0 / 3.0)   # keep overall brightness
    return {k: v / g for k, v in s.items()}


# -- scene model cache ------------------------------------------------

class SceneModeler:
    """Caches the heavy per-scene models (background surface, white balance),
    refreshing them only once the frame count has advanced
    `background_every_n_frames`."""

    def __init__(self) -> None:
        self._bg: dict[str, tuple[int, np.ndarray]] = {}
        self._wb: tuple[int, dict[str, float]] | None = None
        self._mask: tuple[tuple, np.ndarray] | None = None
        self.wb_method: str = "none"

    def source_mask(self, make_lum, cfg: StackConfig, frames: int,
                    key: tuple) -> np.ndarray:
        """The bright-source mask, rebuilt on the background cadence rather than
        for every new frame. `make_lum` is only called on a rebuild. `key` is
        the crop box, so a changed footprint never reuses a stale mask."""
        if self._mask is not None and self._mask[0][0] == key \
                and frames - self._mask[0][1] < cfg.background_every_n_frames:
            return self._mask[1]
        mask = source_mask(make_lum(), cfg)
        self._mask = ((key, frames), mask)
        return mask

    def background(self, name: str, mean: np.ndarray, exclude: np.ndarray,
                   cfg: StackConfig, frames: int) -> np.ndarray:
        hit = self._bg.get(name)
        if hit is not None and frames - hit[0] < cfg.background_every_n_frames \
                and hit[1].shape == mean.shape:
            return hit[1]
        bg = estimate_background(mean, exclude, cfg)
        self._bg[name] = (frames, bg)
        return bg

    def white_balance(self, channels: dict[str, np.ndarray], valid: np.ndarray,
                      cfg: StackConfig, frames: int) -> dict[str, float]:
        if self._wb is not None \
                and frames - self._wb[0] < cfg.background_every_n_frames:
            return self._wb[1]
        scales = star_flux_scales(channels, valid, cfg)
        self.wb_method = "stars"
        if scales is None:
            scales = _percentile_scales(channels, valid)
            self.wb_method = "percentile"
        self._wb = (frames, scales)
        log.info("white balance (%s): %s", self.wb_method,
                 " ".join(f"{k}x{v:.2f}" for k, v in sorted(scales.items())))
        return scales


# -- combine ----------------------------------------------------------

def combine_rgb(
    channels: dict[str, tuple[np.ndarray, np.ndarray, int]],
    cfg: StackConfig,
    modeler: SceneModeler | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build a linear (H, W, 3) RGB array from per-channel
    `(mean, coverage, frame_count)`, and — if a luminance (`L`) channel is
    present — a matching (H, W) luminance array on the same 0..1 scale.

    The result is cropped to the bounding box of the region every present
    colour channel covers, so the ragged dither border never appears. Missing
    colour channels are filled from the mean of the present ones. With only a
    luminance stack, the result is that stack as a grey image (R = G = B).
    """
    present = {k: v for k, v in channels.items()
               if v is not None and v[0] is not None}
    color = {k: v for k, v in present.items() if k in ("R", "G", "B")}
    # luminance only (no colour filters yet): show it as a monochrome image
    mono = not color and "L" in present
    if mono:
        color = {"L": present["L"]}
    if not color:
        return None, None
    if modeler is None:
        modeler = SceneModeler()

    common = None
    for _, (_, cover, _) in color.items():
        common = cover if common is None else (common & cover)
    if cfg.display_border_erode > 0:
        common = binary_erosion(common, iterations=cfg.display_border_erode)
    ys, xs = np.where(common)
    if ys.size == 0:
        return None, None
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    keep = common[sl]

    means = {name: mean[sl].astype(np.float32)
             for name, (mean, _, _) in color.items()}
    frames = {name: n for name, (_, _, n) in color.items()}

    do_bg = cfg.background_enabled and all(
        n >= cfg.min_subs_for_background for n in frames.values())

    exclude = None
    bg_valid = keep
    if do_bg:
        bg_valid = binary_erosion(keep, iterations=cfg.background_border_erode)
        if bg_valid.sum() < 0.05 * keep.sum():
            do_bg = False
    sky_ref = None
    if do_bg:
        smask = modeler.source_mask(
            lambda: np.mean(list(means.values()), axis=0), cfg,
            sum(frames.values()), (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
        exclude = ~bg_valid | smask
        sky_ref = bg_valid & ~smask

    def flatten(name: str, img: np.ndarray, n: int) -> np.ndarray:
        if do_bg:
            img = img - modeler.background(name, img, exclude, cfg, n)
            img = img - np.median(img[sky_ref])
        return np.clip(img, 0.0, None)

    proc = {name: flatten(name, img, frames[name]) for name, img in means.items()}

    if cfg.auto_white_balance and not mono:
        total = sum(frames.values())
        for name, s in modeler.white_balance(proc, keep, cfg, total).items():
            if name in proc:
                proc[name] = proc[name] * s

    fallback = np.mean(np.stack(list(proc.values())), axis=0)
    rgb = np.stack([proc.get(c, fallback) for c in ("R", "G", "B")], axis=-1)
    rgb *= keep[..., None]
    norm = float(np.percentile(rgb.max(axis=-1)[keep], 99.5)) or 1.0
    rgb = np.clip(rgb / norm, 0.0, None).astype(np.float32)

    lum = None
    lch = None if mono else present.get("L")
    if lch is not None and lch[2] >= cfg.min_subs_for_background:
        l_mean = lch[0][sl].astype(np.float32)
        l_cover = lch[1][sl] & keep
        l_flat = flatten("L", l_mean, lch[2])
        l_norm = float(np.percentile(l_flat[l_cover], 99.5)) if l_cover.any() else 0.0
        if l_norm > 0:
            lum = np.where(l_cover, np.clip(l_flat / l_norm, 0.0, None), np.nan)
            lum = lum.astype(np.float32)

    return rgb, lum


# -- luminance (LRGB) -------------------------------------------------

_LUMA = (0.30, 0.59, 0.11)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return _LUMA[0] * rgb[..., 0] + _LUMA[1] * rgb[..., 1] + _LUMA[2] * rgb[..., 2]


def lrgb_chroma(rgb: np.ndarray, smooth: float = 2.0, floor: float = 0.02) -> np.ndarray:
    """Per-pixel colour ratios (R,G,B over luma) of a smoothed copy of `rgb`, for
    colouring a luminance layer. The RGB stack is the noisy part of an LRGB
    image, so its colour is taken from a blurred copy — colour noise is far less
    visible than luminance noise. Where the signal is below `floor` the ratios
    fade to neutral grey instead of amplifying noise."""
    from scipy.ndimage import gaussian_filter
    sm = np.stack([gaussian_filter(rgb[..., i], smooth, truncate=3.0)
                   for i in range(3)], axis=-1)
    lr = _luma(sm)
    w = np.clip(lr / floor, 0.0, 1.0)[..., None]
    ratio = sm / np.maximum(lr, 1e-6)[..., None]
    return (w * ratio + (1.0 - w)).astype(np.float32)


def apply_luminance(rgb: np.ndarray, lum: np.ndarray, blend: float,
                    chroma: np.ndarray | None = None) -> np.ndarray:
    """LRGB: out = (1 - blend) * rgb + blend * chroma * L, so blend 0 is the RGB
    image and 1 is the L stack carrying the RGB colour. `lum` may be NaN where
    the L channel has no coverage — those pixels stay RGB. `chroma` is the
    (cached) output of `lrgb_chroma`; computed here if not given.
    """
    if chroma is None:
        chroma = lrgb_chroma(rgb)
    have = np.isfinite(lum)
    coloured = chroma * np.where(have, lum, 0.0)[..., None]
    out = (1.0 - blend) * rgb + blend * coloured
    out = np.where(have[..., None], out, rgb)
    return np.clip(out, 0.0, None).astype(np.float32)


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


def _scnr_green(x: np.ndarray, amount: float) -> np.ndarray:
    """Subtractive green-noise removal (SCNR, average-neutral). There are almost
    no truly green astronomical objects, so a green pixel above the R/B average
    is noise / a walking-colour artefact — pull it back toward that average."""
    g = x[..., 1]
    cap = 0.5 * (x[..., 0] + x[..., 2])
    x[..., 1] = g + amount * (np.minimum(g, cap) - g)
    return x


def apply_stretch(rgb: np.ndarray, p: StretchParams) -> np.ndarray:
    """Linked black/midtones/gamma/contrast (+ optional SCNR) -> uint8 (H,W,3)."""
    x = (rgb - p.black) / max(1.0 - p.black, 1e-6)
    x = np.clip(x, 0.0, 1.0)
    x = _mtf(x, p.midtones)
    if abs(p.gamma - 1.0) > 1e-3:
        x = np.power(x, 1.0 / max(p.gamma, 1e-3))
    if abs(p.contrast - 1.0) > 1e-3:
        x = (x - 0.5) * p.contrast + 0.5
    x = np.clip(x, 0.0, 1.0)
    if p.green > 1e-3:
        x = _scnr_green(x, min(p.green, 1.0))
    return (x * 255.0 + 0.5).astype(np.uint8)
