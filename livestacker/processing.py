"""Background removal, white balance, channel combination and the linked stretch.

Everything here operates on the *current* accumulated stacks and is cheap
enough to re-run whenever a frame lands or a slider moves. Nothing here is
baked back into the raw accumulation.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import StackConfig, StretchParams


def subtract_background(mean: np.ndarray, sigma: float) -> np.ndarray:
    """Cheap gradient removal: subtract a large-radius Gaussian blur of the
    image from itself, then centre the residual background on zero."""
    bg = gaussian_filter(mean, sigma=sigma, mode="nearest")
    flat = mean - bg
    return flat - np.median(flat)


def _interior(img: np.ndarray, margin: float = 0.06) -> np.ndarray:
    """Drop the outer `margin` fraction so the alignment border and edge
    artefacts don't bias whole-frame statistics."""
    h, w = img.shape
    my, mx = int(h * margin), int(w * margin)
    return img[my:h - my, mx:w - mx]


def _white_balance_scales(channels: dict[str, np.ndarray]) -> dict[str, float]:
    """One scalar per channel so bright stars read roughly neutral. Backgrounds
    are already ~0 post-subtraction, so this matches a high (star-core)
    percentile, measured on the frame interior only."""
    highs = {}
    for name, img in channels.items():
        if img is None:
            continue
        highs[name] = max(float(np.percentile(_interior(img), 99.5)), 1e-6)
    if not highs:
        return {}
    target = float(np.median(list(highs.values())))
    return {name: target / h for name, h in highs.items()}


def combine_rgb(
    channels: dict[str, tuple[np.ndarray, np.ndarray]],
    cfg: StackConfig,
) -> np.ndarray | None:
    """Build a linear (H, W, 3) RGB array from per-channel `(mean, coverage)`.

    Missing channels are filled from the mean of the present ones so a preview
    still appears before every filter has landed. Pixels not covered by every
    present channel are set to black, hiding the ragged alignment border.
    """
    present = {k: v for k, v in channels.items()
               if v is not None and v[0] is not None}
    if not present:
        return None

    proc: dict[str, np.ndarray] = {}
    common = None
    for name, (mean, cover) in present.items():
        img = mean.astype(np.float32)
        if cfg.background_enabled:
            img = subtract_background(img, cfg.background_sigma)
        proc[name] = np.clip(img, 0.0, None)
        common = cover if common is None else (common & cover)

    if cfg.auto_white_balance:
        for name, s in _white_balance_scales(proc).items():
            proc[name] = proc[name] * s

    fallback = np.mean(np.stack(list(proc.values())), axis=0)
    rgb = np.stack([proc.get(c, fallback) for c in ("R", "G", "B")], axis=-1)

    if common is not None:
        rgb *= common[..., None]

    # Normalise to a stable 0..1 working range using a high percentile of the
    # interior so the stretch controls behave consistently as signal builds.
    norm = float(np.percentile(_interior(rgb.max(axis=-1)), 99.5))
    if norm > 0:
        rgb = rgb / norm
    return np.clip(rgb, 0.0, None).astype(np.float32)


def _mtf(x: np.ndarray, m: float) -> np.ndarray:
    """PixInsight-style midtones transfer function. m=0.5 is identity."""
    m = float(np.clip(m, 1e-4, 1 - 1e-4))
    if abs(m - 0.5) < 1e-6:
        return x
    return ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m)


def auto_stretch(rgb: np.ndarray, target_bg: float = 0.20) -> StretchParams:
    """Pick a black point and midtones from image statistics (linked, so from
    the combined luminance) aiming for a given background brightness."""
    lum = _interior(rgb.mean(axis=-1))
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
