"""Colour-sensor (OSC) frames: split a Bayer mosaic into R, G and B planes.

A one-shot-colour sensor records one colour per pixel in a repeating 2x2 tile
(RGGB, GRBG, GBRG or BGGR). The stacker's pipeline works on three colour planes,
so each sub is turned into R, G and B planes here, *after* the calibration steps
that must see the raw mosaic (dark, flat, hot pixels) and *before* registration.

Two methods:
  bilinear    full resolution; each missing colour is interpolated from its
              neighbours. Soft at the pixel level, but the stack recovers detail
              as the dither moves stars across the pattern.
  superpixel  half resolution; each 2x2 tile becomes one pixel (R, mean of the
              two G, B) with no interpolation at all. Faster, a quarter of the
              memory, and no interpolation colour artefacts.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

PATTERNS = ("RGGB", "GRBG", "GBRG", "BGGR")
METHODS = ("bilinear", "superpixel")
CHOICES = ("auto",) + METHODS
# 'auto' goes full resolution up to this many sensor pixels, half resolution above:
# a 24 MP DSLR frame at full resolution needs ~5 GB of memory and several seconds
# a frame, half resolution (6 MP of true colour) about a quarter of that.
AUTO_FULL_RES_MAX_PIXELS = 12_000_000


def choose_method(requested: str, shape: tuple[int, int]) -> str:
    """Resolve 'auto' to 'bilinear' or 'superpixel' for a mosaic of this size."""
    if requested != "auto":
        return requested
    return "bilinear" if shape[0] * shape[1] <= AUTO_FULL_RES_MAX_PIXELS else "superpixel"

# standard bilinear demosaic kernels (each already sums to 1 over the pixels of
# its colour that lie within it, for the Bayer layout)
_K_GREEN = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4.0
_K_REDBLUE = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4.0


def normalise_pattern(pattern: str | None) -> str | None:
    """'rggb' / ' RGGB ' -> 'RGGB'; None if it is not one of the four layouts."""
    p = (pattern or "").strip().upper()
    return p if p in PATTERNS else None


def shift_pattern(pattern: str, x_off: int = 0, y_off: int = 0) -> str:
    """The pattern seen after the sensor's origin moves by (x_off, y_off) pixels
    (FITS XBAYROFF / YBAYROFF, or a crop that starts on an odd row / column)."""
    tile = [pattern[0:2], pattern[2:4]]
    rows = [tile[(r + y_off) % 2] for r in range(2)]
    rows = [row[x_off % 2:] + row[:x_off % 2] for row in rows]
    return rows[0] + rows[1]


def _site_masks(shape: tuple[int, int], pattern: str) -> dict[str, np.ndarray]:
    masks = {c: np.zeros(shape, bool) for c in "RGB"}
    for i, colour in enumerate(pattern):
        masks[colour][i // 2::2, i % 2::2] = True
    return masks


def debayer(mosaic: np.ndarray, pattern: str, method: str = "bilinear") -> np.ndarray:
    """(H, W) mosaic -> (3, H, W) float32 planes in R, G, B order
    ((3, H/2, W/2) for 'superpixel')."""
    pattern = normalise_pattern(pattern)
    if pattern is None:
        raise ValueError("unknown Bayer pattern")
    if method not in METHODS:
        raise ValueError(f"unknown debayer method {method!r}")
    h, w = mosaic.shape
    mosaic = mosaic.astype(np.float32, copy=False)

    if method == "superpixel":
        h2, w2 = h // 2, w // 2
        quad = [mosaic[i // 2:h2 * 2:2, i % 2:w2 * 2:2] for i in range(4)]
        sites = {"R": [], "G": [], "B": []}
        for colour, plane in zip(pattern, quad):
            sites[colour].append(plane)
        return np.stack([np.mean(sites[c], axis=0) for c in "RGB"]).astype(np.float32)

    masks = _site_masks((h, w), pattern)
    out = np.empty((3, h, w), np.float32)
    for k, colour in enumerate("RGB"):
        sparse = np.where(masks[colour], mosaic, np.float32(0))
        kernel = _K_GREEN if colour == "G" else _K_REDBLUE
        # 'mirror' reflects about the edge pixel, so a reflected neighbour is the
        # same colour site as the one it stands in for
        convolve(sparse, kernel, output=out[k], mode="mirror")
    return out


def bayer_planes(mosaic: np.ndarray) -> list[np.ndarray]:
    """The four same-position sub-planes of a mosaic (views): positions (0,0),
    (0,1), (1,0), (1,1) of each 2x2 tile. Each has a single colour, so operations
    that compare a pixel with its neighbours can run on each separately."""
    return [mosaic[i // 2::2, i % 2::2] for i in range(4)]
