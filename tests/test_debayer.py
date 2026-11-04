"""Bayer mosaic -> R, G, B planes."""

import numpy as np
import pytest

from livestacker.debayer import (PATTERNS, bayer_planes, debayer, normalise_pattern,
                                 shift_pattern)


def _mosaic(rgb, pattern):
    """Sample an (3, H, W) colour image through a Bayer pattern."""
    m = np.zeros(rgb.shape[1:], np.float32)
    for i, colour in enumerate(pattern):
        k = "RGB".index(colour)
        m[i // 2::2, i % 2::2] = rgb[k][i // 2::2, i % 2::2]
    return m


@pytest.mark.parametrize("pattern", PATTERNS)
def test_flat_colour_field_is_recovered_exactly_everywhere(pattern):
    level = {"R": 900.0, "G": 300.0, "B": 600.0}
    rgb = np.stack([np.full((20, 26), level[c], np.float32) for c in "RGB"])
    out = debayer(_mosaic(rgb, pattern), pattern)
    assert out.shape == (3, 20, 26)
    for k, c in enumerate("RGB"):
        assert np.allclose(out[k], level[c]), f"{pattern} {c} (edges included)"


@pytest.mark.parametrize("pattern", PATTERNS)
def test_smooth_gradient_is_reproduced_closely(pattern):
    yy, xx = np.mgrid[0:40, 0:48].astype(np.float32)
    rgb = np.stack([1000 + 10 * xx, 500 + 8 * yy, 300 + 5 * (xx + yy)])
    out = debayer(_mosaic(rgb, pattern), pattern)
    inner = (slice(2, -2), slice(2, -2))
    for k in range(3):
        assert np.abs(out[k][inner] - rgb[k][inner]).max() < 1e-2     # linear -> exact


@pytest.mark.parametrize("pattern", PATTERNS)
def test_known_sites_pass_through_unchanged(pattern):
    rng = np.random.default_rng(1)
    rgb = rng.uniform(100, 900, (3, 16, 18)).astype(np.float32)
    mosaic = _mosaic(rgb, pattern)
    out = debayer(mosaic, pattern)
    for i, colour in enumerate(pattern):
        k = "RGB".index(colour)
        assert np.allclose(out[k][i // 2::2, i % 2::2], mosaic[i // 2::2, i % 2::2])


@pytest.mark.parametrize("pattern", PATTERNS)
def test_superpixel_is_half_size_and_uses_no_interpolation(pattern):
    rng = np.random.default_rng(2)
    rgb = rng.uniform(100, 900, (3, 16, 18)).astype(np.float32)
    out = debayer(_mosaic(rgb, pattern), pattern, "superpixel")
    assert out.shape == (3, 8, 9)
    r = rgb[0][[i // 2 for i, c in enumerate(pattern) if c == "R"][0]::2,
               [i % 2 for i, c in enumerate(pattern) if c == "R"][0]::2]
    assert np.allclose(out[0], r)                                       # R is the R site itself
    g_sites = [i for i, c in enumerate(pattern) if c == "G"]
    g = np.mean([rgb[1][i // 2::2, i % 2::2] for i in g_sites], axis=0)
    assert np.allclose(out[1], g)                                       # G averages the two G sites


def test_odd_sized_mosaic_and_bad_input():
    assert debayer(np.ones((15, 17), np.float32), "RGGB", "superpixel").shape == (3, 7, 8)
    assert debayer(np.ones((15, 17), np.float32), "RGGB").shape == (3, 15, 17)
    with pytest.raises(ValueError):
        debayer(np.ones((4, 4), np.float32), "XXXX")
    with pytest.raises(ValueError):
        debayer(np.ones((4, 4), np.float32), "RGGB", "magic")


def test_pattern_names_and_origin_shifts():
    assert normalise_pattern(" rggb ") == "RGGB" and normalise_pattern("RGB") is None
    assert normalise_pattern(None) is None
    assert shift_pattern("RGGB", 0, 0) == "RGGB"
    assert shift_pattern("RGGB", 1, 0) == "GRBG"          # start one column in
    assert shift_pattern("RGGB", 0, 1) == "GBRG"          # start one row down
    assert shift_pattern("RGGB", 1, 1) == "BGGR"
    assert shift_pattern("GRBG", 2, 2) == "GRBG"          # whole tiles change nothing
    # a real mosaic cropped by (1, 1) behaves as the shifted pattern says
    rgb = np.random.default_rng(3).uniform(100, 900, (3, 12, 12)).astype(np.float32)
    cropped = _mosaic(rgb, "RGGB")[1:, 1:]
    out = debayer(cropped, shift_pattern("RGGB", 1, 1))
    assert np.allclose(out[0][1::2, 1::2], rgb[0][2::2, 2::2][:out[0][1::2, 1::2].shape[0], :out[0][1::2, 1::2].shape[1]])


def test_bayer_planes_are_single_colour_views():
    m = np.arange(36, dtype=np.float32).reshape(6, 6)
    planes = bayer_planes(m)
    assert [p.shape for p in planes] == [(3, 3)] * 4
    assert planes[3][0, 0] == m[1, 1] and planes[1][2, 1] == m[4, 3]
    planes[0][0, 0] = -1
    assert m[0, 0] == -1                                   # a view, not a copy
