"""LRGB blend: the L stack supplies the brightness, the (smoothed) RGB the colour."""

import numpy as np

from livestacker.processing import _luma, apply_luminance, lrgb_chroma


def _scene(h=64, w=64):
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 0.6, 0.3, 0.1      # orange-ish, bright
    rgb[:, : w // 4] = 0.0                                     # dark strip
    return rgb


def test_blend_zero_is_rgb_and_one_carries_lum():
    rgb = _scene()
    lum = np.full(rgb.shape[:2], 0.5, np.float32)
    assert np.allclose(apply_luminance(rgb, lum, 0.0), rgb, atol=1e-6)
    out = apply_luminance(rgb, lum, 1.0)
    bright = out[:, 40:]
    assert np.allclose(_luma(bright), 0.5, atol=0.02)          # brightness is L's
    r, g, b = bright[..., 0].mean(), bright[..., 1].mean(), bright[..., 2].mean()
    assert r > g > b                                            # colour is RGB's


def test_dark_areas_do_not_amplify_noise_into_colour():
    rng = np.random.default_rng(1)
    rgb = np.clip(rng.normal(0.003, 0.003, (64, 64, 3)), 0, None).astype(np.float32)
    lum = np.full((64, 64), 0.4, np.float32)
    out = apply_luminance(rgb, lum, 1.0)
    # near-black RGB -> neutral grey carrying L, not a speckle of random colours
    assert out.reshape(-1, 3).std(axis=0).max() < 0.02
    assert np.allclose(out.mean(axis=(0, 1)), 0.4, atol=0.05)


def test_uncovered_luminance_stays_rgb():
    rgb = _scene()
    lum = np.full(rgb.shape[:2], 0.5, np.float32)
    lum[:, 32:] = np.nan
    out = apply_luminance(rgb, lum, 1.0)
    assert np.array_equal(out[:, 32:], rgb[:, 32:])


def test_chroma_has_unit_luma():
    c = lrgb_chroma(_scene())
    assert np.allclose(_luma(c[:, 40:]), 1.0, atol=1e-3)
