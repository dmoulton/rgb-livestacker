"""Bias / dark / flat calibration chain."""

import numpy as np
import pytest
from astropy.io import fits

from livestacker.calibration import (
    Calibrator, MasterBias, MasterDark, build_flats,
)
from livestacker.config import StackConfig
from livestacker.fitsframe import load_frame

H, W = 60, 80


def _write(path, arr, exp, filt="R", typ="LIGHT"):
    hdr = fits.Header()
    hdr["EXPTIME"] = exp
    hdr["GAIN"] = 30
    hdr["XBINNING"] = 2
    hdr["CCD-TEMP"] = -10.0
    hdr["FILTER"] = filt
    hdr["IMAGETYP"] = typ
    fits.PrimaryHDU(arr.astype("float32"), hdr).writeto(path, overwrite=True)


@pytest.fixture
def frames(tmp_path):
    rng = np.random.default_rng(0)
    bias = 100.0 + rng.normal(0, 2, (H, W))
    dark20 = 40.0 + rng.normal(0, 1, (H, W))          # dark current per 20 s
    for name, sub in (("bias", 6), ("dark", 6), ("flat", 4)):
        (tmp_path / name).mkdir()
    for i in range(6):
        _write(tmp_path / "bias" / f"b{i}.fits",
               bias + rng.normal(0, 2, (H, W)), 0.1, typ="BIAS")
    for i in range(6):
        _write(tmp_path / "dark" / f"d{i}.fits",
               bias + dark20 + rng.normal(0, 2, (H, W)), 20.0, typ="DARK")
    for i in range(4):
        _write(tmp_path / "flat" / f"f{i}.fits",
               bias + 3000 * (1 - 0.15 * np.linspace(0, 1, W)[None, :])
               + rng.normal(0, 8, (H, W)), 2.0, typ="FLAT")
    light = bias + dark20 + 1000.0 + rng.normal(0, 4, (H, W))
    light[30, 40] += 20000.0
    _write(tmp_path / "light.fits", light, 20.0)
    return tmp_path


def test_bias_dark_flat_removes_pedestal(frames):
    bias = MasterBias.build(frames / "bias")
    dark = MasterDark.build(frames / "dark", bias)
    flats = build_flats(frames / "flat", bias)
    assert dark.debiased

    cal = Calibrator(StackConfig(hot_pixel_enabled=False), dark, flats, bias)
    out = cal.apply(load_frame(frames / "light.fits"))

    assert abs(np.median(out) - 1000.0) < 25      # bias + dark gone, flat kept scale
    assert abs(out[30, 40] - 21000.0) < 500       # star preserved


def test_debiased_dark_scales_correctly(frames, tmp_path):
    """A 20 s debiased dark scaled to a 40 s light should remove ~2x the dark
    current and nothing else (bias handled separately)."""
    bias = MasterBias.build(frames / "bias")
    dark = MasterDark.build(frames / "dark", bias)

    rng = np.random.default_rng(1)
    b = 100.0 + rng.normal(0, 2, (H, W))
    light40 = b + 80.0 + 1000.0 + rng.normal(0, 4, (H, W))    # 2x the 40/frame dark
    _write(tmp_path / "l40.fits", light40, 40.0)

    cal = Calibrator(StackConfig(hot_pixel_enabled=False), dark, None, bias)
    out = cal.apply(load_frame(tmp_path / "l40.fits"))
    assert abs(np.median(out) - 1000.0) < 30


def test_no_bias_still_works(frames):
    cal = Calibrator(StackConfig(hot_pixel_enabled=False),
                     MasterDark.build(frames / "dark"),
                     build_flats(frames / "flat"))
    out = cal.apply(load_frame(frames / "light.fits"))
    assert abs(np.median(out) - 1000.0) < 25


def test_median_combine_matches_full_median_and_rejects_size_mismatch(tmp_path):
    from livestacker.fitsframe import median_combine
    rng = np.random.default_rng(3)
    arrs = [rng.normal(100, 10, (H, W)).astype("float32") for _ in range(5)]
    paths = []
    for i, a in enumerate(arrs):
        _write(tmp_path / f"m{i}.fits", a, 1.0)
        paths.append(tmp_path / f"m{i}.fits")
    got = median_combine(paths, strip_rows=7)            # strips that don't divide H
    assert np.array_equal(got, np.median(np.stack(arrs), axis=0).astype("float32"))

    _write(tmp_path / "odd.fits", np.zeros((H + 1, W)), 1.0)
    with pytest.raises(ValueError, match="differ in size"):
        median_combine(paths + [tmp_path / "odd.fits"])
