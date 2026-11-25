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


def test_object_name_is_guessed_from_the_first_light_header(tmp_path):
    from livestacker.fitsframe import guess_object
    (tmp_path / "flats").mkdir()
    (tmp_path / "night" / "LIGHT").mkdir(parents=True)

    def put(path, typ, obj):
        hdr = fits.Header()
        hdr["IMAGETYP"] = typ
        if obj:
            hdr["OBJECT"] = obj
        fits.PrimaryHDU(np.zeros((6, 8), "float32"), hdr).writeto(path)
    put(tmp_path / "flats" / "f.fits", "FLAT", "ignored flat target")
    put(tmp_path / "night" / "LIGHT" / "a.fits", "LIGHT", "IC 434")
    put(tmp_path / "night" / "LIGHT" / "b.fits", "LIGHT", "other")
    assert guess_object(tmp_path) == "IC 434"
    assert guess_object(tmp_path / "flats") == ""        # only a flat: nothing
    assert guess_object(tmp_path / "missing") == ""


# -- masters saved 0..1, darks without a baseline, and mixed libraries --------------

from livestacker.calibration import DarkLibrary, BiasLibrary, load_bias, load_dark

BASE, SCALE = 1900.0, 65535.0


def _put(path, data, exp, gain=30, offset=30, temp=-10.0, typ="DARK", unit=False):
    hdr = fits.Header()
    for k, v in (("EXPTIME", exp), ("GAIN", gain), ("OFFSET", offset), ("CCD-TEMP", temp),
                 ("IMAGETYP", typ), ("XBINNING", 1), ("FILTER", "G")):
        hdr[k] = v
    arr = np.asarray(data, np.float32) / (SCALE if unit else 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(arr.astype(np.float32), hdr).writeto(path, overwrite=True)
    return path


def _noisy(level, seed=0, shape=(H, W)):
    return level + np.random.default_rng(seed).normal(0, 1.5, shape)


def _light(path, exp, current, sky=450.0, **kw):
    return load_frame(_put(path, _noisy(BASE + current + sky, 5), exp, typ="LIGHT", **kw))


def _library(root, unit=True):
    """Like a real Siril library: 0..1 files; the 30 s dark has had its baseline
    removed (and is clipped at 0), the 60 s one still has it; other settings too."""
    lib = root / "lib"
    dc30 = np.clip(_noisy(30.0, 1), 0, None)
    dc30[10, 10] = 330.0                                                  # a hot pixel
    _put(lib / "030s_g30.fit", dc30, 30, unit=unit)
    _put(lib / "060s_g30.fit", _noisy(BASE + 60.0, 2), 60, unit=unit)
    _put(lib / "020s_g9.fit", _noisy(5.0, 3), 20, gain=9, offset=100, unit=unit)
    _put(lib / "030s_g30_other_size.fit", _noisy(BASE, 4, (H + 8, W)), 30, unit=unit)
    bias = _put(root / "bias_g30.fit", _noisy(BASE, 6), 0.04, typ="BIAS")
    return lib, bias


def _calibrate(frame, dark, bias):
    cal = Calibrator(StackConfig(hot_pixel_enabled=False), dark, None, bias)
    return cal.apply(frame), cal


def test_a_library_picks_the_matching_dark_and_converts_0_to_1_files(tmp_path):
    lib, bias_path = _library(tmp_path)
    bias = load_bias(bias_path)
    dark = load_dark(lib, bias)
    assert isinstance(dark, DarkLibrary) and len(dark.groups) == 4

    # 30 s: the baseline-free dark; the bias must not be taken off it a second time
    light = _light(tmp_path / "l30.fits", 30, current=30.0)
    light.data[10, 10] += 300.0
    out, cal = _calibrate(light, dark, bias)
    assert abs(np.median(out) - 450.0) < 4 and abs(out[10, 10] - 450.0) < 12
    assert "030s_g30.fit" in cal.applied["dark"] and cal.problems == []

    # 60 s: the dark that still holds the baseline
    out, cal = _calibrate(_light(tmp_path / "l60.fits", 60, current=60.0), dark, bias)
    assert abs(np.median(out) - 450.0) < 4
    assert "060s_g30.fit" in cal.applied["dark"] and cal.problems == []


def test_without_a_bias_the_baseline_free_dark_says_so(tmp_path):
    lib, _ = _library(tmp_path)
    dark = load_dark(lib)
    out, cal = _calibrate(_light(tmp_path / "l.fits", 30, current=30.0), dark, None)
    assert abs(np.median(out) - (BASE + 450.0)) < 4
    assert any("baseline already removed" in p for p in cal.problems)
    # ...while a dark that holds the baseline needs no bias at all
    out, cal = _calibrate(_light(tmp_path / "l60.fits", 60, current=60.0), dark, None)
    assert abs(np.median(out) - 450.0) < 4 and cal.problems == []


def test_a_dark_holding_the_baseline_is_never_scaled(tmp_path):
    _put(tmp_path / "d60.fit", _noisy(BASE + 60.0, 2), 60)
    dark = load_dark(tmp_path / "d60.fit")
    out, cal = _calibrate(_light(tmp_path / "l.fits", 20, current=20.0), dark, None)
    # scaling by 20/60 would take off only a third of the baseline (sky ~1,700)
    assert abs(np.median(out) - 410.0) < 6
    assert any("can't be scaled" in p for p in cal.problems)


def test_with_nothing_that_fits_no_dark_is_applied_and_it_says_so(tmp_path):
    lib, _ = _library(tmp_path)
    light = _light(tmp_path / "l.fits", 30, current=30.0, gain=56, offset=30)
    out, cal = _calibrate(light, load_dark(lib), None)
    assert np.array_equal(out, light.data) and cal.applied["dark"] is None
    assert any("NOT applied" in p and "gain 30" in p for p in cal.problems)
    assert not any(str(tmp_path) in p for p in cal.problems)       # names, not paths


def test_a_bias_that_does_not_fit_is_reported_not_silently_dropped(tmp_path):
    wrong = _put(tmp_path / "canon_offset.fit", _noisy(2048.0, 1, (H + 20, W + 30)), 0.01)
    light = _light(tmp_path / "l.fits", 30, current=30.0)
    out, cal = _calibrate(light, None, load_bias(wrong))
    assert np.array_equal(out, light.data) and cal.applied["bias"] is None
    assert any("canon_offset.fit" in p and "NOT applied" in p for p in cal.problems)


def test_raw_subs_in_one_folder_are_still_combined_into_one_master(tmp_path):
    for i in range(5):
        _put(tmp_path / "d" / f"d{i}.fits", _noisy(BASE + 40.0, i), 20,
             temp=-10.0 + 0.1 * (i % 2))                            # a little jitter
    dark = load_dark(tmp_path / "d")
    assert isinstance(dark, MasterDark) and "(5 frames)" in dark.source
    # two exposures in the same folder become a library of two combined masters
    for i in range(3):
        _put(tmp_path / "d" / f"long{i}.fits", _noisy(BASE + 80.0, i), 40)
    mixed = load_dark(tmp_path / "d")
    assert isinstance(mixed, DarkLibrary) and len(mixed.groups) == 2
    out, cal = _calibrate(_light(tmp_path / "l.fits", 40, current=80.0), mixed, None)
    assert abs(np.median(out) - 450.0) < 4 and "(3 frames)" in cal.applied["dark"]


def test_bias_library_picks_by_gain_and_serves_the_darks(tmp_path):
    _put(tmp_path / "b" / "g30.fit", _noisy(BASE, 1), 0.04, typ="BIAS")
    _put(tmp_path / "b" / "g9.fit", _noisy(400.0, 2), 0.2, gain=9, offset=100, typ="BIAS",
         unit=True)
    lib, _ = _library(tmp_path)
    bias = load_bias(tmp_path / "b")
    assert isinstance(bias, BiasLibrary)
    dark = load_dark(lib, bias)
    out, cal = _calibrate(_light(tmp_path / "l.fits", 60, current=60.0), dark, bias)
    assert abs(np.median(out) - 450.0) < 4 and "g30.fit" in cal.applied["bias"]


def test_status_reports_what_was_applied_and_the_problems(tmp_path):
    from livestacker.pipeline import Pipeline
    lib, _ = _library(tmp_path)
    cal = Calibrator(StackConfig(hot_pixel_enabled=False), load_dark(lib), None, None)
    p = Pipeline(StackConfig(), cal)
    assert "library of 4" in p.status()["calibration"]["dark"]       # before any light
    cal.apply(_light(tmp_path / "l.fits", 30, current=30.0))
    st = p.status()["calibration"]
    assert st["dark"] == "030s_g30.fit" and st["warnings"] == cal.problems != []
    assert str(tmp_path) not in str(st)
