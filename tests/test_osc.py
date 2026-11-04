"""One-shot-colour (OSC) support: Bayer FITS, Canon raw, Bayer-aware calibration."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from livestacker.calibration import (Calibrator, MasterFlat, build_flats,
                                     cosmetic_correct, cosmetic_correct_bayer)
from livestacker.config import StackConfig, StretchParams
from livestacker.debayer import choose_method
from livestacker.fitsframe import (Frame, load_frame, median_combine, read_header)
from livestacker.pipeline import Pipeline
from livestacker.rawframe import kind_from_path
from livestacker.source import _is_fits

CFG = dict(background_enabled=False, auto_white_balance=False,
           sigma_clip_enabled=False, hot_pixel_enabled=False)
STARS = [(30, 40), (70, 110), (95, 55), (45, 130), (60, 80), (20, 100), (105, 140),
         (85, 20), (15, 15), (110, 90), (50, 60), (35, 145)]
SHAPE = (120, 160)          # even, so the tiles line up


def _scene(rng):
    """(3, H, W) colour scene: sky plus coloured stars (first red, second blue)."""
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    colours = [(3.0, 1.0, 0.3), (0.3, 1.0, 3.0)] + [(1.0, 1.0, 1.0)] * (len(STARS) - 2)
    rgb = np.full((3,) + SHAPE, 200.0, np.float32)
    for (cy, cx), col in zip(STARS, colours):
        blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / 3.0)).astype(np.float32)
        for k in range(3):
            rgb[k] += 4000 * col[k] * blob
    return rgb


def _mosaic(rgb, pattern, rng, noise=3.0):
    m = np.zeros(rgb.shape[1:], np.float32)
    for i, colour in enumerate(pattern):
        m[i // 2::2, i % 2::2] = rgb[("RGB".index(colour))][i // 2::2, i % 2::2]
    return (m + rng.normal(0, noise, m.shape)).astype(np.float32)


def _write(path, mosaic, pattern, **extra):
    hdr = fits.Header()
    hdr["BAYERPAT"] = pattern
    hdr["IMAGETYP"] = "Light"
    hdr["EXPTIME"] = 10.0
    for k, v in extra.items():
        hdr[k] = v
    fits.PrimaryHDU(mosaic, hdr).writeto(path, overwrite=True)


@pytest.mark.parametrize("pattern", ["RGGB", "GRBG", "GBRG", "BGGR"])
def test_bayer_fits_session_stacks_as_three_colour_channels(tmp_path, pattern):
    rng = np.random.default_rng(0)
    scene = _scene(rng)
    p = Pipeline(StackConfig(**CFG, osc_debayer="bilinear"))
    for i in range(4):
        path = tmp_path / f"s{i}.fits"
        _write(path, _mosaic(scene, pattern, rng), pattern)
        frame = load_frame(path)
        assert frame.bayer == pattern and frame.data.shape == SHAPE
        assert p.ingest(frame).ok
    st = p.status()
    assert {k: v["frames"] for k, v in st["channels"].items()} == {"R": 4, "G": 4, "B": 4}
    assert st["total_frames"] == 4                  # subs, not channel-frames
    assert st["reference_filter"] == "RGB"
    rgb = p._linear_rgb
    # the red star is red and the blue star is blue — in every layout
    (ry, rx), (by, bx) = STARS[0], STARS[1]
    off = (SHAPE[0] - rgb.shape[0]) // 2, (SHAPE[1] - rgb.shape[1]) // 2
    red = rgb[ry - off[0], rx - off[1]]
    blue = rgb[by - off[0], bx - off[1]]
    assert red[0] > 2 * red[2] and blue[2] > 2 * blue[0]


def test_superpixel_halves_the_stack_and_auto_picks_by_size(tmp_path):
    rng = np.random.default_rng(1)
    scene = _scene(rng)
    p = Pipeline(StackConfig(**CFG, osc_debayer="superpixel"))
    for i in range(3):
        path = tmp_path / f"s{i}.fits"
        _write(path, _mosaic(scene, "RGGB", rng), "RGGB")
        assert p.ingest(load_frame(path)).ok
    assert tuple(p.status()["size"][::-1]) == (SHAPE[0] // 2, SHAPE[1] // 2) or \
        p.status()["size"][1] <= SHAPE[0] // 2
    assert choose_method("auto", (2000, 3000)) == "bilinear"           # 6 MP
    assert choose_method("auto", (4022, 6024)) == "superpixel"         # 24 MP
    assert choose_method("bilinear", (4022, 6024)) == "bilinear"       # explicit wins


def test_bayer_offset_keywords_shift_the_pattern(tmp_path):
    rng = np.random.default_rng(2)
    path = tmp_path / "off.fits"
    _write(path, _mosaic(_scene(rng), "RGGB", rng), "RGGB", XBAYROFF=1, YBAYROFF=0)
    assert load_frame(path).bayer == "GRBG"
    _write(path, np.zeros(SHAPE, np.float32), "xyz")
    assert load_frame(path).bayer is None                 # not a Bayer layout: mono


def test_hot_pixel_on_a_colour_site_is_fixed_from_its_own_colour():
    rng = np.random.default_rng(3)
    levels = {"R": 1000.0, "G": 100.0, "B": 500.0}
    rgb = np.stack([np.full(SHAPE, levels[c], np.float32) for c in "RGB"])
    m = _mosaic(rgb, "RGGB", rng, noise=2.0)
    m[40, 40] = 9000.0                                    # (even, even) = an R site
    fixed = cosmetic_correct_bayer(m)
    assert abs(fixed[40, 40] - 1000.0) < 20               # restored to the R level
    plain = cosmetic_correct(m)                            # a mono 3x3 median is wrong here
    assert abs(plain[40, 40] - 1000.0) > 100
    # and it does not mistake the bright-colour pixels for hot ones
    assert np.allclose(np.delete(fixed.ravel(), 40 * SHAPE[1] + 40),
                       np.delete(m.ravel(), 40 * SHAPE[1] + 40))


def test_bayer_flat_fixes_vignetting_but_keeps_the_colour_balance():
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    vignette = 1.0 - 0.3 * (((yy - 60) / 60.0) ** 2 + ((xx - 80) / 80.0) ** 2) / 2
    levels = {"R": 3000.0, "G": 6000.0, "B": 2000.0}
    flat_rgb = np.stack([levels[c] * vignette for c in "RGB"]).astype(np.float32)
    flat = MasterFlat(_mosaic(flat_rgb, "RGGB", rng, noise=1.0), {}, "x", bayer="RGGB")
    # each colour's sub-plane is normalised to 1 on its own, not all to one global median
    for i in range(4):
        assert abs(np.median(flat.data[i // 2::2, i % 2::2]) - 1.0) < 0.02
    # a uniformly lit scene through that vignette comes out flat, in every colour
    sky = np.full((3,) + SHAPE, 200.0, np.float32)
    light = _mosaic(sky * vignette, "RGGB", rng, noise=0.5)
    out = flat.divide(light, Frame(path=Path("x"), filter="OSC", data=light, date_obs="",
                                   exptime=1.0, bayer="RGGB"))
    centre, corner = out[50:70, 70:90], out[:10, :10]
    assert abs(centre.mean() - corner.mean()) < 0.04 * centre.mean()


# -- Canon raw, with rawpy faked --------------------------------------------------

class _FakeRaw:
    def __init__(self, image, colours, black):
        self.raw_image_visible = image
        self.raw_colors_visible = colours
        self.color_desc = b"RGBG"
        self.black_level_per_channel = black

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_rawpy(monkeypatch, images):
    """A stand-in for rawpy that 'decodes' files by name from `images`."""
    colours = np.zeros(SHAPE, np.uint8)
    colours[0::2, 0::2], colours[0::2, 1::2], colours[1::2, 0::2], colours[1::2, 1::2] = 0, 1, 3, 2
    mod = types.SimpleNamespace(imread=lambda p: _FakeRaw(
        images[Path(p).name].astype(np.uint16), colours, [100, 101, 102, 103]))
    monkeypatch.setattr("livestacker.rawframe._rawpy", lambda: mod)


def test_canon_raw_is_read_as_a_black_subtracted_bayer_mosaic(tmp_path, monkeypatch):
    img = np.full(SHAPE, 1000, np.uint16)
    _fake_rawpy(monkeypatch, {"2026-09-13_20-48-28_20.00s_0000.cr2": img})
    path = tmp_path / "2026-09-13_20-48-28_20.00s_0000.cr2"
    path.write_bytes(b"")
    fr = load_frame(path)
    assert fr.bayer == "RGGB" and fr.data.shape == SHAPE and fr.filter == "OSC"
    assert fr.exptime == 20.0                                # from the file name when no EXIF
    # each site loses the black level of its own colour index: R(0) 100, G(1) 101,
    # G2(3) 103, B(2) 102 in the fake
    assert fr.data[0, 0] == 900 and fr.data[0, 1] == 899
    assert fr.data[1, 0] == 897 and fr.data[1, 1] == 898
    assert read_header(path)[1] == SHAPE and read_header(path)[0].bayer == "RGGB"


def test_raw_calibration_frames_are_recognised_by_name_not_stacked_as_lights():
    assert kind_from_path(Path("/n/2026-09-13/FlatWizard/FLAT/x_0.94s_0000.cr2")) == "FLAT"
    assert kind_from_path(Path("/docs/dark_180s_1600iso_x.cr2")) == "DARK"
    assert kind_from_path(Path("/n/2026-09-13/coathanger/SNAPSHOT/x.cr2")) == "SNAPSHOT"
    assert kind_from_path(Path("/n/BIAS/x.cr2")) == "BIAS"
    assert kind_from_path(Path("/n/2026-09-13/coathanger/LIGHT/x_20.00s_0001.cr2")) == ""
    assert kind_from_path(Path("/n/Dark Nebula/LIGHT/x_20.00s.cr2")) == ""        # not a dark
    assert _is_fits(Path("x.CR2")) and _is_fits(Path("x.cr3")) and _is_fits(Path("x.fits"))
    assert not _is_fits(Path("x.jpg"))


def test_median_of_raw_calibration_frames_and_flats_from_raw(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    images = {f"f{i}_0.9s.cr2": rng.integers(3000, 5000, SHAPE) for i in range(5)}
    _fake_rawpy(monkeypatch, images)
    paths = []
    for name in images:
        (tmp_path / name).write_bytes(b"")
        paths.append(tmp_path / name)
    got = median_combine(paths, strip_rows=7)
    expect = np.median(np.stack([load_frame(p).data for p in paths]), axis=0)
    assert np.array_equal(got, expect.astype(np.float32))
    flats = build_flats(tmp_path)                            # a folder of raw flats
    assert list(flats) == ["OSC"] and flats["OSC"].bayer == "RGGB"


def test_missing_rawpy_gives_a_clear_message(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "rawpy", None)           # as if not installed
    path = tmp_path / "x_1.00s.cr2"
    path.write_bytes(b"")
    with pytest.raises(ImportError, match="pip install rawpy"):
        load_frame(path)


# -- real data (skipped where it isn't available) ---------------------------------

CANON = sorted(Path("~/Pictures/astro/nina/2026-09-13/coathanger/SNAPSHOT").expanduser().glob("*.cr2"))
S50 = sorted(Path("~/Pictures/astro/s50/NGC 6888_sub").expanduser().glob("*.fit*"))


@pytest.mark.skipif(len(CANON) < 2, reason="no Canon CR2 sample frames on this machine")
def test_real_canon_cr2_frames_stack():
    pytest.importorskip("rawpy")
    cfg = StackConfig(osc_debayer="superpixel")
    p = Pipeline(cfg, Calibrator(cfg))
    for path in CANON[:3]:
        fr = load_frame(path)
        assert fr.bayer == "RGGB" and fr.data.shape == (4022, 6024)
        assert fr.exptime == 20.0 and fr.gain == 1600
        assert fr.imagetyp == "SNAPSHOT"                       # (folder name)
        fr.imagetyp = "Light"
        assert p.ingest(fr, recombine=False).ok
    p.recombine()
    assert {k: v["frames"] for k, v in p.status()["channels"].items()} == {"R": 3, "G": 3, "B": 3}
    assert p.render_preview(StretchParams(auto=True)) is not None


@pytest.mark.skipif(len(S50) < 4, reason="no Seestar sample frames on this machine")
def test_real_seestar_bayer_fits_stack():
    cfg = StackConfig()
    p = Pipeline(cfg, Calibrator(cfg))
    for path in S50[:4]:
        fr = load_frame(path)
        assert fr.bayer == "GRBG"
        assert p.ingest(fr, recombine=False).ok
    p.recombine()
    assert p.status()["total_frames"] == 4
    assert p.render_preview(StretchParams(auto=True)) is not None


def test_status_says_whether_this_is_a_colour_camera_session(tmp_path):
    rng = np.random.default_rng(7)
    p = Pipeline(StackConfig(**CFG, osc_debayer="superpixel"))
    assert p.status()["osc"] is False
    path = tmp_path / "s.fits"
    _write(path, _mosaic(_scene(rng), "RGGB", rng), "RGGB")
    assert p.ingest(load_frame(path)).ok
    st = p.status()
    assert st["osc"] is True and st["total_frames"] == 1
    assert sorted(st["channels"]) == ["B", "G", "R"]        # still tracked underneath
