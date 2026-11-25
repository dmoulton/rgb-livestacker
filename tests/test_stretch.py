"""Brightness: bright cores survive, the sky isn't crushed to black, and each
browser chooses how strong the auto stretch is."""

import io

import numpy as np
from PIL import Image

from livestacker.config import StackConfig, StretchParams
from livestacker.pipeline import Pipeline
from livestacker.processing import (_luma, apply_luminance, apply_stretch,
                                    auto_stretch, combine, lrgb_chroma,
                                    midtones_for, stretch_stats, _mtf)

SHAPE = (300, 400)
YY, XX = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
CORE = (150, 200)
CFG = StackConfig(background_enabled=True, background_box=48, background_border_erode=2,
                  auto_white_balance=False, min_subs_for_background=1,
                  display_border_erode=0)


def _globular(rng):
    """Sky + many ordinary stars + a cluster core far brighter than any of them
    but covering well over 0.5% of the picture."""
    img = 1000.0 + rng.normal(0, 4, SHAPE)
    for _ in range(150):
        cy, cx = rng.integers(8, SHAPE[0] - 8), rng.integers(8, SHAPE[1] - 8)
        img += rng.uniform(100, 600) * np.exp(-(((YY - cy) ** 2 + (XX - cx) ** 2) / 3.0))
    # a few bright foreground stars, brighter than the core's centre (as in M2)
    for cy, cx in ((30, 50), (250, 340), (60, 330)):
        img += 70000.0 * np.exp(-(((YY - cy) ** 2 + (XX - cx) ** 2) / 3.0))
    r2 = (YY - CORE[0]) ** 2 + (XX - CORE[1]) ** 2
    img += 20000.0 / (1.0 + r2 / 12.0 ** 2) ** 2           # peaked, with wings
    return img.astype(np.float32)


def _channels(seed=0, names="RGB"):
    rng = np.random.default_rng(seed)
    base = _globular(np.random.default_rng(99))
    full = np.ones(SHAPE, bool)
    return {n: ((base + rng.normal(0, 4, SHAPE)).astype(np.float32), full, 10) for n in names}


def _ring(img, r, width=2):
    rr = np.hypot(YY - CORE[0], XX - CORE[1])
    return float(np.median(img[(rr >= r) & (rr < r + width)]))


def test_a_bright_core_is_not_clipped_to_a_flat_disc():
    out = combine(_channels(), CFG)
    lin = out.rgb.mean(axis=-1)
    assert (out.rgb.max(axis=-1) >= 1.0).mean() < 0.0002     # (it was 0.5% by design)
    shown = apply_stretch(out.rgb, auto_stretch(out.rgb)).mean(axis=-1)
    levels = [_ring(shown, r) for r in (0, 8, 16, 28)]
    assert levels == sorted(levels, reverse=True)             # still graded to the centre
    assert levels[0] - levels[3] > 12 and levels[3] < 245     # not a flat white disc
    # the old white point (99.5th percentile) showed this same core as flat 255
    old = out.rgb - out.pedestal
    old = np.clip(old / np.percentile(old.max(axis=-1), 99.5), 0, None)
    flat = apply_stretch(old, auto_stretch(old)).mean(axis=-1)
    assert _ring(flat, 0) - _ring(flat, 8) < 1
    assert (shown >= 254).mean() < 0.004                     # only the bright stars' centres
    assert _ring(lin, 0) > 3 * _ring(lin, 16)                 # linear detail is all there


def test_the_sky_is_not_crushed_to_black():
    out = combine(_channels(), CFG)
    inside = out.rgb[5:-5, 5:-5]
    assert (inside == 0).mean() < 0.01                        # it used to be about half
    assert out.pedestal > 0
    sky = inside[:60, :60].mean(axis=-1)
    assert abs(np.median(sky) - out.pedestal) < 0.3 * out.pedestal
    shown = apply_stretch(out.rgb, auto_stretch(out.rgb, 0.20))[5:-5, 5:-5].mean(axis=-1)
    corner = shown[:60, :60]
    assert 35 < np.median(corner) < 70                        # a grey sky, about 0.2 * 255
    assert (corner <= 1).mean() < 0.02


def test_auto_stretch_puts_the_sky_at_the_requested_level():
    for sky in (0.001, 0.02, 0.3):
        for target in (0.05, 0.2, 0.4):
            m = midtones_for(sky, target)
            if 1e-4 < m < 0.99:                               # (not clamped)
                assert abs(float(_mtf(np.float64(sky), m)) - target) < 1e-6
    assert stretch_stats(np.zeros((10, 10, 3))) is None


def test_each_browser_sets_its_own_auto_strength():
    p = Pipeline(CFG)
    out = combine(_channels(), CFG)
    p._linear_rgb, p._pedestal = out.rgb, out.pedestal
    p._stats = {"rgb": stretch_stats(out.rgb), "lrgb": None}
    p._auto_params = auto_stretch(out.rgb)

    def sky_and_core(strength):
        img = np.asarray(Image.open(io.BytesIO(p.snapshot_png(
            StretchParams(auto=True, strength=strength))))).astype(float).mean(axis=-1)
        return np.median(img[5:65, 5:65]), _ring(img, 12)
    gentle, normal, strong = sky_and_core(0.08), sky_and_core(0.20), sky_and_core(0.35)
    assert gentle[0] < normal[0] < strong[0]                  # the sky follows the slider
    assert gentle[1] < normal[1] < strong[1]                  # and the core is tamer when low
    resolved = p._resolve_params(StretchParams(auto=True, strength=0.08, gamma=1.7, full=True))
    assert resolved.gamma == 1.7 and resolved.full and resolved.strength == 0.08
    manual = StretchParams(auto=False, black=0.01, midtones=0.002)
    assert p._resolve_params(manual) == manual


def test_new_limits_allow_the_small_midtones_values():
    q = StretchParams.from_json({"midtones": 0.0019, "strength": 9, "black": 0.002})
    assert q.midtones == 0.0019 and q.strength == 0.5 and q.black == 0.002
    assert StretchParams.from_json({"strength": 0}).strength == 0.02


def test_luminance_shares_the_scale_and_is_not_clipped_either():
    out = combine(_channels(names="RGBL"), CFG)
    assert out.lum is not None
    assert np.nanmean(out.lum >= 1.0) < 0.0002
    assert abs(np.nanmedian(out.lum[:60, :60]) - out.pedestal) < 0.3 * out.pedestal
    lrgb = apply_luminance(out.rgb, out.lum, 1.0,
                           lrgb_chroma(out.rgb, floor=0.02 * out.reference,
                                       pedestal=out.pedestal), out.pedestal)
    shown = apply_stretch(lrgb, auto_stretch(lrgb))
    assert (shown.max(axis=-1) >= 254).mean() < 0.003


def test_faint_colour_survives_the_pedestal_in_lrgb():
    ped = 0.02
    rgb = np.full((40, 40, 3), ped, np.float32)
    rgb[..., 0] += 0.012                                      # faint red signal above the sky
    rgb[..., 1] += 0.003
    rgb[..., 2] += 0.003
    lum = np.full((40, 40), ped + 0.006, np.float32)
    chroma = lrgb_chroma(rgb, floor=0.004, pedestal=ped)
    out = apply_luminance(rgb, lum, 1.0, chroma, ped)
    signal = out - ped
    assert signal[..., 0].mean() > 2.5 * signal[..., 2].mean()      # still clearly red
    assert np.allclose(_luma(out), lum, atol=1e-4)                   # brightness is L's
    # a plain grey sky stays grey
    sky = apply_luminance(np.full((8, 8, 3), ped, np.float32), np.full((8, 8), ped, np.float32),
                          1.0, pedestal=ped)
    assert np.allclose(sky, ped, atol=1e-6)
