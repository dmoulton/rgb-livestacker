"""Ha enhancement: nebula-only Ha (stars subtracted) added into red."""

import numpy as np

from livestacker.config import StackConfig, StretchParams
from livestacker.pipeline import Pipeline
from livestacker.processing import combine

STARS = [(30, 40), (70, 110), (95, 55), (45, 130), (60, 80), (20, 100), (105, 140),
         (85, 20), (15, 15), (110, 90), (50, 60), (35, 145)]
SHAPE = (130, 170)
YY, XX = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
BLOB = (65, 100)


def _layer(star_gain, blob_gain, rng):
    d = 1000.0 + rng.normal(0, 3, SHAPE)
    for cy, cx in STARS:
        d = d + star_gain * 9000 * np.exp(-(((YY - cy) ** 2 + (XX - cx) ** 2) / 3.0))
    d = d + blob_gain * np.exp(-(((YY - BLOB[0]) ** 2 + (XX - BLOB[1]) ** 2) / (2 * 8.0 ** 2)))
    return d.astype(np.float32)


def _channels(blob_in_ha=3000.0):
    rng = np.random.default_rng(0)
    full = np.ones(SHAPE, bool)
    return {
        "R": (_layer(1.0, 0.0, rng), full, 10),
        "G": (_layer(1.0, 0.0, rng), full, 10),
        "B": (_layer(1.0, 0.0, rng), full, 10),
        # Ha: stars at 3x the red flux (a different exposure), plus the nebula
        "H": (_layer(3.0, blob_in_ha, rng), full, 10),
    }


CFG = StackConfig(background_enabled=True, background_box=32, background_border_erode=2, auto_white_balance=False,
                  min_subs_for_background=1, display_border_erode=0)  # no crop: keep coordinates


def test_ha_layer_keeps_the_nebula_and_cancels_the_stars():
    out = combine(_channels(), CFG)
    assert out.ha is not None
    ha = out.ha
    blob = ha[BLOB]
    # (a star sitting on the nebula legitimately has nebula under it)
    stars = np.array([ha[y, x] for y, x in STARS
                      if np.hypot(y - BLOB[0], x - BLOB[1]) > 25])
    assert blob > 0.4                                   # nebula sits near HA_PEAK
    assert np.abs(stars).max() < 0.25 * blob            # continuum stars cancelled


def test_no_ha_channel_or_no_red_gives_no_layer():
    ch = _channels()
    assert combine({k: v for k, v in ch.items() if k != "H"}, CFG).ha is None
    assert combine({k: v for k, v in ch.items() if k != "R"}, CFG).ha is None


def test_ha_strength_adds_red_only_where_the_nebula_is():
    p = Pipeline(CFG)
    out = combine(_channels(), CFG)
    p._linear_rgb, p._linear_ha = out.rgb, out.ha
    assert p.status()["ha"] is True
    off = p._render(StretchParams(auto=False, ha=0.0))
    on = p._render(StretchParams(auto=False, ha=2.0))
    a, b = np.asarray(off).astype(int), np.asarray(on).astype(int)
    assert b[BLOB][0] > a[BLOB][0] + 20                  # red up at the nebula
    assert np.abs(b[BLOB][1:] - a[BLOB][1:]).max() == 0  # green / blue untouched
    assert np.abs(b[5, 5] - a[5, 5]).max() <= 2          # empty sky unchanged


def test_narrowband_off_gives_the_plain_rgb_image():
    p = Pipeline(CFG)
    out = combine(_channels(), CFG)
    p._linear_rgb, p._linear_ha = out.rgb, out.ha
    plain = np.asarray(p._render(StretchParams(auto=False, ha=0.0)))
    off = np.asarray(p._render(StretchParams(auto=False, ha=2.0, narrowband=False)))
    on = np.asarray(p._render(StretchParams(auto=False, ha=2.0, narrowband=True)))
    assert np.array_equal(off, plain)
    assert not np.array_equal(on, plain)
    # the page's saved link carries it as a query value
    from livestacker.server import params_from_query
    assert params_from_query("narrowband=false").narrowband is False
    assert params_from_query("narrowband=true").narrowband is True


def test_ha_is_available_from_the_first_ha_sub():
    ch = _channels()
    ch["H"] = (ch["H"][0], ch["H"][1], 1)
    assert combine(ch, CFG).ha is not None
