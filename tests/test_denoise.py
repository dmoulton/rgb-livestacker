"""Colour noise reduction: speckle goes, brightness and star colours stay."""

import io

import numpy as np
from PIL import Image

from livestacker.config import StackConfig, StretchParams
from livestacker.pipeline import Pipeline
from livestacker.processing import (SKY_PEDESTAL_SIGMAS, apply_stretch, auto_stretch,
                                    reduce_colour_noise, reduce_ha_noise, stretch_stats)

SHAPE = (160, 220)
YY, XX = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
SIGMA = 0.001
PED = SKY_PEDESTAL_SIGMAS * SIGMA
RED_STAR, BLUE_STAR = (50, 60), (110, 150)


def _image(seed=0):
    rng = np.random.default_rng(seed)
    rgb = (PED + rng.normal(0, SIGMA * 1.7, SHAPE + (3,))).astype(np.float32)
    for (cy, cx), colour in ((RED_STAR, (1.0, 0.4, 0.2)), (BLUE_STAR, (0.2, 0.4, 1.0))):
        blob = np.exp(-(((YY - cy) ** 2 + (XX - cx) ** 2) / 5.0))
        rgb += (0.5 * blob[..., None] * np.array(colour, np.float32)).astype(np.float32)
    rgb = np.clip(rgb, 0, None)
    rgb[:, :6] = 0.0                                              # the black border
    return rgb


def _speckle(img, area):
    d = img[area][..., 0] - img[area][..., 1]
    return float(d.std())


SKY = (slice(5, 40), slice(150, 215))


def test_colour_speckle_goes_but_brightness_stays():
    rgb = _image()
    out = reduce_colour_noise(rgb, PED)
    assert _speckle(out, SKY) < 0.35 * _speckle(rgb, SKY)
    # brightness (the channel mean) keeps its grain: this is not a blur of the picture
    a, b = rgb[SKY].mean(-1), out[SKY].mean(-1)
    assert abs(b.std() / a.std() - 1.0) < 0.15
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.95
    # the average colour of the sky does not shift
    assert np.abs(out[SKY].mean((0, 1)) - rgb[SKY].mean((0, 1))).max() < 0.15 * SIGMA
    assert not out[:, :6].any()                                    # border still black
    assert out.dtype == np.float32 and out.shape == rgb.shape


def test_bright_stars_keep_their_colour_and_get_no_halo():
    rgb = _image()
    out = reduce_colour_noise(rgb, PED)
    for star in (RED_STAR, BLUE_STAR):
        assert np.allclose(out[star], rgb[star], rtol=0.03)
    # a ring around the red star is not stained red
    rr = np.hypot(YY - RED_STAR[0], XX - RED_STAR[1])
    ring = (rr > 9) & (rr < 15)
    assert (out[ring][:, 0] - out[ring][:, 2]).mean() < 0.5 * SIGMA


def test_noise_free_or_pedestal_free_input_is_left_alone():
    rgb = _image()
    assert reduce_colour_noise(rgb, 0.0) is rgb
    ha = np.zeros(SHAPE, np.float32)
    assert reduce_ha_noise(ha) is ha


def test_ha_layer_speckle_is_reduced_and_bright_ha_kept():
    rng = np.random.default_rng(1)
    ha = rng.normal(0, 0.001, SHAPE).astype(np.float32)
    ha[60:100, 80:140] += 0.05
    out = reduce_ha_noise(ha)
    assert out[5:40, 150:215].std() < 0.4 * ha[5:40, 150:215].std()
    assert abs(out[70:90, 95:125].mean() - ha[70:90, 95:125].mean()) < 0.0005


def _pipeline(rgb, ha=None):
    p = Pipeline(StackConfig())
    p._linear_rgb, p._linear_ha, p._pedestal = rgb, ha, PED
    p._stats = {"rgb": stretch_stats(rgb), "lrgb": None}
    p._auto_params = auto_stretch(rgb)
    return p


def _png(p, **kw):
    return np.asarray(Image.open(io.BytesIO(p.snapshot_png(StretchParams(auto=True, **kw))))
                      ).astype(float)


def test_each_browser_chooses_its_amount_and_they_share_one_clean_copy(monkeypatch):
    import livestacker.pipeline as pipeline_mod
    calls = []
    real = pipeline_mod.reduce_colour_noise
    monkeypatch.setattr(pipeline_mod, "reduce_colour_noise",
                        lambda *a: (calls.append(1), real(*a))[1])
    p = _pipeline(_image())
    off = _png(p)
    assert not calls                                   # nobody asked: no work done
    full, half = _png(p, colour_nr=1.0), _png(p, colour_nr=0.5)
    again = _png(p, colour_nr=1.0)
    assert len(calls) == 1                             # one copy, shared and reused
    assert np.array_equal(full, again)
    assert _speckle(full, SKY) < _speckle(half, SKY) < _speckle(off, SKY)
    assert np.array_equal(off, _png(p, colour_nr=0.0))  # off is exactly the original
    # the auto stretch is measured on the original, so brightness doesn't jump
    assert abs(np.median(full[SKY].mean(-1)) - np.median(off[SKY].mean(-1))) < 6

    p._linear_rgb = _image(seed=7)                     # the stack changed
    _png(p, colour_nr=1.0)
    assert len(calls) == 2                             # ...so the copy is remade


def test_setting_is_validated_and_survives_auto(monkeypatch):
    assert StretchParams.from_json({"colour_nr": 7}).colour_nr == 1.0
    assert StretchParams.from_json({"colour_nr": -1}).colour_nr == 0.0
    assert StretchParams().colour_nr == 0.0
    p = _pipeline(_image())
    assert p._resolve_params(StretchParams(auto=True, colour_nr=0.4)).colour_nr == 0.4
    from livestacker.server import params_from_query
    assert params_from_query("colour_nr=0.35").colour_nr == 0.35


def test_recombine_drops_the_old_copy_and_lrgb_gets_clean_colour(tmp_path):
    from tests.test_stretch import CFG, _channels
    class Done:
        def __init__(self, mean, cover, n):
            self.mean, self._cover, self.frames, self.rejected = mean, cover, n, 0
        def covered_mask(self, frac):
            return self._cover
    p = Pipeline(CFG)
    p.channels = {k: Done(*v) for k, v in _channels(names="RGBL").items()}
    p.recombine()
    assert p._linear_lum is not None
    a = _png(p, colour_nr=0.0, lum=1.0)
    b = _png(p, colour_nr=1.0, lum=1.0)                 # works along the LRGB path
    corner = (slice(5, 60), slice(5, 90))
    assert _speckle(b, corner) <= _speckle(a, corner)
    assert p._nr is not None
    p.recombine()
    assert p._nr is None
