"""Display settings come from the network; bad input must not become a bad render."""

import math

from livestacker.config import StretchParams


def test_from_json_ignores_wrong_types_and_non_finite_values():
    p = StretchParams.from_json({
        "gamma": None, "black": "0.5", "midtones": math.nan, "contrast": math.inf,
        "green": True, "lum": [1], "auto": "yes", "full": 1,
    })
    d = StretchParams()
    assert (p.gamma, p.black, p.midtones, p.contrast, p.green, p.lum) == \
        (d.gamma, d.black, d.midtones, d.contrast, d.green, d.lum)
    assert p.auto is d.auto          # a string is not a bool
    assert p.full is True            # 1 is


def test_from_json_clamps_to_sane_ranges():
    p = StretchParams.from_json({"gamma": 1e9, "black": -5, "midtones": 7, "green": 3})
    assert p.gamma == 5.0 and p.black == 0.0 and p.midtones == 0.99 and p.green == 1.0


def test_from_json_keeps_valid_values():
    p = StretchParams.from_json({"gamma": 1.8, "black": 0.1, "auto": False})
    assert (p.gamma, p.black, p.auto) == (1.8, 0.1, False)
