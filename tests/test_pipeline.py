"""Pipeline behaviour that doesn't need real sky data."""

import numpy as np

from livestacker.config import StackConfig
from livestacker.fitsframe import Frame
from livestacker.pipeline import Pipeline


def _star_frame(shape=(120, 160), seed=0):
    rng = np.random.default_rng(seed)
    data = (1000.0 + rng.normal(0, 5, shape)).astype(np.float32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for cy, cx in [(30, 40), (70, 110), (95, 55), (45, 130), (60, 80)]:
        data += 9000 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / 3.0))
    return Frame(path=type("P", (), {"name": "x.fits"})(), filter="R",
                 data=data, date_obs="", exptime=20.0)


def test_crop_edges_trims_before_registration():
    base = dict(hot_pixel_enabled=False, background_enabled=False,
               auto_white_balance=False, sigma_clip_enabled=False)
    plain = Pipeline(StackConfig(**base))
    cropped = Pipeline(StackConfig(**base, crop_bottom=15, crop_left=8))
    for seed in (0, 1):
        plain.ingest(_star_frame(seed=seed))
        cropped.ingest(_star_frame(seed=seed))

    assert cropped.reference.shape == (120 - 15, 160 - 8)
    ph, pw = plain._linear_rgb.shape[:2]
    ch, cw = cropped._linear_rgb.shape[:2]
    assert (ph - ch, pw - cw) == (15, 8)      # exactly the crop, nothing else


def test_no_crop_keeps_full_frame():
    cfg = StackConfig(hot_pixel_enabled=False, background_enabled=False,
                      auto_white_balance=False, sigma_clip_enabled=False)
    p = Pipeline(cfg)
    p.ingest(_star_frame(seed=0))
    assert p.reference.shape == (120, 160)


def test_full_flag_skips_the_preview_downscale_and_status_reports_size():
    import io
    from PIL import Image
    from livestacker.config import StretchParams

    cfg = StackConfig(hot_pixel_enabled=False, background_enabled=False,
                      auto_white_balance=False, sigma_clip_enabled=False,
                      preview_max_px=60)
    p = Pipeline(cfg)
    p.ingest(_star_frame(seed=0))
    w, h = p.status()["size"]

    small = Image.open(io.BytesIO(p.render_preview(StretchParams())))
    full = Image.open(io.BytesIO(p.render_preview(StretchParams(full=True))))
    assert max(small.size) == 60                      # downscaled to the limit
    assert full.size == (w, h)                        # untouched: the real stack size
    assert p.status()["preview_px"] == 60
