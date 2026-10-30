"""Pipeline behaviour that doesn't need real sky data."""

import numpy as np

from livestacker.config import StackConfig
from livestacker.fitsframe import Frame
from livestacker.pipeline import Pipeline


def _star_frame(shape=(120, 160), seed=0):
    rng = np.random.default_rng(seed)
    data = (1000.0 + rng.normal(0, 5, shape)).astype(np.float32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for cy, cx in [(30, 40), (70, 110), (95, 55), (45, 130), (60, 80), (20, 100), (105, 140),
                   (85, 20), (15, 15), (110, 90), (50, 60), (35, 145)]:
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


def _frame_of(data, imagetyp="LIGHT", name="x.fits"):
    f = _star_frame()
    f.data = data
    f.imagetyp = imagetyp
    f.path = type("P", (), {"name": name})()
    return f


def test_calibration_frames_are_skipped_not_stacked():
    cfg = StackConfig(hot_pixel_enabled=False, background_enabled=False,
                      auto_white_balance=False, sigma_clip_enabled=False)
    p = Pipeline(cfg)
    for kind in ("DARK", "Flat Field", "BIAS", "SNAPSHOT"):
        r = p.ingest(_frame_of(_star_frame().data, imagetyp=kind))
        assert r.skipped and not r.ok
    assert p.reference is None and p.rejected == 0 and not p.channels

    assert p.ingest(_frame_of(_star_frame().data, imagetyp="LIGHT")).ok
    assert p.ingest(_frame_of(_star_frame(seed=1).data, imagetyp="")).ok   # no IMAGETYP: accepted


def test_a_starless_first_frame_is_not_adopted_as_the_reference():
    cfg = StackConfig(hot_pixel_enabled=False, background_enabled=False,
                      auto_white_balance=False, sigma_clip_enabled=False)
    p = Pipeline(cfg)
    rng = np.random.default_rng(3)
    flat = (500 + rng.normal(0, 8, (120, 160))).astype(np.float32)

    r = p.ingest(_frame_of(flat))
    assert not r.ok and "reference" in r.message
    assert p.reference is None                      # still waiting for a usable one

    assert p.ingest(_frame_of(_star_frame(seed=0).data)).ok   # a real frame takes over...
    assert p.ingest(_frame_of(_star_frame(seed=1).data)).ok   # ...and later frames register


def test_luminance_only_session_shows_a_monochrome_image():
    import io
    from PIL import Image
    from livestacker.config import StretchParams
    cfg = StackConfig(hot_pixel_enabled=False, background_enabled=False,
                      sigma_clip_enabled=False, min_subs_for_background=1)
    p = Pipeline(cfg)
    for seed in range(3):
        fr = _star_frame(seed=seed)
        fr.filter = "L"
        assert p.ingest(fr).ok
    assert p.status()["luminance"] is False       # nothing to blend: it *is* the image
    jpeg = p.render_preview(StretchParams(auto=True))
    assert jpeg is not None
    a = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB")).astype(int)
    assert a.std() > 5                             # a real picture, not a blank
    assert np.abs(a[..., 0] - a[..., 1]).max() <= 2 and np.abs(a[..., 1] - a[..., 2]).max() <= 2
