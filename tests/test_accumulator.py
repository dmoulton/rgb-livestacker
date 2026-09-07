"""Sigma-clip and coverage behaviour of the running stack."""

import numpy as np

from livestacker.accumulator import ChannelStack
from livestacker.config import StackConfig


def _sky(rng, shape, level=1000.0, noise=5.0):
    return (level + rng.normal(0, noise, shape)).astype(np.float32)


def test_trail_rejected_but_star_preserved():
    rng = np.random.default_rng(0)
    H, W = 240, 320
    clip = ChannelStack("R", StackConfig(sigma_clip_enabled=True))
    ctrl = ChannelStack("R", StackConfig(sigma_clip_enabled=False))
    yy, xx = np.mgrid[0:H, 0:W]

    for i in range(40):
        f = _sky(rng, (H, W))
        # an undersampled star that jitters sub-pixel each frame
        dy, dx = rng.normal(0, 0.7, 2)
        f += 40000 * np.exp(-(((yy - 120 - dy) ** 2 + (xx - 160 - dx) ** 2) / 1.4))
        if i == 22:                       # one satellite trail
            f[60:63, :] += 9000.0
        clip.add(f)
        ctrl.add(f)

    trail = clip.mean[61].mean()
    assert trail < 1100, f"trail leaked through: {trail}"
    assert ctrl.mean[61].mean() > 1150, "control should be contaminated"

    star_clip = clip.mean[116:125, 156:165].max()
    star_ctrl = ctrl.mean[116:125, 156:165].max()
    assert star_clip / star_ctrl > 0.9, f"sigma-clip ate the star: {star_clip/star_ctrl:.3f}"


def test_coverage_counts_footprint_not_survival():
    """A quiet pixel that takes an outlier hit every other frame gets those
    samples clipped, so its accepted-sample count falls well below the frame
    total — but it must stay 'covered' (footprint), or it is cropped to a
    black speck that multiplies as the session deepens."""
    rng = np.random.default_rng(1)
    H, W = 120, 160
    st = ChannelStack("R", StackConfig(sigma_clip_enabled=True))
    hit = (slice(40, 44), slice(60, 64))

    for i in range(80):
        f = _sky(rng, (H, W))
        # quiet through warm-up, then a cosmic-ray-like hit every other frame
        if i >= 10 and i % 2 == 0:
            f[hit] += 6000.0
        st.add(f)

    assert st._count[hit].mean() < 0.6 * st.frames   # clipping really cut it
    assert (st._seen[hit] == st.frames).all()        # every frame landed on it
    assert st.covered_mask(0.6).all(), "clipped pixels dropped out of coverage"
