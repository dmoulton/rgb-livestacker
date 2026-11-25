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


# -- look-back (trails in the first subs) and sky matching --------------------------

def _run(frames, **cfg):
    st = ChannelStack("G", StackConfig(**cfg))
    for f in frames:
        st.add(f)
    return st


def _session(seed=0, n=30, shape=(120, 160), noise=5.0, levels=None, trail_at=None):
    rng = np.random.default_rng(seed)
    levels = [1000.0] * n if levels is None else levels
    out = []
    for i in range(n):
        f = _sky(rng, shape, level=levels[i], noise=noise)
        if i == trail_at:
            f[60:63, :] += 90.0                              # 18 sigma, like the M27 one
        out.append(f)
    return out


TRAIL = slice(60, 63)


def test_a_trail_in_the_first_subs_is_removed():
    for position in (0, 1, 3, 5, 6, 15, 29):                 # before and after warm-up
        st = _run(_session(trail_at=position))
        left = st.mean[TRAIL].mean() - 1000.0
        assert abs(left) < 0.6, f"sub #{position + 1} left {left:.1f}"
    # (without rejection that trail would sit 3 counts high)
    assert _run(_session(trail_at=1), sigma_clip_enabled=False).mean[TRAIL].mean() > 1002.5


def test_a_trail_disappears_once_a_filter_has_a_few_subs():
    frames = _session(n=8, trail_at=1)
    st = ChannelStack("G", StackConfig())
    left = []
    for f in frames:
        st.add(f)
        left.append(float(st.mean[TRAIL].mean() - 1000.0))
    assert left[1] > 30                                      # nothing to judge it against yet
    trail = [90.0 / (i + 1) for i in range(8)]               # what no rejection would leave
    assert abs(left[3]) < 0.15 * trail[3]                    # four subs: nearly gone
    assert abs(left[5]) < 0.5                                # six: gone


def test_look_back_leaves_ordinary_noise_and_stars_alone():
    frames = _session(n=40)
    rng = np.random.default_rng(9)
    yy, xx = np.mgrid[0:120, 0:160]
    for f in frames:                                         # a star that flickers a lot
        f += (4000 * rng.uniform(0.6, 1.4) *
              np.exp(-(((yy - 30) ** 2 + (xx - 40) ** 2) / 2.0))).astype(np.float32)
    st, plain = _run(frames), _run(frames, sigma_clip_enabled=False)
    sky = (slice(80, 120), slice(0, 160))
    # look-back on its own: the stack as read out against the running mean
    changed = (st.mean - st._mean.astype(np.float32))[sky]
    assert float((changed != 0).mean()) < 0.01                       # rarely steps in
    assert abs(float(changed.mean())) < 0.02                         # no bias to speak of
    assert abs(st.mean[30, 40] / plain.mean[30, 40] - 1) < 0.002
    # reading the stack doesn't change it
    assert np.array_equal(st.mean, st.mean)


def test_a_drifting_sky_is_matched_so_rejection_still_works():
    levels = list(np.linspace(1000, 1400, 30))               # moonrise: 80x the noise
    frames = _session(levels=levels, trail_at=20)
    st = _run(frames)
    assert abs(st.mean[:50].mean() - 1000.0) < 1.0           # held at the first sub's level
    assert abs(st.mean[TRAIL].mean() - st.mean[:50].mean()) < 0.8
    unmatched = _run(frames, sky_match_enabled=False)
    assert abs(unmatched.mean[TRAIL].mean() - unmatched.mean[:50].mean()) > 2.0


def test_sky_matching_leaves_no_step_where_only_some_subs_reach():
    """A dithered edge covered only by late (brighter-sky) subs must not stand out."""
    levels = [1000.0] * 15 + [1200.0] * 15
    frames = _session(levels=levels)
    for f in frames[:15]:
        f[:, 140:] = np.nan                                  # early subs miss this strip
    st = _run(frames)
    assert abs(st.mean[:, 145:].mean() - st.mean[:, :120].mean()) < 1.0
    unmatched = _run(frames, sky_match_enabled=False)
    assert unmatched.mean[:, 145:].mean() - unmatched.mean[:, :120].mean() > 80


def test_sky_matching_ignores_what_each_sub_happens_to_cover():
    """Matched pixel for pixel: a sub covering a bright nebula region and one
    covering plain sky are not 'different sky levels'."""
    rng = np.random.default_rng(4)
    scene = np.full((120, 160), 1000.0, np.float32)
    scene[:, :60] += 500.0                                   # nebula on the left
    frames = []
    for i in range(12):
        f = scene + rng.normal(0, 5, scene.shape).astype(np.float32)
        if i % 2:
            f[:, :50] = np.nan                               # misses most of the nebula
        frames.append(f)
    st = _run(frames)
    assert abs(st.mean[:, 120:].mean() - 1000.0) < 1.0
    assert abs(st.mean[:, :40].mean() - 1500.0) < 2.0
