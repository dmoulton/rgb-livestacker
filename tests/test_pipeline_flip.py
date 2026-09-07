"""A meridian flip must not disturb the stack.

`astroalign`'s triangle matching is rotation-invariant and a 180°-rotated
sensor rectangle covers the same footprint, so a flip needs no special
handling — this test locks that in. It runs only if a session with a
pier-side change is available under NINA_DIR (or the default path).
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astroalign")
from astropy.io import fits  # noqa: E402

from livestacker.config import StackConfig  # noqa: E402
from livestacker.calibration import Calibrator  # noqa: E402
from livestacker.fitsframe import load_frame  # noqa: E402
from livestacker.pipeline import Pipeline  # noqa: E402

NINA = Path(os.environ.get(
    "NINA_DIR", Path.home() / "Pictures/astro/nina"))


def _find_flip_session():
    for light_dir in NINA.glob("*/*/LIGHT"):
        subs = sorted(p for p in light_dir.glob("*.fit*"))
        if len(subs) < 30:
            continue
        rows = []
        for p in subs:
            try:
                h = fits.getheader(p)
            except Exception:
                continue
            if h.get("FILTER") in ("R", "G", "B"):
                rows.append((h.get("DATE-OBS", ""), h.get("PIERSIDE"), p))
        rows.sort()
        sides = [r[1] for r in rows]
        if len(set(s for s in sides if s)) > 1:
            flip = next(i for i in range(1, len(sides)) if sides[i] != sides[i - 1])
            return [r[2] for r in rows], flip
    return None, None


def test_meridian_flip_frames_all_register():
    files, flip = _find_flip_session()
    if not files:
        pytest.skip("no session with a meridian flip available")

    window = files[max(0, flip - 10):flip + 12]
    cfg = StackConfig(hot_pixel_enabled=True)
    p = Pipeline(cfg, Calibrator(cfg))
    rejected = sum(0 if p.ingest(load_frame(f)).ok else 1 for f in window)

    assert rejected == 0, f"{rejected}/{len(window)} frames rejected across the flip"
    rgb = p._linear_rgb
    assert rgb is not None and min(rgb.shape[:2]) > 500, "stack collapsed at the flip"
