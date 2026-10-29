"""FITS loading and the lightweight `Frame` record the pipeline passes around."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

log = logging.getLogger("livestacker.fits")


@dataclass
class Frame:
    path: Path
    filter: str          # normalised FITS FILTER keyword, e.g. "R"
    data: np.ndarray     # float32, shape (H, W)
    date_obs: str        # DATE-OBS string, or "" if absent
    exptime: float       # seconds
    gain: float | None = None
    xbinning: int | None = None
    ccd_temp: float | None = None
    imagetyp: str = ""   # IMAGETYP keyword, e.g. "LIGHT" / "DARK" ("" if absent)


# Map common filter-name spellings onto single-letter RGB channels. Anything
# not listed is passed through uppercased and handled as its own channel
# (so SHO / narrowband names still route somewhere sensible).
_FILTER_ALIASES = {
    "RED": "R", "R": "R",
    "GREEN": "G", "G": "G",
    "BLUE": "B", "B": "B",
    "LUM": "L", "LUMINANCE": "L", "L": "L", "CLEAR": "L",
}


def normalise_filter(name: str | None) -> str:
    if not name:
        return "?"
    key = name.strip().upper()
    return _FILTER_ALIASES.get(key, key)


# Calibration / test exposures that can share a capture tree with the lights.
_NOT_LIGHT = ("DARK", "FLAT", "BIAS", "OFFSET", "SNAPSHOT")


def is_light(frame: Frame) -> bool:
    """False for dark / flat / bias / snapshot frames, judged by IMAGETYP. Frames
    with no IMAGETYP (other capture software) are given the benefit of the doubt."""
    kind = frame.imagetyp.upper()
    return not any(word in kind for word in _NOT_LIGHT)


def load_frame(path: str | Path) -> Frame:
    """Read a FITS file into a `Frame`. Raises on unreadable / non-2D data."""
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        hdu = next((h for h in hdul if h.data is not None), None)
        if hdu is None:
            raise ValueError(f"{path.name}: no image data")
        data = np.asarray(hdu.data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"{path.name}: expected 2-D data, got {data.shape}")
        header = hdu.header

    def _num(key):
        try:
            return float(header[key])
        except (KeyError, TypeError, ValueError):
            return None

    binning = _num("XBINNING")
    return Frame(
        path=path,
        filter=normalise_filter(header.get("FILTER")),
        data=data,
        date_obs=str(header.get("DATE-OBS", "")),
        exptime=float(header.get("EXPTIME", header.get("EXPOSURE", 0.0)) or 0.0),
        gain=_num("GAIN"),
        xbinning=int(binning) if binning is not None else None,
        ccd_temp=_num("CCD-TEMP"),
        imagetyp=str(header.get("IMAGETYP", "") or "").strip(),
    )
