"""Canon raw (CR2 / CR3) frames, read as linear Bayer mosaics.

`rawpy` (LibRaw) decodes the sensor data; no demosaicing, white balance, gamma or
brightness change is applied here: the stacker does its own, and does it on
linear data. Exposure, ISO and time come from the file's EXIF (rawpy doesn't
expose them).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .debayer import normalise_pattern
from .fitsframe import Frame

# folder / file names that mark calibration frames. Raw files carry no IMAGETYP, so
# without this a folder of flats or darks would be stacked as if it were lights.
_KIND_FOLDERS = {
    "dark": "DARK", "darks": "DARK", "flat": "FLAT", "flats": "FLAT",
    "bias": "BIAS", "biases": "BIAS", "offset": "OFFSET", "offsets": "OFFSET",
    "snapshot": "SNAPSHOT", "snapshots": "SNAPSHOT",
}


def _rawpy():
    try:
        import rawpy
    except ImportError as exc:
        raise ImportError("reading Canon raw files needs the 'rawpy' package "
                          "(pip install rawpy)") from exc
    return rawpy


def kind_from_path(path: Path) -> str:
    """'DARK' / 'FLAT' / 'BIAS' / 'OFFSET' / 'SNAPSHOT' if the file name or the
    folder it sits in says so, else '' (a light). Whole words only, so a folder
    called "Dark Nebula" does not count."""
    parent = path.parent.name.strip().lower()
    if parent in _KIND_FOLDERS:
        return _KIND_FOLDERS[parent]
    for token in re.split(r"[^a-z]+", path.stem.lower()):
        if token in _KIND_FOLDERS:
            return _KIND_FOLDERS[token]
    return ""


def _exif(path: Path) -> dict:
    """Exposure (s), ISO and capture time from the EXIF; {} if unreadable."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            exif = im.getexif()
            sub = exif.get_ifd(0x8769)
            out = {}
            if sub.get(33434):
                out["exptime"] = float(sub[33434])
            if sub.get(34855):
                out["iso"] = float(sub[34855])
            when = str(sub.get(36867) or exif.get(306) or "")
            m = re.match(r"(\d{4}):(\d\d):(\d\d)[ T](\d\d:\d\d:\d\d)", when)
            if m:
                out["date_obs"] = f"{m[1]}-{m[2]}-{m[3]}T{m[4]}"
            return out
    except Exception:
        return {}


def _exptime_from_name(path: Path) -> float:
    """NINA names its files like 'x_20.00s_0003.cr2'."""
    m = re.search(r"(\d+(?:\.\d+)?)s(?:_|\.|$)", path.stem)
    return float(m[1]) if m else 0.0


def _meta(path: Path) -> dict:
    info = _exif(path)
    return {
        "exptime": info.get("exptime") or _exptime_from_name(path),
        "gain": info.get("iso"),                   # ISO is the closest thing to gain
        "date_obs": info.get("date_obs", ""),
    }


def _pattern_of(raw) -> str:
    """The 2x2 colour layout at the top-left of the *visible* area: read from the
    per-pixel colour map, so it is right whatever margin LibRaw cropped."""
    desc = raw.color_desc.decode()
    tile = raw.raw_colors_visible[:2, :2].flatten()
    letters = "".join(desc[int(i)] for i in tile)
    pattern = normalise_pattern(letters)
    if pattern is None:
        raise ValueError(f"unsupported sensor layout {letters!r}")
    return pattern


def read_raw_header(path: Path) -> tuple[Frame, tuple[int, int]]:
    """A `Frame` with only the header fields (empty `data`) and the image shape."""
    path = Path(path)
    with _rawpy().imread(str(path)) as raw:
        shape = tuple(raw.raw_image_visible.shape)
        pattern = _pattern_of(raw)
    meta = _meta(path)
    frame = Frame(path=path, filter="OSC", data=np.empty((0, 0), np.float32),
                  date_obs=meta["date_obs"], exptime=meta["exptime"], gain=meta["gain"],
                  imagetyp=kind_from_path(path), bayer=pattern)
    return frame, shape


def load_raw(path: Path) -> Frame:
    """Read a CR2 / CR3 into a `Frame` holding the black-subtracted Bayer mosaic
    (float32, in the camera's own units, so 0 is black) and its layout."""
    path = Path(path)
    with _rawpy().imread(str(path)) as raw:
        mosaic = raw.raw_image_visible.astype(np.float32)       # a copy
        pattern = _pattern_of(raw)
        black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        colours = raw.raw_colors_visible[:2, :2]
        for i in range(2):
            for j in range(2):                                   # each site's own black level
                mosaic[i::2, j::2] -= black[int(colours[i, j])]
    meta = _meta(path)
    return Frame(path=path, filter="OSC", data=mosaic, date_obs=meta["date_obs"],
                 exptime=meta["exptime"], gain=meta["gain"],
                 imagetyp=kind_from_path(path), bayer=pattern)
