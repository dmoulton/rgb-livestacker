"""FITS loading and the lightweight `Frame` record the pipeline passes around."""

from __future__ import annotations

import logging
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

from .debayer import normalise_pattern, shift_pattern

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
    object_name: str = ""  # OBJECT keyword: the target's name ("" if absent)
    bayer: str | None = None  # colour sensor: the mosaic's 2x2 layout ("RGGB", ...); None = mono
    offset: float | None = None  # OFFSET keyword: the camera's baseline setting


# Map common filter-name spellings onto single-letter RGB channels. Anything
# not listed is passed through uppercased and handled as its own channel
# (so SHO / narrowband names still route somewhere sensible).
_FILTER_ALIASES = {
    "RED": "R", "R": "R",
    "GREEN": "G", "G": "G",
    "BLUE": "B", "B": "B",
    "LUM": "L", "LUMINANCE": "L", "L": "L", "CLEAR": "L",
    "HA": "H", "H-ALPHA": "H", "HALPHA": "H", "H": "H",
    "OIII": "O", "O3": "O", "O-III": "O", "O": "O",
    "SII": "S", "S2": "S", "S-II": "S", "S": "S",
}

NARROWBAND_FILTERS = ("H", "O", "S")


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


FITS_SUFFIXES = (".fits", ".fit", ".fts")
RAW_SUFFIXES = (".cr2", ".cr3")
FRAME_SUFFIXES = FITS_SUFFIXES + RAW_SUFFIXES      # everything the stacker can read


def is_raw(path: str | Path) -> bool:
    return Path(path).suffix.lower() in RAW_SUFFIXES


def _bayer_of(header) -> str | None:
    """The mosaic layout from BAYERPAT, moved by XBAYROFF / YBAYROFF if the file says
    the pattern starts off the sensor's first pixel."""
    pattern = normalise_pattern(str(header.get("BAYERPAT", "") or ""))
    if pattern is None:
        return None
    try:
        x, y = int(header.get("XBAYROFF", 0) or 0), int(header.get("YBAYROFF", 0) or 0)
    except (TypeError, ValueError):
        x = y = 0
    return shift_pattern(pattern, x, y)


def _image_hdu(hdul):
    return next((h for h in hdul if h.data is not None), None)


def _frame_from(path: Path, header, data: np.ndarray) -> Frame:
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
        object_name=str(header.get("OBJECT", "") or "").strip(),
        offset=_num("OFFSET"),
        bayer=_bayer_of(header) if data.ndim == 2 else None,
    )


def load_frame(path: str | Path) -> Frame:
    """Read a FITS (or Canon raw) file into a `Frame`. Raises on unreadable / non-2D
    data. A colour-sensor frame comes back as its Bayer mosaic with `bayer` set."""
    path = Path(path)
    if is_raw(path):
        from .rawframe import load_raw
        return load_raw(path)
    with fits.open(path, memmap=False) as hdul:
        hdu = _image_hdu(hdul)
        if hdu is None:
            raise ValueError(f"{path.name}: no image data")
        data = np.asarray(hdu.data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"{path.name}: expected 2-D data, got {data.shape}")
        return _frame_from(path, hdu.header, data)


def read_header(path: str | Path) -> tuple[Frame, tuple[int, int]]:
    """A `Frame` carrying only the header fields (its `data` is empty) plus the
    image shape, without reading the pixels — for deciding what to do with a
    folder of calibration frames before loading any of them."""
    path = Path(path)
    if is_raw(path):
        from .rawframe import read_raw_header
        return read_raw_header(path)
    with fits.open(path, memmap=False) as hdul:
        hdu = _image_hdu(hdul)
        if hdu is None:
            raise ValueError(f"{path.name}: no image data")
        shape = tuple(hdu.shape)
        if len(shape) != 2:
            raise ValueError(f"{path.name}: expected 2-D data, got {shape}")
        return _frame_from(path, hdu.header, np.empty((0, 0), np.float32)), shape


def median_combine(paths: list[Path], strip_rows: int = 128) -> np.ndarray:
    """Per-pixel median of many same-sized frames, built a strip of rows at a
    time so only `len(paths) * strip_rows` rows are in memory rather than every
    frame in full (50 flats at 8 MP would be ~1.7 GB). Same result as
    `np.median(np.stack(frames), axis=0)`."""
    shapes = {read_header(p)[1] for p in paths}
    if len(shapes) != 1:
        raise ValueError("frames differ in size: "
                         + ", ".join(f"{s[1]}x{s[0]}" for s in sorted(shapes)))
    h, w = shapes.pop()
    if any(is_raw(p) for p in paths):
        return _median_combine_raw(paths, (h, w), strip_rows)
    out = np.empty((h, w), np.float32)
    for r0 in range(0, h, strip_rows):
        r1 = min(h, r0 + strip_rows)
        strip = np.empty((len(paths), r1 - r0, w), np.float32)
        for i, p in enumerate(paths):
            with fits.open(p, memmap=False) as hdul:
                strip[i] = _image_hdu(hdul).section[r0:r1]
        out[r0:r1] = np.median(strip, axis=0)
    return out


def guess_object(folder: str | Path, max_files: int = 60) -> str:
    """The target name (OBJECT keyword) of the first light frame found under
    `folder`, or "" — only headers are read, and at most `max_files` files, so
    it is cheap even on a big or networked folder."""
    seen = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames.sort()
        for name in sorted(filenames):
            if not name.lower().endswith((".fits", ".fit", ".fts")):
                continue
            seen += 1
            if seen > max_files:
                return ""
            try:
                frame, _ = read_header(Path(dirpath) / name)
            except Exception:
                continue
            if is_light(frame) and frame.object_name:
                return frame.object_name
    return ""


def _median_combine_raw(paths: list[Path], shape: tuple[int, int],
                        strip_rows: int) -> np.ndarray:
    """`median_combine` for raw files, which can only be decoded whole: decode each
    one once into a scratch file on disk, then take the median strip by strip, so
    30 darks of a 24 MP sensor need disk space (~3 GB) rather than that much RAM."""
    import tempfile
    h, w = shape
    out = np.empty((h, w), np.float32)
    with tempfile.TemporaryDirectory(prefix="livestacker-") as tmp:
        stack = np.memmap(Path(tmp) / "frames.f32", dtype=np.float32, mode="w+",
                          shape=(len(paths), h, w))
        for i, p in enumerate(paths):
            stack[i] = load_frame(p).data
        for r0 in range(0, h, strip_rows):
            r1 = min(h, r0 + strip_rows)
            out[r0:r1] = np.median(stack[:, r0:r1], axis=0)
        del stack
    return out
