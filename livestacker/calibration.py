"""Frame calibration: optional master-dark subtraction + hot-pixel cleanup.

The hot-pixel pass needs no calibration frames and is the thing that stops
un-dark-subtracted hot pixels from smearing into "walking noise" as the mount
dithers. Master-dark subtraction is layered on top when a matching dark set is
available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from .fitsframe import Frame, load_frame
from .config import StackConfig

log = logging.getLogger("livestacker.calibration")

FITS_SUFFIXES = (".fits", ".fit", ".fts")


def cosmetic_correct(data: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """Replace isolated hot/cold pixels with the local 3x3 median.

    A pixel is corrected when its residual from the local median exceeds
    `sigma` times the robust noise level of the residual map — i.e. it stands
    out from its neighbours, not from the scene.
    """
    med = median_filter(data, size=3, mode="nearest")
    resid = data - med
    scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    if scale <= 0:
        return data
    outlier = np.abs(resid) > sigma * scale
    if not outlier.any():
        return data
    out = data.copy()
    out[outlier] = med[outlier]
    return out


class MasterDark:
    def __init__(self, data: np.ndarray, meta: dict, source: str):
        self.data = data.astype(np.float32)
        self.meta = meta
        self.source = source

    # -- construction ---------------------------------------------------

    @classmethod
    def build(cls, path: str | Path) -> "MasterDark":
        path = Path(path)
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.suffix.lower() in FITS_SUFFIXES)
            if not files:
                raise FileNotFoundError(f"no FITS files under {path}")
            frames = [load_frame(f) for f in files]
            data = np.median(np.stack([f.data for f in frames]), axis=0)
            meta = _meta(frames[0])
            log.info("master dark: median of %d frames from %s  %s",
                     len(files), path, _describe(meta))
            return cls(data, meta, f"{path} ({len(files)} frames)")

        frame = load_frame(path)
        meta = _meta(frame)
        log.info("master dark: single file %s  %s", path, _describe(meta))
        return cls(frame.data, meta, str(path))

    # -- application --------------------------------------------------

    def warnings_for(self, frame: Frame) -> list[str]:
        w: list[str] = []
        if self.data.shape != frame.data.shape:
            w.append(f"shape {self.data.shape} != light {frame.data.shape} "
                     "-- dark will be skipped")
            return w
        dm, lm = self.meta, _meta(frame)
        if dm["gain"] is not None and lm["gain"] is not None and dm["gain"] != lm["gain"]:
            w.append(f"gain {dm['gain']} != light {lm['gain']}")
        if dm["exptime"] and lm["exptime"] and abs(dm["exptime"] - lm["exptime"]) / lm["exptime"] > 0.02:
            w.append(f"exposure {dm['exptime']}s != light {lm['exptime']}s "
                     "-- scaling dark by the ratio (approximate: no bias frame)")
        if dm["temp"] is not None and lm["temp"] is not None and abs(dm["temp"] - lm["temp"]) > 3:
            w.append(f"sensor temp {dm['temp']}C != light {lm['temp']}C "
                     "-- dark current will not match")
        return w

    def subtract(self, data: np.ndarray, frame: Frame) -> np.ndarray:
        if self.data.shape != data.shape:
            return data
        scale = 1.0
        de, le = self.meta["exptime"], frame.exptime
        if de and le and abs(de - le) / le > 0.02:
            scale = le / de
        return data - self.data * scale


def _meta(frame: Frame) -> dict:
    return {
        "exptime": frame.exptime,
        "gain": frame.gain,
        "xbinning": frame.xbinning,
        "temp": frame.ccd_temp,
    }


def _describe(m: dict) -> str:
    return (f"exp={m['exptime']}s gain={m['gain']} "
            f"bin={m['xbinning']} temp={m['temp']}C")


class Calibrator:
    """Runs the enabled calibration steps on each incoming light frame."""

    def __init__(self, cfg: StackConfig, master_dark: MasterDark | None = None):
        self.cfg = cfg
        self.master_dark = master_dark
        self._dark_warned = False

    @property
    def active(self) -> bool:
        return self.master_dark is not None or self.cfg.hot_pixel_enabled

    def apply(self, frame: Frame) -> np.ndarray:
        data = frame.data
        if self.master_dark is not None:
            if not self._dark_warned:
                for msg in self.master_dark.warnings_for(frame):
                    log.warning("master dark: %s", msg)
                self._dark_warned = True
            data = self.master_dark.subtract(data, frame)
        if self.cfg.hot_pixel_enabled:
            data = cosmetic_correct(data, self.cfg.hot_pixel_sigma)
        return data
