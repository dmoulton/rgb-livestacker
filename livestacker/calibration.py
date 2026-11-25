"""Frame calibration: master bias, dark, flat field, and hot-pixel cleanup.

Order per light frame: subtract bias, subtract dark, divide by the (normalised)
flat for that filter, then a per-frame hot/cold pixel cosmetic pass. The
hot-pixel pass needs no calibration frames and is what stops un-dark-subtracted
hot pixels from smearing into "walking noise" as the mount dithers; bias, dark
and flat are layered on when matching calibration sets are available. When a
bias is supplied the dark master and the flats are bias-subtracted too, so an
exposure-scaled dark is then correct rather than approximate.
"""

from __future__ import annotations

import collections
import logging
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from .debayer import bayer_planes
from .fitsframe import Frame, load_frame, median_combine, read_header
from .config import StackConfig

log = logging.getLogger("livestacker.calibration")

from .fitsframe import FRAME_SUFFIXES as FITS_SUFFIXES   # FITS and Canon raw: both can be calibration frames
from .fitsframe import RAW_SUFFIXES


def cosmetic_correct(data: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """Replace isolated hot/cold pixels with the local 3x3 median.

    A pixel is corrected when its residual from the local median exceeds
    `sigma` times the robust noise level of the residual map — i.e. it stands
    out from its neighbours, not from the scene.
    """
    med = median_filter(data, size=3, mode="nearest")
    resid = data - med
    # the noise level only needs a sample of the residual map, not all of it
    sample = resid[::3, ::3]
    scale = 1.4826 * np.median(np.abs(sample - np.median(sample)))
    if scale <= 0:
        return data
    outlier = np.abs(resid) > sigma * scale
    if not outlier.any():
        return data
    out = data.copy()
    out[outlier] = med[outlier]
    return out


def cosmetic_correct_bayer(data: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """`cosmetic_correct` for a colour-sensor mosaic. Next to a pixel in a Bayer
    mosaic sit pixels of *other* colours, so a plain 3x3 median would pull a hot
    pixel toward the wrong colour (and flag every bright-colour pixel as an
    outlier); each of the four same-colour sub-planes is cleaned on its own."""
    out = data.copy()
    for plane in bayer_planes(out):
        plane[...] = cosmetic_correct(plane, sigma)
    return out


FULL_SCALE = 65535.0           # 16-bit camera counts
# With no bias to compare against, a dark whose typical value is under this is
# taken to have had its baseline removed: cameras' baselines are higher than this,
# and the dark current of a cooled sensor is lower.
BASELINE_FLOOR = 64.0


def _is_unit_scaled(data: np.ndarray) -> bool:
    """Siril, PixInsight and others save masters as 0..1 floating point, 1.0
    meaning the camera's full scale. Camera counts never all sit below 1."""
    top = float(np.nanmax(data)) if data.size else 0.0
    return 0.0 < top <= 1.0001


def _to_counts(data: np.ndarray, label: str) -> np.ndarray:
    """Bring a 0..1 master onto the lights' scale (camera counts), else as is."""
    if not _is_unit_scaled(data):
        return data
    log.info("%s is saved on a 0..1 scale: converted to camera counts", label)
    return (data * FULL_SCALE).astype(np.float32)


def _combine_files(files: list[Path], label: str) -> tuple[np.ndarray, dict]:
    data = load_frame(files[0]).data if len(files) == 1 else median_combine(files)
    return _to_counts(data, label), _meta(read_header(files[0])[0])


def _frames_under(path: Path) -> list[Path]:
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in FITS_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"no FITS files under {path}")
    return files


def _combine(path: str | Path) -> tuple[np.ndarray, dict, str]:
    """Median-combine every FITS under `path` (or load it, if it's one file).
    Returns (data in camera counts, meta-from-first-frame, human label)."""
    path = Path(path)
    if path.is_dir():
        files = _frames_under(path)
        data, meta = _combine_files(files, str(path))
        return data, meta, f"{path} ({len(files)} frames)"
    data, meta = _combine_files([path], str(path))
    return data, meta, str(path)


def _differs(a, b) -> bool:
    """Two header values that are both known and not the same."""
    return a is not None and b is not None and a != b


class MasterBias:
    """The camera's fixed read-noise / offset pattern (very short exposures)."""

    def __init__(self, data: np.ndarray, meta: dict, source: str):
        self.data = data.astype(np.float32)
        self.meta = meta
        self.source = source
        self.level = float(np.median(self.data[::4, ::4]))

    @classmethod
    def build(cls, path: str | Path) -> "MasterBias":
        data, meta, label = _combine(path)
        log.info("master bias: %s  %s", label, _describe(meta))
        return cls(data, meta, label)

    def for_frame(self, frame: Frame, shape=None) -> tuple["MasterBias | None", list[str]]:
        """(this bias if it can be used on `frame`, what's wrong with it)."""
        shape = shape or frame.data.shape
        if self.data.shape != shape:
            return None, [f"{_name(self.source)} is {_size(self.data.shape)} but the lights "
                          f"are {_size(shape)}: NOT applied"]
        lm = _meta(frame)
        notes = [f"{k} {self.meta[k]} != light {lm[k]}"
                 for k in ("gain", "offset", "xbinning") if _differs(self.meta.get(k), lm.get(k))]
        return self, notes

    def warnings_for(self, frame: Frame) -> list[str]:
        return self.for_frame(frame)[1]

    def subtract(self, data: np.ndarray) -> np.ndarray:
        if self.data.shape != data.shape:
            return data
        return data - self.data


def _baseline_removed(dark: np.ndarray, bias: "MasterBias | None", meta: dict) -> bool:
    """Has the camera's baseline (bias level) already been taken out of this dark?
    Such a dark sits near zero, instead of at about the bias level or above."""
    level = float(np.median(dark[::4, ::4]))
    if bias is not None and bias.data.shape == dark.shape and bias.level > 50:
        return level < 0.5 * bias.level
    # with no bias to compare to: a camera's counts don't sit this close to zero
    # (raw files are excluded: their black level is removed as they are read,
    # from lights and darks alike)
    return not meta.get("raw") and level < BASELINE_FLOOR


class MasterDark:
    def __init__(self, data: np.ndarray, meta: dict, source: str,
                 debiased: bool = False, baseline_removed: bool = False):
        self.data = data.astype(np.float32)
        self.meta = meta
        self.source = source
        self.debiased = debiased   # True -> `data` is pure dark current
        self.baseline_removed = baseline_removed   # ... and it arrived that way

    # -- construction ---------------------------------------------------

    @classmethod
    def build(cls, path: str | Path,
              bias: "MasterBias | None" = None) -> "MasterDark":
        data, meta, label = _combine(path)
        return cls.from_data(data, meta, label, bias)

    @classmethod
    def from_data(cls, data: np.ndarray, meta: dict, label: str,
                  bias: "MasterBias | None" = None) -> "MasterDark":
        removed = _baseline_removed(data, bias, meta)
        debiased = removed
        if not removed and bias is not None and bias.data.shape == data.shape:
            data = data - bias.data
            debiased = True
        log.info("master dark: %s  %s%s", label, _describe(meta),
                 "  (baseline already removed)" if removed
                 else "  (bias-subtracted)" if debiased else "")
        return cls(data, meta, label, debiased, removed)

    # -- application --------------------------------------------------

    def _exposure_differs(self, frame: Frame) -> bool:
        de, le = self.meta["exptime"], frame.exptime
        return bool(de and le and abs(de - le) / le > 0.02)

    def for_frame(self, frame: Frame, bias_applied: bool = True
                  ) -> tuple["MasterDark | None", list[str]]:
        """(this dark if it can be used on `frame`, what's wrong with using it)."""
        if self.data.shape != frame.data.shape:
            return None, [f"{_name(self.source)} is {_size(self.data.shape)} but the lights "
                          f"are {_size(frame.data.shape)}: NOT applied"]
        w: list[str] = []
        dm, lm = self.meta, _meta(frame)
        for k in ("gain", "offset", "xbinning"):
            if _differs(dm.get(k), lm.get(k)):
                w.append(f"{k} {dm[k]} != light {lm[k]}")
        if self._exposure_differs(frame):
            how = ("scaled by the exposure ratio" if self.debiased else
                   "used as it is: without a matching bias its baseline can't be separated "
                   "from the dark current, so it can't be scaled")
            w.append(f"exposure {dm['exptime']}s != light {lm['exptime']}s: {how}")
        if dm["temp"] is not None and lm["temp"] is not None and abs(dm["temp"] - lm["temp"]) > 3:
            w.append(f"sensor temp {dm['temp']}C != light {lm['temp']}C: "
                     "dark current will not match")
        if self.baseline_removed and not bias_applied:
            w.append("this dark has the camera's baseline already removed: add the matching "
                     "master bias, or the baseline stays in the lights")
        return self, w

    def warnings_for(self, frame: Frame) -> list[str]:
        return self.for_frame(frame)[1]

    def subtract(self, data: np.ndarray, frame: Frame) -> np.ndarray:
        if self.data.shape != data.shape:
            return data
        scale = 1.0
        # only pure dark current scales with exposure: scaling a dark that still
        # holds the baseline would scale the baseline too
        if self.debiased and self._exposure_differs(frame):
            scale = frame.exptime / self.meta["exptime"]
        return data - self.data * scale


# -- libraries: a folder holding masters for several settings ---------------

def _scan(path: Path, skip_types: tuple[str, ...]) -> list[tuple[dict, list[Path]]]:
    """Group the files under `path` by what a light must match: size, exposure,
    gain, offset, binning and (to within a few degrees) sensor temperature."""
    entries = []
    for f in _frames_under(path):
        header, shape = read_header(f)
        entries.append((f, header, shape))
    wanted = [e for e in entries
              if not any(t in e[1].imagetyp.upper() for t in skip_types)]
    groups: dict[tuple, list] = collections.defaultdict(list)
    for f, header, shape in (wanted or entries):
        m = _meta(header)
        key = (shape, round(m["exptime"] or 0.0, 3), m["gain"], m["offset"], m["xbinning"])
        groups[key].append((m["temp"], f, m))
    out = []
    for key, members in groups.items():
        members.sort(key=lambda t: (t[0] is None, t[0] or 0.0))
        cluster = [members[0]]
        for item in members[1:]:
            if item[0] is not None and cluster[-1][0] is not None and item[0] - cluster[-1][0] > 3:
                out.append(cluster)
                cluster = []
            cluster.append(item)
        out.append(cluster)
    return [(dict(c[0][2], shape=read_header(c[0][1])[1]), [f for _, f, _ in c]) for c in out]


def _settings(m: dict) -> str:
    bits = [f"{m['exptime']:g}s" if m.get("exptime") else None,
            f"gain {m['gain']:g}" if m.get("gain") is not None else None,
            f"offset {m['offset']:g}" if m.get("offset") is not None else None,
            f"{m['temp']:.0f}C" if m.get("temp") is not None else None]
    return " ".join(b for b in bits if b)


class _Library:
    """A folder of masters (or of calibration subs) for several camera settings.
    The right one is picked for each light, and combined when first needed."""

    kind = ""
    skip_types: tuple[str, ...] = ()

    def __init__(self, path: Path, groups: list):
        self.path = path
        self.groups = groups
        self.source = f"{path} (library of {len(groups)})"
        self._made: dict = {}
        self._picked: dict = {}

    def _matching(self, frame: Frame, shape) -> list[int]:
        lm = _meta(frame)
        return [i for i, (m, _) in enumerate(self.groups)
                if m["shape"] == shape
                and not any(_differs(m.get(k), lm.get(k))
                            for k in ("gain", "offset", "xbinning"))]

    def _none(self, frame: Frame, shape) -> list[str]:
        have = sorted({f"{_size(m['shape'])} {_settings({**m, 'exptime': 0, 'temp': None})}"
                       for m, _ in self.groups})
        return [f"no {self.kind} in {_name(str(self.path))} matches the lights "
                f"({_size(shape)} {_settings(_meta(frame))}): NOT applied. "
                f"It has: {'; '.join(have)}"]


class BiasLibrary(_Library):
    kind = "bias"
    skip_types = ("DARK", "FLAT", "LIGHT")

    def for_frame(self, frame: Frame, shape=None) -> tuple["MasterBias | None", list[str]]:
        shape = shape or frame.data.shape
        found = self._matching(frame, shape)
        if not found:
            return None, self._none(frame, shape)
        i = min(found, key=lambda i: self.groups[i][0].get("exptime") or 0.0)
        if i not in self._made:
            meta, files = self.groups[i]
            label = str(files[0]) if len(files) == 1 else f"{files[0].parent} ({len(files)} frames)"
            data, meta_ = _combine_files(files, label)
            log.info("master bias: %s  %s", label, _describe(meta_))
            self._made[i] = MasterBias(data, meta_, label)
        return self._made[i], []


class DarkLibrary(_Library):
    kind = "dark"
    skip_types = ("BIAS", "OFFSET", "FLAT", "LIGHT")

    def __init__(self, path: Path, groups: list, bias=None):
        super().__init__(path, groups)
        self.bias = bias

    def _closest(self, found: list[int], frame: Frame) -> int:
        """Nearest exposure first (as a ratio), then nearest temperature."""
        def score(i):
            m = self.groups[i][0]
            exp, light = m.get("exptime") or 0.0, frame.exptime or 0.0
            apart = abs(np.log(exp / light)) if exp > 0 and light > 0 else 99.0
            temps = (abs(m["temp"] - frame.ccd_temp)
                     if m.get("temp") is not None and frame.ccd_temp is not None else 0.0)
            return (round(float(apart), 2), temps)
        return min(found, key=score)

    def for_frame(self, frame: Frame, bias_applied: bool = True
                  ) -> tuple["MasterDark | None", list[str]]:
        found = self._matching(frame, frame.data.shape)
        if not found:
            return None, self._none(frame, frame.data.shape)
        i = self._closest(found, frame)
        if i not in self._made:
            meta, files = self.groups[i]
            label = str(files[0]) if len(files) == 1 else f"{files[0].parent} ({len(files)} frames)"
            data, meta_ = _combine_files(files, label)
            bias = self.bias.for_frame(frame)[0] if self.bias else None
            self._made[i] = MasterDark.from_data(data, meta_, label, bias)
        return self._made[i].for_frame(frame, bias_applied)


def load_bias(path: str | Path) -> "MasterBias | BiasLibrary":
    """A bias file, a folder of bias subs, or a folder of masters for several
    settings (the matching one is then picked per light)."""
    path = Path(path)
    if path.is_dir():
        groups = _scan(path, BiasLibrary.skip_types)
        if len(groups) > 1:
            log.info("bias library: %s (%d sets of settings)", path, len(groups))
            return BiasLibrary(path, groups)
        return MasterBias.build(groups[0][1][0] if len(groups[0][1]) == 1 else path)
    return MasterBias.build(path)


def load_dark(path: str | Path, bias=None) -> "MasterDark | DarkLibrary":
    """A dark file, a folder of dark subs, or a library of masters. `bias` is
    whatever `load_bias` returned (or None)."""
    path = Path(path)
    if path.is_dir():
        groups = _scan(path, DarkLibrary.skip_types)
        if len(groups) > 1:
            log.info("dark library: %s (%d sets of settings)", path, len(groups))
            return DarkLibrary(path, groups, bias)
        files = groups[0][1]
        meta_frame = read_header(files[0])[0]
        label = f"{files[0]}" if len(files) == 1 else f"{path} ({len(files)} frames)"
        data, meta = _combine_files(files, label)
    else:
        meta_frame = read_header(path)[0]
        data, meta, label = _combine(path)
    if isinstance(bias, BiasLibrary):                 # pick the bias taken like this dark
        bias = bias.for_frame(meta_frame, shape=data.shape)[0]
    return MasterDark.from_data(data, meta, label, bias)


def _name(label: str) -> str:
    """A source label without its folders: what is safe to show in a browser."""
    m = re.match(r"^(.*?)( \((?:library of )?\d+(?: frames)?\))?$", label)
    base = re.split(r"[\\/]", m.group(1).rstrip("\\/"))[-1]
    return base + (m.group(2) or "")


def _size(shape) -> str:
    return f"{shape[1]}x{shape[0]}"


class MasterFlat:
    """A per-filter flat, normalised so its median is 1.0 — dividing a light by
    it corrects vignetting and dust shadows without changing the signal scale.
    """

    def __init__(self, data: np.ndarray, meta: dict, source: str,
                 bayer: str | None = None):
        flat = data.astype(np.float32)
        if bayer:
            # a colour sensor's flat is a mosaic whose colours sit at different
            # levels: normalise each of the four sub-planes on its own, so the
            # flat corrects vignetting and dust and leaves the colour balance alone
            for plane in bayer_planes(flat):
                plane /= float(np.median(plane)) or 1.0
        else:
            flat /= float(np.median(flat)) or 1.0
        # guard against blow-ups where the flat is near zero (heavy vignette
        # corners, dead pixels)
        flat[flat < 0.2] = 1.0
        self.data = flat
        self.meta = meta
        self.source = source
        self.bayer = bayer

    def warnings_for(self, frame: Frame) -> list[str]:
        w: list[str] = []
        if self.data.shape != frame.data.shape:
            w.append(f"shape {self.data.shape} != light {frame.data.shape} "
                     "-- flat will be skipped")
            return w
        fb, lb = self.meta["xbinning"], frame.xbinning
        if fb is not None and lb is not None and fb != lb:
            w.append(f"binning {fb} != light {lb}")
        return w

    def divide(self, data: np.ndarray, frame: Frame) -> np.ndarray:
        if self.data.shape != data.shape:
            return data
        return data / self.data


def build_flats(path: str | Path,
                bias: "MasterBias | None" = None) -> dict[str, MasterFlat]:
    """Build one `MasterFlat` per filter found under `path` (a single FITS file
    or a folder of raw flats). Keyed by normalised filter name. If a bias is
    supplied it is subtracted before the flat is normalised."""
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in FITS_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"no FITS files under {path}")
    else:
        files = [path]

    by_filter: dict[str, list[Frame]] = collections.defaultdict(list)
    for f in files:
        fr = read_header(f)[0]
        by_filter[fr.filter].append(fr)

    out: dict[str, MasterFlat] = {}
    for filt, frames in by_filter.items():
        data = _to_counts(median_combine([fr.path for fr in frames]), f"flat [{filt}]")
        debiased = False
        use = bias.for_frame(frames[0], shape=data.shape)[0] if bias is not None else None
        if use is not None:
            data = data - use.data
            debiased = True
        out[filt] = MasterFlat(data, _meta(frames[0]),
                               f"{path} ({filt}, {len(frames)} frames)",
                               bayer=frames[0].bayer)
        log.info("master flat [%s]: median of %d frames  %s%s",
                 filt, len(frames), _describe(_meta(frames[0])),
                 "  (bias-subtracted)" if debiased else "")
    return out


def _meta(frame: Frame) -> dict:
    return {
        "exptime": frame.exptime,
        "gain": frame.gain,
        "offset": frame.offset,
        "xbinning": frame.xbinning,
        "raw": frame.path.suffix.lower() in RAW_SUFFIXES if hasattr(frame.path, "suffix") else False,
        "temp": frame.ccd_temp,
    }


def _describe(m: dict) -> str:
    return (f"exp={m['exptime']}s gain={m['gain']} "
            f"bin={m['xbinning']} temp={m['temp']}C")


class Calibrator:
    """Runs the enabled calibration steps on each incoming light frame.

    `master_bias` / `master_dark` may be single masters or libraries. Anything
    wrong (a dark that doesn't fit, one that was skipped) is logged once and
    kept in `problems`, which the web page shows; `applied` says what the latest
    light actually got."""

    def __init__(self, cfg: StackConfig, master_dark: "MasterDark | DarkLibrary | None" = None,
                 flats: dict[str, MasterFlat] | None = None,
                 master_bias: "MasterBias | BiasLibrary | None" = None):
        self.cfg = cfg
        self.master_bias = master_bias
        self.master_dark = master_dark
        self.flats = flats or {}
        self.problems: list[str] = []
        self.applied: dict[str, str | None] = {}
        self._flat_warned: set[str] = set()

    @property
    def active(self) -> bool:
        return (self.master_bias is not None or self.master_dark is not None
                or bool(self.flats) or self.cfg.hot_pixel_enabled)

    def _problem(self, what: str, message: str) -> None:
        text = f"{what}: {message}"
        if text not in self.problems and len(self.problems) < 20:
            self.problems.append(text)
            log.warning("%s", text)

    def _flat_for(self, frame: Frame) -> MasterFlat | None:
        if not self.flats:
            return None
        if frame.filter in self.flats:
            return self.flats[frame.filter]
        if len(self.flats) == 1:
            return next(iter(self.flats.values()))
        return None

    def apply(self, frame: Frame) -> np.ndarray:
        data = frame.data
        if (self.master_bias is not None or self.master_dark is not None) \
                and _is_unit_scaled(data):
            data = data * np.float32(FULL_SCALE)      # a 0..1 light: same units as masters
        bias = None
        if self.master_bias is not None:
            bias, notes = self.master_bias.for_frame(frame)
            for msg in notes:
                self._problem("master bias", msg)
            self.applied["bias"] = None if bias is None else bias.source
            if bias is not None:
                data = bias.subtract(data)

        if self.master_dark is not None:
            dark, notes = self.master_dark.for_frame(frame, bias_applied=bias is not None)
            for msg in notes:
                self._problem("master dark", msg)
            self.applied["dark"] = None if dark is None else dark.source
            if dark is not None:
                data = dark.subtract(data, frame)

        flat = self._flat_for(frame)
        if flat is not None:
            if frame.filter not in self._flat_warned:
                for msg in flat.warnings_for(frame):
                    log.warning("master flat [%s]: %s", frame.filter, msg)
                self._flat_warned.add(frame.filter)
            data = flat.divide(data, frame)
        elif self.flats and frame.filter not in self._flat_warned:
            log.warning("no master flat for filter %s (have: %s)",
                        frame.filter, ", ".join(sorted(self.flats)))
            self._flat_warned.add(frame.filter)

        if self.cfg.hot_pixel_enabled:
            fix = cosmetic_correct_bayer if frame.bayer else cosmetic_correct
            data = fix(data, self.cfg.hot_pixel_sigma)
        return data
