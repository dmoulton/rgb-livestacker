"""The stacking pipeline: reference, per-channel accumulation, preview render.

Not thread-aware by itself. `server.py` funnels every call through a single
worker thread, so the accumulators never need locking.
"""

from __future__ import annotations

import io
import logging
import re
import threading
import time
from dataclasses import dataclass, replace

import numpy as np
import astroalign as aa
import sep
from PIL import Image

from .accumulator import ChannelStack
from .calibration import Calibrator
from .config import RGB_FILTERS, StackConfig, StretchParams
from .debayer import choose_method, debayer
from .fitsframe import NARROWBAND_FILTERS, Frame, is_light
from .processing import (
    SceneModeler, add_ha, apply_luminance, apply_stretch, auto_stretch, combine,
    lrgb_chroma, midtones_for, stretch_stats,
)

log = logging.getLogger("livestacker.pipeline")

# astroalign: be tolerant, these are short subs with varying star counts.
aa.MIN_MATCHES_FRACTION = 0.6
aa.PIXEL_TOL = 2
MAX_CONTROL_POINTS = 50        # stars (brightest first) used to find the transform


def _public_name(label: str) -> str:
    """A calibration source label ("/home/me/darks (12 frames)") reduced to the last
    folder / file name ("darks (12 frames)"): what the browser is shown shouldn't
    carry the machine's directory layout or user name."""
    m = re.match(r"^(.*?)( \(\d+ frames\))?$", label)
    base = re.split(r"[\\/]", m.group(1).rstrip("\\/"))[-1]
    return base + (m.group(2) or "")


def detect_stars(data: np.ndarray, detection_sigma: float = 5.0) -> np.ndarray:
    """(x, y) of the sources in `data`, brightest first — the same detection
    astroalign does internally, done once here so the reference's stars can be
    reused for every sub instead of being re-detected each time."""
    img = np.ascontiguousarray(data, dtype=np.float32)
    bkg = sep.Background(img)
    found = sep.extract(img - bkg.back(), detection_sigma * bkg.globalrms, minarea=5)
    found.sort(order="flux")
    return np.column_stack([found["x"], found["y"]])[::-1]


@dataclass
class IngestResult:
    ok: bool
    filter: str
    path: str
    message: str = ""
    valid_fraction: float = 0.0
    skipped: bool = False      # not a light frame: ignored, not a failure


class Pipeline:
    def __init__(self, cfg: StackConfig, calibrator: Calibrator | None = None):
        self.cfg = cfg
        self.calibrator = calibrator
        self.channels: dict[str, ChannelStack] = {}
        self.reference: np.ndarray | None = None
        self._ref_stars: np.ndarray | None = None
        self.reference_filter: str = ""
        self.reference_path: str = ""
        self._detected_object = ""     # OBJECT of the first light frame seen
        self._linear_rgb: np.ndarray | None = None
        self._linear_lum: np.ndarray | None = None
        self._auto_params: StretchParams | None = None
        self._chroma: np.ndarray | None = None
        self._pedestal = 0.0                 # the sky level in the linear image
        # per image ("rgb", "lrgb"): (black, sky level) for a browser's auto stretch
        self._stats: dict[str, tuple[float, float] | None] = {"rgb": None, "lrgb": None}
        self._linear_ha: np.ndarray | None = None
        self._scene = SceneModeler()
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.rejected = 0
        self._osc_logged = False
        self._osc = False               # a colour-camera session (Bayer frames)
        self.accepted = 0              # subs stacked (an OSC sub feeds three channels)

    # -- ingestion ---------------------------------------------------------

    def _register(self, finder: np.ndarray, planes: dict[str, np.ndarray]):
        """Align `planes` to the reference, finding the transform from `finder` (the
        image the stars are detected in: the plane itself for a mono frame, the
        colour-average for an OSC one). `astroalign`'s triangle matching is
        rotation-invariant, so a meridian flip needs nothing special — but a
        thin-cloud / soft-focus frame can still starve the star detector, so
        drop the detection threshold and retry before giving up. The reference's
        stars are detected once (when it is adopted), not for every sub.
        Returns ({name: aligned plane}, None) or (None, the error)."""
        last = None
        for det_sigma in (5, 3, 2):
            try:
                transform, _ = aa.find_transform(
                    finder, self._ref_stars, max_control_points=MAX_CONTROL_POINTS,
                    detection_sigma=det_sigma)
                return {name: aa.apply_transform(transform, img, self.reference,
                                                 fill_value=np.nan)[0]
                        for name, img in planes.items()}, None
            except Exception as exc:  # too few stars, no transform, etc.
                last = exc
        return None, last

    def _crop_edges(self, data: np.ndarray) -> np.ndarray:
        c = self.cfg
        if not (c.crop_top or c.crop_bottom or c.crop_left or c.crop_right):
            return data
        h, w = data.shape
        t = min(c.crop_top, h - 1)
        b = h - min(c.crop_bottom, h - 1 - t)
        l = min(c.crop_left, w - 1)
        r = w - min(c.crop_right, w - 1 - l)
        return data[t:b, l:r]

    def ingest(self, frame: Frame, recombine: bool = True) -> IngestResult:
        if not is_light(frame):
            return IngestResult(False, frame.filter, frame.path.name,
                                f"skipped: IMAGETYP={frame.imagetyp}", skipped=True)

        if not self._detected_object and frame.object_name:
            self._detected_object = frame.object_name[:120]

        if not self.cfg.narrowband_enabled and frame.filter in NARROWBAND_FILTERS:
            return IngestResult(False, frame.filter, frame.path.name,
                                "skipped: narrowband is switched off", skipped=True)

        data = self.calibrator.apply(frame) if self.calibrator else frame.data
        if frame.bayer:
            # colour sensor: calibrated as a mosaic, now split into R, G, B planes
            # (each then stacked as its own channel) and registered on their average
            method = choose_method(self.cfg.osc_debayer, data.shape)
            if not self._osc_logged:
                self._osc_logged = True
                log.info("colour sensor (%s, %dx%d): debayering %s", frame.bayer,
                         data.shape[1], data.shape[0],
                         "at full resolution" if method == "bilinear"
                         else "at half resolution (superpixel)")
            self._osc = True
            cube = debayer(data, frame.bayer, method)
            planes = {name: self._crop_edges(cube[i]) for i, name in enumerate("RGB")}
            data = (planes["R"] + planes["G"] + planes["B"]) / 3.0
            label = "RGB"
        else:
            data = self._crop_edges(data)
            planes = {frame.filter: data}
            label = frame.filter

        if self.reference is None:
            # Every later frame is aligned to this one, so don't adopt a frame
            # that has no stars to match against; wait for a better one.
            stars = detect_stars(data)
            if len(stars) < self.cfg.min_reference_stars:
                self.rejected += 1
                return IngestResult(
                    False, label, frame.path.name,
                    f"unusable as the reference: only {len(stars)} stars "
                    f"(need {self.cfg.min_reference_stars})")
            self._ref_stars = stars[:MAX_CONTROL_POINTS]
            self.reference = data.copy()
            self.reference_filter = label
            self.reference_path = frame.path.name
            aligned, valid = planes, np.ones(data.shape, bool)
            log.info("reference set from %s (%s)", frame.path.name, label)
        else:
            aligned, exc = self._register(data, planes)
            if aligned is None:
                self.rejected += 1
                return IngestResult(False, label, frame.path.name,
                                    f"registration failed: {exc}")
            valid = np.isfinite(next(iter(aligned.values())))
            frac = float(valid.mean())
            if frac < self.cfg.min_valid_fraction:
                self.rejected += 1
                return IngestResult(False, label, frame.path.name,
                                    f"low overlap ({frac:.0%})", frac)

        for name, img in aligned.items():
            ch = self.channels.setdefault(name, ChannelStack(name, self.cfg))
            ch.add(img, valid)
            ch.last_path = frame.path.name
        self.accepted += 1
        if recombine:
            self.recombine()
        return IngestResult(True, label, frame.path.name,
                            valid_fraction=float(valid.mean()))

    def recombine(self) -> None:
        """Rebuild the combined image from the current channel stacks. Called
        after each frame by default; a caller with a backlog can skip it per
        frame and call it once when the queue drains."""
        data = {name: (st.mean,
                       st.covered_mask(self.cfg.coverage_min_frac),
                       st.frames)
                for name, st in self.channels.items()}
        out = combine(data, self.cfg, self._scene)
        rgb, lum, ha = out.rgb, out.lum, out.ha
        chroma = None
        stats = {"rgb": None if rgb is None else stretch_stats(rgb), "lrgb": None}
        if rgb is not None and lum is not None:
            chroma = lrgb_chroma(rgb, floor=0.02 * out.reference, pedestal=out.pedestal)
            # the L-carrying image has a different brightness distribution, so
            # Auto needs its own black point / midtones when L is shown
            stats["lrgb"] = stretch_stats(
                apply_luminance(rgb, lum, 1.0, chroma, out.pedestal))
        with self._lock:
            self._linear_rgb = rgb
            self._linear_lum = lum
            self._linear_ha = ha
            self._chroma = chroma
            self._pedestal = out.pedestal
            self._stats = stats
            self._auto_params = auto_stretch(rgb) if rgb is not None else None

    # -- rendering -------------------------------------------------------

    def _resolve_params(self, params: StretchParams) -> StretchParams:
        if not params.auto:
            return params
        stats = self._stats["rgb"]
        if params.lum >= 0.5 and self._stats["lrgb"] is not None:
            stats = self._stats["lrgb"]
        if stats is None:
            return params
        black, sky = stats       # the strength is this browser's own choice
        return replace(params, black=black, midtones=midtones_for(sky, params.strength))

    def _render(self, params: StretchParams) -> Image.Image | None:
        with self._lock:
            # recombine() swaps in new arrays and nothing downstream writes to
            # these, so a reference is a safe snapshot — no copy needed
            rgb, lum, chroma = self._linear_rgb, self._linear_lum, self._chroma
            ha = self._linear_ha
            pedestal = self._pedestal
        if rgb is None:
            return None
        eff = self._resolve_params(params)
        if lum is not None and eff.lum > 1e-3:
            rgb = apply_luminance(rgb, lum, min(eff.lum, 1.0), chroma, pedestal)
        if ha is not None and eff.narrowband and eff.ha > 1e-3:
            rgb = add_ha(rgb, ha, eff.ha)
        return Image.fromarray(apply_stretch(rgb, eff), mode="RGB")

    def render_preview(self, params: StretchParams) -> bytes | None:
        img = self._render(params)
        if img is None:
            return None
        longest = max(img.size)
        if not params.full and longest > self.cfg.preview_max_px:
            scale = self.cfg.preview_max_px / longest
            img = img.resize(
                (round(img.width * scale), round(img.height * scale)),
                Image.BILINEAR,
            )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def snapshot_png(self, params: StretchParams) -> bytes | None:
        img = self._render(params)
        if img is None:
            return None
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # -- status --------------------------------------------------------

    def status(self) -> dict:
        # R, G, B first, then any other filters (L / narrowband) alphabetically
        order = {f: i for i, f in enumerate(RGB_FILTERS)}
        names = sorted(self.channels, key=lambda n: (order.get(n, len(order)), n))
        chans = {
            name: {
                "frames": self.channels[name].frames,
                "last": self.channels[name].last_path,
                "clipped_px": self.channels[name].rejected,
            }
            for name in names
        }
        total = self.accepted          # (not the channel sum: an OSC sub feeds three)
        cal = self.calibrator
        bias = _public_name(cal.master_bias.source) if cal and cal.master_bias else None
        dark = _public_name(cal.master_dark.source) if cal and cal.master_dark else None
        flats = sorted(cal.flats) if cal and cal.flats else []
        return {
            "type": "status",
            "osc": self._osc,
            "object": self.cfg.object_name or self._detected_object,
            "reference": self.reference_path,
            "reference_filter": self.reference_filter,
            "calibration": {
                "hot_pixel": self.cfg.hot_pixel_enabled,
                "sigma_clip": self.cfg.sigma_clip_enabled,
                "bias": bias,
                "dark": dark,
                "flats": flats,
                "background": self.cfg.background_enabled,
                "white_balance": (self._scene.wb_method
                                  if self.cfg.auto_white_balance else "off"),
            },
            "luminance": self._linear_lum is not None,
            "ha": self._linear_ha is not None,
            # full stack size + the preview limit, so a page can zoom in real
            # image pixels and know when it needs the full-resolution render
            "size": (None if self._linear_rgb is None
                     else [self._linear_rgb.shape[1], self._linear_rgb.shape[0]]),
            "preview_px": self.cfg.preview_max_px,
            "channels": chans,
            "total_frames": total,
            "rejected": self.rejected,
            "elapsed": round(time.time() - self.started_at, 1),
            "auto_params": self._auto_params.to_json() if self._auto_params else None,
            # (black, sky) per image, so a page can show the numbers its own
            # strength setting leads to
            "auto_stats": self._stats,
        }
