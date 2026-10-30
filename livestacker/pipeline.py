"""The stacking pipeline: reference, per-channel accumulation, preview render.

Not thread-aware by itself. `server.py` funnels every call through a single
worker thread, so the accumulators never need locking.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
import astroalign as aa
import sep
from PIL import Image

from .accumulator import ChannelStack
from .calibration import Calibrator
from .config import RGB_FILTERS, StackConfig, StretchParams
from .fitsframe import NARROWBAND_FILTERS, Frame, is_light
from .processing import (
    SceneModeler, add_ha, apply_luminance, apply_stretch, auto_stretch, combine,
    lrgb_chroma,
)

log = logging.getLogger("livestacker.pipeline")

# astroalign: be tolerant, these are short subs with varying star counts.
aa.MIN_MATCHES_FRACTION = 0.6
aa.PIXEL_TOL = 2
MAX_CONTROL_POINTS = 50        # stars (brightest first) used to find the transform


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
        self._auto_params_lrgb: StretchParams | None = None   # stretch of the L-carrying image
        self._chroma: np.ndarray | None = None
        self._linear_ha: np.ndarray | None = None
        self._scene = SceneModeler()
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.rejected = 0

    # -- ingestion ---------------------------------------------------------

    def _register(self, data: np.ndarray):
        """Align `data` to the reference. `astroalign`'s triangle matching is
        rotation-invariant, so a meridian flip needs nothing special — but a
        thin-cloud / soft-focus frame can still starve the star detector, so
        drop the detection threshold and retry before giving up. The reference's
        stars are detected once (when it is adopted), not for every sub."""
        last = None
        for det_sigma in (5, 3, 2):
            try:
                transform, _ = aa.find_transform(
                    data, self._ref_stars, max_control_points=MAX_CONTROL_POINTS,
                    detection_sigma=det_sigma)
                registered, _ = aa.apply_transform(
                    transform, data, self.reference, fill_value=np.nan)
                return registered, None
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
        data = self._crop_edges(data)

        if self.reference is None:
            # Every later frame is aligned to this one, so don't adopt a frame
            # that has no stars to match against; wait for a better one.
            stars = detect_stars(data)
            if len(stars) < self.cfg.min_reference_stars:
                self.rejected += 1
                return IngestResult(
                    False, frame.filter, frame.path.name,
                    f"unusable as the reference: only {len(stars)} stars "
                    f"(need {self.cfg.min_reference_stars})")
            self._ref_stars = stars[:MAX_CONTROL_POINTS]
            self.reference = data.copy()
            self.reference_filter = frame.filter
            self.reference_path = frame.path.name
            aligned, valid = data, np.ones(data.shape, bool)
            log.info("reference set from %s (%s)", frame.path.name, frame.filter)
        else:
            aligned, exc = self._register(data)
            if aligned is None:
                self.rejected += 1
                return IngestResult(False, frame.filter, frame.path.name,
                                    f"registration failed: {exc}")
            valid = np.isfinite(aligned)
            frac = float(valid.mean())
            if frac < self.cfg.min_valid_fraction:
                self.rejected += 1
                return IngestResult(False, frame.filter, frame.path.name,
                                    f"low overlap ({frac:.0%})", frac)

        ch = self.channels.setdefault(
            frame.filter, ChannelStack(frame.filter, self.cfg))
        ch.add(aligned, valid)
        ch.last_path = frame.path.name
        if recombine:
            self.recombine()
        return IngestResult(True, frame.filter, frame.path.name,
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
        chroma = auto_lrgb = None
        if rgb is not None and lum is not None:
            chroma = lrgb_chroma(rgb)
            # the L-carrying image has a different brightness distribution, so
            # Auto needs its own black point / midtones when L is shown
            auto_lrgb = auto_stretch(apply_luminance(rgb, lum, 1.0, chroma))
        with self._lock:
            self._linear_rgb = rgb
            self._linear_lum = lum
            self._linear_ha = ha
            self._chroma = chroma
            self._auto_params = auto_stretch(rgb) if rgb is not None else None
            self._auto_params_lrgb = auto_lrgb

    # -- rendering -------------------------------------------------------

    def _resolve_params(self, params: StretchParams) -> StretchParams:
        if params.auto and self._auto_params is not None:
            base = self._auto_params
            if params.lum >= 0.5 and self._auto_params_lrgb is not None:
                base = self._auto_params_lrgb
            merged = StretchParams(**base.to_json())
            merged.gamma = params.gamma
            merged.contrast = params.contrast
            merged.green = params.green
            merged.lum = params.lum
            merged.ha = params.ha
            merged.narrowband = params.narrowband
            return merged
        return params

    def _render(self, params: StretchParams) -> Image.Image | None:
        with self._lock:
            # recombine() swaps in new arrays and nothing downstream writes to
            # these, so a reference is a safe snapshot — no copy needed
            rgb, lum, chroma = self._linear_rgb, self._linear_lum, self._chroma
            ha = self._linear_ha
        if rgb is None:
            return None
        eff = self._resolve_params(params)
        if lum is not None and eff.lum > 1e-3:
            rgb = apply_luminance(rgb, lum, min(eff.lum, 1.0), chroma)
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
        total = sum(st.frames for st in self.channels.values())
        cal = self.calibrator
        bias = cal.master_bias.source if cal and cal.master_bias else None
        dark = cal.master_dark.source if cal and cal.master_dark else None
        flats = sorted(cal.flats) if cal and cal.flats else []
        return {
            "type": "status",
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
        }
