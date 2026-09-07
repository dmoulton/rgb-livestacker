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
from PIL import Image

from .accumulator import ChannelStack
from .calibration import Calibrator
from .config import RGB_FILTERS, StackConfig, StretchParams
from .fitsframe import Frame
from .processing import (
    SceneModeler, apply_stretch, auto_stretch, combine_rgb,
)

log = logging.getLogger("livestacker.pipeline")

# astroalign: be tolerant, these are short subs with varying star counts.
aa.MAX_CONTROL_POINTS = 60
aa.MIN_MATCHES_FRACTION = 0.6
aa.PIXEL_TOL = 2


@dataclass
class IngestResult:
    ok: bool
    filter: str
    path: str
    message: str = ""
    valid_fraction: float = 0.0


class Pipeline:
    def __init__(self, cfg: StackConfig, calibrator: Calibrator | None = None):
        self.cfg = cfg
        self.calibrator = calibrator
        self.channels: dict[str, ChannelStack] = {}
        self.reference: np.ndarray | None = None
        self.reference_filter: str = ""
        self.reference_path: str = ""
        self._linear_rgb: np.ndarray | None = None
        self._auto_params: StretchParams | None = None
        self._scene = SceneModeler()
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.rejected = 0

    # -- ingestion ---------------------------------------------------------

    def _register(self, data: np.ndarray):
        """Align `data` to the reference. `astroalign`'s triangle matching is
        rotation-invariant, so a meridian flip needs nothing special — but a
        thin-cloud / soft-focus frame can still starve the star detector, so
        drop the detection threshold and retry before giving up."""
        last = None
        for det_sigma in (5, 3, 2):
            try:
                registered, _ = aa.register(
                    data, self.reference, fill_value=np.nan,
                    detection_sigma=det_sigma,
                )
                return registered, None
            except Exception as exc:  # too few stars, no transform, etc.
                last = exc
        return None, last

    def ingest(self, frame: Frame) -> IngestResult:
        ch = self.channels.setdefault(
            frame.filter, ChannelStack(frame.filter, self.cfg))

        data = self.calibrator.apply(frame) if self.calibrator else frame.data

        if self.reference is None:
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

        ch.add(aligned, valid)
        ch.last_path = frame.path.name
        self._recombine()
        return IngestResult(True, frame.filter, frame.path.name,
                            valid_fraction=float(valid.mean()))

    def _recombine(self) -> None:
        data = {name: (st.mean,
                       st.covered_mask(self.cfg.coverage_min_frac),
                       st.frames)
                for name, st in self.channels.items()}
        rgb = combine_rgb(data, self.cfg, self._scene)
        with self._lock:
            self._linear_rgb = rgb
            self._auto_params = auto_stretch(rgb) if rgb is not None else None

    # -- rendering -------------------------------------------------------

    def _resolve_params(self, params: StretchParams) -> StretchParams:
        if params.auto and self._auto_params is not None:
            merged = StretchParams(**self._auto_params.to_json())
            merged.gamma = params.gamma
            merged.contrast = params.contrast
            merged.green = params.green
            return merged
        return params

    def render_preview(self, params: StretchParams) -> bytes | None:
        with self._lock:
            rgb = None if self._linear_rgb is None else self._linear_rgb.copy()
        if rgb is None:
            return None
        eff = self._resolve_params(params)
        img = Image.fromarray(apply_stretch(rgb, eff), mode="RGB")

        longest = max(img.size)
        if longest > self.cfg.preview_max_px:
            scale = self.cfg.preview_max_px / longest
            img = img.resize(
                (round(img.width * scale), round(img.height * scale)),
                Image.BILINEAR,
            )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def snapshot_png(self, params: StretchParams) -> bytes | None:
        with self._lock:
            rgb = None if self._linear_rgb is None else self._linear_rgb.copy()
        if rgb is None:
            return None
        img = Image.fromarray(apply_stretch(rgb, self._resolve_params(params)),
                              mode="RGB")
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
        dark = (self.calibrator.master_dark.source
                if self.calibrator and self.calibrator.master_dark else None)
        flats = (sorted(self.calibrator.flats)
                 if self.calibrator and self.calibrator.flats else [])
        return {
            "type": "status",
            "reference": self.reference_path,
            "reference_filter": self.reference_filter,
            "calibration": {
                "hot_pixel": self.cfg.hot_pixel_enabled,
                "sigma_clip": self.cfg.sigma_clip_enabled,
                "dark": dark,
                "flats": flats,
                "background": self.cfg.background_enabled,
                "white_balance": (self._scene.wb_method
                                  if self.cfg.auto_white_balance else "off"),
            },
            "channels": chans,
            "total_frames": total,
            "rejected": self.rejected,
            "elapsed": round(time.time() - self.started_at, 1),
            "auto_params": self._auto_params.to_json() if self._auto_params else None,
        }
