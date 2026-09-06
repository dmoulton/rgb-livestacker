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
from .config import StackConfig, StretchParams
from .fitsframe import Frame
from .processing import apply_stretch, auto_stretch, combine_rgb

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
    def __init__(self, cfg: StackConfig):
        self.cfg = cfg
        self.channels: dict[str, ChannelStack] = {}
        self.reference: np.ndarray | None = None
        self.reference_filter: str = ""
        self.reference_path: str = ""
        self._linear_rgb: np.ndarray | None = None
        self._auto_params: StretchParams | None = None
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.rejected = 0

    # -- ingestion ---------------------------------------------------------

    def ingest(self, frame: Frame) -> IngestResult:
        ch = self.channels.setdefault(frame.filter, ChannelStack(frame.filter))

        if self.reference is None:
            self.reference = frame.data.copy()
            self.reference_filter = frame.filter
            self.reference_path = frame.path.name
            aligned, valid = frame.data, np.ones(frame.data.shape, bool)
            log.info("reference set from %s (%s)", frame.path.name, frame.filter)
        else:
            try:
                aligned, _footprint = aa.register(
                    frame.data, self.reference, fill_value=np.nan,
                    detection_sigma=5,
                )
            except Exception as exc:  # too few stars, no transform, etc.
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
        data = {name: (st.mean, st.coverage)
                for name, st in self.channels.items()}
        rgb = combine_rgb(data, self.cfg)
        with self._lock:
            self._linear_rgb = rgb
            self._auto_params = auto_stretch(rgb) if rgb is not None else None

    # -- rendering -------------------------------------------------------

    def _resolve_params(self, params: StretchParams) -> StretchParams:
        if params.auto and self._auto_params is not None:
            merged = StretchParams(**self._auto_params.to_json())
            merged.gamma = params.gamma
            merged.contrast = params.contrast
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
        chans = {
            name: {
                "frames": st.frames,
                "last": st.last_path,
            }
            for name, st in sorted(self.channels.items())
        }
        total = sum(st.frames for st in self.channels.values())
        return {
            "type": "status",
            "reference": self.reference_path,
            "reference_filter": self.reference_filter,
            "channels": chans,
            "total_frames": total,
            "rejected": self.rejected,
            "elapsed": round(time.time() - self.started_at, 1),
            "auto_params": self._auto_params.to_json() if self._auto_params else None,
        }
