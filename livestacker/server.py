"""Web server: static preview page + a websocket that streams the live preview
and receives stretch-control updates.

The stack itself is shared, but the *display* settings (stretch, SCNR,
luminance blend) belong to each connected browser: every websocket carries its
own `StretchParams`, and each is sent a preview rendered with its own settings.

Layout of moving parts:

    source thread ─► frame queue ─► frame worker ─► Pipeline (stack)
                                                       │
    websocket in ─► settings ─► render thread ─────────┤  (reads the latest stack)
                                      │                │
                   asyncio loop ◄── threadsafe send ◄──┘
                         │
                         └─► websocket clients

Frames and renders run on separate threads so a slider move is answered
immediately even while a backlog of subs (a catch-up, a fast replay) is being
stacked.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import logging
import queue
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import websockets

from .calibration import Calibrator
from .config import StackConfig, StretchParams
from .fitsframe import load_frame
from .pipeline import Pipeline

log = logging.getLogger("livestacker.server")

WEB_ROOT = Path(__file__).resolve().parent / "web"
STALLED_BYTES = 32 * 1024 * 1024      # unsent data after which a browser is skipped


def params_from_query(query: str) -> StretchParams:
    """Build display settings from a URL query string (used by /snapshot.png so
    a download matches what *that* browser is showing). Unknown / malformed
    values fall back to the defaults."""
    raw = {k: v[-1] for k, v in parse_qs(query).items()}
    data: dict = {}
    for key, default in StretchParams().to_json().items():
        if key not in raw:
            continue
        if isinstance(default, bool):
            data[key] = raw[key].lower() in ("1", "true", "yes", "on")
        else:
            try:
                data[key] = float(raw[key])
            except ValueError:
                pass
    return StretchParams.from_json(data)


def _params_key(p: StretchParams) -> tuple:
    return tuple(p.to_json().items())


class LiveStacker:
    def __init__(self, cfg: StackConfig, host: str, ws_port: int, web_port: int,
                 calibrator: Calibrator | None = None):
        self.cfg = cfg
        self.host = host
        self.ws_port = ws_port
        self.web_port = web_port

        self.pipeline = Pipeline(cfg, calibrator)
        self._frames: "queue.Queue[Path]" = queue.Queue()
        self._pending = 0                  # frames submitted and not yet finished
        self._pending_lock = threading.Lock()
        self._stale = False                # frames stacked since the last recombine
        self._last_recombine_at = time.time()
        # connected browsers -> their own display settings. Mutated on the
        # event-loop thread, read by the worker threads, hence the lock.
        self._clients: dict = {}
        self._dirty: set = set()           # clients owed a fresh render
        self._clients_lock = threading.Lock()
        self._render_wake = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_preview_at = 0.0
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._renderer = threading.Thread(target=self._render_loop, daemon=True)
        self._stop = threading.Event()

    # -- public API used by the frame source --------------------------------

    def submit_file(self, path: Path) -> None:
        with self._pending_lock:
            self._pending += 1
        self._frames.put(Path(path))

    def _backlog(self) -> bool:
        """True while more frames are queued behind the one being handled."""
        with self._pending_lock:
            return self._pending > 1

    # -- frame worker thread -----------------------------------------------

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                path = self._frames.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._handle_frame(path)
            except Exception:
                log.exception("frame job failed (%s)", path.name)
            finally:
                with self._pending_lock:
                    self._pending -= 1

    def _handle_frame(self, path: Path) -> None:
        try:
            frame = load_frame(path)
        except Exception as exc:
            log.warning("skip %s: %s", path.name, exc)
            return
        # Don't rebuild the combined image after every sub of a backlog (catch-up,
        # fast replay) — only when the queue drains, and at least every few
        # seconds so progress stays visible.
        result = self.pipeline.ingest(frame, recombine=False)
        if result.ok:
            self._stale = True
            log.info("stacked %s  [%s]  overlap %.0f%%",
                     path.name, result.filter, result.valid_fraction * 100)
        elif result.skipped:
            log.info("%s %s", path.name, result.message)
        else:
            log.warning("rejected %s: %s", path.name, result.message)

        now = time.time()
        if self._stale and (not self._backlog()
                            or now - self._last_recombine_at >= self.cfg.backlog_refresh_s):
            self.pipeline.recombine()
            self._stale = False
            self._last_recombine_at = now
            try:
                self._push_preview()
            except Exception:
                log.exception("preview refresh failed")
        self._broadcast_status()

    # -- render thread -------------------------------------------------------

    def _render_loop(self) -> None:
        while not self._stop.is_set():
            if not self._render_wake.wait(0.5):
                continue
            self._render_wake.clear()
            try:
                self._render_dirty()
            except Exception:
                log.exception("render failed")

    # -- preview / status broadcasting ------------------------------------

    def _render_and_send(self, targets: list) -> None:
        """Render a preview for each (ws, params) pair — once per distinct set
        of settings, so browsers sharing settings share the work — and send each
        browser only its own image."""
        rendered: dict = {}
        for ws, params in targets:
            key = _params_key(params)
            if key not in rendered:
                try:
                    rendered[key] = self.pipeline.render_preview(params)
                except Exception as exc:     # one browser's settings must not
                    log.warning("render failed for a browser's settings: %s", exc)
                    rendered[key] = None     # starve the others
            if rendered[key] is not None:
                self._threadsafe(self._send_one, ws, rendered[key])

    def _push_preview(self) -> None:
        """A new frame landed: refresh every browser (throttled)."""
        now = time.time()
        if now - self._last_preview_at < self.cfg.min_preview_interval:
            return
        with self._clients_lock:
            targets = list(self._clients.items())
            self._dirty.clear()           # everyone is about to be refreshed
        if not targets:
            return
        self._last_preview_at = now
        self._render_and_send(targets)

    def _render_dirty(self) -> None:
        """A browser connected or changed its settings: refresh just those."""
        with self._clients_lock:
            targets = [(ws, self._clients[ws]) for ws in self._dirty
                       if ws in self._clients]
            self._dirty.clear()
        if targets:
            self._render_and_send(targets)

    def _mark_dirty(self, ws) -> None:
        with self._clients_lock:
            self._dirty.add(ws)
        self._render_wake.set()

    def _broadcast_status(self) -> None:
        self._threadsafe(self._broadcast_text,
                         json.dumps(self.pipeline.status()))

    def _threadsafe(self, fn, *args) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(fn, *args)

    @staticmethod
    def _stalled(ws) -> bool:
        """A browser that has stopped reading (background tab, bad link):
        `broadcast` never waits, so anything sent to it just piles up in memory."""
        transport = getattr(ws, "transport", None)
        return (transport is not None
                and transport.get_write_buffer_size() > STALLED_BYTES)

    def _send_one(self, ws, data: bytes) -> None:
        if not self._stalled(ws):
            websockets.broadcast([ws], data)

    def _broadcast_text(self, text: str) -> None:
        with self._clients_lock:
            clients = [c for c in self._clients if not self._stalled(c)]
        websockets.broadcast(clients, text)

    # -- websocket handler ----------------------------------------------

    async def _ws_handler(self, ws) -> None:
        with self._clients_lock:
            self._clients[ws] = StretchParams()      # replaced by the page's own
            total = len(self._clients)
        peer = getattr(ws, "remote_address", ("?",))[0]
        log.info("client connected (%s), %d total", peer, total)
        try:
            await ws.send(json.dumps(self.pipeline.status()))
            self._mark_dirty(ws)                     # show something right away

            async for message in ws:
                if isinstance(message, bytes):
                    continue
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "params":
                    with self._clients_lock:
                        self._clients[ws] = StretchParams.from_json(data)
                    self._mark_dirty(ws)
        except websockets.ConnectionClosed:
            pass
        finally:
            with self._clients_lock:
                self._clients.pop(ws, None)
                self._dirty.discard(ws)
                left = len(self._clients)
            log.info("client disconnected (%s), %d left", peer, left)

    # -- http (static + snapshot) --------------------------------------

    def _make_http_handler(self):
        stacker = self

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=str(WEB_ROOT), **kw)

            def log_message(self, *a):  # keep stdlib server quiet
                pass

            def do_GET(self):
                if self.path.startswith("/snapshot"):
                    # the page puts its own display settings in the query string
                    png = stacker.pipeline.snapshot_png(
                        params_from_query(urlparse(self.path).query))
                    if png is None:
                        self.send_error(503, "no image yet")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(png)))
                    self.end_headers()
                    self.wfile.write(png)
                    return
                if self.path == "/" or self.path == "":
                    self.path = "/index.html"
                # inject the ws port so the page knows where to connect
                if self.path == "/index.html":
                    html = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
                    html = html.replace("{{WS_PORT}}", str(stacker.ws_port))
                    body = html.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                super().do_GET()

        return Handler

    def _serve_http(self) -> None:
        handler = self._make_http_handler()
        httpd = http.server.ThreadingHTTPServer((self.host, self.web_port), handler)
        log.info("web UI at http://%s:%d/", self.host or "localhost", self.web_port)
        httpd.serve_forever()

    # -- lifecycle -----------------------------------------------------

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._worker.start()
        self._renderer.start()
        threading.Thread(target=self._serve_http, daemon=True).start()

        # the page only ever sends small JSON settings messages
        async with websockets.serve(self._ws_handler, self.host, self.ws_port,
                                    max_size=64 * 1024):
            log.info("websocket on ws://%s:%d", self.host or "localhost",
                     self.ws_port)
            await asyncio.Future()  # run forever

    def shutdown(self) -> None:
        self._stop.set()
        self._render_wake.set()
