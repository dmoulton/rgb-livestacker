"""Web server: static preview page + a websocket that streams the live preview
and receives stretch-control updates.

The stack itself is shared, but the *display* settings (stretch, SCNR,
luminance blend) belong to each connected browser: every websocket carries its
own `StretchParams`, and each is sent a preview rendered with its own settings.

Layout of moving parts:

    source thread ─┐
                   ├─► job queue ─► worker thread ─► Pipeline
   websocket in  ──┘                     │
                                         ▼
                   asyncio loop ◄── threadsafe send ─► websocket clients
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
        self._jobs: "queue.Queue[tuple[str, object]]" = queue.Queue()
        # connected browsers -> their own display settings. Mutated on the
        # event-loop thread, read by the worker thread, hence the lock.
        self._clients: dict = {}
        self._dirty: set = set()           # clients owed a fresh render
        self._clients_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_preview_at = 0.0
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._stop = threading.Event()

    # -- public API used by the frame source --------------------------------

    def submit_file(self, path: Path) -> None:
        self._jobs.put(("frame", path))

    # -- worker thread -----------------------------------------------------

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                kind, payload = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if kind == "frame":
                    self._handle_frame(Path(payload))
                elif kind == "render":
                    self._render_dirty()
            except Exception:
                log.exception("worker job failed (%s)", kind)

    def _handle_frame(self, path: Path) -> None:
        try:
            frame = load_frame(path)
        except Exception as exc:
            log.warning("skip %s: %s", path.name, exc)
            return
        result = self.pipeline.ingest(frame)
        if result.ok:
            log.info("stacked %s  [%s]  overlap %.0f%%",
                     path.name, result.filter, result.valid_fraction * 100)
        else:
            log.warning("rejected %s: %s", path.name, result.message)
        self._push_preview()
        self._broadcast_status()

    # -- preview / status broadcasting ------------------------------------

    def _render_and_send(self, targets: list) -> None:
        """Render a preview for each (ws, params) pair — once per distinct set
        of settings, so browsers sharing settings share the work — and send each
        browser only its own image."""
        rendered: dict = {}
        for ws, params in targets:
            key = _params_key(params)
            if key not in rendered:
                rendered[key] = self.pipeline.render_preview(params)
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
        self._jobs.put(("render", None))

    def _broadcast_status(self) -> None:
        self._threadsafe(self._broadcast_text,
                         json.dumps(self.pipeline.status()))

    def _threadsafe(self, fn, *args) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(fn, *args)

    def _send_one(self, ws, data: bytes) -> None:
        websockets.broadcast([ws], data)

    def _broadcast_text(self, text: str) -> None:
        with self._clients_lock:
            clients = list(self._clients)
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
        threading.Thread(target=self._serve_http, daemon=True).start()

        async with websockets.serve(self._ws_handler, self.host, self.ws_port,
                                    max_size=None):
            log.info("websocket on ws://%s:%d", self.host or "localhost",
                     self.ws_port)
            await asyncio.Future()  # run forever

    def shutdown(self) -> None:
        self._stop.set()
