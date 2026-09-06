"""Web server: static preview page + a websocket that streams the live preview
and receives stretch-control updates.

Layout of moving parts:

    source thread ─┐
                   ├─► job queue ─► worker thread ─► Pipeline
   websocket in  ──┘                     │
                                         ▼
                   asyncio loop ◄── threadsafe broadcast ─► websocket clients
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

import websockets

from .calibration import Calibrator
from .config import StackConfig, StretchParams
from .fitsframe import load_frame
from .pipeline import Pipeline

log = logging.getLogger("livestacker.server")

WEB_ROOT = Path(__file__).resolve().parent / "web"


class LiveStacker:
    def __init__(self, cfg: StackConfig, host: str, ws_port: int, web_port: int,
                 calibrator: Calibrator | None = None):
        self.cfg = cfg
        self.host = host
        self.ws_port = ws_port
        self.web_port = web_port

        self.pipeline = Pipeline(cfg, calibrator)
        self.params = StretchParams()
        self._jobs: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._latest_preview: bytes | None = None
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
                elif kind == "params":
                    self.params = payload  # type: ignore[assignment]
                    self._push_preview(force=True)
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

    def _push_preview(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_preview_at < self.cfg.min_preview_interval:
            return
        preview = self.pipeline.render_preview(self.params)
        if preview is None:
            return
        self._last_preview_at = now
        self._latest_preview = preview
        self._threadsafe(self._broadcast_bytes, preview)

    def _broadcast_status(self) -> None:
        self._threadsafe(self._broadcast_text,
                         json.dumps(self.pipeline.status()))

    def _threadsafe(self, fn, *args) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(fn, *args)

    def _broadcast_bytes(self, data: bytes) -> None:
        websockets.broadcast(self._clients, data)

    def _broadcast_text(self, text: str) -> None:
        websockets.broadcast(self._clients, text)

    # -- websocket handler ----------------------------------------------

    async def _ws_handler(self, ws) -> None:
        self._clients.add(ws)
        peer = getattr(ws, "remote_address", ("?",))[0]
        log.info("client connected (%s), %d total", peer, len(self._clients))
        try:
            await ws.send(json.dumps({"type": "params", **self.params.to_json()}))
            await ws.send(json.dumps(self.pipeline.status()))
            if self._latest_preview is not None:
                await ws.send(self._latest_preview)

            async for message in ws:
                if isinstance(message, bytes):
                    continue
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "params":
                    self._jobs.put(("params", StretchParams.from_json(data)))
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            log.info("client disconnected (%s), %d left", peer, len(self._clients))

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
                    png = stacker.pipeline.snapshot_png(stacker.params)
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
                    html = (WEB_ROOT / "index.html").read_text()
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
