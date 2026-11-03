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
import hashlib
import hmac
import http.server
import json
import logging
import queue
import secrets
import socket
import threading
import time
from http.cookies import SimpleCookie
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
SESSION_COOKIE = "ls_session"
SESSION_DAYS = 30
LOGIN_MAX_FAILS = 5                   # wrong passwords in a row before a lockout
LOGIN_LOCKOUT_S = 30


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


def _pipe(a: socket.socket, b: socket.socket) -> None:
    """Copy bytes both ways between two sockets until either side closes."""
    def pump(src, dst):
        try:
            while True:
                chunk = src.recv(65536)
                if not chunk:
                    break
                dst.sendall(chunk)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    t = threading.Thread(target=pump, args=(b, a), daemon=True)
    t.start()
    pump(a, b)
    t.join(timeout=2)
    b.close()


def _params_key(p: StretchParams) -> tuple:
    return tuple(p.to_json().items())


class LiveStacker:
    def __init__(self, cfg: StackConfig, host: str, ws_port: int, web_port: int,
                 calibrator: Calibrator | None = None, password: str = ""):
        self.cfg = cfg
        self.password = password          # "" = no password, open to anyone
        self._sessions: dict[str, float] = {}      # login cookie -> expiry time
        self._login_fails = 0
        self._locked_until = 0.0
        self._auth_lock = threading.Lock()
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

    # -- password protection ------------------------------------------------

    def _ws_host(self) -> str:
        """Where the websocket server listens. With a password it is local only:
        the only way in from outside is the web port's /ws relay, which checks
        the login cookie — otherwise the raw port would bypass the password."""
        if self.password:
            return "127.0.0.1"
        return self.host

    def check_login(self, candidate: str) -> str:
        """'ok', 'bad' or 'locked' (too many wrong guesses in a row)."""
        with self._auth_lock:
            if time.time() < self._locked_until:
                return "locked"
            # compare digests so the check takes the same time wherever it differs
            same = hmac.compare_digest(
                hashlib.sha256(candidate.encode()).digest(),
                hashlib.sha256(self.password.encode()).digest())
            if same:
                self._login_fails = 0
                return "ok"
            self._login_fails += 1
            if self._login_fails >= LOGIN_MAX_FAILS:
                self._login_fails = 0
                self._locked_until = time.time() + LOGIN_LOCKOUT_S
            return "bad"

    def new_session(self) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._auth_lock:
            self._sessions = {t: e for t, e in self._sessions.items() if e > now}
            self._sessions[token] = now + SESSION_DAYS * 86400
        return token

    def valid_session(self, token: str | None) -> bool:
        if not self.password:
            return True
        if not token:
            return False
        with self._auth_lock:
            return self._sessions.get(token, 0.0) > time.time()

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

            def _relay_websocket(self):
                """Pass a websocket upgrade on the web port through to the real
                websocket server, so the page needs only one port (a tunnel or
                reverse proxy forwarding just the web port works)."""
                self.close_connection = True
                if "websocket" not in self.headers.get("Upgrade", "").lower():
                    self.send_error(426, "websocket upgrade required")
                    return
                target = stacker._ws_host()
                if target in ("", "0.0.0.0", "::"):
                    target = "127.0.0.1"
                try:
                    up = socket.create_connection((target, stacker.ws_port), timeout=5)
                except OSError:
                    self.send_error(502, "websocket server unavailable")
                    return
                up.settimeout(None)
                head = f"GET / HTTP/1.1\r\n" + "".join(
                    f"{k}: {v}\r\n" for k, v in self.headers.items()) + "\r\n"
                up.sendall(head.encode("latin-1"))
                _pipe(self.connection, up)

            # -- password gate ---------------------------------------------

            def _authorized(self) -> bool:
                if not stacker.password:
                    return True
                try:
                    jar = SimpleCookie(self.headers.get("Cookie", ""))
                except Exception:
                    return False
                morsel = jar.get(SESSION_COOKIE)
                return stacker.valid_session(morsel.value if morsel else None)

            def _send_login(self, status: int = 200, message: str = ""):
                body = (WEB_ROOT / "login.html").read_text(encoding="utf-8").replace(
                    "{{MESSAGE}}", f'<div class="err">{message}</div>' if message else "")
                data = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                if not stacker.password or self.path.split("?")[0] != "/login":
                    self.send_error(404)
                    return
                try:
                    n = min(int(self.headers.get("Content-Length", "0")), 4096)
                except ValueError:
                    n = 0
                form = parse_qs(self.rfile.read(n).decode("utf-8", "replace"),
                                keep_blank_values=True)
                result = stacker.check_login(form.get("password", [""])[-1])
                if result == "ok":
                    token = stacker.new_session()
                    cookie = (f"{SESSION_COOKIE}={token}; HttpOnly; SameSite=Strict; "
                              f"Path=/; Max-Age={SESSION_DAYS * 86400}")
                    if self.headers.get("X-Forwarded-Proto", "").lower() == "https":
                        cookie += "; Secure"     # (behind an https tunnel)
                    self.send_response(303)
                    self.send_header("Location", "/")
                    self.send_header("Set-Cookie", cookie)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif result == "locked":
                    self._send_login(429, "Too many attempts. Wait a little and try again.")
                else:
                    self._send_login(401, "Incorrect password.")

            def do_HEAD(self):
                if not self._authorized():
                    self.send_error(401)
                    return
                super().do_HEAD()

            def do_GET(self):
                path = self.path.split("?")[0]
                if not self._authorized():
                    if path in ("/", "/index.html", "/login"):
                        self._send_login()
                    elif path == "/ws":
                        # a websocket client wants a plain HTTP/1.1 refusal
                        self.close_connection = True
                        self.wfile.write(b"HTTP/1.1 401 Unauthorized\r\n"
                                         b"Content-Length: 0\r\n"
                                         b"Connection: close\r\n\r\n")
                    else:
                        self.send_error(401)
                    return
                if path == "/login":                  # already signed in
                    self.send_response(303)
                    self.send_header("Location", "/")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if path == "/ws":
                    self._relay_websocket()
                    return
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
        async with websockets.serve(self._ws_handler, self._ws_host(), self.ws_port,
                                    max_size=64 * 1024):
            log.info("websocket on ws://%s:%d", self.host or "localhost",
                     self.ws_port)
            await asyncio.Future()  # run forever

    def shutdown(self) -> None:
        self._stop.set()
        self._render_wake.set()
