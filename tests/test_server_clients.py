"""Display settings are per browser: one client's sliders must not change
another client's picture."""

import asyncio
import io
import json
import socket
import threading
import time

import numpy as np
from PIL import Image
from websockets.sync.client import connect

from livestacker.config import StackConfig
from livestacker.fitsframe import Frame
from livestacker.server import LiveStacker, params_from_query


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _star_frame(seed, shape=(120, 160)):
    rng = np.random.default_rng(seed)
    data = (1000.0 + rng.normal(0, 5, shape)).astype(np.float32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for cy, cx in [(30, 40), (70, 110), (95, 55), (45, 130), (60, 80), (20, 100), (105, 140),
                   (85, 20), (15, 15), (110, 90), (50, 60), (35, 145)]:
        data += 9000 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / 3.0))
    return Frame(path=type("P", (), {"name": "x.fits"})(), filter="R",
                 data=data, date_obs="", exptime=20.0)


def _settle(client, quiet=1.0):
    """Collect everything the server sends until it goes quiet; return the
    last JPEG seen (or None)."""
    last = None
    while True:
        try:
            msg = client.recv(timeout=quiet)
        except TimeoutError:
            return last
        if isinstance(msg, bytes):
            last = msg


def _brightness(jpeg: bytes) -> float:
    return float(np.asarray(Image.open(io.BytesIO(jpeg)).convert("L")).mean())


def _params(**kw):
    base = dict(type="params", auto=False, black=0.0, midtones=0.5,
                gamma=1.0, contrast=1.0, green=0.0, lum=1.0)
    base.update(kw)
    return json.dumps(base)


def test_each_browser_keeps_its_own_settings():
    cfg = StackConfig(hot_pixel_enabled=False, background_enabled=False,
                      auto_white_balance=False, sigma_clip_enabled=False,
                      min_preview_interval=0.0)
    ws_port, web_port = _free_port(), _free_port()
    stacker = LiveStacker(cfg, "127.0.0.1", ws_port, web_port)
    for seed in range(3):
        stacker.pipeline.ingest(_star_frame(seed))
    threading.Thread(target=lambda: asyncio.run(stacker.run()), daemon=True).start()

    url = f"ws://127.0.0.1:{ws_port}"
    for _ in range(100):                       # wait for the server to listen
        try:
            socket.create_connection(("127.0.0.1", ws_port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.05)

    with connect(url, max_size=None) as a, connect(url, max_size=None) as b:
        a.send(_params(gamma=1.0))
        b.send(_params(gamma=2.5))
        img_a = _settle(a)
        img_b = _settle(b)
        assert img_a and img_b
        assert _brightness(img_b) > _brightness(img_a) + 3, \
            "browser B's brighter gamma should show only in B"

        # B moves its slider again: B refreshes, A must stay silent.
        b.send(_params(gamma=0.5))
        new_b = _settle(b)
        assert new_b and _brightness(new_b) < _brightness(img_b) - 3
        assert _settle(a, quiet=1.5) is None, "A was sent B's image"


def test_snapshot_query_round_trips_settings():
    p = params_from_query("auto=0&gamma=1.8&black=0.1&green=0.5&lum=0.25&bogus=9")
    assert (p.auto, p.gamma, p.black, p.green, p.lum) == (False, 1.8, 0.1, 0.5, 0.25)
    assert params_from_query("gamma=abc&auto=1").gamma == 1.0   # bad value -> default
    assert params_from_query("auto=1").auto is True


def test_zoomed_browser_gets_full_resolution_while_others_keep_the_preview():
    cfg = StackConfig(hot_pixel_enabled=False, background_enabled=False,
                      auto_white_balance=False, sigma_clip_enabled=False,
                      min_preview_interval=0.0, preview_max_px=60)
    ws_port, web_port = _free_port(), _free_port()
    stacker = LiveStacker(cfg, "127.0.0.1", ws_port, web_port)
    for seed in range(3):
        stacker.pipeline.ingest(_star_frame(seed))
    full_w = stacker.pipeline.status()["size"][0]
    threading.Thread(target=lambda: asyncio.run(stacker.run()), daemon=True).start()

    url = f"ws://127.0.0.1:{ws_port}"
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", ws_port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.05)

    def width(jpeg):
        return Image.open(io.BytesIO(jpeg)).size[0]

    with connect(url, max_size=None) as a, connect(url, max_size=None) as b:
        a.send(_params())
        b.send(_params())
        assert width(_settle(a)) == 60 and width(_settle(b)) == 60

        b.send(_params(full=True))                   # B zooms in past the preview
        assert width(_settle(b)) == full_w
        assert _settle(a, quiet=1.0) is None          # A is untouched

        b.send(_params(full=False))                   # ...and back out again
        assert width(_settle(b)) == 60


def _start(stacker, ws_port):
    threading.Thread(target=lambda: asyncio.run(stacker.run()), daemon=True).start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", ws_port), timeout=0.1).close()
            return
        except OSError:
            time.sleep(0.05)


def test_slider_is_answered_while_a_frame_is_being_stacked():
    cfg = StackConfig(hot_pixel_enabled=False, background_enabled=False,
                      auto_white_balance=False, sigma_clip_enabled=False,
                      min_preview_interval=0.0)
    ws_port, web_port = _free_port(), _free_port()
    stacker = LiveStacker(cfg, "127.0.0.1", ws_port, web_port)
    for seed in range(3):
        stacker.pipeline.ingest(_star_frame(seed))

    release = threading.Event()
    started = threading.Event()

    def slow_load(path):
        started.set()
        release.wait(10)
        raise ValueError("test: give up")
    import livestacker.server as server_mod
    original = server_mod.load_frame
    server_mod.load_frame = slow_load
    try:
        _start(stacker, ws_port)
        stacker.submit_file("busy.fits")           # the frame worker is now stuck
        assert started.wait(5)
        with connect(f"ws://127.0.0.1:{ws_port}", max_size=None) as a:
            a.send(_params(gamma=2.0))
            assert _settle(a, quiet=1.0) is not None, \
                "render must not wait behind the frame worker"
    finally:
        release.set()
        server_mod.load_frame = original


def test_backlog_recombines_once_not_per_frame():
    cfg = StackConfig(hot_pixel_enabled=False, background_enabled=False,
                      auto_white_balance=False, sigma_clip_enabled=False,
                      min_preview_interval=0.0, backlog_refresh_s=3600.0)
    stacker = LiveStacker(cfg, "127.0.0.1", _free_port(), _free_port())
    calls = []
    real = stacker.pipeline.recombine
    stacker.pipeline.recombine = lambda: (calls.append(1), real())[1]

    import livestacker.server as server_mod
    frames = iter(_star_frame(s) for s in range(4))
    original = server_mod.load_frame
    server_mod.load_frame = lambda path: next(frames)
    try:
        for _ in range(4):                          # all queued before any is handled
            with stacker._pending_lock:
                stacker._pending += 1
        for i in range(4):
            stacker._handle_frame(type("P", (), {"name": f"{i}.fits"})())
            with stacker._pending_lock:
                stacker._pending -= 1
    finally:
        server_mod.load_frame = original
    assert len(calls) == 1, "recombine only when the backlog drains"
    assert stacker.pipeline.status()["size"]


def test_websocket_is_reachable_through_the_web_port():
    """A tunnel forwarding only the web port must be enough: /ws on that port
    relays to the websocket server."""
    cfg = StackConfig(hot_pixel_enabled=False, background_enabled=False,
                      auto_white_balance=False, sigma_clip_enabled=False,
                      min_preview_interval=0.0)
    ws_port, web_port = _free_port(), _free_port()
    stacker = LiveStacker(cfg, "127.0.0.1", ws_port, web_port)
    for seed in range(3):
        stacker.pipeline.ingest(_star_frame(seed))
    _start(stacker, ws_port)
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", web_port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.05)

    with connect(f"ws://127.0.0.1:{web_port}/ws", max_size=None) as c:
        c.send(_params())
        status = json.loads(c.recv(timeout=5))
        assert "channels" in status
        assert _settle(c) is not None             # and the image arrives
