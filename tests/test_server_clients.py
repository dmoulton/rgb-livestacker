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
    for cy, cx in [(30, 40), (70, 110), (95, 55), (45, 130), (60, 80)]:
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
