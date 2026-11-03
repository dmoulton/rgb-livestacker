"""Optional password: no password = open; with one, the page, snapshot and the
live websocket all need a login cookie."""

import asyncio
import http.client
import json
import socket
import threading
import time
from urllib.parse import urlencode

import numpy as np
import pytest
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from livestacker.config import StackConfig
from livestacker.server import LOGIN_MAX_FAILS, LiveStacker
from tests.test_server_clients import _free_port, _star_frame

CFG = StackConfig(hot_pixel_enabled=False, background_enabled=False,
                  auto_white_balance=False, sigma_clip_enabled=False,
                  min_preview_interval=0.0)


def _server(password):
    ws_port, web_port = _free_port(), _free_port()
    stacker = LiveStacker(CFG, "127.0.0.1", ws_port, web_port, password=password)
    for seed in range(3):
        stacker.pipeline.ingest(_star_frame(seed))
    threading.Thread(target=lambda: asyncio.run(stacker.run()), daemon=True).start()
    for port in (ws_port, web_port):
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
                break
            except OSError:
                time.sleep(0.05)
    return stacker, web_port


def _http(port, method, path, body=None, cookie=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {}
    if cookie:
        headers["Cookie"] = cookie
    if body is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    conn.request(method, path, body=body, headers=headers)
    r = conn.getresponse()
    data = r.read()
    out = (r.status, dict(r.getheaders()), data)
    conn.close()
    return out


def _login(port, password):
    return _http(port, "POST", "/login", urlencode({"password": password}))


def test_no_password_means_no_prompt():
    _, web = _server("")
    status, _, body = _http(web, "GET", "/")
    assert status == 200 and b"WebSocket" in body            # the real page
    assert _http(web, "GET", "/snapshot.png")[0] == 200


def test_everything_is_locked_until_login():
    _, web = _server("sesame")
    for path in ("/", "/index.html", "/login"):
        status, _, body = _http(web, "GET", path)
        assert status == 200 and b'type="password"' in body   # the login form...
        assert b"WebSocket" not in body                        # ...not the app
        assert b'id="reveal"' in body                          # with a show/hide toggle
    assert _http(web, "GET", "/snapshot.png")[0] == 401
    assert _http(web, "GET", "/ws")[0] == 401
    assert _http(web, "HEAD", "/")[0] == 401                   # HEAD is gated too
    assert _http(web, "POST", "/other", "x=1")[0] == 404

    status, headers, body = _login(web, "wrong")
    assert status == 401 and "Set-Cookie" not in headers and b"Incorrect password" in body


def test_login_unlocks_page_snapshot_and_websocket():
    stacker, web = _server("sesame")
    status, headers, _ = _login(web, "sesame")
    assert status == 303 and headers["Location"] == "/"
    cookie = headers["Set-Cookie"].split(";")[0]
    assert "HttpOnly" in headers["Set-Cookie"] and "SameSite=Strict" in headers["Set-Cookie"]

    status, _, body = _http(web, "GET", "/", cookie=cookie)
    assert status == 200 and b"WebSocket" in body
    assert _http(web, "GET", "/snapshot.png", cookie=cookie)[0] == 200
    assert _http(web, "GET", "/login", cookie=cookie)[0] == 303   # already in

    url = f"ws://127.0.0.1:{web}/ws"
    with pytest.raises(InvalidStatus):                          # no cookie: refused
        connect(url, max_size=None)
    with connect(url, max_size=None, additional_headers={"Cookie": cookie}) as c:
        c.send(json.dumps({"type": "params", "auto": True}))
        assert "channels" in json.loads(c.recv(timeout=5))


def test_a_forged_or_expired_cookie_is_refused():
    stacker, web = _server("sesame")
    assert _http(web, "GET", "/snapshot.png", cookie="ls_session=forged")[0] == 401
    token = stacker.new_session()
    stacker._sessions[token] = time.time() - 1                  # expired
    assert _http(web, "GET", "/snapshot.png", cookie=f"ls_session={token}")[0] == 401


def test_any_password_is_accepted_and_guessing_is_throttled():
    stacker, web = _server(" é 😀  with spaces ")                # no content rules
    assert _login(web, " é 😀  with spaces ")[0] == 303
    for _ in range(LOGIN_MAX_FAILS):
        assert _login(web, "nope")[0] == 401
    assert _login(web, "nope")[0] == 429                        # locked out...
    assert _login(web, " é 😀  with spaces ")[0] == 429          # ...even for the right one
    stacker._locked_until = 0.0                                 # (wait it out)
    assert _login(web, " é 😀  with spaces ")[0] == 303


def test_the_raw_websocket_port_is_local_only_when_a_password_is_set():
    open_ = LiveStacker(CFG, "0.0.0.0", 1, 2)
    locked = LiveStacker(CFG, "0.0.0.0", 1, 2, password="x")
    assert open_._ws_host() == "0.0.0.0"
    assert locked._ws_host() == "127.0.0.1"
