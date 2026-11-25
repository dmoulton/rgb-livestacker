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
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect

from livestacker.config import StackConfig
import livestacker.server as server_mod
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


def _http(port, method, path, body=None, cookie=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = dict(headers or {})
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


def _login(port, password, headers=None):
    return _http(port, "POST", "/login", urlencode({"password": password}),
                 headers=headers)


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


# -- hardening ---------------------------------------------------------------

def test_security_headers_nonce_and_nothing_else_is_served():
    _, web = _server("")
    status, headers, body = _http(web, "GET", "/")
    assert status == 200
    h = {k.lower(): v for k, v in headers.items()}
    assert h["x-content-type-options"] == "nosniff"
    assert h["x-frame-options"] == "DENY" and h["referrer-policy"] == "no-referrer"
    assert h["server"] == "WebStacker"                         # no Python version
    csp = h["content-security-policy"]
    assert "frame-ancestors 'none'" in csp and "unsafe-inline" not in csp
    nonce = csp.split("script-src 'nonce-")[1].split("'")[0]
    assert f'<script nonce="{nonce}">'.encode() in body       # inline script allowed by nonce only
    assert f'<style nonce="{nonce}">'.encode() in body
    for path in ("/login.html", "/index.html.bak", "/livestacker/", "/server.py"):
        assert _http(web, "GET", path)[0] == 404              # no static fallback


def test_login_cookie_is_secure_behind_https_and_on_tunnel_hosts():
    _, web = _server("pw")
    plain = _login(web, "pw")[1]["Set-Cookie"]
    assert "Secure" not in plain and "Max-Age=86400" in plain          # 24 hours
    via_proxy = _login(web, "pw", {"X-Forwarded-Proto": "https"})[1]["Set-Cookie"]
    assert "Secure" in via_proxy
    on_tunnel = _login(web, "pw", {"Host": "abc123.lhr.life"})[1]["Set-Cookie"]
    assert "Secure" in on_tunnel                                         # even with no proxy header


def test_sign_out_ends_the_login_and_cuts_its_live_connection():
    _, web = _server("pw")
    cookie = _login(web, "pw")[1]["Set-Cookie"].split(";")[0]
    with connect(f"ws://127.0.0.1:{web}/ws", max_size=None,
                 additional_headers={"Cookie": cookie}) as c:
        c.recv(timeout=5)                                              # live
        status, headers, _ = _http(web, "POST", "/logout", "", cookie=cookie)
        assert status == 204 and "Max-Age=0" in headers["Set-Cookie"]
        with pytest.raises(ConnectionClosed):
            for _ in range(20):
                c.recv(timeout=5)
    assert _http(web, "GET", "/snapshot.png", cookie=cookie)[0] == 401  # cookie is dead
    assert _http(web, "POST", "/logout", "", cookie="ls_session=junk")[0] == 204  # harmless


def test_lockouts_get_longer_each_time():
    stacker, _ = _server("pw")
    lengths = []
    for _ in range(4):
        for _ in range(LOGIN_MAX_FAILS):
            stacker.check_login("nope")
        lengths.append(round(stacker._locked_until - time.time()))
        stacker._locked_until = 0.0                                    # skip the wait
    assert lengths == [30, 60, 120, 240]
    assert stacker.check_login("pw") == "ok"
    for _ in range(LOGIN_MAX_FAILS):
        stacker.check_login("nope")
    assert round(stacker._locked_until - time.time()) == 30           # a good login resets it


def test_browser_is_sent_file_names_not_paths():
    from types import SimpleNamespace as NS
    from livestacker.pipeline import Pipeline, _public_name
    assert _public_name("/Users/dave/astro/darks (12 frames)") == "darks (12 frames)"
    assert _public_name("C:\\Users\\dave\\bias.fits") == "bias.fits"
    assert _public_name("/a/b/c/") == "c"
    cal = NS(master_bias=NS(source="/Users/dave/x/bias (6 frames)"),
             master_dark=NS(source="C:\\Users\\dave\\dark (9 frames)"), flats={},
             applied={}, problems=[])
    p = Pipeline(CFG, calibrator=cal)
    st = p.status()["calibration"]
    assert st["bias"] == "bias (6 frames)" and st["dark"] == "dark (9 frames)"
    assert "dave" not in json.dumps(p.status())


def test_idle_connections_are_dropped(monkeypatch):
    monkeypatch.setattr(server_mod, "HTTP_TIMEOUT_S", 1)
    _, web = _server("")
    s = socket.create_connection(("127.0.0.1", web))
    s.settimeout(6)
    t = time.time()
    assert s.recv(10) == b""                                          # server hung up
    assert time.time() - t < 5
    s.close()


def test_connections_are_capped(monkeypatch):
    # cap = live-viewer limit + a few extra for page loads: 1 + 2 = 3 here
    monkeypatch.setattr(server_mod, "MAX_VIEWERS", 1)
    monkeypatch.setattr(server_mod, "HTTP_EXTRA_CONNECTIONS", 2)
    _, web = _server("")
    idle = [socket.create_connection(("127.0.0.1", web)) for _ in range(3)]
    try:
        time.sleep(0.5)
        status, headers, _ = _http(web, "GET", "/")
        assert status == 503 and headers.get("Retry-After") == "5"
    finally:
        for s in idle:
            s.close()
    time.sleep(0.5)
    assert _http(web, "GET", "/")[0] == 200                           # recovers


def test_live_browsers_are_capped(monkeypatch):
    monkeypatch.setattr(server_mod, "MAX_VIEWERS", 2)
    stacker, web = _server("")
    url = f"ws://127.0.0.1:{stacker.ws_port}"
    with connect(url, max_size=None) as a, connect(url, max_size=None) as b:
        a.recv(timeout=5), b.recv(timeout=5)
        with connect(url, max_size=None) as c:
            with pytest.raises(ConnectionClosed) as err:
                for _ in range(5):
                    c.recv(timeout=5)
            assert err.value.rcvd.code == 1013                        # "try again later"
    with connect(url, max_size=None) as d:                            # a seat is free again
        assert d.recv(timeout=5)


def test_only_a_couple_of_full_size_pngs_at_once(monkeypatch):
    monkeypatch.setattr(server_mod, "SNAPSHOTS_AT_ONCE", 1)
    stacker, web = _server("")
    release, entered = threading.Event(), threading.Event()
    real = stacker.pipeline.snapshot_png

    def slow(params):
        entered.set()
        release.wait(10)
        return real(params)
    stacker.pipeline.snapshot_png = slow
    first = threading.Thread(target=lambda: _http(web, "GET", "/snapshot.png"))
    first.start()
    try:
        assert entered.wait(5)
        status, headers, _ = _http(web, "GET", "/snapshot.png")
        assert status == 503 and headers.get("Retry-After") == "5"
    finally:
        release.set()
        first.join(10)
    stacker.pipeline.snapshot_png = real
    assert _http(web, "GET", "/snapshot.png")[0] == 200


def test_a_login_lasts_24_hours():
    stacker, web = _server("pw")
    token = stacker.new_session()
    left = stacker._sessions[token] - time.time()
    assert 24 * 3600 - 5 < left <= 24 * 3600
    assert stacker.valid_session(token)
    stacker._sessions[token] = time.time() - 1
    assert not stacker.valid_session(token)


def test_connection_cap_follows_the_viewer_limit(monkeypatch):
    monkeypatch.setattr(server_mod, "MAX_VIEWERS", 1)
    monkeypatch.setattr(server_mod, "HTTP_EXTRA_CONNECTIONS", 2)
    stacker, web = _server("")
    idle = [socket.create_connection(("127.0.0.1", web)) for _ in range(3)]
    try:
        time.sleep(0.4)
        assert _http(web, "GET", "/")[0] == 503                 # 3 of 3 used
        stacker.set_max_viewers(4)                              # cap is now 4 + 2 = 6
        assert _http(web, "GET", "/")[0] == 200                 # room, no restart
    finally:
        for s in idle:
            s.close()


def test_hidden_elements_stay_hidden():
    """The Sign out button showed with no password: a styled `button` sets
    `display`, which beats the `hidden` attribute unless the page says otherwise."""
    _, web = _server("")
    body = _http(web, "GET", "/")[2]
    assert b'id="signout" hidden' in body                    # off until a password says so
    assert b"[hidden] { display: none !important; }" in body


@pytest.mark.parametrize("password", ["", "sesame"])
def test_tab_and_home_screen_icons_are_served_without_a_login(password):
    import io
    from PIL import Image
    _, web = _server(password)
    status, headers, body = _http(web, "GET", "/favicon.ico")
    assert status == 200 and headers["Content-Type"] == "image/x-icon"
    assert {(16, 16), (32, 32), (48, 48)} <= set(Image.open(io.BytesIO(body)).info["sizes"])
    for path in ("/apple-touch-icon.png", "/apple-touch-icon-precomposed.png"):
        status, headers, body = _http(web, "GET", path)
        assert status == 200 and headers["Content-Type"] == "image/png"
        assert Image.open(io.BytesIO(body)).size == (180, 180)
    page = _http(web, "GET", "/")[2]                      # the app, or the login page
    assert b'rel="icon" href="/favicon.ico"' in page and b"apple-touch-icon" in page
    if password:                                          # ...and nothing else opened up
        assert _http(web, "GET", "/snapshot.png")[0] == 401
        assert _http(web, "GET", "/favicon.ico/../index.html")[0] == 401
