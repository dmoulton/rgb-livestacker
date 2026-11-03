"""WatchSource catch-up behaviour."""

import time
from pathlib import Path

from livestacker.source import WatchSource


def test_catch_up_ingests_existing_then_new_without_dupes(tmp_path):
    for n in ("a.fits", "b.fits", "c.fits"):
        (tmp_path / n).write_bytes(b"x" * 128)

    got: list[str] = []
    ws = WatchSource(tmp_path, lambda p: got.append(p.name), catch_up=True)
    ws.start()
    time.sleep(0.4)
    (tmp_path / "d.fits").write_bytes(b"x" * 128)
    time.sleep(3.0)
    ws.stop()

    assert sorted(got) == ["a.fits", "b.fits", "c.fits", "d.fits"]
    assert len(got) == len(set(got)), f"double-ingested: {got}"


def test_no_catch_up_skips_existing(tmp_path):
    (tmp_path / "old.fits").write_bytes(b"x" * 128)

    got: list[str] = []
    ws = WatchSource(tmp_path, lambda p: got.append(p.name), catch_up=False)
    ws.start()
    time.sleep(1.0)
    ws.stop()

    assert got == []


def test_server_exits_when_the_launcher_pipe_closes(tmp_path):
    """--exit-with-launcher: a stacker started by the launcher must not outlive
    it. The OS closes the pipe whether the launcher quit, crashed or was killed."""
    import socket
    import subprocess
    import sys
    import time

    def port():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    proc = subprocess.Popen(
        [sys.executable, "-m", "livestacker", "--replay", str(tmp_path),
         "--web-port", str(port()), "--ws-port", str(port()), "--exit-with-launcher"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(2.0)
        assert proc.poll() is None, "should keep running while the launcher is alive"
        proc.stdin.close()                       # what the OS does when the launcher dies
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_launcher_gets_the_live_browser_count(tmp_path):
    """Started the way the launcher starts it, the server prints '@@viewers N'
    lines on stdout as browsers connect and leave, and the launcher can parse them."""
    import socket
    import subprocess
    import sys
    import threading
    import time

    import pytest
    pytest.importorskip("tkinter")
    from livestacker.launcher import parse_viewers
    from websockets.sync.client import connect

    assert parse_viewers("@@viewers 3\n") == 3
    assert parse_viewers("@@viewers 0") == 0
    assert parse_viewers("@@viewers x") is None
    assert parse_viewers("client connected (1.2.3.4), 3 total") is None

    def port():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    ws_port = port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "livestacker", "--replay", str(tmp_path),
         "--web-port", str(port()), "--ws-port", str(ws_port), "--exit-with-launcher"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    seen = []
    threading.Thread(target=lambda: [seen.append(parse_viewers(l)) for l in proc.stdout],
                     daemon=True).start()
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", ws_port), timeout=0.1).close()
                break
            except OSError:
                time.sleep(0.1)
        with connect(f"ws://127.0.0.1:{ws_port}") as c:
            c.recv(timeout=5)
            for _ in range(100):
                if 1 in seen:
                    break
                time.sleep(0.05)
            assert 1 in seen
        for _ in range(100):
            if [n for n in seen if n is not None][-1:] == [0]:
                break
            time.sleep(0.05)
        assert [n for n in seen if n is not None][-1:] == [0]
    finally:
        proc.kill()
