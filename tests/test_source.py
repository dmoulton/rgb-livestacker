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
