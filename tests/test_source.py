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
