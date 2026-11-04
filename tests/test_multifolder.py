"""Several capture folders: one stacking order (by capture time), no repeats."""

import os
import threading
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

from livestacker.__main__ import build_parser
from livestacker.source import (ReplaySource, WatchSource, capture_time,
                                collapse_folders, gather_frames)


def _fits(path: Path, date_obs: str | None, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = fits.Header()
    if date_obs:
        hdr["DATE-OBS"] = date_obs
    for k, v in extra.items():
        hdr[k] = v
    fits.PrimaryHDU(np.zeros((4, 4), np.float32), hdr).writeto(path)
    return path


def test_folders_are_deduplicated_and_nested_ones_collapsed(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "night1").mkdir(parents=True)
    b.mkdir()
    kept, missing = collapse_folders([a / "night1", a, b, a, tmp_path / "gone", b / "."])
    assert [p.resolve() for p in kept] == [a.resolve(), b.resolve()]    # inner one folded into a
    assert missing == [tmp_path / "gone"]
    kept, _ = collapse_folders([b, a])
    assert [p.resolve() for p in kept] == [b.resolve(), a.resolve()]    # order as listed
    assert os.path.samefile(collapse_folders([a, a])[0][0], a)


def test_capture_time_reads_the_header_and_falls_back_to_mtime(tmp_path):
    early = _fits(tmp_path / "x.fits", "2026-10-16T22:00:00.250")
    late = _fits(tmp_path / "y.fits", "2026-10-16T22:00:00Z")
    assert capture_time(early) - capture_time(late) == 0.25
    split = _fits(tmp_path / "z.fits", "2026-10-16", **{"TIME-OBS": "22:00:01"})
    assert capture_time(split) - capture_time(late) == 1.0              # date and time apart
    bare = _fits(tmp_path / "w.fits", None)
    os.utime(bare, (1_700_000_000, 1_700_000_000))
    assert capture_time(bare) == 1_700_000_000                          # no DATE-OBS: mtime
    junk = tmp_path / "v.fits"
    junk.write_bytes(b"not a fits file")
    os.utime(junk, (1_600_000_000, 1_600_000_000))
    assert capture_time(junk) == 1_600_000_000                          # unreadable: mtime


def test_several_folders_interleave_by_capture_time_not_by_name(tmp_path):
    d1, d2 = tmp_path / "drive1", tmp_path / "drive2"
    # names would sort a,b,c,d / e,f,g — but the nights interleave in time
    _fits(d1 / "a.fits", "2026-10-16T21:00:00")
    _fits(d2 / "e.fits", "2026-10-16T21:10:00")
    _fits(d1 / "b.fits", "2026-10-16T21:20:00")
    _fits(d2 / "f.fits", "2026-10-16T21:30:00")
    _fits(d1 / "sub" / "c.fits", "2026-10-16T21:40:00")        # subfolders count too
    got = [p.name for p in gather_frames([d1, d2])]
    assert got == ["a.fits", "e.fits", "b.fits", "f.fits", "c.fits"]
    # one folder keeps the plain name order
    assert [p.name for p in gather_frames([d1])] == ["a.fits", "b.fits", "c.fits"]
    # the same file reached twice is stacked once
    assert len(gather_frames([d1, d1 / "sub"])) == 3


def test_replay_of_several_folders_is_in_capture_order_and_limit_applies_after(tmp_path):
    d1, d2 = tmp_path / "one", tmp_path / "two"
    _fits(d1 / "z_late.fits", "2026-10-16T23:00:00")
    _fits(d2 / "a_early.fits", "2026-10-16T20:00:00")
    _fits(d1 / "m_mid.fits", "2026-10-16T21:00:00")
    seen = []
    src = ReplaySource([d1, d2, tmp_path / "missing"], lambda p: seen.append(p.name),
                       interval=0.0, limit=2)
    src.start()
    src._thread.join(5)
    assert seen == ["a_early.fits", "m_mid.fits"]                 # earliest two, from both folders


def test_watch_catches_up_in_capture_order_then_sees_new_files_in_every_folder(tmp_path):
    d1, d2 = tmp_path / "one", tmp_path / "two"
    _fits(d1 / "b.fits", "2026-10-16T22:00:00")
    _fits(d2 / "a.fits", "2026-10-16T21:00:00")
    got, lock = [], threading.Lock()

    def add(p):
        with lock:
            got.append(p.name)
    src = WatchSource([d1, d2, tmp_path / "gone"], add)           # a missing one is skipped
    src.start()
    try:
        for _ in range(100):
            if len(got) >= 2:
                break
            time.sleep(0.1)
        assert got[:2] == ["a.fits", "b.fits"]                    # caught up in time order
        _fits(d2 / "new2.fits", "2026-10-16T23:00:00")            # new arrivals: either folder
        _fits(d1 / "new1.fits", "2026-10-16T23:01:00")
        for _ in range(150):
            if len(got) >= 4:
                break
            time.sleep(0.1)
        assert sorted(got[2:]) == ["new1.fits", "new2.fits"]
    finally:
        src.stop()


def test_watch_with_no_usable_folder_is_an_error(tmp_path):
    import pytest
    with pytest.raises(NotADirectoryError):
        WatchSource([tmp_path / "nope", tmp_path / "nada"], lambda p: None).start()


def test_command_line_takes_repeated_folders():
    a = build_parser().parse_args(["--watch", "/x", "--watch", "/y", "--no-catch-up"])
    assert a.watch == ["/x", "/y"] and a.replay is None
    b = build_parser().parse_args(["--replay", "/x"])
    assert b.replay == ["/x"]


# -- the launcher's folder list (logic only: no window needed) ----------------------

def test_launcher_folder_list_rules(tmp_path):
    import pytest
    pytest.importorskip("tkinter")
    from livestacker.launcher import add_folder, split_existing
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "night").mkdir(parents=True)
    b.mkdir()
    folders, note = add_folder([], str(a))
    assert folders == [str(a)] and note == ""
    folders, note = add_folder(folders, str(b))
    assert folders == [str(a), str(b)]
    # the same folder, or one already inside a listed folder: not added
    same, note = add_folder(folders, str(a))
    assert same == folders and "Already covered" in note
    inner, note = add_folder(folders, str(a / "night"))
    assert inner == folders and str(a) in note
    # a parent of listed folders replaces them
    parent, note = add_folder([str(a / "night"), str(b)], str(a))
    assert parent == [str(b), str(a)] and "Replaced 1 folder" in note
    ok, missing = split_existing([str(a), str(tmp_path / "gone"), str(b)])
    assert ok == [str(a), str(b)] and missing == [str(tmp_path / "gone")]


def test_launcher_migrates_the_old_single_folder_setting(tmp_path, monkeypatch):
    import json
    import pytest
    tk = pytest.importorskip("tkinter")
    import livestacker.launcher as launcher
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display for a Tk window")
    try:
        settings = tmp_path / "settings.json"
        monkeypatch.setattr(launcher, "SETTINGS", settings)
        settings.write_text(json.dumps({"folder": "/old/single", "mode": "watch"}))
        assert launcher.Launcher(root).folders == ["/old/single"]
        settings.write_text(json.dumps({"folders": ["/a", "/b"], "folder": "/a"}))
        assert launcher.Launcher(root).folders == ["/a", "/b"]
        settings.write_text(json.dumps({"mode": "watch"}))
        assert launcher.Launcher(root).folders == []
    finally:
        root.destroy()
