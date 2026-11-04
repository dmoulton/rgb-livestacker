"""The launcher's collapsible Source / Calibration sections."""

import json

import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture
def launcher_window(tmp_path, monkeypatch):
    import livestacker.launcher as launcher
    monkeypatch.setattr(launcher, "SETTINGS", tmp_path / "settings.json")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display for a Tk window")
    yield launcher, root
    try:
        root.destroy()
    except tk.TclError:
        pass


def _grid_count(launcher_obj):
    frm = launcher_obj.log.master
    return sum(1 for w in frm.grid_slaves() if w.winfo_manager() == "grid")


def test_collapsing_gives_the_space_to_the_log_and_expanding_takes_it_back(launcher_window):
    launcher, root = launcher_window
    L = launcher.Launcher(root)
    root.update()
    window, log0, widgets0 = root.winfo_height(), L.log.winfo_height(), _grid_count(L)

    L._toggle_section("source")
    root.update()
    log1 = L.log.winfo_height()
    assert log1 > log0 + 100 and root.winfo_height() == window       # taller log, same window
    assert _grid_count(L) < widgets0

    L._toggle_section("calibration")
    root.update()
    log2 = L.log.winfo_height()
    assert log2 > log1 + 100

    L._toggle_section("source")
    L._toggle_section("calibration")
    root.update()
    assert L.log.winfo_height() == log0 and _grid_count(L) == widgets0   # all back, exactly
    assert root.winfo_height() == window


def test_a_collapsed_section_says_what_is_in_it(launcher_window):
    launcher, root = launcher_window
    L = launcher.Launcher(root)
    L.folders = ["/a", "/b"]
    L.v["object_name"].set("M27")
    L.v["dark"].set("/x/darks")
    L.v["crop_bottom"].set("15")
    L._toggle_section("source")
    L._toggle_section("calibration")
    _, arrow, summary = L._section_ui["source"]
    assert arrow.cget("text").startswith("▸") and "watching 2 folders" in summary.cget("text")
    assert "M27" in summary.cget("text")
    _, arrow, summary = L._section_ui["calibration"]
    assert "dark" in summary.cget("text") and "edge crop" in summary.cget("text")
    L.v["dark"].set(""); L.v["crop_bottom"].set("0")
    L._toggle_section("calibration"); L._toggle_section("calibration")
    assert L._section_ui["calibration"][2].cget("text") == "none"
    L._toggle_section("calibration")
    assert L._section_ui["calibration"][1].cget("text").startswith("▾")
    assert L._section_ui["calibration"][2].cget("text") == ""        # open: no summary


def test_collapsed_state_is_remembered_and_the_window_still_opens_full_height(launcher_window):
    launcher, root = launcher_window
    L = launcher.Launcher(root)
    root.update()
    full_height = root.winfo_height()
    L._toggle_section("calibration")
    L._save()
    saved = json.loads(launcher.SETTINGS.read_text())
    assert saved["source_open"] is True and saved["calibration_open"] is False
    root.destroy()

    root2 = tk.Tk()
    try:
        L2 = launcher.Launcher(root2)
        root2.update()
        assert L2._section_ui["calibration"][1].cget("text").startswith("▸")   # starts collapsed
        assert L2._section_ui["source"][1].cget("text").startswith("▾")
        assert root2.winfo_height() == full_height                              # not shrunk to fit
    finally:
        root2.destroy()
