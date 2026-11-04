"""Frame sources: `watch` a live capture folder, or `replay` an existing one.

Both call a supplied `on_file(path)` callback from a background thread. The
callback is expected to be quick (it just enqueues); heavy work happens
downstream on the pipeline worker.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .fitsframe import FRAME_SUFFIXES, RAW_SUFFIXES

log = logging.getLogger("livestacker.source")


def _is_fits(path: Path) -> bool:
    """A file the stacker can read: FITS, or a Canon raw (CR2 / CR3)."""
    return path.suffix.lower() in FRAME_SUFFIXES


def _wait_until_stable(path: Path, timeout: float = 30.0, settle: float = 1.0) -> bool:
    """Return True once the file size has held steady for `settle` seconds."""
    deadline = time.time() + timeout
    last_size = -1
    stable_since = None
    while time.time() < deadline:
        try:
            size = path.stat().st_size
        except OSError:
            return False
        now = time.time()
        if size == last_size and size > 0:
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= settle:
                return True
        else:
            stable_since = None
            last_size = size
        time.sleep(0.25)
    return False


def _key(path: Path) -> str:
    """A stable identity for a path, robust to symlinked parents (e.g. macOS
    /var -> /private/var, which watchdog and rglob disagree on)."""
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


class _Handler(FileSystemEventHandler):
    def __init__(self, on_file: Callable[[Path], None]):
        self._on_file = on_file
        self._seen: set[str] = set()

    def _consider(self, raw_path: str) -> None:
        path = Path(raw_path)
        key = _key(path)
        if not _is_fits(path) or key in self._seen:
            return
        self._seen.add(key)
        if _wait_until_stable(path):
            self._on_file(path)
        else:
            log.warning("file never stabilised, skipping: %s", path.name)

    def on_created(self, event):
        if not event.is_directory:
            self._consider(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._consider(event.dest_path)


# -- several folders, one stacking order ----------------------------------------

def collapse_folders(folders: Iterable[str | Path]) -> tuple[list[Path], list[Path]]:
    """(folders to use, folders that don't exist). The same folder given twice, or
    one that lies inside another listed folder (a watch already searches its
    subfolders), is counted once, so nothing is stacked twice."""
    kept: list[Path] = []
    missing: list[Path] = []
    resolved = []
    for f in folders:
        p = Path(f)
        if not p.is_dir():
            missing.append(p)
            continue
        resolved.append((p.resolve(), p))
    for real, p in sorted(resolved, key=lambda rp: len(rp[0].parts)):   # outermost first
        if not any(real == k.resolve() or k.resolve() in real.parents for k in kept):
            kept.append(p)
    order = {str(p.resolve()): i for i, p in enumerate(Path(f) for f in folders)}
    kept.sort(key=lambda p: order.get(str(p.resolve()), 0))             # as listed
    return kept, missing


def _parse_time(text: str) -> float | None:
    text = (text or "").strip().rstrip("Zz")
    if not text:
        return None
    try:
        when = datetime.fromisoformat(text)
    except ValueError:
        return None
    return (when if when.tzinfo else when.replace(tzinfo=timezone.utc)).timestamp()


def capture_time(path: Path) -> float:
    """When the frame was taken (seconds since the epoch): DATE-OBS from a FITS
    header, the EXIF time of a raw file, or — if neither can be read — the file's
    modification time. Only the header is read, never the pixels."""
    path = Path(path)
    when = None
    try:
        if path.suffix.lower() in RAW_SUFFIXES:
            from .rawframe import _exif
            when = _parse_time(_exif(path).get("date_obs", ""))
        else:
            from astropy.io import fits
            header = fits.getheader(path)
            date = str(header.get("DATE-OBS", "") or "")
            if len(date) == 10 and header.get("TIME-OBS"):          # date and time apart
                date = f"{date}T{header['TIME-OBS']}"
            when = _parse_time(date)
    except Exception:
        when = None
    if when is None:
        try:
            when = path.stat().st_mtime
        except OSError:
            when = 0.0
    return when


def gather_frames(folders: list[Path]) -> list[Path]:
    """Every frame file under the folders, each once, in stacking order: by name
    for a single folder (a capture folder's names already sort by time), by
    capture time once there are several (their names don't interleave)."""
    seen: set[str] = set()
    files: list[Path] = []
    for folder in folders:
        for p in folder.rglob("*"):
            if _is_fits(p) and _key(p) not in seen:
                seen.add(_key(p))
                files.append(p)
    if len(folders) < 2:
        return sorted(files, key=lambda p: p.name)
    # headers are small but may be on a slow network drive: read several at once
    with ThreadPoolExecutor(max_workers=8) as pool:
        times = list(pool.map(capture_time, files))
    return [p for _, _, p in sorted(zip(times, (f.name for f in files), files),
                                    key=lambda t: (t[0], t[1]))]


def _usable(folders: list[Path]) -> list[Path]:
    kept, missing = collapse_folders(folders)
    for p in missing:
        log.warning("folder not found, skipping: %s", p)
    return kept


class WatchSource:
    """Live watch of one or more folders (recursive).

    With `catch_up`, every frame already in the folders is ingested first — in
    capture-time order when there are several folders — before new arrivals are
    handled, so starting late, or restarting after a crash, doesn't throw away
    subs that are on disk.
    """

    def __init__(self, folder: str | Path | Iterable[str | Path],
                 on_file: Callable[[Path], None], catch_up: bool = True):
        self.folders = [Path(f) for f in (
            [folder] if isinstance(folder, (str, Path)) else folder)]
        self.folder = self.folders[0]
        self._on_file = on_file
        self.catch_up = catch_up
        self._handler = _Handler(on_file)
        self._observer = Observer()
        self._stop = threading.Event()

    def start(self) -> None:
        folders = _usable(self.folders)
        if not folders:
            raise NotADirectoryError(self.folder)

        # Snapshot what's already there and mark it seen *before* the observer
        # starts, so a replayed / coalesced create event (macOS FSEvents does
        # this) can't double-ingest it.
        existing = gather_frames(folders)
        for path in existing:
            self._handler._seen.add(_key(path))

        for folder in folders:
            self._observer.schedule(self._handler, str(folder), recursive=True)
            log.info("watching %s", folder)
        self._observer.start()

        if self.catch_up and existing:
            threading.Thread(target=self._ingest_existing, args=(existing,),
                             daemon=True).start()

    def _ingest_existing(self, files: list[Path]) -> None:
        log.info("catch-up: ingesting %d file(s) already in the folder(s)", len(files))
        for path in files:
            if self._stop.is_set():
                return
            self._on_file(path)
        log.info("catch-up done")

    def stop(self) -> None:
        self._stop.set()
        self._observer.stop()
        self._observer.join(timeout=5)


class ReplaySource:
    """Feed the frames of one or more existing folders (in filename order for one
    folder, capture-time order for several), paced to imitate a capture run.
    Great for developing without the rig."""

    def __init__(self, folder: str | Path | Iterable[str | Path],
                 on_file: Callable[[Path], None],
                 interval: float = 2.0, limit: int | None = None):
        self.folders = [Path(f) for f in (
            [folder] if isinstance(folder, (str, Path)) else folder)]
        self.folder = self.folders[0]
        self._on_file = on_file
        self.interval = interval
        self.limit = limit
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        folders = _usable(self.folders)
        files = gather_frames(folders)
        if self.limit:
            files = files[: self.limit]
        where = ", ".join(str(f) for f in folders) or "(no folders)"
        if not files:
            log.warning("no frames found under %s", where)
        log.info("replaying %d files from %s every %.1fs",
                 len(files), where, self.interval)
        for path in files:
            if self._stop.is_set():
                break
            self._on_file(path)
            if self._stop.wait(self.interval):
                break
        log.info("replay finished")
