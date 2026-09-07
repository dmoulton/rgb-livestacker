"""Frame sources: `watch` a live capture folder, or `replay` an existing one.

Both call a supplied `on_file(path)` callback from a background thread. The
callback is expected to be quick (it just enqueues); heavy work happens
downstream on the pipeline worker.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

log = logging.getLogger("livestacker.source")

FITS_SUFFIXES = (".fits", ".fit", ".fts")


def _is_fits(path: Path) -> bool:
    return path.suffix.lower() in FITS_SUFFIXES


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


class WatchSource:
    """Live folder watch (recursive).

    With `catch_up`, every FITS already in the folder is ingested first (in
    filename order) before new arrivals are handled — so starting late, or
    restarting after a crash, doesn't throw away subs that are on disk.
    """

    def __init__(self, folder: str | Path, on_file: Callable[[Path], None],
                 catch_up: bool = True):
        self.folder = Path(folder)
        self._on_file = on_file
        self.catch_up = catch_up
        self._handler = _Handler(on_file)
        self._observer = Observer()
        self._stop = threading.Event()

    def start(self) -> None:
        if not self.folder.is_dir():
            raise NotADirectoryError(self.folder)

        # Snapshot what's already there and mark it seen *before* the observer
        # starts, so a replayed / coalesced create event (macOS FSEvents does
        # this) can't double-ingest it.
        existing = sorted((p for p in self.folder.rglob("*") if _is_fits(p)),
                          key=lambda p: p.name)
        for path in existing:
            self._handler._seen.add(_key(path))

        self._observer.schedule(self._handler, str(self.folder), recursive=True)
        self._observer.start()
        log.info("watching %s", self.folder)

        if self.catch_up and existing:
            threading.Thread(target=self._ingest_existing, args=(existing,),
                             daemon=True).start()

    def _ingest_existing(self, files: list[Path]) -> None:
        log.info("catch-up: ingesting %d file(s) already in the folder", len(files))
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
    """Feed an existing folder's FITS files in filename order, paced to imitate
    a capture run. Great for developing without the rig."""

    def __init__(self, folder: str | Path, on_file: Callable[[Path], None],
                 interval: float = 2.0, limit: int | None = None):
        self.folder = Path(folder)
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
        files = sorted(
            (p for p in self.folder.rglob("*") if _is_fits(p)),
            key=lambda p: p.name,
        )
        if self.limit:
            files = files[: self.limit]
        log.info("replaying %d files from %s every %.1fs",
                 len(files), self.folder, self.interval)
        for path in files:
            if self._stop.is_set():
                break
            self._on_file(path)
            if self._stop.wait(self.interval):
                break
        log.info("replay finished")
