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


class _Handler(FileSystemEventHandler):
    def __init__(self, on_file: Callable[[Path], None]):
        self._on_file = on_file
        self._seen: set[str] = set()

    def _consider(self, raw_path: str) -> None:
        path = Path(raw_path)
        if not _is_fits(path) or raw_path in self._seen:
            return
        self._seen.add(raw_path)
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
    """Live folder watch (recursive)."""

    def __init__(self, folder: str | Path, on_file: Callable[[Path], None]):
        self.folder = Path(folder)
        self._on_file = on_file
        self._observer = Observer()

    def start(self) -> None:
        if not self.folder.is_dir():
            raise NotADirectoryError(self.folder)
        self._observer.schedule(_Handler(self._on_file), str(self.folder),
                                recursive=True)
        self._observer.start()
        log.info("watching %s", self.folder)

    def stop(self) -> None:
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
