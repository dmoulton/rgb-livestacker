"""Command-line entry point.

    python -m livestacker --watch  "/path/to/capture/folder"
    python -m livestacker --replay "/path/to/old/session" --interval 1.5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .config import StackConfig
from .server import LiveStacker
from .source import ReplaySource, WatchSource


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="livestacker", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--watch", metavar="DIR",
                     help="watch a live capture folder for new subs")
    src.add_argument("--replay", metavar="DIR",
                     help="replay an existing folder as if it were being captured")

    p.add_argument("--interval", type=float, default=2.0,
                   help="replay: seconds between frames (default: 2)")
    p.add_argument("--limit", type=int, default=None,
                   help="replay: stop after N frames")

    p.add_argument("--host", default="127.0.0.1",
                   help="bind address (default: 127.0.0.1; use 0.0.0.0 for LAN)")
    p.add_argument("--web-port", type=int, default=8080)
    p.add_argument("--ws-port", type=int, default=8081)

    p.add_argument("--no-background", action="store_true",
                   help="disable background/gradient subtraction")
    p.add_argument("--no-auto-wb", action="store_true",
                   help="disable automatic per-channel white balance")
    p.add_argument("--background-sigma", type=float, default=60.0)
    p.add_argument("--preview-px", type=int, default=1600,
                   help="longest preview edge sent to the browser")

    p.add_argument("-v", "--verbose", action="store_true", help="chatty logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # our own package always logs at INFO so the session is followable
    logging.getLogger("livestacker").setLevel(logging.INFO)

    cfg = StackConfig(
        background_enabled=not args.no_background,
        auto_white_balance=not args.no_auto_wb,
        background_sigma=args.background_sigma,
        preview_max_px=args.preview_px,
    )

    stacker = LiveStacker(cfg, args.host, args.ws_port, args.web_port)

    if args.watch:
        source = WatchSource(args.watch, stacker.submit_file)
    else:
        source = ReplaySource(args.replay, stacker.submit_file,
                              interval=args.interval, limit=args.limit)

    try:
        source.start()
    except (NotADirectoryError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        asyncio.run(stacker.run())
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        stacker.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
