"""Command-line entry point.

    python -m livestacker --watch  "/path/to/capture/folder"
    python -m livestacker --watch  "/drive1/night1" --watch "/drive2/night2"
    python -m livestacker --replay "/path/to/old/session" --interval 1.5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading

from .calibration import Calibrator, MasterBias, MasterDark, build_flats
from .config import StackConfig
from .server import LiveStacker
from .source import ReplaySource, WatchSource


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="livestacker", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--watch", metavar="DIR", action="append",
                     help="watch a live capture folder for new subs (repeat the "
                          "option to watch several; existing subs are stacked in "
                          "capture-time order)")
    src.add_argument("--replay", metavar="DIR", action="append",
                     help="replay an existing folder as if it were being captured "
                          "(repeatable)")

    p.add_argument("--interval", type=float, default=2.0,
                   help="replay: seconds between frames (default: 2)")
    p.add_argument("--limit", type=int, default=None,
                   help="replay: stop after N frames")
    p.add_argument("--no-catch-up", action="store_true",
                   help="watch: don't ingest FITS already in the folder at "
                        "startup (default: do, so a late start / restart keeps "
                        "the subs on disk)")

    p.add_argument("--host", default="127.0.0.1",
                   help="bind address (default: 127.0.0.1; use 0.0.0.0 for LAN)")
    p.add_argument("--web-port", type=int, default=8080)
    p.add_argument("--ws-port", type=int, default=8081)

    p.add_argument("--bias", metavar="PATH",
                   help="master bias: a FITS file, or a folder of bias subs to "
                        "median-combine (also subtracted from the dark and flats)")
    p.add_argument("--dark", metavar="PATH",
                   help="master dark: a FITS file, or a folder of dark subs to "
                        "median-combine")
    p.add_argument("--flat", metavar="PATH",
                   help="master flat(s): a FITS file, or a folder of raw flats "
                        "(grouped by FILTER, one master per filter)")
    p.add_argument("--no-hot-pixel", action="store_true",
                   help="disable per-frame hot/cold pixel correction")
    p.add_argument("--hot-pixel-sigma", type=float, default=8.0,
                   help="hot-pixel detection threshold in robust sigma (default: 8)")
    p.add_argument("--no-sigma-clip", action="store_true",
                   help="disable rolling per-pixel sigma-clip trail/cosmic rejection")
    p.add_argument("--sigma-clip-kappa", type=float, default=3.0,
                   help="sigma-clip rejection threshold (default: 3.0)")

    p.add_argument("--object-name", default="", metavar="NAME",
                   help="the target's name, shown on the web page")
    p.add_argument("--osc-debayer", choices=("auto", "bilinear", "superpixel"),
                   default="auto",
                   help="colour-sensor (OSC) frames: bilinear = full resolution, "
                        "superpixel = half resolution (faster, ~1/4 the memory), "
                        "auto = full up to 12 MP, half above (default)")
    p.add_argument("--max-viewers", type=int, default=16, metavar="N",
                   help="most browsers connected at once (default 16, 1-500)")
    p.add_argument("--no-narrowband", action="store_true",
                   help="ignore narrowband (H / O / S) subs: skip them on arrival "
                        "so the stack is RGB (+L) only")
    p.add_argument("--no-background", action="store_true",
                   help="disable background/gradient subtraction")
    p.add_argument("--no-auto-wb", action="store_true",
                   help="disable automatic per-channel white balance")
    p.add_argument("--background-box", type=int, default=128,
                   help="background mesh box size in px (default: 128)")
    p.add_argument("--preview-px", type=int, default=1600,
                   help="longest preview edge sent to the browser (a browser "
                        "zoomed in past that is sent the full-size image)")

    for edge in ("top", "bottom", "left", "right"):
        p.add_argument(f"--crop-{edge}", type=int, default=0, metavar="N",
                       help=f"trim N rows/columns off the {edge} of every raw "
                            "frame (a permanently bad sensor edge)")

    # set by the launcher: exit when the launcher's pipe to us closes (it quit,
    # crashed or was killed), so a stacker is never left running unattended
    p.add_argument("--exit-with-launcher", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("-v", "--verbose", action="store_true", help="chatty logging")
    return p


VIEWERS_MARKER = "@@viewers "          # the launcher watches for lines starting with this


LIMIT_MARKER = "@@max_viewers "         # launcher -> server: change the viewer limit


def apply_launcher_command(stacker: LiveStacker, line: str) -> None:
    """One line from the launcher: '@@max_viewers N' changes the viewer limit live.
    Anything else (or a bad number) is ignored."""
    if line.startswith(LIMIT_MARKER):
        try:
            stacker.set_max_viewers(int(line[len(LIMIT_MARKER):].strip()))
        except ValueError:
            return
        logging.getLogger("livestacker").info(
            "viewer limit is now %d", stacker.max_viewers)


def _watch_launcher(stacker: LiveStacker) -> None:
    """Read commands from the launcher on stdin until it goes away, then exit. The
    launcher keeps the write end of our stdin open for as long as it lives; the OS
    closes it when the launcher goes away however it goes (quit, crash, kill -9),
    which is what makes this work where a signal handler or `atexit` would not."""
    try:
        for line in sys.stdin:
            apply_launcher_command(stacker, line)
    except Exception:
        pass
    logging.getLogger("livestacker").info("launcher is gone: stopping")
    os._exit(0)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        # no arguments -> open the launcher window
        from .launcher import run as run_launcher
        return run_launcher()

    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # our own package always logs at INFO so the session is followable
    logging.getLogger("livestacker").setLevel(logging.INFO)

    cfg = StackConfig(
        narrowband_enabled=not args.no_narrowband,
        osc_debayer=args.osc_debayer,
        object_name=args.object_name.strip()[:120],
        background_enabled=not args.no_background,
        auto_white_balance=not args.no_auto_wb,
        background_box=args.background_box,
        preview_max_px=args.preview_px,
        hot_pixel_enabled=not args.no_hot_pixel,
        hot_pixel_sigma=args.hot_pixel_sigma,
        sigma_clip_enabled=not args.no_sigma_clip,
        sigma_clip_kappa=args.sigma_clip_kappa,
        crop_top=args.crop_top, crop_bottom=args.crop_bottom,
        crop_left=args.crop_left, crop_right=args.crop_right,
    )

    master_bias = None
    if args.bias:
        try:
            master_bias = MasterBias.build(args.bias)
        except (OSError, ValueError) as exc:
            print(f"error: --bias {exc}", file=sys.stderr)
            return 2

    master_dark = None
    if args.dark:
        try:
            master_dark = MasterDark.build(args.dark, master_bias)
        except (OSError, ValueError) as exc:
            print(f"error: --dark {exc}", file=sys.stderr)
            return 2

    flats = None
    if args.flat:
        try:
            flats = build_flats(args.flat, master_bias)
        except (OSError, ValueError) as exc:
            print(f"error: --flat {exc}", file=sys.stderr)
            return 2

    calibrator = Calibrator(cfg, master_dark, flats, master_bias)

    # The password comes from the environment, not a flag, so it never shows up
    # in a process listing or the launcher's command echo.
    password = os.environ.get("LIVESTACK_PASSWORD", "")
    stacker = LiveStacker(cfg, args.host, args.ws_port, args.web_port, calibrator,
                          password=password, max_viewers=args.max_viewers)
    if password:
        logging.getLogger("livestacker").info("password protection is ON")

    # claim the ports before anything starts: if they are taken (another stacker
    # is running) say so and stop, rather than run half-working beside it
    try:
        stacker.bind()
    except OSError as exc:
        print(f"error: {exc}\nStop the other one, or pick different ports "
              f"(--web-port / --ws-port).", file=sys.stderr)
        return 2

    if args.watch:
        source = WatchSource(args.watch, stacker.submit_file,
                             catch_up=not args.no_catch_up)
    else:
        source = ReplaySource(args.replay, stacker.submit_file,
                              interval=args.interval, limit=args.limit)

    if args.exit_with_launcher:
        threading.Thread(target=_watch_launcher, args=(stacker,), daemon=True).start()
        # tell the launcher how many browsers are connected (a line it reads and
        # does not show in its log)
        stacker.on_viewers = lambda n: print(f"{VIEWERS_MARKER}{n}", flush=True)

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
