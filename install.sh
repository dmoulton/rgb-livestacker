#!/usr/bin/env bash
# One-time setup for the livestacker on macOS / Linux.
# Creates a .venv next to this script and installs the dependencies.
set -euo pipefail
cd "$(dirname "$0")"

# Find a Python 3.11+ interpreter.
PY=""
for c in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$c" >/dev/null 2>&1 \
        && "$c" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' 2>/dev/null; then
        PY="$c"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "error: need Python 3.11 or newer on your PATH." >&2
    echo "  macOS:  brew install python@3.13" >&2
    echo "  or download from https://www.python.org/downloads/" >&2
    exit 1
fi
echo "Using $("$PY" --version) ($(command -v "$PY"))"

if [ ! -d .venv ]; then
    "$PY" -m venv .venv
    echo "Created .venv"
fi

# PIP_USER is sometimes set in the environment and breaks installs into a venv.
export PIP_USER=0
./.venv/bin/python -m pip install --upgrade --quiet pip
./.venv/bin/python -m pip install --quiet -r requirements.txt

if ! ./.venv/bin/python -c 'import tkinter' 2>/dev/null; then
    echo
    echo "note: Tk is missing, so the launcher window won't open."
    echo "      macOS (Homebrew Python): brew install python-tk"
    echo "      Debian/Ubuntu:           sudo apt install python3-tk"
    echo "      The --watch / --replay command line still works without it."
fi

echo
echo "Done. Open the launcher by double-clicking Livestacker.command,"
echo "or run it from the command line:"
echo "  ./livestack.sh                                   # launcher window"
echo "  ./livestack.sh --watch  \"/path/to/capture/folder\"   # headless, live"
echo "  ./livestack.sh --replay \"/path/to/old/session\"      # headless, test"
