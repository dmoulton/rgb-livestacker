#!/usr/bin/env bash
# Run the livestacker from its .venv. Pass through any arguments, e.g.
#   ./livestack.sh --watch "/path/to/folder" -v
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "error: run ./install.sh first." >&2
    exit 1
fi
exec ./.venv/bin/python -m livestacker "$@"
