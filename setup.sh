#!/usr/bin/env bash
# One-command setup: Python 3.12 virtual environment with the exact pinned versions, then the tests.
set -e
cd "$(dirname "$0")"
if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements-lock.txt
    uv pip install --python .venv/bin/python -r requirements-app.txt -c requirements-lock.txt
else
    python3 -m venv .venv
    .venv/bin/pip install -r requirements-lock.txt
    .venv/bin/pip install -r requirements-app.txt -c requirements-lock.txt
fi
.venv/bin/python -m pytest -q
echo
echo "Ready. Double-click 'Start Demo.command' to open the app."
