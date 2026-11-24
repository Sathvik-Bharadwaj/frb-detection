#!/bin/bash
# Double-click this file to open the project app in your web browser.
# Keep this window open while presenting. Close it to stop the app.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/streamlit ]; then
    echo "First-time setup: installing Python and libraries (needs internet, a few minutes)..."
    ./setup.sh || { echo; echo "Setup failed. See the messages above."; read -n 1 -s -r -p "Press any key to close."; exit 1; }
fi
# stop an earlier copy of the app if one is still running
lsof -ti:8501 | xargs kill 2>/dev/null
echo
echo "Starting the app... your browser will open at http://localhost:8501"
echo "Keep this window open while presenting. Close it (or press Ctrl+C) to stop."
echo
.venv/bin/streamlit run app.py --server.port 8501
