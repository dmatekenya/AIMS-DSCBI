#!/bin/bash
# Script to serve the Jupyter Book with disabled caching

echo "Starting a local server to preview the book..."
echo "Open your browser and go to: http://localhost:8000"
echo "Press Ctrl+C to stop the server"

# Check if Python 3 is available
if command -v python3 &>/dev/null; then
    python3 -m http.server 8000 --directory docs/_build/html
else
    # Fall back to python if python3 is not available
    python -m http.server 8000 --directory docs/_build/html
fi
