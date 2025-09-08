#!/bin/bash

# Song Editor 3 Launcher Script
# This script runs the Song Editor 3 application using the virtual environment

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "❌ Error: Virtual environment not found at $SCRIPT_DIR/.venv"
    echo "Please run setup_environment.sh first"
    exit 1
fi

# Activate virtual environment and set Python path
export PYTHONPATH="$SCRIPT_DIR"
source "$SCRIPT_DIR/.venv/bin/activate"

# Run the Song Editor 3 application with all passed arguments
echo "🚀 Starting Song Editor 3..."
exec python "$SCRIPT_DIR/song_editor/app.py" "$@"