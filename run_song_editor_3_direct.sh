#!/bin/bash

# Song Editor 3 Direct Launcher Script
# This script runs Song Editor 3 directly without module imports

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the project directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Check if requirements are installed
if ! python -c "import PySide6" 2>/dev/null; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# Set environment variables for better performance and model support
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export QT_LOGGING_RULES="*.debug=false;qt.qpa.*=false"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export TORCH_USE_CUDA_DSA="1"
export CUDA_LAUNCH_BLOCKING="0"

# Enable all available models and features
export SONG_EDITOR_ENABLE_ALL_MODELS="1"
export SONG_EDITOR_FORCE_GPU="0"  # Set to 1 to force GPU usage

# Run Song Editor 3 directly with any passed arguments
python song_editor/app.py "$@"
