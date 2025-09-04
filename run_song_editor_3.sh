#!/bin/bash

# Song Editor 3 Launcher Script
# This script ensures the correct virtual environment is activated and runs Song Editor 3

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

# Check if requirements are installed (check for a key package)
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

# Run Song Editor 3 with any passed arguments
python -m song_editor "$@"
