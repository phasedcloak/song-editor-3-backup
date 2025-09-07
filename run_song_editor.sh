#!/bin/bash

# Song Editor 3 Launcher Script
# This script provides a reliable way to run the Song Editor application

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if we're in a virtual environment, if not try to activate one
if [[ -z "$VIRTUAL_ENV" ]]; then
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
    elif [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    fi
fi

# Set environment variables for better performance
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# Default Demucs model handling: if caller didn't specify --demucs-model, default to htdemucs
ADD_DEFAULT_ARGS=1
for arg in "$@"; do
    case "$arg" in
        --demucs-model|--demucs-model=*)
            ADD_DEFAULT_ARGS=0
            break
            ;;
    esac
done

DEFAULT_ARGS=()
if [[ $ADD_DEFAULT_ARGS -eq 1 ]]; then
    DEFAULT_ARGS+=(--demucs-model htdemucs)
fi

# Run the application
python -m song_editor "${DEFAULT_ARGS[@]}" "$@"
