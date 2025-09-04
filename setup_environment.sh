#!/bin/bash

# Song Editor 3 Environment Setup Script
# This script sets up the development environment for Song Editor 3

echo "Setting up Song Editor 3 environment..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the project directory
cd "$SCRIPT_DIR"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

echo "Environment setup complete!"
echo ""
echo "To activate the environment manually:"
echo "  source .venv/bin/activate"
echo ""
echo "To run Song Editor 3:"
echo "  ./run_song_editor_3.sh"
echo "  or"
echo "  python -m song_editor.app"
