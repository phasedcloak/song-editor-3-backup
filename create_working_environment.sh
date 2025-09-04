#!/bin/bash

# Create working environment based on known working versions
echo "Creating working environment with known compatible versions..."

# Create new virtual environment
python3 -m venv .venv_working
source .venv_working/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install known working versions
echo "Installing known working package versions..."

# Core packages with working versions
pip install "numpy==1.22.0"
pip install "torch==1.13.1"
pip install "tensorflow==2.13.1"
pip install "numba==0.55.2"
pip install "librosa==0.8.1"

# OpenAI Whisper (working version)
pip install "openai-whisper==20231117"

# Additional packages needed for Song_Editor_3
pip install "soundfile"
pip install "psutil"
pip install "noisereduce"
pip install "pyloudnorm"
pip install "mido"
pip install "PySide6"
pip install "scipy==1.13.1"
pip install "matplotlib"
pip install "pandas"
pip install "scikit-learn"

# Install Song_Editor_3 in development mode
pip install -e .

echo "Working environment created successfully!"
echo "To activate: source .venv_working/bin/activate"
