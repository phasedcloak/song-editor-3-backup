# Song Editor 3 Installation Guide

This guide will help you set up Song Editor 3 on your system.

## Prerequisites

### System Requirements

- **Operating System**: macOS 10.15+, Windows 10+, or Linux (Ubuntu 20.04+)
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space for models and dependencies
- **GPU**: Optional but recommended for faster processing (CUDA 11.8+ or Apple Silicon)

### Python Installation

1. **Download Python**: Visit [python.org](https://www.python.org/downloads/) and download Python 3.10 or higher
2. **Install Python**: Follow the installation instructions for your platform
3. **Verify Installation**: Open a terminal/command prompt and run:
   ```bash
   python3 --version
   ```

## Quick Installation

### Option 1: Automated Setup (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Song_Editor_3
   ```

2. **Run the setup script**:
   ```bash
   # On macOS/Linux:
   ./setup_environment.sh
   
   # On Windows:
   setup_environment.bat
   ```

3. **Run the application**:
   ```bash
   # On macOS/Linux:
   ./run_song_editor_3.sh
   
   # On Windows:
   run_song_editor_3.bat
   ```

### Option 2: Manual Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Song_Editor_3
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   
   # Activate the environment:
   # On macOS/Linux:
   source .venv/bin/activate
   
   # On Windows:
   .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install the package**:
   ```bash
   pip install -e .
   ```

5. **Run the application**:
   ```bash
   python -m song_editor.app
   ```

## Platform-Specific Instructions

### macOS

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python**:
   ```bash
   brew install python@3.10
   ```

3. **Install PortAudio** (for audio support):
   ```bash
   brew install portaudio
   ```

4. **Follow the Quick Installation steps above**

### Windows

1. **Install Visual Studio Build Tools** (for some dependencies):
   - Download from [Microsoft Visual Studio](https://visualstudio.microsoft.com/downloads/)
   - Install the "C++ build tools" workload

2. **Install Python**:
   - Download from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

3. **Follow the Quick Installation steps above**

### Linux (Ubuntu/Debian)

1. **Install system dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   sudo apt install portaudio19-dev python3-pyaudio
   sudo apt install libasound2-dev
   ```

2. **Follow the Quick Installation steps above**

## GPU Support (Optional)

### CUDA Support (NVIDIA GPUs)

1. **Install CUDA Toolkit 11.8**:
   - Download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
   - Follow the installation guide for your platform

2. **Install PyTorch with CUDA**:
   ```bash
   pip uninstall torch torchaudio
   pip install torch==2.2.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```

### Apple Silicon Support (M1/M2 Macs)

1. **Install PyTorch for Apple Silicon**:
   ```bash
   pip uninstall torch torchaudio
   pip install torch torchaudio
   ```

2. **Enable MPS backend** in the application settings

## Verification

After installation, verify that everything is working:

1. **Run the test suite**:
   ```bash
   python run_tests.py --fast
   ```

2. **Run a simple test**:
   ```bash
   python test_song_editor_3.py
   ```

3. **Start the application**:
   ```bash
   python -m song_editor.app
   ```

## Troubleshooting

### Common Issues

#### Import Errors

If you encounter import errors:

1. **Check virtual environment**:
   ```bash
   which python  # Should point to .venv/bin/python
   ```

2. **Reinstall dependencies**:
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

#### Audio Issues

If audio playback doesn't work:

1. **Check audio drivers** are installed and working
2. **Test with a simple audio file**:
   ```bash
   python -c "import sounddevice; print(sounddevice.query_devices())"
   ```

#### Memory Issues

If you encounter memory errors:

1. **Close other applications** to free up RAM
2. **Use smaller models** in the settings
3. **Process shorter audio files** initially

#### Model Download Issues

If models fail to download:

1. **Check internet connection**
2. **Check firewall settings**
3. **Download models manually** and place them in the models directory

### Getting Help

If you continue to have issues:

1. **Check the logs** in the application
2. **Run tests** to identify specific problems
3. **Create an issue** on the project repository with:
   - Your operating system and version
   - Python version
   - Error messages
   - Steps to reproduce the issue

## Next Steps

After successful installation:

1. **Read the README.md** for usage instructions
2. **Try the tutorial** in the application
3. **Load a test audio file** to verify functionality
4. **Explore the features** like transcription, chord detection, and export

## Uninstallation

To remove Song Editor 3:

1. **Deactivate virtual environment**:
   ```bash
   deactivate
   ```

2. **Remove the project directory**:
   ```bash
   rm -rf Song_Editor_3
   ```

3. **Remove any downloaded models** (optional):
   ```bash
   rm -rf ~/.cache/song_editor_3
   ```
