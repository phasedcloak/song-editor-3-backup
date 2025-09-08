#!/usr/bin/env python3
"""
PyInstaller Build Script for Song Editor 3

Creates a standalone executable for Song Editor 3 with all dependencies bundled.
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

def create_spec_file():
    """Create PyInstaller spec file with proper configuration."""

    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_data_files

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(SPEC))
sys.path.insert(0, current_dir)

# Prepare data files
datas_list = [
    ('config.json', '.'),
    ('song_editor/ui', 'song_editor/ui'),
    ('requirements.txt', '.'),
    ('transcription_worker.py', '.'),
    ('melody_worker.py', '.'),
    ('chord_worker.py', '.'),
]
if os.path.exists('models'):
    datas_list.append(('models', 'models'))

# Include package metadata required by importlib.metadata for cmudict
try:
    datas_list += copy_metadata('cmudict')
except Exception:
    pass

# Include whisper assets (mel_filters.npz etc.) if present
try:
    datas_list += collect_data_files('whisper', subdir='assets')
except Exception:
    pass

# Bundle demucs remote assets directory if present in site-packages
try:
    import demucs, importlib.util, os as _os
    _spec = importlib.util.find_spec('demucs')
    if _spec and _spec.submodule_search_locations:
        _pkg_dir = list(_spec.submodule_search_locations)[0]
        _remote = _os.path.join(_pkg_dir, 'remote')
        if _os.path.isdir(_remote):
            datas_list.append((_remote, 'demucs/remote'))
except Exception:
    pass

# Define analysis
a = Analysis(
    ['song_editor/app.py'],
    pathex=[current_dir],
    binaries=[],
    datas=datas_list,
    hiddenimports=[
        # Core dependencies
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'torch',
        'torchaudio',
        'numpy',
        'scipy',
        'librosa',
        'soundfile',
        'pydub',

        # Whisper models
        'whisper',
        'faster_whisper',
        'whisperx',
        'mlx_whisper',

        # Audio processing
        'demucs',
        'demucs.separate',
        'demucs.hdemucs',
        'demucs.pretrained',
        'noisereduce',
        'pyloudnorm',

        # Analysis
        'chord_extractor',
        'crepe',
        'vamp',
        'basic_pitch',
        'pretty_midi',
        'cmudict',
        'mido',

        # Additional hidden imports for PyTorch and CUDA
        'torchvision',
        'torchaudio.lib.libtorchaudio',
        'torch._C',
        'torch._C._fft',
        'torch._C._linalg',
        'torch._C._nn',
        'torch._C._sparse',
        'torch._C._special',
        'torchvision._C',

        # NumPy and SciPy
        'numpy.core._methods',
        'numpy.lib.format',
        'scipy.sparse.csgraph._validation',
        'scipy._lib.messagestream',

        # Audio libraries
        'soundfile',
        'audioread',
        'librosa.core.audio',
        'librosa.core.spectrum',
        'librosa.core.pitch',

        # Qt platform plugins
        'PySide6.QtPlugins.platforms',
        'PySide6.QtPlugins.styles',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'tkinter',
        'matplotlib.tests',
        'numpy.tests',
        'PIL.ImageQt',
        'PIL.ImageTk',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Remove debug and unnecessary files
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SongEditor3',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False for production (no console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path if available
)

# Create app bundle for macOS
app = BUNDLE(
    exe,
    name='Song Editor 3.app',
    icon=None,
    bundle_identifier='com.songeditor.songeditor3',
    version='3.0.0',
    info_plist={
        'NSHighResolutionCapable': 'True',
        'NSRequiresAquaSystemAppearance': 'False',
        'CFBundleShortVersionString': '3.0.0',
        'CFBundleVersion': '3.0.0',
        'CFBundleIdentifier': 'com.songeditor.songeditor3',
        'LSMinimumSystemVersion': '10.12.0',
        'NSHumanReadableCopyright': '© 2024 Song Editor 3',
        'CFBundleName': 'Song Editor 3',
        'CFBundleDisplayName': 'Song Editor 3',
        'CFBundlePackageType': 'APPL',
        'CFBundleSignature': '????',
        'CFBundleExecutable': 'SongEditor3',
    },
)
'''

    with open('song_editor_3.spec', 'w') as f:
        f.write(spec_content)

    print("✅ Created PyInstaller spec file: song_editor_3.spec")

def create_build_script():
    """Create build script for different platforms."""

    build_script = '''#!/bin/bash

# Song Editor 3 PyInstaller Build Script
# Creates standalone executables for deployment

set -e  # Exit on any error

echo "🎵 Building Song Editor 3 Standalone Application"
echo "================================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Run setup first:"
    echo "   ./run_song_editor_3.sh"
    exit 1
fi

# Activate virtual environment and resolve interpreter
echo "🔧 Activating virtual environment..."
source .venv/bin/activate
VENV_PY="$(pwd)/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "❌ Virtualenv python not found at $VENV_PY"
    exit 1
fi

# Upgrade pip and install PyInstaller if needed
echo "📦 Installing/Upgrading PyInstaller..."
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install --upgrade pyinstaller

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build dist *.spec

# Create spec file
echo "📝 Creating PyInstaller spec file..."
"$VENV_PY" build_app.py

# Build the application
echo "🏗️  Building standalone application..."
echo "   This may take several minutes..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS build
    echo "🍎 Building for macOS..."
    "$VENV_PY" -m PyInstaller --clean song_editor_3.spec

    # Create DMG for distribution
    echo "📦 Creating DMG for distribution..."
    if command -v create-dmg &> /dev/null; then
        create-dmg \\
            --volname "Song Editor 3" \\
            --volicon "icon.icns" \\
            --window-pos 200 120 \\
            --window-size 800 400 \\
            --icon-size 100 \\
            --icon "Song Editor 3.app" 200 190 \\
            --hide-extension "Song Editor 3.app" \\
            --app-drop-link 600 185 \\
            "SongEditor3.dmg" \\
            "dist/"
    fi

elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows build
    echo "🪟 Building for Windows..."
    "$VENV_PY" -m PyInstaller --clean --noconsole song_editor_3.spec

    # Create installer
    echo "📦 Creating Windows installer..."
    if command -v makensis &> /dev/null; then
        # NSIS installer script would go here
        echo "   NSIS installer creation not implemented yet"
    fi

else
    # Linux build
    echo "🐧 Building for Linux..."
    "$VENV_PY" -m PyInstaller --clean song_editor_3.spec

    # Create AppImage
    echo "📦 Creating AppImage..."
    if command -v appimagetool &> /dev/null; then
        echo "   AppImage creation not implemented yet"
    fi
fi

echo ""
echo "✅ Build completed!"
echo ""
echo "📂 Output location: dist/"
echo ""
echo "🎯 To run the application:"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "   open dist/Song\\ Editor\\ 3.app"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "   dist/SongEditor3.exe"
else
    echo "   ./dist/SongEditor3"
fi

echo ""
echo "📊 Build size:"
du -sh dist/

echo ""
echo "🎉 Song Editor 3 is ready for deployment!"
'''

    with open('build_app_executable.sh', 'w') as f:
        f.write(build_script)

    # Make executable
    os.chmod('build_app_executable.sh', 0o755)

    print("✅ Created build script: build_app_executable.sh")

def create_requirements_build():
    """Create a requirements file optimized for PyInstaller builds."""

    build_requirements = '''# Song Editor 3 - Build Requirements
# Optimized for PyInstaller packaging

# Core dependencies
PySide6>=6.5.0
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
librosa>=0.10.0

# Audio processing
soundfile>=0.12.0
noisereduce>=3.0.0
pyloudnorm>=0.4.1
demucs>=4.0.0

# Transcription models
openai-whisper>=20230314
faster-whisper>=1.0.0
whisperx>=3.0.0

# Analysis
chord-extractor>=0.1.0
crepe>=0.0.16
basic-pitch>=0.4.0
pretty-midi>=0.2.10
mido>=1.3.0

# Additional utilities
pathlib2>=2.3.0
typing-extensions>=4.5.0
packaging>=23.0

# Build tools
pyinstaller>=6.0.0

# Optional MLX Whisper (Apple Silicon)
# mlx-whisper>=0.1.0  # Uncomment if building for Apple Silicon
'''

    with open('requirements-build.txt', 'w') as f:
        f.write(build_requirements)

    print("✅ Created optimized build requirements: requirements-build.txt")

def create_icon_placeholder():
    """Create a placeholder for application icon."""

    icon_script = '''#!/bin/bash

# Create application icon for Song Editor 3
# This is a placeholder - you'll need to create actual icons

echo "🎨 Creating application icons..."
echo ""
echo "For macOS (.icns):"
echo "1. Create a 1024x1024 PNG icon"
echo "2. Use iconutil to convert to .icns:"
echo "   iconutil -c icns icon.iconset"
echo ""
echo "For Windows (.ico):"
echo "1. Create PNG icons in sizes: 16x16, 32x32, 48x48, 256x256"
echo "2. Use ImageMagick to convert:"
echo "   convert icon-16.png icon-32.png icon-48.png icon-256.png icon.ico"
echo ""
echo "For Linux (.png):"
echo "   Use 512x512 PNG icon"
echo ""
echo "Place icons in the project root directory as:"
echo "   icon.icns (macOS)"
echo "   icon.ico (Windows)"
echo "   icon.png (Linux)"
'''

    with open('create_icons.sh', 'w') as f:
        f.write(icon_script)

    os.chmod('create_icons.sh', 0o755)

    print("✅ Created icon creation script: create_icons.sh")

def create_deployment_readme():
    """Create deployment documentation."""

    deployment_docs = '''# 🚀 Song Editor 3 Deployment Guide

## PyInstaller Build Process

### Prerequisites

1. **Python Environment**: Ensure all dependencies are installed
2. **PyInstaller**: Install with `pip install pyinstaller`
3. **System Dependencies**: Make sure audio libraries are available

### Building the Application

#### Option 1: Automated Build Script (Recommended)
```bash
# Make executable
chmod +x build_app_executable.sh

# Run the build
./build_app_executable.sh
```

#### Option 2: Manual Build
```bash
# Activate virtual environment
source .venv/bin/activate

# Install PyInstaller
pip install pyinstaller

# Create spec file
python build_app.py

# Build application
pyinstaller --clean song_editor_3.spec
```

### Output Locations

- **macOS**: `dist/Song Editor 3.app`
- **Windows**: `dist/SongEditor3.exe`
- **Linux**: `dist/SongEditor3`

### Distribution

#### macOS
```bash
# Create DMG (requires create-dmg)
create-dmg --volname "Song Editor 3" dist/SongEditor3.dmg dist/
```

#### Windows
```bash
# Create installer (requires NSIS)
makensis installer.nsi
```

#### Linux
```bash
# Create AppImage (requires appimagetool)
# AppImage creation requires additional setup
```

## Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Add missing modules to `hiddenimports` in spec file
   - Check that all dependencies are installed

2. **Qt/GUI issues**
   - Ensure PySide6 is properly included
   - Check platform plugins are bundled

3. **Torch/CUDA issues**
   - PyTorch may need special handling for CUDA
   - Consider building without CUDA for broader compatibility

4. **Large bundle size**
   - Use `excludes` to remove unnecessary packages
   - Consider UPX compression

### Optimization Tips

1. **Reduce Size**
   ```python
   # In spec file
   excludes=['matplotlib.tests', 'numpy.tests', 'PIL']
   upx=True
   ```

2. **Improve Startup Time**
   ```python
   # In spec file
   noarchive=False
   ```

3. **Cross-Platform Compatibility**
   - Test on target platform before distribution
   - Consider platform-specific builds

## Testing the Build

1. **Run the executable**
2. **Test all features**
3. **Verify model loading**
4. **Check GUI responsiveness**

```bash
# Test the built application
cd dist
./SongEditor3  # or SongEditor3.exe on Windows
```

## Deployment Checklist

- [ ] Build tested on target platform
- [ ] All models load correctly
- [ ] GUI displays properly
- [ ] Audio processing works
- [ ] File I/O functions correctly
- [ ] Application exits cleanly
- [ ] Bundle size is reasonable
- [ ] Distribution package created

## Platform-Specific Notes

### macOS
- Ensure code signing for distribution
- Test on multiple macOS versions
- Consider notarization for App Store

### Windows
- Test on different Windows versions
- Consider Windows Store packaging
- Handle UAC permissions if needed

### Linux
- Test on different distributions
- Consider snap/flatpak packaging
- Handle library dependencies

---

**🎯 Remember**: Test thoroughly on the target platform before distribution!
'''

    with open('DEPLOYMENT_README.md', 'w') as f:
        f.write(deployment_docs)

    print("✅ Created deployment documentation: DEPLOYMENT_README.md")

def main():
    """Main build setup function."""

    print("🎵 Setting up PyInstaller build for Song Editor 3")
    print("=" * 60)

    # Create all necessary files
    create_spec_file()
    create_build_script()
    create_requirements_build()
    create_icon_placeholder()
    create_deployment_readme()

    print("\n" + "=" * 60)
    print("✅ PyInstaller setup complete!")
    print("\n📁 Files created:")
    print("   • song_editor_3.spec - PyInstaller specification")
    print("   • build_app_executable.sh - Build script")
    print("   • requirements-build.txt - Build requirements")
    print("   • create_icons.sh - Icon creation helper")
    print("   • DEPLOYMENT_README.md - Deployment guide")
    print("\n🚀 To build the application:")
    print("   ./build_app_executable.sh")
    print("\n🎯 The resulting executable will be in: dist/")

if __name__ == "__main__":
    main()
