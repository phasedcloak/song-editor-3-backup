# 🚀 Song Editor 3 - PyInstaller Build Guide

## Overview

This guide covers building Song Editor 3 into a standalone executable using PyInstaller, creating a deployable application that doesn't require Python or external dependencies on the target system.

## 🎯 Quick Start

### 1. Prepare Environment
```bash
# Ensure virtual environment is set up
./run_song_editor_3.sh

# Or create manually
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build Executable
```bash
# Use the automated build script
./build_executable.sh
```

### 3. Test Build
```bash
# Test the built executable
python test_built_executable.py
```

## 📁 Build Output

After successful build, you'll find:

- **macOS**: `dist/Song Editor 3.app` (app bundle)
- **Windows**: `dist/SongEditor3.exe` (standalone executable)
- **Linux**: `dist/SongEditor3` (standalone executable)

## 🛠️ Build Scripts

### `build_executable.sh` - Main Build Script
- Automated build process
- Platform detection and optimization
- Distribution package creation
- Build verification

### `build_app.py` - Spec File Generator
- Generates PyInstaller spec files
- Configures hidden imports
- Sets up data file inclusion
- Optimizes for each platform

### `test_built_executable.py` - Verification Script
- Tests all imports and functionality
- Verifies model loading
- Checks GUI creation
- Validates file operations

## 🔧 Build Configuration

### PyInstaller Spec File (`song_editor_3.spec`)
Key configurations:
- **Hidden Imports**: All dependencies explicitly included
- **Data Files**: Config files, models, UI resources
- **Exclusions**: Unnecessary packages removed for size
- **Platform Specific**: macOS app bundle, Windows console hiding

### Custom Hooks
- **`hooks/hook-torch.py`**: PyTorch CUDA and MKL library handling
- Ensures GPU acceleration works in built executable

## 🎵 Included Features

### ✅ Working Models
- **OpenAI Whisper**: Full transcription with all model sizes
- **Faster Whisper**: GPU-accelerated transcription
- **WhisperX**: Word-level timestamp alignment
- **MLX Whisper**: Apple Silicon optimization (with fallback)

### ✅ Audio Processing
- **Demucs**: Source separation (vocals, drums, bass, other)
- **Noise Reduction**: Audio denoising
- **Loudness Normalization**: EBU R128 compliant
- **Format Support**: WAV, MP3, FLAC, M4A, AAC, OGG

### ✅ Analysis Features
- **Chord Detection**: chord_extractor + VAMP plugins
- **Pitch Detection**: CREPE algorithm
- **Melody Extraction**: Basic Pitch
- **MIDI Processing**: Export capabilities

## 🏗️ Build Process Details

### Step 1: Environment Setup
```bash
source .venv/bin/activate
pip install pyinstaller
```

### Step 2: Generate Spec File
```bash
python build_app.py
```

### Step 3: Build Executable
```bash
pyinstaller --clean song_editor_3.spec
```

### Step 4: Verify Build
```bash
python test_built_executable.py
```

## 📦 Distribution

### macOS
```bash
# Create DMG (requires create-dmg)
brew install create-dmg
create-dmg --volname "Song Editor 3" SongEditor3.dmg dist/
```

### Windows
```bash
# Use NSIS or Inno Setup for installer
# The dist/ folder contains the standalone executable
```

### Linux
```bash
# Create AppImage or distribute the executable directly
# The dist/ folder contains the standalone executable
```

## 🔍 Troubleshooting

### Common Issues

#### 1. "Module not found" in built executable
**Solution**: Add missing module to `hiddenimports` in spec file
```python
hiddenimports = [
    # Add your missing module here
    'your_missing_module',
]
```

#### 2. PyTorch CUDA not working
**Solution**: Ensure CUDA libraries are included
- Check `hooks/hook-torch.py`
- Verify CUDA installation on build machine

#### 3. GUI not displaying
**Solution**: Platform plugin issues
- Ensure PySide6 platform plugins are included
- Check `QT_QPA_PLATFORM` environment variable

#### 4. Large bundle size
**Solution**: Optimize exclusions
```python
excludes = [
    'matplotlib.tests',
    'numpy.tests',
    'torch.test',
    # Add more exclusions
]
```

### Build Machine Requirements

#### Minimum
- Python 3.8+
- 8GB RAM
- 10GB free disk space

#### Recommended
- Python 3.10+
- 16GB RAM
- 20GB free disk space
- GPU with CUDA (for GPU-accelerated models)

## 🚀 Advanced Configuration

### Custom Spec File
Edit `song_editor_3.spec` for:
- Custom icons
- Version information
- Bundle identifiers
- Code signing

### Environment Variables
```bash
# GPU acceleration
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Qt platform
export QT_QPA_PLATFORM="cocoa"  # macOS
export QT_QPA_PLATFORM="windows"  # Windows

# Debugging
export PYINSTALLER_DEBUG=1
```

### Size Optimization
```python
# In spec file
upx=True  # Enable UPX compression
strip=True  # Strip debug symbols
exclude_binaries=True  # Create one-folder distribution
```

## 🧪 Testing Strategy

### Pre-Build Testing
```bash
# Test all functionality before building
python test_models.py
```

### Post-Build Testing
```bash
# Test the built executable
python test_built_executable.py

# Manual testing
./dist/SongEditor3  # Run the executable
```

### Cross-Platform Testing
- Test on multiple target platforms
- Verify all features work
- Check file I/O operations
- Validate model loading

## 📋 Deployment Checklist

- [ ] Build completed successfully
- [ ] All tests pass
- [ ] Executable runs on target platform
- [ ] All models load correctly
- [ ] GUI displays properly
- [ ] File operations work
- [ ] Audio processing functions
- [ ] Reasonable bundle size
- [ ] Distribution package created
- [ ] Version information correct
- [ ] Documentation included

## 🎯 Performance Considerations

### Build Time
- **Small build**: ~5-10 minutes
- **Full build**: ~15-30 minutes
- **With CUDA**: ~30-45 minutes

### Bundle Size
- **Minimal**: ~200-300MB
- **Full**: ~500-800MB
- **With CUDA**: ~1-2GB

### Startup Time
- **First run**: ~10-30 seconds (model loading)
- **Subsequent runs**: ~5-10 seconds

## 🔐 Code Signing & Notarization

### macOS
```bash
# Sign the app
codesign --deep --force --verbose --sign "Developer ID" dist/Song\ Editor\ 3.app

# Notarize (requires Apple Developer account)
xcrun notarytool submit dist/SongEditor3.dmg --keychain-profile "notary-profile"
```

### Windows
```bash
# Sign the executable
signtool sign /f cert.pfx /p password dist/SongEditor3.exe
```

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run the test scripts to identify specific failures
3. Verify all dependencies are installed
4. Check PyInstaller and Python versions
5. Review the build log for error messages

---

**🎉 Happy building! Your Song Editor 3 executable will be ready for deployment in minutes.**
