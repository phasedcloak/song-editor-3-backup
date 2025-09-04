# 🚀 Song Editor 3 Deployment Guide

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
