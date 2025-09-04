# Cross-Platform Implementation Guide

## Overview

Song Editor 3 has been enhanced with comprehensive cross-platform support for:
- **macOS** (desktop)
- **iOS** (mobile)
- **Windows** (desktop)
- **Android** (mobile)
- **Linux** (desktop)

The implementation maintains all existing functionality while providing platform-appropriate user experiences.

## Architecture

### Platform Detection System

The application automatically detects the current platform and applies appropriate configurations:

```python
from song_editor.platform_utils import PlatformUtils

# Detect platform
platform_type = PlatformUtils.detect_platform()  # Returns Platform enum

# Get platform-specific configuration
config = PlatformUtils.get_platform_config()
```

### Platform-Specific Features

| Platform | UI Style | Font | Touch Support | High DPI | Dark Mode | Gestures |
|----------|----------|------|---------------|----------|-----------|----------|
| macOS | Native | SF Pro Display | No | Yes | Yes | No |
| iOS | Mobile | SF Pro Display | Yes | Yes | Yes | Yes |
| Windows | Native | Segoe UI | No | Yes | Yes | No |
| Android | Mobile | Roboto | Yes | Yes | Yes | Yes |
| Linux | Native | Ubuntu | No | Yes | Yes | No |

## Implementation Details

### 1. Platform Utilities (`song_editor/platform_utils.py`)

**Key Features:**
- Automatic platform detection
- Platform-specific configurations
- Touch support detection
- High DPI support
- Mobile vs desktop detection

**Usage:**
```python
# Check if running on mobile
if PlatformUtils.is_mobile():
    # Apply mobile-specific optimizations
    
# Check touch support
if PlatformUtils.is_touch_supported():
    # Enable touch interactions
    
# Get platform-specific colors
accent_color = PlatformUtils.get_accent_color()
background_color = PlatformUtils.get_background_color()
```

### 2. Platform-Aware Styling (`song_editor/ui/platform_styles.py`)

**Key Features:**
- Platform-specific stylesheets
- Mobile optimizations
- High DPI settings
- Touch target sizing

**Platform-Specific Styles:**

#### macOS Style
- SF Pro Display font
- Blue accent color (#007AFF)
- Rounded corners
- Native window shadows

#### iOS Style
- Larger touch targets (44px minimum)
- Increased font sizes
- Gesture support
- Safe area insets

#### Windows Style
- Segoe UI font
- Windows blue accent (#0078D4)
- Square corners
- Native Windows styling

#### Android Style
- Roboto font
- Material Design purple (#6200EE)
- Material Design styling
- Touch-optimized controls

### 3. Platform-Aware Main Window (`song_editor/ui/main_window.py`)

**Key Features:**
- Automatic UI layout adaptation
- Mobile-optimized controls
- Desktop splitter layout
- Platform-specific window sizing

**Mobile Layout:**
- Vertical scroll layout
- Large touch targets
- Simplified menu structure
- Tab-based navigation

**Desktop Layout:**
- Horizontal splitter
- Traditional menu bar
- Toolbar
- Side-by-side editors

### 4. Cross-Platform Build System (`build_config.py`)

**Supported Build Targets:**
- macOS: `.app` bundle with PyInstaller
- iOS: iOS app bundle
- Windows: `.exe` with optional installer
- Android: `.apk` with Python-for-Android
- Linux: AppImage and Snap packages

## Usage Examples

### Platform Detection
```python
from song_editor.platform_utils import PlatformUtils

# Get comprehensive platform info
info = PlatformUtils.get_platform_info()
print(f"Running on: {info['platform']}")
print(f"Mobile: {info['is_mobile']}")
print(f"Touch support: {info['is_touch_supported']}")
```

### Platform-Aware UI Creation
```python
from song_editor.ui.main_window import MainWindow
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)

# Main window automatically adapts to platform
window = MainWindow()
window.show()
```

### Platform-Specific Configuration
```python
from song_editor.platform_utils import PlatformUtils

# Get platform-specific settings
config = PlatformUtils.get_platform_config()

# Apply platform-specific behavior
if config['ui_style'] == 'mobile':
    # Mobile-specific optimizations
    pass
elif config['ui_style'] == 'macos':
    # macOS-specific features
    pass
```

## Build Instructions

### Prerequisites

**For all platforms:**
```bash
pip install -r requirements.txt
```

**Platform-specific requirements:**

#### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Install PyInstaller
pip install pyinstaller
```

#### Windows
```bash
# Install Visual Studio Build Tools
# Install PyInstaller
pip install pyinstaller
```

#### Linux
```bash
# Install system dependencies
sudo apt-get install python3-dev build-essential

# Install PyInstaller
pip install pyinstaller
```

#### Android
```bash
# Install Python-for-Android
pip install python-for-android

# Install Android SDK and NDK
# Set ANDROID_HOME environment variable
```

### Building

**Build for current platform:**
```bash
python build_config.py
```

**Build for specific platform:**
```bash
python build_config.py --platform macos
python build_config.py --platform windows
python build_config.py --platform android
python build_config.py --platform linux
```

**Show build configuration:**
```bash
python build_config.py --config
```

**Clean build artifacts:**
```bash
python build_config.py --clean
```

## Platform-Specific Considerations

### macOS

**Features:**
- Native macOS appearance
- High DPI support
- Dark mode support
- App bundle packaging
- Code signing and notarization

**Requirements:**
- macOS 10.14 or later
- Xcode command line tools
- Apple Developer account (for distribution)

### iOS

**Features:**
- Touch-optimized interface
- Gesture support
- Safe area handling
- iOS-specific styling
- App Store distribution

**Requirements:**
- macOS with Xcode
- iOS Developer account
- iOS 14.0+ deployment target

### Windows

**Features:**
- Native Windows appearance
- Windows installer (NSIS)
- MSI package support
- Windows Store distribution

**Requirements:**
- Windows 10 or later
- Visual Studio Build Tools
- Windows Developer account (for Store)

### Android

**Features:**
- Material Design styling
- Touch-optimized controls
- Android permissions handling
- Google Play Store distribution

**Requirements:**
- Android SDK and NDK
- Python-for-Android
- Google Play Developer account

### Linux

**Features:**
- Native Linux appearance
- AppImage packaging
- Snap package support
- Flatpak support

**Requirements:**
- Linux distribution with Python 3.8+
- Build tools (gcc, make)
- AppImage tools (for AppImage)

## Testing

### Platform Testing

**Test platform detection:**
```bash
python -m song_editor.app --platform-info
```

**Test on different platforms:**
- Use virtual machines for cross-platform testing
- Use cloud-based CI/CD for automated testing
- Test on physical devices for mobile platforms

### UI Testing

**Test responsive design:**
- Resize windows on desktop platforms
- Test different screen orientations on mobile
- Test high DPI displays

**Test touch interactions:**
- Test on touch-enabled devices
- Verify touch target sizes
- Test gesture support

## Deployment

### Desktop Platforms

**macOS:**
```bash
# Build app bundle
python build_config.py --platform macos

# Code sign (requires developer account)
codesign --deep --force --verify --verbose --sign "Developer ID Application: Your Name" dist/Song\ Editor\ 3.app

# Notarize (requires Apple ID)
xcrun altool --notarize-app --primary-bundle-id "com.songeditor.app" --username "your-apple-id" --password "app-specific-password" --file "Song Editor 3.zip"
```

**Windows:**
```bash
# Build executable
python build_config.py --platform windows

# Create installer (requires NSIS)
makensis installer.nsi
```

**Linux:**
```bash
# Build AppImage
python build_config.py --platform linux

# Create Snap package
snapcraft
```

### Mobile Platforms

**iOS:**
```bash
# Build iOS app
python build_config.py --platform ios

# Archive for App Store
xcodebuild -archivePath "Song Editor 3.xcarchive" -exportPath "Song Editor 3.ipa" -exportOptionsPlist "exportOptions.plist"
```

**Android:**
```bash
# Build APK
python build_config.py --platform android

# Sign APK
jarsigner -verbose -sigalg SHA1withRSA -digestalg SHA1 -keystore my-release-key.keystore app-release-unsigned.apk alias_name

# Optimize APK
zipalign -v 4 app-release-unsigned.apk Song-Editor-3.apk
```

## Troubleshooting

### Common Issues

**Build Failures:**
- Ensure all dependencies are installed
- Check platform-specific requirements
- Verify Python version compatibility
- Check for missing system libraries

**UI Issues:**
- Verify platform detection is working
- Check stylesheet application
- Test on target platform
- Verify touch support detection

**Performance Issues:**
- Optimize for target platform
- Use platform-specific optimizations
- Profile on target hardware
- Consider platform limitations

### Platform-Specific Issues

**macOS:**
- Code signing issues: Check certificates and provisioning profiles
- Notarization failures: Verify Apple ID and app-specific passwords
- High DPI issues: Test on Retina displays

**Windows:**
- DLL missing errors: Include all required libraries
- Installer issues: Check NSIS script syntax
- UAC issues: Request appropriate permissions

**Linux:**
- Library dependencies: Install required system packages
- AppImage issues: Check AppImage tools installation
- Snap issues: Verify snapcraft configuration

**Mobile:**
- Touch target size: Ensure minimum 44px touch targets
- Performance: Optimize for mobile hardware
- Permissions: Request appropriate permissions
- App store guidelines: Follow platform-specific guidelines

## Future Enhancements

### Planned Features

1. **Enhanced Mobile Support**
   - Offline processing capabilities
   - Cloud sync integration
   - Mobile-optimized processing

2. **Platform-Specific Features**
   - macOS: Touch Bar support
   - Windows: Windows Ink support
   - iOS: Apple Pencil support
   - Android: Stylus support

3. **Advanced Build System**
   - Automated CI/CD pipelines
   - Cross-compilation support
   - Automated testing
   - Distribution automation

4. **Performance Optimizations**
   - Platform-specific optimizations
   - Hardware acceleration
   - Memory management improvements
   - Battery optimization for mobile

## Conclusion

The cross-platform implementation of Song Editor 3 provides a consistent, high-quality experience across all supported platforms while maintaining the full functionality of the application. The platform-aware design ensures that users get the most appropriate interface for their device while preserving all the powerful audio processing capabilities.

The modular architecture makes it easy to add new platforms or enhance existing platform support, ensuring that Song Editor 3 can continue to evolve and support new devices and operating systems as they become available.
