#!/bin/bash

# Create a working macOS app bundle for Song Editor 3
# This avoids PyInstaller dyld issues by running Python directly

echo "🍎 Creating macOS app bundle for Song Editor 3..."

# App bundle structure
APP_NAME="Song Editor 3"
APP_BUNDLE="$APP_NAME.app"
CONTENTS_DIR="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

# Clean up any existing bundle
rm -rf "$APP_BUNDLE"

# Create directory structure
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Create Info.plist
cat > "$CONTENTS_DIR/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>SongEditor3</string>
    <key>CFBundleIdentifier</key>
    <string>com.songeditor.songeditor3</string>
    <key>CFBundleName</key>
    <string>Song Editor 3</string>
    <key>CFBundleDisplayName</key>
    <string>Song Editor 3</string>
    <key>CFBundleVersion</key>
    <string>3.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>3.0.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>NSHumanReadableCopyright</key>
    <string>© 2024 Song Editor 3</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon.icns</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.12.0</string>
    <key>NSHighResolutionCapable</key>
    <string>True</string>
    <key>NSRequiresAquaSystemAppearance</key>
    <string>False</string>
</dict>
</plist>
EOF

# Create the executable script
cat > "$MACOS_DIR/SongEditor3" << 'EOF'
#!/bin/bash

# Song Editor 3 Launcher
# This script runs the Python application directly to avoid PyInstaller issues

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
RESOURCES_DIR="$APP_DIR/Resources"

# Set up environment
export PYTHONPATH="$APP_DIR:$APP_DIR/Resources:$PYTHONPATH"
export DYLD_LIBRARY_PATH="/usr/lib:/System/Library/Frameworks:$DYLD_LIBRARY_PATH"

# Find Python executable
PYTHON_EXE=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON_EXE="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_EXE="python"
else
    echo "❌ Error: Python not found!"
    exit 1
fi

# Check if we're in a virtual environment (and it actually exists)
if [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    echo "📍 Using virtual environment: $VIRTUAL_ENV"
    PYTHON_EXE="$VIRTUAL_ENV/bin/python"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment detected but not found: $VIRTUAL_ENV"
    echo "📍 Falling back to system Python"
fi

# Run the application
echo "🚀 Starting Song Editor 3..."
echo "🐍 Python: $PYTHON_EXE"
echo "📂 App directory: $APP_DIR"
echo "📋 Command line: $@"
echo ""

# Capture original working directory before changing
export SONG_EDITOR_ORIGINAL_CWD="$(pwd)"

# Change to the app directory
cd "$APP_DIR"

# Run the Python script directly
exec "$PYTHON_EXE" -m song_editor.app "$@"
EOF

# Make the executable script executable
chmod +x "$MACOS_DIR/SongEditor3"

# Copy Python files and resources
echo "📋 Copying Python files..."
cp -r song_editor "$CONTENTS_DIR/"
cp transcription_worker.py "$CONTENTS_DIR/"
cp melody_worker.py "$CONTENTS_DIR/"
cp chord_worker.py "$CONTENTS_DIR/"
cp song_editor_3.spec "$CONTENTS_DIR/" 2>/dev/null || true

# Also copy to Resources for backup
cp -r song_editor "$RESOURCES_DIR/" 2>/dev/null || true
cp transcription_worker.py "$RESOURCES_DIR/" 2>/dev/null || true
cp melody_worker.py "$RESOURCES_DIR/" 2>/dev/null || true
cp chord_worker.py "$RESOURCES_DIR/" 2>/dev/null || true

# Copy data files if they exist
if [ -f "config.json" ]; then
    cp config.json "$RESOURCES_DIR/"
fi

if [ -f "requirements.txt" ]; then
    cp requirements.txt "$RESOURCES_DIR/"
fi

echo ""
echo "✅ macOS app bundle created successfully!"
echo ""
echo "📁 Bundle location: $APP_BUNDLE"
echo ""
echo "🧪 To test the app:"
echo "   open \"$APP_BUNDLE\""
echo ""
echo "🖥️  Or run from command line:"
echo "   \"$APP_BUNDLE/Contents/MacOS/SongEditor3\" --help"
echo ""
echo "🎵 To test transcription:"
echo "   \"$APP_BUNDLE/Contents/MacOS/SongEditor3\" --no-gui --whisper-model mlx-whisper your_audio.wav"
echo ""
echo "💡 This approach avoids PyInstaller dyld issues by running Python directly!"
