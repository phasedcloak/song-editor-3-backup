#!/bin/bash

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

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip and install PyInstaller if needed
echo "📦 Installing/Upgrading PyInstaller..."
pip install --upgrade pip
pip install pyinstaller

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build dist *.spec

# Create spec file
echo "📝 Creating PyInstaller spec file..."
python build_app.py

# Build the application
echo "🏗️  Building standalone application..."
echo "   This may take several minutes..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS build
    echo "🍎 Building for macOS..."
    pyinstaller --clean song_editor_3.spec

    # Create DMG for distribution
    echo "📦 Creating DMG for distribution..."
    if command -v create-dmg &> /dev/null; then
        create-dmg \
            --volname "Song Editor 3" \
            --volicon "icon.icns" \
            --window-pos 200 120 \
            --window-size 800 400 \
            --icon-size 100 \
            --icon "Song Editor 3.app" 200 190 \
            --hide-extension "Song Editor 3.app" \
            --app-drop-link 600 185 \
            "SongEditor3.dmg" \
            "dist/"
    fi

elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows build
    echo "🪟 Building for Windows..."
    pyinstaller --clean --noconsole song_editor_3.spec

    # Create installer
    echo "📦 Creating Windows installer..."
    if command -v makensis &> /dev/null; then
        # NSIS installer script would go here
        echo "   NSIS installer creation not implemented yet"
    fi

else
    # Linux build
    echo "🐧 Building for Linux..."
    pyinstaller --clean song_editor_3.spec

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
    echo "   open dist/Song\ Editor\ 3.app"
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
