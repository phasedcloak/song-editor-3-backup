#!/bin/bash

# Song Editor 3 Installation from Archive
# This script extracts and sets up Song Editor 3 from the distribution archive

set -e  # Exit on any error

echo "🎵 Song Editor 3 - Archive Installation"
echo "======================================"

# Find the archive file
ARCHIVE_FILE=""
if [ -f "Song_Editor_3.tar.gz" ]; then
    ARCHIVE_FILE="Song_Editor_3.tar.gz"
elif [ -f "~/Desktop/Song_Editor_3.tar.gz" ]; then
    ARCHIVE_FILE="~/Desktop/Song_Editor_3.tar.gz"
else
    echo "❌ Could not find Song_Editor_3.tar.gz"
    echo "Please place the archive file in the current directory or on your Desktop"
    exit 1
fi

echo "📦 Found archive: $ARCHIVE_FILE"

# Check if target directory already exists
if [ -d "Song_Editor_3_Installed" ]; then
    echo "⚠️  Installation directory already exists"
    read -p "Remove existing installation? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf Song_Editor_3_Installed
    else
        echo "Installation cancelled"
        exit 0
    fi
fi

# Extract the archive
echo "📤 Extracting archive..."
tar -xzf "$ARCHIVE_FILE"

# Rename the extracted directory
if [ -d "dist" ]; then
    mv dist Song_Editor_3_Installed
fi

echo ""
echo "✅ Extraction complete!"
echo ""
echo "🚀 To run Song Editor 3:"
echo "   cd Song_Editor_3_Installed"
echo "   chmod +x setup.sh run_song_editor.sh"
echo "   ./setup.sh"
echo "   ./run_song_editor.sh"
echo ""
echo "📁 Installation directory: $(pwd)/Song_Editor_3_Installed"

