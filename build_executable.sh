#!/bin/bash

echo "🎯 Building Song Editor 3 Executable with Forked Process Support"
echo "================================================================"

# Check if we're in the right directory
if [ ! -f "song_editor_3.spec" ]; then
    echo "❌ Error: song_editor_3.spec not found. Are you in the correct directory?"
    exit 1
fi

# Check if transcription_worker.py exists
if [ ! -f "transcription_worker.py" ]; then
    echo "❌ Error: transcription_worker.py not found. This is required for forked transcription."
    exit 1
fi

echo "✅ Found required files:"
echo "   - song_editor_3.spec"
echo "   - transcription_worker.py"
echo "   - song_editor/app.py"
echo ""

# Clean previous build
echo "🧹 Cleaning previous build..."
rm -rf build dist
echo "✅ Cleaned build directories"
echo ""

# Build the executable
echo "🔨 Building executable with PyInstaller..."
echo "This may take several minutes..."
echo ""

pyinstaller --clean song_editor_3.spec

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Build completed successfully!"
    echo ""
    echo "📁 Build output:"
    ls -la dist/
    echo ""
    echo "🧪 To test the executable, run:"
    echo "   ./dist/SongEditor3 --help"
    echo ""
    echo "🎵 To test transcription:"
    echo "   ./dist/SongEditor3 --no-gui --whisper-model mlx-whisper your_audio_file.wav"
    echo ""
    echo "📝 Note: The transcription worker will run in subprocesses to avoid"
    echo "   library conflicts that were causing segmentation faults before."
else
    echo ""
    echo "❌ Build failed!"
    echo ""
    echo "🔍 Check the error messages above for details."
    echo "💡 Common issues:"
    echo "   - Missing dependencies (run: pip install -r requirements.txt)"
    echo "   - PyInstaller issues (try: pip install --upgrade pyinstaller)"
    echo "   - Permission issues (run with sudo if needed)"
fi