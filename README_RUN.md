# Quick Start - Song Editor 3

## 🚀 Running the Application

Use the provided launcher script for the most reliable experience:

```bash
# Make sure the script is executable (first time only)
chmod +x run_song_editor.sh

# Run with GUI (default)
./run_song_editor.sh

# Process a single audio file
./run_song_editor.sh path/to/your/audio.wav

# Process without GUI (batch mode)
./run_song_editor.sh path/to/your/audio.wav --no-gui

# Process all files in a directory
./run_song_editor.sh --input-dir ./audio_files --no-gui

# Use specific Whisper model
./run_song_editor.sh audio.wav --whisper-model mlx-whisper --no-gui
```

## 📁 What the Script Does

1. **Activates the virtual environment** (`.venv`)
2. **Sets up Python path** correctly
3. **Runs the Song Editor 3 application** with your arguments

## 🎯 Features

- ✅ **MLX Whisper** support with `large-v3-turbo` model
- ✅ **Audio source separation** (Demucs)
- ✅ **Chord detection** and **melody extraction**
- ✅ **Multiple output formats**: JSON, MIDI, CCLI text
- ✅ **Batch processing** capability
- ✅ **GUI and CLI** modes

## 📋 Requirements

- macOS (tested on macOS 15)
- Virtual environment set up (run `setup_environment.sh` if needed)
- Audio files in WAV, MP3, M4A, FLAC, or OGG format

## 🔧 Troubleshooting

If you get permission errors:
```bash
chmod +x run_song_editor.sh
```

If you get environment errors, make sure the virtual environment exists:
```bash
ls -la .venv/
```

The script automatically handles all the complex setup - just run it!
