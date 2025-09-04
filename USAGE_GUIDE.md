# 🎵 Song Editor 3 - Usage Guide

## How to Run Song Editor 3

### 📋 Quick Reference

```bash
# Navigate to your project directory
cd /Volumes/SDMusicProd/CascadeProjects/Song_Editor_3

# Your preferred command (exactly as you requested):
./run_song_editor.sh --no-gui --whisper-model faster-whisper "25-03-12 we see your love - 02.wav"
```

### 🔧 Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--no-gui` | Run without GUI (batch mode) | GUI mode |
| `--whisper-model MODEL` | Choose: `openai-whisper`, `faster-whisper`, `whisperx`, `mlx-whisper` | `openai-whisper` |
| `--no-demucs` | Skip source separation | Use Demucs |
| `--no-chordino` | Skip chord detection | Use Chordino |
| `--save-intermediate` | Keep temporary files | Don't save |
| `--output-dir DIR` | Custom output directory | Auto-generated |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `--content-type TYPE` | `general`, `christian`, `gospel`, `worship`, `hymn`, `clean` | `general` |

### 🚀 Common Usage Patterns

**1. Your preferred setup (no-GUI, faster-whisper):**
```bash
./run_song_editor.sh --no-gui --whisper-model faster-whisper "25-03-12 we see your love - 02.wav"
```

**2. GUI mode (interactive):**
```bash
./run_song_editor.sh "25-03-12 we see your love - 02.wav"
```

**3. Fast processing (skip heavy features):**
```bash
./run_song_editor.sh --no-gui --no-demucs --no-chordino --whisper-model faster-whisper "25-03-12 we see your love - 02.wav"
```

**4. Debug mode (verbose output):**
```bash
./run_song_editor.sh --no-gui --log-level DEBUG --whisper-model faster-whisper "25-03-12 we see your love - 02.wav"
```

### 📁 Output Files

The program generates several output files:
- `*_processed.json` - Complete song data with lyrics, chords, melody
- `*_processed.mid` - MIDI file with chords and melody
- `*_processed.ccli` - CCLI format for church software
- Various intermediate files (if `--save-intermediate` is used)

### ⚡ Tips

- Use quotes around file paths with spaces
- The script automatically activates the virtual environment
- All models work: **Demucs**, **Basic Pitch**, **Faster Whisper**, **CREPE**
- Chord detection falls back gracefully if vamp has issues
- Supported audio formats: WAV, MP3, FLAC, M4A, AAC, OGG, WMA, OPUS, AIFF, ALAC

### 🎯 Features

✅ **Core Functionality:**
- Audio transcription using multiple Whisper models
- Melody extraction with Basic Pitch
- Source separation with Demucs
- Chord detection (with fallback)
- Multi-format export (JSON, MIDI, CCLI)

⚠️ **Optional Features:**
- VAMP-based chord detection (architecture compatibility issues on Apple Silicon)
- MLX Whisper (not available on this system)

### 🏃‍♂️ Quick Start Examples

```bash
# Basic transcription only
./run_song_editor.sh --no-gui --no-demucs --no-chordino "your-song.wav"

# Full processing pipeline
./run_song_editor.sh --no-gui --whisper-model faster-whisper "your-song.wav"

# Process with GUI for review
./run_song_editor.sh "your-song.wav"
```

---

**Created:** $(date)
**Version:** Song Editor 3.0.0
