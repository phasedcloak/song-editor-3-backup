# Song Editor 3

A professional song editing and transcription application with OpenAI Whisper, Chordino, and advanced audio processing capabilities.

## Features

- **Audio Transcription**: Powered by OpenAI Whisper for accurate speech-to-text conversion
- **Chord Detection**: Advanced chord recognition using Chordino and other algorithms
- **Melody Extraction**: Extract melodic lines from audio using Basic Pitch and CREPE
- **Source Separation**: Separate vocals, drums, bass, and other instruments using Demucs
- **MIDI Export**: Export detected chords and melodies to MIDI format
- **Lyrics Editor**: Advanced lyrics editing with timing synchronization
- **Block View**: Visual block-based interface for song structure editing
- **CCLI Export**: Export song data in CCLI format for worship services
- **JSON Export**: Comprehensive data export in JSON format

## Requirements

- Python 3.10 or higher
- macOS, Windows, or Linux
- Sufficient RAM for audio processing (8GB+ recommended)
- GPU support optional but recommended for faster processing

## Installation

### Quick Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Song_Editor_3
```

2. Run the setup script:
```bash
./setup_environment.sh
```

3. Run the application:
```bash
./run_song_editor_3.sh
```

See [USAGE_GUIDE.md](./USAGE_GUIDE.md) for detailed usage instructions and command-line options.

### Manual Setup

1. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Run the application:
```bash
python -m song_editor.app
```

See [USAGE_GUIDE.md](./USAGE_GUIDE.md) for detailed usage instructions and command-line options.

## Usage

### Quick Start

For detailed command-line usage and examples, see [USAGE_GUIDE.md](./USAGE_GUIDE.md).

**Basic command:**
```bash
./run_song_editor.sh --no-gui --whisper-model faster-whisper "your-song.wav"
```

### Basic Workflow

1. **Load Audio**: Open an audio file (WAV, MP3, FLAC, etc.)
2. **Transcribe**: Use Whisper to transcribe the audio
3. **Detect Chords**: Run chord detection algorithms
4. **Extract Melody**: Extract melodic lines from the audio
5. **Edit**: Use the lyrics editor and block view to refine the song
6. **Export**: Export to MIDI, JSON, or CCLI format

### Advanced Features

- **Source Separation**: Separate different instruments for individual processing
- **Batch Processing**: Process multiple files at once
- **Custom Models**: Use different Whisper models for various languages and accuracy levels
- **Real-time Preview**: Preview changes in real-time during editing

## Project Structure

```
song_editor/
├── app.py                 # Main application entry point
├── batch.py              # Batch processing functionality
├── core/                 # Core audio processing modules
│   ├── audio_player.py   # Audio playback functionality
│   ├── audio_processor.py # Audio processing utilities
│   ├── chord_detector.py # Chord detection algorithms
│   ├── melody_extractor.py # Melody extraction
│   └── transcriber.py    # Transcription engines
├── export/               # Export modules
│   ├── ccli_exporter.py  # CCLI format export
│   ├── json_exporter.py  # JSON format export
│   └── midi_exporter.py  # MIDI format export
├── models/               # Data models
│   ├── lyrics.py         # Lyrics data structures
│   ├── metadata.py       # Metadata handling
│   └── song_data.py      # Main song data model
├── processing/           # Audio processing utilities
│   ├── chords.py         # Chord processing
│   ├── separate.py       # Source separation
│   └── transcriber.py    # Transcription processing
├── services/             # External services
│   └── gemini_client.py  # Gemini AI integration
└── ui/                   # User interface components
    ├── block_view.py     # Block view interface
    ├── chord_editor.py   # Chord editing interface
    ├── enhanced_lyrics_editor.py # Advanced lyrics editor
    ├── lyrics_editor.py  # Basic lyrics editor
    ├── main_window.py    # Main application window
    └── melody_editor.py  # Melody editing interface
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests

# Run with coverage
pytest --cov=song_editor
```

## Development

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement the feature
3. Add tests in the `tests/` directory
4. Update documentation
5. Submit pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Write unit tests for new functionality

## Troubleshooting

### Common Issues

1. **Audio playback issues**: Ensure you have the correct audio drivers installed
2. **Memory errors**: Close other applications to free up RAM
3. **Slow processing**: Consider using GPU acceleration if available
4. **Model download issues**: Check your internet connection and firewall settings

### Getting Help

- Check the test files for usage examples
- Review the source code documentation
- Create an issue on the project repository

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- OpenAI for Whisper
- Chordino for chord detection
- Facebook Research for Demucs
- The open-source audio processing community
