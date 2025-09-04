#!/usr/bin/env python3
"""
Song Editor 3 - Main Application Entry Point

A professional desktop song editing and transcription application that combines
the best features of Song Editor 2 and wav_to_karoke.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication

# Handle both module execution and standalone executable execution
try:
    # Try relative imports (works when run as module)
    from .ui.main_window import MainWindow
    from .core.audio_processor import AudioProcessor
    from .core.transcriber import Transcriber
    # Import chord_detector lazily to avoid bus errors
    from .core.melody_extractor import MelodyExtractor
    from .export.midi_exporter import MidiExporter
    from .export.ccli_exporter import CCLIExporter
    from .export.json_exporter import JSONExporter
except ImportError:
    # Fallback to absolute imports (works in PyInstaller bundle)
    from song_editor.ui.main_window import MainWindow
    from song_editor.core.audio_processor import AudioProcessor
    from song_editor.core.transcriber import Transcriber
    # Import chord_detector lazily to avoid bus errors
    from song_editor.core.melody_extractor import MelodyExtractor
    from song_editor.export.midi_exporter import MidiExporter
    from song_editor.export.ccli_exporter import CCLIExporter
    from song_editor.export.json_exporter import JSONExporter

# ChordDetector will be imported lazily when needed


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_audio_file(file_path: str) -> bool:
    """Validate that the file exists and is a supported audio format"""
    audio_extensions = {
        '.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg',
        '.wma', '.opus', '.aiff', '.alac'
    }

    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return False

    ext = Path(file_path).suffix.lower()
    if ext not in audio_extensions:
        print(f"Error: File '{file_path}' is not a supported audio format.")
        print(f"Supported formats: {', '.join(audio_extensions)}")
        return False

    return True


def process_audio_file(
    input_path: str,
    output_dir: Optional[str] = None,
    whisper_model: str = "openai-whisper",
    use_chordino: bool = True,
    use_demucs: bool = True,
    save_intermediate: bool = False,
    no_gui: bool = False
) -> bool:
    """Process a single audio file with the full pipeline"""
    try:
        logging.info(f"🎵 Processing audio file: {input_path}")

        # Initialize processors
        audio_processor = AudioProcessor(
            use_demucs=use_demucs,
            save_intermediate=save_intermediate
        )

        transcriber = Transcriber(
            model=whisper_model,
            alternatives_count=5
        )

        # Import chord_detector lazily to avoid bus errors
        try:
            try:
                from .core.chord_detector import ChordDetector
            except ImportError:
                from song_editor.core.chord_detector import ChordDetector
            chord_detector = ChordDetector(
                use_chordino=False  # Disable chordino to avoid bus errors
            )
        except Exception as e:
            logging.warning(f"Chord detection not available: {e}")
            chord_detector = None

        melody_extractor = MelodyExtractor()

        # Process audio
        logging.info("🔧 Processing audio...")
        audio_data = audio_processor.process(input_path)

        # Transcribe lyrics
        logging.info("🎤 Transcribing lyrics...")
        lyrics = transcriber.transcribe(audio_data['vocals'], audio_data['sample_rate'])

        # Detect chords
        logging.info("🎸 Detecting chords...")
        chords = chord_detector.detect(audio_data['accompaniment'], audio_data['sample_rate'])

        # Extract melody
        logging.info("🎼 Extracting melody...")
        melody = melody_extractor.extract(audio_data['vocals'], audio_data['sample_rate'])

        # Prepare song data
        song_data = {
            'metadata': {
                'version': '3.0.0',
                'created_at': audio_processor.get_timestamp(),
                'source_audio': input_path,
                'processing_tool': 'Song Editor 3',
                'transcription': {
                    'engine': whisper_model,
                    'alternatives_count': 5
                },
                'audio_processing': {
                    'use_demucs': use_demucs,
                    'use_chordino': use_chordino,
                    'denoise': True,
                    'normalize': True
                }
            },
            'audio_analysis': audio_data['analysis'],
            'words': lyrics,
            'chords': chords,
            'notes': melody,
            'segments': []
        }

        # Export results
        logging.info("📤 Exporting results...")
        base_name = Path(input_path).stem
        export_dir = output_dir or Path(input_path).parent

        # Export JSON
        json_exporter = JSONExporter()
        json_path = export_dir / f"{base_name}.song_data.json"
        json_exporter.export(song_data, json_path)

        # Export MIDI
        midi_exporter = MidiExporter()
        midi_path = export_dir / f"{base_name}.karaoke.mid"
        midi_exporter.export(song_data, midi_path)

        # Export CCLI text
        ccli_exporter = CCLIExporter()
        ccli_path = export_dir / f"{base_name}_chord_lyrics_table.txt"
        ccli_exporter.export(song_data, ccli_path)

        logging.info(f"✅ Processing complete! Results saved to: {export_dir}")
        return True

    except Exception as e:
        logging.error(f"❌ Error processing audio file: {e}")
        return False


def main() -> int:
    """Main application entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Song Editor 3 - Professional Audio Transcription and Editing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  song-editor-3                                    # Open GUI without audio file
  song-editor-3 song.wav                          # Open GUI with audio file
  song-editor-3 song.wav --no-gui                 # Process without GUI
  song-editor-3 song.wav --output-dir ./output    # Specify output directory
  song-editor-3 song.wav --whisper-model faster-whisper  # Use different model
        """
    )

    parser.add_argument(
        'input_path',
        nargs='?',
        help='Audio file to process'
    )

    parser.add_argument(
        '--output-dir',
        help='Output directory for processed files'
    )

    parser.add_argument(
        '--whisper-model',
        default='openai-whisper',
        choices=['openai-whisper', 'faster-whisper', 'whisperx', 'mlx-whisper'],
        help='Whisper model to use for transcription (default: openai-whisper)'
    )

    parser.add_argument(
        '--use-chordino',
        action='store_true',
        default=True,
        help='Use Chordino for chord detection (default: True)'
    )

    parser.add_argument(
        '--no-chordino',
        dest='use_chordino',
        action='store_false',
        help='Disable Chordino chord detection'
    )

    parser.add_argument(
        '--use-demucs',
        action='store_true',
        default=True,
        help='Use Demucs for source separation (default: True)'
    )

    parser.add_argument(
        '--no-demucs',
        dest='use_demucs',
        action='store_false',
        help='Disable Demucs source separation'
    )

    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='Save intermediate processing files'
    )

    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Process without GUI (batch mode)'
    )

    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--content-type',
        default='general',
        choices=['general', 'christian', 'gospel', 'worship', 'hymn', 'clean'],
        help='Content type for transcription prompts (default: general)'
    )

    parser.add_argument(
        '--platform-info',
        action='store_true',
        help='Display platform information and exit'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Song Editor 3.0.0'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Display platform information if requested
    if args.platform_info:
        try:
            from .platform_utils import PlatformUtils
        except ImportError:
            from song_editor.platform_utils import PlatformUtils
        platform_info = PlatformUtils.get_platform_info()
        print("Platform Information:")
        for key, value in platform_info.items():
            if key != "config":
                print(f"  {key}: {value}")
        print("  Platform Config:")
        for key, value in platform_info["config"].items():
            print(f"    {key}: {value}")
        return 0

    # Best-effort: use spawn to avoid semaphore leaks from forked workers (Demucs/torch)
    try:
        import multiprocessing as _mp
        _mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    # Best-effort: clear resource tracker semaphores on startup
    try:
        from multiprocessing import resource_tracker as _rt  # type: ignore
        if hasattr(_rt, "_resource_tracker"):
            _rt._resource_tracker._cleanup()  # type: ignore[attr-defined]
    except Exception:
        pass

    # If no input path provided, launch GUI
    if not args.input_path:
        if args.no_gui:
            print("Error: Input path is required when using --no-gui")
            return 1

        # Launch GUI without audio file
        app = QApplication(sys.argv)
        app.setApplicationName("Song Editor 3")

        # Note: High DPI and touch attributes are deprecated in newer Qt versions
        # Modern Qt handles these automatically

        window = MainWindow()

        # Cleanup on quit
        def _on_quit() -> None:
            try:
                window.prepare_shutdown()
            except Exception:
                pass
        app.aboutToQuit.connect(_on_quit)

        window.show()
        return app.exec()

    # Validate input file
    if not validate_audio_file(args.input_path):
        return 1

    # Process audio file
    if args.no_gui:
        # Batch processing mode
        success = process_audio_file(
            input_path=args.input_path,
            output_dir=args.output_dir,
            whisper_model=args.whisper_model,
            use_chordino=args.use_chordino,
            use_demucs=args.use_demucs,
            save_intermediate=args.save_intermediate,
            no_gui=True
        )
        return 0 if success else 1
    else:
        # GUI mode
        app = QApplication(sys.argv)
        app.setApplicationName("Song Editor 3")

        # Note: High DPI and touch attributes are deprecated in newer Qt versions
        # Modern Qt handles these automatically

        window = MainWindow()

        # Load audio file
        window.load_audio_from_path(args.input_path)

        # Cleanup on quit
        def _on_quit() -> None:
            try:
                window.prepare_shutdown()
            except Exception:
                pass
        app.aboutToQuit.connect(_on_quit)

        window.show()
        return app.exec()


if __name__ == "__main__":
    sys.exit(main())
