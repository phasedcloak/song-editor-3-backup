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

# Capture original working directory from environment variable
ORIGINAL_CWD = os.environ.get('SONG_EDITOR_ORIGINAL_CWD', os.getcwd())

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


def print_separation_models_info():
    """Print information about available separation models."""
    print("🎵 Song Editor 3 - Available Separation Models")
    print("=" * 60)

    # Print Demucs models
    print("\n🔸 Demucs Models (Built-in):")
    print("   Engine: demucs")
    print("   Models:")
    demucs_models = [
        "htdemucs        - Default 4-stem model (drums, bass, vocals, guitar/piano/other)",
        "htdemucs_ft     - Fine-tuned version with better quality",
        "htdemucs_6s     - 6-stem model (adds guitar and piano separation)",
        "hdemucs_mmi     - MMI version with improved quality",
        "bag_of_models   - Ensemble of multiple models for best quality"
    ]
    for model in demucs_models:
        print(f"     • {model}")

    # Print Audio-Separator models if available
    print("\n🔸 Audio-Separator Models (Enhanced Performance):")
    try:
        from song_editor.core.audio_separator_processor import AudioSeparatorProcessor

        if AudioSeparatorProcessor.is_available():
            print("   Engine: audio-separator")
            print("   Status: ✅ Available")
            print("   GPU Support: CoreML (Apple Silicon), CUDA (NVIDIA)")

            separator = AudioSeparatorProcessor()
            models = separator.get_available_models()

            print(f"   Available Models ({len(models)}):")
            for i, model in enumerate(models, 1):
                status = "💾" if model.get('cached', False) else "⬇️"
                print(f"     {i:2d}. {status} {model['name']} ({model['estimated_size_mb']}MB)")
                print(f"         {model['description']}")
                print(f"         Quality: {model['quality']} | Speed: {model['speed']}")
                print(f"         Use: {model['recommended_use']}")
                print()

        else:
            print("   Status: ❌ Not available (install with: pip install 'audio-separator[gpu]')")
            print("   Models: Use --separation-engine demucs instead")

    except Exception as e:
        print(f"   Error loading audio-separator: {e}")

    # Print usage examples
    print("\n📋 Usage Examples:")
    print("   # Use Demucs (default)")
    print("   song-editor-3 song.wav --separation-engine demucs --demucs-model htdemucs_ft")
    print()
    print("   # Use Audio-Separator (better performance)")
    print("   song-editor-3 song.wav --separation-engine audio-separator")
    print("   song-editor-3 song.wav --separation-engine audio-separator --audio-separator-model UVR_MDXNET_21_OVERLAP_9")
    print()
    print("   # GPU acceleration options")
    print("   song-editor-3 song.wav --separation-engine audio-separator --use-coreml  # Apple Silicon")
    print("   song-editor-3 song.wav --separation-engine audio-separator --use-cuda    # NVIDIA GPUs")


def find_audio_files(directory: str) -> list:
    """Find all audio files in a directory."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    audio_files = []

    for file_path in Path(directory).rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            audio_files.append(str(file_path))

    # Sort files for consistent processing order
    audio_files.sort()
    return audio_files

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


def validate_audio_file(file_path: str, original_cwd: str = None) -> tuple[bool, str]:
    """Validate that the file exists and is a supported audio format.

    Returns:
        tuple: (is_valid, resolved_path)
    """
    audio_extensions = {
        '.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg',
        '.wma', '.opus', '.aiff', '.alac'
    }

    # Resolve relative paths
    path_obj = Path(file_path)
    if not path_obj.is_absolute():
        # Use original working directory if provided, otherwise current
        base_dir = Path(original_cwd) if original_cwd else Path.cwd()
        resolved_path = base_dir / path_obj
        if resolved_path.exists():
            file_path = str(resolved_path)

    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return False, file_path

    ext = Path(file_path).suffix.lower()
    if ext not in audio_extensions:
        print(f"Error: File '{file_path}' is not a supported audio format.")
        print(f"Supported formats: {', '.join(audio_extensions)}")
        return False, file_path

    return True, file_path


def process_audio_file(
    input_path: str,
    output_dir: Optional[str] = None,
    whisper_model: str = "openai-whisper",
    use_chordino: bool = True,
    separation_engine: str = 'demucs',
    audio_separator_model: str = 'UVR_MDXNET_KARA_2',
    use_cuda: bool = False,
    use_coreml: bool = True,
    use_demucs: bool = True,
    save_intermediate: bool = False,
    demucs_model: str = 'htdemucs',
    no_gui: bool = False
) -> bool:
    """Process a single audio file with the full pipeline"""
    try:
        logging.info(f"🎵 Processing audio file: {input_path}")

        # Initialize processors
        audio_processor = AudioProcessor(
            use_demucs=use_demucs,
            save_intermediate=save_intermediate,
            demucs_model=demucs_model,
            # Audio-separator parameters
            separation_engine=separation_engine,
            audio_separator_model=audio_separator_model,
            use_cuda=use_cuda,
            use_coreml=use_coreml
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

        melody_extractor = MelodyExtractor(method='basic-pitch')

        # Process audio
        logging.info("🔧 Processing audio...")
        audio_data = audio_processor.process(input_path)

        # Transcribe lyrics
        logging.info("🎤 Transcribing lyrics...")
        vocals_audio = audio_data.get('vocals', audio_data['audio'])
        lyrics = transcriber.transcribe(vocals_audio, audio_data['sample_rate'])

        # Detect chords
        logging.info("🎸 Detecting chords...")
        instrumental_audio = audio_data.get('other', audio_data['audio'])
        chords = chord_detector.detect(instrumental_audio, audio_data['sample_rate'])

        # Extract melody
        logging.info("🎼 Extracting melody...")
        # Melody extractor worker script expects a file path
        temp_vocal_path = audio_processor._save_audio_temp(vocals_audio, audio_data['sample_rate'])
        melody = melody_extractor.extract(temp_vocal_path)
        # Clean up temporary file
        if os.path.exists(temp_vocal_path):
            os.unlink(temp_vocal_path)

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
    # Suppress pkg_resources deprecation warning (scheduled for removal 2025-11-30)
    import warnings
    warnings.filterwarnings('ignore', message='pkg_resources is deprecated', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*pkg_resources.*', category=UserWarning)

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

  # Audio separation options:
  song-editor-3 song.wav --list-separation-models    # List available separation models
  song-editor-3 song.wav --separation-engine audio-separator  # Use audio-separator
  song-editor-3 song.wav --separation-engine audio-separator --audio-separator-model UVR_MDXNET_21_OVERLAP_9  # Use specific model
  song-editor-3 song.wav --separation-engine demucs --demucs-model htdemucs_ft  # Use Demucs with fine-tuned model

  # Batch processing:
  song-editor-3 --input-dir ./audio_files --no-gui  # Process all audio files in directory
  song-editor-3 --input-dir ./songs --output-dir ./processed  # Batch process with custom output
        """
    )
    # Hidden/internal: worker dispatch for packaged subprocesses
    parser.add_argument(
        '--worker',
        choices=['melody', 'transcription', 'chord'],
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--worker-params',
        help=argparse.SUPPRESS
    )

    parser.add_argument(
        'input_path',
        nargs='?',
        help='Audio file to process (or use --input-dir for batch processing)'
    )

    parser.add_argument(
        '--input-dir',
        help='Directory containing audio files to process (batch mode)'
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
        '--demucs-model',
        default='htdemucs',
        help="Demucs pretrained model to use (e.g., 'htdemucs', 'htdemucs_ft', 'bag_of_models')"
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

    # New separation engine options
    parser.add_argument(
        '--separation-engine',
        default='demucs',
        choices=['demucs', 'audio-separator'],
        help='Source separation engine to use (default: demucs). Audio-separator provides better performance and more models.'
    )

    parser.add_argument(
        '--audio-separator-model',
        default='UVR_MDXNET_KARA_2',
        help='Audio-separator model to use (default: UVR_MDXNET_KARA_2). Use --list-separation-models to see available options.'
    )

    parser.add_argument(
        '--list-separation-models',
        action='store_true',
        help='List all available separation models and exit'
    )

    parser.add_argument(
        '--use-cuda',
        action='store_true',
        help='Enable CUDA acceleration for audio-separator (NVIDIA GPUs only)'
    )

    parser.add_argument(
        '--use-coreml',
        action='store_true',
        default=True,
        help='Enable CoreML acceleration for audio-separator (Apple Silicon, default: True)'
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

    # Ignore unknown args injected by PyInstaller/multiprocessing (-B, -S, -I, -c, etc.)
    args, _unknown = parser.parse_known_args()

    # Use the original working directory captured at import time
    original_cwd = ORIGINAL_CWD

    # Setup logging
    setup_logging(args.log_level)
    # Handle internal worker dispatch for packaged subprocesses
    if args.worker and args.worker_params:
        try:
            import runpy as _runpy
            import importlib
            # Resolve worker path
            worker_map = {
                'melody': 'melody_worker.py',
                'transcription': 'transcription_worker.py',
                'chord': 'chord_worker.py',
            }
            worker_file = worker_map[args.worker]
            base_dir = Path(__file__).resolve().parent.parent
            worker_path = str(base_dir / worker_file)
            if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                worker_path = os.path.join(sys._MEIPASS, worker_file)  # type: ignore[attr-defined]

            # Emulate running the worker script as __main__ with params path argument
            old_argv = sys.argv[:]
            sys.argv = [worker_path, args.worker_params]
            try:
                _runpy.run_path(worker_path, run_name='__main__')
            finally:
                sys.argv = old_argv
            return 0
        except SystemExit as e:
            return int(getattr(e, 'code', 0) or 0)
        except Exception as e:
            print(f"WORKER_DISPATCH_ERROR: {e}", file=sys.stderr)
            return 1


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

    # Handle separation model listing
    if args.list_separation_models:
        print_separation_models_info()
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

    # Handle directory processing
    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"Error: Directory '{args.input_dir}' does not exist")
            return 1

        if not args.no_gui:
            print("Error: Directory processing requires --no-gui flag")
            return 1

        # Find all audio files in directory
        audio_files = find_audio_files(args.input_dir)
        if not audio_files:
            print(f"Error: No audio files found in directory '{args.input_dir}'")
            print("Supported formats: .wav, .mp3, .flac, .aac, .ogg, .m4a, .wma")
            return 1

        print(f"📁 Found {len(audio_files)} audio files in directory:")
        for i, audio_file in enumerate(audio_files, 1):
            print(f"  {i}. {os.path.basename(audio_file)}")

        # Process each file
        successful = 0
        failed = 0

        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n🎵 Processing file {i}/{len(audio_files)}: {os.path.basename(audio_file)}")
            print("=" * 60)

            try:
                success = process_audio_file(
                    input_path=audio_file,
                    output_dir=args.output_dir,
                    whisper_model=args.whisper_model,
                    use_chordino=args.use_chordino,
                    # New separation engine parameters
                    separation_engine=args.separation_engine.replace('-', '_'),
                    audio_separator_model=args.audio_separator_model,
                    use_cuda=args.use_cuda,
                    use_coreml=args.use_coreml,
                    use_demucs=args.use_demucs,
                    save_intermediate=args.save_intermediate,
                    demucs_model=args.demucs_model
                )
                if success:
                    successful += 1
                    print(f"✅ Successfully processed: {os.path.basename(audio_file)}")
                else:
                    failed += 1
                    print(f"❌ Failed to process: {os.path.basename(audio_file)}")

            except Exception as e:
                failed += 1
                print(f"❌ Error processing {os.path.basename(audio_file)}: {e}")

        print(f"\n📊 Batch processing complete!")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📁 Total files: {len(audio_files)}")

        return 0 if failed == 0 else 1

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
    is_valid, resolved_path = validate_audio_file(args.input_path, original_cwd)
    if not is_valid:
        return 1

    # Process audio file
    if args.no_gui:
        # Batch processing mode
        success = process_audio_file(
            input_path=resolved_path,
            output_dir=args.output_dir,
            whisper_model=args.whisper_model,
            use_chordino=args.use_chordino,
            # New separation engine parameters
            separation_engine=args.separation_engine.replace('-', '_'),
            audio_separator_model=args.audio_separator_model,
            use_cuda=args.use_cuda,
            use_coreml=args.use_coreml,
            use_demucs=args.use_demucs,
            save_intermediate=args.save_intermediate,
            demucs_model=args.demucs_model,
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
        window.load_audio_from_path(resolved_path)

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
    # Ensure safe multiprocessing on frozen apps
    try:
        import multiprocessing as _mp
        _mp.freeze_support()
    except Exception:
        pass
    sys.exit(main())
