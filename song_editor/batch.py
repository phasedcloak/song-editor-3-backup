#!/usr/bin/env python3
"""
Song Editor 3 - Batch Processing Module

Handles processing of multiple audio files with progress tracking,
error handling, and resource management.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json

from .app import process_audio_file, validate_audio_file, setup_logging


class BatchProcessor:
    """Handles batch processing of multiple audio files"""

    def __init__(
        self,
        input_dir: Optional[str] = None,
        file_list: Optional[str] = None,
        output_dir: Optional[str] = None,
        max_workers: int = 1,
        **kwargs
    ):
        self.input_dir = input_dir
        self.file_list = file_list
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.kwargs = kwargs

        self.results: Dict[str, Dict[str, Any]] = {}
        self.failed_files: List[str] = []
        self.successful_files: List[str] = []

        # Audio file extensions
        self.audio_extensions = {
            '.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg',
            '.wma', '.opus', '.aiff', '.alac'
        }

    def get_audio_files(self) -> List[str]:
        """Get list of audio files to process"""
        audio_files = []

        if self.file_list:
            # Read from file list
            try:
                with open(self.file_list, 'r', encoding='utf-8') as f:
                    for line in f:
                        file_path = line.strip()
                        if file_path and validate_audio_file(file_path):
                            audio_files.append(file_path)
            except Exception as e:
                logging.error(f"Error reading file list {self.file_list}: {e}")
                return []

        elif self.input_dir:
            # Scan directory for audio files
            input_path = Path(self.input_dir)
            if not input_path.exists():
                logging.error(f"Input directory does not exist: {self.input_dir}")
                return []

            if not input_path.is_dir():
                logging.error(f"Input path is not a directory: {self.input_dir}")
                return []

            for file_path in input_path.rglob('*'):
                if (file_path.is_file() and
                        file_path.suffix.lower() in self.audio_extensions):
                    audio_files.append(str(file_path))

        return audio_files

    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single audio file"""
        start_time = time.time()
        result = {
            'file_path': file_path,
            'success': False,
            'error': None,
            'processing_time': 0.0,
            'audio_duration': 0.0,
            'output_files': []
        }
        
        # Get audio duration for timing analysis
        try:
            # Use subprocess to get audio duration to avoid cffi conflicts
            audio_info = self._get_audio_duration_subprocess(file_path)
            result['audio_duration'] = audio_info
        except Exception as e:
            logging.warning(f"Failed to get audio duration for {file_path}: {e}")
            # Fallback: try to get duration from the processed JSON if it exists
            result['audio_duration'] = 0.0

        try:
            logging.info(f"🎵 Processing: {file_path}")

            # Check if output already exists (unless force overwrite)
            if not self.kwargs.get('force_overwrite', False):
                base_name = Path(file_path).stem
                output_dir_path = Path(self.output_dir) if self.output_dir else Path(file_path).parent
                song_data_file = output_dir_path / f"{base_name}.song_data.json"
                if song_data_file.exists():
                    logging.info(f"⚠️ Skipping {file_path} - output already exists")
                    result['success'] = True
                    result['skipped'] = True
                    return result

            # Process the file
            success = process_audio_file(
                input_path=file_path,
                output_dir=self.output_dir,
                no_gui=True,
                **self.kwargs
            )

            result['success'] = success
            result['processing_time'] = time.time() - start_time

            if success:
                # List output files
                base_name = Path(file_path).stem
                output_dir = Path(self.output_dir or Path(file_path).parent)
                result['output_files'] = [
                    str(output_dir / f"{base_name}.song_data.json"),
                    str(output_dir / f"{base_name}.karaoke.mid"),
                    str(output_dir / f"{base_name}_chord_lyrics_table.txt")
                ]

                if self.kwargs.get('save_intermediate', False):
                    result['output_files'].extend([
                        str(output_dir / f"{base_name}_vocals.wav"),
                        str(output_dir / f"{base_name}_accompaniment.wav")
                    ])

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            logging.error(f"❌ Error processing {file_path}: {e}")

        return result

    def process_batch(self) -> Dict[str, Any]:
        """Process all audio files in batch"""
        audio_files = self.get_audio_files()

        if not audio_files:
            logging.error("No audio files found to process")
            return {
                'success': False,
                'error': 'No audio files found',
                'total_files': 0,
                'successful': 0,
                'failed': 0,
                'results': []
            }

        logging.info(f"🎯 Found {len(audio_files)} audio files to process")

        start_time = time.time()
        results = []

        if self.max_workers == 1:
            # Sequential processing
            for i, file_path in enumerate(audio_files, 1):
                logging.info(f"📁 [{i}/{len(audio_files)}] Processing: {Path(file_path).name}")
                result = self.process_single_file(file_path)
                results.append(result)

                if result['success']:
                    self.successful_files.append(file_path)
                else:
                    self.failed_files.append(file_path)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self.process_single_file, file_path): file_path
                    for file_path in audio_files
                }

                # Process completed tasks
                for i, future in enumerate(as_completed(future_to_file), 1):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)

                        if result['success']:
                            self.successful_files.append(file_path)
                        else:
                            self.failed_files.append(file_path)

                        logging.info(f"📁 [{i}/{len(audio_files)}] Completed: {Path(file_path).name}")

                    except Exception as e:
                        logging.error(f"❌ Unexpected error processing {file_path}: {e}")
                        results.append({
                            'file_path': file_path,
                            'success': False,
                            'error': str(e),
                            'processing_time': 0.0,
                            'output_files': []
                        })
                        self.failed_files.append(file_path)

        total_time = time.time() - start_time

        # Prepare summary
        summary = {
            'success': len(self.failed_files) == 0,
            'total_files': len(audio_files),
            'successful': len(self.successful_files),
            'failed': len(self.failed_files),
            'total_processing_time': total_time,
            'average_processing_time': total_time / len(audio_files) if audio_files else 0,
            'results': results
        }

        return summary

    def save_results(self, output_path: str) -> None:
        """Save batch processing results to JSON file"""
        summary = {
            'batch_info': {
                'input_dir': self.input_dir,
                'file_list': self.file_list,
                'output_dir': self.output_dir,
                'max_workers': self.max_workers,
                'processing_options': self.kwargs
            },
            'summary': {
                'total_files': len(self.successful_files) + len(self.failed_files),
                'successful': len(self.successful_files),
                'failed': len(self.failed_files),
                'successful_files': self.successful_files,
                'failed_files': self.failed_files
            },
            'detailed_results': self.results
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 Batch results saved to: {output_path}")
        except Exception as e:
            logging.error(f"Error saving batch results: {e}")

    def _get_audio_duration_subprocess(self, file_path: str) -> float:
        """Get audio duration using subprocess to avoid cffi conflicts."""
        import subprocess
        import tempfile
        import sys
        
        # Create a temporary script for getting audio duration
        script_content = f"""
import sys
import os
import tempfile
from pathlib import Path

# Set temp directory to external drive
input_drive = Path(r"{file_path}").anchor or Path(r"{file_path}").parents[-1]
temp_dir = Path(input_drive) / "tmp" / "song_editor_temp"
temp_dir.mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(temp_dir)
tempfile.tempdir = str(temp_dir)

try:
    import librosa
    duration = librosa.get_duration(filename=r"{file_path}")
    print(f"DURATION:{{duration}}")
except Exception as e:
    print(f"ERROR:{{e}}", file=sys.stderr)
    sys.exit(1)
"""
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                script_path = f.name
            
            # Run the script with system Python
            result = subprocess.run(
                ['/usr/local/bin/python3', script_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up
            Path(script_path).unlink(missing_ok=True)
            
            if result.returncode == 0:
                # Parse the output
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('DURATION:'):
                        return float(line.split(':', 1)[1])
                raise ValueError("No duration found in subprocess output")
            else:
                raise Exception(f"Subprocess failed with code {result.returncode}: {result.stderr}")
                
        except Exception as e:
            logging.warning(f"Failed to get audio duration via subprocess: {e}")
            return 0.0


def main():
    """Main entry point for batch processing"""
    # Note: pkg_resources deprecation warning from pretty_midi has been fixed by patching the library
    
    parser = argparse.ArgumentParser(
        description="Song Editor 3 - Batch Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  song-editor-3-batch --input-dir ./audio_files --output-dir ./output
  song-editor-3-batch --file-list files.txt --output-dir ./output
  song-editor-3-batch --input-dir ./audio_files --max-workers 4 --whisper-model faster-whisper
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-dir',
        help='Directory containing audio files to process'
    )
    input_group.add_argument(
        '--file-list',
        help='Text file containing list of audio files to process (one per line)'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        help='Output directory for processed files'
    )

    # Processing options
    parser.add_argument(
        '--max-workers',
        type=int,
        default=1,
        help='Maximum number of parallel workers (default: 1)'
    )

    parser.add_argument(
        '--whisper-model',
        default='openai-whisper',
        choices=['openai-whisper', 'faster-whisper', 'whisperx', 'mlx-whisper'],
        help='Whisper model to use for transcription'
    )

    parser.add_argument(
        '--use-chordino',
        action='store_true',
        default=True,
        help='Use Chordino for chord detection'
    )

    parser.add_argument(
        '--no-chordino',
        dest='use_chordino',
        action='store_false',
        help='Disable Chordino chord detection'
    )
    
    # Separation engine options
    parser.add_argument(
        '--separation-engine',
        default='demucs',
        choices=['demucs', 'audio-separator'],
        help='Audio separation engine to use (default: demucs)'
    )
    
    parser.add_argument(
        '--audio-separator-model',
        default='UVR_MDXNET_KARA_2',
        help='Audio-separator model to use (default: UVR_MDXNET_KARA_2)'
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
        '--use-demucs',
        action='store_true',
        default=True,
        help='Use Demucs for source separation'
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
        '--force-overwrite',
        action='store_true',
        help='Force overwrite existing output files'
    )

    parser.add_argument(
        '--results-file',
        help='Save batch processing results to JSON file'
    )

    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Create batch processor
    processor = BatchProcessor(
        input_dir=args.input_dir,
        file_list=args.file_list,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        whisper_model=args.whisper_model,
        use_chordino=args.use_chordino,
        use_demucs=args.use_demucs,
        save_intermediate=args.save_intermediate,
        force_overwrite=args.force_overwrite,
        # New separation engine parameters
        separation_engine=args.separation_engine.replace('-', '_'),
        audio_separator_model=args.audio_separator_model,
        use_cuda=args.use_cuda,
        use_coreml=args.use_coreml
    )

    # Process batch
    logging.info("🚀 Starting batch processing...")
    summary = processor.process_batch()

    # Print summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total files: {summary['total_files']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total time: {summary['total_processing_time']:.1f} seconds")
    print(f"Average time per file: {summary['average_processing_time']:.1f} seconds")

    if summary['failed'] > 0:
        print("\nFailed files:")
        for file_path in processor.failed_files:
            print(f"  - {file_path}")

    # Save results if requested
    if args.results_file:
        processor.save_results(args.results_file)

    # Generate timing table if we have multiple files
    if summary['total_files'] > 1:
        try:
            # Add timing table generation logic
            from pathlib import Path
            import sys as _sys
            
            # Add the project root to Python path to import timing_table_builder
            project_root = Path(__file__).resolve().parent.parent
            _sys.path.insert(0, str(project_root))
            
            from timing_table_builder import TimingTableBuilder
            
            builder = TimingTableBuilder()
            builder.add_batch_results(summary)
            
            print("\n" + "="*80)
            timing_table = builder.generate_table()
            print(timing_table)
            
            # Save timing table to file if results file was specified
            if args.results_file:
                timing_file = str(Path(args.results_file).with_suffix('.timing.txt'))
                with open(timing_file, 'w') as f:
                    f.write(timing_table)
                print(f"\n📊 Timing analysis saved to: {timing_file}")
                
        except Exception as e:
            logging.warning(f"Could not generate timing table: {e}")

    # Exit with appropriate code
    sys.exit(0 if summary['success'] else 1)


if __name__ == "__main__":
    main()
