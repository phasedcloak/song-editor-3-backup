#!/usr/bin/env python3
"""
Processing Service Layer for Song Editor 3

Provides a unified interface for audio processing operations,
decoupling UI components from core processing modules.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future

from ..core.audio_processor import AudioProcessor
from ..core.transcriber import Transcriber
from ..core.chord_detector import ChordDetector
from ..core.melody_extractor import MelodyExtractor
from ..config import get_transcription_config


class ProcessingService:
    """Service layer for audio processing operations."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="Processing")
        self.active_tasks: Dict[str, Future] = {}
        self.task_callbacks: Dict[str, List[Callable]] = {}

    def process_audio_file(
        self,
        file_path: str,
        task_id: str,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        completion_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process an audio file asynchronously."""
        if task_id in self.active_tasks:
            raise ValueError(f"Task {task_id} is already running")

        # Default configuration
        if config is None:
            config = {
                'use_demucs': True,
                'save_intermediate': False,
                'whisper_model': get_transcription_config().default_model,
                'model_size': get_transcription_config().default_model_size,
                'use_chordino': True,
                'simplify_chords': False,
                'preserve_chord_richness': True,
                'use_basic_pitch': True,
                'min_note_duration': 0.1,
                'language': get_transcription_config().default_language
            }

        # Submit task
        future = self.executor.submit(
            self._process_audio_file_sync,
            file_path,
            config,
            progress_callback
        )

        self.active_tasks[task_id] = future

        # Set up completion callback
        if completion_callback:
            if task_id not in self.task_callbacks:
                self.task_callbacks[task_id] = []
            self.task_callbacks[task_id].append(completion_callback)

            future.add_done_callback(
                lambda f: self._handle_task_completion(task_id, f)
            )

        return task_id

    def _process_audio_file_sync(
        self,
        file_path: str,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[str, int], None]]
    ) -> Dict[str, Any]:
        """Synchronous audio processing."""
        try:
            # Update progress
            if progress_callback:
                progress_callback("Initializing processors...", 5)

            # Initialize processors
            audio_processor = AudioProcessor(
                use_demucs=config.get('use_demucs', True),
                save_intermediate=config.get('save_intermediate', False)
            )

            transcriber = Transcriber(
                model=config.get('whisper_model', 'openai-whisper'),
                model_size=config.get('model_size', 'large-v2'),
                language=config.get('language', None)
            )

            chord_detector = ChordDetector(
                use_chordino=config.get('use_chordino', True),
                chord_simplification=config.get('simplify_chords', False),
                preserve_chord_richness=config.get('preserve_chord_richness', True)
            )

            melody_extractor = MelodyExtractor(
                use_basic_pitch=config.get('use_basic_pitch', True),
                min_note_duration=config.get('min_note_duration', 0.1)
            )

            # Process audio
            if progress_callback:
                progress_callback("Loading and preprocessing audio...", 15)

            audio_data = audio_processor.process(file_path)

            # Calculate audio duration and estimated processing time
            audio_duration = len(audio_data['audio']) / audio_data['sample_rate']

            if progress_callback:
                if audio_duration > 300:  # 5 minutes
                    progress_callback(
                        f"Long audio file ({audio_duration:.0f}s) - processing may take several minutes", 20
                    )

            # Transcribe lyrics
            if progress_callback:
                progress_callback("Transcribing lyrics...", 30)

            start_time = time.time()
            words = transcriber.transcribe(audio_data['vocals'], audio_data['sample_rate'])
            transcription_time = time.time() - start_time

            if progress_callback:
                progress_callback(f"Transcription completed ({transcription_time:.1f}s)", 50)

            # Detect chords
            if progress_callback:
                progress_callback("Detecting chords...", 60)

            chords = chord_detector.detect(audio_data['accompaniment'], audio_data['sample_rate'])

            if progress_callback:
                progress_callback("Chord detection completed", 75)

            # Extract melody
            if progress_callback:
                progress_callback("Extracting melody...", 80)

            melody = melody_extractor.extract(audio_data['vocals'], audio_data['sample_rate'])

            if progress_callback:
                progress_callback("Melody extraction completed", 95)

            # Prepare result
            result = {
                'success': True,
                'audio_data': audio_data,
                'words': words,
                'chords': chords,
                'melody': melody,
                'processing_stats': {
                    'audio_duration': audio_duration,
                    'transcription_time': transcription_time,
                    'total_processing_time': time.time() - time.time(),  # Will be set by caller
                    'word_count': len(words),
                    'chord_count': len(chords),
                    'note_count': len(melody)
                }
            }

            if progress_callback:
                progress_callback("Processing completed successfully", 100)

            return result

        except Exception as e:
            logging.error(f"Processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }

    def _handle_task_completion(self, task_id: str, future: Future) -> None:
        """Handle task completion."""
        try:
            result = future.result()
            result['processing_stats']['total_processing_time'] = time.time()  # Approximate

            # Call callbacks
            if task_id in self.task_callbacks:
                for callback in self.task_callbacks[task_id]:
                    try:
                        callback(result)
                    except Exception as e:
                        logging.error(f"Error in completion callback: {e}")

                # Clean up callbacks
                del self.task_callbacks[task_id]

        except Exception as e:
            logging.error(f"Task {task_id} failed: {e}")

            # Call callbacks with error
            if task_id in self.task_callbacks:
                error_result = {
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                for callback in self.task_callbacks[task_id]:
                    try:
                        callback(error_result)
                    except Exception as callback_error:
                        logging.error(f"Error in error callback: {callback_error}")

                del self.task_callbacks[task_id]

        finally:
            # Clean up task
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self.active_tasks:
            future = self.active_tasks[task_id]
            if not future.done():
                future.cancel()
                return True
        return False

    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get the status of a task."""
        if task_id in self.active_tasks:
            future = self.active_tasks[task_id]
            if future.done():
                return "completed"
            elif future.cancelled():
                return "cancelled"
            else:
                return "running"
        return None

    def get_active_tasks(self) -> List[str]:
        """Get list of active task IDs."""
        return list(self.active_tasks.keys())

    def shutdown(self) -> None:
        """Shutdown the processing service."""
        self.executor.shutdown(wait=True)
        self.active_tasks.clear()
        self.task_callbacks.clear()


# Global service instance
_processing_service = None


def get_processing_service() -> ProcessingService:
    """Get the global processing service."""
    global _processing_service
    if _processing_service is None:
        _processing_service = ProcessingService()
    return _processing_service


def init_processing_service() -> ProcessingService:
    """Initialize the global processing service."""
    global _processing_service
    _processing_service = ProcessingService()
    return _processing_service
