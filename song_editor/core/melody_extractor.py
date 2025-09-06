#!/usr/bin/env python3
"""
Melody Extractor
Extracts melody from an audio file.
"""

import sys
import os
import logging
import subprocess
import json
import tempfile
from typing import List, Dict, Any


class MelodyExtractor:
    """Extracts melody from an audio file."""

    def __init__(
        self,
        method: str = 'librosa_fallback',
        min_note_duration: float = 0.1,
        min_pitch: int = 36,  # C2
        max_pitch: int = 84,  # C6
        min_confidence: float = 0.5
    ):
        """
        Initialize MelodyExtractor.

        Args:
            method (str): Method to use ('basic_pitch', 'crepe', 'librosa_fallback').
            min_note_duration (float): Minimum duration of a note to be considered valid.
            min_pitch (int): Minimum MIDI pitch to consider.
            max_pitch (int): Maximum MIDI pitch to consider.
            min_confidence (float): Minimum confidence for a note to be considered valid.
        """
        self.method = method
        self.min_note_duration = min_note_duration
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.min_confidence = min_confidence

    def extract(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Extract melody from an audio file by calling an external worker script.
        This isolates the melody extraction process to prevent library conflicts.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            List[Dict[str, Any]]: List of note events.
        """
        try:
            logging.info(f"🎤 Starting melody extraction with method: {self.method}")

            melody_params = {
                'audio_path': audio_path,
                'method': self.method,
                'min_note_duration': self.min_note_duration,
                'min_pitch': self.min_pitch,
                'max_pitch': self.max_pitch,
                'min_confidence': self.min_confidence
            }

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(melody_params, temp_file)
                params_file = temp_file.name

            # This path logic works for both running from source and from the macOS app bundle
            script_dir = os.path.join(os.path.dirname(__file__), '..', '..')
            script_path = os.path.join(script_dir, 'melody_worker.py')

            if not os.path.exists(script_path):
                # Fallback for bundle where CWD is .../Contents
                bundle_script_path = os.path.join(os.getcwd(), 'Resources', 'melody_worker.py')
                if os.path.exists(bundle_script_path):
                    script_path = bundle_script_path
                else:
                    logging.error(f"Melody worker script not found at {script_path} or {bundle_script_path}")
                    return []
            
            python_executable = sys.executable

            command = [python_executable, script_path, params_file]
            
            logging.info(f"Running melody worker command: {' '.join(command)}")

            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )

            if process.stderr:
                for line in process.stderr.strip().split('\n'):
                    logging.info(f"[MelodyWorker] {line}")
            
            notes = json.loads(process.stdout)
            
            logging.info(f"Melody extraction succeeded, found {len(notes)} notes.")
            return notes

        except subprocess.CalledProcessError as e:
            logging.error("Error in melody extraction worker process.")
            logging.error(f"Return code: {e.returncode}")
            logging.error(f"Stdout: {e.stdout}")
            logging.error(f"Stderr: {e.stderr}")
            return []
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from melody worker: {e}")
            logging.error(f"Received output: {process.stdout}")
            return []
        except Exception as e:
            logging.error(f"An unexpected error occurred during melody extraction: {e}")
            return []
        finally:
            if 'params_file' in locals() and os.path.exists(params_file):
                os.unlink(params_file)

    def get_extractor_info(self) -> Dict[str, Any]:
        """Get information about the melody extractor configuration."""
        return {
            'method': self.method,
            'min_note_duration': self.min_note_duration,
            'min_pitch': self.min_pitch,
            'max_pitch': self.max_pitch,
            'min_confidence': self.min_confidence
        }
