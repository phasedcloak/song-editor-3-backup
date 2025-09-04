"""
Song Editor 3 - Core Processing Modules

Core audio processing, transcription, chord detection, and melody extraction modules.
"""

from .audio_processor import AudioProcessor
from .transcriber import Transcriber
from .chord_detector import ChordDetector
from .melody_extractor import MelodyExtractor

__all__ = [
    "AudioProcessor",
    "Transcriber",
    "ChordDetector",
    "MelodyExtractor"
]
