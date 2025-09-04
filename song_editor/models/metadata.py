#!/usr/bin/env python3
"""
Metadata Models

Defines metadata structures for Song Editor 3.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class TranscriptionInfo:
    """Information about the transcription process."""
    model: str
    model_size: Optional[str] = None
    language: Optional[str] = None
    confidence_threshold: Optional[float] = None
    word_timestamps: bool = True
    alternatives: bool = False
    processing_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            'model': self.model,
            'word_timestamps': self.word_timestamps,
            'alternatives': self.alternatives
        }

        if self.model_size:
            result['model_size'] = self.model_size

        if self.language:
            result['language'] = self.language

        if self.confidence_threshold is not None:
            result['confidence_threshold'] = self.confidence_threshold

        if self.processing_time is not None:
            result['processing_time'] = self.processing_time

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptionInfo':
        """Create from dictionary representation."""
        return cls(
            model=data.get('model', ''),
            model_size=data.get('model_size'),
            language=data.get('language'),
            confidence_threshold=data.get('confidence_threshold'),
            word_timestamps=data.get('word_timestamps', True),
            alternatives=data.get('alternatives', False),
            processing_time=data.get('processing_time')
        )


@dataclass
class AudioProcessingInfo:
    """Information about audio processing."""
    denoising: bool = False
    normalization: bool = False
    source_separation: bool = False
    separation_model: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    processing_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            'denoising': self.denoising,
            'normalization': self.normalization,
            'source_separation': self.source_separation
        }

        if self.separation_model:
            result['separation_model'] = self.separation_model

        if self.sample_rate:
            result['sample_rate'] = self.sample_rate

        if self.channels:
            result['channels'] = self.channels

        if self.processing_time is not None:
            result['processing_time'] = self.processing_time

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudioProcessingInfo':
        """Create from dictionary representation."""
        return cls(
            denoising=data.get('denoising', False),
            normalization=data.get('normalization', False),
            source_separation=data.get('source_separation', False),
            separation_model=data.get('separation_model'),
            sample_rate=data.get('sample_rate'),
            channels=data.get('channels'),
            processing_time=data.get('processing_time')
        )


@dataclass
class Metadata:
    """Main metadata container."""
    version: str = "3.0.0"
    created_at: Optional[str] = None
    source_audio: Optional[str] = None
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    year: Optional[int] = None
    genre: Optional[str] = None
    duration: Optional[float] = None
    transcription: Optional[TranscriptionInfo] = None
    audio_processing: Optional[AudioProcessingInfo] = None
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set default values after initialization."""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            'version': self.version,
            'created_at': self.created_at
        }

        if self.source_audio:
            result['source_audio'] = self.source_audio

        if self.title:
            result['title'] = self.title

        if self.artist:
            result['artist'] = self.artist

        if self.album:
            result['album'] = self.album

        if self.year:
            result['year'] = self.year

        if self.genre:
            result['genre'] = self.genre

        if self.duration is not None:
            result['duration'] = self.duration

        if self.transcription:
            result['transcription'] = self.transcription.to_dict()

        if self.audio_processing:
            result['audio_processing'] = self.audio_processing.to_dict()

        if self.custom_fields:
            result['custom_fields'] = self.custom_fields

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Metadata':
        """Create from dictionary representation."""
        transcription_data = data.get('transcription')
        audio_processing_data = data.get('audio_processing')

        return cls(
            version=data.get('version', '3.0.0'),
            created_at=data.get('created_at'),
            source_audio=data.get('source_audio'),
            title=data.get('title'),
            artist=data.get('artist'),
            album=data.get('album'),
            year=data.get('year'),
            genre=data.get('genre'),
            duration=data.get('duration'),
            transcription=TranscriptionInfo.from_dict(transcription_data) if transcription_data else None,
            audio_processing=AudioProcessingInfo.from_dict(audio_processing_data) if audio_processing_data else None,
            custom_fields=data.get('custom_fields', {})
        )

    def set_title(self, title: str) -> None:
        """Set the song title."""
        self.title = title

    def set_artist(self, artist: str) -> None:
        """Set the artist name."""
        self.artist = artist

    def set_album(self, album: str) -> None:
        """Set the album name."""
        self.album = album

    def set_year(self, year: int) -> None:
        """Set the release year."""
        self.year = year

    def set_genre(self, genre: str) -> None:
        """Set the genre."""
        self.genre = genre

    def set_duration(self, duration: float) -> None:
        """Set the song duration."""
        self.duration = duration

    def set_transcription_info(self, transcription: TranscriptionInfo) -> None:
        """Set transcription information."""
        self.transcription = transcription

    def set_audio_processing_info(self, audio_processing: AudioProcessingInfo) -> None:
        """Set audio processing information."""
        self.audio_processing = audio_processing

    def add_custom_field(self, key: str, value: Any) -> None:
        """Add a custom field."""
        self.custom_fields[key] = value

    def get_custom_field(self, key: str, default: Any = None) -> Any:
        """Get a custom field value."""
        return self.custom_fields.get(key, default)

    def remove_custom_field(self, key: str) -> bool:
        """Remove a custom field."""
        if key in self.custom_fields:
            del self.custom_fields[key]
            return True
        return False

    def get_basic_info(self) -> Dict[str, Any]:
        """Get basic song information."""
        info = {}

        if self.title:
            info['title'] = self.title

        if self.artist:
            info['artist'] = self.artist

        if self.album:
            info['album'] = self.album

        if self.year:
            info['year'] = self.year

        if self.genre:
            info['genre'] = self.genre

        if self.duration is not None:
            info['duration'] = self.duration

        return info

    def get_processing_info(self) -> Dict[str, Any]:
        """Get processing information."""
        info = {}

        if self.transcription:
            info['transcription'] = self.transcription.to_dict()

        if self.audio_processing:
            info['audio_processing'] = self.audio_processing.to_dict()

        return info

    def validate(self) -> List[str]:
        """Validate the metadata and return any errors."""
        errors = []

        # Check required fields
        if not self.version:
            errors.append("Missing version")

        if not self.created_at:
            errors.append("Missing created_at")

        # Check version format
        if not self.version.startswith('3.'):
            errors.append("Invalid version format (should start with '3.')")

        # Check year if present
        if self.year is not None:
            if not isinstance(self.year, int) or self.year < 1900 or self.year > 2100:
                errors.append("Invalid year")

        # Check duration if present
        if self.duration is not None:
            if not isinstance(self.duration, (int, float)) or self.duration <= 0:
                errors.append("Invalid duration")

        return errors

    def to_json(self, file_path: str, pretty: bool = True) -> bool:
        """Save metadata to JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)
                else:
                    json.dump(self.to_dict(), f, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            print(f"Error saving metadata: {e}")
            return False

    @classmethod
    def from_json(cls, file_path: str) -> Optional['Metadata']:
        """Load metadata from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return None
