#!/usr/bin/env python3
"""
Song Data Models

Defines the core data structures for Song Editor 3.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import re


def strip_punctuation(text: str) -> str:
    """
    Strip common punctuation from transcribed text while preserving word boundaries.
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned text with punctuation removed
    """
    if not text:
        return text
    
    # Common punctuation marks to remove
    punctuation_pattern = r'[.,!?;:"\'()\[\]{}<>/\\|`~@#$%^&*+=_\-]'
    
    # Remove punctuation but preserve spaces
    cleaned = re.sub(punctuation_pattern, '', text)
    
    # Clean up multiple spaces that might result from punctuation removal
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Strip leading/trailing whitespace
    return cleaned.strip()


@dataclass
class Word:
    """Represents a single word with timing and confidence."""
    text: str
    start: float
    end: float
    confidence: float
    alternatives: Optional[List[str]] = None
    chord: Optional[str] = None
    line_break: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            'text': self.text,
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence
        }

        if self.alternatives:
            result['alternatives'] = self.alternatives

        if self.chord:
            result['chord'] = self.chord

        if self.line_break:
            result['line_break'] = True

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Word':
        """Create from dictionary representation."""
        # Strip punctuation from loaded text
        raw_text = data.get('text', '')
        cleaned_text = strip_punctuation(raw_text)
        
        return cls(
            text=cleaned_text,
            start=data.get('start', 0.0),
            end=data.get('end', 0.0),
            confidence=data.get('confidence', 0.0),
            alternatives=data.get('alternatives'),
            chord=data.get('chord'),
            line_break=bool(data.get('line_break', False))
        )


@dataclass
class Chord:
    """Represents a chord with timing and properties."""
    symbol: str
    root: str
    quality: str
    start: float
    end: float
    bass: Optional[str] = None
    duration: Optional[float] = None
    confidence: Optional[float] = None
    detection_method: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            'symbol': self.symbol,
            'root': self.root,
            'quality': self.quality,
            'start': self.start,
            'end': self.end
        }

        if self.bass:
            result['bass'] = self.bass

        if self.duration is not None:
            result['duration'] = self.duration

        if self.confidence is not None:
            result['confidence'] = self.confidence

        if self.detection_method:
            result['detection_method'] = self.detection_method

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chord':
        """Create from dictionary representation."""
        return cls(
            symbol=data.get('symbol', ''),
            root=data.get('root', ''),
            quality=data.get('quality', ''),
            start=data.get('start', 0.0),
            end=data.get('end', 0.0),
            bass=data.get('bass'),
            duration=data.get('duration'),
            confidence=data.get('confidence'),
            detection_method=data.get('detection_method')
        )


@dataclass
class Note:
    """Represents a musical note with timing and properties."""
    pitch_midi: int
    start: float
    end: float
    pitch_name: Optional[str] = None
    duration: Optional[float] = None
    velocity: Optional[int] = None
    confidence: Optional[float] = None
    detection_method: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            'pitch_midi': self.pitch_midi,
            'start': self.start,
            'end': self.end
        }

        if self.pitch_name:
            result['pitch_name'] = self.pitch_name

        if self.duration is not None:
            result['duration'] = self.duration

        if self.velocity is not None:
            result['velocity'] = self.velocity

        if self.confidence is not None:
            result['confidence'] = self.confidence

        if self.detection_method:
            result['detection_method'] = self.detection_method

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Note':
        """Create from dictionary representation."""
        return cls(
            pitch_midi=data.get('pitch_midi', 60),
            start=data.get('start', 0.0),
            end=data.get('end', 0.0),
            pitch_name=data.get('pitch_name'),
            duration=data.get('duration'),
            velocity=data.get('velocity'),
            confidence=data.get('confidence'),
            detection_method=data.get('detection_method')
        )


@dataclass
class SongData:
    """Main song data container."""
    metadata: Dict[str, Any] = field(default_factory=dict)
    words: List[Word] = field(default_factory=list)
    chords: List[Chord] = field(default_factory=list)
    notes: List[Note] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            'metadata': self.metadata,
            'words': [word.to_dict() for word in self.words],
            'chords': [chord.to_dict() for chord in self.chords],
            'notes': [note.to_dict() for note in self.notes]
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SongData':
        """Create from dictionary representation."""
        return cls(
            metadata=data.get('metadata', {}),
            words=[Word.from_dict(w) for w in data.get('words', [])],
            chords=[Chord.from_dict(c) for c in data.get('chords', [])],
            notes=[Note.from_dict(n) for n in data.get('notes', [])]
        )

    def get_duration(self) -> float:
        """Get the total duration of the song."""
        if self.words:
            return max(word.end for word in self.words)
        if self.chords:
            return max(chord.end for chord in self.chords)
        if self.notes:
            return max(note.end for note in self.notes)
        return 0.0

    def get_word_count(self) -> int:
        """Get the total number of words."""
        return len(self.words)

    def get_chord_count(self) -> int:
        """Get the total number of chords."""
        return len(self.chords)

    def get_note_count(self) -> int:
        """Get the total number of notes."""
        return len(self.notes)
