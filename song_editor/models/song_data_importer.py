"""
Song Data Importer Module

Handles importing pre-processed song data from JSON files that follow the song_data_schema.json format.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from .lyrics import WordRow


@dataclass
class ChordData:
    """Represents chord data from imported song data"""
    symbol: str
    root: str
    quality: str
    bass: Optional[str]
    start: float
    end: float
    confidence: float


@dataclass
class NoteData:
    """Represents note data from imported song data"""
    pitch_midi: int
    pitch_name: Optional[str]
    start: float
    end: float
    velocity: Optional[int]
    confidence: float


@dataclass
class SegmentData:
    """Represents segment data from imported song data"""
    type: str
    label: Optional[str]
    start: float
    end: float
    confidence: float


@dataclass
class SongData:
    """Complete song data structure"""
    metadata: Dict
    words: List[WordRow]
    chords: List[ChordData]
    notes: List[NoteData]
    segments: List[SegmentData]


class SongDataImporter:
    """Handles importing song data from JSON files"""

    def __init__(self):
        self.schema_path = Path(__file__).parent.parent.parent / "song_data_schema.json"

    def find_song_data_file(self, audio_path: str) -> Optional[str]:
        """
        Look for a .song_data file with the same basename as the audio file

        Args:
            audio_path: Path to the audio file

        Returns:
            Path to the song data file if found, None otherwise
        """
        audio_path = Path(audio_path)
        possible_extensions = ['.song_data', '.song_data.json', '.json']

        for ext in possible_extensions:
            song_data_path = audio_path.with_suffix(ext)
            if song_data_path.exists():
                return str(song_data_path)

        return None

    def validate_song_data(self, data: Dict) -> bool:
        """
        Basic validation of song data structure

        Args:
            data: The song data dictionary

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required top-level fields
            if 'metadata' not in data or 'words' not in data:
                return False

            # Check metadata
            metadata = data['metadata']
            required_metadata = ['version', 'created_at', 'source_audio']
            if not all(field in metadata for field in required_metadata):
                return False

            # Check words array
            words = data['words']
            if not isinstance(words, list):
                return False

            # Check each word has required fields
            required_word_fields = ['text', 'start', 'end', 'confidence']
            for word in words:
                if not isinstance(word, dict):
                    return False
                if not all(field in word for field in required_word_fields):
                    return False
                if not isinstance(word['text'], str):
                    return False
                if not isinstance(word['start'], (int, float)) or not isinstance(word['end'], (int, float)):
                    return False
                if not isinstance(word['confidence'], (int, float)):
                    return False

            return True

        except Exception:
            return False

    def parse_chord_data(self, chord_dict: Dict) -> ChordData:
        """Parse chord data from dictionary"""
        return ChordData(
            symbol=chord_dict.get('symbol', ''),
            root=chord_dict.get('root', ''),
            quality=chord_dict.get('quality', ''),
            bass=chord_dict.get('bass'),
            start=chord_dict.get('start', 0.0),
            end=chord_dict.get('end', 0.0),
            confidence=chord_dict.get('confidence', 1.0)
        )

    def parse_note_data(self, note_dict: Dict) -> NoteData:
        """Parse note data from dictionary"""
        return NoteData(
            pitch_midi=note_dict.get('pitch_midi', 0),
            pitch_name=note_dict.get('pitch_name'),
            start=note_dict.get('start', 0.0),
            end=note_dict.get('end', 0.0),
            velocity=note_dict.get('velocity'),
            confidence=note_dict.get('confidence', 1.0)
        )

    def parse_segment_data(self, segment_dict: Dict) -> SegmentData:
        """Parse segment data from dictionary"""
        return SegmentData(
            type=segment_dict.get('type', 'other'),
            label=segment_dict.get('label'),
            start=segment_dict.get('start', 0.0),
            end=segment_dict.get('end', 0.0),
            confidence=segment_dict.get('confidence', 1.0)
        )

    def convert_to_word_rows(self, words_data: List[Dict]) -> List[WordRow]:
        """Convert imported word data to WordRow objects"""
        word_rows = []

        for word_data in words_data:
            # Check if this word has alternatives
            alt_text = None
            if 'alternatives' in word_data and word_data['alternatives']:
                # Use the highest confidence alternative
                best_alternative = max(word_data['alternatives'], key=lambda x: x.get('confidence', 0.0))
                alt_text = best_alternative['text']

            # Create WordRow with alternative text if available
            word_row = WordRow(
                text=word_data['text'],
                start=word_data['start'],
                end=word_data['end'],
                confidence=word_data['confidence'],
                chord=None,  # Don't set chord here - it will be assigned from chords section
                alt_text=alt_text,
                alt_chord=None,
                alt_start=word_data['start'],  # Use same timing for alternatives
                alt_end=word_data['end']
            )

            word_rows.append(word_row)

        return word_rows

    def import_song_data(self, song_data_path: str) -> Optional[SongData]:
        """
        Import song data from a JSON file

        Args:
            song_data_path: Path to the song data JSON file

        Returns:
            SongData object if successful, None otherwise
        """
        try:
            with open(song_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate the data
            if not self.validate_song_data(data):
                print(f"Invalid song data format in {song_data_path}")
                return None

            # Parse metadata
            metadata = data['metadata']

            # Convert words to WordRow objects
            words = self.convert_to_word_rows(data['words'])

            # Parse chords
            chords = []
            if 'chords' in data:
                for chord_dict in data['chords']:
                    chords.append(self.parse_chord_data(chord_dict))

            # Parse notes
            notes = []
            if 'notes' in data:
                for note_dict in data['notes']:
                    notes.append(self.parse_note_data(note_dict))

            # Parse segments
            segments = []
            if 'segments' in data:
                for segment_dict in data['segments']:
                    segments.append(self.parse_segment_data(segment_dict))

            return SongData(
                metadata=metadata,
                words=words,
                chords=chords,
                notes=notes,
                segments=segments
            )

        except FileNotFoundError:
            print(f"Song data file not found: {song_data_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in song data file: {e}")
            return None
        except Exception as e:
            print(f"Error importing song data: {e}")
            return None

    def export_song_data(self, song_data: SongData, output_path: str) -> bool:
        """
        Export song data to JSON format

        Args:
            song_data: The song data to export
            output_path: Path where to save the JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert WordRow objects back to dictionaries
            words_data = []
            for word in song_data.words:
                word_dict = {
                    'text': word.text,
                    'start': word.start,
                    'end': word.end,
                    'confidence': word.confidence
                }

                # Add chord information if available
                if word.chord:
                    word_dict['chord'] = {
                        'symbol': word.chord,
                        'root': word.chord[0] if word.chord else '',
                        'quality': word.chord[1:] if len(word.chord) > 1 else 'maj',
                        'bass': None,
                        'confidence': 1.0
                    }

                words_data.append(word_dict)

            # Convert other data structures
            chords_data = []
            for chord in song_data.chords:
                chords_data.append({
                    'symbol': chord.symbol,
                    'root': chord.root,
                    'quality': chord.quality,
                    'bass': chord.bass,
                    'start': chord.start,
                    'end': chord.end,
                    'confidence': chord.confidence
                })

            notes_data = []
            for note in song_data.notes:
                note_dict = {
                    'pitch_midi': note.pitch_midi,
                    'start': note.start,
                    'end': note.end,
                    'confidence': note.confidence
                }
                if note.pitch_name:
                    note_dict['pitch_name'] = note.pitch_name
                if note.velocity:
                    note_dict['velocity'] = note.velocity
                notes_data.append(note_dict)

            segments_data = []
            for segment in song_data.segments:
                segment_dict = {
                    'type': segment.type,
                    'start': segment.start,
                    'end': segment.end,
                    'confidence': segment.confidence
                }
                if segment.label:
                    segment_dict['label'] = segment.label
                segments_data.append(segment_dict)

            # Create the complete data structure
            export_data = {
                'metadata': song_data.metadata,
                'words': words_data,
                'chords': chords_data,
                'notes': notes_data,
                'segments': segments_data
            }

            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"Error exporting song data: {e}")
            return False
