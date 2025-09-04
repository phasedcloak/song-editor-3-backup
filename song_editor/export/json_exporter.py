#!/usr/bin/env python3
"""
JSON Exporter Module

Handles export of song data to enhanced JSON format for Song Editor 3.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class JSONExporter:
    """Handles export of song data to enhanced JSON format."""

    def __init__(
        self,
        include_processing_info: bool = True,
        include_audio_analysis: bool = True,
        include_alternatives: bool = True,
        pretty_print: bool = True,
        validate_schema: bool = True
    ):
        self.include_processing_info = include_processing_info
        self.include_audio_analysis = include_audio_analysis
        self.include_alternatives = include_alternatives
        self.pretty_print = pretty_print
        self.validate_schema = validate_schema

    def _validate_song_data(self, song_data: Dict[str, Any]) -> bool:
        """Validate song data against schema requirements."""
        try:
            # Check required fields
            required_fields = ['metadata', 'words']
            for field in required_fields:
                if field not in song_data:
                    logging.error(f"Missing required field: {field}")
                    return False

            # Validate metadata
            metadata = song_data.get('metadata', {})
            required_metadata = ['version', 'created_at', 'source_audio']
            for field in required_metadata:
                if field not in metadata:
                    logging.error(f"Missing required metadata field: {field}")
                    return False

            # Validate words
            words = song_data.get('words', [])
            if not isinstance(words, list):
                logging.error("Words must be a list")
                return False

            for i, word in enumerate(words):
                if not isinstance(word, dict):
                    logging.error(f"Word {i} must be a dictionary")
                    return False

                required_word_fields = ['text', 'start', 'end', 'confidence']
                for field in required_word_fields:
                    if field not in word:
                        logging.error(f"Word {i} missing required field: {field}")
                        return False

                # Validate timing
                start = word.get('start', 0)
                end = word.get('end', 0)
                # Allow for small floating point differences or edge cases
                if start > end + 0.001:  # Allow 1ms tolerance
                    logging.warning(f"Word {i} has suspicious timing: start ({start}) > end ({end})")
                    # Fix the timing by setting end = start + small duration
                    word['end'] = start + 0.01  # 10ms minimum duration

            return True

        except Exception as e:
            logging.error(f"Error validating song data: {e}")
            return False

    def _clean_word_data(self, word: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate word data."""
        cleaned_word = {
            'text': str(word.get('text', '')),
            'start': float(word.get('start', 0.0)),
            'end': float(word.get('end', 0.0)),
            'confidence': float(word.get('confidence', 0.0))
        }

        # Add alternatives if available and requested
        if self.include_alternatives and 'alternatives' in word:
            alternatives = word.get('alternatives', [])
            if isinstance(alternatives, list) and alternatives:
                cleaned_word['alternatives'] = alternatives

        # Add chord information if available
        if 'chord' in word and word['chord']:
            cleaned_word['chord'] = word['chord']

        return cleaned_word

    def _clean_chord_data(self, chord: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate chord data."""
        cleaned_chord = {
            'symbol': str(chord.get('symbol', '')),
            'root': str(chord.get('root', '')),
            'quality': str(chord.get('quality', '')),
            'start': float(chord.get('start', 0.0)),
            'end': float(chord.get('end', 0.0))
        }

        # Add optional fields
        if 'bass' in chord:
            cleaned_chord['bass'] = chord['bass']

        if 'duration' in chord:
            cleaned_chord['duration'] = float(chord.get('duration', 0.0))

        if 'confidence' in chord:
            cleaned_chord['confidence'] = float(chord.get('confidence', 0.0))

        if 'detection_method' in chord:
            cleaned_chord['detection_method'] = str(chord.get('detection_method', ''))

        return cleaned_chord

    def _clean_note_data(self, note: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate note data."""
        cleaned_note = {
            'pitch_midi': int(note.get('pitch_midi', 60)),
            'start': float(note.get('start', 0.0)),
            'end': float(note.get('end', 0.0))
        }

        # Add optional fields
        if 'pitch_name' in note:
            cleaned_note['pitch_name'] = str(note.get('pitch_name', ''))

        if 'duration' in note:
            cleaned_note['duration'] = float(note.get('duration', 0.0))

        if 'velocity' in note:
            cleaned_note['velocity'] = int(note.get('velocity', 80))

        if 'confidence' in note:
            cleaned_note['confidence'] = float(note.get('confidence', 0.0))

        if 'detection_method' in note:
            cleaned_note['detection_method'] = str(note.get('detection_method', ''))

        return cleaned_note

    def _clean_segment_data(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate segment data."""
        cleaned_segment = {
            'type': str(segment.get('type', 'other')),
            'start': float(segment.get('start', 0.0)),
            'end': float(segment.get('end', 0.0))
        }

        # Add optional fields
        if 'label' in segment:
            cleaned_segment['label'] = str(segment.get('label', ''))

        if 'confidence' in segment:
            cleaned_segment['confidence'] = float(segment.get('confidence', 0.0))

        return cleaned_segment

    def _prepare_export_data(self, song_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare song data for export."""
        export_data = {}

        # Copy metadata
        if 'metadata' in song_data:
            export_data['metadata'] = song_data['metadata'].copy()

        # Clean and add words
        if 'words' in song_data:
            export_data['words'] = [
                self._clean_word_data(word) for word in song_data['words']
            ]

        # Clean and add chords
        if 'chords' in song_data:
            export_data['chords'] = [
                self._clean_chord_data(chord) for chord in song_data['chords']
            ]

        # Clean and add notes
        if 'notes' in song_data:
            export_data['notes'] = [
                self._clean_note_data(note) for note in song_data['notes']
            ]

        # Clean and add segments
        if 'segments' in song_data:
            export_data['segments'] = [
                self._clean_segment_data(segment) for segment in song_data['segments']
            ]

        # Add audio analysis if requested
        if self.include_audio_analysis and 'audio_analysis' in song_data:
            export_data['audio_analysis'] = song_data['audio_analysis'].copy()

        # Add processing info if requested
        if self.include_processing_info and 'processing_info' in song_data:
            export_data['processing_info'] = song_data['processing_info'].copy()

        return export_data

    def _add_export_metadata(self, export_data: Dict[str, Any], output_path: str) -> None:
        """Add export-specific metadata."""
        if 'metadata' not in export_data:
            export_data['metadata'] = {}

        metadata = export_data['metadata']

        # Add export information
        metadata['exported_at'] = datetime.now().isoformat()
        metadata['export_format'] = 'json'
        metadata['export_version'] = '3.0.0'

        # Add file information
        if output_path:
            metadata['export_file'] = str(Path(output_path).absolute())

        # Add exporter information
        metadata['exporter_info'] = {
            'include_processing_info': self.include_processing_info,
            'include_audio_analysis': self.include_audio_analysis,
            'include_alternatives': self.include_alternatives,
            'pretty_print': self.pretty_print,
            'validate_schema': self.validate_schema
        }

    def export(self, song_data: Dict[str, Any], output_path: str) -> bool:
        """Export song data to JSON file."""
        try:
            logging.info(f"Exporting JSON to: {output_path}")

            # Validate song data if requested
            if self.validate_schema:
                if not self._validate_song_data(song_data):
                    logging.error("Song data validation failed")
                    return False

            # Prepare export data
            export_data = self._prepare_export_data(song_data)

            # Add export metadata
            self._add_export_metadata(export_data, output_path)

            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                if self.pretty_print:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
                else:
                    json.dump(export_data, f, ensure_ascii=False, default=str)

            # Log export statistics
            word_count = len(export_data.get('words', []))
            chord_count = len(export_data.get('chords', []))
            note_count = len(export_data.get('notes', []))
            segment_count = len(export_data.get('segments', []))

            logging.info("JSON exported successfully:")
            logging.info(f"  - Words: {word_count}")
            logging.info(f"  - Chords: {chord_count}")
            logging.info(f"  - Notes: {note_count}")
            logging.info(f"  - Segments: {segment_count}")

            return True

        except Exception as e:
            logging.error(f"Error exporting JSON: {e}")
            return False

    def export_minimal(self, song_data: Dict[str, Any], output_path: str) -> bool:
        """Export minimal JSON with only essential data."""
        try:
            logging.info(f"Exporting minimal JSON to: {output_path}")

            # Create minimal export data
            minimal_data = {
                'metadata': song_data.get('metadata', {}).copy(),
                'words': [
                    {
                        'text': word.get('text', ''),
                        'start': word.get('start', 0.0),
                        'end': word.get('end', 0.0),
                        'confidence': word.get('confidence', 0.0)
                    }
                    for word in song_data.get('words', [])
                ]
            }

            # Add basic audio analysis
            if 'audio_analysis' in song_data:
                audio_analysis = song_data['audio_analysis']
                minimal_data['audio_analysis'] = {
                    'duration': audio_analysis.get('duration'),
                    'tempo': audio_analysis.get('tempo'),
                    'key': audio_analysis.get('key', {})
                }

            # Add export metadata
            self._add_export_metadata(minimal_data, output_path)

            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                if self.pretty_print:
                    json.dump(minimal_data, f, indent=2, ensure_ascii=False, default=str)
                else:
                    json.dump(minimal_data, f, ensure_ascii=False, default=str)

            logging.info("Minimal JSON exported successfully")
            return True

        except Exception as e:
            logging.error(f"Error exporting minimal JSON: {e}")
            return False

    def export_analysis_only(self, song_data: Dict[str, Any], output_path: str) -> bool:
        """Export only analysis data (no lyrics/words)."""
        try:
            logging.info(f"Exporting analysis-only JSON to: {output_path}")

            # Create analysis-only export data
            analysis_data = {
                'metadata': song_data.get('metadata', {}).copy()
            }

            # Add audio analysis
            if 'audio_analysis' in song_data:
                analysis_data['audio_analysis'] = song_data['audio_analysis'].copy()

            # Add chords
            if 'chords' in song_data:
                analysis_data['chords'] = [
                    self._clean_chord_data(chord) for chord in song_data['chords']
                ]

            # Add notes
            if 'notes' in song_data:
                analysis_data['notes'] = [
                    self._clean_note_data(note) for note in song_data['notes']
                ]

            # Add segments
            if 'segments' in song_data:
                analysis_data['segments'] = [
                    self._clean_segment_data(segment) for segment in song_data['segments']
                ]

            # Add processing info
            if 'processing_info' in song_data:
                analysis_data['processing_info'] = song_data['processing_info'].copy()

            # Add export metadata
            self._add_export_metadata(analysis_data, output_path)

            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                if self.pretty_print:
                    json.dump(analysis_data, f, indent=2, ensure_ascii=False, default=str)
                else:
                    json.dump(analysis_data, f, ensure_ascii=False, default=str)

            logging.info("Analysis-only JSON exported successfully")
            return True

        except Exception as e:
            logging.error(f"Error exporting analysis-only JSON: {e}")
            return False

    def validate_json_file(self, file_path: str) -> bool:
        """Validate a JSON file against the schema."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return self._validate_song_data(data)

        except Exception as e:
            logging.error(f"Error validating JSON file: {e}")
            return False

    def get_exporter_info(self) -> Dict[str, Any]:
        """Get information about the JSON exporter."""
        return {
            'include_processing_info': self.include_processing_info,
            'include_audio_analysis': self.include_audio_analysis,
            'include_alternatives': self.include_alternatives,
            'pretty_print': self.pretty_print,
            'validate_schema': self.validate_schema
        }
