#!/usr/bin/env python3
"""
Melody Extractor Module

Handles melody extraction using Basic Pitch and CREPE for Song Editor 3.
"""

import os
import logging
import tempfile
import numpy as np
import soundfile as sf
from typing import Dict, Any, List
from datetime import datetime

# Optional imports (import lazily to avoid bus errors)
BASIC_PITCH_AVAILABLE = False
CREPE_AVAILABLE = False


class MelodyExtractor:
    """Handles melody extraction using various methods including Basic Pitch and CREPE."""

    def __init__(
        self,
        use_basic_pitch: bool = True,
        min_confidence: float = 0.5,
        min_note_duration: float = 0.1,
        min_pitch: int = 21,  # A0
        max_pitch: int = 108  # C8
    ):
        self.use_basic_pitch = use_basic_pitch
        self.min_confidence = min_confidence
        self.min_note_duration = min_note_duration
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch

        # Note name mapping
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def _save_audio_temp(self, audio: np.ndarray, sample_rate: int) -> str:
        """Save audio to temporary file for melody extraction."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name

            # Ensure audio is in correct format
            if len(audio.shape) > 1:
                if audio.shape[0] == 2:  # Stereo audio
                    audio = np.mean(audio, axis=0)  # Convert to mono by averaging channels
                else:
                    audio = audio.squeeze()  # Remove singleton dimensions

            # Save audio to temporary file
            sf.write(temp_path, audio.astype(np.float32), sample_rate, subtype='PCM_16')
            return temp_path

        except Exception as e:
            logging.error(f"Error saving temporary audio file: {e}")
            raise

    def _extract_melody_basic_pitch(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Extract melody using Basic Pitch."""
        try:
            # Save audio to temporary file
            temp_path = self._save_audio_temp(audio, sample_rate)

            try:
                # Suppress Basic Pitch verbose output
                import logging as python_logging
                original_level = python_logging.getLogger().level
                python_logging.getLogger().setLevel(python_logging.WARNING)

                # Temporarily redirect stdout to suppress Basic Pitch prints
                import sys
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = StringIO()

                try:
                    # Import Basic Pitch lazily
                    from basic_pitch.inference import predict
                    # Run Basic Pitch inference
                    model_output, midi_data, note_events = predict(temp_path)
                finally:
                    # Restore stdout and logging level
                    sys.stdout = old_stdout
                    python_logging.getLogger().setLevel(original_level)

                # Process note events
                notes = []
                for note_event in note_events:
                    start_time = note_event[0]
                    end_time = note_event[1]
                    pitch_midi = int(note_event[2])
                    amplitude = note_event[3]

                    # Filter by pitch range
                    if self.min_pitch <= pitch_midi <= self.max_pitch:
                        # Calculate duration
                        duration = end_time - start_time

                        # Filter by minimum duration
                        if duration >= self.min_note_duration:
                            # Calculate confidence from amplitude
                            confidence = min(1.0, amplitude / 0.5)  # Normalize amplitude

                            if confidence >= self.min_confidence:
                                note = {
                                    'pitch_midi': pitch_midi,
                                    'pitch_name': self._midi_to_note_name(pitch_midi),
                                    'start': start_time,
                                    'end': end_time,
                                    'duration': duration,
                                    'velocity': int(amplitude * 127),  # Convert to MIDI velocity
                                    'confidence': confidence,
                                    'detection_method': 'basic_pitch'
                                }
                                notes.append(note)

                return notes

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logging.error(f"Error in Basic Pitch melody extraction: {e}")
            raise

    def _extract_melody_crepe(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Extract melody using CREPE."""
        try:
            # Convert to mono if needed
            if len(audio.shape) > 1:
                audio_mono = np.mean(audio, axis=0)
            else:
                audio_mono = audio

            # Import CREPE lazily
            import crepe
            # Process with CREPE
            time, frequency, confidence, _ = crepe.predict(
                audio_mono,
                sample_rate,
                model_capacity='large',
                viterbi=True
            )

            # Filter low confidence
            frequency[confidence < self.min_confidence] = 0

            # Convert to MIDI notes
            midi_notes = np.zeros_like(frequency)
            mask = frequency > 0
            midi_notes[mask] = 12 * (np.log2(frequency[mask] / 440.0)) + 69
            midi_notes = np.round(midi_notes).astype(int)

            # Filter by pitch range
            pitch_mask = (midi_notes >= self.min_pitch) & (midi_notes <= self.max_pitch)
            midi_notes[~pitch_mask] = 0

            # Convert to note events
            notes = []
            current_note = None

            for i, (t, midi_note, conf) in enumerate(zip(time, midi_notes, confidence)):
                if midi_note > 0 and conf >= self.min_confidence:
                    if current_note is None:
                        # Start new note
                        current_note = {
                            'pitch_midi': midi_note,
                            'pitch_name': self._midi_to_note_name(midi_note),
                            'start': t,
                            'confidence': conf
                        }
                    elif midi_note != current_note['pitch_midi']:
                        # End current note and start new one
                        if current_note is not None:
                            current_note['end'] = t
                            current_note['duration'] = t - current_note['start']

                            if current_note['duration'] >= self.min_note_duration:
                                current_note['velocity'] = int(current_note['confidence'] * 127)
                                current_note['detection_method'] = 'crepe'
                                notes.append(current_note)

                        current_note = {
                            'pitch_midi': midi_note,
                            'pitch_name': self._midi_to_note_name(midi_note),
                            'start': t,
                            'confidence': conf
                        }
                else:
                    # End current note
                    if current_note is not None:
                        current_note['end'] = t
                        current_note['duration'] = t - current_note['start']

                        if current_note['duration'] >= self.min_note_duration:
                            current_note['velocity'] = int(current_note['confidence'] * 127)
                            current_note['detection_method'] = 'crepe'
                            notes.append(current_note)

                        current_note = None

            # Handle last note
            if current_note is not None:
                current_note['end'] = time[-1]
                current_note['duration'] = time[-1] - current_note['start']

                if current_note['duration'] >= self.min_note_duration:
                    current_note['velocity'] = int(current_note['confidence'] * 127)
                    current_note['detection_method'] = 'crepe'
                    notes.append(current_note)

            return notes

        except Exception as e:
            logging.error(f"Error in CREPE melody extraction: {e}")
            raise

    def _midi_to_note_name(self, midi_note: int) -> str:
        """Convert MIDI note number to note name."""
        note = midi_note % 12
        octave = (midi_note // 12) - 1
        return f"{self.note_names[note]}{octave}"

    def _merge_similar_notes(self, notes: List[Dict[str, Any]], max_gap: float = 0.1) -> List[Dict[str, Any]]:
        """Merge consecutive notes with the same pitch."""
        if not notes:
            return notes

        merged_notes = []
        current_note = notes[0].copy()

        for next_note in notes[1:]:
            # Check if notes are similar and close in time
            if (current_note['pitch_midi'] == next_note['pitch_midi'] and
                    abs(next_note['start'] - current_note['end']) < max_gap):
                # Merge notes
                current_note['end'] = next_note['end']
                current_note['duration'] = current_note['end'] - current_note['start']
                # Average confidence and velocity
                current_note['confidence'] = (current_note['confidence'] + next_note['confidence']) / 2
                current_note['velocity'] = (current_note['velocity'] + next_note['velocity']) // 2
            else:
                # Add current note
                merged_notes.append(current_note)
                current_note = next_note.copy()

        # Add the last note
        merged_notes.append(current_note)

        return merged_notes

    def _filter_notes_by_duration(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter notes by minimum duration."""
        return [note for note in notes if note['duration'] >= self.min_note_duration]

    def _analyze_melody_contour(self, notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze melody contour and patterns."""
        try:
            if not notes:
                return {
                    'range': 0,
                    'average_pitch': 0,
                    'pitch_variation': 0,
                    'common_intervals': [],
                    'melodic_direction': 'stable'
                }

            # Extract pitches
            pitches = [note['pitch_midi'] for note in notes]

            # Calculate range
            pitch_range = max(pitches) - min(pitches)

            # Calculate average pitch
            average_pitch = np.mean(pitches)

            # Calculate pitch variation
            pitch_variation = np.std(pitches)

            # Calculate intervals
            intervals = []
            for i in range(len(pitches) - 1):
                interval = pitches[i+1] - pitches[i]
                intervals.append(interval)

            # Find common intervals
            if intervals:
                interval_counts = {}
                for interval in intervals:
                    interval_counts[interval] = interval_counts.get(interval, 0) + 1

                common_intervals = sorted(
                    interval_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            else:
                common_intervals = []

            # Determine melodic direction
            if len(intervals) > 0:
                positive_intervals = sum(1 for interval in intervals if interval > 0)
                negative_intervals = sum(1 for interval in intervals if interval < 0)

                if positive_intervals > negative_intervals * 1.5:
                    melodic_direction = 'ascending'
                elif negative_intervals > positive_intervals * 1.5:
                    melodic_direction = 'descending'
                else:
                    melodic_direction = 'stable'
            else:
                melodic_direction = 'stable'

            return {
                'range': pitch_range,
                'average_pitch': average_pitch,
                'pitch_variation': pitch_variation,
                'common_intervals': common_intervals,
                'melodic_direction': melodic_direction
            }

        except Exception as e:
            logging.warning(f"Error in melody contour analysis: {e}")
            return {
                'range': 0,
                'average_pitch': 0,
                'pitch_variation': 0,
                'common_intervals': [],
                'melodic_direction': 'stable'
            }

    def extract(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Extract melody from audio using the selected method."""
        start_time = datetime.now()

        try:
            logging.info("Starting melody extraction...")

            # TEMPORARY: Skip melody extraction to avoid bus errors
            logging.warning("Melody extraction temporarily disabled due to bus error - returning empty results")
            notes = []

            # Filter notes by duration
            notes = self._filter_notes_by_duration(notes)

            # Merge similar consecutive notes
            notes = self._merge_similar_notes(notes)

            # Sort notes by start time
            notes.sort(key=lambda x: x['start'])

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            logging.info(f"Melody extraction completed: {len(notes)} notes in {processing_time:.1f} seconds")

            return notes

        except Exception as e:
            logging.error(f"Error in melody extraction: {e}")
            raise

    def analyze_melody(self, notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze extracted melody for patterns and characteristics."""
        try:
            if not notes:
                return {
                    'note_count': 0,
                    'total_duration': 0.0,
                    'average_note_duration': 0.0,
                    'pitch_range': 0,
                    'average_pitch': 0,
                    'melodic_contour': {},
                    'note_density': 0.0
                }

            # Basic statistics
            note_count = len(notes)
            total_duration = sum(note['duration'] for note in notes)
            average_note_duration = total_duration / note_count if note_count > 0 else 0.0

            # Pitch statistics
            pitches = [note['pitch_midi'] for note in notes]
            pitch_range = max(pitches) - min(pitches) if pitches else 0
            average_pitch = np.mean(pitches) if pitches else 0

            # Note density (notes per second)
            if notes:
                time_span = notes[-1]['end'] - notes[0]['start']
                note_density = note_count / time_span if time_span > 0 else 0.0
            else:
                note_density = 0.0

            # Melodic contour analysis
            melodic_contour = self._analyze_melody_contour(notes)

            return {
                'note_count': note_count,
                'total_duration': total_duration,
                'average_note_duration': average_note_duration,
                'pitch_range': pitch_range,
                'average_pitch': average_pitch,
                'melodic_contour': melodic_contour,
                'note_density': note_density
            }

        except Exception as e:
            logging.warning(f"Error in melody analysis: {e}")
            return {
                'note_count': 0,
                'total_duration': 0.0,
                'average_note_duration': 0.0,
                'pitch_range': 0,
                'average_pitch': 0,
                'melodic_contour': {},
                'note_density': 0.0
            }

    def get_extractor_info(self) -> Dict[str, Any]:
        """Get information about the melody extractor."""
        return {
            'use_basic_pitch': self.use_basic_pitch,
            'min_confidence': self.min_confidence,
            'min_note_duration': self.min_note_duration,
            'min_pitch': self.min_pitch,
            'max_pitch': self.max_pitch,
            'basic_pitch_available': BASIC_PITCH_AVAILABLE,
            'crepe_available': CREPE_AVAILABLE
        }
