#!/usr/bin/env python3
"""
Chord Detector Module

Handles chord detection using Chordino and other methods for Song Editor 3.
"""

import os
import logging
import tempfile
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Any, List
from datetime import datetime
# import vamp - optional

# Optional imports
try:
    from chord_extractor import extract_chords
    CHORD_EXTRACTOR_AVAILABLE = True
except ImportError:
    CHORD_EXTRACTOR_AVAILABLE = False
    logging.warning("Chord Extractor not available")


class ChordDetector:
    """Handles chord detection using various methods including Chordino."""

    def __init__(
        self,
        use_chordino: bool = True,
        min_confidence: float = 0.3,
        chord_simplification: bool = False,  # Default to False to preserve richness
        preserve_chord_richness: bool = True,  # New option to explicitly preserve richness
        window_size: float = 0.5
    ):
        self.use_chordino = use_chordino and CHORD_EXTRACTOR_AVAILABLE
        self.min_confidence = min_confidence
        self.chord_simplification = chord_simplification and not preserve_chord_richness
        self.preserve_chord_richness = preserve_chord_richness
        self.window_size = window_size

        # Chord mapping for consistency (preserves richness, no simplification)
        self.chord_mapping = {
            # Major chords - preserve as-is
            'C': 'C', 'C#': 'C#', 'D': 'D', 'D#': 'D#', 'E': 'E', 'F': 'F',
            'F#': 'F#', 'G': 'G', 'G#': 'G#', 'A': 'A', 'A#': 'A#', 'B': 'B',
            # Minor chords - preserve as-is
            'Cm': 'Cm', 'C#m': 'C#m', 'Dm': 'Dm', 'D#m': 'D#m', 'Em': 'Em', 'Fm': 'Fm',
            'F#m': 'F#m', 'Gm': 'Gm', 'G#m': 'G#m', 'Am': 'Am', 'A#m': 'A#m', 'Bm': 'Bm',
            # Extended chords - preserve full richness
            'C7': 'C7', 'Cm7': 'Cm7', 'Cmaj7': 'Cmaj7', 'Cdim': 'Cdim', 'Caug': 'Caug',
            'C9': 'C9', 'Cm9': 'Cm9', 'Cmaj9': 'Cmaj9', 'C11': 'C11', 'C13': 'C13',
            'Csus2': 'Csus2', 'Csus4': 'Csus4', 'Cadd9': 'Cadd9', 'Cadd11': 'Cadd11',
            'D7': 'D7', 'Dm7': 'Dm7', 'Dmaj7': 'Dmaj7', 'Ddim': 'Ddim', 'Daug': 'Daug',
            'D9': 'D9', 'Dm9': 'Dm9', 'Dmaj9': 'Dmaj9', 'D11': 'D11', 'D13': 'D13',
            'Dsus2': 'Dsus2', 'Dsus4': 'Dsus4', 'Dadd9': 'Dadd9', 'Dadd11': 'Dadd11',
            'E7': 'E7', 'Em7': 'Em7', 'Emaj7': 'Emaj7', 'Edim': 'Edim', 'Eaug': 'Eaug',
            'E9': 'E9', 'Em9': 'Em9', 'Emaj9': 'Emaj9', 'E11': 'E11', 'E13': 'E13',
            'Esus2': 'Esus2', 'Esus4': 'Esus4', 'Eadd9': 'Eadd9', 'Eadd11': 'Eadd11',
            'F7': 'F7', 'Fm7': 'Fm7', 'Fmaj7': 'Fmaj7', 'Fdim': 'Fdim', 'Faug': 'Faug',
            'F9': 'F9', 'Fm9': 'Fm9', 'Fmaj9': 'Fmaj9', 'F11': 'F11', 'F13': 'F13',
            'Fsus2': 'Fsus2', 'Fsus4': 'Fsus4', 'Fadd9': 'Fadd9', 'Fadd11': 'Fadd11',
            'G7': 'G7', 'Gm7': 'Gm7', 'Gmaj7': 'Gmaj7', 'Gdim': 'Gdim', 'Gaug': 'Gaug',
            'G9': 'G9', 'Gm9': 'Gm9', 'Gmaj9': 'Gmaj9', 'G11': 'G11', 'G13': 'G13',
            'Gsus2': 'Gsus2', 'Gsus4': 'Gsus4', 'Gadd9': 'Gadd9', 'Gadd11': 'Gadd11',
            'A7': 'A7', 'Am7': 'Am7', 'Amaj7': 'Amaj7', 'Adim': 'Adim', 'Aaug': 'Aaug',
            'A9': 'A9', 'Am9': 'Am9', 'Amaj9': 'Amaj9', 'A11': 'A11', 'A13': 'A13',
            'Asus2': 'Asus2', 'Asus4': 'Asus4', 'Aadd9': 'Aadd9', 'Aadd11': 'Aadd11',
            'B7': 'B7', 'Bm7': 'Bm7', 'Bmaj7': 'Bmaj7', 'Bdim': 'Bdim', 'Baug': 'Baug',
            'B9': 'B9', 'Bm9': 'Bm9', 'Bmaj9': 'Bmaj9', 'B11': 'B11', 'B13': 'B13',
            'Bsus2': 'Bsus2', 'Bsus4': 'Bsus4', 'Badd9': 'Badd9', 'Badd11': 'Badd11'
        }

    def _save_audio_temp(self, audio: np.ndarray, sample_rate: int) -> str:
        """Save audio to temporary file for chord detection."""
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

    def _detect_chords_chordino(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Detect chords using Chordino."""
        try:
            # Save audio to temporary file
            temp_path = self._save_audio_temp(audio, sample_rate)

            try:
                # Extract chords using chord_extractor
                chords = extract_chords(temp_path)

                # Process results
                processed_chords = []
                for chord_info in chords:
                    chord_symbol = chord_info['chord']
                    start_time = chord_info['start']
                    end_time = chord_info['end']
                    confidence = chord_info.get('confidence', 0.5)

                    # Filter by confidence
                    if confidence >= self.min_confidence:
                        # Parse chord information
                        chord_data = self._parse_chord_symbol(chord_symbol)

                        processed_chord = {
                            'symbol': chord_symbol,
                            'root': chord_data['root'],
                            'quality': chord_data['quality'],
                            'bass': chord_data['bass'],
                            'start': start_time,
                            'end': end_time,
                            'duration': end_time - start_time,
                            'confidence': confidence,
                            'detection_method': 'chordino'
                        }

                        processed_chords.append(processed_chord)

                return processed_chords

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logging.error(f"Error in Chordino chord detection: {e}")
            raise

    def _detect_chords_chromagram(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Detect chords using chromagram analysis."""
        try:
            # Convert to mono if needed
            if len(audio.shape) > 1:
                audio_mono = np.mean(audio, axis=0)
            else:
                audio_mono = audio

            # Extract chromagram
            hop_length = int(self.window_size * sample_rate)
            chromagram = librosa.feature.chroma_cqt(
                y=audio_mono,
                sr=sample_rate,
                hop_length=hop_length
            )

            # Define chord templates
            chord_templates = self._get_chord_templates()

            # Detect chords
            chords = []
            frame_times = librosa.frames_to_time(
                np.arange(chromagram.shape[1]),
                sr=sample_rate,
                hop_length=hop_length
            )

            for i, frame_time in enumerate(frame_times):
                frame_chroma = chromagram[:, i]

                # Find best matching chord
                best_chord = None
                best_correlation = -1.0

                for chord_symbol, template in chord_templates.items():
                    correlation = np.corrcoef(frame_chroma, template)[0, 1]
                    if correlation > best_correlation:
                        best_correlation = correlation
                        best_chord = chord_symbol

                # Calculate confidence
                confidence = max(0.0, min(1.0, (best_correlation + 1) / 2))

                if confidence >= self.min_confidence:
                    # Parse chord information
                    chord_data = self._parse_chord_symbol(best_chord)

                    chord = {
                        'symbol': best_chord,
                        'root': chord_data['root'],
                        'quality': chord_data['quality'],
                        'bass': chord_data['bass'],
                        'start': frame_time,
                        'end': frame_time + self.window_size,
                        'duration': self.window_size,
                        'confidence': confidence,
                        'detection_method': 'chromagram'
                    }

                    chords.append(chord)

            return chords

        except Exception as e:
            logging.error(f"Error in chromagram chord detection: {e}")
            raise

    def _get_chord_templates(self) -> Dict[str, np.ndarray]:
        """Get chord templates for chromagram analysis."""
        # Define comprehensive chord templates to preserve richness
        templates = {}

        # Major chords
        major_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
            templates[note] = np.roll(major_template, i)

        # Minor chords
        minor_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
            templates[f"{note}m"] = np.roll(minor_template, i)

        # Dominant 7th chords
        seventh_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
            templates[f"{note}7"] = np.roll(seventh_template, i)

        # Major 7th chords
        maj7_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])  # Same as major for now
        for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
            templates[f"{note}maj7"] = np.roll(maj7_template, i)

        # Minor 7th chords
        min7_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])  # Same as minor for now
        for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
            templates[f"{note}m7"] = np.roll(min7_template, i)

        # Diminished chords
        dim_template = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
        for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
            templates[f"{note}dim"] = np.roll(dim_template, i)

        # Augmented chords
        aug_template = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
            templates[f"{note}aug"] = np.roll(aug_template, i)

        return templates

    def _parse_chord_symbol(self, chord_symbol: str) -> Dict[str, str]:
        """Parse chord symbol into components."""
        chord_symbol = chord_symbol.strip()

        # Default values
        root = 'C'
        quality = 'maj'
        bass = None

        if not chord_symbol:
            return {'root': root, 'quality': quality, 'bass': bass}

        # Handle inversions (e.g., C/E)
        if '/' in chord_symbol:
            parts = chord_symbol.split('/')
            chord_symbol = parts[0]
            bass = parts[1]

        # Extract root note
        if len(chord_symbol) >= 1:
            if len(chord_symbol) >= 2 and chord_symbol[1] in ['#', 'b']:
                root = chord_symbol[:2]
                quality_part = chord_symbol[2:]
            else:
                root = chord_symbol[0]
                quality_part = chord_symbol[1:]
        else:
            quality_part = ''

        # Determine quality - preserve full chord richness
        if not quality_part:
            quality = 'maj'
        elif quality_part == 'm':
            quality = 'min'
        elif quality_part == '7':
            quality = '7'
        elif quality_part == 'maj7':
            quality = 'maj7'
        elif quality_part == 'm7':
            quality = 'min7'
        elif quality_part == 'dim':
            quality = 'dim'
        elif quality_part == 'aug':
            quality = 'aug'
        elif quality_part == 'sus2':
            quality = 'sus2'
        elif quality_part == 'sus4':
            quality = 'sus4'
        elif quality_part == '9':
            quality = '9'
        elif quality_part == 'm9':
            quality = 'm9'
        elif quality_part == 'maj9':
            quality = 'maj9'
        elif quality_part == '11':
            quality = '11'
        elif quality_part == '13':
            quality = '13'
        elif quality_part == 'add9':
            quality = 'add9'
        elif quality_part == 'add11':
            quality = 'add11'
        else:
            # Preserve any other quality as-is to maintain richness
            quality = quality_part

        return {
            'root': root,
            'quality': quality,
            'bass': bass
        }

    def _simplify_chords(self, chords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standardize chord symbols while preserving full richness."""
        if not self.chord_simplification:
            return chords

        standardized_chords = []
        for chord in chords:
            symbol = chord['symbol']

            # Apply standardization mapping (preserves richness, no simplification)
            if symbol in self.chord_mapping:
                standardized_symbol = self.chord_mapping[symbol]
                chord['symbol'] = standardized_symbol

                # Update parsed components while preserving full chord information
                chord_data = self._parse_chord_symbol(standardized_symbol)
                chord['root'] = chord_data['root']
                chord['quality'] = chord_data['quality']
                chord['bass'] = chord_data['bass']
            else:
                # For unknown chords, preserve as-is to maintain richness
                chord_data = self._parse_chord_symbol(symbol)
                chord['root'] = chord_data['root']
                chord['quality'] = chord_data['quality']
                chord['bass'] = chord_data['bass']

            standardized_chords.append(chord)

        return standardized_chords

    def _merge_similar_chords(self, chords: List[Dict[str, Any]], min_duration: float = 0.5) -> List[Dict[str, Any]]:
        """Merge consecutive similar chords."""
        if not chords:
            return chords

        merged_chords = []
        current_chord = chords[0].copy()

        for next_chord in chords[1:]:
            # Check if chords are similar
            if (current_chord['symbol'] == next_chord['symbol'] and
                    abs(next_chord['start'] - current_chord['end']) < 0.1):
                # Merge chords
                current_chord['end'] = next_chord['end']
                current_chord['duration'] = current_chord['end'] - current_chord['start']
                # Average confidence
                current_chord['confidence'] = (current_chord['confidence'] + next_chord['confidence']) / 2
            else:
                # Add current chord if it meets minimum duration
                if current_chord['duration'] >= min_duration:
                    merged_chords.append(current_chord)
                current_chord = next_chord.copy()

        # Add the last chord
        if current_chord['duration'] >= min_duration:
            merged_chords.append(current_chord)

        return merged_chords

    def detect(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Detect chords in audio using the selected method."""
        start_time = datetime.now()

        try:
            logging.info("Starting chord detection...")

            if self.use_chordino:
                chords = self._detect_chords_chordino(audio, sample_rate)
                logging.info(f"Chordino detected {len(chords)} chords")
            else:
                chords = self._detect_chords_chromagram(audio, sample_rate)
                logging.info(f"Chromagram detected {len(chords)} chords")

            # Simplify chords if requested
            if self.chord_simplification:
                chords = self._simplify_chords(chords)

            # Merge similar consecutive chords
            chords = self._merge_similar_chords(chords)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            logging.info(f"Chord detection completed: {len(chords)} chords in {processing_time:.1f} seconds")

            return chords

        except Exception as e:
            logging.error(f"Error in chord detection: {e}")
            raise

    def analyze_chord_progression(self, chords: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze chord progression for patterns and key detection."""
        try:
            if not chords:
                return {
                    'key': 'Unknown',
                    'confidence': 0.0,
                    'common_progressions': [],
                    'chord_frequencies': {}
                }

            # Count chord frequencies
            chord_counts = {}
            for chord in chords:
                root = chord['root']
                chord_counts[root] = chord_counts.get(root, 0) + 1

            # Find most common chord (potential key)
            if chord_counts:
                most_common_chord = max(chord_counts.items(), key=lambda x: x[1])
                key = most_common_chord[0]
                confidence = min(1.0, most_common_chord[1] / len(chords))
            else:
                key = 'Unknown'
                confidence = 0.0

            # Find common progressions
            progressions = []
            for i in range(len(chords) - 1):
                progression = f"{chords[i]['root']} -> {chords[i+1]['root']}"
                progressions.append(progression)

            # Count progression frequencies
            progression_counts = {}
            for prog in progressions:
                progression_counts[prog] = progression_counts.get(prog, 0) + 1

            # Get most common progressions
            common_progressions = sorted(
                progression_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            return {
                'key': key,
                'confidence': confidence,
                'common_progressions': common_progressions,
                'chord_frequencies': chord_counts
            }

        except Exception as e:
            logging.warning(f"Error in chord progression analysis: {e}")
            return {
                'key': 'Unknown',
                'confidence': 0.0,
                'common_progressions': [],
                'chord_frequencies': {}
            }

    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the chord detector."""
        return {
            'use_chordino': self.use_chordino,
            'min_confidence': self.min_confidence,
            'chord_simplification': self.chord_simplification,
            'preserve_chord_richness': self.preserve_chord_richness,
            'window_size': self.window_size,
            'chord_extractor_available': CHORD_EXTRACTOR_AVAILABLE
        }
