#!/usr/bin/env python3
"""
MIDI Exporter Module

Handles export of song data to multi-track MIDI format with lyrics,
chords, and melody for Song Editor 3.
"""

import logging
import mido
from typing import Dict, Any, List, Optional

# Optional imports
try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False
    logging.warning("Pretty MIDI not available")


class MidiExporter:
    """Handles export of song data to MIDI format."""

    def __init__(
        self,
        ticks_per_beat: int = 480,
        include_tempo_map: bool = True,
        include_lyrics: bool = True,
        include_chords: bool = True,
        include_melody: bool = True
    ):
        self.ticks_per_beat = ticks_per_beat
        self.include_tempo_map = include_tempo_map
        self.include_lyrics = include_lyrics
        self.include_chords = include_chords
        self.include_melody = include_melody

    def _sanitize_text_for_midi(self, text: str) -> str:
        """Sanitize text to be compatible with MIDI latin-1 encoding."""
        if not text:
            return ""

        # Convert to string if needed
        text = str(text)

        # Replace common problematic characters
        replacements = {
            '—': '-',  # em dash to hyphen
            '–': '-',  # en dash to hyphen
            '"': '"',  # smart quotes to regular quotes
            '"': '"',
            ''': "'",  # smart apostrophes to regular apostrophes
            ''': "'",
            '…': '...',  # ellipsis to three dots
            '–': '-',  # various dashes
            '—': '-',
            '−': '-',
            '…': '...',
            '…': '...',
            '…': '...',
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove or replace any remaining non-ASCII characters
        try:
            # Try to encode as latin-1
            text.encode('latin-1')
            return text
        except UnicodeEncodeError:
            # If encoding fails, remove problematic characters
            cleaned_text = ""
            for char in text:
                try:
                    char.encode('latin-1')
                    cleaned_text += char
                except UnicodeEncodeError:
                    # Replace with a safe character
                    cleaned_text += '?'
            return cleaned_text

    def _time_to_ticks(self, time_seconds: float, tempo_bpm: float) -> int:
        """Convert time in seconds to MIDI ticks."""
        beats_per_second = tempo_bpm / 60.0
        beats = time_seconds * beats_per_second
        return int(beats * self.ticks_per_beat)

    def _create_tempo_track(self, tempo_bpm: float) -> mido.MidiTrack:
        """Create tempo track with fixed tempo and time signature."""
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name='Tempo & Time Signature', time=0))

        # Set tempo
        tempo_us = int(60000000 / tempo_bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))

        # Set time signature (default to 4/4)
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))

        return track

    def _create_variable_tempo_track(self, tempo_data: List[Dict[str, Any]]) -> mido.MidiTrack:
        """Create tempo track with variable tempo changes."""
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name='Variable Tempo', time=0))

        # Set initial time signature (default to 4/4)
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))

        last_tick = 0
        for tempo_change in tempo_data:
            time_seconds = tempo_change.get('time', 0)
            tempo_bpm = tempo_change.get('tempo', 120.0)

            # Validate time_seconds is numeric
            try:
                time_seconds = float(time_seconds)
            except (ValueError, TypeError):
                time_seconds = 0.0  # Default fallback

            # Validate tempo_bpm is numeric
            try:
                tempo_bpm = float(tempo_bpm)
            except (ValueError, TypeError):
                tempo_bpm = 120.0  # Default fallback

            # Convert time to ticks (assuming 120 BPM for time conversion)
            beats_per_second = 120.0 / 60.0
            beats = time_seconds * beats_per_second
            current_tick = int(beats * self.ticks_per_beat)
            delta_ticks = current_tick - last_tick

            # Set tempo
            tempo_us = int(60000000 / tempo_bpm)
            track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=delta_ticks))

            last_tick = current_tick

        return track

    def _create_lyrics_track(self, words: List[Dict[str, Any]], tempo_bpm: float) -> mido.MidiTrack:
        """Create lyrics track with word-level timing."""
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name='Lyrics', time=0))

        if not words:
            return track

        last_tick = 0
        for word in words:
            # Calculate tick position
            start_time = word.get('start', 0)
            current_tick = self._time_to_ticks(start_time, tempo_bpm)
            delta_ticks = current_tick - last_tick

            # Add lyric event
            text = self._sanitize_text_for_midi(word.get('text', ''))
            if text:
                track.append(mido.MetaMessage('lyrics', text=text, time=delta_ticks))
                last_tick = current_tick

        return track

    def _create_chords_track(self, chords: List[Dict[str, Any]], tempo_bpm: float) -> mido.MidiTrack:
        """Create chords track with chord symbols and block notes."""
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name='Chords', time=0))

        if not chords:
            return track

        last_tick = 0
        for chord in chords:
            # Calculate tick position
            start_time = chord.get('start', 0)
            end_time = chord.get('end', start_time + 1.0)

            current_tick = self._time_to_ticks(start_time, tempo_bpm)
            delta_ticks = current_tick - last_tick

            # Add chord symbol as text event
            chord_symbol = chord.get('symbol', '')
            if chord_symbol:
                track.append(mido.MetaMessage('text', text=chord_symbol, time=delta_ticks))

            # Add chord block notes (simplified chord voicing)
            chord_notes = self._get_chord_notes(chord_symbol)
            if chord_notes:
                # Note on events
                for note in chord_notes:
                    track.append(mido.Message('note_on', note=note, velocity=60, time=0))

                # Note off events
                chord_duration = self._time_to_ticks(end_time - start_time, tempo_bpm)
                for i, note in enumerate(chord_notes):
                    # All notes off at the same time
                    track.append(mido.Message('note_off', note=note, velocity=0, time=0 if i == 0 else chord_duration))

            last_tick = current_tick

        return track

    def _get_chord_notes(self, chord_symbol: str) -> List[int]:
        """Get MIDI note numbers for a chord symbol."""
        # Simplified chord voicing - just root, third, and fifth
        chord_notes = []

        # Parse chord root
        if len(chord_symbol) >= 1:
            if len(chord_symbol) >= 2 and chord_symbol[1] in ['#', 'b']:
                root_note = chord_symbol[:2]
                quality = chord_symbol[2:]
            else:
                root_note = chord_symbol[0]
                quality = chord_symbol[1:]
        else:
            return chord_notes

        # Note name to MIDI number mapping
        note_to_midi = {
            'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63,
            'E': 64, 'F': 65, 'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68, 'Ab': 68,
            'A': 69, 'A#': 70, 'Bb': 70, 'B': 71
        }

        if root_note not in note_to_midi:
            return chord_notes

        root_midi = note_to_midi[root_note]

        # Add root
        chord_notes.append(root_midi)

        # Add third and fifth based on chord quality
        if not quality or quality == 'maj':
            # Major chord: root, major third, perfect fifth
            chord_notes.extend([root_midi + 4, root_midi + 7])
        elif quality == 'm' or quality == 'min':
            # Minor chord: root, minor third, perfect fifth
            chord_notes.extend([root_midi + 3, root_midi + 7])
        elif quality == '7':
            # Dominant 7th: root, major third, perfect fifth, minor seventh
            chord_notes.extend([root_midi + 4, root_midi + 7, root_midi + 10])
        elif quality == 'm7' or quality == 'min7':
            # Minor 7th: root, minor third, perfect fifth, minor seventh
            chord_notes.extend([root_midi + 3, root_midi + 7, root_midi + 10])
        elif quality == 'maj7':
            # Major 7th: root, major third, perfect fifth, major seventh
            chord_notes.extend([root_midi + 4, root_midi + 7, root_midi + 11])
        elif quality == 'dim':
            # Diminished: root, minor third, diminished fifth
            chord_notes.extend([root_midi + 3, root_midi + 6])
        elif quality == 'aug':
            # Augmented: root, major third, augmented fifth
            chord_notes.extend([root_midi + 4, root_midi + 8])
        else:
            # Default to major triad
            chord_notes.extend([root_midi + 4, root_midi + 7])

        return chord_notes

    def _create_melody_track(self, notes: List[Dict[str, Any]], tempo_bpm: float) -> mido.MidiTrack:
        """Create melody track with extracted notes."""
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name='Melody', time=0))

        if not notes:
            return track

        last_tick = 0
        for note in notes:
            # Calculate tick position
            start_time = note.get('start', 0)
            end_time = note.get('end', start_time + 0.5)

            current_tick = self._time_to_ticks(start_time, tempo_bpm)
            delta_ticks = current_tick - last_tick

            # Get note parameters
            pitch = note.get('pitch_midi', 60)
            velocity = note.get('velocity', 80)
            duration = self._time_to_ticks(end_time - start_time, tempo_bpm)

            # Add note on event
            track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=delta_ticks))

            # Add note off event
            track.append(mido.Message('note_off', note=pitch, velocity=0, time=duration))

            last_tick = current_tick + duration

        return track

    def _create_pretty_midi(
        self,
        song_data: Dict[str, Any],
        use_variable_tempo: bool = False
    ) -> Optional['pretty_midi.PrettyMIDI']:
        """Create PrettyMIDI object if available."""
        if not PRETTY_MIDI_AVAILABLE:
            return None

        try:
            # Extract data
            audio_analysis = song_data.get('audio_analysis', {})
            tempo_bpm = audio_analysis.get('tempo', 120.0)
            tempo_changes = audio_analysis.get('tempo_changes', [])
            words = song_data.get('words', [])
            chords = song_data.get('chords', [])
            notes = song_data.get('notes', [])

            # Create PrettyMIDI object
            midi = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)

            # Add variable tempo changes if requested
            if use_variable_tempo and tempo_changes:
                for tempo_change in tempo_changes:
                    time_seconds = tempo_change.get('time', 0)
                    tempo_bpm = tempo_change.get('tempo', 120.0)
                    try:
                        tempo_bpm = float(tempo_bpm)
                    except (ValueError, TypeError):
                        tempo_bpm = 120.0

                    # Add tempo change
                    tempo_us = int(60000000 / tempo_bpm)
                    midi.tempo_changes.append(pretty_midi.TempoChange(tempo_us, time_seconds))

            # Add lyrics as text events
            if self.include_lyrics and words:
                lyrics_program = pretty_midi.Instrument(program=0, is_drum=False, name='Lyrics')
                for word in words:
                    start_time = word.get('start', 0)
                    text = self._sanitize_text_for_midi(word.get('text', ''))
                    if text:
                        note = pretty_midi.Note(velocity=0, pitch=60, start=start_time, end=start_time + 0.1)
                        note.lyric = text
                        lyrics_program.notes.append(note)
                midi.instruments.append(lyrics_program)

            # Add chords
            if self.include_chords and chords:
                chords_program = pretty_midi.Instrument(program=0, is_drum=False, name='Chords')
                for chord in chords:
                    start_time = chord.get('start', 0)
                    end_time = chord.get('end', start_time + 1.0)
                    chord_notes = self._get_chord_notes(chord.get('symbol', ''))

                    for note_pitch in chord_notes:
                        note = pretty_midi.Note(velocity=60, pitch=note_pitch, start=start_time, end=end_time)
                        chords_program.notes.append(note)
                midi.instruments.append(chords_program)

            # Add melody
            if self.include_melody and notes:
                melody_program = pretty_midi.Instrument(program=0, is_drum=False, name='Melody')
                for note_data in notes:
                    start_time = note_data.get('start', 0)
                    end_time = note_data.get('end', start_time + 0.5)
                    pitch = note_data.get('pitch_midi', 60)
                    velocity = note_data.get('velocity', 80)

                    note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
                    melody_program.notes.append(note)
                midi.instruments.append(melody_program)

            return midi

        except Exception as e:
            logging.warning(f"Error creating PrettyMIDI: {e}")
            return None

    def export(self, song_data: Dict[str, Any], output_path: str, use_variable_tempo: bool = False) -> bool:
        """Export song data to MIDI file."""
        try:
            logging.info(f"Exporting MIDI to: {output_path}")

            # Extract data
            audio_analysis = song_data.get('audio_analysis', {})
            tempo_bpm = audio_analysis.get('tempo', 120.0)
            tempo_changes = audio_analysis.get('tempo_changes', [])
            words = song_data.get('words', [])
            chords = song_data.get('chords', [])
            notes = song_data.get('notes', [])

            # Try PrettyMIDI first if available
            if PRETTY_MIDI_AVAILABLE:
                pretty_midi_obj = self._create_pretty_midi(song_data, use_variable_tempo)
                if pretty_midi_obj:
                    # Convert pathlib.Path to string for PrettyMIDI
                    output_path_str = str(output_path) if hasattr(output_path, '__fspath__') else output_path
                    pretty_midi_obj.write(output_path_str)
                    logging.info("MIDI exported using PrettyMIDI")
                    return True

            # Fallback to mido
            midi_file = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)

            # Add tempo track
            if self.include_tempo_map:
                if use_variable_tempo and tempo_changes:
                    tempo_track = self._create_variable_tempo_track(tempo_changes)
                else:
                    tempo_track = self._create_tempo_track(tempo_bpm)
                midi_file.tracks.append(tempo_track)

            # Add lyrics track
            if self.include_lyrics and words:
                lyrics_track = self._create_lyrics_track(words, tempo_bpm)
                midi_file.tracks.append(lyrics_track)

            # Add chords track
            if self.include_chords and chords:
                chords_track = self._create_chords_track(chords, tempo_bpm)
                midi_file.tracks.append(chords_track)

            # Add melody track
            if self.include_melody and notes:
                melody_track = self._create_melody_track(notes, tempo_bpm)
                midi_file.tracks.append(melody_track)

            # Save MIDI file - convert pathlib.Path to string for mido
            output_path_str = str(output_path) if hasattr(output_path, '__fspath__') else output_path
            midi_file.save(output_path_str)

            logging.info(f"MIDI exported successfully: {len(midi_file.tracks)} tracks")
            return True

        except Exception as e:
            logging.error(f"Error exporting MIDI: {e}")
            return False

    def export_tempo_map(self, song_data: Dict[str, Any], output_path: str, use_variable_tempo: bool = False) -> bool:
        """Export tempo map as separate MIDI file."""
        try:
            audio_analysis = song_data.get('audio_analysis', {})
            tempo_bpm = audio_analysis.get('tempo', 120.0)
            tempo_changes = audio_analysis.get('tempo_changes', [])

            # Create tempo-only MIDI
            midi_file = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)

            # Tempo track
            if use_variable_tempo and tempo_changes:
                tempo_track = self._create_variable_tempo_track(tempo_changes)
            else:
                tempo_track = self._create_tempo_track(tempo_bpm)
            midi_file.tracks.append(tempo_track)

            # Save
            midi_file.save(output_path)

            logging.info(f"Tempo map exported to: {output_path}")
            return True

        except Exception as e:
            logging.error(f"Error exporting tempo map: {e}")
            return False

    def get_exporter_info(self) -> Dict[str, Any]:
        """Get information about the MIDI exporter."""
        return {
            'ticks_per_beat': self.ticks_per_beat,
            'include_tempo_map': self.include_tempo_map,
            'include_lyrics': self.include_lyrics,
            'include_chords': self.include_chords,
            'include_melody': self.include_melody,
            'pretty_midi_available': PRETTY_MIDI_AVAILABLE
        }
