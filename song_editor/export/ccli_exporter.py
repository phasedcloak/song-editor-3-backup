#!/usr/bin/env python3
"""
CCLI Exporter Module

Handles export of song data to CCLI-compatible chord/lyrics text format
for Song Editor 3.
"""

import logging
import string
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class CCLIExporter:
    """Handles export of song data to CCLI-compatible text format."""

    def __init__(
        self,
        include_metadata: bool = True,
        include_timing: bool = False,
        chord_format: str = "brackets",  # "brackets", "inline", "separate"
        remove_punctuation: bool = True,
        max_line_length: int = 80
    ):
        self.include_metadata = include_metadata
        self.include_timing = include_timing
        self.chord_format = chord_format
        self.remove_punctuation = remove_punctuation
        self.max_line_length = max_line_length

    def _remove_punctuation_for_ccli(self, text: str) -> str:
        """Remove punctuation from text for CCLI compatibility."""
        if not text:
            return ""

        # Convert to string if needed
        text = str(text)

        # Define punctuation to remove (but keep apostrophes in contractions)
        punctuation_to_remove = string.punctuation.replace("'", "")  # Keep apostrophes

        # Remove punctuation except apostrophes
        cleaned_text = ""
        for char in text:
            if char not in punctuation_to_remove:
                cleaned_text += char
            elif char in ".,!?;:":  # Replace common punctuation with spaces
                cleaned_text += " "

        # Clean up multiple spaces
        while "  " in cleaned_text:
            cleaned_text = cleaned_text.replace("  ", " ")

        # Strip leading/trailing whitespace
        cleaned_text = cleaned_text.strip()

        return cleaned_text

    def _format_chord_brackets(self, chord: str, word: str) -> str:
        """Format chord and word with brackets: [C]word"""
        return f"[{chord}]{word}"

    def _format_chord_inline(self, chord: str, word: str) -> str:
        """Format chord and word inline: Cword"""
        return f"{chord}{word}"

    def _format_chord_separate(self, chord: str, word: str) -> str:
        """Format chord and word separately: C word"""
        return f"{chord} {word}"

    def _find_chord_for_word(self, word: Dict[str, Any], chords: List[Dict[str, Any]]) -> Optional[str]:
        """Find the chord that corresponds to a word based on timing."""
        word_start = word.get('start', 0)
        word_end = word.get('end', word_start + 0.5)

        # Find chord that overlaps with word timing
        for chord in chords:
            chord_start = chord.get('start', 0)
            chord_end = chord.get('end', chord_start + 1.0)

            # Check for overlap
            if (chord_start <= word_end and chord_end >= word_start):
                return chord.get('symbol', '')

        return None

    def _create_metadata_section(self, song_data: Dict[str, Any]) -> str:
        """Create metadata section for the CCLI file."""
        metadata_lines = []

        # Song title (from filename or metadata)
        source_audio = song_data.get('metadata', {}).get('source_audio', '')
        if source_audio:
            title = Path(source_audio).stem
            metadata_lines.append(f"Title: {title}")

        # Key information
        key_info = song_data.get('audio_analysis', {}).get('key', {})
        if key_info and key_info.get('root') != 'Unknown':
            key = key_info.get('root', '')
            mode = key_info.get('mode', 'major')
            confidence = key_info.get('confidence', 0.0)
            metadata_lines.append(f"Key: {key} {mode}")
            if confidence > 0:
                metadata_lines.append(f"Key Confidence: {confidence:.1%}")

        # Tempo information
        tempo = song_data.get('audio_analysis', {}).get('tempo')
        if tempo:
            metadata_lines.append(f"Tempo: {tempo:.1f} BPM")

        # Duration
        duration = song_data.get('audio_analysis', {}).get('duration')
        if duration:
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            metadata_lines.append(f"Duration: {minutes}:{seconds:02d}")

        # Processing information
        metadata = song_data.get('metadata', {})
        if metadata:
            created_at = metadata.get('created_at', '')
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    formatted_date = dt.strftime('%Y-%m-%d %H:%M:%S')
                    metadata_lines.append(f"Processed: {formatted_date}")
                except (ValueError, TypeError):
                    pass

            transcription_engine = metadata.get('transcription', {}).get('engine', '')
            if transcription_engine:
                metadata_lines.append(f"Transcription: {transcription_engine}")

        if metadata_lines:
            return "\n".join(metadata_lines) + "\n"
        else:
            return ""

    def _create_chord_lyrics_table(self, song_data: Dict[str, Any]) -> str:
        """Create chord and lyrics alignment table."""
        words = song_data.get('words', [])
        chords = song_data.get('chords', [])

        if not words:
            return ""

        table_lines = []
        table_lines.append("CHORD AND LYRICS TIME ALIGNMENT TABLE")
        table_lines.append("=" * 70)
        table_lines.append("")

        # Header
        if self.include_timing:
            table_lines.append(f"{'Time':<8} {'Chord':<15} {'Duration':<8} {'Lyrics':<40}")
            table_lines.append("-" * 71)
        else:
            table_lines.append(f"{'Chord':<15} {'Lyrics':<40}")
            table_lines.append("-" * 55)

        # Create combined timeline
        events = []

        # Add chord events
        for chord in chords:
            events.append(('chord', chord.get('start', 0), chord))

        # Add lyric events
        for word in words:
            events.append(('lyrics', word.get('start', 0), word))

        # Sort by time
        events.sort(key=lambda x: x[1])

        # Write events
        for event_type, time, content in events:
            if event_type == 'chord':
                chord_symbol = content.get('symbol', '')
                duration = content.get('duration', 0)

                if self.include_timing:
                    minutes = int(time // 60)
                    seconds = time % 60
                    time_str = f"{minutes}:{seconds:05.2f}"
                    duration_str = f"{duration:.2f}s" if duration > 0 else "N/A"
                    table_lines.append(f"{time_str:<8} {chord_symbol:<15} {duration_str:<8} {'':<40}")
                else:
                    table_lines.append(f"{chord_symbol:<15} {'':<40}")
            else:  # lyrics
                word_text = content.get('text', '')
                if self.include_timing:
                    minutes = int(time // 60)
                    seconds = time % 60
                    time_str = f"{minutes}:{seconds:05.2f}"
                    table_lines.append(f"{time_str:<8} {'':<15} {'':<8} {word_text:<40}")
                else:
                    table_lines.append(f"{'':<15} {word_text:<40}")

        return "\n".join(table_lines)

    def _create_lyrics_only_section(self, song_data: Dict[str, Any]) -> str:
        """Create lyrics-only section without chords."""
        words = song_data.get('words', [])

        if not words:
            return ""

        lyrics_lines = []
        lyrics_lines.append("LYRICS ONLY")
        lyrics_lines.append("=" * 50)
        lyrics_lines.append("")

        current_line = ""
        for word in words:
            word_text = word.get('text', '')

            # Clean text if requested
            if self.remove_punctuation:
                word_text = self._remove_punctuation_for_ccli(word_text)

            # Check if adding this word would exceed line length
            if current_line and len(current_line + " " + word_text) > self.max_line_length:
                lyrics_lines.append(current_line)
                current_line = word_text
            else:
                if current_line:
                    current_line += " " + word_text
                else:
                    current_line = word_text

        # Add the last line
        if current_line:
            lyrics_lines.append(current_line)

        return "\n".join(lyrics_lines)

    def _create_chordpro_section(self, song_data: Dict[str, Any]) -> str:
        """Create ChordPro format section with inline chords."""
        words = song_data.get('words', [])
        chords = song_data.get('chords', [])

        if not words:
            return ""

        chordpro_lines = []
        chordpro_lines.append("CHORDPRO FORMAT")
        chordpro_lines.append("=" * 50)
        chordpro_lines.append("")

        current_line = ""
        for word in words:
            word_text = word.get('text', '')

            # Clean text if requested
            if self.remove_punctuation:
                word_text = self._remove_punctuation_for_ccli(word_text)

            # Find chord for this word
            chord_symbol = self._find_chord_for_word(word, chords)

            # Format chord and word
            if chord_symbol:
                if self.chord_format == "brackets":
                    formatted_word = self._format_chord_brackets(chord_symbol, word_text)
                elif self.chord_format == "inline":
                    formatted_word = self._format_chord_inline(chord_symbol, word_text)
                else:  # separate
                    formatted_word = self._format_chord_separate(chord_symbol, word_text)
            else:
                formatted_word = word_text

            # Check line length
            if current_line and len(current_line + " " + formatted_word) > self.max_line_length:
                chordpro_lines.append(current_line)
                current_line = formatted_word
            else:
                if current_line:
                    current_line += " " + formatted_word
                else:
                    current_line = formatted_word

        # Add the last line
        if current_line:
            chordpro_lines.append(current_line)

        return "\n".join(chordpro_lines)

    def _create_chord_chart_section(self, song_data: Dict[str, Any]) -> str:
        """Create chord chart section showing chord progression."""
        chords = song_data.get('chords', [])

        if not chords:
            return ""

        chart_lines = []
        chart_lines.append("CHORD CHART")
        chart_lines.append("=" * 50)
        chart_lines.append("")

        # Group chords by time sections
        sections = []
        current_section = []
        current_time = 0

        for chord in chords:
            chord_time = chord.get('start', 0)

            # Start new section if there's a significant gap
            if chord_time - current_time > 2.0 and current_section:
                sections.append(current_section)
                current_section = []

            current_section.append(chord)
            current_time = chord_time

        # Add the last section
        if current_section:
            sections.append(current_section)

        # Format sections
        for i, section in enumerate(sections):
            chart_lines.append(f"Section {i+1}:")

            # Get unique chords in this section
            unique_chords = []
            for chord in section:
                chord_symbol = chord.get('symbol', '')
                if chord_symbol not in unique_chords:
                    unique_chords.append(chord_symbol)

            # Format chord progression
            if unique_chords:
                progression = " | ".join(unique_chords)
                chart_lines.append(f"  {progression}")

            chart_lines.append("")

        return "\n".join(chart_lines)

    def export(self, song_data: Dict[str, Any], output_path: str) -> bool:
        """Export song data to CCLI-compatible text file."""
        try:
            logging.info(f"Exporting CCLI text to: {output_path}")

            # Create output content
            content_sections = []

            # Add metadata section
            if self.include_metadata:
                metadata = self._create_metadata_section(song_data)
                if metadata:
                    content_sections.append(metadata)

            # Add chord/lyrics table
            table = self._create_chord_lyrics_table(song_data)
            if table:
                content_sections.append(table)
                content_sections.append("")  # Empty line

            # Add ChordPro format
            chordpro = self._create_chordpro_section(song_data)
            if chordpro:
                content_sections.append(chordpro)
                content_sections.append("")  # Empty line

            # Add chord chart
            chart = self._create_chord_chart_section(song_data)
            if chart:
                content_sections.append(chart)
                content_sections.append("")  # Empty line

            # Add lyrics only section
            lyrics = self._create_lyrics_only_section(song_data)
            if lyrics:
                content_sections.append(lyrics)

            # Combine all sections
            content = "\n".join(content_sections)

            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

            logging.info(f"CCLI text exported successfully: {len(content)} characters")
            return True

        except Exception as e:
            logging.error(f"Error exporting CCLI text: {e}")
            return False

    def export_lyrics_only(self, song_data: Dict[str, Any], output_path: str) -> bool:
        """Export lyrics-only version for CCLI submission."""
        try:
            logging.info(f"Exporting lyrics-only to: {output_path}")

            words = song_data.get('words', [])
            if not words:
                logging.warning("No words found for lyrics-only export")
                return False

            # Create lyrics content
            lyrics_lines = []

            # Add metadata
            metadata = self._create_metadata_section(song_data)
            if metadata:
                lyrics_lines.append(metadata)

            # Add lyrics
            lyrics = self._create_lyrics_only_section(song_data)
            if lyrics:
                lyrics_lines.append(lyrics)

            # Combine content
            content = "\n".join(lyrics_lines)

            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

            logging.info(f"Lyrics-only exported successfully: {len(content)} characters")
            return True

        except Exception as e:
            logging.error(f"Error exporting lyrics-only: {e}")
            return False

    def export_chordpro(self, song_data: Dict[str, Any], output_path: str) -> bool:
        """Export ChordPro format file."""
        try:
            logging.info(f"Exporting ChordPro to: {output_path}")

            # Create ChordPro content
            content_sections = []

            # Add metadata
            metadata = self._create_metadata_section(song_data)
            if metadata:
                content_sections.append(metadata)

            # Add ChordPro format
            chordpro = self._create_chordpro_section(song_data)
            if chordpro:
                content_sections.append(chordpro)

            # Combine content
            content = "\n".join(content_sections)

            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

            logging.info(f"ChordPro exported successfully: {len(content)} characters")
            return True

        except Exception as e:
            logging.error(f"Error exporting ChordPro: {e}")
            return False

    def get_exporter_info(self) -> Dict[str, Any]:
        """Get information about the CCLI exporter."""
        return {
            'include_metadata': self.include_metadata,
            'include_timing': self.include_timing,
            'chord_format': self.chord_format,
            'remove_punctuation': self.remove_punctuation,
            'max_line_length': self.max_line_length
        }
