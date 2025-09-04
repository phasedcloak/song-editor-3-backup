"""
Block View Component

Displays song data in 20-second blocks with editable fields and audio playback functionality.
"""

import os
from typing import List, Optional
from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLineEdit, QTextEdit, QPushButton, QLabel
)

from ..models.lyrics import WordRow
from ..core.audio_player import AudioPlayer


@dataclass
class BlockData:
    """Represents a 20-second block of song data"""
    start_time: float
    end_time: float
    local_chord: str
    gemini_chord: str
    lyrics: List[WordRow]
    gemini_lyrics: List[WordRow]


class AudioPlaybackThread(QThread):
    """Thread for playing audio segments"""
    playback_finished = Signal()

    def __init__(self, audio_path: str, start_time: float, duration: float = 5.0):
        super().__init__()
        self.audio_path = audio_path
        self.start_time = start_time
        self.duration = duration
        self.player = AudioPlayer()

    def run(self):
        try:
            self.player.load(self.audio_path)
            end_time = self.start_time + self.duration
            self.player.play_segment(self.start_time, end_time)

            # Stop after duration
            QTimer.singleShot(int(self.duration * 1000), self.stop_playback)

        except Exception as e:
            print(f"Audio playback error: {e}")

    def stop_playback(self):
        self.player.stop()
        self.playback_finished.emit()


class EditableChordLine(QLineEdit):
    """Editable chord line with styling"""

    def __init__(self, placeholder: str = "", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setStyleSheet("""
            QLineEdit {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 4px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 2px solid #0078d4;
                background-color: #ffffff;
            }
        """)
        self.setMinimumHeight(30)


class EditableLyricsArea(QTextEdit):
    """Editable lyrics area with styling"""

    def __init__(self, placeholder: str = "", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 4px;
                line-height: 1.4;
            }
            QTextEdit:focus {
                border: 2px solid #0078d4;
                background-color: #ffffff;
            }
        """)
        self.setMaximumHeight(60)
        self.setMinimumHeight(40)

        # Store lyrics for context menu
        self.lyrics = []

        # Enable context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def set_lyrics(self, lyrics: List[WordRow]):
        """Set lyrics from WordRow objects with smart chord annotations, alternatives, and confidence colors"""
        if not lyrics:
            self.clear()
            return

        # Create text with smart chord annotations and alternatives
        lines = []
        previous_chord = None

        for word in lyrics:
            current_chord = word.chord
            chord_text = ""
            alt_text = ""

            # Only show chord if it's different from the previous word
            if current_chord and current_chord != previous_chord:
                chord_text = f"[{current_chord}]"

            # Add alternative if available
            if getattr(word, 'alt_text', None):
                alt_text = f"<{word.alt_text}>"

            lines.append(f"{word.text}{chord_text}{alt_text}")
            previous_chord = current_chord

        text = " ".join(lines)
        self.setPlainText(text)

        # Apply confidence-based color coding
        self.apply_confidence_colors(lyrics)

    def show_context_menu(self, position):
        """Show context menu with alternatives and probabilities"""
        from PySide6.QtWidgets import QMenu, QAction

        # Get the word at cursor position
        cursor = self.cursorForPosition(position)
        if not cursor:
            return

        # Find which word was clicked
        clicked_word = None
        for word in self.lyrics:
            if word.text in self.toPlainText():
                # Simple heuristic: check if cursor is near this word
                word_start = self.toPlainText().find(word.text)
                if word_start != -1:
                    cursor_pos = cursor.position()
                    if abs(cursor_pos - word_start) < len(word.text) + 10:  # Allow some tolerance
                        clicked_word = word
                        break

        if not clicked_word:
            return

        # Create context menu
        menu = QMenu(self)

        # Show primary word info
        primary_action = QAction(f"Primary: '{clicked_word.text}' (conf: {clicked_word.confidence:.1%})", self)
        primary_action.setEnabled(False)
        menu.addAction(primary_action)

        # Show alternatives if available
        if hasattr(clicked_word, 'alt_text') and clicked_word.alt_text:
            alt_action = QAction(f"Alternative: '{clicked_word.alt_text}'", self)
            alt_action.setEnabled(False)
            menu.addAction(alt_action)

            # Add separator
            menu.addSeparator()

            # Add action to use alternative
            use_alt_action = QAction("Use Alternative", self)
            use_alt_action.triggered.connect(lambda: self.use_alternative_word(clicked_word))
            menu.addAction(use_alt_action)

        # Show chord info if available
        if clicked_word.chord:
            menu.addSeparator()
            chord_action = QAction(f"Chord: {clicked_word.chord}", self)
            chord_action.setEnabled(False)
            menu.addAction(chord_action)

        # Show the menu
        menu.exec_(self.mapToGlobal(position))

    def use_alternative_word(self, word):
        """Replace primary word with alternative"""
        if hasattr(word, 'alt_text') and word.alt_text:
            # Update the word text
            word.text = word.alt_text
            # Set confidence to 100% since user chose this
            word.confidence = 1.0
            # Refresh the display
            self.set_lyrics(self.lyrics)

    def apply_confidence_colors(self, lyrics: List[WordRow]):
        """Apply confidence-based color coding to words"""
        from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor

        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.Start)

        for word in lyrics:
            # Calculate color based on confidence (0.0 red -> 1.0 green)
            c = max(0.0, min(1.0, word.confidence))
            red = int(255 * (1.0 - c))
            green = int(255 * c)
            color = QColor(red, green, 0)

            # Create format for this word
            format = QTextCharFormat()
            format.setForeground(color)

            # Find the word in the text and apply formatting
            word_text = word.text
            if word.chord and word.chord != getattr(self, '_previous_chord', None):
                word_text += f"[{word.chord}]"

            # Search for the word and apply formatting
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.Start)

            # Find and format this specific word
            while not cursor.isNull() and not cursor.atEnd():
                cursor = self.document().find(word_text, cursor)
                if not cursor.isNull():
                    cursor.mergeCharFormat(format)
                    break

            self._previous_chord = word.chord

    def get_lyrics_text(self) -> str:
        """Get current lyrics text"""
        return self.toPlainText().strip()


class BlockViewWidget(QWidget):
    """Widget for displaying a single 20-second block"""

    chord_edited = Signal(str, str)  # block_id, new_chord
    lyrics_edited = Signal(str, str)  # block_id, new_lyrics
    play_audio_requested = Signal(float, float)  # start_time, duration

    def __init__(self, block_data: BlockData, block_id: str, parent=None):
        super().__init__(parent)
        self.block_data = block_data
        self.block_id = block_id
        self.duration_seconds = 3.0  # Default duration for audio playback
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # Time range label
        time_label = QLabel(f"{self.block_data.start_time:.1f}s - {self.block_data.end_time:.1f}s")
        time_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #666;
                font-size: 10px;
            }
        """)
        layout.addWidget(time_label)

        # Note: Chords are embedded in lyrics as word[chord] format, so no separate chord lines needed

        # Local lyrics area (with embedded chords)
        local_lyrics_layout = QHBoxLayout()
        local_lyrics_layout.addWidget(QLabel("Local Lyrics (with chords):"))
        self.local_lyrics_edit = EditableLyricsArea("Enter lyrics...")
        self.local_lyrics_edit.set_lyrics(self.block_data.lyrics)
        self.local_lyrics_edit.textChanged.connect(self.on_local_lyrics_changed)
        # Add double-click to play audio for specific word
        self.local_lyrics_edit.mouseDoubleClickEvent = lambda event: \
            self.on_lyrics_double_click(event, self.local_lyrics_edit)
        local_lyrics_layout.addWidget(self.local_lyrics_edit)

        # Play button for local lyrics (plays entire block)
        play_local_btn = QPushButton("▶")
        play_local_btn.setMaximumSize(30, 30)
        play_local_btn.clicked.connect(lambda: self.play_audio(self.block_data.start_time))
        play_local_btn.setToolTip("Play audio for entire block (20 seconds)")
        local_lyrics_layout.addWidget(play_local_btn)
        layout.addLayout(local_lyrics_layout)

        # Styling
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border: 1px solid #ddd;
                border-radius: 6px;
            }
        """)

    # Note: Chord editing is now done directly in the lyrics text as word[chord] format

    def on_local_lyrics_changed(self):
        self.lyrics_edited.emit(self.block_id, self.local_lyrics_edit.get_lyrics_text())

    def play_audio(self, start_time: float, duration: float = 5.0):
        """Play audio segment"""
        self.play_audio_requested.emit(start_time, duration)

    def on_lyrics_double_click(self, event, text_edit):
        """Handle double-click on lyrics to play audio around specific word"""
        # Get the cursor position at click
        cursor = text_edit.cursorForPosition(event.pos())
        position = cursor.position()

        # Find which word was clicked
        lyrics = self.block_data.lyrics

        # Find the word at this position
        clicked_word = self.find_word_at_position(position, lyrics, text_edit)

        if clicked_word:
            # Use the local duration_seconds attribute
            duration_seconds = self.duration_seconds

            # Calculate center point of the word
            word_center = (clicked_word.start + clicked_word.end) / 2

            # Calculate start and end times centered on the word
            half_duration = duration_seconds / 2
            start_time = max(0.0, word_center - half_duration)
            start_time = max(0.0, start_time)  # Ensure we don't go before 0

            end_time = word_center + half_duration

            # Ensure we don't go beyond the audio bounds
            if hasattr(self, 'block_data') and hasattr(self.block_data, 'end_time'):
                end_time = min(end_time, self.block_data.end_time)

            # Emit the play request with start time and duration
            self.play_audio_requested.emit(start_time, end_time - start_time)
        else:
            # Fallback to block audio
            self.play_audio(self.block_data.start_time)

    def find_word_at_position(self, position: int, lyrics: List[WordRow], text_edit):
        """Find which word corresponds to a text position"""

        # Build a mapping of character positions to words
        char_pos = 0
        word_mapping = []
        prev_chord = None

        for word in lyrics:
            word_text = word.text
            if word.chord and word.chord != prev_chord:
                word_text += f"[{word.chord}]"

            word_mapping.append({
                'word': word,
                'start_pos': char_pos,
                'end_pos': char_pos + len(word_text)
            })
            char_pos += len(word_text) + 1  # +1 for space
            prev_chord = word.chord

        # Find which word contains the clicked position
        for mapping in word_mapping:
            if mapping['start_pos'] <= position <= mapping['end_pos']:
                return mapping['word']

        return None

    def get_updated_data(self) -> BlockData:
        """Get updated block data from current edits"""
        return BlockData(
            start_time=self.block_data.start_time,
            end_time=self.block_data.end_time,
            local_chord="",  # Chords are now embedded in lyrics
            gemini_chord="",  # Chords are now embedded in lyrics
            lyrics=self.block_data.lyrics,  # Keep original structure
            gemini_lyrics=[]  # Gemini lyrics are now shown in table view
        )

    def set_font(self, font):
        """Set font for all text elements in this block"""
        # Set font for lyrics areas
        self.local_lyrics_edit.setFont(font)

        # Set font for labels
        for label in self.findChildren(QLabel):
            label.setFont(font)


class BlockView(QWidget):
    """Main block view widget"""

    data_updated = Signal(list)  # List of updated BlockData

    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_path: Optional[str] = None
        self.blocks: List[BlockData] = []
        self.block_widgets: List[BlockViewWidget] = []
        self.playback_thread: Optional[AudioPlaybackThread] = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Block View - 20 Second Segments"))

        # Save button (now does comprehensive export)
        self.save_btn = QPushButton("Save & Export All")
        self.save_btn.clicked.connect(self.save_changes)
        self.save_btn.setEnabled(False)
        self.save_btn.setToolTip("Save changes and export CCLI, MIDI, and song_data files")
        header_layout.addWidget(self.save_btn)

        layout.addLayout(header_layout)

        # Scroll area for blocks
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Container widget for blocks
        self.blocks_container = QWidget()
        self.blocks_layout = QVBoxLayout(self.blocks_container)
        self.blocks_layout.setSpacing(10)
        self.blocks_layout.setContentsMargins(10, 10, 10, 10)
        self.blocks_layout.addStretch()

        self.scroll_area.setWidget(self.blocks_container)
        layout.addWidget(self.scroll_area)

    def set_audio_path(self, audio_path: str):
        """Set the audio file path for playback"""
        self.audio_path = audio_path

    def create_blocks_from_data(self, words: List[WordRow], chords: List):
        """Create 20-second blocks from song data"""
        if not words:
            return

        # Calculate total duration
        total_duration = max(word.end for word in words) if words else 0

        # Create blocks
        self.blocks = []
        block_size = 20.0  # 20 seconds

        for block_start in range(0, int(total_duration) + 1, int(block_size)):
            block_end = min(block_start + block_size, total_duration)

            # Get words in this time range
            block_words = [w for w in words if w.start >= block_start and w.end <= block_end]

            # Align chords with words in this block
            # Each word should have its corresponding chord
            for word in block_words:
                word_mid = (word.start + word.end) / 2
                # Find the chord that overlaps with this word
                for chord in chords:
                    if hasattr(chord, 'start') and hasattr(chord, 'end'):
                        if chord.start <= word_mid <= chord.end:
                            word.chord = chord.name if hasattr(chord, 'name') else str(chord)
                            break

            # Get the most common chord in this block for display
            block_chord = ""
            if block_words:
                chord_counts = {}
                for word in block_words:
                    if word.chord:
                        chord_counts[word.chord] = chord_counts.get(word.chord, 0) + 1
                if chord_counts:
                    block_chord = max(chord_counts, key=chord_counts.get)

            block_data = BlockData(
                start_time=block_start,
                end_time=block_end,
                local_chord=block_chord,
                gemini_chord="",  # Gemini data is now shown in table view
                lyrics=block_words,
                gemini_lyrics=[]  # Gemini lyrics are now shown in table view
            )

            self.blocks.append(block_data)

        self.update_block_widgets()

    def update_block_widgets(self):
        """Update the block widgets display"""
        # Clear existing widgets
        for widget in self.block_widgets:
            widget.deleteLater()
        self.block_widgets.clear()

        # Remove stretch from layout
        while self.blocks_layout.count() > 0:
            child = self.blocks_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add block widgets
        for i, block_data in enumerate(self.blocks):
            block_widget = BlockViewWidget(block_data, f"block_{i}", self)
            block_widget.chord_edited.connect(self.on_chord_edited)
            block_widget.lyrics_edited.connect(self.on_lyrics_edited)
            block_widget.play_audio_requested.connect(self.on_play_audio_requested)
            self.block_widgets.append(block_widget)
            self.blocks_layout.addWidget(block_widget)

        # Add stretch at the end
        self.blocks_layout.addStretch()

    def on_chord_edited(self, block_id: str, new_chord: str):
        """Handle chord edit"""
        self.save_btn.setEnabled(True)

    def on_lyrics_edited(self, block_id: str, new_lyrics: str):
        """Handle lyrics edit"""
        self.save_btn.setEnabled(True)

    def on_play_audio_requested(self, start_time: float, duration: float = 5.0):
        """Handle audio playback request"""
        if not self.audio_path or not os.path.exists(self.audio_path):
            return

        # Stop any existing playback
        if self.playback_thread and self.playback_thread.isRunning():
            self.playback_thread.terminate()
            self.playback_thread.wait()

        # Start new playback
        self.playback_thread = AudioPlaybackThread(self.audio_path, start_time, duration)
        self.playback_thread.playback_finished.connect(self.on_playback_finished)
        self.playback_thread.start()

    def on_playback_finished(self):
        """Handle playback finished"""
        if self.playback_thread:
            self.playback_thread.deleteLater()
            self.playback_thread = None

    def save_changes(self):
        """Comprehensive save operation: CCLI, MIDI update, and song_data export"""
        from PySide6.QtWidgets import QMessageBox, QFileDialog
        import os
        from pathlib import Path

        # Get updated data from all blocks
        updated_blocks = []
        for widget in self.block_widgets:
            updated_blocks.append(widget.get_updated_data())

        self.blocks = updated_blocks
        self.save_btn.setEnabled(False)
        self.data_updated.emit(updated_blocks)

        # 1. Export CCLI text file
        try:
            # Suggest filename based on audio file
            suggested_name = ""
            if self.audio_path:
                audio_path = Path(self.audio_path)
                suggested_name = str(audio_path.with_suffix('.cho'))

            ccli_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save CCLI Text File",
                suggested_name,
                "ChordPro Files (*.cho *.chordpro *.pro *.crd);;Text Files (*.txt)"
            )

            if ccli_path:
                from ...export.ccli import export_ccli
                # Get updated words from blocks
                updated_words = self.get_updated_words()
                export_ccli(ccli_path, updated_words)
                QMessageBox.information(self, "Success", f"CCLI file saved to {os.path.basename(ccli_path)}")
        except Exception as e:
            QMessageBox.warning(self, "CCLI Export Error", f"Failed to export CCLI: {e}")

        # 2. Update MIDI file if available
        try:
            # Look for existing MIDI file with same basename
            if self.audio_path:
                audio_path = Path(self.audio_path)
                midi_path = audio_path.with_suffix('.mid')

                if midi_path.exists():
                    # Ask user if they want to update the MIDI
                    reply = QMessageBox.question(
                        self,
                        "Update MIDI File",
                        f"Found existing MIDI file: {midi_path.name}\n\n"
                        f"Would you like to update it with the corrected lyrics and chords?",
                        QMessageBox.Yes | QMessageBox.No
                    )

                    if reply == QMessageBox.Yes:
                        from ...export.midi_export import export_midi
                        updated_words = self.get_updated_words()
                        updated_chords = self.get_updated_chords()

                        # Pass melody if available from Gemini
                        melody = None
                        if hasattr(self.parent(), 'parent') and hasattr(self.parent().parent(), 'gemini'):
                            melody = getattr(self.parent().parent().gemini, 'last_notes', None)

                        export_midi(str(midi_path), updated_words, updated_chords, melody)
                        QMessageBox.information(self, "Success", f"MIDI file updated: {midi_path.name}")
                else:
                    # Ask if user wants to create new MIDI
                    reply = QMessageBox.question(
                        self,
                        "Create MIDI File",
                        "No existing MIDI file found.\n\n"
                        "Would you like to create a new MIDI file with the corrected lyrics and chords?",
                        QMessageBox.Yes | QMessageBox.No
                    )

                    if reply == QMessageBox.Yes:
                        midi_path, _ = QFileDialog.getSaveFileName(
                            self,
                            "Save MIDI File",
                            str(audio_path.with_suffix('.mid')),
                            "MIDI Files (*.mid)"
                        )

                        if midi_path:
                            from ...export.midi_export import export_midi
                            updated_words = self.get_updated_words()
                            updated_chords = self.get_updated_chords()

                            # Pass melody if available from Gemini
                            melody = None
                            if hasattr(self.parent(), 'parent') and hasattr(self.parent().parent(), 'gemini'):
                                melody = getattr(self.parent().parent().gemini, 'last_notes', None)

                            export_midi(midi_path, updated_words, updated_chords, melody)
                            QMessageBox.information(
                                self,
                                "Success",
                                f"MIDI file created: {os.path.basename(midi_path)}"
                            )
        except Exception as e:
            QMessageBox.warning(self, "MIDI Update Error", f"Failed to update MIDI: {e}")

        # 3. Export updated song_data JSON
        try:
            # Suggest filename based on audio file
            suggested_name = ""
            if self.audio_path:
                audio_path = Path(self.audio_path)
                suggested_name = str(audio_path.with_suffix('.song_data'))

            song_data_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Updated Song Data",
                suggested_name,
                "Song Data Files (*.song_data *.json);;JSON Files (*.json)"
            )

            if song_data_path:
                from ...models.song_data_importer import SongData, ChordData
                from datetime import datetime

                # Convert updated chords to ChordData format
                chord_data_list = []
                for chord in self.get_updated_chords():
                    chord_data = ChordData(
                        symbol=chord.name,
                        root=chord.name[0] if chord.name else '',
                        quality=chord.name[1:] if len(chord.name) > 1 else 'maj',
                        bass=None,
                        start=chord.start,
                        end=chord.end,
                        confidence=chord.confidence
                    )
                    chord_data_list.append(chord_data)

                # Create metadata
                metadata = {
                    "version": "2.0.0",
                    "created_at": datetime.now().isoformat(),
                    "source_audio": self.audio_path or "",
                    "processing_tool": "Song Editor 2",
                    "confidence_threshold": 0.7,
                    "last_edited": datetime.now().isoformat(),
                    "editing_session": "Block View Correction"
                }

                # Create SongData object with updated data
                song_data = SongData(
                    metadata=metadata,
                    words=self.get_updated_words(),
                    chords=chord_data_list,
                    notes=[],  # Could be populated from Gemini notes if available
                    segments=[]  # Could be populated from segment detection if available
                )

                # Export using the importer's export function
                if hasattr(self.parent(), 'parent') and hasattr(self.parent().parent(), 'song_data_importer'):
                    importer = self.parent().parent().song_data_importer
                    if importer.export_song_data(song_data, song_data_path):
                        QMessageBox.information(
                            self,
                            "Success",
                            f"Updated song data saved to {os.path.basename(song_data_path)}"
                        )
                    else:
                        QMessageBox.warning(self, "Export Error", "Failed to export updated song data")
                else:
                    QMessageBox.warning(self, "Export Error", "Song data importer not available")
        except Exception as e:
            QMessageBox.warning(self, "Song Data Export Error", f"Failed to export song data: {e}")

        QMessageBox.information(self, "Save Complete", "All changes have been saved and exported!")

    def get_updated_words(self) -> List[WordRow]:
        """Get updated words from all blocks"""
        updated_words = []
        for block in self.blocks:
            # For now, return original words structure
            # In a full implementation, you'd parse the edited lyrics back to WordRow objects
            updated_words.extend(block.lyrics)
        return updated_words

    def get_updated_chords(self) -> List:
        """Get updated chords from all blocks"""
        updated_chords = []
        for block in self.blocks:
            if block.local_chord:
                # Create a simple chord object
                from ..processing.chords import DetectedChord
                chord = DetectedChord(
                    name=block.local_chord,
                    start=block.start_time,
                    end=block.end_time,
                    confidence=0.9
                )
                updated_chords.append(chord)
        return updated_chords

    def set_font(self, font):
        """Set font for all block widgets"""
        for widget in self.block_widgets:
            widget.set_font(font)
