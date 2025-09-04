#!/usr/bin/env python3
"""
Lyrics Editor UI

Lyrics editing interface for Song Editor 3.
"""

from typing import List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QTableWidget,
    QTableWidgetItem, QPushButton, QLabel, QDoubleSpinBox,
    QGroupBox, QGridLayout, QHeaderView, QMessageBox, QSplitter
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from ..models.song_data import SongData, Word


class LyricsEditor(QWidget):
    """Lyrics editing interface."""

    lyrics_changed = Signal(list)

    def __init__(self):
        super().__init__()
        self.song_data = None
        self.words = []
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Create splitter for text and table views
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Text view
        left_panel = self.create_text_panel()
        splitter.addWidget(left_panel)

        # Right panel - Table view
        right_panel = self.create_table_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([400, 600])

        # Bottom panel - Controls
        bottom_panel = self.create_control_panel()
        layout.addWidget(bottom_panel)

    def create_text_panel(self):
        """Create the text editing panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Text editor
        layout.addWidget(QLabel("Lyrics Text:"))
        self.text_editor = QTextEdit()
        self.text_editor.setFont(QFont("Courier", 12))
        self.text_editor.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.text_editor)

        # Word count
        self.word_count_label = QLabel("Words: 0")
        layout.addWidget(self.word_count_label)

        return panel

    def create_table_panel(self):
        """Create the table editing panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Table
        layout.addWidget(QLabel("Word Details:"))
        self.word_table = QTableWidget()
        self.word_table.setColumnCount(5)
        self.word_table.setHorizontalHeaderLabels([
            "Text", "Start Time", "End Time", "Confidence", "Chord"
        ])

        # Set table properties
        header = self.word_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)

        self.word_table.itemChanged.connect(self.on_table_item_changed)
        layout.addWidget(self.word_table)

        return panel

    def create_control_panel(self):
        """Create the control panel."""
        panel = QWidget()
        layout = QHBoxLayout(panel)

        # Timing controls
        timing_group = QGroupBox("Timing")
        timing_layout = QGridLayout(timing_group)

        self.start_time_spin = QDoubleSpinBox()
        self.start_time_spin.setRange(0, 9999)
        self.start_time_spin.setDecimals(3)
        self.start_time_spin.setSuffix(" s")
        timing_layout.addWidget(QLabel("Start:"), 0, 0)
        timing_layout.addWidget(self.start_time_spin, 0, 1)

        self.end_time_spin = QDoubleSpinBox()
        self.end_time_spin.setRange(0, 9999)
        self.end_time_spin.setDecimals(3)
        self.end_time_spin.setSuffix(" s")
        timing_layout.addWidget(QLabel("End:"), 0, 2)
        timing_layout.addWidget(self.end_time_spin, 0, 3)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0, 1)
        self.confidence_spin.setDecimals(3)
        self.confidence_spin.setSingleStep(0.1)
        timing_layout.addWidget(QLabel("Confidence:"), 1, 0)
        timing_layout.addWidget(self.confidence_spin, 1, 1)

        layout.addWidget(timing_group)

        # Action buttons
        button_group = QGroupBox("Actions")
        button_layout = QHBoxLayout(button_group)

        self.add_word_btn = QPushButton("Add Word")
        self.add_word_btn.clicked.connect(self.add_word)
        button_layout.addWidget(self.add_word_btn)

        self.delete_word_btn = QPushButton("Delete Word")
        self.delete_word_btn.clicked.connect(self.delete_word)
        button_layout.addWidget(self.delete_word_btn)

        self.move_up_btn = QPushButton("Move Up")
        self.move_up_btn.clicked.connect(self.move_word_up)
        button_layout.addWidget(self.move_up_btn)

        self.move_down_btn = QPushButton("Move Down")
        self.move_down_btn.clicked.connect(self.move_word_down)
        button_layout.addWidget(self.move_down_btn)

        layout.addWidget(button_group)

        # Analysis controls
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)

        self.auto_align_btn = QPushButton("Auto-Align Timing")
        self.auto_align_btn.clicked.connect(self.auto_align_timing)
        analysis_layout.addWidget(self.auto_align_btn)

        self.fix_confidence_btn = QPushButton("Fix Low Confidence")
        self.fix_confidence_btn.clicked.connect(self.fix_low_confidence)
        analysis_layout.addWidget(self.fix_confidence_btn)

        layout.addWidget(analysis_group)

        layout.addStretch()

        return panel

    def set_song_data(self, song_data: SongData):
        """Set the song data to edit."""
        self.song_data = song_data
        self.words = song_data.words.copy()
        self.update_display()

    def update_display(self):
        """Update the display with current word data."""
        # Update text editor
        text = ' '.join(word.text for word in self.words)
        self.text_editor.blockSignals(True)
        self.text_editor.setPlainText(text)
        self.text_editor.blockSignals(False)

        # Update word count
        self.word_count_label.setText(f"Words: {len(self.words)}")

        # Update table
        self.update_table()

    def update_table(self):
        """Update the word table."""
        self.word_table.blockSignals(True)
        self.word_table.setRowCount(len(self.words))

        for row, word in enumerate(self.words):
            # Text
            text_item = QTableWidgetItem(word.text)
            self.word_table.setItem(row, 0, text_item)

            # Start time
            start_item = QTableWidgetItem(f"{word.start:.3f}")
            self.word_table.setItem(row, 1, start_item)

            # End time
            end_item = QTableWidgetItem(f"{word.end:.3f}")
            self.word_table.setItem(row, 2, end_item)

            # Confidence
            conf_item = QTableWidgetItem(f"{word.confidence:.3f}")
            self.word_table.setItem(row, 3, conf_item)

            # Chord
            chord_item = QTableWidgetItem(word.chord or "")
            self.word_table.setItem(row, 4, chord_item)

        self.word_table.blockSignals(False)

    def on_text_changed(self):
        """Handle text editor changes."""
        text = self.text_editor.toPlainText()
        new_words = text.split()

        # Update word texts while preserving timing
        for i, word in enumerate(self.words):
            if i < len(new_words):
                word.text = new_words[i]
            else:
                # Remove extra words
                self.words = self.words[:len(new_words)]
                break

        # Add new words if needed
        while len(self.words) < len(new_words):
            # Create new word with default timing
            start_time = len(self.words) * 0.5  # Default 0.5s per word
            end_time = start_time + 0.5
            new_word = Word(
                text=new_words[len(self.words)],
                start=start_time,
                end=end_time,
                confidence=0.5
            )
            self.words.append(new_word)

        self.update_table()
        self.word_count_label.setText(f"Words: {len(self.words)}")
        self.lyrics_changed.emit(self.words)

    def on_table_item_changed(self, item):
        """Handle table item changes."""
        row = item.row()
        col = item.column()

        if row >= len(self.words):
            return

        word = self.words[row]

        try:
            if col == 0:  # Text
                word.text = item.text()
            elif col == 1:  # Start time
                word.start = float(item.text())
            elif col == 2:  # End time
                word.end = float(item.text())
            elif col == 3:  # Confidence
                word.confidence = float(item.text())
            elif col == 4:  # Chord
                word.chord = item.text() if item.text() else None

            # Update text editor
            self.text_editor.blockSignals(True)
            text = ' '.join(w.text for w in self.words)
            self.text_editor.setPlainText(text)
            self.text_editor.blockSignals(False)

            self.lyrics_changed.emit(self.words)

        except ValueError:
            # Revert invalid input
            self.update_table()
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")

    def add_word(self):
        """Add a new word."""
        # Get current selection
        current_row = self.word_table.currentRow()
        if current_row < 0:
            current_row = len(self.words)

        # Create new word
        start_time = current_row * 0.5
        end_time = start_time + 0.5
        new_word = Word(
            text="new_word",
            start=start_time,
            end=end_time,
            confidence=0.5
        )

        # Insert word
        self.words.insert(current_row, new_word)
        self.update_display()

        # Select the new word
        self.word_table.selectRow(current_row)
        self.word_table.setFocus()

        self.lyrics_changed.emit(self.words)

    def delete_word(self):
        """Delete the selected word."""
        current_row = self.word_table.currentRow()
        if current_row >= 0 and current_row < len(self.words):
            del self.words[current_row]
            self.update_display()
            self.lyrics_changed.emit(self.words)

    def move_word_up(self):
        """Move the selected word up."""
        current_row = self.word_table.currentRow()
        if current_row > 0:
            self.words[current_row], self.words[current_row - 1] = \
                self.words[current_row - 1], self.words[current_row]
            self.update_display()
            self.word_table.selectRow(current_row - 1)
            self.lyrics_changed.emit(self.words)

    def move_word_down(self):
        """Move the selected word down."""
        current_row = self.word_table.currentRow()
        if current_row < len(self.words) - 1:
            self.words[current_row], self.words[current_row + 1] = \
                self.words[current_row + 1], self.words[current_row]
            self.update_display()
            self.word_table.selectRow(current_row + 1)
            self.lyrics_changed.emit(self.words)

    def auto_align_timing(self):
        """Automatically align word timing."""
        if not self.words:
            return

        # Simple linear alignment
        total_duration = self.song_data.get_duration() if self.song_data else 60.0
        word_duration = total_duration / len(self.words)

        for i, word in enumerate(self.words):
            word.start = i * word_duration
            word.end = (i + 1) * word_duration

        self.update_table()
        self.lyrics_changed.emit(self.words)

        QMessageBox.information(
            self,
            "Timing Aligned",
            f"Aligned {len(self.words)} words over {total_duration:.1f} seconds."
        )

    def fix_low_confidence(self):
        """Fix words with low confidence."""
        fixed_count = 0
        threshold = 0.3

        for word in self.words:
            if word.confidence < threshold:
                word.confidence = threshold
                fixed_count += 1

        if fixed_count > 0:
            self.update_table()
            self.lyrics_changed.emit(self.words)

            QMessageBox.information(
                self,
                "Confidence Fixed",
                f"Fixed confidence for {fixed_count} words."
            )
        else:
            QMessageBox.information(
                self,
                "No Fixes Needed",
                "All words have acceptable confidence levels."
            )

    def get_words(self) -> List[Word]:
        """Get the current word list."""
        return self.words.copy()

    def set_words(self, words: List[Word]):
        """Set the word list."""
        self.words = words.copy()
        self.update_display()

    def export_lyrics_text(self) -> str:
        """Export lyrics as plain text."""
        return ' '.join(word.text for word in self.words)

    def import_lyrics_text(self, text: str):
        """Import lyrics from plain text."""
        words = text.split()

        # Create word objects with default timing
        self.words = []
        for i, word_text in enumerate(words):
            start_time = i * 0.5
            end_time = start_time + 0.5
            word = Word(
                text=word_text,
                start=start_time,
                end=end_time,
                confidence=0.5
            )
            self.words.append(word)

        self.update_display()
        self.lyrics_changed.emit(self.words)
