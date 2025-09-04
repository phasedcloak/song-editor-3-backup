#!/usr/bin/env python3
"""
Melody Editor UI

Melody editing interface for Song Editor 3.
"""

from typing import List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QPushButton, QLabel, QDoubleSpinBox,
    QGroupBox, QGridLayout, QHeaderView, QMessageBox, QSpinBox,
    QSplitter, QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QFont

from ..models.song_data import SongData, Note


class MelodyEditor(QWidget):
    """Melody editing interface."""

    melody_changed = Signal(list)

    def __init__(self):
        super().__init__()
        self.song_data = None
        self.notes = []
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Create splitter for different views
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Melody visualization and list
        left_panel = self.create_melody_panel()
        splitter.addWidget(left_panel)

        # Right panel - Detailed table view
        right_panel = self.create_table_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([400, 600])

        # Bottom panel - Controls
        bottom_panel = self.create_control_panel()
        layout.addWidget(bottom_panel)

    def create_melody_panel(self):
        """Create the melody visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Melody visualization
        layout.addWidget(QLabel("Melody Visualization:"))
        self.melody_widget = MelodyVisualizationWidget()
        layout.addWidget(self.melody_widget)

        # Note list
        layout.addWidget(QLabel("Notes:"))
        self.note_list = QListWidget()
        self.note_list.itemClicked.connect(self.on_note_item_clicked)
        layout.addWidget(self.note_list)

        # Melody statistics
        stats_group = QGroupBox("Melody Statistics")
        stats_layout = QGridLayout(stats_group)

        self.total_notes_label = QLabel("0")
        stats_layout.addWidget(QLabel("Total Notes:"), 0, 0)
        stats_layout.addWidget(self.total_notes_label, 0, 1)

        self.range_label = QLabel("0")
        stats_layout.addWidget(QLabel("Range:"), 1, 0)
        stats_layout.addWidget(self.range_label, 1, 1)

        self.avg_duration_label = QLabel("0.0s")
        stats_layout.addWidget(QLabel("Avg Duration:"), 2, 0)
        stats_layout.addWidget(self.avg_duration_label, 2, 1)

        layout.addWidget(stats_group)

        return panel

    def create_table_panel(self):
        """Create the detailed table panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Table
        layout.addWidget(QLabel("Note Details:"))
        self.note_table = QTableWidget()
        self.note_table.setColumnCount(8)
        self.note_table.setHorizontalHeaderLabels([
            "Pitch", "Note Name", "Start Time", "End Time", "Duration", "Velocity", "Confidence", "Method"
        ])

        # Set table properties
        header = self.note_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.Stretch)

        self.note_table.itemChanged.connect(self.on_table_item_changed)
        self.note_table.itemSelectionChanged.connect(self.on_table_selection_changed)
        layout.addWidget(self.note_table)

        return panel

    def create_control_panel(self):
        """Create the control panel."""
        panel = QWidget()
        layout = QHBoxLayout(panel)

        # Note input controls
        note_group = QGroupBox("Note Input")
        note_layout = QGridLayout(note_group)

        # Pitch (MIDI)
        self.pitch_spin = QSpinBox()
        self.pitch_spin.setRange(0, 127)
        self.pitch_spin.setValue(60)  # Middle C
        note_layout.addWidget(QLabel("MIDI Pitch:"), 0, 0)
        note_layout.addWidget(self.pitch_spin, 0, 1)

        # Note name display
        self.note_name_label = QLabel("C")
        note_layout.addWidget(QLabel("Note Name:"), 0, 2)
        note_layout.addWidget(self.note_name_label, 0, 3)

        # Velocity
        self.velocity_spin = QSpinBox()
        self.velocity_spin.setRange(1, 127)
        self.velocity_spin.setValue(80)
        note_layout.addWidget(QLabel("Velocity:"), 1, 0)
        note_layout.addWidget(self.velocity_spin, 1, 1)

        # Connect signals for note name update
        self.pitch_spin.valueChanged.connect(self.update_note_name)

        layout.addWidget(note_group)

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

        layout.addWidget(timing_group)

        # Action buttons
        button_group = QGroupBox("Actions")
        button_layout = QHBoxLayout(button_group)

        self.add_note_btn = QPushButton("Add Note")
        self.add_note_btn.clicked.connect(self.add_note)
        button_layout.addWidget(self.add_note_btn)

        self.delete_note_btn = QPushButton("Delete Note")
        self.delete_note_btn.clicked.connect(self.delete_note)
        button_layout.addWidget(self.delete_note_btn)

        self.move_up_btn = QPushButton("Move Up")
        self.move_up_btn.clicked.connect(self.move_note_up)
        button_layout.addWidget(self.move_up_btn)

        self.move_down_btn = QPushButton("Move Down")
        self.move_down_btn.clicked.connect(self.move_note_down)
        button_layout.addWidget(self.move_down_btn)

        layout.addWidget(button_group)

        # Analysis controls
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)

        self.filter_short_btn = QPushButton("Filter Short Notes")
        self.filter_short_btn.clicked.connect(self.filter_short_notes)
        analysis_layout.addWidget(self.filter_short_btn)

        self.merge_similar_btn = QPushButton("Merge Similar")
        self.merge_similar_btn.clicked.connect(self.merge_similar_notes)
        analysis_layout.addWidget(self.merge_similar_btn)

        layout.addWidget(analysis_group)

        layout.addStretch()

        return panel

    def set_song_data(self, song_data: SongData):
        """Set the song data to edit."""
        self.song_data = song_data
        self.notes = song_data.notes.copy()
        self.update_display()

    def update_display(self):
        """Update the display with current note data."""
        self.update_note_list()
        self.update_table()
        self.update_statistics()
        self.melody_widget.set_notes(self.notes)

    def update_note_list(self):
        """Update the note list."""
        self.note_list.clear()

        for note in self.notes:
            note_name = self.midi_to_note_name(note.pitch_midi)
            duration = note.end - note.start
            item_text = f"{note_name} ({note.pitch_midi}) - {duration:.2f}s"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, note)
            self.note_list.addItem(item)

    def update_table(self):
        """Update the note table."""
        self.note_table.blockSignals(True)
        self.note_table.setRowCount(len(self.notes))

        for row, note in enumerate(self.notes):
            # MIDI pitch
            pitch_item = QTableWidgetItem(str(note.pitch_midi))
            self.note_table.setItem(row, 0, pitch_item)

            # Note name
            note_name = self.midi_to_note_name(note.pitch_midi)
            name_item = QTableWidgetItem(note_name)
            self.note_table.setItem(row, 1, name_item)

            # Start time
            start_item = QTableWidgetItem(f"{note.start:.3f}")
            self.note_table.setItem(row, 2, start_item)

            # End time
            end_item = QTableWidgetItem(f"{note.end:.3f}")
            self.note_table.setItem(row, 3, end_item)

            # Duration
            duration = note.end - note.start
            duration_item = QTableWidgetItem(f"{duration:.3f}")
            self.note_table.setItem(row, 4, duration_item)

            # Velocity
            velocity = note.velocity or 80
            velocity_item = QTableWidgetItem(str(velocity))
            self.note_table.setItem(row, 5, velocity_item)

            # Confidence
            confidence = note.confidence or 0.5
            conf_item = QTableWidgetItem(f"{confidence:.3f}")
            self.note_table.setItem(row, 6, conf_item)

            # Detection method
            method_item = QTableWidgetItem(note.detection_method or "")
            self.note_table.setItem(row, 7, method_item)

        self.note_table.blockSignals(False)

    def update_statistics(self):
        """Update the statistics display."""
        total_notes = len(self.notes)

        if total_notes > 0:
            pitches = [note.pitch_midi for note in self.notes]
            pitch_range = max(pitches) - min(pitches)
            avg_duration = sum(note.end - note.start for note in self.notes) / total_notes
        else:
            pitch_range = 0
            avg_duration = 0.0

        self.total_notes_label.setText(str(total_notes))
        self.range_label.setText(str(pitch_range))
        self.avg_duration_label.setText(f"{avg_duration:.2f}s")

    def update_note_name(self):
        """Update the note name display."""
        pitch = self.pitch_spin.value()
        note_name = self.midi_to_note_name(pitch)
        self.note_name_label.setText(note_name)

    def midi_to_note_name(self, midi_pitch: int) -> str:
        """Convert MIDI pitch to note name."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_pitch // 12) - 1
        note_name = note_names[midi_pitch % 12]
        return f"{note_name}{octave}"

    def on_note_item_clicked(self, item):
        """Handle note list item click."""
        note = item.data(Qt.UserRole)
        if note:
            # Find the note in the table and select it
            for row in range(self.note_table.rowCount()):
                if self.notes[row] == note:
                    self.note_table.selectRow(row)
                    self.note_table.scrollToItem(self.note_table.item(row, 0))
                    break

    def on_table_item_changed(self, item):
        """Handle table item changes."""
        row = item.row()
        col = item.column()

        if row >= len(self.notes):
            return

        note = self.notes[row]

        try:
            if col == 0:  # MIDI pitch
                note.pitch_midi = int(item.text())
                note.pitch_name = self.midi_to_note_name(note.pitch_midi)
            elif col == 2:  # Start time
                note.start = float(item.text())
            elif col == 3:  # End time
                note.end = float(item.text())
            elif col == 5:  # Velocity
                note.velocity = int(item.text())
            elif col == 6:  # Confidence
                note.confidence = float(item.text())
            elif col == 7:  # Detection method
                note.detection_method = item.text()

            # Update duration column
            duration = note.end - note.start
            duration_item = QTableWidgetItem(f"{duration:.3f}")
            self.note_table.setItem(row, 4, duration_item)

            # Update note name column
            note_name = self.midi_to_note_name(note.pitch_midi)
            name_item = QTableWidgetItem(note_name)
            self.note_table.setItem(row, 1, name_item)

            self.update_note_list()
            self.update_statistics()
            self.melody_widget.set_notes(self.notes)
            self.melody_changed.emit(self.notes)

        except ValueError:
            # Revert invalid input
            self.update_table()
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")

    def on_table_selection_changed(self):
        """Handle table selection changes."""
        current_row = self.note_table.currentRow()
        if current_row >= 0 and current_row < len(self.notes):
            note = self.notes[current_row]

            # Update timing controls
            self.start_time_spin.setValue(note.start)
            self.end_time_spin.setValue(note.end)

            # Update note input controls
            self.pitch_spin.setValue(note.pitch_midi)
            self.velocity_spin.setValue(note.velocity or 80)

    def add_note(self):
        """Add a new note."""
        # Get current selection
        current_row = self.note_table.currentRow()
        if current_row < 0:
            current_row = len(self.notes)

        # Get note details from controls
        pitch = self.pitch_spin.value()
        velocity = self.velocity_spin.value()
        start_time = self.start_time_spin.value()
        end_time = self.end_time_spin.value()

        # Create new note
        new_note = Note(
            pitch_midi=pitch,
            pitch_name=self.midi_to_note_name(pitch),
            start=start_time,
            end=end_time,
            velocity=velocity,
            confidence=0.5,
            detection_method="manual"
        )

        # Insert note
        self.notes.insert(current_row, new_note)
        self.update_display()

        # Select the new note
        self.note_table.selectRow(current_row)
        self.note_table.setFocus()

        self.melody_changed.emit(self.notes)

    def delete_note(self):
        """Delete the selected note."""
        current_row = self.note_table.currentRow()
        if current_row >= 0 and current_row < len(self.notes):
            del self.notes[current_row]
            self.update_display()
            self.melody_changed.emit(self.notes)

    def move_note_up(self):
        """Move the selected note up."""
        current_row = self.note_table.currentRow()
        if current_row > 0:
            self.notes[current_row], self.notes[current_row - 1] = \
                self.notes[current_row - 1], self.notes[current_row]
            self.update_display()
            self.note_table.selectRow(current_row - 1)
            self.melody_changed.emit(self.notes)

    def move_note_down(self):
        """Move the selected note down."""
        current_row = self.note_table.currentRow()
        if current_row < len(self.notes) - 1:
            self.notes[current_row], self.notes[current_row + 1] = \
                self.notes[current_row + 1], self.notes[current_row]
            self.update_display()
            self.note_table.selectRow(current_row + 1)
            self.melody_changed.emit(self.notes)

    def filter_short_notes(self):
        """Filter out very short notes."""
        if not self.notes:
            return

        min_duration = 0.1  # 100ms minimum
        original_count = len(self.notes)

        self.notes = [note for note in self.notes if (note.end - note.start) >= min_duration]

        filtered_count = original_count - len(self.notes)

        if filtered_count > 0:
            self.update_display()
            self.melody_changed.emit(self.notes)

            QMessageBox.information(
                self,
                "Short Notes Filtered",
                f"Removed {filtered_count} notes shorter than {min_duration}s."
            )
        else:
            QMessageBox.information(
                self,
                "No Filtering Needed",
                f"No notes shorter than {min_duration}s found."
            )

    def merge_similar_notes(self):
        """Merge consecutive notes with the same pitch."""
        if len(self.notes) < 2:
            return

        merged_count = 0
        i = 0

        while i < len(self.notes) - 1:
            current_note = self.notes[i]
            next_note = self.notes[i + 1]

            # Check if notes are consecutive and have same pitch
            if (current_note.pitch_midi == next_note.pitch_midi and
                    abs(current_note.end - next_note.start) < 0.1):  # 100ms gap tolerance
                # Merge notes
                current_note.end = next_note.end
                del self.notes[i + 1]
                merged_count += 1
            else:
                i += 1

        if merged_count > 0:
            self.update_display()
            self.melody_changed.emit(self.notes)

            QMessageBox.information(
                self,
                "Notes Merged",
                f"Merged {merged_count} consecutive notes with same pitch."
            )
        else:
            QMessageBox.information(
                self,
                "No Merging Needed",
                "No consecutive notes with same pitch found."
            )

    def get_notes(self) -> List[Note]:
        """Get the current note list."""
        return self.notes.copy()

    def set_notes(self, notes: List[Note]):
        """Set the note list."""
        self.notes = notes.copy()
        self.update_display()

    def export_melody_midi(self) -> List[int]:
        """Export melody as list of MIDI pitches."""
        return [note.pitch_midi for note in self.notes]

    def import_melody_midi(self, midi_pitches: List[int]):
        """Import melody from list of MIDI pitches."""
        # Create note objects with default timing
        self.notes = []
        for i, pitch in enumerate(midi_pitches):
            start_time = i * 0.5  # Default 0.5s per note
            end_time = start_time + 0.5

            note = Note(
                pitch_midi=pitch,
                pitch_name=self.midi_to_note_name(pitch),
                start=start_time,
                end=end_time,
                velocity=80,
                confidence=0.5,
                detection_method="imported"
            )
            self.notes.append(note)

        self.update_display()
        self.melody_changed.emit(self.notes)


class MelodyVisualizationWidget(QWidget):
    """Widget for visualizing melody as a piano roll."""

    def __init__(self):
        super().__init__()
        self.notes = []
        self.setMinimumHeight(200)
        self.setMaximumHeight(300)

    def set_notes(self, notes: List[Note]):
        """Set the notes to visualize."""
        self.notes = notes
        self.update()

    def paintEvent(self, event):
        """Paint the melody visualization."""
        if not self.notes:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Get dimensions
        width = self.width()
        height = self.height()

        if width <= 0 or height <= 0:
            return

        # Find time range
        if self.notes:
            min_time = min(note.start for note in self.notes)
            max_time = max(note.end for note in self.notes)
            time_range = max_time - min_time
        else:
            time_range = 1.0

        # Find pitch range
        if self.notes:
            min_pitch = min(note.pitch_midi for note in self.notes)
            max_pitch = max(note.pitch_midi for note in self.notes)
            pitch_range = max_pitch - min_pitch + 1
        else:
            pitch_range = 12

        # Draw background
        painter.fillRect(0, 0, width, height, QColor(240, 240, 240))

        # Draw grid
        painter.setPen(QPen(QColor(200, 200, 200), 1))

        # Vertical lines (time)
        for i in range(11):
            x = (i * width) // 10
            painter.drawLine(x, 0, x, height)

        # Horizontal lines (pitch)
        for i in range(11):
            y = (i * height) // 10
            painter.drawLine(0, y, width, y)

        # Draw notes
        for note in self.notes:
            # Calculate position and size
            x1 = ((note.start - min_time) / time_range) * width
            x2 = ((note.end - min_time) / time_range) * width
            y = ((max_pitch - note.pitch_midi) / pitch_range) * height

            note_width = x2 - x1
            note_height = height / pitch_range

            # Draw note rectangle
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.setBrush(QBrush(QColor(100, 150, 255)))
            painter.drawRect(int(x1), int(y), int(note_width), int(note_height))

            # Draw note name
            if note_width > 20:  # Only draw text if note is wide enough
                painter.setPen(QColor(255, 255, 255))
                painter.setFont(QFont("Arial", 8))
                note_name = self.midi_to_note_name(note.pitch_midi)
                painter.drawText(int(x1 + 2), int(y + note_height - 2), note_name)

    def midi_to_note_name(self, midi_pitch: int) -> str:
        """Convert MIDI pitch to note name."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_pitch // 12) - 1
        note_name = note_names[midi_pitch % 12]
        return f"{note_name}{octave}"
