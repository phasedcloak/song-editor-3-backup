#!/usr/bin/env python3
"""
Chord Editor UI

Chord editing interface for Song Editor 3.
"""

from typing import List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QPushButton, QLabel, QDoubleSpinBox,
    QGroupBox, QGridLayout, QHeaderView, QMessageBox, QComboBox,
    QLineEdit, QSplitter, QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, Signal

from ..models.song_data import SongData, Chord


class ChordEditor(QWidget):
    """Chord editing interface."""

    chords_changed = Signal(list)

    def __init__(self):
        super().__init__()
        self.song_data = None
        self.chords = []
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Create splitter for different views
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Chord progression view
        left_panel = self.create_progression_panel()
        splitter.addWidget(left_panel)

        # Right panel - Detailed table view
        right_panel = self.create_table_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([400, 600])

        # Bottom panel - Controls
        bottom_panel = self.create_control_panel()
        layout.addWidget(bottom_panel)

    def create_progression_panel(self):
        """Create the chord progression panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Chord progression list
        layout.addWidget(QLabel("Chord Progression:"))
        self.progression_list = QListWidget()
        self.progression_list.itemClicked.connect(self.on_progression_item_clicked)
        layout.addWidget(self.progression_list)

        # Progression statistics
        stats_group = QGroupBox("Progression Statistics")
        stats_layout = QGridLayout(stats_group)

        self.total_chords_label = QLabel("0")
        stats_layout.addWidget(QLabel("Total Chords:"), 0, 0)
        stats_layout.addWidget(self.total_chords_label, 0, 1)

        self.unique_chords_label = QLabel("0")
        stats_layout.addWidget(QLabel("Unique Chords:"), 1, 0)
        stats_layout.addWidget(self.unique_chords_label, 1, 1)

        self.duration_label = QLabel("0.0s")
        stats_layout.addWidget(QLabel("Total Duration:"), 2, 0)
        stats_layout.addWidget(self.duration_label, 2, 1)

        layout.addWidget(stats_group)

        return panel

    def create_table_panel(self):
        """Create the detailed table panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Table
        layout.addWidget(QLabel("Chord Details:"))
        self.chord_table = QTableWidget()
        self.chord_table.setColumnCount(7)
        self.chord_table.setHorizontalHeaderLabels([
            "Symbol", "Root", "Quality", "Start Time", "End Time", "Duration", "Method"
        ])

        # Set table properties
        header = self.chord_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.Stretch)

        self.chord_table.itemChanged.connect(self.on_table_item_changed)
        self.chord_table.itemSelectionChanged.connect(self.on_table_selection_changed)
        layout.addWidget(self.chord_table)

        return panel

    def create_control_panel(self):
        """Create the control panel."""
        panel = QWidget()
        layout = QHBoxLayout(panel)

        # Chord input controls
        chord_group = QGroupBox("Chord Input")
        chord_layout = QGridLayout(chord_group)

        # Root note
        self.root_combo = QComboBox()
        self.root_combo.addItems(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
        chord_layout.addWidget(QLabel("Root:"), 0, 0)
        chord_layout.addWidget(self.root_combo, 0, 1)

        # Quality
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(['major', 'minor', 'dim', 'aug', '7', 'm7', 'maj7', 'dim7', 'sus2', 'sus4'])
        chord_layout.addWidget(QLabel("Quality:"), 0, 2)
        chord_layout.addWidget(self.quality_combo, 0, 3)

        # Bass note
        self.bass_combo = QComboBox()
        self.bass_combo.addItems(['', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
        chord_layout.addWidget(QLabel("Bass:"), 1, 0)
        chord_layout.addWidget(self.bass_combo, 1, 1)

        # Symbol preview
        self.symbol_preview = QLineEdit()
        self.symbol_preview.setReadOnly(True)
        chord_layout.addWidget(QLabel("Symbol:"), 1, 2)
        chord_layout.addWidget(self.symbol_preview, 1, 3)

        # Connect signals for symbol preview
        self.root_combo.currentTextChanged.connect(self.update_symbol_preview)
        self.quality_combo.currentTextChanged.connect(self.update_symbol_preview)
        self.bass_combo.currentTextChanged.connect(self.update_symbol_preview)

        layout.addWidget(chord_group)

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

        self.add_chord_btn = QPushButton("Add Chord")
        self.add_chord_btn.clicked.connect(self.add_chord)
        button_layout.addWidget(self.add_chord_btn)

        self.delete_chord_btn = QPushButton("Delete Chord")
        self.delete_chord_btn.clicked.connect(self.delete_chord)
        button_layout.addWidget(self.delete_chord_btn)

        self.move_up_btn = QPushButton("Move Up")
        self.move_up_btn.clicked.connect(self.move_chord_up)
        button_layout.addWidget(self.move_up_btn)

        self.move_down_btn = QPushButton("Move Down")
        self.move_down_btn.clicked.connect(self.move_chord_down)
        button_layout.addWidget(self.move_down_btn)

        layout.addWidget(button_group)

        # Analysis controls
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)

        self.simplify_chords_btn = QPushButton("Simplify Chords")
        self.simplify_chords_btn.clicked.connect(self.simplify_chords)
        analysis_layout.addWidget(self.simplify_chords_btn)

        self.merge_similar_btn = QPushButton("Merge Similar")
        self.merge_similar_btn.clicked.connect(self.merge_similar_chords)
        analysis_layout.addWidget(self.merge_similar_btn)

        layout.addWidget(analysis_group)

        layout.addStretch()

        return panel

    def set_song_data(self, song_data: SongData):
        """Set the song data to edit."""
        self.song_data = song_data
        self.chords = song_data.chords.copy()
        self.update_display()

    def update_display(self):
        """Update the display with current chord data."""
        self.update_progression_list()
        self.update_table()
        self.update_statistics()

    def update_progression_list(self):
        """Update the chord progression list."""
        self.progression_list.clear()

        for chord in self.chords:
            item = QListWidgetItem(chord.symbol)
            item.setData(Qt.UserRole, chord)
            self.progression_list.addItem(item)

    def update_table(self):
        """Update the chord table."""
        self.chord_table.blockSignals(True)
        self.chord_table.setRowCount(len(self.chords))

        for row, chord in enumerate(self.chords):
            # Symbol
            symbol_item = QTableWidgetItem(chord.symbol)
            self.chord_table.setItem(row, 0, symbol_item)

            # Root
            root_item = QTableWidgetItem(chord.root)
            self.chord_table.setItem(row, 1, root_item)

            # Quality
            quality_item = QTableWidgetItem(chord.quality)
            self.chord_table.setItem(row, 2, quality_item)

            # Start time
            start_item = QTableWidgetItem(f"{chord.start:.3f}")
            self.chord_table.setItem(row, 3, start_item)

            # End time
            end_item = QTableWidgetItem(f"{chord.end:.3f}")
            self.chord_table.setItem(row, 4, end_item)

            # Duration
            duration = chord.end - chord.start
            duration_item = QTableWidgetItem(f"{duration:.3f}")
            self.chord_table.setItem(row, 5, duration_item)

            # Detection method
            method_item = QTableWidgetItem(chord.detection_method or "")
            self.chord_table.setItem(row, 6, method_item)

        self.chord_table.blockSignals(False)

    def update_statistics(self):
        """Update the statistics display."""
        total_chords = len(self.chords)
        unique_chords = len(set(chord.symbol for chord in self.chords))
        total_duration = sum(chord.end - chord.start for chord in self.chords)

        self.total_chords_label.setText(str(total_chords))
        self.unique_chords_label.setText(str(unique_chords))
        self.duration_label.setText(f"{total_duration:.1f}s")

    def update_symbol_preview(self):
        """Update the chord symbol preview."""
        root = self.root_combo.currentText()
        quality = self.quality_combo.currentText()
        bass = self.bass_combo.currentText()

        symbol = root
        if quality == 'major':
            pass  # No suffix for major
        elif quality == 'minor':
            symbol += 'm'
        elif quality == 'dim':
            symbol += 'dim'
        elif quality == 'aug':
            symbol += 'aug'
        elif quality == '7':
            symbol += '7'
        elif quality == 'm7':
            symbol += 'm7'
        elif quality == 'maj7':
            symbol += 'maj7'
        elif quality == 'dim7':
            symbol += 'dim7'
        elif quality == 'sus2':
            symbol += 'sus2'
        elif quality == 'sus4':
            symbol += 'sus4'

        if bass and bass != root:
            symbol += f"/{bass}"

        self.symbol_preview.setText(symbol)

    def on_progression_item_clicked(self, item):
        """Handle progression list item click."""
        chord = item.data(Qt.UserRole)
        if chord:
            # Find the chord in the table and select it
            for row in range(self.chord_table.rowCount()):
                if self.chords[row] == chord:
                    self.chord_table.selectRow(row)
                    self.chord_table.scrollToItem(self.chord_table.item(row, 0))
                    break

    def on_table_item_changed(self, item):
        """Handle table item changes."""
        row = item.row()
        col = item.column()

        if row >= len(self.chords):
            return

        chord = self.chords[row]

        try:
            if col == 0:  # Symbol
                chord.symbol = item.text()
            elif col == 1:  # Root
                chord.root = item.text()
            elif col == 2:  # Quality
                chord.quality = item.text()
            elif col == 3:  # Start time
                chord.start = float(item.text())
            elif col == 4:  # End time
                chord.end = float(item.text())
            elif col == 6:  # Detection method
                chord.detection_method = item.text()

            # Update duration column
            duration = chord.end - chord.start
            duration_item = QTableWidgetItem(f"{duration:.3f}")
            self.chord_table.setItem(row, 5, duration_item)

            self.update_progression_list()
            self.update_statistics()
            self.chords_changed.emit(self.chords)

        except ValueError:
            # Revert invalid input
            self.update_table()
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")

    def on_table_selection_changed(self):
        """Handle table selection changes."""
        current_row = self.chord_table.currentRow()
        if current_row >= 0 and current_row < len(self.chords):
            chord = self.chords[current_row]

            # Update timing controls
            self.start_time_spin.setValue(chord.start)
            self.end_time_spin.setValue(chord.end)

            # Update chord input controls
            self.root_combo.setCurrentText(chord.root)
            self.quality_combo.setCurrentText(chord.quality)
            self.bass_combo.setCurrentText(chord.bass or "")

    def add_chord(self):
        """Add a new chord."""
        # Get current selection
        current_row = self.chord_table.currentRow()
        if current_row < 0:
            current_row = len(self.chords)

        # Get chord details from controls
        root = self.root_combo.currentText()
        quality = self.quality_combo.currentText()
        bass = self.bass_combo.currentText() if self.bass_combo.currentText() else None
        symbol = self.symbol_preview.text()

        # Create new chord
        start_time = current_row * 1.0  # Default 1s per chord
        end_time = start_time + 1.0
        new_chord = Chord(
            symbol=symbol,
            root=root,
            quality=quality,
            start=start_time,
            end=end_time,
            bass=bass,
            detection_method="manual"
        )

        # Insert chord
        self.chords.insert(current_row, new_chord)
        self.update_display()

        # Select the new chord
        self.chord_table.selectRow(current_row)
        self.chord_table.setFocus()

        self.chords_changed.emit(self.chords)

    def delete_chord(self):
        """Delete the selected chord."""
        current_row = self.chord_table.currentRow()
        if current_row >= 0 and current_row < len(self.chords):
            del self.chords[current_row]
            self.update_display()
            self.chords_changed.emit(self.chords)

    def move_chord_up(self):
        """Move the selected chord up."""
        current_row = self.chord_table.currentRow()
        if current_row > 0:
            self.chords[current_row], self.chords[current_row - 1] = \
                self.chords[current_row - 1], self.chords[current_row]
            self.update_display()
            self.chord_table.selectRow(current_row - 1)
            self.chords_changed.emit(self.chords)

    def move_chord_down(self):
        """Move the selected chord down."""
        current_row = self.chord_table.currentRow()
        if current_row < len(self.chords) - 1:
            self.chords[current_row], self.chords[current_row + 1] = \
                self.chords[current_row + 1], self.chords[current_row]
            self.update_display()
            self.chord_table.selectRow(current_row + 1)
            self.chords_changed.emit(self.chords)

    def simplify_chords(self):
        """Simplify chord symbols."""
        simplified_count = 0

        for chord in self.chords:
            original_symbol = chord.symbol

            # Simple simplifications
            if chord.quality == 'major' and not chord.bass:
                chord.symbol = chord.root
            elif chord.quality == 'minor' and not chord.bass:
                chord.symbol = chord.root + 'm'
            elif chord.quality == '7' and not chord.bass:
                chord.symbol = chord.root + '7'
            elif chord.quality == 'm7' and not chord.bass:
                chord.symbol = chord.root + 'm7'

            if chord.symbol != original_symbol:
                simplified_count += 1

        if simplified_count > 0:
            self.update_display()
            self.chords_changed.emit(self.chords)

            QMessageBox.information(
                self,
                "Chords Simplified",
                f"Simplified {simplified_count} chord symbols."
            )
        else:
            QMessageBox.information(
                self,
                "No Simplification Needed",
                "All chords are already in simplified form."
            )

    def merge_similar_chords(self):
        """Merge consecutive identical chords."""
        if len(self.chords) < 2:
            return

        merged_count = 0
        i = 0

        while i < len(self.chords) - 1:
            current_chord = self.chords[i]
            next_chord = self.chords[i + 1]

            if current_chord.symbol == next_chord.symbol:
                # Merge chords
                current_chord.end = next_chord.end
                del self.chords[i + 1]
                merged_count += 1
            else:
                i += 1

        if merged_count > 0:
            self.update_display()
            self.chords_changed.emit(self.chords)

            QMessageBox.information(
                self,
                "Chords Merged",
                f"Merged {merged_count} consecutive identical chords."
            )
        else:
            QMessageBox.information(
                self,
                "No Merging Needed",
                "No consecutive identical chords found."
            )

    def get_chords(self) -> List[Chord]:
        """Get the current chord list."""
        return self.chords.copy()

    def set_chords(self, chords: List[Chord]):
        """Set the chord list."""
        self.chords = chords.copy()
        self.update_display()

    def export_progression(self) -> str:
        """Export chord progression as text."""
        return ' '.join(chord.symbol for chord in self.chords)

    def import_progression(self, progression_text: str):
        """Import chord progression from text."""
        symbols = progression_text.split()

        # Create chord objects with default timing
        self.chords = []
        for i, symbol in enumerate(symbols):
            start_time = i * 1.0
            end_time = start_time + 1.0

            # Simple parsing of chord symbols
            if 'm' in symbol and not symbol.endswith('7'):
                root = symbol.replace('m', '')
                quality = 'minor'
            elif symbol.endswith('7'):
                if 'm' in symbol:
                    root = symbol.replace('m7', '')
                    quality = 'm7'
                else:
                    root = symbol.replace('7', '')
                    quality = '7'
            else:
                root = symbol
                quality = 'major'

            chord = Chord(
                symbol=symbol,
                root=root,
                quality=quality,
                start=start_time,
                end=end_time,
                detection_method="imported"
            )
            self.chords.append(chord)

        self.update_display()
        self.chords_changed.emit(self.chords)
