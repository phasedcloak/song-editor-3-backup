#!/usr/bin/env python3
"""
Song Information Dialog

Dialog window for editing song metadata.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QDialogButtonBox, QTextEdit
)
from PySide6.QtCore import Qt


class SongInfoDialog(QDialog):
    """Dialog for editing song metadata and information."""
    
    def __init__(self, parent=None, song_data=None):
        super().__init__(parent)
        self.setWindowTitle("Song Information")
        self.setModal(True)
        self.resize(400, 300)
        
        self.song_data = song_data
        self.setup_ui()
        self.load_song_data()
    
    def setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Basic information group
        basic_group = QGroupBox("Basic Information")
        basic_layout = QGridLayout(basic_group)
        
        # Title
        basic_layout.addWidget(QLabel("Title:"), 0, 0)
        self.title_edit = QLineEdit()
        basic_layout.addWidget(self.title_edit, 0, 1)
        
        # Artist
        basic_layout.addWidget(QLabel("Artist:"), 1, 0)
        self.artist_edit = QLineEdit()
        basic_layout.addWidget(self.artist_edit, 1, 1)
        
        # Album
        basic_layout.addWidget(QLabel("Album:"), 2, 0)
        self.album_edit = QLineEdit()
        basic_layout.addWidget(self.album_edit, 2, 1)
        
        # Genre
        basic_layout.addWidget(QLabel("Genre:"), 3, 0)
        self.genre_edit = QLineEdit()
        basic_layout.addWidget(self.genre_edit, 3, 1)
        
        # Year
        basic_layout.addWidget(QLabel("Year:"), 4, 0)
        self.year_edit = QLineEdit()
        self.year_edit.setPlaceholderText("e.g., 2025")
        basic_layout.addWidget(self.year_edit, 4, 1)
        
        layout.addWidget(basic_group)
        
        # Additional information group
        additional_group = QGroupBox("Additional Information")
        additional_layout = QGridLayout(additional_group)
        
        # Composer
        additional_layout.addWidget(QLabel("Composer:"), 0, 0)
        self.composer_edit = QLineEdit()
        additional_layout.addWidget(self.composer_edit, 0, 1)
        
        # Key signature
        additional_layout.addWidget(QLabel("Key:"), 1, 0)
        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText("e.g., C major, Am")
        additional_layout.addWidget(self.key_edit, 1, 1)
        
        # Tempo (BPM)
        additional_layout.addWidget(QLabel("Tempo (BPM):"), 2, 0)
        self.tempo_edit = QLineEdit()
        self.tempo_edit.setPlaceholderText("e.g., 120")
        additional_layout.addWidget(self.tempo_edit, 2, 1)
        
        layout.addWidget(additional_group)
        
        # Notes/Description group
        notes_group = QGroupBox("Notes")
        notes_layout = QVBoxLayout(notes_group)
        
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Additional notes about this song...")
        self.notes_edit.setMaximumHeight(80)
        notes_layout.addWidget(self.notes_edit)
        
        layout.addWidget(notes_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        # Rename OK button to "Save"
        ok_button = button_box.button(QDialogButtonBox.Ok)
        ok_button.setText("Save")
        
        layout.addWidget(button_box)
    
    def load_song_data(self):
        """Load existing song data into the dialog."""
        if not self.song_data:
            return
        
        # Get metadata from song data
        metadata = self.song_data.get('metadata', {}) if isinstance(self.song_data, dict) else {}
        
        # Load basic information
        self.title_edit.setText(metadata.get('title', ''))
        self.artist_edit.setText(metadata.get('artist', ''))
        self.album_edit.setText(metadata.get('album', ''))
        self.genre_edit.setText(metadata.get('genre', ''))
        self.year_edit.setText(str(metadata.get('year', '')) if metadata.get('year') else '')
        
        # Load additional information
        self.composer_edit.setText(metadata.get('composer', ''))
        
        # Try to get key from audio analysis
        if isinstance(self.song_data, dict):
            audio_analysis = self.song_data.get('audio_analysis', {})
            key_info = audio_analysis.get('key', {})
            if isinstance(key_info, dict) and 'key' in key_info:
                key_str = f"{key_info['key']} {key_info.get('mode', 'major')}"
                self.key_edit.setText(key_str)
            
            # Try to get tempo from audio analysis
            tempo = audio_analysis.get('tempo')
            if tempo:
                self.tempo_edit.setText(str(round(tempo)))
        
        self.notes_edit.setText(metadata.get('notes', ''))
    
    def get_song_info(self) -> dict:
        """Get the song information from the dialog."""
        return {
            'title': self.title_edit.text().strip(),
            'artist': self.artist_edit.text().strip(),
            'album': self.album_edit.text().strip(),
            'genre': self.genre_edit.text().strip(),
            'year': int(self.year_edit.text().strip()) if self.year_edit.text().strip().isdigit() else None,
            'composer': self.composer_edit.text().strip(),
            'key': self.key_edit.text().strip(),
            'tempo': int(self.tempo_edit.text().strip()) if self.tempo_edit.text().strip().isdigit() else None,
            'notes': self.notes_edit.toPlainText().strip()
        }
