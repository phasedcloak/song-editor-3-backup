#!/usr/bin/env python3
"""
Main Window UI

Main application window for Song Editor 3.
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QFileDialog, QProgressBar,
    QTabWidget, QGroupBox, QGridLayout, QComboBox, QCheckBox,
    QLineEdit, QMessageBox, QStatusBar,
    QToolBar, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QAction

from ..platform_utils import PlatformUtils, PlatformAwareWidget
from .platform_styles import PlatformStyles

from ..core.audio_processor import AudioProcessor
from ..core.transcriber import Transcriber
from ..core.chord_detector import ChordDetector
from ..core.melody_extractor import MelodyExtractor
from ..export.midi_exporter import MidiExporter
from ..export.ccli_exporter import CCLIExporter
from ..export.json_exporter import JSONExporter
from ..models.song_data import SongData
from ..models.metadata import Metadata
from .lyrics_editor import LyricsEditor
from .chord_editor import ChordEditor
from .melody_editor import MelodyEditor


class ProcessingThread(QThread):
    """Background thread for audio processing."""

    progress_updated = Signal(str, int)
    stage_completed = Signal(str, dict)
    processing_finished = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, audio_file: str, config: Dict[str, Any]):
        super().__init__()
        self.audio_file = audio_file
        self.config = config
        self.song_data = None
        self.timeout_seconds = 1800  # 30 minute timeout for long audio files
        self.start_time = None

    def run(self):
        """Run the audio processing pipeline using CLI subprocess isolation."""
        self.start_time = time.time()
        try:
            # Import the CLI processing function that already has subprocess isolation
            from ..app import process_audio_file
            
            # Convert GUI config to CLI parameters
            separation_engine = self.config.get('separation_engine', 'demucs')
            use_demucs = self.config.get('use_demucs', True)
            use_chordino = (self.config.get('chord_method') == 'chordino')
            
            # Emit progress updates
            self.progress_updated.emit("Starting audio processing...", 10)
            logging.info(f"GUI processing audio file using CLI function: {self.audio_file}")
            
            # Monitor progress by patching the logging system temporarily
            self._setup_progress_monitoring()
            
            # Use the CLI processing function that has subprocess isolation
            success = process_audio_file(
                input_path=self.audio_file,
                output_dir=None,  # Use same directory as input
                whisper_model=self.config.get('whisper_model', 'mlx-whisper'),
                use_chordino=use_chordino,
                separation_engine=separation_engine,
                audio_separator_model=self.config.get('audio_separator_model', 'UVR_MDXNET_KARA_2'),
                use_cuda=self.config.get('use_cuda', False),
                use_coreml=self.config.get('use_coreml', True),
                use_demucs=use_demucs,
                save_intermediate=self.config.get('save_intermediate', False),
                demucs_model='htdemucs',
                no_gui=True,  # Important: tells CLI function we're in GUI mode
                force_overwrite=True  # GUI should always reprocess when user clicks the button
            )
            
            if not success:
                raise Exception("CLI processing function failed")
                
            # Load the generated JSON file to get the results
            from pathlib import Path
            audio_path = Path(self.audio_file)
            json_path = audio_path.parent / f"{audio_path.stem}.song_data.json"
            
            if json_path.exists():
                import json
                with open(json_path, 'r') as f:
                    song_data = json.load(f)
                
                # Convert to the format expected by the GUI
                audio_data = {
                    'audio': None,  # GUI doesn't need raw audio data
                    'sample_rate': song_data.get('audio_analysis', {}).get('sample_rate', 44100),
                    'analysis': song_data.get('audio_analysis', {}),
                    'processing_info': song_data.get('processing_info', {})
                }

                # Extract data for GUI
                words = song_data.get('words', [])
                chords = song_data.get('chords', [])
                notes = song_data.get('notes', [])
                analysis = song_data.get('audio_analysis', {})
                
                # Emit progress updates
                self.progress_updated.emit(f"Processing completed: {len(words)} words found", 80)
                self.progress_updated.emit(f"Detected {len(chords)} chords", 90)
                self.progress_updated.emit(f"Extracted {len(notes)} notes", 95)
                
                logging.info(f"CLI processing completed successfully")
                logging.info(f"Results: {len(words)} words, {len(chords)} chords, {len(notes)} notes")
            else:
                raise Exception(f"JSON file not found: {json_path}")
                
            # Emit progress completion
            self.progress_updated.emit("Processing completed successfully!", 100)

            # Create the song data structure for the GUI
            self.song_data = {
                'metadata': {
                    'source_audio': self.audio_file,
                    'processing_config': self.config,
                    'processing_time': time.time() - self.start_time
                },
                'words': words,
                'chords': chords,
                'notes': notes,
                'audio_data': audio_data,
                'audio_analysis': analysis
            }
            
            # Emit final signals
            self.stage_completed.emit("transcription", {'words': words})
            self.stage_completed.emit("chords", {'chords': chords})
            self.stage_completed.emit("melody", {'notes': notes})
            self.processing_finished.emit(self.song_data)
            
        except Exception as e:
            logging.error(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))
            
        finally:
            # Clean up progress monitoring
            self._cleanup_progress_monitoring()
    
    def _setup_progress_monitoring(self):
        """Set up progress monitoring by intercepting log messages."""
        # Store the original log handler
        self.original_handlers = logging.getLogger().handlers.copy()
        
        # Create a custom handler that emits progress signals
        class ProgressHandler(logging.Handler):
            def __init__(self, thread):
                super().__init__()
                self.thread = thread
                self.progress_map = {
                    'Loading audio': ("Loading audio...", 15),
                    'Loaded audio data': ("Audio loaded successfully", 20),
                    'Denoising audio': ("Denoising audio...", 25),
                    'Denoising completed': ("Denoising completed", 30),
                    'Normalizing audio': ("Normalizing audio...", 35),
                    'normalization completed': ("Normalization completed", 40),
                    'Source separation': ("Separating audio sources...", 45),
                    'Audio processing completed': ("Audio processing completed", 50),
                    'Transcribing lyrics': ("Transcribing lyrics...", 55),
                    'Transcription completed': ("Transcription completed", 70),
                    'Detecting chords': ("Detecting chords...", 75),
                    'Chord detection': ("Chord detection completed", 80),
                    'Extracting melody': ("Extracting melody...", 85),
                    'melody extraction': ("Melody extraction completed", 90),
                    'Exporting': ("Exporting results...", 95),
                    'Processing complete': ("Processing completed!", 100)
                }
            
            def emit(self, record):
                message = record.getMessage()
                for key, (status, progress) in self.progress_map.items():
                    if key.lower() in message.lower():
                        self.thread.progress_updated.emit(status, progress)
                        break
        
        # Add our progress handler
        progress_handler = ProgressHandler(self)
        logging.getLogger().addHandler(progress_handler)
        self.progress_handler = progress_handler
    
    def _cleanup_progress_monitoring(self):
        """Clean up progress monitoring by removing our custom handler."""
        try:
            if hasattr(self, 'progress_handler'):
                logging.getLogger().removeHandler(self.progress_handler)
        except Exception as e:
            logging.warning(f"Error cleaning up progress handler: {e}")

    def _save_temp_audio(self, audio_data: "np.ndarray", sr: int) -> str:
        """Saves a numpy array to a temporary WAV file and returns the path."""
        import soundfile as sf
        import tempfile
        import numpy as np

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        # Audio from Demucs is (channels, samples), soundfile expects (samples, channels).
        # We must transpose 2D arrays. 1D (mono) arrays are fine as is.
        if audio_data.ndim > 1:
            sf.write(temp_file.name, audio_data.T, sr)
        else:
            sf.write(temp_file.name, audio_data, sr)
        return temp_file.name

    def stop(self):
        """Stop the processing thread."""
        self.terminate()


class MainWindow(QMainWindow):
    """Main application window with platform-aware design."""

    def __init__(self):
        super().__init__()

        # Initialize platform-aware widget functionality
        self.platform_utils = PlatformUtils()
        self.platform_aware = PlatformAwareWidget()

        self.song_data = None
        self.processing_thread = None
        self.settings = QSettings('SongEditor3', 'SongEditor3')

        # Platform-specific setup
        self.setup_platform_specific_behavior()
        self.init_ui()
        self.load_settings()

    def setup_platform_specific_behavior(self):
        """Setup platform-specific behavior for the main window."""
        # Set window title and size
        self.setWindowTitle("Song Editor 3")

        # Get platform-specific window size
        width, height = self.platform_utils.get_recommended_window_size()
        self.resize(width, height)

        # Apply platform-specific stylesheet
        self.setStyleSheet(PlatformStyles.get_main_window_style())

        # Platform-specific window flags
        if self.platform_utils.is_mobile():
            # Mobile: full screen or large modal
            self.setWindowFlags(Qt.Window | Qt.WindowMaximizeButtonHint)
        else:
            # Desktop: normal window with minimize/maximize
            self.setWindowFlags(Qt.Window)

        # Note: High DPI and touch attributes should be set on QApplication, not QWidget
        # These are handled in the main application setup

    def init_ui(self):
        """Initialize the user interface with platform-aware design."""
        # Get platform-specific optimizations
        mobile_opts = PlatformStyles.get_mobile_optimizations()

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout with platform-specific spacing
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(mobile_opts["spacing"])
        main_layout.setContentsMargins(
            mobile_opts["safe_area_margin"],
            mobile_opts["safe_area_margin"],
            mobile_opts["safe_area_margin"],
            mobile_opts["safe_area_margin"]
        )

        # Create menu bar (minimal for mobile)
        if not self.platform_utils.is_mobile():
            self.create_menu_bar()
            self.create_toolbar()

        # Create main content area
        if self.platform_utils.is_mobile():
            # Mobile: vertical layout with scroll area
            self.create_mobile_ui(main_layout)
        else:
            # Desktop: horizontal splitter layout
            self.create_desktop_ui(main_layout)

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Set status
        self.status_bar.showMessage("Ready")

    def create_mobile_ui(self, main_layout):
        """Create mobile-optimized UI layout."""
        # Create scroll area for mobile
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create scroll content widget
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(16)  # Mobile spacing

        # Add controls at top
        controls_panel = self.create_mobile_controls_panel()
        scroll_layout.addWidget(controls_panel)

        # Add tab widget for editors
        self.tab_widget = QTabWidget()
        self.lyrics_editor = self.create_lyrics_editor()
        self.chord_editor = self.create_chord_editor()
        self.melody_editor = self.create_melody_editor()
        self.tab_widget.addTab(self.lyrics_editor, "Lyrics")
        self.tab_widget.addTab(self.chord_editor, "Chords")
        self.tab_widget.addTab(self.melody_editor, "Melody")
        scroll_layout.addWidget(self.tab_widget)

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

    def create_desktop_ui(self, main_layout):
        """Create desktop-optimized UI layout."""
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # Create left panel (controls and info)
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)

        # Create right panel (editors)
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setSizes([300, 900])

    def create_mobile_controls_panel(self):
        """Create mobile-optimized controls panel."""
        panel = QGroupBox("Audio Processing")
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)

        select_button = QPushButton("Select Audio")
        select_button.clicked.connect(self.open_audio_file)
        select_button.setMinimumHeight(48)  # Mobile touch target
        file_layout.addWidget(select_button)
        layout.addLayout(file_layout)

        # Processing options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)

        # Whisper model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["openai-whisper", "faster-whisper", "whisperx"])
        model_layout.addWidget(self.model_combo)
        options_layout.addLayout(model_layout)

        # Whisper model size selection
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Model Size:"))
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItems([
            'large-v3-turbo', 'large-v3', 'large-v2', 'large', 'medium', 'small', 'base', 'tiny'
        ])
        self.model_size_combo.setToolTip(
            "Preferred size for Whisper engines. 'large-v3-turbo' requires recent Whisper/Faster-Whisper. "
            "If unavailable, the app will fall back to large-v2 automatically."
        )
        size_layout.addWidget(self.model_size_combo)
        options_layout.addLayout(size_layout)

        # Content type selection
        content_layout = QHBoxLayout()
        content_layout.addWidget(QLabel("Content:"))
        self.content_combo = QComboBox()
        self.content_combo.addItems(["general", "christian", "gospel", "worship", "hymn", "clean"])
        content_layout.addWidget(self.content_combo)
        options_layout.addLayout(content_layout)

        layout.addWidget(options_group)

        # Process button
        self.process_button = QPushButton("Process Audio")
        self.process_button.clicked.connect(self.process_audio)
        self.process_button.setMinimumHeight(48)  # Mobile touch target
        layout.addWidget(self.process_button)

        return panel

    def create_lyrics_editor(self):
        """Create lyrics editor widget."""
        from .enhanced_lyrics_editor import EnhancedLyricsEditor
        return EnhancedLyricsEditor()

    def create_chord_editor(self):
        """Create chord editor widget."""
        from .chord_editor import ChordEditor
        return ChordEditor()

    def create_melody_editor(self):
        """Create melody editor widget."""
        from .melody_editor import MelodyEditor
        return MelodyEditor()

    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('&File')

        open_action = QAction('&Open Audio...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_audio_file)
        file_menu.addAction(open_action)

        save_action = QAction('&Save Song Data...', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_song_data)
        file_menu.addAction(save_action)

        export_menu = file_menu.addMenu('&Export')

        export_midi_action = QAction('Export &MIDI...', self)
        export_midi_action.triggered.connect(self.export_midi)
        export_menu.addAction(export_midi_action)

        export_ccli_action = QAction('Export &CCLI...', self)
        export_ccli_action.triggered.connect(self.export_ccli)
        export_menu.addAction(export_ccli_action)

        export_json_action = QAction('Export &JSON...', self)
        export_json_action.triggered.connect(self.export_json)
        export_menu.addAction(export_json_action)

        file_menu.addSeparator()

        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu('&Edit')

        undo_action = QAction('&Undo', self)
        undo_action.setShortcut('Ctrl+Z')
        edit_menu.addAction(undo_action)

        redo_action = QAction('&Redo', self)
        redo_action.setShortcut('Ctrl+Y')
        edit_menu.addAction(redo_action)

        # Help menu
        help_menu = menubar.addMenu('&Help')

        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_toolbar(self):
        """Create the toolbar."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        # Open button
        open_btn = QPushButton("Open Audio")
        open_btn.clicked.connect(self.open_audio_file)
        toolbar.addWidget(open_btn)

        # Process button
        self.process_btn = QPushButton("Process Audio")
        self.process_btn.clicked.connect(self.process_audio)
        self.process_btn.setEnabled(False)
        toolbar.addWidget(self.process_btn)

        toolbar.addSeparator()

        # Export buttons
        export_midi_btn = QPushButton("Export MIDI")
        export_midi_btn.clicked.connect(self.export_midi)
        toolbar.addWidget(export_midi_btn)

        export_ccli_btn = QPushButton("Export CCLI")
        export_ccli_btn.clicked.connect(self.export_ccli)
        toolbar.addWidget(export_ccli_btn)

    def create_left_panel(self):
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # File info group
        file_group = QGroupBox("File Information")
        file_layout = QGridLayout(file_group)

        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(QLabel("Audio File:"), 0, 0)
        file_layout.addWidget(self.file_label, 0, 1)

        self.duration_label = QLabel("--")
        file_layout.addWidget(QLabel("Duration:"), 1, 0)
        file_layout.addWidget(self.duration_label, 1, 1)

        layout.addWidget(file_group)

        # Processing options group
        options_group = QGroupBox("Processing Options")
        options_layout = QGridLayout(options_group)

        # Whisper model
        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.addItems(['openai-whisper', 'faster-whisper', 'whisperx', 'mlx-whisper'])
        options_layout.addWidget(QLabel("Whisper Model:"), 0, 0)
        options_layout.addWidget(self.whisper_model_combo, 0, 1)

        # Use shorter, more compact labels and layout
        # Model size (shortened)
        options_layout.addWidget(QLabel("Whisper Size:"), 1, 0)
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItems(['large-v3-turbo', 'large-v3', 'large-v2', 'large', 'medium', 'small', 'base', 'tiny'])
        self.model_size_combo.setToolTip(
            "Whisper model size. Large-v3-turbo is fastest and most accurate."
        )
        options_layout.addWidget(self.model_size_combo, 1, 1)

        # Chord detection
        options_layout.addWidget(QLabel("Chords:"), 2, 0)
        self.chord_method_combo = QComboBox()
        self.chord_method_combo.addItems(['chordino', 'chromagram'])
        options_layout.addWidget(self.chord_method_combo, 2, 1)

        # Melody extraction
        options_layout.addWidget(QLabel("Melody:"), 3, 0)
        self.melody_method_combo = QComboBox()
        self.melody_method_combo.addItems(['basic-pitch', 'crepe'])
        index = self.melody_method_combo.findText('basic-pitch')
        if index >= 0:
            self.melody_method_combo.setCurrentIndex(index)
        options_layout.addWidget(self.melody_method_combo, 3, 1)

        # Source separation (compact)
        options_layout.addWidget(QLabel("Separation:"), 4, 0)
        self.separation_method_combo = QComboBox()
        self.separation_method_combo.addItems(['demucs', 'audio-separator'])
        index = self.separation_method_combo.findText('demucs')
        if index >= 0:
            self.separation_method_combo.setCurrentIndex(index)
        options_layout.addWidget(self.separation_method_combo, 4, 1)

        # Audio-separator model selection (compact display)
        options_layout.addWidget(QLabel("UVR Model:"), 5, 0)
        self.audio_separator_model_combo = QComboBox()

        # Use compact model names for UI
        try:
            from song_editor.core.audio_separator_processor import AudioSeparatorProcessor
            if AudioSeparatorProcessor.is_available():
                separator = AudioSeparatorProcessor()
                model_names = separator.get_models_for_ui()
                self.audio_separator_model_combo.addItems(model_names)
                # Set default to karaoke model
                for i, name in enumerate(model_names):
                    if 'KARA_2' in name:
                        self.audio_separator_model_combo.setCurrentIndex(i)
                        break
            else:
                # Fallback to basic list
                self.audio_separator_model_combo.addItems([
                    'UVR_MDXNET_KARA_2',
                    'UVR_MDXNET_21_OVERLAP_5',
                    'UVR_MDXNET_21_OVERLAP_7',
                    'UVR_MDXNET_21_OVERLAP_9'
                ])
        except Exception as e:
            logger.warning(f"Failed to load model list: {e}")
            self.audio_separator_model_combo.addItems([
                'UVR_MDXNET_KARA_2',
                'UVR_MDXNET_21_OVERLAP_5'
            ])

        # Set fixed width for combo box to prevent UI from being too wide
        self.audio_separator_model_combo.setMaximumWidth(300)
        options_layout.addWidget(self.audio_separator_model_combo, 5, 1)

        # GPU options (compact layout)
        gpu_layout = QHBoxLayout()
        self.use_cuda_check = QCheckBox("CUDA")
        self.use_cuda_check.setChecked(False)
        self.use_cuda_check.setToolTip("Enable CUDA acceleration (NVIDIA GPUs)")
        gpu_layout.addWidget(self.use_cuda_check)

        self.use_coreml_check = QCheckBox("CoreML")
        self.use_coreml_check.setChecked(True)
        self.use_coreml_check.setToolTip("Enable CoreML acceleration (Apple Silicon)")
        gpu_layout.addWidget(self.use_coreml_check)

        # Add GPU options to the grid
        options_layout.addWidget(QLabel("GPU:"), 6, 0)
        options_layout.addLayout(gpu_layout, 6, 1)

        # Legacy checkbox (hidden)
        self.use_demucs_check = QCheckBox("Use Demucs")
        self.use_demucs_check.setChecked(True)
        self.use_demucs_check.setVisible(False)

        # Save intermediate files
        self.save_intermediate_check = QCheckBox("Save intermediate files")
        self.save_intermediate_check.setChecked(True)
        options_layout.addWidget(self.save_intermediate_check, 7, 0, 1, 2)

        layout.addWidget(options_group)

        # Song info group
        song_group = QGroupBox("Song Information")
        song_layout = QGridLayout(song_group)

        self.title_edit = QLineEdit()
        song_layout.addWidget(QLabel("Title:"), 0, 0)
        song_layout.addWidget(self.title_edit, 0, 1)

        self.artist_edit = QLineEdit()
        song_layout.addWidget(QLabel("Artist:"), 1, 0)
        song_layout.addWidget(self.artist_edit, 1, 1)

        self.album_edit = QLineEdit()
        song_layout.addWidget(QLabel("Album:"), 2, 0)
        song_layout.addWidget(self.album_edit, 2, 1)

        self.genre_edit = QLineEdit()
        song_layout.addWidget(QLabel("Genre:"), 3, 0)
        song_layout.addWidget(self.genre_edit, 3, 1)

        layout.addWidget(song_group)

        # Statistics group
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout(stats_group)

        self.word_count_label = QLabel("0")
        stats_layout.addWidget(QLabel("Words:"), 0, 0)
        stats_layout.addWidget(self.word_count_label, 0, 1)

        self.chord_count_label = QLabel("0")
        stats_layout.addWidget(QLabel("Chords:"), 1, 0)
        stats_layout.addWidget(self.chord_count_label, 1, 1)

        self.note_count_label = QLabel("0")
        stats_layout.addWidget(QLabel("Notes:"), 2, 0)
        stats_layout.addWidget(self.note_count_label, 2, 1)

        layout.addWidget(stats_group)

        layout.addStretch()

        return panel

    def create_right_panel(self):
        """Create the right editor panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Lyrics editor tab
        self.lyrics_editor = self.create_lyrics_editor()
        self.tab_widget.addTab(self.lyrics_editor, "Lyrics")

        # Chord editor tab
        self.chord_editor = self.create_chord_editor()
        self.tab_widget.addTab(self.chord_editor, "Chords")

        # Melody editor tab
        self.melody_editor = self.create_melody_editor()
        self.tab_widget.addTab(self.melody_editor, "Melody")

        layout.addWidget(self.tab_widget)
        return panel

    def open_audio_file(self):
        """Open an audio file for processing with platform-aware dialog."""
        try:
            logging.info("Opening audio file dialog...")
            
            # Ensure the window is active and raised
            self.raise_()
            self.activateWindow()
            
            # Use a more robust approach that works across platforms
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open Audio File",
                str(Path.home()),  # Start from home directory
                "Audio Files (*.mp3 *.wav *.m4a *.flac *.ogg *.aac *.opus);;All Files (*)",
                options=QFileDialog.ReadOnly
            )
            
            logging.info(f"File dialog returned: {file_path}")

            if file_path:
                logging.info(f"Selected audio file: {file_path}")
                self.audio_file_path = file_path
                self.file_label.setText(Path(file_path).name)
                
                # Enable both process buttons (toolbar and panel)
                if hasattr(self, 'process_btn'):
                    self.process_btn.setEnabled(True)
                if hasattr(self, 'process_button'):
                    self.process_button.setEnabled(True)
                
                # Check for existing processed data and load it
                self.load_audio_from_path(file_path)
            else:
                logging.info("No file selected or dialog was cancelled")
                
        except Exception as e:
            logging.error(f"Error opening file dialog: {e}")
            # Fallback: show a message to help debug
            QMessageBox.warning(self, "File Dialog Error", f"Could not open file dialog: {e}")

    def process_audio(self):
        """Process the loaded audio file."""
        try:
            logging.info("Process audio button clicked")
            
            if not hasattr(self, 'audio_file_path'):
                logging.warning("No audio file path set")
                QMessageBox.warning(self, "No File", "Please select an audio file first.")
                return

            logging.info(f"Starting audio processing for: {self.audio_file_path}")

            # Disable process buttons during processing
            if hasattr(self, 'process_btn'):
                self.process_btn.setEnabled(False)
            if hasattr(self, 'process_button'):
                self.process_button.setEnabled(False)

            # Get processing configuration
            config = {
                'whisper_model': self.whisper_model_combo.currentText(),
                'model_size': self.model_size_combo.currentText(),
                'chord_method': self.chord_method_combo.currentText(),
                'melody_method': self.melody_method_combo.currentText(),
                # Audio-separator options
                'separation_engine': self.separation_method_combo.currentText().replace('-', '_'),
                'audio_separator_model': self._extract_model_name(self.audio_separator_model_combo.currentText()),
                'use_cuda': self.use_cuda_check.isChecked(),
                'use_coreml': self.use_coreml_check.isChecked(),
                # Legacy compatibility
                'use_demucs': self.separation_method_combo.currentText() == 'demucs',
                'save_intermediate': self.save_intermediate_check.isChecked(),
                'language': None  # None for auto-detection
            }

            logging.info(f"Processing configuration: {config}")

            # Start processing thread
            self.processing_thread = ProcessingThread(self.audio_file_path, config)
            self.processing_thread.progress_updated.connect(self.update_progress)
            self.processing_thread.stage_completed.connect(self.stage_completed)
            self.processing_thread.processing_finished.connect(self.processing_finished)
            
            # Actually start the thread!
            logging.info("Starting processing thread...")
            self.processing_thread.start()
            
            # Update UI to show processing has started
            self.status_bar.showMessage("Processing audio...")
            
        except Exception as e:
            logging.error(f"Error starting audio processing: {e}")
            QMessageBox.critical(self, "Processing Error", f"Failed to start processing: {e}")
            
            # Re-enable buttons on error
            if hasattr(self, 'process_btn'):
                self.process_btn.setEnabled(True)
            if hasattr(self, 'process_button'):
                self.process_button.setEnabled(True)

    def _extract_model_name(self, display_name: str) -> str:
        """
        Extract clean model name from formatted UI display text.

        Args:
            display_name: Formatted display name (e.g., "UVR_MDXNET_KARA_2 (85MB) - description")

        Returns:
            Clean model name (e.g., "UVR_MDXNET_KARA_2")
        """
        # Extract model name from format: "Model_Name (Size) - Description"
        if ' (' in display_name and ') - ' in display_name:
            return display_name.split(' (')[0]

        # Try to use AudioSeparatorProcessor method if available
        try:
            from song_editor.core.audio_separator_processor import AudioSeparatorProcessor
            if AudioSeparatorProcessor.is_available():
                separator = AudioSeparatorProcessor()
                return separator.get_model_name_from_display(display_name)
        except Exception:
            pass

        # Fallback: return as-is if parsing fails
        return display_name

    def update_progress(self, message: str, value: int):
        """Update the progress bar."""
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(message)

    def stage_completed(self, stage: str, data: Dict[str, Any]):
        """Handle stage completion."""
        logging.info(f"Stage completed: {stage}")

    def processing_finished(self, song_data: Dict[str, Any]):
        """Handle processing completion."""
        self.song_data = SongData.from_dict(song_data)

        # Update song information
        if self.song_data.metadata.get('title'):
            self.title_edit.setText(self.song_data.metadata['title'])
        if self.song_data.metadata.get('artist'):
            self.artist_edit.setText(self.song_data.metadata['artist'])
        if self.song_data.metadata.get('album'):
            self.album_edit.setText(self.song_data.metadata['album'])
        if self.song_data.metadata.get('genre'):
            self.genre_edit.setText(self.song_data.metadata['genre'])

        # Update statistics
        self.word_count_label.setText(str(self.song_data.get_word_count()))
        self.chord_count_label.setText(str(self.song_data.get_chord_count()))
        self.note_count_label.setText(str(self.song_data.get_note_count()))

        # Update duration
        duration = self.song_data.get_duration()
        if duration > 0:
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            self.duration_label.setText(f"{minutes}:{seconds:02d}")

        # Update editors
        self.lyrics_editor.set_song_data(self.song_data)
        self.chord_editor.set_song_data(self.song_data)
        self.melody_editor.set_song_data(self.song_data)

        # Update UI
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(
            f"Processing complete • Words: {self.song_data.get_word_count()} • Chords: {self.song_data.get_chord_count()} • Notes: {self.song_data.get_note_count()}"
        )

    def processing_error(self, error_message: str):
        """Handle processing error."""
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Processing failed!")

        QMessageBox.critical(
            self,
            "Processing Error",
            f"An error occurred during processing:\n{error_message}"
        )

    def save_song_data(self):
        """Save the song data to a JSON file."""
        if not self.song_data:
            QMessageBox.warning(self, "No Data", "No song data to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Song Data",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            if self.song_data.save_json(file_path):
                self.status_bar.showMessage(f"Saved: {Path(file_path).name}")
            else:
                QMessageBox.critical(self, "Save Error", "Failed to save song data.")

    def export_midi(self):
        """Export song data to MIDI format."""
        if not self.song_data:
            QMessageBox.warning(self, "No Data", "No song data to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export MIDI",
            "",
            "MIDI Files (*.mid);;All Files (*)"
        )

        if file_path:
            exporter = MidiExporter()
            if exporter.export(self.song_data.to_dict(), file_path):
                self.status_bar.showMessage(f"Exported MIDI: {Path(file_path).name}")
            else:
                QMessageBox.critical(self, "Export Error", "Failed to export MIDI.")

    def export_ccli(self):
        """Export song data to CCLI format."""
        if not self.song_data:
            QMessageBox.warning(self, "No Data", "No song data to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export CCLI",
            "",
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            exporter = CCLIExporter()
            if exporter.export(self.song_data.to_dict(), file_path):
                self.status_bar.showMessage(f"Exported CCLI: {Path(file_path).name}")
            else:
                QMessageBox.critical(self, "Export Error", "Failed to export CCLI.")

    def export_json(self):
        """Export song data to JSON format."""
        if not self.song_data:
            QMessageBox.warning(self, "No Data", "No song data to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export JSON",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            exporter = JSONExporter()
            if exporter.export(self.song_data.to_dict(), file_path):
                self.status_bar.showMessage(f"Exported JSON: {Path(file_path).name}")
            else:
                QMessageBox.critical(self, "Export Error", "Failed to export JSON.")

    def show_about(self):
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About Song Editor 3",
            "Song Editor 3\n\n"
            "A comprehensive audio processing and song analysis tool.\n\n"
            "Features:\n"
            "• Audio transcription with multiple Whisper models\n"
            "• Chord detection with Chordino\n"
            "• Melody extraction with Basic Pitch/CREPE\n"
            "• Source separation with Demucs\n"
            "• Export to MIDI, CCLI, and JSON formats\n\n"
            "Version 3.0.0"
        )

    def load_settings(self):
        """Load application settings."""
        # Load window geometry
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)

        # Load processing options
        whisper_model = self.settings.value('whisper_model', 'openai-whisper')
        index = self.whisper_model_combo.findText(whisper_model)
        if index >= 0:
            self.whisper_model_combo.setCurrentIndex(index)

        model_size = self.settings.value('model_size', 'large-v3-turbo')
        index = self.model_size_combo.findText(model_size)
        if index >= 0:
            self.model_size_combo.setCurrentIndex(index)

        chord_method = self.settings.value('chord_method', 'chordino')
        index = self.chord_method_combo.findText(chord_method)
        if index >= 0:
            self.chord_method_combo.setCurrentIndex(index)

        melody_method = self.settings.value('melody_method', 'basic-pitch')
        index = self.melody_method_combo.findText(melody_method)
        if index >= 0:
            self.melody_method_combo.setCurrentIndex(index)

        # Load audio-separator settings
        separation_engine = self.settings.value('separation_engine', 'demucs')
        # Convert underscore back to hyphen for UI display
        ui_separation_engine = separation_engine.replace('_', '-')
        index = self.separation_method_combo.findText(ui_separation_engine)
        if index >= 0:
            self.separation_method_combo.setCurrentIndex(index)

        # Load audio-separator model - handle both clean names and formatted display names
        saved_model = self.settings.value('audio_separator_model', 'UVR_MDXNET_KARA_2')

        # Find the correct index for the saved model in the formatted list
        found_index = -1
        for i in range(self.audio_separator_model_combo.count()):
            display_text = self.audio_separator_model_combo.itemText(i)
            clean_name = self._extract_model_name(display_text)
            if clean_name == saved_model:
                found_index = i
                break

        if found_index >= 0:
            self.audio_separator_model_combo.setCurrentIndex(found_index)
        else:
            # Fallback: try to find by partial match
            for i in range(self.audio_separator_model_combo.count()):
                display_text = self.audio_separator_model_combo.itemText(i)
                if saved_model in display_text:
                    self.audio_separator_model_combo.setCurrentIndex(i)
                    break

        self.use_cuda_check.setChecked(self.settings.value('use_cuda', False, type=bool))
        self.use_coreml_check.setChecked(self.settings.value('use_coreml', True, type=bool))

        # Legacy compatibility
        self.use_demucs_check.setChecked(self.settings.value('use_demucs', True, type=bool))
        self.save_intermediate_check.setChecked(self.settings.value('save_intermediate', True, type=bool))

    def save_settings(self):
        """Save application settings."""
        # Save window geometry
        self.settings.setValue('geometry', self.saveGeometry())

        # Save processing options
        self.settings.setValue('whisper_model', self.whisper_model_combo.currentText())
        self.settings.setValue('model_size', self.model_size_combo.currentText())
        self.settings.setValue('chord_method', self.chord_method_combo.currentText())
        self.settings.setValue('melody_method', self.melody_method_combo.currentText())
        # Audio-separator settings
        self.settings.setValue('separation_engine', self.separation_method_combo.currentText().replace('-', '_'))
        # Save clean model name (not formatted display name)
        clean_model_name = self._extract_model_name(self.audio_separator_model_combo.currentText())
        self.settings.setValue('audio_separator_model', clean_model_name)
        self.settings.setValue('use_cuda', self.use_cuda_check.isChecked())
        self.settings.setValue('use_coreml', self.use_coreml_check.isChecked())
        # Legacy compatibility
        self.settings.setValue('use_demucs', self.separation_method_combo.currentText() == 'demucs')
        self.settings.setValue('save_intermediate', self.save_intermediate_check.isChecked())

    def load_audio_from_path(self, audio_path: str):
        """Load an audio file and check for existing processed data."""
        if not audio_path:
            return


        audio_path = Path(audio_path)

        # If path is not absolute, try resolving relative to current working directory
        # (this handles when app is launched from command line with relative paths)
        if not audio_path.is_absolute():
            # Try the current working directory first
            cwd_path = Path.cwd() / audio_path
            logging.info(f"Trying relative path: {cwd_path}")

            if cwd_path.exists():
                audio_path = cwd_path
                logging.info(f"Found file at: {audio_path}")
            else:
                logging.error(f"File not found at: {cwd_path}")
                QMessageBox.warning(
                    self,
                    "File Not Found",
                    f"Audio file not found: {audio_path}\n"
                    f"Also tried: {cwd_path}"
                )
                return
        elif not audio_path.exists():
            logging.error(f"Absolute path not found: {audio_path}")
            QMessageBox.warning(self, "File Not Found", f"Audio file not found: {audio_path}")
            return

        # Set the audio file path
        self.audio_file_path = str(audio_path)
        self.file_label.setText(audio_path.name)
        self.process_btn.setEnabled(True)
        self.status_bar.showMessage(f"Loaded: {audio_path.name}")

        # Check for existing processed data
        base_name = audio_path.stem
        json_path = audio_path.parent / f"{base_name}.song_data.json"

        if json_path.exists():
            try:
                # Load existing processed data
                with open(json_path, 'r', encoding='utf-8') as f:
                    song_data = json.load(f)

                logging.info(f"Loading existing processed data from: {json_path}")

                # Update UI with loaded data
                self._display_loaded_data(song_data, json_path)

                # Status bar summary instead of popup
                self.status_bar.showMessage(
                    f"Loaded existing data • Words: {len(song_data.get('words', []))} • Chords: {len(song_data.get('chords', []))} • Notes: {len(song_data.get('notes', []))}"
                )

            except Exception as e:
                logging.error(f"Error loading existing data: {e}")
                QMessageBox.warning(
                    self,
                    "Data Load Error",
                    f"Could not load existing processed data:\n{str(e)}\n\n"
                    "The file will need to be re-processed."
                )
        else:
            logging.info(f"No existing processed data found for: {audio_path.name}")
            self.status_bar.showMessage(f"Loaded: {audio_path.name} (not processed yet)")

    def _display_loaded_data(self, song_data: Dict[str, Any], json_path: Path):
        """Display loaded song data in the UI."""
        try:
            # Create SongData object from loaded data
            from ..models.song_data import SongData
            loaded_song_data = SongData.from_dict(song_data)

            # Determine source audio path (prefer metadata, then current file)
            metadata = song_data.get('metadata', {})
            source_audio = metadata.get('source_audio')
            try:
                from pathlib import Path as _Path
                if source_audio and _Path(source_audio).exists():
                    _audio_path_for_player = _Path(source_audio)
                elif hasattr(self, 'audio_file_path'):
                    _audio_path_for_player = _Path(self.audio_file_path)
                else:
                    _audio_path_for_player = None
            except Exception:
                _audio_path_for_player = None

            # Update lyrics editor
            if hasattr(self, 'lyrics_editor'):
                self.lyrics_editor.set_song_data(loaded_song_data)
                # Set audio path for enhanced lyrics editor playback features
                if hasattr(self.lyrics_editor, 'set_audio_path'):
                    if _audio_path_for_player is not None:
                        self.lyrics_editor.set_audio_path(str(_audio_path_for_player))

            # Update chord editor
            if hasattr(self, 'chord_editor'):
                self.chord_editor.set_song_data(loaded_song_data)

            # Update melody editor
            if hasattr(self, 'melody_editor'):
                self.melody_editor.set_song_data(loaded_song_data)

            # Update statistics panel like in processing_finished
            try:
                if hasattr(self, 'word_count_label'):
                    self.word_count_label.setText(str(loaded_song_data.get_word_count()))
                if hasattr(self, 'chord_count_label'):
                    self.chord_count_label.setText(str(loaded_song_data.get_chord_count()))
                if hasattr(self, 'note_count_label'):
                    self.note_count_label.setText(str(loaded_song_data.get_note_count()))

                duration = loaded_song_data.get_duration()
                if duration and hasattr(self, 'duration_label'):
                    minutes = int(duration // 60)
                    seconds = int(duration % 60)
                    self.duration_label.setText(f"{minutes}:{seconds:02d}")
            except Exception:
                pass

            # Update metadata display
            metadata = song_data.get('metadata', {})
            source_audio = metadata.get('source_audio', 'Unknown')
            created_at = metadata.get('created_at', 'Unknown')

            self.status_bar.showMessage(
                f"Loaded processed data from {json_path.name} "
                f"(created: {created_at})"
            )

        except Exception as e:
            logging.error(f"Error displaying loaded data: {e}")
            QMessageBox.warning(
                self,
                "Display Error",
                f"Could not display loaded data:\n{str(e)}"
            )

    def closeEvent(self, event):
        """Handle window close event."""
        self.save_settings()
        event.accept()
