#!/usr/bin/env python3
"""
Processing Options Dialog

Dialog window for configuring audio processing options.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QCheckBox, QDialogButtonBox
)
from PySide6.QtCore import Qt, QSettings


class ProcessingOptionsDialog(QDialog):
    """Dialog for configuring processing options before starting audio processing."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Options")
        self.setModal(True)
        self.resize(500, 400)
        
        # Load settings
        self.settings = QSettings('SongEditor3', 'SongEditor3')
        
        self.setup_ui()
        self.load_settings()
    
    def setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Transcription options
        transcription_group = QGroupBox("Transcription Options")
        transcription_layout = QGridLayout(transcription_group)
        
        # Whisper model
        transcription_layout.addWidget(QLabel("Whisper Model:"), 0, 0)
        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.addItems(['openai-whisper', 'faster-whisper', 'whisperx', 'mlx-whisper'])
        self.whisper_model_combo.setCurrentText('faster-whisper')
        transcription_layout.addWidget(self.whisper_model_combo, 0, 1)
        
        # Model size
        transcription_layout.addWidget(QLabel("Model Size:"), 1, 0)
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItems(['large-v3-turbo', 'large-v3', 'large-v2', 'large', 'medium', 'small', 'base', 'tiny'])
        self.model_size_combo.setCurrentText('large-v3-turbo')
        self.model_size_combo.setToolTip("Large-v3-turbo is fastest and most accurate.")
        transcription_layout.addWidget(self.model_size_combo, 1, 1)
        
        layout.addWidget(transcription_group)
        
        # Analysis options
        analysis_group = QGroupBox("Music Analysis Options")
        analysis_layout = QGridLayout(analysis_group)
        
        # Chord detection
        analysis_layout.addWidget(QLabel("Chord Detection:"), 0, 0)
        self.chord_method_combo = QComboBox()
        self.chord_method_combo.addItems(['chordino', 'chromagram'])
        self.chord_method_combo.setCurrentText('chromagram')
        analysis_layout.addWidget(self.chord_method_combo, 0, 1)
        
        # Melody extraction
        analysis_layout.addWidget(QLabel("Melody Extraction:"), 1, 0)
        self.melody_method_combo = QComboBox()
        self.melody_method_combo.addItems(['basic-pitch', 'crepe'])
        self.melody_method_combo.setCurrentText('basic-pitch')
        analysis_layout.addWidget(self.melody_method_combo, 1, 1)
        
        layout.addWidget(analysis_group)
        
        # Source separation options
        separation_group = QGroupBox("Audio Separation Options")
        separation_layout = QGridLayout(separation_group)
        
        # Separation engine
        separation_layout.addWidget(QLabel("Separation Engine:"), 0, 0)
        self.separation_method_combo = QComboBox()
        self.separation_method_combo.addItems(['demucs', 'audio-separator'])
        self.separation_method_combo.setCurrentText('demucs')
        separation_layout.addWidget(self.separation_method_combo, 0, 1)
        
        # Audio-separator model
        separation_layout.addWidget(QLabel("UVR Model:"), 1, 0)
        self.audio_separator_model_combo = QComboBox()
        
        # Load UVR models
        try:
            from ..core.audio_separator_processor import AudioSeparatorProcessor
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
                self.audio_separator_model_combo.addItems([
                    'UVR_MDXNET_KARA_2',
                    'UVR_MDXNET_21_OVERLAP_5',
                    'UVR_MDXNET_21_OVERLAP_7',
                    'UVR_MDXNET_21_OVERLAP_9'
                ])
                self.audio_separator_model_combo.setCurrentText('UVR_MDXNET_KARA_2')
        except Exception as e:
            self.audio_separator_model_combo.addItems([
                'UVR_MDXNET_KARA_2',
                'UVR_MDXNET_21_OVERLAP_5'
            ])
            self.audio_separator_model_combo.setCurrentText('UVR_MDXNET_KARA_2')
        
        separation_layout.addWidget(self.audio_separator_model_combo, 1, 1)
        
        layout.addWidget(separation_group)
        
        # Hardware acceleration options
        hardware_group = QGroupBox("Hardware Acceleration")
        hardware_layout = QGridLayout(hardware_group)
        
        # GPU options
        hardware_layout.addWidget(QLabel("GPU Acceleration:"), 0, 0)
        gpu_layout = QHBoxLayout()
        
        self.use_cuda_check = QCheckBox("CUDA")
        self.use_cuda_check.setChecked(False)
        self.use_cuda_check.setToolTip("Enable CUDA acceleration (NVIDIA GPUs)")
        gpu_layout.addWidget(self.use_cuda_check)
        
        self.use_coreml_check = QCheckBox("CoreML")
        self.use_coreml_check.setChecked(True)
        self.use_coreml_check.setToolTip("Enable CoreML acceleration (Apple Silicon)")
        gpu_layout.addWidget(self.use_coreml_check)
        
        gpu_layout.addStretch()
        hardware_layout.addLayout(gpu_layout, 0, 1)
        
        layout.addWidget(hardware_group)
        
        # Additional options
        additional_group = QGroupBox("Additional Options")
        additional_layout = QVBoxLayout(additional_group)
        
        self.save_intermediate_check = QCheckBox("Save intermediate audio files")
        self.save_intermediate_check.setChecked(False)
        self.save_intermediate_check.setToolTip("Save separated audio files (vocals, accompaniment, etc.)")
        additional_layout.addWidget(self.save_intermediate_check)
        
        layout.addWidget(additional_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        # Rename OK button to "Start Processing"
        ok_button = button_box.button(QDialogButtonBox.Ok)
        ok_button.setText("Start Processing")
        
        layout.addWidget(button_box)
    
    def get_config(self) -> dict:
        """Get the current configuration from the dialog."""
        return {
            'whisper_model': self.whisper_model_combo.currentText(),
            'model_size': self.model_size_combo.currentText(),
            'chord_method': self.chord_method_combo.currentText(),
            'melody_method': self.melody_method_combo.currentText(),
            'separation_engine': self.separation_method_combo.currentText(),
            'audio_separator_model': self.audio_separator_model_combo.currentText(),
            'use_cuda': self.use_cuda_check.isChecked(),
            'use_coreml': self.use_coreml_check.isChecked(),
            'save_intermediate': self.save_intermediate_check.isChecked(),
            'use_demucs': True,  # Legacy compatibility
            'language': None
        }
    
    def load_settings(self):
        """Load settings from QSettings."""
        self.whisper_model_combo.setCurrentText(
            self.settings.value('whisper_model', 'faster-whisper')
        )
        self.model_size_combo.setCurrentText(
            self.settings.value('model_size', 'large-v3-turbo')
        )
        self.chord_method_combo.setCurrentText(
            self.settings.value('chord_method', 'chromagram')
        )
        self.melody_method_combo.setCurrentText(
            self.settings.value('melody_method', 'basic-pitch')
        )
        self.separation_method_combo.setCurrentText(
            self.settings.value('separation_engine', 'demucs')
        )
        self.audio_separator_model_combo.setCurrentText(
            self.settings.value('audio_separator_model', 'UVR_MDXNET_KARA_2')
        )
        self.use_cuda_check.setChecked(
            self.settings.value('use_cuda', False, type=bool)
        )
        self.use_coreml_check.setChecked(
            self.settings.value('use_coreml', True, type=bool)
        )
        self.save_intermediate_check.setChecked(
            self.settings.value('save_intermediate', False, type=bool)
        )
    
    def save_settings(self):
        """Save settings to QSettings."""
        self.settings.setValue('whisper_model', self.whisper_model_combo.currentText())
        self.settings.setValue('model_size', self.model_size_combo.currentText())
        self.settings.setValue('chord_method', self.chord_method_combo.currentText())
        self.settings.setValue('melody_method', self.melody_method_combo.currentText())
        self.settings.setValue('separation_engine', self.separation_method_combo.currentText())
        self.settings.setValue('audio_separator_model', self.audio_separator_model_combo.currentText())
        self.settings.setValue('use_cuda', self.use_cuda_check.isChecked())
        self.settings.setValue('use_coreml', self.use_coreml_check.isChecked())
        self.settings.setValue('save_intermediate', self.save_intermediate_check.isChecked())
    
    def accept(self):
        """Accept dialog and save settings."""
        self.save_settings()
        super().accept()
