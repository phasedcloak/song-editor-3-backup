#!/usr/bin/env python3
"""
Centralized Configuration Management for Song Editor 3

Provides a unified interface for configuration management across the application.
Supports loading from JSON files, environment variables, and runtime overrides.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class AppConfig:
    """Application configuration."""
    name: str = "Song Editor 3"
    version: str = "3.0.0"
    debug: bool = False
    log_level: str = "INFO"


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    default_sample_rate: int = 44100
    default_channels: int = 2
    buffer_size: int = 1024
    chunk_size: int = 4096
    supported_formats: list = field(default_factory=lambda: ["wav", "mp3", "flac", "m4a", "aac", "ogg"])


@dataclass
class TranscriptionConfig:
    """Transcription configuration."""
    default_model: str = "openai-whisper"
    default_model_size: str = "large-v2"
    default_language: str = "en"
    confidence_threshold: float = 0.5
    max_audio_length: float = 300.0


@dataclass
class ChordDetectionConfig:
    """Chord detection configuration."""
    default_method: str = "chordino"
    confidence_threshold: float = 0.6
    min_chord_duration: float = 0.1


@dataclass
class MelodyExtractionConfig:
    """Melody extraction configuration."""
    default_method: str = "basic-pitch"
    confidence_threshold: float = 0.7
    min_note_duration: float = 0.05


@dataclass
class SourceSeparationConfig:
    """Source separation configuration."""
    default_model: str = "demucs"
    separate_tracks: list = field(default_factory=lambda: ["vocals", "drums", "bass", "other"])
    sample_rate: int = 44100


@dataclass
class ExportConfig:
    """Export configuration."""
    default_format: str = "json"
    include_metadata: bool = True
    include_analysis: bool = True
    compression: bool = False


@dataclass
class UIConfig:
    """UI configuration."""
    theme: str = "default"
    window_size: list = field(default_factory=lambda: [1200, 800])
    auto_save_interval: int = 300
    show_tooltips: bool = True


@dataclass
class PathsConfig:
    """Paths configuration."""
    models_dir: str = "models"
    temp_dir: str = "temp"
    output_dir: str = "output"
    cache_dir: str = "cache"


@dataclass
class Config:
    """Main configuration container."""
    app: AppConfig = field(default_factory=AppConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    chord_detection: ChordDetectionConfig = field(default_factory=ChordDetectionConfig)
    melody_extraction: MelodyExtractionConfig = field(default_factory=MelodyExtractionConfig)
    source_separation: SourceSeparationConfig = field(default_factory=SourceSeparationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


class ConfigManager:
    """Centralized configuration manager."""

    def __init__(self, config_file: Optional[str] = None):
        self.config = Config()
        self.config_file = config_file or self._find_config_file()
        self.load_config()

    def _find_config_file(self) -> str:
        """Find the configuration file."""
        # Check current directory first
        if Path("config.json").exists():
            return "config.json"

        # Check in song_editor directory
        if Path("song_editor/config.json").exists():
            return "song_editor/config.json"

        # Use default
        return "config.json"

    def load_config(self) -> None:
        """Load configuration from file and environment variables."""
        # Load from JSON file
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._update_from_dict(data)
                logging.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logging.warning(f"Failed to load config from {self.config_file}: {e}")

        # Override with environment variables
        self._load_from_env()

    def _update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for section_name, section_data in data.items():
            if hasattr(self.config, section_name):
                section = getattr(self.config, section_name)
                if hasattr(section, '__dict__'):
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # App settings
        if "SONG_EDITOR_DEBUG" in os.environ:
            self.config.app.debug = os.environ["SONG_EDITOR_DEBUG"].lower() == "true"
        if "SONG_EDITOR_LOG_LEVEL" in os.environ:
            self.config.app.log_level = os.environ["SONG_EDITOR_LOG_LEVEL"]

        # Audio settings
        if "SONG_EDITOR_SAMPLE_RATE" in os.environ:
            self.config.audio.default_sample_rate = int(os.environ["SONG_EDITOR_SAMPLE_RATE"])

        # Transcription settings
        if "SONG_EDITOR_WHISPER_MODEL" in os.environ:
            self.config.transcription.default_model = os.environ["SONG_EDITOR_WHISPER_MODEL"]
        if "SONG_EDITOR_WHISPER_MODEL_SIZE" in os.environ:
            self.config.transcription.default_model_size = os.environ["SONG_EDITOR_WHISPER_MODEL_SIZE"]
        if "SONG_EDITOR_LANGUAGE" in os.environ:
            self.config.transcription.default_language = os.environ["SONG_EDITOR_LANGUAGE"]

    def save_config(self, file_path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        save_path = file_path or self.config_file
        try:
            data = self._config_to_dict()
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved configuration to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save config to {save_path}: {e}")

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for section_name in ['app', 'audio', 'transcription', 'chord_detection',
                             'melody_extraction', 'source_separation', 'export', 'ui', 'paths']:
            if hasattr(self.config, section_name):
                section = getattr(self.config, section_name)
                if hasattr(section, '__dict__'):
                    result[section_name] = section.__dict__.copy()
        return result

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if hasattr(self.config, section):
            section_obj = getattr(self.config, section)
            if hasattr(section_obj, key):
                return getattr(section_obj, key)
        return default

    def set(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value."""
        if hasattr(self.config, section):
            section_obj = getattr(self.config, section)
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)

    def get_section(self, section: str) -> Optional[Any]:
        """Get a configuration section."""
        if hasattr(self.config, section):
            return getattr(self.config, section)
        return None


# Global configuration instance
_config_manager = None


def get_config() -> ConfigManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def init_config(config_file: Optional[str] = None) -> ConfigManager:
    """Initialize the global configuration manager."""
    global _config_manager
    _config_manager = ConfigManager(config_file)
    return _config_manager


# Convenience functions for common config access
def get_app_config() -> AppConfig:
    """Get application configuration."""
    return get_config().config.app


def get_audio_config() -> AudioConfig:
    """Get audio configuration."""
    return get_config().config.audio


def get_transcription_config() -> TranscriptionConfig:
    """Get transcription configuration."""
    return get_config().config.transcription


def get_chord_detection_config() -> ChordDetectionConfig:
    """Get chord detection configuration."""
    return get_config().config.chord_detection


def get_melody_extraction_config() -> MelodyExtractionConfig:
    """Get melody extraction configuration."""
    return get_config().config.melody_extraction


def get_source_separation_config() -> SourceSeparationConfig:
    """Get source separation configuration."""
    return get_config().config.source_separation


def get_export_config() -> ExportConfig:
    """Get export configuration."""
    return get_config().config.export


def get_ui_config() -> UIConfig:
    """Get UI configuration."""
    return get_config().config.ui


def get_paths_config() -> PathsConfig:
    """Get paths configuration."""
    return get_config().config.paths
