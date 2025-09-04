#!/usr/bin/env python3
"""
Dependency Management for Song Editor 3

Handles optional dependencies gracefully and provides fallback functionality.
"""

import logging
import importlib
from typing import Dict, Any, Optional


class DependencyManager:
    """Manages optional dependencies and provides fallbacks."""

    def __init__(self):
        self._dependencies: Dict[str, Dict[str, Any]] = {}
        self._setup_dependencies()

    def _setup_dependencies(self) -> None:
        """Set up dependency configurations."""
        self._dependencies = {
            'torch': {
                'module': 'torch',
                'required_for': ['Demucs source separation'],
                'fallback_message': 'Torch not available - source separation disabled',
                'available': False,
                'version': None
            },
            'torchaudio': {
                'module': 'torchaudio',
                'required_for': ['Audio processing with Torch'],
                'fallback_message': 'Torchaudio not available - using librosa fallback',
                'available': False,
                'version': None
            },
            'demucs': {
                'module': 'demucs.separate',
                'required_for': ['Source separation (HTDemucs)'],
                'fallback_message': 'Demucs not available - using basic source separation',
                'available': False,
                'version': None
            },
            'whisper': {
                'module': 'whisper',
                'required_for': ['OpenAI Whisper transcription'],
                'fallback_message': 'OpenAI Whisper not available - using faster-whisper',
                'available': False,
                'version': None
            },
            'faster_whisper': {
                'module': 'faster_whisper',
                'required_for': ['Faster Whisper transcription'],
                'fallback_message': 'Faster Whisper not available - basic transcription disabled',
                'available': False,
                'version': None
            },
            'whisperx': {
                'module': 'whisperx',
                'required_for': ['WhisperX transcription'],
                'fallback_message': 'WhisperX not available - using alternative transcription',
                'available': False,
                'version': None
            },
            'mlx_whisper': {
                'module': 'mlx_whisper',
                'required_for': ['MLX Whisper transcription (Apple Silicon)'],
                'fallback_message': 'MLX Whisper not available - using CPU-based transcription',
                'available': False,
                'version': None
            },
            'chord_extractor': {
                'module': 'chord_extractor',
                'required_for': ['Chord data structures and utilities'],
                'fallback_message': 'Chord Extractor available for data structures',
                'available': False,
                'version': None
            },
            'basic_pitch': {
                'module': 'basic_pitch',
                'required_for': ['Melody extraction'],
                'fallback_message': 'Basic Pitch not available - melody extraction disabled',
                'available': False,
                'version': None
            },
            'crepe': {
                'module': 'crepe',
                'required_for': ['CREPE pitch detection'],
                'fallback_message': 'CREPE not available - using basic pitch detection',
                'available': False,
                'version': None
            },
            'librosa': {
                'module': 'librosa',
                'required_for': ['Audio analysis and processing'],
                'fallback_message': 'Librosa not available - audio processing limited',
                'available': False,
                'version': None
            },
            'noisereduce': {
                'module': 'noisereduce',
                'required_for': ['Audio denoising'],
                'fallback_message': 'NoiseReduce not available - denoising disabled',
                'available': False,
                'version': None
            },
            'pyloudnorm': {
                'module': 'pyloudnorm',
                'required_for': ['Audio normalization'],
                'fallback_message': 'PyLoudNorm not available - using RMS normalization',
                'available': False,
                'version': None
            },
            'vamp': {
                'module': 'vamp',
                'required_for': ['VAMP plugins for chord detection'],
                'fallback_message': 'VAMP not available - chord detection limited',
                'available': False,
                'version': None
            },
            'mido': {
                'module': 'mido',
                'required_for': ['MIDI file handling'],
                'fallback_message': 'Mido not available - MIDI export disabled',
                'available': False,
                'version': None
            },
            'pretty_midi': {
                'module': 'pretty_midi',
                'required_for': ['Advanced MIDI processing'],
                'fallback_message': 'PrettyMIDI not available - basic MIDI export only',
                'available': False,
                'version': None
            }
        }

        # Check availability of all dependencies
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check which dependencies are available."""
        for name, config in self._dependencies.items():
            try:
                module = importlib.import_module(config['module'])
                config['available'] = True

                # Try to get version
                try:
                    if hasattr(module, '__version__'):
                        config['version'] = module.__version__
                    elif hasattr(module, 'version'):
                        config['version'] = module.version
                    else:
                        config['version'] = 'unknown'
                except (AttributeError, TypeError):
                    config['version'] = 'unknown'

                logging.debug(f"Dependency {name} available (version: {config['version']})")

            except ImportError:
                config['available'] = False
                logging.debug(f"Dependency {name} not available: {config['fallback_message']}")

    def is_available(self, name: str) -> bool:
        """Check if a dependency is available."""
        return self._dependencies.get(name, {}).get('available', False)

    def get_version(self, name: str) -> Optional[str]:
        """Get the version of a dependency."""
        return self._dependencies.get(name, {}).get('version')

    def get_fallback_message(self, name: str) -> str:
        """Get the fallback message for a dependency."""
        return self._dependencies.get(name, {}).get('fallback_message', f'{name} not available')

    def get_required_for(self, name: str) -> str:
        """Get what a dependency is required for."""
        return self._dependencies.get(name, {}).get('required_for', 'Unknown functionality')

    def get_unavailable_dependencies(self) -> Dict[str, str]:
        """Get all unavailable dependencies with their fallback messages."""
        unavailable = {}
        for name, config in self._dependencies.items():
            if not config['available']:
                unavailable[name] = config['fallback_message']
        return unavailable

    def get_available_dependencies(self) -> Dict[str, str]:
        """Get all available dependencies with their versions."""
        available = {}
        for name, config in self._dependencies.items():
            if config['available']:
                available[name] = config.get('version', 'unknown')
        return available

    def safe_import(self, name: str, fallback: Any = None) -> Any:
        """Safely import a module with fallback."""
        if self.is_available(name):
            try:
                return importlib.import_module(self._dependencies[name]['module'])
            except ImportError:
                pass

        if fallback is not None:
            return fallback

        # Create a dummy module that raises informative errors
        class DummyModule:
            def __init__(self, dep_name: str):
                self._dep_name = dep_name

            def __getattr__(self, name: str):
                raise ImportError(
                    f"{self._dep_name} is not available. "
                    f"{self.get_fallback_message(self._dep_name)}"
                )

        return DummyModule(name)

    def log_dependency_status(self) -> None:
        """Log the status of all dependencies."""
        logging.info("Dependency Status:")
        logging.info("=" * 50)

        available = self.get_available_dependencies()
        if available:
            logging.info("Available dependencies:")
            for name, version in available.items():
                logging.info(f"  ✓ {name} (v{version})")
        else:
            logging.info("No optional dependencies available")

        unavailable = self.get_unavailable_dependencies()
        if unavailable:
            logging.info("\nUnavailable dependencies:")
            for name, message in unavailable.items():
                logging.info(f"  ✗ {name}: {message}")
        else:
            logging.info("All optional dependencies available!")

        logging.info("=" * 50)


# Global dependency manager instance
_dependency_manager = None


def get_dependency_manager() -> DependencyManager:
    """Get the global dependency manager."""
    global _dependency_manager
    if _dependency_manager is None:
        _dependency_manager = DependencyManager()
    return _dependency_manager


def init_dependency_manager() -> DependencyManager:
    """Initialize the global dependency manager."""
    global _dependency_manager
    _dependency_manager = DependencyManager()
    return _dependency_manager


# Convenience functions
def is_dependency_available(name: str) -> bool:
    """Check if a dependency is available."""
    return get_dependency_manager().is_available(name)


def safe_import(name: str, fallback: Any = None) -> Any:
    """Safely import a module with fallback."""
    return get_dependency_manager().safe_import(name, fallback)


def get_dependency_version(name: str) -> Optional[str]:
    """Get the version of a dependency."""
    return get_dependency_manager().get_version(name)


def log_dependency_status() -> None:
    """Log the status of all dependencies."""
    get_dependency_manager().log_dependency_status()
