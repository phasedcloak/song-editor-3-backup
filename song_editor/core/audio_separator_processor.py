#!/usr/bin/env python3
"""
Audio Separator Processor
Integrates audio-separator library for advanced source separation with GPU acceleration.

This module provides an alternative to Demucs with better cross-platform GPU support
and more separation models, offering superior performance on Apple Silicon and NVIDIA GPUs.
"""

import os
import logging
import tempfile
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional import for audio-separator
AUDIO_SEPARATOR_AVAILABLE = False
Separator = None

# Try multiple import strategies
try:
    from audio_separator import Separator
    AUDIO_SEPARATOR_AVAILABLE = True
except ImportError:
    try:
        # Try importing from system Python path
        import sys
        import os
        # Add system Python site-packages to path
        system_site_packages = '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages'
        if system_site_packages not in sys.path:
            sys.path.insert(0, system_site_packages)

        from audio_separator import Separator
        AUDIO_SEPARATOR_AVAILABLE = True
        logger.info("Audio-separator loaded from system Python path")
    except ImportError:
        try:
            # Try with different import path
            import audio_separator
            Separator = audio_separator.Separator
            AUDIO_SEPARATOR_AVAILABLE = True
        except (ImportError, AttributeError):
            AUDIO_SEPARATOR_AVAILABLE = False
            Separator = None
            logger.warning("Audio-separator library not available")


class AudioSeparatorProcessor:
    """
    Audio source separation processor using audio-separator library.

    Provides high-performance audio separation with GPU acceleration across platforms:
    - Apple Silicon: CoreML acceleration
    - NVIDIA GPUs: CUDA acceleration
    - AMD GPUs: DirectML support
    - CPU fallback for systems without GPU
    """

    # Popular UVR models for different use cases
    RECOMMENDED_MODELS = {
        # Best for vocals/instrumental separation (karaoke)
        'karaoke': 'UVR_MDXNET_KARA_2',

        # High quality vocal separation
        'vocals': 'UVR_MDXNET_KARA_2',

        # Instrumental separation
        'instrumental': 'UVR_MDXNET_KARA_2',

        # Multi-instrument separation (drums, bass, piano, guitar)
        'multi_stem': 'UVR_MDXNET_21_OVERLAP_9',

        # Voice isolation (removes everything except vocals)
        'voice_only': 'UVR_MDXNET_KARA_2',

        # High quality for music production
        'hq_music': 'UVR_MDXNET_21_OVERLAP_9',
    }

    def __init__(
        self,
        model_name: str = 'UVR_MDXNET_KARA_2',
        output_dir: Optional[str] = None,
        use_cuda: bool = False,
        use_coreml: bool = True,  # Enable for Apple Silicon
        log_level: int = 20,  # INFO level
        normalization_enabled: bool = True,
        denoise_enabled: bool = True,
        output_format: str = 'WAV'
    ):
        """
        Initialize AudioSeparatorProcessor.

        Args:
            model_name: UVR model to use for separation
            output_dir: Directory to save separated files (None = auto)
            use_cuda: Enable CUDA acceleration (NVIDIA GPUs)
            use_coreml: Enable CoreML acceleration (Apple Silicon)
            log_level: Logging verbosity level
            normalization_enabled: Enable audio normalization
            denoise_enabled: Enable denoising
            output_format: Output audio format (WAV, MP3, FLAC)
        """
        if not AUDIO_SEPARATOR_AVAILABLE:
            raise ImportError(
                "audio-separator library is not installed. "
                "Install with: pip install 'audio-separator[gpu]'"
            )

        self.model_name = model_name
        self.output_dir = output_dir
        self.use_cuda = use_cuda
        self.use_coreml = use_coreml
        self.log_level = log_level
        self.normalization_enabled = normalization_enabled
        self.denoise_enabled = denoise_enabled
        self.output_format = output_format.upper()

        # Model cache directory for downloaded models
        self.model_cache_dir = os.path.expanduser("~/.cache/audio-separator-models")

        logger.info(f"AudioSeparatorProcessor initialized with model: {model_name}")
        logger.info(f"GPU acceleration - CUDA: {use_cuda}, CoreML: {use_coreml}")

    def separate_audio(
        self,
        audio_path: str,
        stems: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Separate audio into stems using audio-separator.

        Args:
            audio_path: Path to input audio file
            stems: List of stems to extract (None = auto-detect from model)

        Returns:
            Dict containing separation results and file paths
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Starting audio separation with {self.model_name}")
        logger.info(f"Input file: {audio_path}")

        # Create temporary directory for output if not specified
        temp_output_dir = None
        if self.output_dir is None:
            temp_output_dir = tempfile.mkdtemp(prefix="audio_separator_")
            output_dir = temp_output_dir
        else:
            output_dir = self.output_dir
            os.makedirs(output_dir, exist_ok=True)

        try:
            # Initialize separator with audio file
            separator = Separator(
                audio_file_path=audio_path,
                model_name=self.model_name,
                output_dir=output_dir,
                use_cuda=self.use_cuda,
                use_coreml=self.use_coreml,
                log_level=self.log_level,
                normalization_enabled=self.normalization_enabled,
                denoise_enabled=self.denoise_enabled,
                output_format=self.output_format
            )

            logger.info(f"Model: {separator.model_name}")
            logger.info(f"Output directory: {separator.output_dir}")

            # Perform separation
            separator.separate()

            # Collect output files
            output_files = self._collect_output_files(output_dir)

            # Create result dictionary
            result = {
                'success': True,
                'model_used': self.model_name,
                'output_dir': output_dir,
                'input_file': audio_path,
                'output_files': output_files,
                'stems_found': list(output_files.keys()),
                'gpu_acceleration': {
                    'cuda': self.use_cuda,
                    'coreml': self.use_coreml
                }
            }

            logger.info(f"Separation completed successfully")
            logger.info(f"Generated {len(output_files)} stems: {list(output_files.keys())}")

            return result

        except Exception as e:
            logger.error(f"Audio separation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_used': self.model_name,
                'input_file': audio_path
            }

        finally:
            # Clean up temporary directory if created
            if temp_output_dir and os.path.exists(temp_output_dir):
                try:
                    shutil.rmtree(temp_output_dir)
                    logger.debug(f"Cleaned up temporary directory: {temp_output_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {e}")

    def _collect_output_files(self, output_dir: str) -> Dict[str, str]:
        """
        Collect and organize output files from separation.

        Args:
            output_dir: Directory containing separated files

        Returns:
            Dict mapping stem names to file paths
        """
        output_files = {}

        # Common stem patterns from UVR models
        stem_patterns = [
            ('vocals', ['*vocals*', '*voice*', '*vocal*']),
            ('instrumental', ['*instrumental*', '*music*', '*accompaniment*']),
            ('drums', ['*drums*', '*drum*', '*percussion*']),
            ('bass', ['*bass*', '*low*']),
            ('piano', ['*piano*', '*keys*', '*keyboard*']),
            ('guitar', ['*guitar*', '*gtr*']),
            ('other', ['*other*', '*rest*', '*remaining*']),
            ('noise', ['*noise*', '*denoise*']),
        ]

        # Find all audio files in output directory
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        for file_path in Path(output_dir).glob('*'):
            if file_path.suffix.lower() in audio_extensions:
                filename = file_path.name.lower()

                # Try to identify stem type
                stem_type = 'unknown'
                for stem_name, patterns in stem_patterns:
                    for pattern in patterns:
                        if pattern.replace('*', '') in filename:
                            stem_type = stem_name
                            break
                    if stem_type != 'unknown':
                        break

                output_files[stem_type] = str(file_path)
                logger.debug(f"Found stem: {stem_type} -> {file_path.name}")

        return output_files

    def get_available_models(self) -> List[str]:
        """
        Get list of available UVR models.

        Note: This is a static list based on known UVR models.
        The actual available models depend on what's been trained and released.

        Returns:
            List of available model names
        """
        # Common UVR models (subset of available models)
        models = [
            'UVR_MDXNET_KARA_2',          # Best for karaoke (vocals/instrumental)
            'UVR_MDXNET_21_OVERLAP_9',    # High quality multi-stem
            'UVR_MDXNET_21_OVERLAP_7',    # Good balance quality/speed
            'UVR_MDXNET_21_OVERLAP_5',    # Faster processing
            'UVR_MDXNET_MAIN_21',         # Main vocal model
            'UVR_MDXNET_BASS_21',         # Bass focused
            'UVR_MDXNET_DRUMS_21',        # Drums focused
            'UVR_MDXNET_GUITAR_21',       # Guitar focused
            'UVR_MDXNET_PIANO_21',        # Piano focused
            'UVR_MDXNET_STRINGS_21',      # Strings focused
            'UVR_MDXNET_WIND_21',         # Wind instruments focused
        ]

        return models

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dict with model information
        """
        model_info = {
            'name': model_name,
            'description': self._get_model_description(model_name),
            'recommended_use': self._get_model_use_case(model_name),
            'quality': self._get_model_quality(model_name),
            'speed': self._get_model_speed(model_name)
        }

        return model_info

    def _get_model_description(self, model_name: str) -> str:
        """Get human-readable description for model."""
        descriptions = {
            'UVR_MDXNET_KARA_2': 'Optimized for karaoke - separates vocals from instrumental',
            'UVR_MDXNET_21_OVERLAP_9': 'High-quality multi-stem separation (vocals, drums, bass, etc.)',
            'UVR_MDXNET_21_OVERLAP_7': 'Balanced quality and speed for general use',
            'UVR_MDXNET_21_OVERLAP_5': 'Fast processing with good quality',
            'UVR_MDXNET_MAIN_21': 'Focused on main vocals and lead instruments',
            'UVR_MDXNET_BASS_21': 'Optimized for bass guitar and low-frequency instruments',
            'UVR_MDXNET_DRUMS_21': 'Specialized for drum and percussion separation',
            'UVR_MDXNET_GUITAR_21': 'Focused on guitar and similar instruments',
            'UVR_MDXNET_PIANO_21': 'Optimized for piano and keyboard separation',
            'UVR_MDXNET_STRINGS_21': 'Focused on string instruments (violin, cello, etc.)',
            'UVR_MDXNET_WIND_21': 'Optimized for wind instruments (flute, saxophone, etc.)'
        }
        return descriptions.get(model_name, f'UVR model: {model_name}')

    def _get_model_use_case(self, model_name: str) -> str:
        """Get recommended use case for model."""
        use_cases = {
            'UVR_MDXNET_KARA_2': 'Karaoke creation, vocal removal',
            'UVR_MDXNET_21_OVERLAP_9': 'Music production, remixing, multi-track separation',
            'UVR_MDXNET_21_OVERLAP_7': 'General music analysis, balanced workflow',
            'UVR_MDXNET_21_OVERLAP_5': 'Quick processing, real-time applications',
            'UVR_MDXNET_MAIN_21': 'Lead vocal extraction, main melody isolation',
            'UVR_MDXNET_BASS_21': 'Bass guitar removal/addition, low-end analysis',
            'UVR_MDXNET_DRUMS_21': 'Drum isolation, rhythm analysis',
            'UVR_MDXNET_GUITAR_21': 'Guitar extraction, solo analysis',
            'UVR_MDXNET_PIANO_21': 'Piano isolation, keyboard analysis',
            'UVR_MDXNET_STRINGS_21': 'Orchestral analysis, string section extraction',
            'UVR_MDXNET_WIND_21': 'Wind instrument analysis, solo extraction'
        }
        return use_cases.get(model_name, 'General audio separation')

    def _get_model_quality(self, model_name: str) -> str:
        """Get quality rating for model."""
        if 'OVERLAP_9' in model_name:
            return 'Very High'
        elif 'OVERLAP_7' in model_name:
            return 'High'
        elif 'OVERLAP_5' in model_name:
            return 'Good'
        elif '21' in model_name:
            return 'High'
        else:
            return 'Standard'

    def _get_model_speed(self, model_name: str) -> str:
        """Get speed rating for model."""
        if 'OVERLAP_9' in model_name:
            return 'Slow'
        elif 'OVERLAP_7' in model_name:
            return 'Medium'
        elif 'OVERLAP_5' in model_name:
            return 'Fast'
        elif '21' in model_name:
            return 'Medium'
        else:
            return 'Medium'

    @staticmethod
    def is_available() -> bool:
        """Check if audio-separator library is available."""
        return AUDIO_SEPARATOR_AVAILABLE

    @staticmethod
    def get_gpu_status() -> Dict[str, bool]:
        """Get GPU acceleration availability status."""
        status = {
            'cuda_available': False,
            'coreml_available': False,
            'cpu_fallback': True
        }

        if AUDIO_SEPARATOR_AVAILABLE:
            try:
                # Test CUDA availability
                import torch
                status['cuda_available'] = torch.cuda.is_available()

                # Test CoreML availability (Apple Silicon)
                try:
                    import platform
                    status['coreml_available'] = platform.machine() == 'arm64' and 'macOS' in platform.platform()
                except:
                    pass

            except ImportError:
                pass

        return status
