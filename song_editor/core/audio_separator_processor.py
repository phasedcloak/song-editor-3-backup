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
        self.model_info_file = os.path.join(self.model_cache_dir, "model_info.json")

        # Initialize cache system
        self._initialize_model_cache()

        logger.info(f"AudioSeparatorProcessor initialized with model: {model_name}")
        logger.info(f"GPU acceleration - CUDA: {use_cuda}, CoreML: {use_coreml}")
        logger.info(f"Model cache directory: {self.model_cache_dir}")

    def _initialize_model_cache(self):
        """Initialize the model cache system."""
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(self.model_cache_dir, exist_ok=True)

            # Initialize model info tracking
            self.model_info = {}
            if os.path.exists(self.model_info_file):
                try:
                    import json
                    with open(self.model_info_file, 'r') as f:
                        self.model_info = json.load(f)
                    logger.info(f"Loaded model cache info for {len(self.model_info)} models")
                except Exception as e:
                    logger.warning(f"Failed to load model cache info: {e}")
                    self.model_info = {}

            # Clean up old cache entries
            self._cleanup_cache()

        except Exception as e:
            logger.error(f"Failed to initialize model cache: {e}")
            self.model_info = {}

    def _cleanup_cache(self, max_age_days: int = 30):
        """Clean up old cached models and invalid entries."""
        try:
            import json
            import time
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60

            # Remove entries for non-existent models
            models_to_remove = []
            for model_name, info in self.model_info.items():
                model_path = info.get('path', '')
                if not os.path.exists(model_path):
                    models_to_remove.append(model_name)
                    logger.info(f"Removing cache entry for missing model: {model_name}")

            for model_name in models_to_remove:
                del self.model_info[model_name]

            # Remove old entries (older than max_age_days)
            old_entries = []
            for model_name, info in self.model_info.items():
                last_used = info.get('last_used', 0)
                if current_time - last_used > max_age_seconds:
                    old_entries.append(model_name)
                    logger.info(f"Removing old cache entry: {model_name}")

            for model_name in old_entries:
                del self.model_info[model_name]

            # Save updated cache info
            if self.model_info:
                with open(self.model_info_file, 'w') as f:
                    json.dump(self.model_info, f, indent=2)

        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")

    def _update_model_cache_info(self, model_name: str, model_path: str, size_mb: float = 0):
        """Update cache information for a model."""
        try:
            import json
            import time

            self.model_info[model_name] = {
                'path': model_path,
                'size_mb': size_mb,
                'last_used': time.time(),
                'download_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            # Save to file
            with open(self.model_info_file, 'w') as f:
                json.dump(self.model_info, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to update model cache info: {e}")

    def _get_model_size(self, model_path: str) -> float:
        """Get model file size in MB."""
        try:
            if os.path.exists(model_path):
                size_bytes = os.path.getsize(model_path)
                return size_bytes / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.warning(f"Failed to get model size: {e}")
        return 0.0

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models."""
        try:
            total_size = 0
            model_count = len(self.model_info)

            for info in self.model_info.values():
                total_size += info.get('size_mb', 0)

            return {
                'cache_directory': self.model_cache_dir,
                'total_models': model_count,
                'total_size_mb': round(total_size, 2),
                'models': self.model_info
            }
        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return {'error': str(e)}

    def clear_cache(self, model_name: Optional[str] = None):
        """Clear cache for specific model or all models."""
        try:
            import json

            if model_name:
                # Clear specific model
                if model_name in self.model_info:
                    model_path = self.model_info[model_name].get('path', '')
                    if os.path.exists(model_path):
                        try:
                            os.remove(model_path)
                            logger.info(f"Removed cached model file: {model_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove model file: {e}")

                    del self.model_info[model_name]
                    logger.info(f"Cleared cache for model: {model_name}")
            else:
                # Clear all models
                for info in self.model_info.values():
                    model_path = info.get('path', '')
                    if os.path.exists(model_path):
                        try:
                            os.remove(model_path)
                            logger.info(f"Removed cached model file: {model_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove model file: {e}")

                self.model_info.clear()
                logger.info("Cleared all model cache")

            # Update cache info file
            with open(self.model_info_file, 'w') as f:
                json.dump(self.model_info, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

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

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available UVR models with detailed information.

        Returns:
            List of model dictionaries with name, description, size, and quality info
        """
        # Enhanced UVR models with detailed information
        models = [
            {
                'name': 'UVR_MDXNET_KARA_2',
                'description': 'Best for karaoke - separates vocals from instrumental',
                'estimated_size_mb': 85,
                'quality': 'High',
                'speed': 'Medium',
                'recommended_use': 'Karaoke creation, vocal removal',
                'category': 'vocals'
            },
            {
                'name': 'UVR_MDXNET_21_OVERLAP_9',
                'description': 'High-quality multi-stem separation (vocals, drums, bass, etc.)',
                'estimated_size_mb': 120,
                'quality': 'Very High',
                'speed': 'Slow',
                'recommended_use': 'Music production, remixing, multi-track separation',
                'category': 'multi_stem'
            },
            {
                'name': 'UVR_MDXNET_21_OVERLAP_7',
                'description': 'Balanced quality and speed for general use',
                'estimated_size_mb': 95,
                'quality': 'High',
                'speed': 'Medium',
                'recommended_use': 'General music analysis, balanced workflow',
                'category': 'multi_stem'
            },
            {
                'name': 'UVR_MDXNET_21_OVERLAP_5',
                'description': 'Fast processing with good quality',
                'estimated_size_mb': 75,
                'quality': 'Good',
                'speed': 'Fast',
                'recommended_use': 'Quick processing, real-time applications',
                'category': 'multi_stem'
            },
            {
                'name': 'UVR_MDXNET_MAIN_21',
                'description': 'Focused on main vocals and lead instruments',
                'estimated_size_mb': 80,
                'quality': 'High',
                'speed': 'Medium',
                'recommended_use': 'Lead vocal extraction, main melody isolation',
                'category': 'vocals'
            },
            {
                'name': 'UVR_MDXNET_BASS_21',
                'description': 'Optimized for bass guitar and low-frequency instruments',
                'estimated_size_mb': 70,
                'quality': 'High',
                'speed': 'Medium',
                'recommended_use': 'Bass guitar removal/addition, low-end analysis',
                'category': 'bass'
            },
            {
                'name': 'UVR_MDXNET_DRUMS_21',
                'description': 'Specialized for drum and percussion separation',
                'estimated_size_mb': 75,
                'quality': 'High',
                'speed': 'Medium',
                'recommended_use': 'Drum isolation, rhythm analysis',
                'category': 'drums'
            },
            {
                'name': 'UVR_MDXNET_GUITAR_21',
                'description': 'Focused on guitar and similar instruments',
                'estimated_size_mb': 80,
                'quality': 'High',
                'speed': 'Medium',
                'recommended_use': 'Guitar extraction, solo analysis',
                'category': 'guitar'
            },
            {
                'name': 'UVR_MDXNET_PIANO_21',
                'description': 'Optimized for piano and keyboard separation',
                'estimated_size_mb': 85,
                'quality': 'High',
                'speed': 'Medium',
                'recommended_use': 'Piano isolation, keyboard analysis',
                'category': 'piano'
            },
            {
                'name': 'UVR_MDXNET_STRINGS_21',
                'description': 'Focused on string instruments (violin, cello, etc.)',
                'estimated_size_mb': 75,
                'quality': 'High',
                'speed': 'Medium',
                'recommended_use': 'Orchestral analysis, string section extraction',
                'category': 'strings'
            },
            {
                'name': 'UVR_MDXNET_WIND_21',
                'description': 'Optimized for wind instruments (flute, saxophone, etc.)',
                'estimated_size_mb': 70,
                'quality': 'High',
                'speed': 'Medium',
                'recommended_use': 'Wind instrument analysis, solo extraction',
                'category': 'wind'
            }
        ]

        return models

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dict with model information including cache status
        """
        # Get base model info from enhanced list
        available_models = self.get_available_models()
        model_data = None

        for model in available_models:
            if model['name'] == model_name:
                model_data = model
                break

        if model_data:
            # Add cache information if available
            cache_info = self.model_info.get(model_name, {})
            model_data.update({
                'cached': bool(cache_info),
                'cache_path': cache_info.get('path', ''),
                'cache_size_mb': cache_info.get('size_mb', 0),
                'last_used': cache_info.get('last_used', 0),
                'download_date': cache_info.get('download_date', '')
            })
            return model_data
        else:
            # Fallback for unknown models
            return {
                'name': model_name,
                'description': f'UVR model: {model_name}',
                'estimated_size_mb': 0,
                'quality': 'Unknown',
                'speed': 'Unknown',
                'recommended_use': 'General audio separation',
                'category': 'unknown',
                'cached': False
            }

    def get_models_for_ui(self) -> List[str]:
        """
        Get model names formatted for UI dropdown display.

        Returns:
            List of formatted model names for UI display
        """
        models = self.get_available_models()
        formatted_names = []

        for model in models:
            # Format: "Model Name (Size MB) - Description"
            display_name = f"{model['name']} ({model['estimated_size_mb']}MB) - {model['description']}"
            formatted_names.append(display_name)

        return formatted_names

    def get_model_name_from_display(self, display_name: str) -> str:
        """
        Extract model name from UI display format.

        Args:
            display_name: Formatted display name from UI

        Returns:
            Clean model name
        """
        # Extract model name from format: "Model_Name (Size) - Description"
        if ' (' in display_name and ') - ' in display_name:
            return display_name.split(' (')[0]
        return display_name

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
