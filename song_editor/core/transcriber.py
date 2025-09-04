#!/usr/bin/env python3
"""
Transcriber Module

Handles audio transcription using various Whisper models including
OpenAI Whisper, WhisperX, and MLX Whisper for Song Editor 3.
"""

import os
import logging
import tempfile
import numpy as np
import soundfile as sf
from typing import Dict, Any, List, Optional
from datetime import datetime

# Optional imports for different Whisper engines
try:
    import whisper
    OPENAI_WHISPER_AVAILABLE = True
except ImportError:
    OPENAI_WHISPER_AVAILABLE = False
    logging.warning("OpenAI Whisper not available")

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logging.warning("Faster Whisper not available")

try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    logging.warning("WhisperX not available")

try:
    import mlx_whisper
    MLX_WHISPER_AVAILABLE = True
except ImportError:
    MLX_WHISPER_AVAILABLE = False
    logging.warning("MLX Whisper not available")


class Transcriber:
    """Handles audio transcription using various Whisper models."""

    def __init__(
        self,
        model: str = "faster-whisper",  # Changed default to faster-whisper for GPU acceleration
        model_size: str = "base",
        alternatives_count: int = 5,
        confidence_threshold: float = 0.5,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        content_type: str = "general"
    ):
        self.model = model
        self.model_size = model_size
        self.alternatives_count = alternatives_count
        self.confidence_threshold = confidence_threshold
        self.language = language
        self.prompt = prompt
        self.content_type = content_type

        # Set default prompts based on content type
        if self.prompt is None:
            self.prompt = self._get_default_prompt()

        # Initialize model based on type
        self.whisper_model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the appropriate Whisper model."""
        try:
            if self.model == "openai-whisper" and OPENAI_WHISPER_AVAILABLE:
                # Use large-v2 for best accuracy like working wav_to_karoke implementation
                model_size = "large-v2" if self.model_size in ["tiny", "base", "small"] else self.model_size
                logging.info(f"Loading OpenAI Whisper model: {model_size} (requested: {self.model_size})")
                self.whisper_model = whisper.load_model(model_size)

            elif self.model == "faster-whisper" and FASTER_WHISPER_AVAILABLE:
                logging.info(f"Loading Faster Whisper model: {self.model_size} with GPU acceleration")
                # Use GPU acceleration with auto device selection and float32 compute type
                self.whisper_model = WhisperModel(self.model_size, device='auto', compute_type='float32')

            elif self.model == "whisperx" and WHISPERX_AVAILABLE:
                logging.info(f"Loading WhisperX model: {self.model_size}")
                # WhisperX models are loaded on-demand
                self.whisper_model = self.model_size

            elif self.model == "mlx-whisper" and MLX_WHISPER_AVAILABLE:
                logging.info(f"Loading MLX Whisper model: {self.model_size}")
                self.whisper_model = mlx_whisper.load_model(self.model_size)

            else:
                # Fallback to faster-whisper if available
                if FASTER_WHISPER_AVAILABLE:
                    logging.warning(
                        f"Model {self.model} not available, falling back to faster-whisper with GPU acceleration"
                    )
                    self.model = "faster-whisper"
                    self.whisper_model = WhisperModel(self.model_size, device='auto', compute_type='float32')
                else:
                    raise ValueError(f"Model {self.model} not available and no fallback found")

        except Exception as e:
            logging.error(f"Error initializing model {self.model}: {e}")
            raise

    def _get_default_prompt(self) -> str:
        """Get default prompt based on content type."""
        prompts = {
            "general": "",
            "christian": "This is a Christian worship song with clean, family-friendly lyrics. "
                         "No profanity or inappropriate language.",
            "gospel": "This is a gospel song with spiritual and uplifting lyrics. "
                         "No profanity or inappropriate language.",
            "worship": "This is a worship song with reverent and spiritual lyrics. "
                        "No profanity or inappropriate language.",
            "hymn": "This is a traditional hymn with sacred and reverent lyrics. "
                        "No profanity or inappropriate language.",
            "clean": "This is a family-friendly song with clean lyrics. "
                        "No profanity or inappropriate language."
        }
        return prompts.get(self.content_type, "")

    def _save_audio_temp(self, audio: np.ndarray, sample_rate: int) -> str:
        """Save audio to temporary file for Whisper processing."""
        try:
            # Debug: Log audio properties
            logging.debug(f"Saving audio: shape={audio.shape}, dtype={audio.dtype}, sr={sample_rate}")
            logging.debug(f"Audio stats: min={np.min(audio):.6f}, max={np.max(audio):.6f}, mean={np.mean(audio):.6f}")

            # Create temporary file with explicit cleanup
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)  # Close the file descriptor

            # Ensure audio is in correct format
            if len(audio.shape) > 1:
                if audio.shape[0] == 2:  # Stereo audio
                    audio = np.mean(audio, axis=0)  # Convert to mono by averaging channels
                else:
                    audio = audio.squeeze()  # Remove singleton dimensions

            # Ensure audio is in valid range [-1, 1]
            audio = np.clip(audio, -1.0, 1.0)

            # Save using soundfile with explicit parameters
            sf.write(temp_path, audio.astype(np.float32), sample_rate, subtype='PCM_16')

            # Verify the file was created and is readable
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise ValueError("Temporary audio file was not created properly")

            logging.debug(f"Successfully saved audio to: {temp_path}")
            return temp_path

        except Exception as e:
            logging.error(f"Error saving temporary audio file: {e}")
            logging.error(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")
            raise

    def _transcribe_openai_whisper(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Transcribe using OpenAI Whisper."""
        try:
            # Save audio to temporary file
            temp_path = self._save_audio_temp(audio, sample_rate)

            try:
                # Transcribe with OpenAI Whisper - use exact working parameters from wav_to_karoke
                result = self.whisper_model.transcribe(
                    temp_path,
                    language="en",  # Hardcode to English like working implementation
                    word_timestamps=True,
                    verbose=False,
                    beam_size=5,  # Use beam search like working implementation
                    temperature=0.0,  # Make deterministic like working implementation
                    initial_prompt=self.prompt if self.prompt else None
                )

                # Process results
                words = []
                for segment in result.get('segments', []):
                    for word_info in segment.get('words', []):
                        word = {
                            'text': word_info['word'].strip(),
                            'start': word_info['start'],
                            'end': word_info['end'],
                            'confidence': word_info.get('confidence', 0.5),
                            'alternatives': []
                        }

                        # Filter by confidence threshold
                        if word['confidence'] >= self.confidence_threshold:
                            words.append(word)

                return words

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logging.error(f"Error in OpenAI Whisper transcription: {e}")
            raise

    def _transcribe_faster_whisper(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Transcribe using Faster Whisper."""
        try:
            # Save audio to temporary file
            temp_path = self._save_audio_temp(audio, sample_rate)

            try:
                # Transcribe with Faster Whisper - use working parameters from wav_to_karoke
                segments, info = self.whisper_model.transcribe(
                    temp_path,
                    language=self.language if self.language else None,
                    word_timestamps=True,  # Re-enable word timestamps (they work fine)
                    beam_size=1,  # Keep beam size at 1
                    initial_prompt=self.prompt if self.prompt else None
                )

                # Process results
                words = []
                for segment in segments:
                    for word_info in segment.words:
                        word = {
                            'text': word_info.word.strip(),
                            'start': word_info.start,
                            'end': word_info.end,
                            'confidence': word_info.probability,
                            'alternatives': []
                        }

                        # Filter by confidence threshold
                        if word['confidence'] >= self.confidence_threshold:
                            words.append(word)

                return words

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logging.error(f"Error in Faster Whisper transcription: {e}")
            raise

    def _transcribe_whisperx(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Transcribe using WhisperX."""
        try:
            # Save audio to temporary file
            temp_path = self._save_audio_temp(audio, sample_rate)

            try:
                # Load WhisperX model
                model = whisperx.load_model(self.whisper_model)

                # Transcribe with WhisperX
                result = model.transcribe(
                    temp_path,
                    language=self.language if self.language else None,
                    initial_prompt=self.prompt if self.prompt else None
                )

                # Align timestamps
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"],
                    device="cpu"
                )
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    temp_path,
                    "cpu"
                )

                # Process results
                words = []
                for segment in result.get('segments', []):
                    for word_info in segment.get('words', []):
                        word = {
                            'text': word_info['word'].strip(),
                            'start': word_info['start'],
                            'end': word_info['end'],
                            'confidence': word_info.get('confidence', 0.5),
                            'alternatives': []
                        }

                        # Filter by confidence threshold
                        if word['confidence'] >= self.confidence_threshold:
                            words.append(word)

                return words

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logging.error(f"Error in WhisperX transcription: {e}")
            raise

    def _transcribe_mlx_whisper(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Transcribe using MLX Whisper."""
        try:
            # Save audio to temporary file
            temp_path = self._save_audio_temp(audio, sample_rate)

            try:
                # Transcribe with MLX Whisper
                result = mlx_whisper.transcribe(
                    self.whisper_model,
                    temp_path,
                    language=self.language if self.language else None,
                    word_timestamps=True,
                    initial_prompt=self.prompt if self.prompt else None
                )

                # Process results
                words = []
                for segment in result.get('segments', []):
                    for word_info in segment.get('words', []):
                        word = {
                            'text': word_info['word'].strip(),
                            'start': word_info['start'],
                            'end': word_info['end'],
                            'confidence': word_info.get('confidence', 0.5),
                            'alternatives': []
                        }

                        # Filter by confidence threshold
                        if word['confidence'] >= self.confidence_threshold:
                            words.append(word)

                return words

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logging.error(f"Error in MLX Whisper transcription: {e}")
            raise

    def _generate_alternatives(self, word: str, confidence: float) -> List[Dict[str, Any]]:
        """Generate alternative transcriptions for a word."""
        alternatives = []

        # Simple alternative generation based on common misheard words
        # In a real implementation, this could use a language model or API
        common_alternatives = {
            'the': ['thee', 'thuh'],
            'a': ['ay', 'uh'],
            'and': ['an', 'end'],
            'to': ['too', 'two'],
            'for': ['four', 'fore'],
            'you': ['u', 'yew'],
            'are': ['r', 'our'],
            'your': ['you\'re', 'yore'],
            'there': ['their', 'they\'re'],
            'here': ['hear', 'heer']
        }

        word_lower = word.lower().strip()
        if word_lower in common_alternatives:
            for alt in common_alternatives[word_lower]:
                alternatives.append({
                    'text': alt,
                    'confidence': confidence * 0.8  # Slightly lower confidence for alternatives
                })

        return alternatives

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Transcribe audio using the selected Whisper model."""
        start_time = datetime.now()

        try:
            logging.info(f"Starting transcription with {self.model}...")

            # Select transcription method with fallback
            words = None
            models_to_try = []

            # Build list of models to try in order
            if self.model == "faster-whisper":
                models_to_try = ["faster-whisper", "openai-whisper"]
            elif self.model == "openai-whisper":
                models_to_try = ["openai-whisper", "faster-whisper"]
            else:
                models_to_try = [self.model]

            # Try each model until one works
            for model_name in models_to_try:
                try:
                    logging.info(f"Trying transcription with {model_name}...")

                    if model_name == "openai-whisper":
                        words = self._transcribe_openai_whisper(audio, sample_rate)
                    elif model_name == "faster-whisper":
                        words = self._transcribe_faster_whisper(audio, sample_rate)
                    elif model_name == "whisperx":
                        words = self._transcribe_whisperx(audio, sample_rate)
                    elif model_name == "mlx-whisper":
                        words = self._transcribe_mlx_whisper(audio, sample_rate)

                    if words is not None:
                        logging.info(f"Successfully transcribed with {model_name}")
                        break

                except Exception as e:
                    logging.warning(f"Transcription failed with {model_name}: {e}")
                    if model_name == models_to_try[-1]:  # Last model failed
                        raise ValueError(f"All transcription models failed: {e}")
                    continue

            if words is None:
                raise ValueError("No transcription model succeeded")

            # Generate alternatives if requested
            if self.alternatives_count > 0:
                for word in words:
                    word['alternatives'] = self._generate_alternatives(
                        word['text'],
                        word['confidence']
                    )[:self.alternatives_count]

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            logging.info(f"Transcription completed: {len(words)} words in {processing_time:.1f} seconds")

            return words

        except Exception as e:
            logging.error(f"Error in transcription: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model': self.model,
            'model_size': self.model_size,
            'alternatives_count': self.alternatives_count,
            'confidence_threshold': self.confidence_threshold,
            'language': self.language,
            'content_type': self.content_type,
            'prompt': self.prompt,
            'available_models': {
                'openai-whisper': OPENAI_WHISPER_AVAILABLE,
                'faster-whisper': FASTER_WHISPER_AVAILABLE,
                'whisperx': WHISPERX_AVAILABLE,
                'mlx-whisper': MLX_WHISPER_AVAILABLE
            }
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clear model from memory
            if self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None

            logging.debug("Transcriber cleanup completed")

        except Exception as e:
            logging.warning(f"Error during transcriber cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
