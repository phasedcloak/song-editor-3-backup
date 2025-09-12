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
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

# Optional imports for different Whisper engines (import lazily to avoid bus errors)
OPENAI_WHISPER_AVAILABLE = False
FASTER_WHISPER_AVAILABLE = False

WHISPERX_AVAILABLE = False
# Try to import MLX-Whisper (should work on Apple Silicon)
try:
    import mlx_whisper
    MLX_WHISPER_AVAILABLE = True
except ImportError:
    MLX_WHISPER_AVAILABLE = False
    logging.warning("MLX Whisper not available")


def strip_punctuation(text: str) -> str:
    """
    Strip common punctuation from transcribed text while preserving word boundaries.
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned text with punctuation removed
    """
    if not text:
        return text
    
    # Common punctuation marks to remove
    punctuation_pattern = r'[.,!?;:"\'()\[\]{}<>/\\|`~@#$%^&*+=_\-]'
    
    # Remove punctuation but preserve spaces
    cleaned = re.sub(punctuation_pattern, '', text)
    
    # Clean up multiple spaces that might result from punctuation removal
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Strip leading/trailing whitespace
    return cleaned.strip()


class Transcriber:
    """Handles audio transcription using various Whisper models."""

    def __init__(
        self,
        model: str = "faster-whisper",  # Changed default to faster-whisper for GPU acceleration
        model_size: str = "large-v3-turbo",
        alternatives_count: int = 5,
        confidence_threshold: float = 0.5,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        content_type: str = "general"
    ):
        self.model = model
        self.model_size = self._normalize_model_size(model_size)
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
            # PRIORITY: Try MLX Whisper first (best for Apple Silicon)
            if MLX_WHISPER_AVAILABLE:
                try:
                    # Try requested size first; the worker will handle runtime fallbacks if needed
                    requested = str(self.model_size).lower()
                    self.whisper_model = f"mlx-community/whisper-{requested}"
                    logging.info(f"MLX Whisper ready with model: {self.whisper_model} (requested: {self.model_size})")
                    return
                except Exception as e:
                    logging.warning(f"MLX Whisper setup failed with requested size {self.model_size}: {e}")
                    try:
                        # Consistent fallback to large-v2
                        self.whisper_model = "mlx-community/whisper-large-v2"
                        logging.info("Falling back to MLX Whisper large-v2 model")
                        return
                    except Exception as e2:
                        logging.warning(f"MLX Whisper large-v2 fallback failed: {e2}")

            # Fallback: Try to import and use OpenAI Whisper
            try:
                import whisper
                # Try requested size (map large-v3-turbo to 'turbo' for OpenAI), then fall back
                if self.model_size in ("large-v3-turbo", "turbo"):
                    candidates = ["turbo", "large-v3", "large-v2"]
                else:
                    candidates = [self.model_size, "large-v3", "large-v2"]
                last_error = None
                for candidate in candidates:
                    try:
                        logging.info(f"Loading OpenAI Whisper model: {candidate} (requested: {self.model_size})")
                        # Prefer local bundled cache if available
                        model_root = None
                        try:
                            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                            local_dir = os.path.join(base_dir, 'models', 'whisper')
                            if os.path.isdir(local_dir):
                                model_root = local_dir
                        except Exception:
                            pass
                        if getattr(__import__('sys'), 'frozen', False):
                            try:
                                import sys as _sys
                                meipass_dir = getattr(_sys, '_MEIPASS', None)
                                if meipass_dir:
                                    local_dir = os.path.join(meipass_dir, 'models', 'whisper')
                                    if os.path.isdir(local_dir):
                                        model_root = local_dir
                            except Exception:
                                pass
                        if model_root:
                            self.whisper_model = whisper.load_model(candidate, download_root=model_root)
                        else:
                            self.whisper_model = whisper.load_model(candidate)
                        return
                    except Exception as candidate_error:
                        last_error = candidate_error
                        logging.warning(f"OpenAI Whisper failed to load {candidate}: {candidate_error}")
                if last_error is not None:
                    raise last_error
            except Exception as e:
                logging.warning(f"OpenAI Whisper failed to load after fallbacks: {e}")

            # Fallback: Try to import and use Faster Whisper
            try:
                from faster_whisper import WhisperModel
                # Try requested size; if it fails, try large-v3 and large-v2
                candidates = [self.model_size, "large-v3", "large-v2"]
                last_error = None
                for candidate in candidates:
                    try:
                        # Prefer local bundled models if available
                        model_arg = candidate
                        try:
                            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                            local_dir = os.path.join(base_dir, 'models', 'faster-whisper', candidate)
                            if os.path.isdir(local_dir):
                                model_arg = local_dir
                        except Exception:
                            pass
                        if getattr(__import__('sys'), 'frozen', False):
                            try:
                                import sys as _sys
                                meipass_dir = getattr(_sys, '_MEIPASS', None)
                                if meipass_dir:
                                    local_dir = os.path.join(meipass_dir, 'models', 'faster-whisper', candidate)
                                    if os.path.isdir(local_dir):
                                        model_arg = local_dir
                            except Exception:
                                pass

                        logging.info(f"Loading Faster Whisper model: {model_arg} (requested: {candidate})")
                        self.whisper_model = WhisperModel(model_arg, device='auto', compute_type='float32')
                        return
                    except Exception as candidate_error:
                        last_error = candidate_error
                        logging.warning(f"Faster Whisper failed to load {candidate}: {candidate_error}")
                if last_error is not None:
                    raise last_error
            except Exception as e:
                logging.warning(f"Faster Whisper failed to load after fallbacks: {e}")

            # Fallback: Try to use WhisperX
            try:
                import whisperx
                logging.info(f"Loading WhisperX model: {self.model_size}")
                # WhisperX models are loaded on-demand
                self.whisper_model = self.model_size
                return
            except Exception as e:
                logging.warning(f"WhisperX failed to load: {e}")

            # No models available
            raise ValueError("No Whisper models available. Please install at least one: mlx-whisper, openai-whisper, faster-whisper, or whisperx")

        except Exception as e:
            logging.error(f"Error initializing model {self.model}: {e}")
            raise

    def _normalize_model_size(self, size: Optional[str]) -> str:
        """Return canonical model size string (e.g., 'turbo' -> 'large-v3-turbo')."""
        try:
            value = (size or "").strip().lower()
            if value in ("turbo", "v3-turbo", "large-turbo"):
                return "large-v3-turbo"
            return value if value else "large-v2"
        except Exception:
            return "large-v2"

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

    def _initialize_openai_whisper(self) -> None:
        """Initialize OpenAI Whisper model."""
        try:
            import whisper
            global OPENAI_WHISPER_AVAILABLE
            OPENAI_WHISPER_AVAILABLE = True
            candidates = self._get_model_candidates()
            last_error = None
            for candidate in candidates:
                try:
                    logging.info(f"Loading OpenAI Whisper model: {candidate}")
                    self.whisper_model = whisper.load_model(candidate)
                    return
                except Exception as candidate_error:
                    last_error = candidate_error
                    logging.warning(f"OpenAI Whisper failed to load {candidate}: {candidate_error}")
            if last_error is not None:
                raise last_error
        except Exception as e:
            logging.warning(f"OpenAI Whisper failed to load after fallbacks: {e}")
            raise

    def _initialize_faster_whisper(self) -> None:
        """Initialize Faster Whisper model."""
        try:
            from faster_whisper import WhisperModel
            global FASTER_WHISPER_AVAILABLE
            FASTER_WHISPER_AVAILABLE = True
            candidates = self._get_model_candidates()
            last_error = None
            for candidate in candidates:
                try:
                    logging.info(f"Loading Faster Whisper model: {candidate}")
                    self.whisper_model = WhisperModel(candidate, device='auto', compute_type='float32')
                    return
                except Exception as candidate_error:
                    last_error = candidate_error
                    logging.warning(f"Faster Whisper failed to load {candidate}: {candidate_error}")
            if last_error is not None:
                raise last_error
        except Exception as e:
            logging.warning(f"Faster Whisper failed to load after fallbacks: {e}")
            raise

    def _initialize_whisperx(self) -> None:
        """Initialize WhisperX model."""
        try:
            import whisperx
            global WHISPERX_AVAILABLE
            WHISPERX_AVAILABLE = True
            logging.info(f"Loading WhisperX model: {self.model_size}")
            # WhisperX models are loaded on-demand, just set the size
            self.whisper_model = self.model_size
        except Exception as e:
            logging.warning(f"WhisperX failed to load: {e}")
            raise

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
                # Add threading and multiprocessing controls to avoid segmentation faults
                import os
                os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
                os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads

                result = self.whisper_model.transcribe(
                    temp_path,
                    language="en",  # Hardcode to English like working implementation
                    word_timestamps=True,
                    verbose=False,
                    beam_size=5,  # Use beam search like working implementation
                    temperature=0.0,  # Make deterministic like working implementation
                    initial_prompt=self.prompt if self.prompt else None,
                    fp16=False  # Force FP32 to avoid GPU issues
                )

                # Process results
                words = []
                for segment in result.get('segments', []):
                    for word_info in segment.get('words', []):
                        word = {
                            'text': strip_punctuation(word_info['word'].strip()),
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
                # Add threading controls to avoid segmentation faults
                import os
                os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
                os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads

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
                            'text': strip_punctuation(word_info.word.strip()),
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
                # Import and load WhisperX model lazily
                import whisperx
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
                            'text': strip_punctuation(word_info['word'].strip()),
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

    def _transcribe_with_forked_process(self, audio: np.ndarray, sample_rate: int, model_type: str) -> List[Dict[str, Any]]:
        """Transcribe using a forked process to avoid library conflicts."""
        start_time = datetime.now()
        try:
            # Save audio to temporary file
            temp_path = self._save_audio_temp(audio, sample_rate)

            try:
                # Create parameters file for the worker
                import subprocess
                import json
                import tempfile
                import sys

                # Prepare transcription parameters
                transcriber_params = {
                    'audio_path': temp_path,
                    'model_type': model_type,
                    'model_size': self.model_size,
                    'language': self.language,
                    'prompt': self.prompt,
                    'confidence_threshold': self.confidence_threshold
                }

                # Write parameters to temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as params_file:
                    json.dump(transcriber_params, params_file)
                    params_file_path = params_file.name

                try:
                    # Get path to transcription worker
                    # Resolve worker path in both source and PyInstaller frozen modes
                    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    worker_path = os.path.join(base_dir, 'transcription_worker.py')
                    if getattr(sys, 'frozen', False):
                        try:
                            worker_path = os.path.join(sys._MEIPASS, 'transcription_worker.py')  # type: ignore[attr-defined]
                        except Exception:
                            pass

                    # Run the transcription worker in a separate process
                    logging.debug(f"Starting forked transcription process for {model_type}")
                    logging.debug(f"Worker path: {worker_path}")
                    logging.debug(f"Python executable: {sys.executable}")

                    # Prefer invoking packaged entrypoint to avoid flags and path issues in frozen mode
                    try:
                        if getattr(sys, 'frozen', False):
                            cmd = [sys.executable, '--worker', 'transcription', '--worker-params', params_file_path]
                        else:
                            cmd = [sys.executable, worker_path, params_file_path]
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=600,
                            env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
                        )
                    except FileNotFoundError as e:
                        logging.error(f"Worker script not found: {worker_path}")
                        logging.error(f"Current directory: {os.getcwd()}")
                        logging.error(f"Python executable: {sys.executable}")
                        return []
                    except Exception as e:
                        logging.error(f"Subprocess execution failed: {e}")
                        return []

                    logging.debug(f"Subprocess completed with return code: {result.returncode}")
                    logging.debug(f"STDOUT length: {len(result.stdout)}")
                    logging.debug(f"STDERR length: {len(result.stderr)}")
                    logging.debug(f"STDERR content: {result.stderr[:200]}")

                    if result.returncode == 0:
                        try:
                            words = json.loads(result.stdout.strip())
                            # Calculate and log transcription timing
                            processing_time = (datetime.now() - start_time).total_seconds()
                            logging.info(f"Forked {model_type} transcription succeeded: {len(words)} words in {processing_time:.1f} seconds")
                            return words
                        except json.JSONDecodeError as e:
                            logging.error(f"Failed to parse transcription output: {e}")
                            logging.error(f"Raw output: {result.stdout[:500]}")
                            return []
                    else:
                        logging.error(f"Forked {model_type} transcription failed (code {result.returncode})")
                        logging.error(f"Stderr: {result.stderr}")
                        logging.error(f"Stdout: {result.stdout}")
                        if "TRANSCRIPTION_ERROR" in result.stderr:
                            logging.error("Worker script reported transcription error")
                        else:
                            logging.error("Worker script failed with unknown error")
                        return []

                finally:
                    # Clean up parameters file
                    try:
                        os.unlink(params_file_path)
                    except:
                        pass

            finally:
                # Clean up audio file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logging.error(f"Error in forked {model_type} transcription: {e}")
            logging.error(f"Exception type: {type(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return []

    def _transcribe_mlx_whisper(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Transcribe using MLX Whisper (legacy method, now uses forked process)."""
        return self._transcribe_with_forked_process(audio, sample_rate, "mlx-whisper")

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
            if self.model == "mlx-whisper":
                models_to_try = ["mlx-whisper", "openai-whisper", "faster-whisper"]
            elif self.model == "faster-whisper":
                models_to_try = ["faster-whisper", "mlx-whisper", "openai-whisper"]
            elif self.model == "openai-whisper":
                models_to_try = ["openai-whisper", "mlx-whisper", "faster-whisper"]
            else:
                models_to_try = [self.model]

            # PRIORITY: Try MLX-Whisper first with forked process isolation
            if self.model == "mlx-whisper" or MLX_WHISPER_AVAILABLE:
                try:
                    logging.info("Trying MLX-Whisper with forked process isolation...")
                    words = self._transcribe_with_forked_process(audio, sample_rate, "mlx-whisper")
                    if words is not None and len(words) > 0:
                        logging.info(f"Successfully transcribed with MLX-Whisper: {len(words)} words")
                        return words
                    else:
                        logging.warning("MLX-Whisper returned empty results")
                except Exception as e:
                    logging.warning(f"MLX-Whisper failed: {e}")

            # Try faster-whisper with forked process isolation
            if self.model == "faster-whisper" or FASTER_WHISPER_AVAILABLE:
                try:
                    logging.info("Trying faster-whisper with forked process isolation...")
                    words = self._transcribe_with_forked_process(audio, sample_rate, "faster-whisper")
                    if words is not None and len(words) > 0:
                        logging.info(f"Successfully transcribed with faster-whisper: {len(words)} words")
                        return words
                    else:
                        logging.warning("faster-whisper returned empty results")
                except Exception as e:
                    logging.warning(f"faster-whisper failed: {e}")

            # Fallback: Try the requested model
            for model_name in models_to_try:
                if model_name == "mlx-whisper" and words is not None:
                    continue  # Already tried MLX-Whisper

                try:
                    logging.info(f"Trying {model_name}...")

                    if model_name == "openai-whisper":
                        # Re-initialize model if needed
                        if not isinstance(self.whisper_model, str) or not hasattr(self.whisper_model, 'transcribe'):
                            self._initialize_openai_whisper()
                        words = self._transcribe_openai_whisper(audio, sample_rate)
                    elif model_name == "faster-whisper":
                        # Re-initialize model if needed
                        if not hasattr(self.whisper_model, 'transcribe'):
                            self._initialize_faster_whisper()
                        words = self._transcribe_faster_whisper(audio, sample_rate)
                    elif model_name == "whisperx":
                        # Re-initialize model if needed
                        if not hasattr(self.whisper_model, 'transcribe'):
                            self._initialize_whisperx()
                        words = self._transcribe_whisperx(audio, sample_rate)
                    elif model_name == "mlx-whisper":
                        # MLX-Whisper uses forked process isolation, skip direct call
                        continue

                    if words is not None and len(words) > 0:
                        logging.info(f"Successfully transcribed with {model_name}: {len(words)} words")
                        break
                    else:
                        logging.warning(f"{model_name} returned empty results")

                except Exception as e:
                    logging.warning(f"{model_name} failed: {e}")
                    if model_name == models_to_try[-1]:  # Last model failed
                        logging.error(f"All transcription models failed")
                        words = []
                    continue

            # If no model worked and we haven't tried MLX-Whisper yet, try it as last resort
            if (words is None or len(words) == 0) and not self.model == "mlx-whisper" and MLX_WHISPER_AVAILABLE:
                try:
                    logging.info("Trying MLX-Whisper as last resort...")
                    words = self._transcribe_mlx_whisper(audio, sample_rate)
                    if words is not None and len(words) > 0:
                        logging.info(f"Successfully transcribed with MLX-Whisper: {len(words)} words")
                except Exception as e:
                    logging.error(f"MLX-Whisper also failed: {e}")
                    words = []

            if words is None:
                raise ValueError("No transcription model succeeded")

            # Generate alternatives if requested
            if self.alternatives_count > 0:
                for word in words:
                    word['alternatives'] = self._generate_alternatives(
                        word['text'],
                        word['confidence']
                    )[:self.alternatives_count]

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

    def get_transcription_info(self) -> Dict[str, Any]:
        """Get information about the transcriber configuration."""
        return {
            'model': self.model,
            'model_size': self.model_size,
            'language': self.language,
            'confidence_threshold': self.confidence_threshold,
            'prompt': self.prompt,
            'openai_whisper_available': OPENAI_WHISPER_AVAILABLE,
            'faster_whisper_available': FASTER_WHISPER_AVAILABLE,
            'whisperx_available': WHISPERX_AVAILABLE,
            'mlx_whisper_available': MLX_WHISPER_AVAILABLE
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
