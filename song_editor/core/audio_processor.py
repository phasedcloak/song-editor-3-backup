#!/usr/bin/env python3
"""
Audio Processor Module

Handles audio loading, denoising, normalization, and source separation.
"""

import os
import logging
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import psutil
import time

DEMUCS_AVAILABLE = False
NOISEREDUCE_AVAILABLE = False

PYLN_AVAILABLE = False


class AudioProcessor:
    """Handles audio processing pipeline."""

    def __init__(
        self,
        use_demucs: bool = True,
        save_intermediate: bool = True,
        target_sr: int = 44100,
        denoise_strength: float = 0.5,
        normalize_lufs: float = -23.0,
        demucs_model: str = 'htdemucs'
    ):
        self.use_demucs = use_demucs
        self.save_intermediate = save_intermediate
        self.target_sr = target_sr
        self.denoise_strength = denoise_strength
        self.normalize_lufs = normalize_lufs
        self.demucs_model = demucs_model

        self.separator = None
        self.audio_data = None
        self.processing_info = {}

        if self.use_demucs:
            self._initialize_demucs()

    def _initialize_demucs(self):
        """Initialize Demucs separator."""
        try:
            # Import demucs lazily
            from demucs.separate import HTDemucs, apply_model
            from demucs.pretrained import get_model
            # Load chosen pretrained model
            model_name = (self.demucs_model or 'htdemucs').strip()
            self.separator = get_model(model_name)
            logging.info(f"Demucs initialized successfully (model: {model_name})")
        except ImportError:
            logging.warning("Demucs not available, using fallback methods")
        except Exception as e:
            logging.error(f"Failed to initialize Demucs: {e}")
            self.use_demucs = False

    def _log_memory_usage(self, stage: str):
        """Log memory usage for a processing stage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024

        self.processing_info[f"{stage}_memory_mb"] = memory_mb
        logging.debug(f"{stage} memory usage: {memory_mb:.1f} MB")

    def _calculate_audio_levels(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Calculate audio levels and statistics."""
        levels = {}

        # RMS levels
        levels['rms'] = float(np.sqrt(np.mean(audio**2)))
        levels['rms_db'] = float(20 * np.log10(levels['rms'] + 1e-10))

        # Peak levels
        levels['peak'] = float(np.max(np.abs(audio)))
        levels['peak_db'] = float(20 * np.log10(levels['peak'] + 1e-10))

        # Dynamic range
        levels['dynamic_range_db'] = levels['peak_db'] - levels['rms_db']

        # Crest factor
        levels['crest_factor'] = float(levels['peak'] / (levels['rms'] + 1e-10))

        return levels

    def _detect_tempo(self, audio: np.ndarray, sr: int) -> Optional[float]:
        """Detect tempo using librosa."""
        try:
            start = time.time()
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            self.processing_info['tempo_time'] = time.time() - start
            return float(tempo)
        except Exception as e:
            logging.warning(f"Tempo detection failed: {e}")
            return None

    def _detect_key(self, audio: np.ndarray, sr: int) -> Optional[Dict[str, Any]]:
        """Detect musical key using librosa."""
        try:
            start = time.time()
            # Extract chromagram
            chromagram = librosa.feature.chroma_cqt(y=audio, sr=sr)

            # Get key profile
            key_profile = np.mean(chromagram, axis=1)

            # Find dominant key
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            dominant_key_idx = np.argmax(key_profile)
            dominant_key = key_names[dominant_key_idx]

            # Determine major/minor
            # This is a simplified approach - in practice you'd use more sophisticated methods
            major_scale = [0, 2, 4, 5, 7, 9, 11]
            minor_scale = [0, 2, 3, 5, 7, 8, 10]

            major_score = sum(key_profile[(dominant_key_idx + note) % 12] for note in major_scale)
            minor_score = sum(key_profile[(dominant_key_idx + note) % 12] for note in minor_scale)

            mode = "major" if major_score > minor_score else "minor"

            return {
                'key': dominant_key,
                'mode': mode,
                'confidence': float(max(major_score, minor_score) / sum(key_profile))
            }
            # not reached

        except Exception as e:
            logging.warning(f"Key detection failed: {e}")
            return None
        finally:
            try:
                self.processing_info['key_time'] = time.time() - start
            except Exception:
                pass

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        try:
            logging.info(f"Loading audio: {file_path}")
            start_time = time.time()

            # Load audio
            audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)

            # Store original info
            self.audio_data = {
                'original_path': file_path,
                'original_sr': sr,
                'duration': len(audio) / sr,
                'channels': 1
            }

            load_time = time.time() - start_time
            self.processing_info['load_time'] = load_time
            self._log_memory_usage('load')

            logging.info(f"Audio loaded: {len(audio)} samples, {sr} Hz, {self.audio_data['duration']:.2f}s")
            return audio, sr

        except Exception as e:
            logging.error(f"Failed to load audio: {e}")
            raise

    def denoise_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Denoise audio using noisereduce."""
        try:
            # Import noisereduce lazily
            import noisereduce as nr

            logging.info("Denoising audio...")
            start_time = time.time()

            # Apply noise reduction
            denoised = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=True,
                prop_decrease=self.denoise_strength
            )

            denoise_time = time.time() - start_time
            self.processing_info['denoise_time'] = denoise_time
            self._log_memory_usage('denoise')

            logging.info(f"Denoising completed in {denoise_time:.2f}s")
            return denoised

        except ImportError:
            logging.warning("Noisereduce not available, skipping denoising")
            return audio
        except Exception as e:
            logging.error(f"Denoising failed: {e}")
            return audio

    def normalize_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Normalize audio to target LUFS."""
        try:
            # Import pyloudnorm lazily
            import pyloudnorm as pyln

            logging.info("Normalizing audio...")
            start_time = time.time()

            # Create loudness meter
            meter = pyln.Meter(sr)

            # Measure current loudness
            current_lufs = meter.integrated_loudness(audio)

            # Calculate gain needed
            gain_db = self.normalize_lufs - current_lufs
            gain_linear = 10**(gain_db / 20)

            # Apply normalization
            normalized = audio * gain_linear

            # Ensure no clipping
            max_val = np.max(np.abs(normalized))
            if max_val > 0.95:
                normalized = normalized * (0.95 / max_val)

            normalize_time = time.time() - start_time
            self.processing_info['normalize_time'] = normalize_time
            self._log_memory_usage('normalize')

            logging.info(f"Normalization completed: {current_lufs:.1f} LUFS -> {self.normalize_lufs:.1f} LUFS")
            return normalized

        except ImportError:
            logging.warning("Pyloudnorm not available, using RMS normalization")
            # Fallback to RMS normalization
            rms = np.sqrt(np.mean(audio**2))
            target_rms = 0.1
            normalized = audio * (target_rms / (rms + 1e-10))
            return normalized
        except Exception as e:
            logging.error(f"Normalization failed: {e}")
            return audio

    def separate_sources(self, audio_path: str, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Separate audio sources using Demucs or fallback."""
        if self.use_demucs and self.separator:
            try:
                logging.info("Separating audio sources with Demucs...")
                start_time = time.time()

                # Import required modules
                import torch
                from demucs.separate import apply_model
                # Ensure stereo by duplicating mono channel: [batch=1, channels=2, time]
                if len(audio.shape) == 1:
                    # Duplicate mono channel to create stereo
                    audio_stereo = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).repeat(1, 2, 1)
                else:
                    audio_stereo = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

                # Pick best available device (Metal/MPS on Apple Silicon, otherwise CPU)
                device = 'cpu'
                try:
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        device = 'mps'
                    elif torch.cuda.is_available():
                        device = 'cuda'
                except Exception:
                    device = 'cpu'

                # Move model to device if possible
                try:
                    self.separator.to(device)
                except Exception:
                    device = 'cpu'

                audio_tensor = audio_stereo.to(device)

                # Log model/device
                try:
                    model_name = type(self.separator).__name__
                    logging.info(f"Demucs model: {model_name} on device: {device}")
                except Exception:
                    logging.info(f"Demucs device: {device}")

                # Separate sources using apply_model - correct API usage
                # Safe apply_model call (avoid odd segment sizes that cause reshape errors)
                try:
                    sources_tensor = apply_model(
                        self.separator,
                        audio_tensor,
                        shifts=0,           # no test-time augmentation for speed
                        split=True,
                        overlap=0.25,
                        device=device
                    )
                except Exception as dm_e:
                    logging.warning(f"Demucs apply_model failed on {device}: {dm_e}. Retrying on CPU with safe params...")
                    try:
                        self.separator.to('cpu')
                    except Exception:
                        pass
                    audio_tensor_cpu = audio_stereo.to('cpu')
                    sources_tensor = apply_model(
                        self.separator,
                        audio_tensor_cpu,
                        shifts=1,          # default TTA for stability
                        split=True,
                        overlap=0.25,
                        device='cpu'
                    )

                # Convert back to dict format
                separated = {}
                source_names = ['drums', 'bass', 'other', 'vocals']  # HTDemucs order

                # sources_tensor should be [batch, sources, channels, time]
                if sources_tensor.dim() == 4:
                    for i, source_name in enumerate(source_names):
                        if i < sources_tensor.shape[1]:  # Check if source index exists
                            source_tensor = sources_tensor[0, i]  # [channels, time]
                            if source_tensor.dim() == 2:  # [channels, time]
                                source_tensor = source_tensor.squeeze(0)  # Remove channel dimension if mono
                            # Bring back to CPU for numpy conversion
                            separated[source_name] = source_tensor.to('cpu').numpy()
                        else:
                            # Fallback if source not available
                            separated[source_name] = audio
                else:
                    # Fallback for unexpected tensor shape
                    logging.warning(f"Unexpected Demucs output shape: {sources_tensor.shape}")
                    return {'vocals': audio, 'accompaniment': audio}

                separate_time = time.time() - start_time
                self.processing_info['separate_time'] = separate_time
                self._log_memory_usage('separate')

                logging.info(f"Source separation completed in {separate_time:.2f}s")
                
                # Always save the separated sources as they are needed by other modules
                self._save_separated_sources(audio_path, separated, sr)
                
                return separated

            except Exception as e:
                logging.error(f"Demucs separation failed: {e}")
                # Fall back to no separation
                return {'vocals': audio, 'accompaniment': audio}
        else:
            logging.info("Using fallback source separation (no separation)")
            # Simple fallback: assume vocals are in center frequencies
            # This is a very basic approach
            return {'vocals': audio, 'accompaniment': audio}

    def _save_separated_sources(self, audio_path: str, sources: Dict[str, np.ndarray], sr: int):
        """Save separated audio sources to files."""
        try:
            total_write_start = time.time()
            base_name = Path(audio_path).stem

            # Use the same original working directory as the main application
            original_cwd = os.environ.get('SONG_EDITOR_ORIGINAL_CWD', '.')
            output_dir = Path(original_cwd) / 'separated' / 'htdemucs' / base_name
            output_dir.mkdir(parents=True, exist_ok=True)

            per_file_times = {}
            for name, audio_data in sources.items():
                file_start = time.time()
                file_path = output_dir / f"{name}.wav"
                import soundfile as sf
                sf.write(file_path, audio_data.T, sr)
                logging.info(f"Saved separated source: {file_path}")
                per_file_times[name] = time.time() - file_start

            total_write_time = time.time() - total_write_start
            self.processing_info['save_separated_sources_time'] = total_write_time
            self.processing_info['save_separated_sources_per_file'] = per_file_times

        except Exception as e:
            logging.error(f"Failed to save separated sources: {e}")

    def _save_audio_temp(self, audio: np.ndarray, sr: int) -> str:
        """Save audio to temporary file."""
        import tempfile
        import os

        try:
            # Create temporary file with explicit cleanup
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)  # Close the file descriptor

            # Ensure audio is in correct format
            if len(audio.shape) > 1:
                if audio.shape[0] == 2:  # Stereo audio
                    audio = np.mean(audio, axis=0)  # Convert to mono by averaging channels
                else:
                    audio = audio.squeeze()  # Remove singleton dimensions

            # Save using soundfile with explicit parameters
            import soundfile as sf
            sf.write(temp_path, audio.astype(np.float32), sr, subtype='PCM_16')

            # Verify the file was created and is readable
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise ValueError("Temporary audio file was not created properly")

            return temp_path

        except Exception as e:
            logging.error(f"Error saving temporary audio file: {e}")
            raise

    def _save_intermediate_files(self, audio: np.ndarray, sr: int, stage: str):
        """Save intermediate audio files if requested."""
        if not self.save_intermediate:
            return

        try:
            write_start = time.time()
            output_dir = Path("intermediate_outputs")
            output_dir.mkdir(exist_ok=True)

            output_path = output_dir / f"{stage}_{int(time.time())}.wav"

            import soundfile as sf
            sf.write(str(output_path), audio, sr)

            logging.info(f"Saved intermediate file: {output_path}")
            dur = time.time() - write_start
            key = f"intermediate_write_time_{stage}"
            self.processing_info[key] = dur
            self.processing_info['intermediate_write_time_total'] = (
                self.processing_info.get('intermediate_write_time_total', 0.0) + dur
            )

        except Exception as e:
            logging.warning(f"Failed to save intermediate file: {e}")

    def process(self, audio_path: str) -> Dict[str, Any]:
        """Process a single audio file."""
        self.processing_info = {}
        try:
            logging.info(f"Starting audio processing: {audio_path}")
            total_start_time = time.time()

            # Load audio
            audio, sr = self.load_audio(audio_path)

            # Save intermediate if requested
            if self.save_intermediate:
                self._save_intermediate_files(audio, sr, "01_loaded")

            # Denoise
            audio = self.denoise_audio(audio, sr)
            if self.save_intermediate:
                self._save_intermediate_files(audio, sr, "02_denoised")

            # Normalize
            audio = self.normalize_audio(audio, sr)
            if self.save_intermediate:
                self._save_intermediate_files(audio, sr, "03_normalized")

            # 4. Source Separation
            separated_sources = self.separate_sources(audio_path, audio, sr)

            # 5. Assemble results
            final_audio_data = {
                'audio': audio,
                'sample_rate': sr,
                'analysis': {
                    'duration': len(audio) / sr,
                    'sample_rate': sr,
                    'channels': 1,
                    'audio_levels': self._calculate_audio_levels(audio, sr),
                    'tempo': self._detect_tempo(audio, sr),
                    'key': self._detect_key(audio, sr)
                },
                'processing_info': self.processing_info
            }
            final_audio_data.update(separated_sources)

            # Calculate total processing time
            total_time = time.time() - total_start_time
            self.processing_info['total_time'] = total_time

            # Print a concise per-stage summary if available
            tempo_t = self.processing_info.get('tempo_time')
            key_t = self.processing_info.get('key_time')
            sep_t = self.processing_info.get('separate_time')
            save_sep_t = self.processing_info.get('save_separated_sources_time')
            interm_t = self.processing_info.get('intermediate_write_time_total')

            summary_parts = [f"total {total_time:.2f}s"]
            if sep_t is not None:
                summary_parts.append(f"demucs {sep_t:.2f}s")
            if save_sep_t is not None:
                summary_parts.append(f"write_sep {save_sep_t:.2f}s")
            if tempo_t is not None:
                summary_parts.append(f"tempo {tempo_t:.2f}s")
            if key_t is not None:
                summary_parts.append(f"key {key_t:.2f}s")
            if interm_t is not None:
                summary_parts.append(f"intermed {interm_t:.2f}s")

            logging.info("Audio processing completed (" + ", ".join(summary_parts) + ")")

            return final_audio_data

        except Exception as e:
            logging.error(f"Audio processing failed: {e}")
            raise

    def get_timestamp(self) -> str:
        """Get current timestamp string."""
        return time.strftime("%Y%m%d_%H%M%S")

    def cleanup(self):
        """Clean up resources."""
        if self.separator:
            del self.separator
            self.separator = None

        if self.audio_data:
            self.audio_data.clear()

        self.processing_info.clear()

        logging.info("Audio processor cleaned up")
