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
import subprocess

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
        demucs_model: str = 'htdemucs',
        # Audio-separator options
        separation_engine: str = 'demucs',  # 'demucs' or 'audio_separator'
        audio_separator_model: str = 'UVR_MDXNET_KARA_2',
        use_cuda: bool = False,
        use_coreml: bool = True
    ):
        self.use_demucs = use_demucs
        self.save_intermediate = save_intermediate
        self.target_sr = target_sr
        self.denoise_strength = denoise_strength
        self.normalize_lufs = normalize_lufs
        self.demucs_model = demucs_model

        # Audio-separator settings
        self.separation_engine = separation_engine.lower()
        self.audio_separator_model = audio_separator_model
        self.use_cuda = use_cuda
        self.use_coreml = use_coreml

        self.separator = None
        self.audio_separator_processor = None
        self.audio_data = None
        self.processing_info = {}

        # Initialize appropriate separation engine
        if self.separation_engine == 'audio_separator':
            self._initialize_audio_separator()
            self.using_audio_separator = True
        elif self.separation_engine == 'demucs':
            if self.use_demucs:
                self._initialize_demucs()
            self.using_audio_separator = False
        else:
            logging.warning(f"Unknown separation engine: {self.separation_engine}, falling back to demucs")
            self.separation_engine = 'demucs'
            if self.use_demucs:
                self._initialize_demucs()
            self.using_audio_separator = False

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

    def _initialize_audio_separator(self):
        """Initialize Audio-Separator processor."""
        try:
            # Import audio-separator lazily
            from .audio_separator_processor import AudioSeparatorProcessor

            if not AudioSeparatorProcessor.is_available():
                raise ImportError("audio-separator library not available")

            # Set output directory for audio-separator
            original_cwd = os.environ.get('SONG_EDITOR_ORIGINAL_CWD', os.getcwd())
            output_dir = os.path.join(original_cwd, 'separated', 'htdemucs')
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            self.audio_separator_processor = AudioSeparatorProcessor(
                model_name=self.audio_separator_model,
                output_dir=output_dir,
                use_cuda=self.use_cuda,
                use_coreml=self.use_coreml,
                log_level=20  # INFO level
            )

            logging.info(f"Audio-Separator initialized successfully (model: {self.audio_separator_model})")
            logging.info(f"GPU acceleration - CUDA: {self.use_cuda}, CoreML: {self.use_coreml}")

        except ImportError as e:
            logging.warning(f"Audio-Separator not available: {e}")
            logging.info("Falling back to Demucs")
            self.separation_engine = 'demucs'
            if self.use_demucs:
                self._initialize_demucs()
        except Exception as e:
            logging.error(f"Failed to initialize Audio-Separator: {e}")
            logging.info("Falling back to Demucs")
            self.separation_engine = 'demucs'
            if self.use_demucs:
                self._initialize_demucs()

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
        """Detect tempo using system Python subprocess to avoid cffi conflicts."""
        try:
            import tempfile
            import numpy as _np
            start = time.time()
            tmpdir = tempfile.mkdtemp(prefix="tempo_")
            in_npy = os.path.join(tmpdir, "audio.npy")
            _np.save(in_npy, audio)

            script = f'''\
import sys, numpy as np
sys.path.insert(0, '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages')
try:
    # Fix scipy/librosa compatibility issue
    import scipy.signal
    if not hasattr(scipy.signal, 'hann'):
        import scipy.signal.windows
        scipy.signal.hann = scipy.signal.windows.hann
    
    import librosa
    y = np.load(r"{in_npy}")
    t, _ = librosa.beat.beat_track(y=y, sr={sr})
    print("TEMPO:", float(t))
except ImportError as err:
    print("ERROR: librosa not available -", str(err), file=sys.stderr)
    sys.exit(1)
except Exception as err:
    print("ERROR:", str(err), file=sys.stderr)
    sys.exit(1)
'''
            sp = subprocess.run(['/usr/local/bin/python3', '-c', script], capture_output=True, text=True, timeout=60)
            self.processing_info['tempo_time'] = time.time() - start
            if sp.returncode != 0:
                logging.warning(f"Tempo subprocess failed: {sp.stderr.strip()}")
                return None
            for line in sp.stdout.splitlines():
                if line.startswith('TEMPO:'):
                    return float(line.split(':', 1)[1].strip())
            return None
        except Exception as e:
            logging.warning(f"Tempo detection failed: {e}")
            return None

    def _detect_key(self, audio: np.ndarray, sr: int) -> Optional[Dict[str, Any]]:
        """Detect musical key using system Python subprocess to avoid cffi conflicts."""
        try:
            import tempfile
            import numpy as _np
            start = time.time()
            tmpdir = tempfile.mkdtemp(prefix="key_")
            in_npy = os.path.join(tmpdir, "audio.npy")
            _np.save(in_npy, audio)

            script = f'''\
import sys, numpy as np
sys.path.insert(0, '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages')
try:
    # Fix scipy/librosa compatibility issue
    import scipy.signal
    if not hasattr(scipy.signal, 'hann'):
        import scipy.signal.windows
        scipy.signal.hann = scipy.signal.windows.hann
    
    import librosa
    y = np.load(r"{in_npy}")
    chroma = librosa.feature.chroma_cqt(y=y, sr={sr})
    prof = np.mean(chroma, axis=1)
    idx = int(np.argmax(prof))
    names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    key = names[idx]
    maj = [0,2,4,5,7,9,11]
    mino = [0,2,3,5,7,8,10]
    maj_s = float(sum(prof[(idx+n)%12] for n in maj))
    min_s = float(sum(prof[(idx+n)%12] for n in mino))
    mode = 'major' if maj_s >= min_s else 'minor'
    conf = float(max(maj_s, min_s) / (float(sum(prof)) + 1e-12))
    print("KEY:", key, mode, conf)
except ImportError as err:
    print("ERROR: librosa not available -", str(err), file=sys.stderr)
    sys.exit(1)
except Exception as err:
    print("ERROR:", str(err), file=sys.stderr)
    sys.exit(1)
'''
            sp = subprocess.run(['/usr/local/bin/python3', '-c', script], capture_output=True, text=True, timeout=60)
            self.processing_info['key_time'] = time.time() - start
            if sp.returncode != 0:
                logging.warning(f"Key subprocess failed: {sp.stderr.strip()}")
                return None
            for line in sp.stdout.splitlines():
                if line.startswith('KEY:'):
                    parts = line.split()
                    return {'key': f"{parts[1]}", 'mode': parts[2], 'confidence': float(parts[3])}
            return None
        except Exception as e:
            logging.warning(f"Key detection failed: {e}")
            return None

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        try:
            logging.info(f"Loading audio: {file_path}")
            start_time = time.time()

            # Always use subprocess isolation for librosa to avoid cffi conflicts
            logging.info("Loading audio via subprocess to avoid cffi conflicts")
            try:
                audio, sr = self._load_audio_with_subprocess(file_path)
                # Ensure we have valid audio data
                if len(audio) == 0:
                    raise ValueError("Empty audio data from subprocess")
            except Exception as e:
                logging.error(f"Failed to load audio via subprocess: {e}, falling back to direct load")
                # Fallback to direct loading (will have cffi issues but at least works)
                logging.warning("⚠️  Using direct librosa.load - may encounter cffi conflicts")
                audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)

            # Store original info
            # Ensure audio and sr are proper types before calculating duration
            try:
                duration = float(len(audio)) / float(sr) if audio is not None and sr > 0 else 0.0
            except (TypeError, ValueError, ZeroDivisionError) as e:
                logging.error(f"Duration calculation error: {e}")
                logging.error(f"  audio type={type(audio)}, audio value preview={str(audio)[:100] if audio is not None else 'None'}")
                logging.error(f"  sr type={type(sr)}, sr value={sr}")
                duration = 0.0
                
            self.audio_data = {
                'original_path': file_path,
                'original_sr': sr,
                'duration': duration,
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
            # Skip denoising if audio is empty
            if audio is None or len(audio) == 0:
                logging.info("Skipping denoising (no audio)")
                return audio

            logging.info("Denoising audio via subprocess...")
            start_time = time.time()

            # Apply noise reduction via subprocess
            denoised = self._denoise_with_subprocess(audio, sr)

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

    def _denoise_with_subprocess(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Denoise audio via system Python subprocess using noisereduce."""
        try:
            import tempfile
            import numpy as _np
            tmpdir = tempfile.mkdtemp(prefix="denoise_")
            in_npy = os.path.join(tmpdir, "in.npy")
            out_npy = os.path.join(tmpdir, "out.npy")
            _np.save(in_npy, audio)

            script = f'''\
import sys, numpy as np
sys.path.insert(0, '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages')
try:
    import noisereduce as nr
    y = np.load(r"{in_npy}")
    out = nr.reduce_noise(y=y, sr={sr}, stationary=True, prop_decrease=0.8)
    np.save(r"{out_npy}", out)
    print("DENOISED")
except ImportError:
    # noisereduce not available, return original audio
    y = np.load(r"{in_npy}")
    np.save(r"{out_npy}", y)
    print("SKIPPED: noisereduce not available")
except Exception as err:
    print("ERROR:", str(err), file=sys.stderr)
    sys.exit(1)
'''
            sp = subprocess.run(['/usr/local/bin/python3', '-c', script], capture_output=True, text=True, timeout=120)
            if sp.returncode != 0:
                logging.warning(f"Subprocess denoise failed: {sp.stderr.strip()}")
                return audio
            
            # Check if denoising was skipped
            if "SKIPPED" in sp.stdout:
                logging.info("Denoising skipped - noisereduce not available in system Python")
            elif "DENOISED" in sp.stdout:
                logging.info("Audio denoised successfully")
                
            return _np.load(out_npy)
        except Exception as e:
            logging.warning(f"Denoise subprocess error: {e}")
            return audio

    def normalize_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Normalize audio to target LUFS."""
        try:
            # Skip normalization if audio is empty (audio-separator case)
            if len(audio) == 0:
                logging.info("Skipping normalization for audio-separator (empty audio)")
                return audio

            # Import pyloudnorm lazily
            import pyloudnorm as pyln

            logging.info("Normalizing audio...")
            start_time = time.time()

            # Create loudness meter
            meter = pyln.Meter(sr)

            # Measure current loudness - this can take a long time for long audio files
            logging.info(f"Measuring loudness for {len(audio)/sr:.1f}s audio file...")
            try:
                # Add timeout for very long audio files
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Loudness measurement timeout")
                
                # Set 30 second timeout for loudness measurement
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
                
                current_lufs = meter.integrated_loudness(audio)
                signal.alarm(0)  # Cancel the alarm
                logging.info(f"Current loudness: {current_lufs:.1f} LUFS")
                
            except (TimeoutError, Exception) as e:
                signal.alarm(0)  # Cancel the alarm
                logging.warning(f"Loudness measurement failed or timed out: {e}, using fallback RMS normalization")
                # Fall back to RMS normalization
                rms = np.sqrt(np.mean(audio**2))
                target_rms = 0.1
                if rms > 0:
                    gain_linear = target_rms / rms
                    normalized = audio * gain_linear
                    max_val = np.max(np.abs(normalized))
                    if max_val > 0.95:
                        normalized = normalized * (0.95 / max_val)
                    
                    normalize_time = time.time() - start_time
                    self.processing_info['normalize_time'] = normalize_time
                    logging.info(f"RMS normalization completed: RMS {rms:.3f} -> {target_rms:.3f}")
                    return normalized
                else:
                    logging.warning("Audio RMS is zero, skipping normalization")
                    return audio

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
        """Separate audio sources using selected engine (Demucs or Audio-Separator)."""
        start_time = time.time()

        # Use Audio-Separator if selected and available
        if self.separation_engine == 'audio_separator' and self.audio_separator_processor:
            try:
                logging.info(f"Separating audio sources with Audio-Separator ({self.audio_separator_model})...")

                # Audio-separator handles file I/O internally
                result = self.audio_separator_processor.separate_audio(audio_path)

                if result['success']:
                    # Convert audio-separator output to expected format
                    separated = {}

                    # Map audio-separator stems to expected format
                    stem_mapping = {
                        'vocals': 'vocals',
                        'instrumental': 'accompaniment',
                        'drums': 'drums',
                        'bass': 'bass',
                        'other': 'other',
                        'piano': 'piano',
                        'guitar': 'guitar'
                    }

                    # For audio-separator, skip loading individual stems to avoid cffi conflicts
                    # We'll load the vocals stem later in the main processing pipeline
                    logging.info("Audio-separator completed - stems available in output directory")

                    # Create placeholder entries for required stems
                    for stem_name in result['stems_found']:
                        expected_name = stem_mapping.get(stem_name, stem_name)
                        # Don't load audio here to avoid cffi issues - will load vocals later
                        separated[expected_name] = None  # Placeholder

                    # Ensure we have vocals and accompaniment (required by other modules)
                    if 'vocals' not in separated:
                        if 'instrumental' in separated:
                            # Calculate vocals as difference from instrumental
                            separated['vocals'] = audio - separated['instrumental']
                        else:
                            separated['vocals'] = audio

                    if 'accompaniment' not in separated:
                        if 'vocals' in separated:
                            separated['accompaniment'] = audio - separated['vocals']
                        else:
                            separated['accompaniment'] = audio

                    separate_time = time.time() - start_time
                    self.processing_info['separate_time'] = separate_time
                    self._log_memory_usage('separate')

                    logging.info(f"Audio-Separator separation completed in {separate_time:.2f}s")
                    logging.info(f"Generated stems: {list(separated.keys())}")

                    # Save separated sources for compatibility with other modules
                    self._save_separated_sources(audio_path, separated, sr)

                    return separated

                else:
                    logging.error(f"Audio-Separator failed: {result.get('error', 'Unknown error')}")
                    # Fall back to Demucs or no separation
                    if self.use_demucs and self.separator:
                        logging.info("Falling back to Demucs...")
                        return self._separate_with_demucs(audio_path, audio, sr, start_time)
                    else:
                        return self._fallback_separation(audio)

            except Exception as e:
                logging.error(f"Audio-Separator separation failed: {e}")
                # Fall back to Demucs or no separation
                if self.use_demucs and self.separator:
                    logging.info("Falling back to Demucs...")
                    return self._separate_with_demucs(audio_path, audio, sr, start_time)
                else:
                    return self._fallback_separation(audio)

        # Use Demucs (original implementation)
        elif self.separation_engine == 'demucs' and self.use_demucs and self.separator:
            return self._separate_with_demucs(audio_path, audio, sr, start_time)

        # Fallback to no separation
        else:
            logging.info("Using fallback source separation (no separation)")
            return self._fallback_separation(audio)

    def _load_audio_with_subprocess(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio using system Python subprocess to avoid cffi conflicts."""
        try:
            # Create a script to load audio with system Python
            script_content = f'''
import sys
import warnings
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add system site-packages to path
sys.path.insert(0, '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages')

try:
    import librosa

    # Load audio file
    audio, sr = librosa.load("{audio_path}", sr=None, mono=True)

    # Save as numpy array temporarily
    np.save("/tmp/audio_data.npy", audio)
    print(f"SAMPLE_RATE: {{sr}}")
    print(f"SUCCESS: Audio loaded")

except Exception as err:
    print("ERROR:", str(err), file=sys.stderr)
    sys.exit(1)
'''

            # Write the script
            script_path = "/tmp/load_audio.py"
            with open(script_path, 'w') as f:
                f.write(script_content)

            # Run the script with system Python
            cmd = ['/usr/local/bin/python3', script_path]
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout
            )

            # Clean up script
            try:
                os.remove(script_path)
            except:
                pass

            if process.returncode == 0:
                # Load the saved numpy array
                temp_file = "/tmp/audio_data.npy"
                if os.path.exists(temp_file):
                    audio_data = np.load(temp_file)
                    logging.info(f"Loaded audio data from subprocess: {len(audio_data)} samples")
                else:
                    logging.error("Temp audio file not found")
                    return np.array([]), 44100

                # Parse sample rate from output
                sr = None
                for line in process.stdout.strip().split('\n'):
                    if line.startswith('SAMPLE_RATE:'):
                        sr_str = line.split(':', 1)[1].strip()
                        try:
                            # Try to parse as float first, then convert to int
                            sr = int(float(sr_str))
                        except (ValueError, TypeError):
                            logging.warning(f"Could not parse sample rate '{sr_str}', using default 44100")
                            sr = 44100
                        break

                if sr is None:
                    sr = 44100
                    logging.warning("Could not parse sample rate, using default 44100")

                # Clean up temp file
                try:
                    os.remove(temp_file)
                except:
                    pass

                logging.info(f"Returning audio data: {len(audio_data)} samples at {sr} Hz")
                return audio_data, sr
            else:
                error_msg = process.stderr.strip()
                raise Exception(f"Failed to load audio with subprocess: {error_msg}")

        except Exception as e:
            logging.error(f"Failed to load audio with subprocess: {e}")
            raise

    def _reconstruct_vocals_from_stems(self, audio_path: str, output_dir: str, target_sr: int) -> Optional[np.ndarray]:
        """Reconstruct vocals by subtracting working stems from original audio."""
        try:
            # First load the original audio
            original_audio, original_sr = self._load_audio_with_subprocess(audio_path)
            if len(original_audio) == 0:
                return None

            # Resample to target rate if needed
            if original_sr != target_sr:
                import librosa
                original_audio = librosa.resample(original_audio, orig_sr=original_sr, target_sr=target_sr)

            logging.info(f"Loaded original audio: {len(original_audio)} samples for reconstruction")

            # Try to load working stems
            stems_to_subtract = []
            stem_files = ['bass.wav', 'drums.wav', 'other.wav']

            for stem_file in stem_files:
                stem_path = os.path.join(output_dir, os.path.splitext(os.path.basename(audio_path))[0], stem_file)
                if os.path.exists(stem_path) and os.path.getsize(stem_path) > 1000:  # Check if file is not empty
                    try:
                        stem_audio, stem_sr = self._load_audio_with_subprocess(stem_path)
                        if len(stem_audio) > 0:
                            # Resample if needed
                            if stem_sr != target_sr:
                                import librosa
                                stem_audio = librosa.resample(stem_audio, orig_sr=stem_sr, target_sr=target_sr)

                            # Ensure same length as original
                            if len(stem_audio) > len(original_audio):
                                stem_audio = stem_audio[:len(original_audio)]
                            elif len(stem_audio) < len(original_audio):
                                # Pad with zeros if shorter
                                padding = np.zeros(len(original_audio) - len(stem_audio))
                                stem_audio = np.concatenate([stem_audio, padding])

                            stems_to_subtract.append(stem_audio)
                            logging.info(f"Loaded stem for reconstruction: {stem_file} ({len(stem_audio)} samples)")
                    except Exception as e:
                        logging.warning(f"Failed to load stem {stem_file}: {e}")

            if not stems_to_subtract:
                logging.warning("No working stems found for vocals reconstruction")
                return None

            # Reconstruct vocals: original - sum of other stems
            reconstructed_vocals = original_audio.copy()
            for stem_audio in stems_to_subtract:
                reconstructed_vocals -= stem_audio

            # Ensure we don't have negative values (clamp to reasonable range)
            reconstructed_vocals = np.clip(reconstructed_vocals, -1.0, 1.0)

            logging.info(f"Reconstructed vocals from {len(stems_to_subtract)} stems: {len(reconstructed_vocals)} samples")
            return reconstructed_vocals

        except Exception as e:
            logging.error(f"Failed to reconstruct vocals: {e}")
            return None

    def _separate_with_demucs(self, audio_path: str, audio: np.ndarray, sr: int, start_time: float) -> Dict[str, np.ndarray]:
        """Separate audio sources using Demucs."""
        try:
            logging.info("Separating audio sources with Demucs...")

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
                return self._fallback_separation(audio)

            separate_time = time.time() - start_time
            self.processing_info['separate_time'] = separate_time
            self._log_memory_usage('separate')

            logging.info(f"Demucs separation completed in {separate_time:.2f}s")

            # Always save the separated sources as they are needed by other modules
            self._save_separated_sources(audio_path, separated, sr)

            return separated

        except Exception as e:
            logging.error(f"Demucs separation failed: {e}")
            return self._fallback_separation(audio)

    def _fallback_separation(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Fallback separation when no engine is available."""
        logging.info("Using fallback source separation (no separation)")
        # Simple fallback: assume vocals are in center frequencies
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
                # Skip None values (placeholders from audio-separator)
                if audio_data is None:
                    continue

                file_start = time.time()
                file_path = output_dir / f"{name}.wav"

                # Check if file already exists (from audio-separator)
                if file_path.exists():
                    logging.info(f"Separated source already exists: {file_path}")
                else:
                    # Save the audio data
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

            # For audio-separator, use original audio for now (separation files created but loading has cffi issues)
            vocals_audio = audio.copy() if audio is not None else None  # Make a copy to avoid reference issues
            if self.separation_engine == 'audio_separator':
                logging.info("Audio-separator processing completed - using original audio for analysis (separation files saved)")

            # 5. Assemble results
            # Safety check for audio data
            if vocals_audio is None or len(vocals_audio) == 0:
                logging.error("Audio data is None or empty, cannot perform analysis")
                vocals_audio = audio if audio is not None and len(audio) > 0 else None

            if vocals_audio is None or len(vocals_audio) == 0:
                logging.error("No valid audio data available, skipping analysis")
                # Safety check for duration calculation
                try:
                    duration = float(len(audio)) / float(sr) if audio is not None and sr > 0 else 0.0
                except (TypeError, ValueError, ZeroDivisionError) as e:
                    logging.error(f"Duration calculation error in error path: {e}")
                    logging.error(f"  audio type={type(audio)}, audio value preview={str(audio)[:100] if audio is not None else 'None'}")
                    logging.error(f"  sr type={type(sr)}, sr value={sr}")
                    duration = 0.0
                    
                final_audio_data = {
                    'audio': audio,
                    'sample_rate': sr,
                    'analysis': {
                        'duration': duration,
                        'sample_rate': sr,
                        'channels': 1,
                        'audio_levels': [],
                        'tempo': None,
                        'key': None
                    },
                    'processing_info': self.processing_info
                }
            else:
                # Use original audio for analysis; offload tempo/key safely
                safe_audio = self.denoise_audio(audio, sr)
                tempo_val = self._detect_tempo(safe_audio, sr)
                key_val = self._detect_key(safe_audio, sr)

                # Safety check for duration calculation
                try:
                    duration = float(len(audio)) / float(sr) if audio is not None and sr > 0 else 0.0
                except (TypeError, ValueError, ZeroDivisionError) as e:
                    logging.error(f"Duration calculation error in error path: {e}")
                    logging.error(f"  audio type={type(audio)}, audio value preview={str(audio)[:100] if audio is not None else 'None'}")
                    logging.error(f"  sr type={type(sr)}, sr value={sr}")
                    duration = 0.0
                    
                final_audio_data = {
                    'audio': audio,
                    'sample_rate': sr,
                    'analysis': {
                        'duration': duration,
                        'sample_rate': sr,
                        'channels': 1,
                        'audio_levels': self._calculate_audio_levels(audio, sr),
                        'tempo': tempo_val,
                        'key': key_val
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
        """Clean up resources and temporary files."""
        if self.separator:
            del self.separator
            self.separator = None

        if self.audio_data:
            self.audio_data.clear()

        self.processing_info.clear()
        
        # Clean up intermediate output files
        try:
            intermediate_dir = Path("intermediate_outputs")
            if intermediate_dir.exists():
                for file in intermediate_dir.glob("*"):
                    file.unlink()
                logging.info("Cleaned up intermediate output files")
        except Exception as e:
            logging.warning(f"Could not clean up intermediate files: {e}")
        
        # Clean up separated audio files older than 1 hour
        try:
            separated_dir = Path("separated")
            if separated_dir.exists():
                import time
                current_time = time.time()
                for subdir in separated_dir.rglob("*"):
                    if subdir.is_file():
                        # Remove files older than 1 hour (3600 seconds)
                        if current_time - subdir.stat().st_mtime > 3600:
                            subdir.unlink()
                            logging.debug(f"Cleaned up old separated file: {subdir}")
                # Remove empty directories
                for subdir in sorted(separated_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                    if subdir.is_dir() and not any(subdir.iterdir()):
                        subdir.rmdir()
                        logging.debug(f"Removed empty directory: {subdir}")
        except Exception as e:
            logging.warning(f"Could not clean up separated files: {e}")

        logging.info("Audio processor cleaned up")
