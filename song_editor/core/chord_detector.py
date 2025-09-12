#!/usr/bin/env python3
"""
Chord Detector Module

Handles chord detection using Chordino and other methods for Song Editor 3.
"""

import os
import logging
import tempfile
import numpy as np
import librosa
import soundfile as sf
import subprocess
import json
from typing import Tuple, Dict, List, Any
import sys
from datetime import datetime
# vamp is no longer imported here, it will be used in the worker process

# Optional imports (import lazily to avoid bus errors)
CHORD_EXTRACTOR_AVAILABLE = False

# Check vamp availability (import lazily to avoid bus errors)
VAMP_AVAILABLE = False


class ChordDetector:
    """Detects chords in an audio file using various methods."""

    def __init__(
        self,
        use_chordino: bool = True,
        chord_simplification: bool = False,
        preserve_chord_richness: bool = True,
        min_confidence: float = 0.3,
        window_size: float = 0.5
    ):
        """Initialize ChordDetector."""
        self.use_chordino = use_chordino
        self.chord_simplification = chord_simplification
        self.preserve_chord_richness = preserve_chord_richness
        self.min_confidence = min_confidence
        self.window_size = window_size
        self.vamp_host = None
        self.chordino_plugin = None
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        # Set chromagram parameters as instance variables for get_detector_info
        self.chromagram_tuning_error = 0.0
        self.chromagram_n_fft = 4096
        self.chromagram_hop_length = 1024
        
        # Chord mapping for standardization
        self.chord_mapping = {
            'C': 'C', 'Cm': 'Cm', 'C7': 'C7', 'Cm7': 'Cm7', 'Cmaj7': 'Cmaj7',
            'D': 'D', 'Dm': 'Dm', 'D7': 'D7', 'Dm7': 'Dm7', 'Dmaj7': 'Dmaj7',
            'E': 'E', 'Em': 'Em', 'E7': 'E7', 'Em7': 'Em7', 'Emaj7': 'Emaj7',
            'F': 'F', 'Fm': 'Fm', 'F7': 'F7', 'Fm7': 'Fm7', 'Fmaj7': 'Fmaj7',
            'G': 'G', 'Gm': 'Gm', 'G7': 'G7', 'Gm7': 'Gm7', 'Gmaj7': 'Gmaj7',
            'A': 'A', 'Am': 'Am', 'A7': 'A7', 'Am7': 'Am7', 'Amaj7': 'Amaj7',
            'B': 'B', 'Bm': 'Bm', 'B7': 'B7', 'Bm7': 'Bm7', 'Bmaj7': 'Bmaj7',
            # Add more chord variations as needed
            'C#': 'C#', 'C#m': 'C#m', 'C#7': 'C#7', 'C#m7': 'C#m7',
            'D#': 'D#', 'D#m': 'D#m', 'D#7': 'D#7', 'D#m7': 'D#m7',
            'F#': 'F#', 'F#m': 'F#m', 'F#7': 'F#7', 'F#m7': 'F#m7',
            'G#': 'G#', 'G#m': 'G#m', 'G#7': 'G#7', 'G#m7': 'G#m7',
            'A#': 'A#', 'A#m': 'A#m', 'A#7': 'A#7', 'A#m7': 'A#m7',
        }

    def _detect_chords_chordino_worker(self, audio_path: str) -> List[Dict[str, Any]]:
        """Detect chords by calling the external chord_worker.py script."""
        params_file_path = None
        try:
            # 1. Prepare parameters for the worker, passing the direct file path
            worker_params = {
                'audio_path': audio_path,
                'use_chordino': self.use_chordino,
                'min_confidence': self.min_confidence,
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_params_file:
                json.dump(worker_params, temp_params_file)
                params_file_path = temp_params_file.name

            # 2. Find the worker script path
            script_dir = os.path.dirname(__file__)
            script_path = os.path.join(script_dir, '..', '..', 'chord_worker.py')
            if not os.path.exists(script_path):
                try:
                    import sys as _sys
                    if getattr(_sys, 'frozen', False) and hasattr(_sys, '_MEIPASS'):
                        script_path = os.path.join(_sys._MEIPASS, 'chord_worker.py')  # type: ignore[attr-defined]
                except Exception:
                    pass
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"chord_worker.py not found at {script_path}")

            # 3. Run the worker script
            python_executable = sys.executable
            if getattr(sys, 'frozen', False):
                command = [python_executable, '--worker', 'chord', '--worker-params', params_file_path]
            else:
                command = [python_executable, script_path, params_file_path]
            
            logging.info(f"Running chord worker command: {' '.join(command)}")
            process = subprocess.run(command, capture_output=True, text=True, check=True)

            if process.stderr:
                for line in process.stderr.strip().split('\n'):
                    logging.info(f"[ChordWorker] {line}")

            # 4. Parse the results
            raw_chords = json.loads(process.stdout)

            # 5. Post-process the results
            # Get audio duration for post-processing using subprocess isolation
            audio_duration = self._get_audio_duration_with_subprocess(audio_path)
            processed_chords = self._post_process_chordino_results(raw_chords, audio_duration)
            return processed_chords

        except subprocess.CalledProcessError as e:
            logging.error(f"Chord worker process failed with exit code {e.returncode}")
            logging.error(f"Stderr: {e.stderr}")
            logging.error(f"Falling back to chromagram chord detection.")
            # Fallback requires loading audio using subprocess isolation
            audio, sr = self._load_audio_with_subprocess(audio_path)
            return self._detect_chords_chromagram(audio, sr)
        except Exception as e:
            logging.error(f"An error occurred during chordino worker processing: {e}", exc_info=True)
            logging.error("Falling back to chromagram chord detection.")
            audio, sr = self._load_audio_with_subprocess(audio_path)
            return self._detect_chords_chromagram(audio, sr)
        finally:
            # 6. Clean up temporary params file
            if params_file_path and os.path.exists(params_file_path):
                os.unlink(params_file_path)

    def _post_process_chordino_results(self, raw_chords: List[Dict[str, Any]], audio_duration: float) -> List[Dict[str, Any]]:
        """Converts the list of chord events from the worker into timed chord segments."""
        processed_chords = []
        if not raw_chords:
            return []

        for i, event in enumerate(raw_chords):
            start_time = event['timestamp']
            end_time = raw_chords[i + 1]['timestamp'] if i + 1 < len(raw_chords) else audio_duration
            
            chord_symbol = event['chord']
            parsed_symbol = self._parse_chord_symbol(chord_symbol)

            processed_chords.append({
                'symbol': chord_symbol,
                'root': parsed_symbol['root'],
                'quality': parsed_symbol['quality'],
                'bass': parsed_symbol['bass'],
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'confidence': 0.85,  # Default confidence for Chordino
                'detection_method': 'chordino_worker'
            })
        
        return processed_chords

    def _detect_chords_chordino(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Wrapper for the worker-based implementation, accepting a file path.
        """
        return self._detect_chords_chordino_worker(audio_path)

    def _detect_chords_chromagram(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Detect chords using chromagram analysis via system Python subprocess to avoid cffi issues."""
        try:
            import tempfile
            tmpdir = tempfile.mkdtemp(prefix="chords_")
            in_npy = os.path.join(tmpdir, "in.npy")
            mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)
            np.save(in_npy, mono)
            hop = max(1, int(self.window_size * sample_rate))

            script = f'''\
import sys, numpy as np
sys.path.insert(0, '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages')
try:
    import librosa
    y = np.load(r"{in_npy}")
    chroma = librosa.feature.chroma_cqt(y=y, sr={sample_rate}, hop_length={hop})
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr={sample_rate}, hop_length={hop})
    chroma = chroma / (np.max(chroma) + 1e-12)
    for i in range(chroma.shape[1]):
        vec = chroma[:, i]
        t = float(times[i])
        print("FRAME:", t, ",", ",".join(str(float(v)) for v in vec))
except Exception as err:
    print("ERROR:", str(err), file=sys.stderr)
    sys.exit(1)
'''
            sp = subprocess.run(['/usr/local/bin/python3', '-c', script], capture_output=True, text=True, timeout=180)
            if sp.returncode != 0:
                logging.error(f"Error in chromagram chord detection: {sp.stderr.strip()}")
                return []

            chord_templates = self._get_chord_templates()
            chords: List[Dict[str, Any]] = []

            for line in sp.stdout.splitlines():
                if not line.startswith('FRAME:'):
                    continue
                try:
                    _, payload = line.split(':', 1)
                    time_str, vec_str = payload.split(',', 1)
                    t = float(time_str.strip())
                    vec = np.array([float(x) for x in vec_str.strip().split(',')], dtype=float)
                    best = None
                    best_corr = -1.0
                    for symbol, tmpl in chord_templates.items():
                        corr = np.corrcoef(vec, tmpl)[0, 1]
                        if corr > best_corr:
                            best_corr = corr
                            best = symbol
                    conf = max(0.0, min(1.0, (best_corr + 1) / 2))
                    if conf >= self.min_confidence and best:
                        parsed = self._parse_chord_symbol(best)
                        chords.append({
                            'symbol': best,
                            'root': parsed['root'],
                            'quality': parsed['quality'],
                            'bass': parsed['bass'],
                            'start': float(t),
                            'end': float(t + self.window_size),
                            'duration': float(self.window_size),
                            'confidence': float(conf),
                            'detection_method': 'chromagram'
                        })
                except Exception:
                    continue

            return self._merge_similar_chords(chords)

        except Exception as e:
            logging.error(f"Error in chromagram chord detection: {e}")
            return []

    def _get_chord_templates(self) -> Dict[str, np.ndarray]:
        """Get chord templates for chromagram analysis."""
        # Define comprehensive chord templates to preserve richness
        templates = {}

        # Major chords
        major_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
            templates[note] = np.roll(major_template, i)

        # Minor chords
        minor_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
            templates[f"{note}m"] = np.roll(minor_template, i)

        # Dominant 7th chords
        seventh_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
        for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
            templates[f"{note}7"] = np.roll(seventh_template, i)

        # Major 7th chords
        maj7_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])  # Same as major for now
        for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
            templates[f"{note}maj7"] = np.roll(maj7_template, i)

        # Minor 7th chords
        min7_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])  # Same as minor for now
        for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
            templates[f"{note}m7"] = np.roll(min7_template, i)

        # Diminished chords
        dim_template = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
        for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
            templates[f"{note}dim"] = np.roll(dim_template, i)

        # Augmented chords
        aug_template = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        for i, note in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']):
            templates[f"{note}aug"] = np.roll(aug_template, i)

        return templates

    def _parse_chord_symbol(self, chord_symbol: str) -> Dict[str, str]:
        """Parse chord symbol into components."""
        chord_symbol = chord_symbol.strip()

        # Default values
        root = 'C'
        quality = 'maj'
        bass = None

        if not chord_symbol:
            return {'root': root, 'quality': quality, 'bass': bass}

        # Handle inversions (e.g., C/E)
        if '/' in chord_symbol:
            parts = chord_symbol.split('/')
            chord_symbol = parts[0]
            bass = parts[1]

        # Extract root note
        if len(chord_symbol) >= 1:
            if len(chord_symbol) >= 2 and chord_symbol[1] in ['#', 'b']:
                root = chord_symbol[:2]
                quality_part = chord_symbol[2:]
            else:
                root = chord_symbol[0]
                quality_part = chord_symbol[1:]
        else:
            quality_part = ''

        # Determine quality - preserve full chord richness
        if not quality_part:
            quality = 'maj'
        elif quality_part == 'm':
            quality = 'min'
        elif quality_part == '7':
            quality = '7'
        elif quality_part == 'maj7':
            quality = 'maj7'
        elif quality_part == 'm7':
            quality = 'min7'
        elif quality_part == 'dim':
            quality = 'dim'
        elif quality_part == 'aug':
            quality = 'aug'
        elif quality_part == 'sus2':
            quality = 'sus2'
        elif quality_part == 'sus4':
            quality = 'sus4'
        elif quality_part == '9':
            quality = '9'
        elif quality_part == 'm9':
            quality = 'm9'
        elif quality_part == 'maj9':
            quality = 'maj9'
        elif quality_part == '11':
            quality = '11'
        elif quality_part == '13':
            quality = '13'
        elif quality_part == 'add9':
            quality = 'add9'
        elif quality_part == 'add11':
            quality = 'add11'
        else:
            # Preserve any other quality as-is to maintain richness
            quality = quality_part

        return {
            'root': root,
            'quality': quality,
            'bass': bass
        }

    def _simplify_chords(self, chords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standardize chord symbols while preserving full richness."""
        if not self.chord_simplification:
            return chords

        standardized_chords = []
        for chord in chords:
            symbol = chord['symbol']

            # Apply standardization mapping (preserves richness, no simplification)
            if symbol in self.chord_mapping:
                standardized_symbol = self.chord_mapping[symbol]
                chord['symbol'] = standardized_symbol

                # Update parsed components while preserving full chord information
                chord_data = self._parse_chord_symbol(standardized_symbol)
                chord['root'] = chord_data['root']
                chord['quality'] = chord_data['quality']
                chord['bass'] = chord_data['bass']
            else:
                # For unknown chords, preserve as-is to maintain richness
                chord_data = self._parse_chord_symbol(symbol)
                chord['root'] = chord_data['root']
                chord['quality'] = chord_data['quality']
                chord['bass'] = chord_data['bass']

            standardized_chords.append(chord)

        return standardized_chords

    def _merge_similar_chords(self, chords: List[Dict[str, Any]], min_duration: float = 0.5) -> List[Dict[str, Any]]:
        """Merge consecutive similar chords."""
        if not chords:
            return chords

        merged_chords = []
        current_chord = chords[0].copy()

        for next_chord in chords[1:]:
            # Check if chords are similar
            if (current_chord['symbol'] == next_chord['symbol'] and
                    abs(next_chord['start'] - current_chord['end']) < 0.1):
                # Merge chords
                current_chord['end'] = next_chord['end']
                current_chord['duration'] = current_chord['end'] - current_chord['start']
                # Average confidence
                current_chord['confidence'] = (current_chord['confidence'] + next_chord['confidence']) / 2
            else:
                # Add current chord if it meets minimum duration
                if current_chord['duration'] >= min_duration:
                    merged_chords.append(current_chord)
                current_chord = next_chord.copy()

        # Add the last chord
        if current_chord['duration'] >= min_duration:
            merged_chords.append(current_chord)

        return merged_chords

    def _get_audio_duration_with_subprocess(self, audio_path: str) -> float:
        """Get audio duration using subprocess isolation to avoid cffi conflicts."""
        try:
            import subprocess
            import tempfile
            import os
            
            script_content = f'''
import sys
import warnings
import tempfile
import os

warnings.filterwarnings("ignore")
sys.path.insert(0, '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages')

try:
    import librosa
    duration = librosa.get_duration(filename=r"{audio_path}")
    print("DURATION:" + str(duration))
    
except Exception as e:
    print("ERROR:" + str(e), file=sys.stderr)
    sys.exit(1)
'''
            
            result = subprocess.run(
                ['/usr/local/bin/python3', '-c', script_content],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise Exception(f"Subprocess failed: {result.stderr}")
            
            for line in result.stdout.strip().split('\n'):
                if line.startswith('DURATION:'):
                    return float(line.split(':', 1)[1])
            
            raise Exception("Failed to parse duration from subprocess output")
            
        except Exception as e:
            logging.warning(f"Subprocess duration loading failed: {e}, using fallback")
            # Fallback: estimate duration from file size (very rough)
            try:
                import os
                file_size = os.path.getsize(audio_path)
                # Rough estimate: assume 16-bit, 44.1kHz, mono
                estimated_duration = file_size / (44100 * 2)
                return estimated_duration
            except:
                return 180.0  # Default 3 minutes

    def _load_audio_with_subprocess(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio using subprocess isolation to avoid cffi conflicts."""
        try:
            import subprocess
            import tempfile
            import os
            
            script_content = f'''
import sys
import warnings
import numpy as np
import tempfile
import os

warnings.filterwarnings("ignore")
sys.path.insert(0, '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages')

try:
    import librosa
    audio, sr = librosa.load(r"{audio_path}", sr=44100, mono=True)
    
    audio = np.array(audio, dtype=np.float32)
    sr = int(sr)
    
    temp_fd, temp_path = tempfile.mkstemp(suffix='.npy')
    os.close(temp_fd)
    
    np.save(temp_path, audio)
    print("AUDIO_DATA_PATH:" + temp_path)
    print("SAMPLE_RATE:" + str(sr))
    print("SHAPE:" + str(audio.shape))
    
except Exception as e:
    print("ERROR:" + str(e), file=sys.stderr)
    sys.exit(1)
'''
            
            result = subprocess.run(
                ['/usr/local/bin/python3', '-c', script_content],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise Exception(f"Subprocess failed: {result.stderr}")
            
            audio_data_path = None
            sample_rate = None
            
            for line in result.stdout.strip().split('\n'):
                if line.startswith('AUDIO_DATA_PATH:'):
                    audio_data_path = line.split(':', 1)[1]
                elif line.startswith('SAMPLE_RATE:'):
                    sample_rate = int(line.split(':', 1)[1])
            
            if not audio_data_path or sample_rate is None:
                raise Exception("Failed to parse subprocess output")
            
            audio = np.load(audio_data_path)
            
            # Clean up temp file
            try:
                os.unlink(audio_data_path)
            except:
                pass
                
            return audio, sample_rate
            
        except Exception as e:
            logging.warning(f"Subprocess audio loading failed: {e}, falling back to direct load")
            # This fallback will likely fail due to cffi conflicts, but we try anyway
            import librosa
            return librosa.load(audio_path, sr=44100, mono=True)

    def detect(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Detect chords in audio using the selected method."""
        start_time = datetime.now()

        # This can be used to debug the contents of the audio buffer
        # if needed in the future.
        # sf.write('debug_chord_audio.wav', audio, sample_rate)
        
        try:
            # --- REMOVING DIAGNOSTIC LOGGING ---
            # logging.info("Starting chord detection...")
            # logging.info(f"[DIAGNOSTIC] ChordDetector init params: use_chordino={self.use_chordino}")
            # logging.info(f"[DIAGNOSTIC] Received audio data: type={type(audio)}, shape={audio.shape}, dtype={audio.dtype}")
            # logging.info(f"[DIAGNOSTIC] Audio stats: min={np.min(audio):.4f}, max={np.max(audio):.4f}, mean={np.mean(audio):.4f}")
            # logging.info(f"[DIAGNOSTIC] Sample rate: {sample_rate}")
            
            if self.use_chordino:
                # The public detect method now expects a file path for chordino
                raise NotImplementedError("Chordino detection requires a file path passed to detect_from_path.")
            else:
                # Chromagram still works on in-memory audio
                chords = self._detect_chords_chromagram(audio, sample_rate)
                logging.info(f"Chromagram detected {len(chords)} chords")
            
            # Simplify chords if requested
            if self.chord_simplification:
                chords = self._simplify_chords(chords)

            # Merge similar consecutive chords
            chords = self._merge_similar_chords(chords)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            logging.info(f"Chord detection completed: {len(chords)} chords in {processing_time:.1f} seconds")

            return chords

        except Exception as e:
            logging.error(f"Error in chord detection: {e}")
            raise

    def detect_from_path(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Detect chords from an audio file path. This is the new primary method.
        """
        start_time = datetime.now()
        try:
            if self.use_chordino:
                chords = self._detect_chords_chordino(audio_path)
                logging.info(f"Chordino worker detected {len(chords)} chords")
            else:
                # For chromagram, we need to load the audio into memory using subprocess isolation
                audio, sr = self._load_audio_with_subprocess(audio_path)
                chords = self._detect_chords_chromagram(audio, sr)
                logging.info(f"Chromagram detected {len(chords)} chords")

            # Common post-processing
            if self.chord_simplification:
                chords = self._simplify_chords(chords)
            chords = self._merge_similar_chords(chords)

            processing_time = (datetime.now() - start_time).total_seconds()
            logging.info(f"Chord detection completed: {len(chords)} chords in {processing_time:.1f} seconds")
            return chords

        except Exception as e:
            logging.error(f"Error in chord detection from path: {e}", exc_info=True)
            raise

    def analyze_chord_progression(self, chords: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze chord progression for patterns and key detection."""
        try:
            if not chords:
                return {
                    'key': 'Unknown',
                    'confidence': 0.0,
                    'common_progressions': [],
                    'chord_frequencies': {}
                }

            # Count chord frequencies
            chord_counts = {}
            for chord in chords:
                root = chord['root']
                chord_counts[root] = chord_counts.get(root, 0) + 1

            # Find most common chord (potential key)
            if chord_counts:
                most_common_chord = max(chord_counts.items(), key=lambda x: x[1])
                key = most_common_chord[0]
                confidence = min(1.0, most_common_chord[1] / len(chords))
            else:
                key = 'Unknown'
                confidence = 0.0

            # Find common progressions
            progressions = []
            for i in range(len(chords) - 1):
                progression = f"{chords[i]['root']} -> {chords[i+1]['root']}"
                progressions.append(progression)

            # Count progression frequencies
            progression_counts = {}
            for prog in progressions:
                progression_counts[prog] = progression_counts.get(prog, 0) + 1

            # Get most common progressions
            common_progressions = sorted(
                progression_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            return {
                'key': key,
                'confidence': confidence,
                'common_progressions': common_progressions,
                'chord_frequencies': chord_counts
            }

        except Exception as e:
            logging.warning(f"Error in chord progression analysis: {e}")
            return {
                'key': 'Unknown',
                'confidence': 0.0,
                'common_progressions': [],
                'chord_frequencies': {}
            }

    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the chord detector configuration."""
        return {
            'use_chordino': self.use_chordino,
            'vamp_available': VAMP_AVAILABLE,
            'chord_simplification': self.chord_simplification,
            'preserve_chord_richness': self.preserve_chord_richness,
            'chromagram_tuning_error': self.chromagram_tuning_error,
            'chromagram_n_fft': self.chromagram_n_fft,
            'chromagram_hop_length': self.chromagram_hop_length
        }

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
    audio, sr = librosa.load("{audio_path}", sr=44100, mono=True)

    # Save as numpy array temporarily
    np.save("/tmp/chord_audio_data.npy", audio)
    print(f"SAMPLE_RATE: {{sr}}")
    print(f"SUCCESS: Audio loaded")

except Exception as err:
    print("ERROR:", str(err), file=sys.stderr)
    sys.exit(1)
'''

            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                script_path = f.name

            # Run the script with system Python
            result = subprocess.run(
                ['/usr/local/bin/python3', script_path],
                capture_output=True,
                text=True,
                timeout=60
            )

            # Clean up script file
            os.unlink(script_path)

            if result.returncode == 0:
                # Parse the output to get sample rate
                sr = None
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('SAMPLE_RATE:'):
                        sr = int(float(line.split(':')[1].strip()))

                # Load the audio data
                audio = np.load("/tmp/chord_audio_data.npy")
                
                # Clean up temp file
                try:
                    os.unlink("/tmp/chord_audio_data.npy")
                except:
                    pass

                if sr is None:
                    raise ValueError("Could not parse sample rate from subprocess output")

                logging.info(f"Loaded audio data from subprocess: {len(audio)} samples")
                return audio, sr

            else:
                raise Exception(f"Subprocess failed: {result.stderr}")

        except Exception as e:
            logging.error(f"Failed to load audio via subprocess: {e}")
            # Clean up any temp files
            try:
                os.unlink("/tmp/chord_audio_data.npy")
            except:
                pass
            raise
