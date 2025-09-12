from __future__ import annotations

import threading
import time
import subprocess
import tempfile
import os
import logging
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import librosa


class AudioPlayer:
    def __init__(self) -> None:
        self.audio: Optional[np.ndarray] = None
        self.sr: int = 0
        self._stream: Optional[sd.OutputStream] = None
        self._lock = threading.RLock()
        self._play_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._paused = False
        self._pos = 0

    def load(self, path: str) -> None:
        # Use subprocess isolation to avoid cffi conflicts (same as audio processing pipeline)
        try:
            data, sr = self._load_audio_with_subprocess(path)
        except Exception as e:
            logging.warning(f"Subprocess audio loading failed: {e}, falling back to direct load")
            # Fallback to direct loading (will have cffi issues but at least works)
            data, sr = librosa.load(path, sr=None, mono=False)
        
        # Ensure data is in the right format for sounddevice
        if len(data.shape) == 1:
            # Mono audio - convert to 2D for sounddevice
            data = data.reshape(1, -1)
        elif data.shape[0] > 2:
            # More than 2 channels - take first 2 for stereo
            data = data[:2]
        elif data.shape[0] == 1:
            # Single channel - duplicate for stereo
            data = np.vstack([data, data])
        
        # Transpose to (samples, channels) format for sounddevice
        data = data.T
        
        self.audio = data.astype(np.float32)
        self.sr = sr
        self._pos = 0

    def _load_audio_with_subprocess(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio using system Python subprocess to avoid cffi conflicts."""
        try:
            # Create a script to load audio with system Python
            script_content = f'''
import sys
import warnings
import numpy as np
import tempfile
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Add system Python path
sys.path.insert(0, '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages')

try:
    import librosa
    data, sr = librosa.load(r"{audio_path}", sr=None, mono=False)
    
    # Convert to numpy arrays and ensure proper types
    data = np.array(data, dtype=np.float32)
    sr = int(sr)
    
    # Save to temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.npy')
    os.close(temp_fd)
    
    np.save(temp_path, data)
    print("AUDIO_DATA_PATH:" + temp_path)
    print("SAMPLE_RATE:" + str(sr))
    print("SHAPE:" + str(data.shape))
    
except Exception as e:
    print("ERROR:" + str(e), file=sys.stderr)
    sys.exit(1)
'''
            
            # Run the script with system Python
            result = subprocess.run(
                ['/usr/local/bin/python3', '-c', script_content],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise Exception(f"Subprocess failed: {result.stderr}")
            
            # Parse the output
            audio_data_path = None
            sample_rate = None
            shape = None
            
            for line in result.stdout.strip().split('\n'):
                if line.startswith('AUDIO_DATA_PATH:'):
                    audio_data_path = line.split(':', 1)[1]
                elif line.startswith('SAMPLE_RATE:'):
                    sample_rate = int(line.split(':', 1)[1])
                elif line.startswith('SHAPE:'):
                    shape_str = line.split(':', 1)[1]
                    shape = eval(shape_str)  # Safe since we control the output
            
            if not audio_data_path or sample_rate is None or shape is None:
                raise Exception("Failed to parse subprocess output")
            
            # Load the audio data
            audio_data = np.load(audio_data_path)
            
            # Clean up temporary file
            try:
                os.unlink(audio_data_path)
            except:
                pass
            
            return audio_data, sample_rate
            
        except Exception as e:
            logging.error(f"Subprocess audio loading failed: {e}")
            raise

    def toggle_play_pause(self) -> None:
        with self._lock:
            self._paused = not self._paused

    def stop(self) -> None:
        self._stop_flag.set()
        if self._stream is not None:
            self._stream.abort()
            self._stream.close()
            self._stream = None
        self._play_thread = None

    def play_segment(self, start_s: float, end_s: float) -> None:
        if self.audio is None or self.sr <= 0:
            return
        start = max(0, int(start_s * self.sr))
        end = min(self.audio.shape[0], int(end_s * self.sr))
        segment = self.audio[start:end]
        if segment.size == 0:
            return
        self.stop()

        self._stop_flag.clear()
        self._paused = False

        def run() -> None:
            with sd.OutputStream(samplerate=self.sr, channels=segment.shape[1]) as stream:
                self._stream = stream
                idx = 0
                block = 1024
                while idx < len(segment) and not self._stop_flag.is_set():
                    if self._paused:
                        time.sleep(0.02)
                        continue
                    end_idx = min(idx + block, len(segment))
                    stream.write(segment[idx:end_idx])
                    idx = end_idx

        self._play_thread = threading.Thread(target=run, daemon=True)
        self._play_thread.start()
