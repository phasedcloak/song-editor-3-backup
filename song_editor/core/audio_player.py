from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


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
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        self.audio = data
        self.sr = sr
        self._pos = 0

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
