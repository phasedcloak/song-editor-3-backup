from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import librosa


ROOTS = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]


@dataclass
class DetectedChord:
    name: str  # e.g., C, Cm, D#, D#m
    start: float
    end: float
    confidence: float


def _build_templates() -> list[tuple[str, np.ndarray]]:
    major = np.zeros(12)
    major[[0, 4, 7]] = [1.0, 0.9, 0.9]
    minor = np.zeros(12)
    minor[[0, 3, 7]] = [1.0, 0.9, 0.9]
    templates: list[tuple[str, np.ndarray]] = []
    for i, r in enumerate(ROOTS):
        maj = np.roll(major, i)
        minr = np.roll(minor, i)
        templates.append((f"{r}", maj))
        templates.append((f"{r}m", minr))
    return templates


class ChordDetector:
    def __init__(self) -> None:
        self.templates = _build_templates()

    def detect(self, audio_path: str) -> List[DetectedChord]:
        y, sr = librosa.load(audio_path, mono=True)
        hop_length = 2048
        chromagram = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        # normalize per-frame
        chromagram += 1e-6
        chromagram = chromagram / np.maximum(np.sum(chromagram, axis=0, keepdims=True), 1e-6)
        times = librosa.times_like(chromagram, sr=sr, hop_length=hop_length)

        best_labels: list[str] = []
        best_scores: list[float] = []
        for i in range(chromagram.shape[1]):
            v = chromagram[:, i]
            label = "N"
            score = 0.0
            for name, tmpl in self.templates:
                val = float(np.dot(v, tmpl))
                if val > score:
                    score = val
                    label = name
            best_labels.append(label)
            best_scores.append(score)

        # median filter smoothing over 7 frames
        win = 7
        pad = win // 2
        labels_sm = best_labels[:]
        for i in range(len(best_labels)):
            lo = max(0, i - pad)
            hi = min(len(best_labels), i + pad + 1)
            window = best_labels[lo:hi]
            labels_sm[i] = max(set(window), key=window.count)

        # segment labels to chords
        chords: List[DetectedChord] = []
        current = None
        start_time = float(times[0]) if len(times) else 0.0
        max_conf = 0.0
        for i, lab in enumerate(labels_sm):
            t = float(times[i])
            conf = best_scores[i]
            if current is None:
                current = lab
                start_time = t
                max_conf = conf
                continue
            if lab != current:
                chords.append(DetectedChord(current, start_time, t, max_conf))
                current = lab
                start_time = t
                max_conf = conf
            else:
                max_conf = max(max_conf, conf)
        # tail
        if current is not None:
            end_time = float(times[-1]) if len(times) else start_time
            chords.append(DetectedChord(current, start_time, end_time, max_conf))

        # merge very short chords under 250ms
        merged: List[DetectedChord] = []
        for ch in chords:
            if not merged:
                merged.append(ch)
                continue
            if ch.end - ch.start < 0.25 and merged[-1].name == ch.name:
                prev = merged[-1]
                merged[-1] = DetectedChord(prev.name, prev.start, ch.end, max(prev.confidence, ch.confidence))
            else:
                merged.append(ch)
        return merged
