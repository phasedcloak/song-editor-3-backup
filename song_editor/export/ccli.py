from __future__ import annotations

from typing import List

from ..models.lyrics import WordRow


def export_ccli(path: str, words: List[WordRow]) -> None:
    """
    Write ChordPro-like text, injecting chords in square brackets before the word where the chord changes.
    Lines are split on gaps (>0.5s) in the word stream.
    Uses detected chord from each word's mid-point; if missing, no bracket is injected.
    """
    if not words:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    lines: list[list[str]] = []
    current: list[str] = []
    prev_chord: str | None = None
    for i, w in enumerate(words):
        # inject chord when it changes and exists
        if w.chord and w.chord != prev_chord:
            current.append(f"[{w.chord}]")
            prev_chord = w.chord
        current.append(w.text)
        if i + 1 < len(words):
            next_w = words[i + 1]
            if next_w.start - w.end > 0.5:
                lines.append(current)
                current = []
                prev_chord = None
    if current:
        lines.append(current)

    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(" ".join(line).strip() + "\n")
