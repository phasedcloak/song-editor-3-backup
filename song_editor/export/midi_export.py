from __future__ import annotations

from typing import List, Optional

import mido

from ..models.lyrics import WordRow
from ..services.gemini_client import AltNoteTimed
from ..processing.chords import DetectedChord


def seconds_to_ticks(seconds: float, tempo: int, ticks_per_beat: int) -> int:
    beats = (seconds * 1_000_000) / tempo
    return int(beats * ticks_per_beat)


def export_midi(
    path: str,
    words: List[WordRow],
    chords: Optional[List[DetectedChord]] = None,
    melody: Optional[List[AltNoteTimed]] = None
) -> None:
    mid = mido.MidiFile()
    mid.ticks_per_beat = 480

    # Track 0: tempo
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    tempo = mido.bpm2tempo(120)
    meta.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    # Track 1: lyrics
    lyr = mido.MidiTrack()
    mid.tracks.append(lyr)
    last_tick = 0
    for w in words:
        start_ticks = seconds_to_ticks(w.start, tempo, mid.ticks_per_beat)
        delta = max(0, start_ticks - last_tick)
        lyr.append(mido.MetaMessage("lyrics", text=w.text, time=delta))
        last_tick = start_ticks

    # Track 2: chords as block triads
    if chords:
        ch_track = mido.MidiTrack()
        mid.tracks.append(ch_track)
        root_map = {
            "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63,
            "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68, "Ab": 68,
            "A": 69, "A#": 70, "Bb": 70, "B": 71,
        }
        last_tick = 0
        for ch in chords:
            name = ch.name
            # derive root token (best-effort) for mapping; fall back to C
            root_token = name.split("/")[0]
            # strip quality extensions from root token
            root_only = ''.join([c for c in root_token if c.isalpha() or c == '#'])
            if len(root_only) >= 2 and root_only[1] == 'b':
                root_only = root_only[:2]
            else:
                root_only = root_only[:2] if len(root_only) > 1 and root_only[1] == '#' else root_only[:1]
            root_key = root_map.get(root_only, 60)
            # simple major triad (placeholder; could parse quality for min/7/etc.)
            notes = [root_key, root_key + 4, root_key + 7]
            start_ticks = seconds_to_ticks(ch.start, tempo, mid.ticks_per_beat)
            end_ticks = seconds_to_ticks(ch.end, tempo, mid.ticks_per_beat)
            delta = max(0, start_ticks - last_tick)
            for n in notes:
                ch_track.append(mido.Message("note_on", note=n, velocity=64, time=delta))
                delta = 0
            duration = max(1, end_ticks - start_ticks)
            for i, n in enumerate(notes):
                ch_track.append(mido.Message("note_off", note=n, velocity=0, time=duration if i == 0 else 0))
            last_tick = end_ticks

    # Track 3: melody (if provided)
    if melody:
        mel = mido.MidiTrack()
        mid.tracks.append(mel)
        last_tick = 0
        for n in melody:
            start_ticks = seconds_to_ticks(n.start, tempo, mid.ticks_per_beat)
            end_ticks = seconds_to_ticks(n.end, tempo, mid.ticks_per_beat)
            delta = max(0, start_ticks - last_tick)
            mel.append(mido.Message("note_on", note=max(0, min(127, n.pitch_midi)), velocity=70, time=delta))
            duration = max(1, end_ticks - start_ticks)
            mel.append(mido.Message("note_off", note=max(0, min(127, n.pitch_midi)), velocity=0, time=duration))
            last_tick = end_ticks

    mid.save(path)
