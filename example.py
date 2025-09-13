# Step 1: Run Basic Pitch on audio file to get MIDI + note events
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

audio_path = "your_audio.wav"
model_output, midi_data, note_events = predict(audio_path, model_path=ICASSP_2022_MODEL_PATH)

# Step 2: Group note events into segments (e.g., by every X ms or beats)
# Note events are dictionaries with keys: "start", "end", "pitch", "velocity"
import numpy as np
from pychord import find_chords_from_notes

segment_length = 1.0  # seconds per segment
max_time = max(event["end"] for event in note_events)
num_segments = int(np.ceil(max_time / segment_length))

for i in range(num_segments):
    t_start = i * segment_length
    t_end = (i + 1) * segment_length
    notes_this_segment = [event for event in note_events if event["start"] < t_end and event["end"] > t_start]
    midi_pitches = list({event["pitch"] for event in notes_this_segment})
    # Convert MIDI to note names (C, C#, D, etc.)
    note_map = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_names = [note_map[n % 12] for n in midi_pitches]
    # Step 3: Use pychord to label the chord(s)
    if note_names:
        chords = find_chords_from_notes(note_names)
        print(f"Segment {i}: Notes {note_names} -> Chord(s): {[str(chord) for chord in chords]}")
    else:
        print(f"Segment {i}: No notes detected.")

# To install:
# pip install basic-pitch pychord numpy
