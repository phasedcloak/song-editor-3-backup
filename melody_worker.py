#!/usr/bin/env python3
"""
Melody Extraction Worker - Isolated process for melody extraction
This script runs in a separate process to avoid library conflicts
"""

import sys
import json
import os
import numpy as np
import soundfile as sf
import tempfile

def main():
    # Read parameters from command line argument
    if len(sys.argv) != 2:
        print("Usage: melody_worker.py <params.json>", file=sys.stderr)
        sys.exit(1)

    params_file = sys.argv[1]
    print(f"MELODY_WORKER: Reading params from {params_file}", file=sys.stderr)

    try:
        # Load parameters from file
        with open(params_file, 'r') as f:
            params = json.load(f)

        # Set environment to minimize conflicts
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        # Load audio
        print(f"MELODY_WORKER: Loading audio from {params['audio_path']}", file=sys.stderr)
        audio, sample_rate = sf.read(params['audio_path'])

        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)

        notes = []

        print(f"MELODY_WORKER: Method selection - requested: '{params['method']}'", file=sys.stderr, flush=True)

        if params['method'] == 'basic_pitch':
            print("MELODY_WORKER: Using Basic Pitch for melody extraction", file=sys.stderr)
            print(f"MELODY_WORKER: Requested method: {params['method']}", flush=True)
            try:
                notes = extract_basic_pitch(audio, sample_rate, params)
            except Exception as e:
                print(f"MELODY_WORKER: Basic Pitch failed ({e}), trying CREPE as fallback", file=sys.stderr)
                try:
                    notes = extract_crepe(audio, sample_rate, params)
                except Exception as crepe_error:
                    print(f"MELODY_WORKER: CREPE also failed ({crepe_error}), trying librosa fallback", file=sys.stderr)
                    try:
                        notes = extract_librosa_fallback(audio, sample_rate, params)
                    except Exception as librosa_error:
                        print(f"MELODY_WORKER: Librosa fallback also failed ({librosa_error})", file=sys.stderr)
                        notes = []
        elif params['method'] == 'crepe':
            print("MELODY_WORKER: Using CREPE for melody extraction", file=sys.stderr)
            try:
                notes = extract_crepe(audio, sample_rate, params)
            except Exception as e:
                print(f"MELODY_WORKER: CREPE failed ({e}), trying Basic Pitch as fallback", file=sys.stderr)
                try:
                    notes = extract_basic_pitch(audio, sample_rate, params)
                except Exception as bp_error:
                    print(f"MELODY_WORKER: Basic Pitch also failed ({bp_error}), trying librosa fallback", file=sys.stderr)
                    try:
                        notes = extract_librosa_fallback(audio, sample_rate, params)
                    except Exception as librosa_error:
                        print(f"MELODY_WORKER: Librosa fallback also failed ({librosa_error})", file=sys.stderr)
                        notes = []
        else:
            # Default to librosa fallback
            print(f"MELODY_WORKER: Using librosa fallback for melody extraction (method: '{params['method']}')", file=sys.stderr, flush=True)
            try:
                notes = extract_librosa_fallback(audio, sample_rate, params)
            except Exception as e:
                print(f"MELODY_WORKER: Librosa fallback failed ({e})", file=sys.stderr)
                notes = []

        print(f"MELODY_WORKER: Extracted {len(notes)} notes", file=sys.stderr)
        print(json.dumps(notes))

    except Exception as e:
        print(f"MELODY_ERROR: {e}", file=sys.stderr)
        sys.exit(1)

def extract_basic_pitch(audio, sample_rate, params):
    """Extract melody using Basic Pitch in isolated process."""
    print(f"MELODY_WORKER: Basic Pitch - starting function", file=sys.stderr, flush=True)
    try:
        print(f"MELODY_WORKER: Basic Pitch - importing", file=sys.stderr, flush=True)
        # Import Basic Pitch (this can cause bus errors on macOS)
        from basic_pitch.inference import predict
        print(f"MELODY_WORKER: Basic Pitch - import successful", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"MELODY_WORKER: Failed to import Basic Pitch: {e}", file=sys.stderr, flush=True)
        return []

    # Save audio to temporary file
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name

        print(f"MELODY_WORKER: Basic Pitch - saving audio to {temp_path}", file=sys.stderr, flush=True)
        # Save audio
        sf.write(temp_path, audio.astype(np.float32), sample_rate, subtype='PCM_16')
        print(f"MELODY_WORKER: Basic Pitch - audio saved, running predict", file=sys.stderr, flush=True)

        # Suppress Basic Pitch verbose output
        import logging as python_logging
        original_level = python_logging.getLogger().level
        python_logging.getLogger().setLevel(python_logging.WARNING)

        # Run Basic Pitch inference
        model_output, midi_data, note_events = predict(temp_path)
        print(f"MELODY_WORKER: Basic Pitch - predict completed, found {len(note_events)} note events", file=sys.stderr, flush=True)

        # Restore logging level
        python_logging.getLogger().setLevel(original_level)

        # Process note events
        notes = []
        for note_event in note_events:
            start_time = note_event[0]
            end_time = note_event[1]
            pitch_midi = int(note_event[2])
            amplitude = note_event[3]

            # Filter by pitch range
            if params['min_pitch'] <= pitch_midi <= params['max_pitch']:
                duration = end_time - start_time

                # Filter by minimum duration
                if duration >= params['min_note_duration']:
                    # Calculate confidence from amplitude
                    confidence = min(1.0, amplitude / 0.5)

                    if confidence >= params['min_confidence']:
                        # Convert MIDI to note name
                        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                        note = pitch_midi % 12
                        octave = (pitch_midi // 12) - 1
                        pitch_name = f"{note_names[note]}{octave}"

                        note_data = {
                            'pitch_midi': pitch_midi,
                            'pitch_name': pitch_name,
                            'start': start_time,
                            'end': end_time,
                            'duration': duration,
                            'velocity': int(amplitude * 127),
                            'confidence': confidence,
                            'detection_method': 'basic_pitch'
                        }
                        notes.append(note_data)

        print(f"MELODY_WORKER: Basic Pitch - returning {len(notes)} notes", file=sys.stderr, flush=True)
        return notes

    except Exception as e:
        print(f"MELODY_WORKER: Basic Pitch error: {e}", file=sys.stderr, flush=True)
        return []
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"MELODY_WORKER: Failed to cleanup temp file: {e}", file=sys.stderr)

def extract_crepe(audio, sample_rate, params):
    """Extract melody using CREPE in isolated process."""
    try:
        # Import CREPE (this can also cause issues on macOS)
        import crepe
    except Exception as e:
        print(f"MELODY_WORKER: Failed to import CREPE: {e}", file=sys.stderr)
        return []

        # Process with CREPE
        time, frequency, confidence, _ = crepe.predict(
            audio,
            sample_rate,
            model_capacity='large',
            viterbi=True
        )

        # Filter low confidence
        frequency[confidence < params['min_confidence']] = 0

        # Convert to MIDI notes
        midi_notes = np.zeros_like(frequency)
        mask = frequency > 0
        midi_notes[mask] = 12 * (np.log2(frequency[mask] / 440.0)) + 69
        midi_notes = np.round(midi_notes).astype(int)

        # Filter by pitch range
        pitch_mask = (midi_notes >= params['min_pitch']) & (midi_notes <= params['max_pitch'])
        midi_notes[~pitch_mask] = 0

        # Convert to note events
        notes = []
        current_note = None

        for i, (t, midi_note, conf) in enumerate(zip(time, midi_notes, confidence)):
            if midi_note > 0 and conf >= params['min_confidence']:
                if current_note is None:
                    # Start new note
                    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    note = midi_note % 12
                    octave = (midi_note // 12) - 1
                    pitch_name = f"{note_names[note]}{octave}"

                    current_note = {
                        'pitch_midi': midi_note,
                        'pitch_name': pitch_name,
                        'start': t,
                        'confidence': conf
                    }
                elif midi_note != current_note['pitch_midi']:
                    # End current note and start new one
                    if current_note is not None:
                        current_note['end'] = t
                        current_note['duration'] = t - current_note['start']

                        if current_note['duration'] >= params['min_note_duration']:
                            current_note['velocity'] = int(current_note['confidence'] * 127)
                            current_note['detection_method'] = 'crepe'
                            notes.append(current_note)

                    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    note = midi_note % 12
                    octave = (midi_note // 12) - 1
                    pitch_name = f"{note_names[note]}{octave}"

                    current_note = {
                        'pitch_midi': midi_note,
                        'pitch_name': pitch_name,
                        'start': t,
                        'confidence': conf
                    }
            else:
                # End current note
                if current_note is not None:
                    current_note['end'] = t
                    current_note['duration'] = t - current_note['start']

                    if current_note['duration'] >= params['min_note_duration']:
                        current_note['velocity'] = int(current_note['confidence'] * 127)
                        current_note['detection_method'] = 'crepe'
                        notes.append(current_note)

                    current_note = None

        # Handle last note
        if current_note is not None:
            current_note['end'] = time[-1]
            current_note['duration'] = time[-1] - current_note['start']

            if current_note['duration'] >= params['min_note_duration']:
                current_note['velocity'] = int(current_note['confidence'] * 127)
                current_note['detection_method'] = 'crepe'
                notes.append(current_note)

        return notes

    except Exception as e:
        print(f"MELODY_WORKER: CREPE error: {e}", file=sys.stderr)
        return []

def extract_librosa_fallback(audio, sample_rate, params):
    """Extract melody using librosa-based pitch detection (macOS-compatible fallback)."""
    try:
        # Import librosa for pitch detection
        import librosa

        # Use librosa's YIN algorithm for fundamental frequency estimation
        f0 = librosa.yin(audio,
                        fmin=librosa.note_to_hz('C2'),  # C2 is a reasonable low note
                        fmax=librosa.note_to_hz('C7'),  # C7 is a reasonable high note
                        sr=sample_rate,
                        frame_length=2048,
                        win_length=1024,
                        hop_length=256)

        # Convert to MIDI notes
        midi_notes = np.zeros_like(f0)
        mask = f0 > 0
        midi_notes[mask] = 12 * (np.log2(f0[mask] / 440.0)) + 69

        # Filter by pitch range
        pitch_mask = (midi_notes >= params['min_pitch']) & (midi_notes <= params['max_pitch'])
        midi_notes[~pitch_mask] = 0

        # Create time axis
        times = librosa.times_like(f0, sr=sample_rate, hop_length=256)

        # Extract note segments
        notes = []
        current_note = None
        min_duration = params['min_note_duration']
        confidence_threshold = params['min_confidence']

        for i, (time, midi_note) in enumerate(zip(times, midi_notes)):
            if midi_note > 0:
                # Calculate confidence based on stability (simplified)
                confidence = 0.7  # Default confidence for librosa-based detection

                if current_note is None:
                    # Start new note
                    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    note = int(midi_note) % 12
                    octave = (int(midi_note) // 12) - 1
                    pitch_name = f"{note_names[note]}{octave}"

                    current_note = {
                        'pitch_midi': int(midi_note),
                        'pitch_name': pitch_name,
                        'start': time,
                        'confidence': confidence
                    }
                elif abs(midi_note - current_note['pitch_midi']) > 1:  # Allow small variations
                    # End current note and start new one
                    if current_note is not None:
                        current_note['end'] = time
                        current_note['duration'] = time - current_note['start']

                        if current_note['duration'] >= min_duration and current_note['confidence'] >= confidence_threshold:
                            current_note['velocity'] = int(current_note['confidence'] * 127)
                            current_note['detection_method'] = 'librosa_yin'
                            notes.append(current_note)

                    # Start new note
                    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    note = int(midi_note) % 12
                    octave = (int(midi_note) // 12) - 1
                    pitch_name = f"{note_names[note]}{octave}"

                    current_note = {
                        'pitch_midi': int(midi_note),
                        'pitch_name': pitch_name,
                        'start': time,
                        'confidence': confidence
                    }
            else:
                # End current note
                if current_note is not None:
                    current_note['end'] = time
                    current_note['duration'] = time - current_note['start']

                    if current_note['duration'] >= min_duration and current_note['confidence'] >= confidence_threshold:
                        current_note['velocity'] = int(current_note['confidence'] * 127)
                        current_note['detection_method'] = 'librosa_yin'
                        notes.append(current_note)

                    current_note = None

        # Handle last note
        if current_note is not None:
            current_note['end'] = times[-1]
            current_note['duration'] = times[-1] - current_note['start']

            if current_note['duration'] >= min_duration and current_note['confidence'] >= confidence_threshold:
                current_note['velocity'] = int(current_note['confidence'] * 127)
                current_note['detection_method'] = 'librosa_yin'
                notes.append(current_note)

        return notes

    except Exception as e:
        print(f"MELODY_WORKER: Librosa fallback error: {e}", file=sys.stderr)
        return []

if __name__ == '__main__':
    main()
