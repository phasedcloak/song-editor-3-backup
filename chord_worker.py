#!/usr/bin/env python3
"""
Chord Detection Worker

A standalone script to run chord detection in an isolated process.
This prevents C++ library conflicts and threading issues within the main application.
"""

import sys
import json
import logging
import numpy as np
import librosa
import vamp

def detect_chords(audio_path, params):
    """Load audio and run chord detection."""
    try:
        logging.info(f"CHORD_WORKER: Loading audio from {audio_path}")
        audio, sr = librosa.load(audio_path, sr=44100, mono=True)
        
        logging.info("CHORD_WORKER: Running Chordino VAMP plugin...")
        
        # Correctly process the generator returned by vamp.process_audio
        result_generator = vamp.process_audio(audio, sr, 'nnls-chroma:chordino')

        chords = []
        # The generator yields dictionaries, one per processing step.
        # Chord events are in the value associated with key 0.
        for step_result in result_generator:
            if 0 in step_result:
                for event in step_result[0]:
                    label = event.get('label')
                    if label and label != 'N':
                        chords.append({
                            'timestamp': float(event.get('timestamp', 0.0)),
                            'chord': label
                        })
        
        logging.info(f"CHORD_WORKER: Detected {len(chords)} chord events.")
        return chords

    except Exception as e:
        logging.error(f"CHORD_WORKER: Error during chord detection: {e}", exc_info=True)
        return []

def main():
    """Main entry point for the worker script."""
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    if len(sys.argv) != 2:
        print("Usage: python chord_worker.py <params_file>", file=sys.stderr)
        sys.exit(1)

    params_file = sys.argv[1]

    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        audio_path = params.get('audio_path')
        if not audio_path:
            raise ValueError("audio_path not provided in params file.")

        results = detect_chords(audio_path, params)
        
        # Output results as JSON to stdout
        print(json.dumps(results))

    except Exception as e:
        logging.error(f"CHORD_WORKER: Fatal error: {e}", exc_info=True)
        # Print an empty JSON array on error to avoid breaking the main app's parser
        print(json.dumps([]))
        sys.exit(1)

if __name__ == "__main__":
    main()
