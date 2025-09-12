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
        
        # Use subprocess isolation to avoid cffi conflicts
        try:
            import subprocess
            import tempfile
            import os
            
            # Create a subprocess script to load audio
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
            sr = sample_rate
            
            # Clean up temp file
            try:
                os.unlink(audio_data_path)
            except:
                pass
                
        except Exception as e:
            logging.warning(f"Subprocess audio loading failed: {e}, falling back to direct load")
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
        
        logging.info(f"CHORD_WORKER: Detected {len(chords)} chord events (Chordino).")
        if len(chords) > 0:
            return chords

        # Fallback to librosa-based chroma template matching if Chordino returns none
        logging.info("CHORD_WORKER: Chordino returned no chords. Falling back to librosa template detector...")
        try:
            from song_editor.processing.chords import ChordDetector
            cd = ChordDetector()
            detected = cd.detect(audio_path)
            chords_librosa = [
                { 'timestamp': float(ch.start), 'chord': ch.name }
                for ch in detected if ch.name != 'N'
            ]
            logging.info(f"CHORD_WORKER: Librosa fallback detected {len(chords_librosa)} chords.")
            return chords_librosa
        except Exception as fe:
            logging.error(f"CHORD_WORKER: Librosa fallback failed: {fe}")
            return []

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
