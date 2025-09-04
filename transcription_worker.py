#!/usr/bin/env python3
"""
Transcription Worker - Isolated process for Whisper transcription
This script runs in a separate process to avoid library conflicts
"""

import sys
import json
import os

def main():
    # Debug: Log that we're starting
    print("TRANSCRIPTION_WORKER: Starting worker script", file=sys.stderr)

    # Read parameters from command line argument
    if len(sys.argv) != 2:
        print("TRANSCRIPTION_WORKER: Usage error - wrong number of arguments", file=sys.stderr)
        print("Usage: transcription_worker.py <params.json>", file=sys.stderr)
        sys.exit(1)

    params_file = sys.argv[1]
    print(f"TRANSCRIPTION_WORKER: Reading params from {params_file}", file=sys.stderr)

    try:
        # Load parameters from file
        with open(params_file, 'r') as f:
            params = json.load(f)

        # Set environment to minimize conflicts
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        if params['model_type'] == 'mlx-whisper':
            import mlx_whisper
            # MLX-Whisper only supports tiny model
            model_size = 'tiny'  # Force tiny model for MLX-Whisper
            model_path = f"mlx-community/whisper-{model_size}"
            print(f"TRANSCRIPTION_WORKER: Using MLX-Whisper model: {model_path}", file=sys.stderr)

            # Redirect stdout to avoid extra output from MLX-Whisper
            import io
            from contextlib import redirect_stdout

            with redirect_stdout(io.StringIO()) as captured_output:
                result = mlx_whisper.transcribe(
                    params['audio_path'],
                    path_or_hf_repo=model_path,
                    language=params['language'],
                    word_timestamps=True,
                    initial_prompt=params['prompt'],
                    verbose=False
                )
        elif params['model_type'] == 'faster-whisper':
            from faster_whisper import WhisperModel
            model = WhisperModel(params['model_size'], device='cpu', compute_type='float32')
            segments, info = model.transcribe(
                params['audio_path'],
                language=params['language'],
                word_timestamps=True,
                beam_size=1,
                initial_prompt=params['prompt']
            )
            # Convert to dict format
            result = {
                'segments': [
                    {
                        'words': [
                            {
                                'word': word.word,
                                'start': word.start,
                                'end': word.end,
                                'probability': word.probability
                            }
                            for word in segment.words
                        ] if segment.words else []
                    }
                    for segment in segments
                ]
            }
        else:
            raise ValueError(f"Unsupported model type: {params['model_type']}")

        # Process results
        words = []
        for segment in result.get('segments', []):
            for word_info in segment.get('words', []):
                word = {
                    'text': word_info['word'].strip(),
                    'start': word_info['start'],
                    'end': word_info['end'],
                    'confidence': word_info.get('probability', 0.8),
                    'alternatives': []
                }
                if word['confidence'] >= params['confidence_threshold']:
                    words.append(word)

        # Print only the JSON output (no extra text)
        print(json.dumps(words), flush=True)

    except Exception as e:
        print(f"TRANSCRIPTION_ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
