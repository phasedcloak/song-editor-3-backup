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
            # Prefer running MLX on Apple GPU when available
            try:
                import mlx.core as mx
                if getattr(mx, 'gpu', None) is not None and mx.gpu.is_available():
                    mx.set_default_device(mx.gpu)
                    print(f"TRANSCRIPTION_WORKER: MLX default device -> {mx.default_device()}", file=sys.stderr)
                else:
                    print("TRANSCRIPTION_WORKER: MLX GPU not available; using CPU", file=sys.stderr)
            except Exception as _e:
                print(f"TRANSCRIPTION_WORKER: Failed to set MLX device: {_e}", file=sys.stderr)

            import mlx_whisper
            # Try requested size first (normalize 'turbo' to 'large-v3-turbo'); fallback to tiny
            requested_size = str(params.get('model_size', 'large-v3-turbo')).strip().lower()
            if requested_size in ('turbo', 'v3-turbo', 'large-turbo'):
                requested_size = 'large-v3-turbo'
            # Prefer local bundled models if present
            model_path = f"mlx-community/whisper-{requested_size}"
            try:
                base_dir = os.path.dirname(__file__)
                local_dir = os.path.join(os.path.dirname(base_dir), 'models', f'whisper-{requested_size}')
                if os.path.isdir(local_dir):
                    model_path = local_dir
            except Exception:
                pass
            if getattr(sys, 'frozen', False):
                try:
                    meipass_dir = getattr(sys, '_MEIPASS', None)
                    if meipass_dir:
                        local_dir = os.path.join(meipass_dir, 'models', f'whisper-{requested_size}')
                        if os.path.isdir(local_dir):
                            model_path = local_dir
                except Exception:
                    pass
            print(f"TRANSCRIPTION_WORKER: Using MLX-Whisper model: {model_path}", file=sys.stderr)

            # Redirect stdout to avoid extra output from MLX-Whisper
            import io
            from contextlib import redirect_stdout

            with redirect_stdout(io.StringIO()) as captured_output:
                try:
                    result = mlx_whisper.transcribe(
                        params['audio_path'],
                        path_or_hf_repo=model_path,
                        language=params['language'],
                        word_timestamps=True,
                        initial_prompt=params['prompt'],
                        verbose=False
                    )
                except Exception as _me:
                    # Fallback to large-v2 if requested size fails
                    fallback_path = "mlx-community/whisper-large-v2"
                    print(f"TRANSCRIPTION_WORKER: Requested MLX model failed ({_me}); falling back to {fallback_path}", file=sys.stderr)
                    result = mlx_whisper.transcribe(
                        params['audio_path'],
                        path_or_hf_repo=fallback_path,
                        language=params['language'],
                        word_timestamps=True,
                        initial_prompt=params['prompt'],
                        verbose=False
                    )
        elif params['model_type'] == 'faster-whisper':
            from faster_whisper import WhisperModel
            # Allow GPU/Metal if available; otherwise fallback to CPU
            # Note: faster-whisper uses CTranslate2; on Apple Silicon recent versions can use 'metal'
            # but 'auto' will choose best available backend.
            model = WhisperModel(params['model_size'], device='auto', compute_type='float32')
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

        # Process results with guard against trailing repetition/zero-length words
        words = []
        last_text = None
        last_end = None
        for segment in result.get('segments', []):
            for word_info in segment.get('words', []):
                text = word_info['word'].strip()
                start = float(word_info['start'])
                end = float(word_info['end'])
                conf = float(word_info.get('probability', 0.8))

                # Skip zero/negative duration
                if end <= start:
                    continue
                # Skip repeated short tail loops (same text and time)
                if last_text == text and last_end is not None and abs(end - last_end) < 1e-3:
                    continue

                last_text = text
                last_end = end

                word = {
                    'text': text,
                    'start': start,
                    'end': end,
                    'confidence': conf,
                    'alternatives': []
                }
                if conf >= params['confidence_threshold']:
                    words.append(word)

        # Print only the JSON output (no extra text)
        print(json.dumps(words), flush=True)

    except Exception as e:
        print(f"TRANSCRIPTION_ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
