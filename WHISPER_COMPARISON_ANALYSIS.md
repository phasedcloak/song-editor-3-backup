# Whisper Transcription Comparison Analysis

## Working Implementation: wav_to_karoke/audio_to_karaoke.py

### Key Success Factors:

1. **Uses WhisperX as Primary Method** (lines ~2950-3000):
   ```python
   # Transcribe with word-level timestamps
   result = model.transcribe(debug_vocal_file, language="en")
   
   # Align whisper output for better word-level timestamps
   model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
   result = whisperx.align(result["segments"], model_a, metadata, debug_vocal_file, device)
   ```

2. **OpenAI Whisper Fallback** (lines ~2830-2900):
   ```python
   # Transcribe with word-level timestamps and multiple candidates
   result = model.transcribe(
       debug_vocal_file,
       language="en",
       word_timestamps=True,
       beam_size=5,  # Use beam search
       best_of=self.alternatives_count,  # Generate multiple candidates
       temperature=0.0  # Deterministic for consistent alternatives
   )
   ```

3. **Proper Model Loading**:
   - Uses `whisper.load_model("large-v2")` for best accuracy
   - Handles GPU/CPU fallbacks gracefully
   - Uses appropriate compute types (float16/float32)

## Current Song_Editor_3 Implementation Issues:

### 1. **Core Transcriber Problems**:
   - **openai-whisper**: Hangs indefinitely on CPU
   - **faster-whisper**: Works but may have accuracy issues
   - **Complex fallback logic** that can fail

### 2. **Missing WhisperX Support**:
   - No WhisperX implementation (which is the working method in wav_to_karoke)
   - No proper alignment model for word-level timestamps

### 3. **Model Configuration Issues**:
   - Uses "tiny" model instead of "large-v2" for accuracy
   - Missing proper beam search and alternatives generation
   - No temperature control for deterministic results

## Key Differences Summary:

| Aspect | wav_to_karoke (Working) | Song_Editor_3 (Current) |
|--------|-------------------------|-------------------------|
| **Primary Method** | WhisperX with alignment | OpenAI Whisper (hangs) |
| **Fallback** | OpenAI Whisper (working) | Faster Whisper (limited) |
| **Model Size** | large-v2 (best accuracy) | tiny (fast but less accurate) |
| **Word Timestamps** | WhisperX alignment | Direct from model |
| **Beam Search** | beam_size=5 | Not configured |
| **Alternatives** | Multiple candidates | Single result only |
| **Language** | Hardcoded "en" | Configurable (can cause issues) |
| **Error Handling** | Graceful fallbacks | Complex fallback logic |

## Recommended Fixes:

### 1. **Implement WhisperX as Primary Method**:
   ```python
   def _transcribe_whisperx(self, audio: np.ndarray, sample_rate: int):
       # Use the working implementation from wav_to_karoke
       model = whisperx.load_model("large-v2", device="cpu", compute_type="float32")
       result = model.transcribe(temp_path, language="en")
       # Add alignment step
   ```

### 2. **Fix OpenAI Whisper Parameters**:
   ```python
   result = self.whisper_model.transcribe(
       temp_path,
       language="en",  # Hardcode like wav_to_karoke
       word_timestamps=True,
       beam_size=5,  # Add beam search
       temperature=0.0  # Make deterministic
   )
   ```

### 3. **Use Working Processing Module**:
   - The `song_editor/processing/transcriber.py` already works
   - Use this as the foundation instead of the broken core module

### 4. **Simplify Model Selection**:
   - Remove complex fallback logic
   - Use WhisperX → OpenAI Whisper → Faster Whisper order
   - Each with proper timeout handling

## Immediate Action Items:

1. **Use the working processing module** for now
2. **Test with actual audio** (not silent test file)
3. **Implement WhisperX** using wav_to_karoke code
4. **Fix OpenAI Whisper parameters** to match working version
5. **Add proper timeout handling** to prevent hanging

## Why wav_to_karoke Works:

1. **WhisperX**: Better word-level timestamp accuracy
2. **Proper Model Size**: large-v2 instead of tiny
3. **Hardcoded Language**: "en" instead of configurable
4. **Simple Fallback**: Direct model calls without complex logic
5. **GPU/CPU Handling**: Proper device and compute type selection
