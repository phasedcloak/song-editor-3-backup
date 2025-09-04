# OpenAI Whisper Debugging Project

## Problem Description

OpenAI Whisper is hanging during word-level transcription (`word_timestamps=True`) on macOS. The process becomes unresponsive with:
- **No CPU usage** (process goes quiet)
- **No memory growth** (stuck at same memory level)
- **No I/O activity** (not reading/writing)
- **Process appears frozen** but doesn't crash

## Key Findings

1. **Basic transcription works fine** - completes in ~6.7 seconds for 5+ minute audio
2. **Word-level transcription hangs** - process becomes unresponsive
3. **OpenMP conflicts resolved** - no more crashes, just hanging
4. **Working implementation exists** - wav_to_karaoke project has working word-level transcription

## Test Files

### 1. `standalone_openai_whisper_test.py`
- **Purpose**: Basic OpenAI Whisper test (crashes with OpenMP error)
- **Status**: ❌ Crashes due to OpenMP library conflicts
- **Use Case**: Baseline test to show the crash issue

### 2. `standalone_openai_whisper_test_fixed.py`
- **Purpose**: OpenAI Whisper test with OpenMP fix
- **Status**: ⚠️ No crashes, but hangs on word-level transcription
- **Use Case**: Shows the hanging issue after fixing OpenMP conflicts

### 3. `test_wav_to_karaoke_exact_copy.py`
- **Purpose**: Exact copy of working wav_to_karaoke implementation
- **Status**: ⚠️ Still hangs (suggesting issue is not with parameters)
- **Use Case**: Proves the problem exists even with exact working code

### 4. `debug_resource_usage.py`
- **Purpose**: Monitor system resources during hang
- **Status**: 🔍 Diagnostic tool to identify what's blocking
- **Use Case**: Real-time monitoring of CPU, memory, and I/O during hang

### 5. `WHISPER_COMPARISON_ANALYSIS.md`
- **Purpose**: Detailed analysis of differences between implementations
- **Status**: 📋 Documentation of findings
- **Use Case**: Reference for understanding the problem

## Audio File

- **File**: `25-03-12 we see your love - 02.wav`
- **Duration**: 303 seconds (5+ minutes)
- **Content**: Worship song with clear vocals
- **Format**: 48kHz, stereo, float64

## Environment

- **OS**: macOS (darwin 24.6.0)
- **Python**: 3.10 (virtual environment)
- **OpenAI Whisper**: 20231117
- **Hardware**: Apple Silicon (no GPU acceleration)

## How to Reproduce

1. **Setup environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Run basic test** (works):
   ```bash
   python standalone_openai_whisper_test_fixed.py
   # This will hang on step 5 (word-level transcription)
   ```

3. **Run resource monitoring** (recommended):
   ```bash
   python debug_resource_usage.py
   # This will show real-time resource usage during hang
   ```

## Expected Behavior

- **Steps 1-4**: Should complete successfully
- **Step 5**: Process hangs with no CPU/memory activity
- **Resource monitoring**: Shows process stuck at same resource levels

## Working Implementation

The `wav_to_karaoke` project has a working OpenAI Whisper implementation that successfully produces word-level timestamps. Key differences:

1. **Uses WhisperX as primary method**
2. **Different model loading strategy**
3. **Simpler parameter handling**
4. **GPU acceleration support**

## Debugging Approach

1. **Resource monitoring** - identify what's blocking
2. **Parameter comparison** - find differences from working version
3. **System call tracing** - see where process gets stuck
4. **Library version comparison** - check for version differences

## Next Steps

1. **Run resource monitoring** to see exactly what's happening
2. **Compare with working wav_to_karaoke implementation**
3. **Check for system resource limits** on macOS
4. **Test on different hardware** (Intel Mac, Linux, Windows)

## Contact

This is a debugging project to identify why OpenAI Whisper hangs during word-level transcription on macOS. The goal is to make word-level transcription work reliably like it does in the wav_to_karaoke project.

## Files to Upload to GitHub

- All test files listed above
- Audio file (`25-03-12 we see your love - 02.wav`)
- This README
- Any output logs from failed runs
