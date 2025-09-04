#!/usr/bin/env python3
"""
Standalone OpenAI Whisper Test (Fixed for OpenMP conflicts)

This script tests OpenAI Whisper in isolation to identify the hanging issue.
Run this on another system to debug the transcription problem.
"""

import os
import sys
import time
import signal
import tempfile
import numpy as np
import soundfile as sf

# Fix OpenMP library conflicts on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Operation timed out")

def test_openai_whisper_standalone():
    """Test OpenAI Whisper in complete isolation"""
    
    print("=== Standalone OpenAI Whisper Test (Fixed) ===")
    print("This test isolates OpenAI Whisper to identify hanging issues.")
    print("OpenMP conflict workaround enabled.")
    print()
    
    # Test 1: Basic import and model loading
    print("1. Testing OpenAI Whisper import...")
    try:
        import whisper
        print("✅ OpenAI Whisper imported successfully")
        print(f"   Version: {whisper.__version__ if hasattr(whisper, '__version__') else 'Unknown'}")
    except ImportError as e:
        print(f"❌ OpenAI Whisper import failed: {e}")
        return False
    
    # Test 2: Model loading
    print("\n2. Testing model loading...")
    try:
        start_time = time.time()
        model = whisper.load_model("tiny")  # Start with tiny model
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        print(f"   Model type: {type(model)}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Test 3: Load actual audio file
    print("\n3. Loading actual audio file...")
    try:
        audio_file = "25-03-12 we see your love - 02.wav"
        
        # Check if file exists
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return False
        
        # Load the audio file
        audio, sample_rate = sf.read(audio_file)
        duration = len(audio) / sample_rate
        
        print(f"✅ Audio file loaded: {audio_file}")
        print(f"   Duration: {duration:.2f}s, Sample rate: {sample_rate}Hz")
        print(f"   Audio shape: {audio.shape}, Type: {audio.dtype}")
        
        # Use the original file path for transcription
        temp_path = audio_file
        
    except Exception as e:
        print(f"❌ Audio file loading failed: {e}")
        return False
    
    # Test 4: Basic transcription (no word timestamps)
    print("\n4. Testing basic transcription (no word timestamps)...")
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # 1 minute timeout
        
        start_time = time.time()
        result = model.transcribe(temp_path, language="en")
        signal.alarm(0)  # Cancel timeout
        
        transcribe_time = time.time() - start_time
        print(f"✅ Basic transcription completed in {transcribe_time:.2f}s")
        print(f"   Text: '{result['text']}'")
        print(f"   Language: {result.get('language', 'Unknown')}")
        
    except TimeoutError:
        signal.alarm(0)
        print("❌ Basic transcription timed out after 1 minute")
        print("   This indicates a fundamental hanging issue")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Basic transcription failed: {e}")
        return False
    
    # Test 5: Word-level transcription (the problematic one)
    print("\n5. Testing word-level transcription (this is where it hangs)...")
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)  # 2 minutes timeout
        
        start_time = time.time()
        result = model.transcribe(
            temp_path,
            language="en",
            word_timestamps=True,
            beam_size=5,
            best_of=5,
            temperature=0.0
        )
        signal.alarm(0)  # Cancel timeout
        
        transcribe_time = time.time() - start_time
        print(f"✅ Word-level transcription completed in {transcribe_time:.2f}s")
        
        if 'segments' in result and result['segments']:
            first_segment = result['segments'][0]
            if 'words' in first_segment and first_segment['words']:
                print(f"   First word: '{first_segment['words'][0]['word']}' at {first_segment['words'][0]['start']:.2f}s")
                print(f"   Total words: {sum(len(seg.get('words', [])) for seg in result['segments'])}")
            else:
                print("   ⚠️  No word timestamps found in result")
        else:
            print("   ⚠️  No segments found in result")
            
    except TimeoutError:
        signal.alarm(0)
        print("❌ Word-level transcription timed out after 2 minutes")
        print("   This confirms the hanging issue is with word-level timestamps")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"❌ Word-level transcription failed: {e}")
        return False
    
    # Cleanup
    print(f"\n✅ Test completed - no cleanup needed (using original file)")
    
    print("\n=== Test Summary ===")
    print("✅ All tests passed - OpenAI Whisper is working correctly")
    print("   If you're experiencing hanging issues, the problem is likely:")
    print("   1. Audio file format/complexity")
    print("   2. System resources (CPU/memory)")
    print("   3. Model size (larger models are much slower)")
    print("   4. Missing GPU acceleration")
    
    return True

if __name__ == "__main__":
    try:
        success = test_openai_whisper_standalone()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
