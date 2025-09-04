#!/usr/bin/env python3
"""
Debug Resource Usage During OpenAI Whisper Hang

This script monitors CPU, memory, and I/O during transcription to identify what's blocking
"""

import os
import sys
import time
import signal
import psutil
import threading
from datetime import datetime

# Fix OpenMP library conflicts on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class ResourceMonitor:
    """Monitor system resources during transcription"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 Resource monitoring started in background...")
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        print("🔍 Resource monitoring stopped")
        
    def _monitor_loop(self):
        """Monitor loop that runs in background"""
        while self.monitoring:
            try:
                # Get current process info
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # Get system-wide info
                system_cpu = psutil.cpu_percent(interval=0.1)
                system_memory = psutil.virtual_memory()
                
                # Get I/O info
                io_counters = self.process.io_counters()
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] CPU: {cpu_percent:5.1f}% | "
                      f"Memory: {memory_mb:7.1f}MB | "
                      f"Sys CPU: {system_cpu:5.1f}% | "
                      f"Sys Mem: {system_memory.percent:3.0f}% | "
                      f"IO Read: {io_counters.read_bytes/1024/1024:6.1f}MB | "
                      f"IO Write: {io_counters.write_bytes/1024/1024:6.1f}MB")
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                time.sleep(1)

def test_with_resource_monitoring():
    """Test OpenAI Whisper with resource monitoring"""
    
    print("=== OpenAI Whisper Resource Usage Debug ===")
    print("This will monitor resources during transcription to identify what's blocking")
    print()
    
    try:
        # Import whisper
        print("1. Importing OpenAI Whisper...")
        import whisper
        print("✅ OpenAI Whisper imported successfully")
        
        # Load model
        print("\n2. Loading model...")
        start_time = time.time()
        model = whisper.load_model("tiny")  # Use tiny for faster testing
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Load audio file
        print("\n3. Loading audio file...")
        audio_file = "25-03-12 we see your love - 02.wav"
        
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return False
        
        print(f"✅ Audio file: {audio_file}")
        
        # Start resource monitoring
        print("\n4. Starting resource monitoring...")
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        # Wait a moment for monitoring to start
        time.sleep(2)
        
        # Test transcription with monitoring
        print("\n5. Testing transcription with word timestamps...")
        print("   Watch the resource usage above - if it hangs, we'll see what's happening")
        print("   Press Ctrl+C to stop if it hangs")
        print()
        
        try:
            transcribe_start = time.time()
            
            # This is where it hangs
            result = model.transcribe(
                audio_file,
                language="en",
                word_timestamps=True,
                beam_size=5,
                temperature=0.0
            )
            
            transcribe_time = time.time() - transcribe_start
            print(f"\n✅ Transcription completed in {transcribe_time:.2f}s")
            
            if 'segments' in result and result['segments']:
                word_count = sum(len(seg.get('words', [])) for seg in result['segments'])
                print(f"   Found {word_count} words with timestamps")
            
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️  Transcription interrupted by user")
            print("   This confirms it was hanging")
            return False
        except Exception as e:
            print(f"\n❌ Transcription failed: {e}")
            return False
        finally:
            monitor.stop_monitoring()
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_with_resource_monitoring()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
