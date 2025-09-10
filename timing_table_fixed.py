#!/usr/bin/env python3
"""
Generate timing table for Song Editor 3 processing run
"""

# Timing data from the terminal output
steps = [
    ("Audio Loading", "08:36:08", "08:36:10", 2.0),
    ("Audio Denoising", "08:36:10", "08:36:13", 2.41),
    ("Audio Normalization", "08:36:13", "08:36:13", 0.0),
    ("Source Separation (Demucs)", "08:36:14", "08:36:42", 29.12),
    ("Save Separated Sources", "08:36:42", "08:36:59", 17.0),
    ("Lyrics Transcription", "08:36:59", "08:38:02", 63.0),
    ("Chord Detection", "08:38:02", "08:38:05", 3.0),
    ("Melody Extraction", "08:38:05", "08:38:14", 9.0),
    ("Results Export", "08:38:14", "08:38:14", 0.0),
]

total_time = sum(step[3] for step in steps)

print("🎵 Song Editor 3 Processing Timeline")
print("=" * 80)
print("<20")
print("-" * 80)

for i, (step, start, end, duration) in enumerate(steps, 1):
    percentage = (duration / total_time * 100) if total_time > 0 else 0
    print("<20")

print("-" * 80)
print("<20")
print("=" * 80)

print("\n📊 Processing Breakdown:")
print("• Total processing time: 125.53 seconds")
print("• Audio file duration: 303.13 seconds")
print("• Processing ratio: 2.4x real-time")

print("\n🔧 Performance Insights:")
print("• Most time-intensive step: Lyrics Transcription (50.2% of total time)")
print("• Source separation took 23.2% of total time") 
print("• Audio preprocessing (load + denoise + normalize): 3.5%")
print("• Analysis (chords + melody): 9.6%")
print("• File I/O operations: 13.5%")

print("\n⚡ Efficiency Notes:")
print("• MLX Whisper processed 303s of audio in 63s (4.8x real-time)")
print("• Demucs source separation: 29.12s for 303s audio (~10.4x real-time)")
print("• Basic Pitch melody extraction: 9s for 303s audio (~33.7x real-time)")
