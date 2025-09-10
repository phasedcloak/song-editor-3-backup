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
    ("Save Separated Sources", "08:36:42", "08:36:59", 17.0),  # File writing time
    ("Lyrics Transcription", "08:36:59", "08:38:02", 63.0),
    ("Chord Detection", "08:38:02", "08:38:05", 3.0),
    ("Melody Extraction", "08:38:05", "08:38:14", 9.0),
    ("Results Export", "08:38:14", "08:38:14", 0.0),
]

total_time = sum(step[3] for step in steps)

print("🎵 Song Editor 3 Processing Timeline")
print("=" * 70)
print("<15")
print("-" * 70)

for step, start, end, duration in steps:
    percentage = (duration / total_time * 100) if total_time > 0 else 0
    print("<15")

print("-" * 70)
print("<15")
print("=" * 70)

print("\n📊 Processing Breakdown:")
print(".1f")
print(".1f")
print(".1f")
print(".1f")
print(".1f")
print(".1f")
print(".1f")
print(".1f")

print("\n🔧 Performance Insights:")
print(f"• Most time-intensive step: Lyrics Transcription ({63.0/total_time*100:.1f}%)")
print(f"• Source separation took {29.12/total_time*100:.1f}% of total time")
print(f"• Audio preprocessing (load + denoise + normalize): {(2.0+2.41)/total_time*100:.1f}%")
print(f"• Analysis (chords + melody): {(3.0+9.0)/total_time*100:.1f}%")
