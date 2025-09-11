#!/usr/bin/env python3
"""
Enhanced Timing Table Builder for Song Editor 3
Supports both single file and batch processing analysis
"""

import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import re
from dataclasses import dataclass
import sys


@dataclass
class ProcessingStep:
    """Represents a single processing step with timing information."""
    name: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    
    def __post_init__(self):
        if self.duration is None and self.start_time is not None and self.end_time is not None:
            self.duration = self.end_time - self.start_time


@dataclass
class FileProcessingStats:
    """Processing statistics for a single file."""
    file_path: str
    file_name: str
    audio_duration: float
    total_processing_time: float
    processing_ratio: float  # processing_time / audio_duration
    steps: List[ProcessingStep]
    success: bool = True
    error: Optional[str] = None
    word_count: int = 0
    chord_count: int = 0
    note_count: int = 0


class TimingTableBuilder:
    """Builds comprehensive timing tables for Song Editor 3 processing runs."""
    
    def __init__(self):
        self.file_stats: List[FileProcessingStats] = []
        self.batch_start_time: Optional[float] = None
        self.batch_end_time: Optional[float] = None
    
    def parse_log_output(self, log_text: str) -> List[FileProcessingStats]:
        """Parse timing information from Song Editor 3 log output."""
        files_data = []
        current_file = None
        current_steps = []
        
        lines = log_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Match file processing start
            if "🎵 Processing audio file:" in line:
                # Save previous file if exists
                if current_file and current_steps:
                    files_data.append(self._create_file_stats(current_file, current_steps))
                
                # Extract file path
                match = re.search(r"🎵 Processing audio file: (.+)$", line)
                if match:
                    current_file = match.group(1)
                    current_steps = []
            
            # Parse individual step timings
            elif current_file:
                self._parse_step_timing(line, current_steps)
        
        # Don't forget the last file
        if current_file and current_steps:
            files_data.append(self._create_file_stats(current_file, current_steps))
        
        self.file_stats = files_data
        return files_data
    
    def _parse_step_timing(self, line: str, steps: List[ProcessingStep]):
        """Parse timing information from a log line."""
        
        # Audio loading
        if "Audio loaded:" in line and "samples" in line and "Hz" in line:
            match = re.search(r"Audio loaded: (\d+) samples, (\d+) Hz, ([\d.]+)s", line)
            if match:
                samples, sr, duration = match.groups()
                steps.append(ProcessingStep("Audio Loading", duration=float(duration)))
        
        # Audio processing completed (contains multiple timings)
        elif "Audio processing completed" in line:
            # Example: "Audio processing completed (total 23.08s, demucs 15.12s, write_sep 0.00s, tempo 3.02s, key 3.18s)"
            match = re.search(r"total ([\d.]+)s", line)
            if match:
                total_time = float(match.group(1))
                steps.append(ProcessingStep("Audio Processing Total", duration=total_time))
            
            # Parse individual components
            component_matches = re.findall(r"(\w+) ([\d.]+)s", line)
            for component, duration in component_matches:
                if component != "total":
                    step_name = self._normalize_step_name(component)
                    steps.append(ProcessingStep(step_name, duration=float(duration)))
        
        # Transcription timing
        elif "transcription succeeded:" in line:
            match = re.search(r"(\d+) words in ([\d.]+) seconds", line)
            if match:
                word_count, duration = match.groups()
                steps.append(ProcessingStep("Transcription", duration=float(duration)))
        
        # Chord detection
        elif "Chord detection completed:" in line:
            match = re.search(r"(\d+) chords in ([\d.]+) seconds", line)
            if match:
                chord_count, duration = match.groups()
                steps.append(ProcessingStep("Chord Detection", duration=float(duration)))
        
        # Melody extraction (from worker output)
        elif "Melody extraction succeeded" in line:
            # Look for timing in previous lines or estimate
            steps.append(ProcessingStep("Melody Extraction", duration=2.0))  # Estimate if not found
    
    def _normalize_step_name(self, component: str) -> str:
        """Normalize component names for display."""
        name_map = {
            'demucs': 'Source Separation (Demucs)',
            'write_sep': 'Save Separated Sources',
            'tempo': 'Tempo Detection',
            'key': 'Key Detection',
            'intermed': 'Save Intermediate Files'
        }
        return name_map.get(component, component.replace('_', ' ').title())
    
    def _create_file_stats(self, file_path: str, steps: List[ProcessingStep]) -> FileProcessingStats:
        """Create file statistics from parsed steps."""
        
        # Get audio duration from loading step
        audio_duration = 0.0
        for step in steps:
            if step.name == "Audio Loading" and step.duration:
                audio_duration = step.duration
                break
        
        # Calculate total processing time
        total_processing_time = sum(step.duration for step in steps if step.duration)
        
        # Calculate processing ratio
        processing_ratio = total_processing_time / audio_duration if audio_duration > 0 else 0.0
        
        return FileProcessingStats(
            file_path=file_path,
            file_name=Path(file_path).name,
            audio_duration=audio_duration,
            total_processing_time=total_processing_time,
            processing_ratio=processing_ratio,
            steps=steps
        )
    
    def add_batch_results(self, batch_results: Dict[str, Any]):
        """Add batch processing results to the timing analysis."""
        if 'results' in batch_results:
            for result in batch_results['results']:
                file_path = result.get('file_path', '')
                processing_time = result.get('processing_time', 0.0)
                audio_duration = result.get('audio_duration', 0.0)
                success = result.get('success', False)
                error = result.get('error')
                
                # Calculate processing ratio
                processing_ratio = processing_time / audio_duration if audio_duration > 0 else 0.0
                
                # Create basic file stats for batch results
                file_stats = FileProcessingStats(
                    file_path=file_path,
                    file_name=Path(file_path).name,
                    audio_duration=audio_duration,
                    total_processing_time=processing_time,
                    processing_ratio=processing_ratio,
                    steps=[ProcessingStep("Total Processing", duration=processing_time)],
                    success=success,
                    error=error
                )
                self.file_stats.append(file_stats)
    
    def generate_table(self, include_steps: bool = True) -> str:
        """Generate a comprehensive timing table."""
        if not self.file_stats:
            return "No processing data available."
        
        output = []
        
        # Header
        output.append("🎵 Song Editor 3 Processing Analysis")
        output.append("=" * 80)
        
        if len(self.file_stats) == 1:
            # Single file analysis
            output.extend(self._generate_single_file_table(self.file_stats[0], include_steps))
        else:
            # Batch analysis
            output.extend(self._generate_batch_table(include_steps))
        
        return "\n".join(output)
    
    def _generate_single_file_table(self, stats: FileProcessingStats, include_steps: bool) -> List[str]:
        """Generate table for single file processing."""
        output = []
        
        # File information
        output.append(f"📁 File: {stats.file_name}")
        output.append(f"📂 Path: {stats.file_path}")
        output.append(f"⏱️  Audio Duration: {stats.audio_duration:.2f} seconds ({stats.audio_duration/60:.1f} minutes)")
        output.append(f"🔧 Processing Time: {stats.total_processing_time:.2f} seconds")
        output.append(f"⚡ Processing Ratio: {stats.processing_ratio:.1f}x real-time")
        output.append("-" * 80)
        
        if include_steps and stats.steps:
            # Step-by-step breakdown
            output.append("\n📊 Processing Steps:")
            output.append(f"{'Step':<25} {'Duration (s)':<12} {'% of Total':<12}")
            output.append("-" * 50)
            
            for step in stats.steps:
                if step.duration is not None:
                    percentage = (step.duration / stats.total_processing_time * 100) if stats.total_processing_time > 0 else 0
                    output.append(f"{step.name:<25} {step.duration:<12.2f} {percentage:<12.1f}")
            
            output.append("-" * 50)
            output.append(f"{'TOTAL':<25} {stats.total_processing_time:<12.2f} {'100.0':<12}")
        
        # Performance insights
        output.append("\n🔧 Performance Insights:")
        if stats.steps:
            longest_step = max(stats.steps, key=lambda s: s.duration or 0)
            if longest_step.duration:
                percentage = (longest_step.duration / stats.total_processing_time * 100) if stats.total_processing_time > 0 else 0
                output.append(f"• Most time-intensive step: {longest_step.name} ({percentage:.1f}% of total time)")
        
        output.append(f"• Processing efficiency: {1/stats.processing_ratio:.1f}x faster than real-time" if stats.processing_ratio > 0 else "• Processing efficiency: N/A")
        
        return output
    
    def _generate_batch_table(self, include_steps: bool) -> List[str]:
        """Generate table for batch processing."""
        output = []
        
        # Batch summary
        total_files = len(self.file_stats)
        successful_files = sum(1 for f in self.file_stats if f.success)
        failed_files = total_files - successful_files
        total_audio_duration = sum(f.audio_duration for f in self.file_stats)
        total_processing_time = sum(f.total_processing_time for f in self.file_stats)
        
        output.append(f"📊 Batch Processing Summary")
        output.append("-" * 80)
        output.append(f"📁 Total Files: {total_files}")
        output.append(f"✅ Successful: {successful_files}")
        output.append(f"❌ Failed: {failed_files}")
        output.append(f"⏱️  Total Audio Duration: {total_audio_duration:.1f}s ({total_audio_duration/60:.1f} min)")
        output.append(f"🔧 Total Processing Time: {total_processing_time:.1f}s ({total_processing_time/60:.1f} min)")
        
        if total_audio_duration > 0:
            batch_ratio = total_processing_time / total_audio_duration
            output.append(f"⚡ Batch Processing Ratio: {batch_ratio:.1f}x real-time")
        else:
            batch_ratio = 0.0
            output.append(f"⚡ Batch Processing Ratio: N/A (no audio duration data)")
        
        output.append(f"💨 Average per file: {total_processing_time/total_files:.1f}s processing time")
        
        # Per-file breakdown
        output.append("\n📋 Per-File Breakdown:")
        output.append(f"{'File Name':<30} {'Audio (s)':<10} {'Process (s)':<12} {'Ratio':<8} {'Status':<8}")
        output.append("-" * 80)
        
        for stats in self.file_stats:
            status = "✅ OK" if stats.success else "❌ FAIL"
            ratio_str = f"{stats.processing_ratio:.1f}x" if stats.processing_ratio > 0 else "N/A"
            
            file_name = stats.file_name
            if len(file_name) > 28:
                file_name = file_name[:25] + "..."
            
            output.append(f"{file_name:<30} {stats.audio_duration:<10.1f} {stats.total_processing_time:<12.1f} {ratio_str:<8} {status:<8}")
        
        output.append("-" * 80)
        ratio_str = f"{batch_ratio:.1f}x" if batch_ratio > 0 else "N/A"
        output.append(f"{'TOTAL':<30} {total_audio_duration:<10.1f} {total_processing_time:<12.1f} {ratio_str:<8} {'':8}")
        
        # Failed files details
        if failed_files > 0:
            output.append("\n❌ Failed Files:")
            for stats in self.file_stats:
                if not stats.success:
                    output.append(f"  • {stats.file_name}: {stats.error or 'Unknown error'}")
        
        return output
    
    def export_json(self, output_path: str):
        """Export timing data to JSON format."""
        data = {
            'batch_summary': {
                'total_files': len(self.file_stats),
                'successful_files': sum(1 for f in self.file_stats if f.success),
                'failed_files': sum(1 for f in self.file_stats if not f.success),
                'total_audio_duration': sum(f.audio_duration for f in self.file_stats),
                'total_processing_time': sum(f.total_processing_time for f in self.file_stats)
            },
            'files': []
        }
        
        for stats in self.file_stats:
            file_data = {
                'file_path': stats.file_path,
                'file_name': stats.file_name,
                'audio_duration': stats.audio_duration,
                'total_processing_time': stats.total_processing_time,
                'processing_ratio': stats.processing_ratio,
                'success': stats.success,
                'error': stats.error,
                'word_count': stats.word_count,
                'chord_count': stats.chord_count,
                'note_count': stats.note_count,
                'steps': [
                    {
                        'name': step.name,
                        'duration': step.duration
                    }
                    for step in stats.steps
                ]
            }
            data['files'].append(file_data)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    """CLI interface for timing table builder."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python timing_table_builder.py <log_file>")
        print("  python timing_table_builder.py <batch_results.json>")
        print("  cat processing.log | python timing_table_builder.py -")
        sys.exit(1)
    
    input_source = sys.argv[1]
    
    builder = TimingTableBuilder()
    
    if input_source == "-":
        # Read from stdin
        log_text = sys.stdin.read()
        builder.parse_log_output(log_text)
    elif input_source.endswith('.json'):
        # Read batch results JSON
        with open(input_source, 'r') as f:
            batch_results = json.load(f)
        builder.add_batch_results(batch_results)
    else:
        # Read log file
        with open(input_source, 'r') as f:
            log_text = f.read()
        builder.parse_log_output(log_text)
    
    # Generate and print table
    table = builder.generate_table()
    print(table)
    
    # Export JSON if requested
    if len(sys.argv) > 2:
        json_output = sys.argv[2]
        builder.export_json(json_output)
        print(f"\n📊 Timing data exported to: {json_output}")


if __name__ == "__main__":
    main()

