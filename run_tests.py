#!/usr/bin/env python3
"""
Test Runner for Song Editor 3

This script runs all tests for the Song Editor 3 project.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_pytest_tests(test_paths=None, markers=None, verbose=False):
    """Run pytest tests."""
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if markers:
        cmd.extend(["-m", markers])
    
    if test_paths:
        cmd.extend(test_paths)
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.getcwd())
    return result.returncode == 0

def run_specific_test(test_file):
    """Run a specific test file."""
    print(f"Running specific test: {test_file}")
    result = subprocess.run([sys.executable, test_file], cwd=os.getcwd())
    return result.returncode == 0

def run_integration_tests():
    """Run integration tests."""
    print("Running integration tests...")
    return run_pytest_tests(["tests/"], markers="integration", verbose=True)

def run_unit_tests():
    """Run unit tests."""
    print("Running unit tests...")
    return run_pytest_tests(["tests/"], markers="unit", verbose=True)

def run_gui_tests():
    """Run GUI tests."""
    print("Running GUI tests...")
    return run_pytest_tests(["tests/"], markers="gui", verbose=True)

def run_audio_tests():
    """Run audio processing tests."""
    print("Running audio processing tests...")
    return run_pytest_tests(["tests/"], markers="audio", verbose=True)

def run_whisper_tests():
    """Run Whisper-related tests."""
    print("Running Whisper tests...")
    return run_pytest_tests(["tests/"], markers="whisper", verbose=True)

def run_chordino_tests():
    """Run Chordino-related tests."""
    print("Running Chordino tests...")
    return run_pytest_tests(["tests/"], markers="chordino", verbose=True)

def run_demucs_tests():
    """Run Demucs-related tests."""
    print("Running Demucs tests...")
    return run_pytest_tests(["tests/"], markers="demucs", verbose=True)

def run_fast_tests():
    """Run fast tests (skip slow ones)."""
    print("Running fast tests...")
    return run_pytest_tests(["tests/"], markers="not slow", verbose=True)

def run_coverage():
    """Run tests with coverage."""
    print("Running tests with coverage...")
    cmd = ["python", "-m", "pytest", "--cov=song_editor", "--cov-report=html", "--cov-report=term"]
    result = subprocess.run(cmd, cwd=os.getcwd())
    return result.returncode == 0

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Song Editor 3 tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--gui", action="store_true", help="Run GUI tests only")
    parser.add_argument("--audio", action="store_true", help="Run audio processing tests only")
    parser.add_argument("--whisper", action="store_true", help="Run Whisper tests only")
    parser.add_argument("--chordino", action="store_true", help="Run Chordino tests only")
    parser.add_argument("--demucs", action="store_true", help="Run Demucs tests only")
    parser.add_argument("--fast", action="store_true", help="Run fast tests (skip slow ones)")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    parser.add_argument("--file", type=str, help="Run a specific test file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("song_editor").exists():
        print("Error: song_editor directory not found. Please run this script from the project root.")
        sys.exit(1)
    
    success = True
    
    if args.file:
        success = run_specific_test(args.file)
    elif args.unit:
        success = run_unit_tests()
    elif args.integration:
        success = run_integration_tests()
    elif args.gui:
        success = run_gui_tests()
    elif args.audio:
        success = run_audio_tests()
    elif args.whisper:
        success = run_whisper_tests()
    elif args.chordino:
        success = run_chordino_tests()
    elif args.demucs:
        success = run_demucs_tests()
    elif args.fast:
        success = run_fast_tests()
    elif args.coverage:
        success = run_coverage()
    elif args.all:
        print("Running all tests...")
        success = run_pytest_tests(verbose=args.verbose)
    else:
        # Default: run fast tests
        success = run_fast_tests()
    
    if success:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
