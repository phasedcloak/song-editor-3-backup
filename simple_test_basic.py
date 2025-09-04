#!/usr/bin/env python3
"""
Basic Test for Song Editor 3

Tests only the core models without problematic dependencies.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_models():
    """Test basic model creation."""
    print("Testing basic models...")
    
    try:
        # Test Word model
        from song_editor.models.song_data import Word
        word = Word(text="test", start=0.0, end=1.0, confidence=0.8)
        print("✓ Word model created")
        
        # Test Chord model
        from song_editor.models.song_data import Chord
        chord = Chord(symbol="C", root="C", quality="major", start=0.0, end=1.0)
        print("✓ Chord model created")
        
        # Test Note model
        from song_editor.models.song_data import Note
        note = Note(pitch_midi=60, start=0.0, end=1.0)
        print("✓ Note model created")
        
        # Test SongData model
        from song_editor.models.song_data import SongData
        song_data = SongData()
        song_data.words.append(word)
        song_data.chords.append(chord)
        song_data.notes.append(note)
        print("✓ SongData model created and populated")
        
        # Test Metadata model
        from song_editor.models.metadata import Metadata
        metadata = Metadata(version="3.0.0")
        print("✓ Metadata model created")
        
        print("✓ All basic models work!")
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_package_structure():
    """Test package structure."""
    print("\nTesting package structure...")
    
    try:
        # Test package import
        import song_editor
        print("✓ song_editor package imported")
        
        # Test models package
        from song_editor import models
        print("✓ models package imported")
        
        # Test core package
        from song_editor import core
        print("✓ core package imported")
        
        # Test export package
        from song_editor import export
        print("✓ export package imported")
        
        # Test UI package
        from song_editor import ui
        print("✓ ui package imported")
        
        print("✓ All packages imported successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Package import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("Song Editor 3 Basic Test")
    print("=" * 40)
    
    # Test package structure
    if not test_package_structure():
        print("\n✗ Package structure tests failed!")
        return False
    
    # Test basic models
    if not test_basic_models():
        print("\n✗ Model tests failed!")
        return False
    
    print("\n" + "=" * 40)
    print("✓ All basic tests passed! Song Editor 3 structure is working.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
