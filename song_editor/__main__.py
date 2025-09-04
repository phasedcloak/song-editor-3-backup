#!/usr/bin/env python3
"""
Song Editor 3 - Package Entry Point

This allows the package to be run with `python -m song_editor`
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path for proper imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import and run the main application
from song_editor.app import main

if __name__ == "__main__":
    sys.exit(main())
