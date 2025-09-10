#!/usr/bin/env python3
"""
Song Editor 3 - Package Entry Point

This allows the package to be run with `python -m song_editor`
"""

import sys
import os
import warnings
from pathlib import Path

# Suppress pkg_resources deprecation warning (scheduled for removal 2025-11-30)
# Apply filters as early as possible
warnings.filterwarnings('ignore', message='pkg_resources is deprecated', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*', category=UserWarning)

# Add the parent directory to Python path for proper imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import and run the main application
from song_editor.app import main

if __name__ == "__main__":
    # Set PYTHONWARNINGS environment variable to suppress pretty_midi warning
    # This must be done before any imports that might trigger the warning
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:pretty_midi.instrument'
    sys.exit(main())
