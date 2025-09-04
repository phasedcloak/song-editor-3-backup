"""
UI package for Song Editor 3.

This package contains the user interface components.
"""

from .main_window import MainWindow, ProcessingThread
from .lyrics_editor import LyricsEditor
from .chord_editor import ChordEditor
from .melody_editor import MelodyEditor, MelodyVisualizationWidget

__all__ = [
    "MainWindow",
    "ProcessingThread",
    "LyricsEditor",
    "ChordEditor",
    "MelodyEditor",
    "MelodyVisualizationWidget"
]
