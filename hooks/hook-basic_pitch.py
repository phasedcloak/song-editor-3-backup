"""
PyInstaller hook for Basic Pitch

Ensures Basic Pitch model files are properly included.
"""

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect Basic Pitch submodules
hiddenimports = collect_submodules('basic_pitch')

# Add specific Basic Pitch modules
hiddenimports += [
    'basic_pitch.predict',
    'basic_pitch.note_creation',
    'basic_pitch.midi',
    'basic_pitch.algorithms',
]

# Collect data files
datas = []

# Include Basic Pitch saved models
basic_pitch_paths = [
    '/opt/homebrew/anaconda3/lib/python3.10/site-packages/basic_pitch/saved_models',
]

for bp_path in basic_pitch_paths:
    if os.path.exists(bp_path):
        datas.append((bp_path, 'basic_pitch/saved_models'))

# Include any cached Basic Pitch models
import os
home_dir = os.path.expanduser('~')
cache_dirs = [
    os.path.join(home_dir, '.cache', 'basic_pitch'),
]

for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        datas.append((cache_dir, 'cache/basic_pitch'))
