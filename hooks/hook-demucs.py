"""
PyInstaller hook for Demucs

Ensures Demucs model files and configuration are properly included.
"""

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect Demucs submodules
hiddenimports = collect_submodules('demucs')

# Add specific Demucs modules
hiddenimports += [
    'demucs.api',
    'demucs.separate',
    'demucs.hdemucs',
    'demucs.htdemucs',
    'demucs.pretrained',
    'demucs.remote',
    'demucs.audio',
    'demucs.utils',
    'demucs.wdemucs',
    'demucs.demucs',
]

# Collect data files
datas = []

# Include Demucs remote configuration and model files
demucs_paths = [
    '/opt/homebrew/anaconda3/lib/python3.10/site-packages/demucs/remote',
    '/opt/homebrew/anaconda3/lib/python3.10/site-packages/demucs/models',
]

for demucs_path in demucs_paths:
    if os.path.exists(demucs_path):
        datas.append((demucs_path, os.path.join('demucs', os.path.basename(demucs_path))))

# Include any cached Demucs models
import os
home_dir = os.path.expanduser('~')
cache_dirs = [
    os.path.join(home_dir, '.cache', 'demucs'),
    os.path.join(home_dir, '.cache', 'torch', 'hub', 'checkpoints'),
]

for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        datas.append((cache_dir, os.path.join('cache', 'demucs')))
