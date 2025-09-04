"""
PyInstaller hook for PyTorch

Ensures all PyTorch dependencies are included in the bundle,
especially CUDA libraries and optimized kernels.
"""

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

# Collect all torch submodules
hiddenimports = collect_submodules('torch')

# Add specific torch modules that might be missed
hiddenimports += [
    'torch._C',
    'torch._C._fft',
    'torch._C._linalg',
    'torch._C._nn',
    'torch._C._sparse',
    'torch._C._special',
    'torch.nn.functional',
    'torch.optim',
    'torch.utils',
    'torch.utils.data',
    'torchvision',
    'torchvision.transforms',
    'torchaudio',
]

# Collect data files (models, etc.)
datas = collect_data_files('torch')

# Collect dynamic libraries (CUDA, MKL, etc.)
binaries = collect_dynamic_libs('torch')

# Add CUDA libraries if available
try:
    import torch
    if torch.cuda.is_available():
        # Add CUDA runtime libraries
        cuda_libs = [
            'libcudart.so*',
            'libnvrtc.so*',
            'libnvToolsExt.so*',
            'libcublas.so*',
            'libcufft.so*',
            'libcurand.so*',
            'libcusolver.so*',
            'libcusparse.so*',
        ]

        # Try to find CUDA installation
        cuda_paths = [
            '/usr/local/cuda/lib64',
            '/usr/lib/x86_64-linux-gnu',
            '/opt/cuda/lib64',
        ]

        for cuda_path in cuda_paths:
            if os.path.exists(cuda_path):
                for lib_pattern in cuda_libs:
                    lib_path = os.path.join(cuda_path, lib_pattern)
                    if os.path.exists(lib_path) or os.path.exists(lib_path.replace('so*', 'so')):
                        binaries.append((lib_path, '.'))

except ImportError:
    pass

# Add MKL libraries for Intel CPUs
mkl_libs = [
    'libmkl_core.so*',
    'libmkl_intel_lp64.so*',
    'libmkl_intel_thread.so*',
    'libmkl_def.so*',
    'libiomp5.so*',
]

mkl_paths = [
    '/opt/intel/mkl/lib/intel64',
    '/usr/lib/x86_64-linux-gnu',
    '/usr/local/lib',
]

for mkl_path in mkl_paths:
    if os.path.exists(mkl_path):
        for lib_pattern in mkl_libs:
            lib_path = os.path.join(mkl_path, lib_pattern)
            if os.path.exists(lib_path) or os.path.exists(lib_path.replace('so*', 'so')):
                binaries.append((lib_path, '.'))
