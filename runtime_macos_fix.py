"""
Runtime hook to fix macOS library linking issues on macOS 15+
"""

import os
import sys
import platform

def fix_macos_library_paths():
    """Fix library paths for macOS compatibility."""
    if platform.system() != 'Darwin':
        return

    # Get macOS version
    mac_ver = platform.mac_ver()[0]
    major_version = int(mac_ver.split('.')[0]) if mac_ver else 0

    # Aggressive library path setting for all macOS versions
    lib_paths = [
        '/usr/lib',
        '/System/Library/Frameworks',
        '/System/Library/PrivateFrameworks',
        '/usr/local/lib',
        '/opt/homebrew/lib',
        '/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib',
        '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/lib',
    ]

    # Filter out paths that don't exist
    existing_paths = [path for path in lib_paths if os.path.exists(path)]
    lib_path_str = ':'.join(existing_paths)

    os.environ['DYLD_LIBRARY_PATH'] = lib_path_str
    os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = lib_path_str

    # Also set rpath-related variables
    os.environ['DYLD_FRAMEWORK_PATH'] = '/System/Library/Frameworks:/System/Library/PrivateFrameworks'
    os.environ['DYLD_FALLBACK_FRAMEWORK_PATH'] = '/System/Library/Frameworks:/System/Library/PrivateFrameworks'

    # Try to find libSystem in various locations and create symbolic links if needed
    libsystem_paths = [
        '/usr/lib/libSystem.B.dylib',
        '/usr/lib/libSystem.dylib',
        '/System/Library/Frameworks/System.framework/Versions/B/System',
        '/System/Library/Frameworks/System.framework/System',
    ]

    libsystem_found = None
    for path in libsystem_paths:
        if os.path.exists(path):
            libsystem_found = path
            break

    if libsystem_found:
        # Create symlink in common locations if it doesn't exist
        symlink_locations = ['/usr/local/lib/libSystem.B.dylib', '/usr/local/lib/libSystem.dylib']
        for symlink_loc in symlink_locations:
            symlink_dir = os.path.dirname(symlink_loc)
            if os.path.exists(symlink_dir) and not os.path.exists(symlink_loc):
                try:
                    os.symlink(libsystem_found, symlink_loc)
                    print(f"Created symlink: {symlink_loc} -> {libsystem_found}")
                except Exception as e:
                    print(f"Failed to create symlink {symlink_loc}: {e}")

    # Additional fix for libSystem.B.dylib issue
    try:
        import subprocess
        # Try to find the actual libSystem location
        result = subprocess.run(['otool', '-L', sys.executable],
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'libSystem' in line and '@rpath' in line:
                    # This indicates we need to fix the rpath
                    print("Found rpath reference to libSystem, attempting to fix...")
                    break
    except Exception as e:
        print(f"Error checking executable dependencies: {e}")

def pre_run():
    """Called before the main application starts."""
    fix_macos_library_paths()

# Call fix on import
fix_macos_library_paths()
