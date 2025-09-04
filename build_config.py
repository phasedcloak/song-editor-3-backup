#!/usr/bin/env python3
"""
Cross-Platform Build Configuration for Song Editor 3

Handles building and packaging for different platforms:
- macOS (desktop and iOS)
- Windows (desktop)
- Android
- Linux (desktop)
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Any


class CrossPlatformBuilder:
    """Cross-platform build system for Song Editor 3."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.platform = platform.system().lower()
        self.arch = platform.machine().lower()
        
        # Create build directories
        self.build_dir.mkdir(exist_ok=True)
        self.dist_dir.mkdir(exist_ok=True)
    
    def detect_platform(self) -> str:
        """Detect the current platform."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if system == "darwin":
            if "iphone" in machine or "ipad" in machine:
                return "ios"
            else:
                return "macos"
        elif system == "windows":
            return "windows"
        elif system == "linux":
            if os.path.exists("/system/build.prop"):
                return "android"
            else:
                return "linux"
        else:
            return "unknown"
    
    def get_platform_config(self) -> Dict[str, Any]:
        """Get platform-specific build configuration."""
        platform_type = self.detect_platform()
        
        configs = {
            "macos": {
                "app_name": "Song Editor 3",
                "bundle_id": "com.songeditor.app",
                "icon": "assets/icon.icns",
                "entitlements": "assets/macos.entitlements",
                "frameworks": ["Qt6", "Python"],
                "dylibs": [],
                "codesign": True,
                "notarize": True,
                "target_arch": ["x86_64", "arm64"]
            },
            "ios": {
                "app_name": "Song Editor 3",
                "bundle_id": "com.songeditor.ios",
                "icon": "assets/icon.icns",
                "entitlements": "assets/ios.entitlements",
                "frameworks": ["Qt6", "Python"],
                "target_arch": ["arm64"],
                "deployment_target": "14.0"
            },
            "windows": {
                "app_name": "Song Editor 3",
                "icon": "assets/icon.ico",
                "installer": True,
                "nsis_script": "assets/installer.nsi",
                "target_arch": ["x86_64"],
                "msi": True
            },
            "android": {
                "app_name": "Song Editor 3",
                "package_name": "com.songeditor.android",
                "icon": "assets/icon.png",
                "permissions": [
                    "android.permission.RECORD_AUDIO",
                    "android.permission.READ_EXTERNAL_STORAGE",
                    "android.permission.WRITE_EXTERNAL_STORAGE"
                ],
                "target_sdk": 33,
                "min_sdk": 21,
                "target_arch": ["arm64-v8a", "x86_64"]
            },
            "linux": {
                "app_name": "Song Editor 3",
                "icon": "assets/icon.png",
                "desktop_file": "assets/song-editor.desktop",
                "appimage": True,
                "snap": True,
                "target_arch": ["x86_64"]
            }
        }
        
        return configs.get(platform_type, {})
    
    def build_macos(self) -> bool:
        """Build for macOS."""
        print("Building for macOS...")
        config = self.get_platform_config()
        
        try:
            # Use pyinstaller for macOS
            cmd = [
                "pyinstaller",
                "--onefile",
                "--windowed",
                f"--name={config['app_name']}",
                f"--icon={config['icon']}",
                "--add-data=song_editor:song_editor",
                "--hidden-import=PySide6",
                "--hidden-import=librosa",
                "--hidden-import=soundfile",
                "--hidden-import=numpy",
                "--hidden-import=scipy",
                "--hidden-import=mido",
                "--hidden-import=pretty_midi",
                "--hidden-import=whisper",
                "--hidden-import=faster_whisper",
                "--hidden-import=whisperx",
                "--hidden-import=mlx_whisper",
                "--hidden-import=chord_extractor",
                "--hidden-import=basic_pitch",
                "--hidden-import=crepe",
                "--hidden-import=demucs",
                "--hidden-import=noisereduce",
                "--hidden-import=pyloudnorm",
                "--collect-all=song_editor",
                "song_editor/app.py"
            ]
            
            subprocess.run(cmd, check=True)
            
            # Create app bundle
            self._create_macos_bundle(config)
            
            print("✅ macOS build completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ macOS build failed: {e}")
            return False
    
    def build_ios(self) -> bool:
        """Build for iOS."""
        print("Building for iOS...")
        config = self.get_platform_config()
        
        try:
            # Use pyinstaller with iOS-specific options
            cmd = [
                "pyinstaller",
                "--onefile",
                "--windowed",
                f"--name={config['app_name']}",
                f"--icon={config['icon']}",
                "--add-data=song_editor:song_editor",
                "--target-arch=arm64",
                "--osx-bundle-identifier=" + config['bundle_id'],
                "--hidden-import=PySide6",
                "--hidden-import=librosa",
                "--hidden-import=soundfile",
                "--hidden-import=numpy",
                "--hidden-import=scipy",
                "--hidden-import=mido",
                "--hidden-import=pretty_midi",
                "--hidden-import=whisper",
                "--hidden-import=faster_whisper",
                "--hidden-import=whisperx",
                "--hidden-import=mlx_whisper",
                "--hidden-import=chord_extractor",
                "--hidden-import=basic_pitch",
                "--hidden-import=crepe",
                "--hidden-import=demucs",
                "--hidden-import=noisereduce",
                "--hidden-import=pyloudnorm",
                "--collect-all=song_editor",
                "song_editor/app.py"
            ]
            
            subprocess.run(cmd, check=True)
            
            print("✅ iOS build completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ iOS build failed: {e}")
            return False
    
    def build_windows(self) -> bool:
        """Build for Windows."""
        print("Building for Windows...")
        config = self.get_platform_config()
        
        try:
            # Use pyinstaller for Windows
            cmd = [
                "pyinstaller",
                "--onefile",
                "--windowed",
                f"--name={config['app_name']}",
                f"--icon={config['icon']}",
                "--add-data=song_editor;song_editor",
                "--hidden-import=PySide6",
                "--hidden-import=librosa",
                "--hidden-import=soundfile",
                "--hidden-import=numpy",
                "--hidden-import=scipy",
                "--hidden-import=mido",
                "--hidden-import=pretty_midi",
                "--hidden-import=whisper",
                "--hidden-import=faster_whisper",
                "--hidden-import=whisperx",
                "--hidden-import=mlx_whisper",
                "--hidden-import=chord_extractor",
                "--hidden-import=basic_pitch",
                "--hidden-import=crepe",
                "--hidden-import=demucs",
                "--hidden-import=noisereduce",
                "--hidden-import=pyloudnorm",
                "--collect-all=song_editor",
                "song_editor/app.py"
            ]
            
            subprocess.run(cmd, check=True)
            
            # Create installer if requested
            if config.get("installer"):
                self._create_windows_installer(config)
            
            print("✅ Windows build completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Windows build failed: {e}")
            return False
    
    def build_android(self) -> bool:
        """Build for Android."""
        print("Building for Android...")
        config = self.get_platform_config()
        
        try:
            # Use python-for-android or similar tool
            # This is a simplified version - in practice, you'd use p4a or similar
            cmd = [
                "python", "-m", "pythonforandroid",
                "apk",
                "--private", ".",
                "--package", config['package_name'],
                "--name", config['app_name'],
                "--version", "1.0.0",
                "--bootstrap", "sdl2",
                "--requirements", "python3,pyqt6,librosa,soundfile,numpy,scipy,mido,whisper"
            ]
            
            subprocess.run(cmd, check=True)
            
            print("✅ Android build completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Android build failed: {e}")
            return False
    
    def build_linux(self) -> bool:
        """Build for Linux."""
        print("Building for Linux...")
        config = self.get_platform_config()
        
        try:
            # Use pyinstaller for Linux
            cmd = [
                "pyinstaller",
                "--onefile",
                "--windowed",
                f"--name={config['app_name']}",
                f"--icon={config['icon']}",
                "--add-data=song_editor:song_editor",
                "--hidden-import=PySide6",
                "--hidden-import=librosa",
                "--hidden-import=soundfile",
                "--hidden-import=numpy",
                "--hidden-import=scipy",
                "--hidden-import=mido",
                "--hidden-import=pretty_midi",
                "--hidden-import=whisper",
                "--hidden-import=faster_whisper",
                "--hidden-import=whisperx",
                "--hidden-import=mlx_whisper",
                "--hidden-import=chord_extractor",
                "--hidden-import=basic_pitch",
                "--hidden-import=crepe",
                "--hidden-import=demucs",
                "--hidden-import=noisereduce",
                "--hidden-import=pyloudnorm",
                "--collect-all=song_editor",
                "song_editor/app.py"
            ]
            
            subprocess.run(cmd, check=True)
            
            # Create AppImage if requested
            if config.get("appimage"):
                self._create_linux_appimage(config)
            
            print("✅ Linux build completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Linux build failed: {e}")
            return False
    
    def _create_macos_bundle(self, config: Dict[str, Any]) -> None:
        """Create macOS app bundle."""
        # Implementation for creating .app bundle
        pass
    
    def _create_windows_installer(self, config: Dict[str, Any]) -> None:
        """Create Windows installer."""
        # Implementation for creating NSIS installer
        pass
    
    def _create_linux_appimage(self, config: Dict[str, Any]) -> None:
        """Create Linux AppImage."""
        # Implementation for creating AppImage
        pass
    
    def build_all(self) -> bool:
        """Build for all supported platforms."""
        platform_type = self.detect_platform()
        
        print(f"Detected platform: {platform_type}")
        print(f"Architecture: {self.arch}")
        
        if platform_type == "macos":
            return self.build_macos()
        elif platform_type == "ios":
            return self.build_ios()
        elif platform_type == "windows":
            return self.build_windows()
        elif platform_type == "android":
            return self.build_android()
        elif platform_type == "linux":
            return self.build_linux()
        else:
            print(f"❌ Unsupported platform: {platform_type}")
            return False
    
    def clean(self) -> None:
        """Clean build artifacts."""
        import shutil
        
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        
        if self.dist_dir.exists():
            shutil.rmtree(self.dist_dir)
        
        # Clean pyinstaller artifacts
        spec_file = self.project_root / "song_editor.spec"
        if spec_file.exists():
            spec_file.unlink()
        
        print("✅ Build artifacts cleaned!")


def main():
    """Main build script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-platform build system for Song Editor 3")
    parser.add_argument("--platform", choices=["macos", "ios", "windows", "android", "linux", "all"],
                       help="Target platform to build for")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--config", action="store_true", help="Show build configuration")
    
    args = parser.parse_args()
    
    builder = CrossPlatformBuilder()
    
    if args.clean:
        builder.clean()
        return
    
    if args.config:
        config = builder.get_platform_config()
        print("Build Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        return
    
    if args.platform:
        if args.platform == "all":
            success = builder.build_all()
        elif args.platform == "macos":
            success = builder.build_macos()
        elif args.platform == "ios":
            success = builder.build_ios()
        elif args.platform == "windows":
            success = builder.build_windows()
        elif args.platform == "android":
            success = builder.build_android()
        elif args.platform == "linux":
            success = builder.build_linux()
        else:
            print(f"❌ Unknown platform: {args.platform}")
            return 1
        
        return 0 if success else 1
    else:
        # Build for current platform
        success = builder.build_all()
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
