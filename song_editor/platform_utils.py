#!/usr/bin/env python3
"""
Platform Utilities for Song Editor 3

Handles platform detection and provides platform-specific configurations
for cross-platform compatibility across macOS, iOS, Windows, and Android.
"""

import sys
import os
import platform
from typing import Dict, Any, Optional
from enum import Enum


class Platform(Enum):
    """Supported platforms."""
    MACOS = "macos"
    IOS = "ios"
    WINDOWS = "windows"
    ANDROID = "android"
    LINUX = "linux"
    UNKNOWN = "unknown"


class PlatformUtils:
    """Platform detection and configuration utilities."""

    @staticmethod
    def detect_platform() -> Platform:
        """Detect the current platform."""
        system = platform.system().lower()
        machine = platform.machine().lower()

        # macOS detection
        if system == "darwin":
            # Check if running on iOS (iOS apps run on Darwin)
            if "iphone" in machine or "ipad" in machine:
                return Platform.IOS
            else:
                return Platform.MACOS

        # Windows detection
        elif system == "windows":
            return Platform.WINDOWS

        # Android detection (Android apps can run on Linux)
        elif system == "linux":
            # Check for Android-specific indicators
            if os.path.exists("/system/build.prop") or "android" in machine:
                return Platform.ANDROID
            else:
                return Platform.LINUX

        return Platform.UNKNOWN

    @staticmethod
    def get_platform_config() -> Dict[str, Any]:
        """Get platform-specific configuration."""
        platform_type = PlatformUtils.detect_platform()

        configs = {
            Platform.MACOS: {
                "ui_style": "macos",
                "font_family": "SF Pro Display",
                "font_size": 13,
                "accent_color": "#007AFF",
                "background_color": "#F5F5F7",
                "window_style": "native",
                "menu_bar_style": "native",
                "touch_support": False,
                "high_dpi": True,
                "dark_mode_support": True,
                "file_dialog_style": "native",
                "icon_style": "flat",
                "animation_speed": "smooth",
                "window_shadows": True,
                "rounded_corners": True
            },
            Platform.IOS: {
                "ui_style": "ios",
                "font_family": "SF Pro Display",
                "font_size": 16,
                "accent_color": "#007AFF",
                "background_color": "#F2F2F7",
                "window_style": "mobile",
                "menu_bar_style": "minimal",
                "touch_support": True,
                "high_dpi": True,
                "dark_mode_support": True,
                "file_dialog_style": "mobile",
                "icon_style": "flat",
                "animation_speed": "fast",
                "window_shadows": False,
                "rounded_corners": True,
                "gesture_support": True,
                "safe_area_insets": True
            },
            Platform.WINDOWS: {
                "ui_style": "windows",
                "font_family": "Segoe UI",
                "font_size": 9,
                "accent_color": "#0078D4",
                "background_color": "#F3F3F3",
                "window_style": "native",
                "menu_bar_style": "native",
                "touch_support": False,
                "high_dpi": True,
                "dark_mode_support": True,
                "file_dialog_style": "native",
                "icon_style": "flat",
                "animation_speed": "normal",
                "window_shadows": True,
                "rounded_corners": False
            },
            Platform.ANDROID: {
                "ui_style": "android",
                "font_family": "Roboto",
                "font_size": 14,
                "accent_color": "#6200EE",
                "background_color": "#FAFAFA",
                "window_style": "mobile",
                "menu_bar_style": "minimal",
                "touch_support": True,
                "high_dpi": True,
                "dark_mode_support": True,
                "file_dialog_style": "mobile",
                "icon_style": "material",
                "animation_speed": "fast",
                "window_shadows": False,
                "rounded_corners": True,
                "gesture_support": True,
                "safe_area_insets": True
            },
            Platform.LINUX: {
                "ui_style": "linux",
                "font_family": "Ubuntu",
                "font_size": 10,
                "accent_color": "#E95420",
                "background_color": "#F5F5F5",
                "window_style": "native",
                "menu_bar_style": "native",
                "touch_support": False,
                "high_dpi": True,
                "dark_mode_support": True,
                "file_dialog_style": "native",
                "icon_style": "flat",
                "animation_speed": "normal",
                "window_shadows": True,
                "rounded_corners": False
            }
        }

        return configs.get(platform_type, configs[Platform.LINUX])

    @staticmethod
    def is_mobile() -> bool:
        """Check if running on a mobile platform."""
        platform_type = PlatformUtils.detect_platform()
        return platform_type in [Platform.IOS, Platform.ANDROID]

    @staticmethod
    def is_desktop() -> bool:
        """Check if running on a desktop platform."""
        platform_type = PlatformUtils.detect_platform()
        return platform_type in [Platform.MACOS, Platform.WINDOWS, Platform.LINUX]

    @staticmethod
    def is_touch_supported() -> bool:
        """Check if touch input is supported."""
        config = PlatformUtils.get_platform_config()
        return config.get("touch_support", False)

    @staticmethod
    def is_high_dpi() -> bool:
        """Check if high DPI display is supported."""
        config = PlatformUtils.get_platform_config()
        return config.get("high_dpi", True)

    @staticmethod
    def get_recommended_window_size() -> tuple:
        """Get recommended window size for the current platform."""
        if PlatformUtils.is_mobile():
            # Mobile platforms - full screen or large modal
            return (800, 600)
        else:
            # Desktop platforms - standard window size
            return (1200, 800)

    @staticmethod
    def get_recommended_font_size() -> int:
        """Get recommended font size for the current platform."""
        config = PlatformUtils.get_platform_config()
        return config.get("font_size", 12)

    @staticmethod
    def get_accent_color() -> str:
        """Get platform-specific accent color."""
        config = PlatformUtils.get_platform_config()
        return config.get("accent_color", "#007AFF")

    @staticmethod
    def get_background_color() -> str:
        """Get platform-specific background color."""
        config = PlatformUtils.get_platform_config()
        return config.get("background_color", "#F5F5F7")

    @staticmethod
    def get_ui_style() -> str:
        """Get platform-specific UI style."""
        config = PlatformUtils.get_platform_config()
        return config.get("ui_style", "default")

    @staticmethod
    def get_font_family() -> str:
        """Get platform-specific font family."""
        config = PlatformUtils.get_platform_config()
        return config.get("font_family", "Arial")

    @staticmethod
    def should_use_native_dialogs() -> bool:
        """Check if native file dialogs should be used."""
        config = PlatformUtils.get_platform_config()
        return config.get("file_dialog_style", "native") == "native"

    @staticmethod
    def should_use_rounded_corners() -> bool:
        """Check if rounded corners should be used."""
        config = PlatformUtils.get_platform_config()
        return config.get("rounded_corners", False)

    @staticmethod
    def should_use_window_shadows() -> bool:
        """Check if window shadows should be used."""
        config = PlatformUtils.get_platform_config()
        return config.get("window_shadows", True)

    @staticmethod
    def get_animation_speed() -> str:
        """Get platform-specific animation speed."""
        config = PlatformUtils.get_platform_config()
        return config.get("animation_speed", "normal")

    @staticmethod
    def supports_dark_mode() -> bool:
        """Check if dark mode is supported."""
        config = PlatformUtils.get_platform_config()
        return config.get("dark_mode_support", False)

    @staticmethod
    def supports_gestures() -> bool:
        """Check if gesture support is available."""
        config = PlatformUtils.get_platform_config()
        return config.get("gesture_support", False)

    @staticmethod
    def needs_safe_area_insets() -> bool:
        """Check if safe area insets are needed (mobile platforms)."""
        config = PlatformUtils.get_platform_config()
        return config.get("safe_area_insets", False)

    @staticmethod
    def get_platform_info() -> Dict[str, Any]:
        """Get comprehensive platform information."""
        platform_type = PlatformUtils.detect_platform()
        config = PlatformUtils.get_platform_config()

        return {
            "platform": platform_type.value,
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "is_mobile": PlatformUtils.is_mobile(),
            "is_desktop": PlatformUtils.is_desktop(),
            "is_touch_supported": PlatformUtils.is_touch_supported(),
            "is_high_dpi": PlatformUtils.is_high_dpi(),
            "config": config
        }


class PlatformAwareWidget:
    """Base class for platform-aware widgets."""

    def __init__(self):
        self.platform_config = PlatformUtils.get_platform_config()
        self.setup_platform_specific_behavior()

    def setup_platform_specific_behavior(self):
        """Setup platform-specific behavior for the widget."""
        # Override in subclasses
        pass

    def apply_platform_style(self, widget):
        """Apply platform-specific styling to a widget."""
        # Override in subclasses
        pass

    def get_platform_font(self, size: Optional[int] = None) -> str:
        """Get platform-specific font."""
        if size is None:
            size = self.platform_config.get("font_size", 12)

        font_family = self.platform_config.get("font_family", "Arial")
        return f"{font_family}, {size}pt"

    def get_platform_color(self, color_type: str) -> str:
        """Get platform-specific color."""
        color_map = {
            "accent": self.platform_config.get("accent_color", "#007AFF"),
            "background": self.platform_config.get("background_color", "#F5F5F7"),
            "text": "#000000",
            "text_secondary": "#666666",
            "border": "#CCCCCC",
            "success": "#34C759",
            "warning": "#FF9500",
            "error": "#FF3B30"
        }

        return color_map.get(color_type, "#000000")
