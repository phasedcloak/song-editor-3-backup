#!/usr/bin/env python3
"""
Platform-Specific Styles for Song Editor 3

Provides platform-appropriate styling for macOS, iOS, Windows, and Android
while maintaining all functionality across platforms.
"""

from typing import Dict, Any
from ..platform_utils import PlatformUtils


class PlatformStyles:
    """Platform-specific stylesheet generator."""

    @staticmethod
    def get_main_window_style() -> str:
        """Get platform-specific main window stylesheet."""
        config = PlatformUtils.get_platform_config()
        ui_style = config.get("ui_style", "default")

        if ui_style == "macos":
            return PlatformStyles._get_macos_style()
        elif ui_style == "ios":
            return PlatformStyles._get_ios_style()
        elif ui_style == "windows":
            return PlatformStyles._get_windows_style()
        elif ui_style == "android":
            return PlatformStyles._get_android_style()
        else:
            return PlatformStyles._get_default_style()

    @staticmethod
    def _get_macos_style() -> str:
        """Get macOS-specific stylesheet."""
        return """
        QMainWindow {
            background-color: #F5F5F7;
            color: #000000;
            font-family: "SF Pro Display";
            font-size: 13px;
        }

        QMenuBar {
            background-color: #F5F5F7;
            border-bottom: 1px solid #E5E5E7;
            padding: 4px;
        }

        QMenuBar::item {
            background-color: transparent;
            padding: 6px 12px;
            border-radius: 6px;
        }

        QMenuBar::item:selected {
            background-color: #007AFF;
            color: white;
        }

        QPushButton {
            background-color: #007AFF;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 500;
            min-height: 20px;
        }

        QPushButton:hover {
            background-color: #0056CC;
        }

        QPushButton:pressed {
            background-color: #004499;
        }

        QPushButton:disabled {
            background-color: #CCCCCC;
            color: #666666;
        }

        QTabWidget::pane {
            border: 1px solid #E5E5E7;
            border-radius: 8px;
            background-color: white;
        }

        QTabBar::tab {
            background-color: #F5F5F7;
            color: #666666;
            padding: 8px 16px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            margin-right: 2px;
        }

        QTabBar::tab:selected {
            background-color: white;
            color: #007AFF;
            font-weight: 500;
        }

        QGroupBox {
            font-weight: 600;
            border: 1px solid #E5E5E7;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 8px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 8px 0 8px;
        }

        QLineEdit, QTextEdit, QComboBox {
            border: 1px solid #E5E5E7;
            border-radius: 6px;
            padding: 6px 8px;
            background-color: white;
        }

        QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
            border-color: #007AFF;
        }

        QProgressBar {
            border: 1px solid #E5E5E7;
            border-radius: 6px;
            text-align: center;
            background-color: #F5F5F7;
        }

        QProgressBar::chunk {
            background-color: #007AFF;
            border-radius: 5px;
        }
        """

    @staticmethod
    def _get_ios_style() -> str:
        """Get iOS-specific stylesheet."""
        return """
        QMainWindow {
            background-color: #F2F2F7;
            color: #000000;
            font-family: "SF Pro Display";
            font-size: 16px;
        }

        QMenuBar {
            background-color: #F2F2F7;
            border-bottom: 1px solid #E5E5E7;
            padding: 8px;
        }

        QMenuBar::item {
            background-color: transparent;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 16px;
        }

        QMenuBar::item:selected {
            background-color: #007AFF;
            color: white;
        }

        QPushButton {
            background-color: #007AFF;
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 16px;
            min-height: 44px;
        }

        QPushButton:hover {
            background-color: #0056CC;
        }

        QPushButton:pressed {
            background-color: #004499;
        }

        QPushButton:disabled {
            background-color: #CCCCCC;
            color: #666666;
        }

        QTabWidget::pane {
            border: 1px solid #E5E5E7;
            border-radius: 12px;
            background-color: white;
        }

        QTabBar::tab {
            background-color: #F2F2F7;
            color: #666666;
            padding: 12px 20px;
            border-top-left-radius: 12px;
            border-top-right-radius: 12px;
            margin-right: 4px;
            font-size: 16px;
        }

        QTabBar::tab:selected {
            background-color: white;
            color: #007AFF;
            font-weight: 600;
        }

        QGroupBox {
            font-weight: 600;
            border: 1px solid #E5E5E7;
            border-radius: 12px;
            margin-top: 16px;
            padding-top: 12px;
            font-size: 16px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 16px;
            padding: 0 12px 0 12px;
        }

        QLineEdit, QTextEdit, QComboBox {
            border: 1px solid #E5E5E7;
            border-radius: 8px;
            padding: 12px 16px;
            background-color: white;
            font-size: 16px;
            min-height: 44px;
        }

        QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
            border-color: #007AFF;
        }

        QProgressBar {
            border: 1px solid #E5E5E7;
            border-radius: 8px;
            text-align: center;
            background-color: #F2F2F7;
            min-height: 8px;
        }

        QProgressBar::chunk {
            background-color: #007AFF;
            border-radius: 6px;
        }
        """

    @staticmethod
    def _get_windows_style() -> str:
        """Get Windows-specific stylesheet."""
        return """
        QMainWindow {
            background-color: #F3F3F3;
            color: #000000;
            font-family: "Segoe UI";
            font-size: 9px;
        }

        QMenuBar {
            background-color: #F3F3F3;
            border-bottom: 1px solid #D4D4D4;
            padding: 2px;
        }

        QMenuBar::item {
            background-color: transparent;
            padding: 4px 8px;
            border-radius: 3px;
        }

        QMenuBar::item:selected {
            background-color: #0078D4;
            color: white;
        }

        QPushButton {
            background-color: #0078D4;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            font-weight: normal;
            min-height: 16px;
        }

        QPushButton:hover {
            background-color: #106EBE;
        }

        QPushButton:pressed {
            background-color: #005A9E;
        }

        QPushButton:disabled {
            background-color: #CCCCCC;
            color: #666666;
        }

        QTabWidget::pane {
            border: 1px solid #D4D4D4;
            border-radius: 4px;
            background-color: white;
        }

        QTabBar::tab {
            background-color: #F3F3F3;
            color: #666666;
            padding: 6px 12px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: 1px;
        }

        QTabBar::tab:selected {
            background-color: white;
            color: #0078D4;
            font-weight: normal;
        }

        QGroupBox {
            font-weight: normal;
            border: 1px solid #D4D4D4;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 6px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 6px 0 6px;
        }

        QLineEdit, QTextEdit, QComboBox {
            border: 1px solid #D4D4D4;
            border-radius: 3px;
            padding: 4px 6px;
            background-color: white;
        }

        QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
            border-color: #0078D4;
        }

        QProgressBar {
            border: 1px solid #D4D4D4;
            border-radius: 3px;
            text-align: center;
            background-color: #F3F3F3;
        }

        QProgressBar::chunk {
            background-color: #0078D4;
            border-radius: 2px;
        }
        """

    @staticmethod
    def _get_android_style() -> str:
        """Get Android-specific stylesheet."""
        return """
        QMainWindow {
            background-color: #FAFAFA;
            color: #000000;
            font-family: "Roboto";
            font-size: 14px;
        }

        QMenuBar {
            background-color: #FAFAFA;
            border-bottom: 1px solid #E0E0E0;
            padding: 8px;
        }

        QMenuBar::item {
            background-color: transparent;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 14px;
        }

        QMenuBar::item:selected {
            background-color: #6200EE;
            color: white;
        }

        QPushButton {
            background-color: #6200EE;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 500;
            font-size: 14px;
            min-height: 48px;
        }

        QPushButton:hover {
            background-color: #3700B3;
        }

        QPushButton:pressed {
            background-color: #30009C;
        }

        QPushButton:disabled {
            background-color: #CCCCCC;
            color: #666666;
        }

        QTabWidget::pane {
            border: 1px solid #E0E0E0;
            border-radius: 8px;
            background-color: white;
        }

        QTabBar::tab {
            background-color: #FAFAFA;
            color: #666666;
            padding: 10px 18px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            margin-right: 2px;
            font-size: 14px;
        }

        QTabBar::tab:selected {
            background-color: white;
            color: #6200EE;
            font-weight: 500;
        }

        QGroupBox {
            font-weight: 500;
            border: 1px solid #E0E0E0;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 10px;
            font-size: 14px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 10px 0 10px;
        }

        QLineEdit, QTextEdit, QComboBox {
            border: 1px solid #E0E0E0;
            border-radius: 6px;
            padding: 10px 14px;
            background-color: white;
            font-size: 14px;
            min-height: 48px;
        }

        QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
            border-color: #6200EE;
        }

        QProgressBar {
            border: 1px solid #E0E0E0;
            border-radius: 6px;
            text-align: center;
            background-color: #FAFAFA;
            min-height: 8px;
        }

        QProgressBar::chunk {
            background-color: #6200EE;
            border-radius: 5px;
        }
        """

    @staticmethod
    def _get_default_style() -> str:
        """Get default stylesheet for unsupported platforms."""
        return """
        QMainWindow {
            background-color: #F5F5F5;
            color: #000000;
            font-family: "Arial";
            font-size: 12px;
        }

        QMenuBar {
            background-color: #F5F5F5;
            border-bottom: 1px solid #CCCCCC;
            padding: 4px;
        }

        QMenuBar::item {
            background-color: transparent;
            padding: 6px 12px;
            border-radius: 4px;
        }

        QMenuBar::item:selected {
            background-color: #007AFF;
            color: white;
        }

        QPushButton {
            background-color: #007AFF;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: normal;
            min-height: 20px;
        }

        QPushButton:hover {
            background-color: #0056CC;
        }

        QPushButton:pressed {
            background-color: #004499;
        }

        QPushButton:disabled {
            background-color: #CCCCCC;
            color: #666666;
        }

        QTabWidget::pane {
            border: 1px solid #CCCCCC;
            border-radius: 6px;
            background-color: white;
        }

        QTabBar::tab {
            background-color: #F5F5F5;
            color: #666666;
            padding: 8px 16px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            margin-right: 2px;
        }

        QTabBar::tab:selected {
            background-color: white;
            color: #007AFF;
            font-weight: normal;
        }

        QGroupBox {
            font-weight: normal;
            border: 1px solid #CCCCCC;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 8px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 8px 0 8px;
        }

        QLineEdit, QTextEdit, QComboBox {
            border: 1px solid #CCCCCC;
            border-radius: 4px;
            padding: 6px 8px;
            background-color: white;
        }

        QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
            border-color: #007AFF;
        }

        QProgressBar {
            border: 1px solid #CCCCCC;
            border-radius: 4px;
            text-align: center;
            background-color: #F5F5F5;
        }

        QProgressBar::chunk {
            background-color: #007AFF;
            border-radius: 3px;
        }
        """

    @staticmethod
    def get_mobile_optimizations() -> Dict[str, Any]:
        """Get mobile-specific optimizations."""
        if PlatformUtils.is_mobile():
            return {
                "touch_target_size": 44,  # Minimum touch target size in pixels
                "spacing": 16,  # Increased spacing for touch interfaces
                "font_size_multiplier": 1.2,  # Larger fonts for mobile
                "button_height": 48,  # Taller buttons for touch
                "scroll_speed": 1.5,  # Faster scrolling for touch
                "gesture_enabled": True,
                "safe_area_margin": 20
            }
        else:
            return {
                "touch_target_size": 20,
                "spacing": 8,
                "font_size_multiplier": 1.0,
                "button_height": 32,
                "scroll_speed": 1.0,
                "gesture_enabled": False,
                "safe_area_margin": 0
            }

    @staticmethod
    def get_high_dpi_settings() -> Dict[str, Any]:
        """Get high DPI display settings."""
        if PlatformUtils.is_high_dpi():
            return {
                "scale_factor": 1.0,  # Let Qt handle scaling
                "icon_size": 24,
                "font_size_adjustment": 0,
                "border_width": 1,
                "shadow_blur": 10
            }
        else:
            return {
                "scale_factor": 1.0,
                "icon_size": 16,
                "font_size_adjustment": 0,
                "border_width": 1,
                "shadow_blur": 5
            }
