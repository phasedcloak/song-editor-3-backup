#!/usr/bin/env python3
"""
Setup script for Song Editor 3
"""
from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "A professional song editing and transcription application with OpenAI Whisper, Chordino, and advanced audio processing"

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

setup(
    name="song-editor-3",
    version="3.0.0",
    author="Song Editor Team",
    description="A professional song editing and transcription application with OpenAI Whisper, Chordino, and advanced audio processing",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "song-editor-3=song_editor.app:main",
            "song-editor-3-batch=song_editor.batch:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Editors",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    include_package_data=True,
    package_data={
        "song_editor": ["*.ui", "*.qml", "*.qrc", "assets/*", "*.json"],
    },
    keywords=[
        "audio", "transcription", "whisper", "chordino", "demucs", 
        "midi", "karaoke", "lyrics", "chords", "music", "editing"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/song-editor-3/issues",
        "Source": "https://github.com/your-username/song-editor-3",
        "Documentation": "https://github.com/your-username/song-editor-3/blob/main/README.md",
    },
)
