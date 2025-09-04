from __future__ import annotations

from typing import Optional, Tuple

import os
import tempfile


def separate_vocals_instrumental(audio_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Optionally separate stems using Demucs if available.
    Returns paths to (vocals_wav, instrumental_wav). If separation is not available,
    returns (None, None).
    """
    try:
        from demucs.separate import main as demucs_main  # type: ignore
        # Use a persistent cache dir so files remain after function returns
        base = os.path.join(tempfile.gettempdir(), "song_editor_2_stems")
        os.makedirs(base, exist_ok=True)
        args = [
            "-n", "htdemucs",
            "-o", base,
            "--two-stems", "vocals",
            audio_path,
        ]
        demucs_main(args)
        # Discover output recursively under base
        voc_path: Optional[str] = None
        inst_path: Optional[str] = None
        for root, dirs, files in os.walk(base):
            if any(f.endswith("vocals.wav") for f in files) and any(f.endswith("no_vocals.wav") for f in files):
                vocals = next(f for f in files if f.endswith("vocals.wav"))
                no_vocals = next(f for f in files if f.endswith("no_vocals.wav"))
                voc_path = os.path.join(root, vocals)
                inst_path = os.path.join(root, no_vocals)
                break
        return (voc_path, inst_path)
    except Exception:
        return (None, None)
