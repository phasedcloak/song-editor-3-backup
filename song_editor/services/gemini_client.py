from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import requests
import base64
import io
import time
import soundfile as sf
import numpy as np


@dataclass
class AltWord:
    text: str
    confidence: float


@dataclass
class AltWordTimed:
    text: str
    start: float
    end: float


@dataclass
class AltChordTimed:
    symbol: str
    start: float
    end: float


@dataclass
class AltNoteTimed:
    pitch_midi: int
    start: float
    end: float


class GeminiClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model_name = "gemini-2.5-flash"
        self.last_debug: str = ""
        self.last_notes: list[AltNoteTimed] = []

    def ensure_api_key(self) -> bool:
        return bool(self.api_key)

    def rewrite_lyrics(self, text: str, words_with_alternatives: List = None) -> List[AltWord]:
        self.last_debug = ""
        if not self.api_key:
            self.last_debug = "No API key set"
            return []

        # Build enhanced prompt with alternatives if available
        if words_with_alternatives and any(getattr(word, 'alt_text', None) for word in words_with_alternatives):
            # Create a detailed prompt that includes alternatives
            enhanced_text = "Original transcript with alternatives:\n"
            for word in words_with_alternatives:
                enhanced_text += f"'{word.text}' (alt: '{word.alt_text}' if available)\n"
            enhanced_text += f"\nFull text: {text}\n\n"

            prompt = (
                "You are a lyrics improvement expert. I have a transcript with some words that have "
                "alternative transcriptions. Use the alternatives when they make more sense or sound better. "
                "Rewrite this as improved lyrics, keeping word count the same where possible. "
                "Consider the alternatives and choose the best version of each word. "
                "Return JSON list of {text, confidence in [0,1]} for each word in order.\n\n" + enhanced_text
            )
        else:
            # Standard prompt for when no alternatives are available
            prompt = (
                "Rewrite this transcript as improved lyrics, keep word count the same where possible, "
                "return JSON list of {text, confidence in [0,1]} for each word in order.\n\n" + text
            )

        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
            )
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            self.last_debug = f"POST {url}\nModel: {self.model_name}\nPayload chars: {len(prompt)}\n"
            resp = requests.post(url, params={"key": self.api_key}, json=payload, timeout=30)
            self.last_debug += f"HTTP {resp.status_code}\n"
            # Log a snippet of response text for debugging
            try:
                snippet = resp.text[:500]
                self.last_debug += f"Resp head: {snippet}\n"
            except Exception:
                pass
            resp.raise_for_status()
            data = resp.json()
            candidate_text = None
            try:
                candidate_text = data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                self.last_debug += f"Parse err (no text field): {e}\n"
                candidate_text = None
            items: List[AltWord] = []
            if candidate_text and candidate_text.strip().startswith("["):
                import json as _json
                try:
                    arr = _json.loads(candidate_text)
                    for it in arr:
                        items.append(AltWord(text=str(it.get("text", "")), confidence=float(it.get("confidence", 0.5))))
                    return items
                except Exception as e:
                    self.last_debug += f"JSON list parse err: {e}\n"
            else:
                self.last_debug += "No JSON list returned in candidate_text\n"
        except Exception as e:
            self.last_debug += f"Request failed: {e}\n"
            return []
        return []

    def infer_chords(self, text: str) -> List[AltWord]:
        # Placeholder: chord tokens aligned with spaces; in a real impl, call a music understanding model
        words = text.split()
        # Repeat a simple pattern C, G, Am, F
        pattern = ["C", "G", "Am", "F"]
        return [AltWord(pattern[i % len(pattern)], 0.5) for i in range(len(words))]

    def analyze_audio_alt(self, audio_path: str) -> tuple[list[AltWordTimed], list[AltChordTimed]]:
        """
        Send audio to Gemini to get alternative lyrics and chords. Returns (alt_words, alt_chords).
        This uses inline_data with FLAC-encoded mono to reduce payload size.
        """
        self.last_debug = ""
        if not self.api_key:
            self.last_debug = "No API key set"
            return ([], [])
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        y = data.mean(axis=1)
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="FLAC")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        prompt = (
            "Analyze the given audio (full mix).\n"
            "1) Transcribe the lead vocal and rewrite as improved lyrics, with times per word.\n"
            "2) For each word, infer the harmonic chord WITH QUALITY using standard chord symbols and details.\n"
            "   Examples: C, Am, D7, Gmaj7/B, Fsus4, Eadd9, Bdim, Aaug, Em9, C#7b9, F#maj7#11.\n"
            "Return STRICT JSON with keys 'words' and 'chords'.\n"
            "- words: array of objects with keys: 'text', 'start_sec', 'end_sec'.\n"
            "- chords: array of objects with keys: 'symbol', 'root', 'quality', 'bass', 'start_sec', 'end_sec'.\n"
            "- Ensure both arrays are the same LENGTH and index-aligned.\n"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"inline_data": {"mime_type": "audio/flac", "data": b64}},
                        {"text": prompt}
                    ]
                }
            ]
        }
        try:
            self.last_debug = f"POST {url}\nModel: {self.model_name}\nAudio bytes: {len(b64)} (b64)\n"
            resp = requests.post(url, params={"key": self.api_key}, json=payload, timeout=60)
            self.last_debug += f"HTTP {resp.status_code}\n"
            self.last_debug += f"Resp head: {resp.text[:800]}\n"
            resp.raise_for_status()
            data = resp.json()
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                self.last_debug += f"Parse err (no text field): {e}\n"
                return ([], [])
            import json as _json
            try:
                obj = _json.loads(text)
                words_t: list[AltWordTimed] = []
                chords_t: list[AltChordTimed] = []
                self.last_notes = []
                for w in obj.get("words", []):
                    words_t.append(
                        AltWordTimed(
                            text=str(w.get("text", "")),
                            start=float(w.get("start_sec", 0.0)),
                            end=float(w.get("end_sec", 0.0))
                        )
                    )
                for c in obj.get("chords", []):
                    sym = str(c.get("symbol") or "")
                    if not sym:
                        root = str(c.get("root") or "")
                        qual = str(c.get("quality") or "")
                        bass = str(c.get("bass") or "")
                        sym = root + (qual if qual else "") + ("/" + bass if bass else "")
                    chords_t.append(
                        AltChordTimed(
                            symbol=sym,
                            start=float(c.get("start_sec", 0.0)),
                            end=float(c.get("end_sec", 0.0))
                        )
                    )
                for n in obj.get("notes", []) or []:
                    try:
                        self.last_notes.append(
                            AltNoteTimed(
                                pitch_midi=int(n.get("pitch_midi")),
                                start=float(n.get("start_sec", 0.0)),
                                end=float(n.get("end_sec", 0.0))
                            )
                        )
                    except Exception:
                        pass
                m = min(len(words_t), len(chords_t))
                return (words_t[:m], chords_t[:m])
            except Exception as e:
                self.last_debug += f"JSON parse err: {e}\n"
                return ([], [])
        except Exception as e:
            self.last_debug += f"Request failed: {e}\n"
            return ([], [])

    def analyze_audio_alt_chunked(
        self,
        audio_path: str,
        chunk_seconds: int,
        sleep_between: int
    ) -> tuple[list[AltWordTimed], list[AltChordTimed]]:
        """Chunked analysis with backoff and stitching absolute times."""
        self.last_debug = ""
        self.last_notes = []
        if not self.api_key:
            self.last_debug = "No API key set"
            return ([], [])
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        y = data.mean(axis=1)
        samples_per_chunk = max(1, int(sr * chunk_seconds))
        words_all: list[AltWordTimed] = []
        chords_all: list[AltChordTimed] = []
        backoffs = [60, 300]
        unavailable_hits = 0
        start = 0
        while start < len(y):
            end = min(len(y), start + samples_per_chunk)
            buf = io.BytesIO()
            sf.write(buf, y[start:end], sr, format="FLAC")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            chunk_idx = (start // samples_per_chunk) + 1
            total_chunks = int(np.ceil(len(y) / samples_per_chunk))
            prompt = (
                f"This is chunk {chunk_idx}/{total_chunks} of the song. Analyze only this chunk.\n"
                "1) Transcribe the lead vocal and rewrite as improved lyrics, with times per word.\n"
                "2) For each word, infer the harmonic chord WITH QUALITY using standard chord symbols "
                "(maj/min/7/maj7/min7/dim/aug/sus/add extensions, alterations, slash bass).\n"
                "Return STRICT JSON with keys 'words' and 'chords'.\n"
                "- words: array of objects with keys: 'text', 'start_sec', 'end_sec'.\n"
                "- chords: array of objects with keys: 'symbol', 'root', 'quality', 'bass', 'start_sec', 'end_sec'.\n"
                "- Ensure both arrays are the same LENGTH and index-aligned.\n"
            )
            # backoff loop
            skipped_this_chunk = False
            while True:
                res = self._post_audio_payload(b64, prompt)
                if not self._is_unavailable(res):
                    break
                if unavailable_hits >= len(backoffs):
                    self.last_debug += f"Skipping chunk {chunk_idx}/{total_chunks}: UNAVAILABLE after backoffs\n"
                    skipped_this_chunk = True
                    break
                delay = backoffs[unavailable_hits]
                unavailable_hits += 1
                time.sleep(delay)
            # parse
            if not skipped_this_chunk:
                try:
                    text = res.get("json", {}).get("candidates", [{}])[0]\
                           .get("content", {}).get("parts", [{}])[0].get("text", "")
                except Exception:
                    text = ""
                if text:
                    import json as _json
                    try:
                        obj = _json.loads(self.strip_code_fences(text))
                        offset = start / sr
                        words_list = obj.get("words", []) or []
                        chords_list = obj.get("chords", []) or []
                        notes_list = obj.get("notes", []) or []
                        m_local = min(len(words_list), len(chords_list))
                        for i in range(m_local):
                            w = words_list[i] or {}
                            c = chords_list[i] or {}
                            words_all.append(
                                AltWordTimed(
                                    text=str(w.get("text", "")),
                                    start=offset + float(w.get("start_sec", 0.0)),
                                    end=offset + float(w.get("end_sec", 0.0)),
                                )
                            )
                            sym = str(c.get("symbol") or "")
                            if not sym:
                                root = str(c.get("root") or "")
                                qual = str(c.get("quality") or "")
                                bass = str(c.get("bass") or "")
                                sym = root + (qual if qual else "") + ("/" + bass if bass else "")
                            chords_all.append(
                                AltChordTimed(
                                    symbol=sym,
                                    start=offset + float(c.get("start_sec", 0.0)),
                                    end=offset + float(c.get("end_sec", 0.0)),
                                )
                            )
                        for n in notes_list:
                            try:
                                self.last_notes.append(
                                    AltNoteTimed(
                                        pitch_midi=int(n.get("pitch_midi")),
                                        start=offset + float(n.get("start_sec", 0.0)),
                                        end=offset + float(n.get("end_sec", 0.0))
                                    )
                                )
                            except Exception:
                                pass
                    except Exception as e:
                        self.last_debug += f"Chunk JSON parse err: {e}\n"
            # advance
            start = end
            if start < len(y):
                time.sleep(max(0, sleep_between))
        m = min(len(words_all), len(chords_all))
        return (words_all[:m], chords_all[:m])

    def _post_audio_payload(self, b64: str, prompt: str) -> dict:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"inline_data": {"mime_type": "audio/flac", "data": b64}},
                        {"text": prompt}
                    ]
                }
            ]
        }
        try:
            resp = requests.post(url, params={"key": self.api_key}, json=payload, timeout=60)
            return {
                "status": resp.status_code,
                "text": resp.text,
                "json": (resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {})
            }
        except Exception as e:
            self.last_debug += f"Request err: {e}\n"
            return {"status": 0, "text": "", "json": {}}

    def _is_unavailable(self, res: dict) -> bool:
        if res.get("status") == 503:
            return True
        data = res.get("json")
        try:
            err = data.get("error") if isinstance(data, dict) else None
            if err and err.get("status") == "UNAVAILABLE":
                return True
        except Exception:
            pass
        return False

    def strip_code_fences(self, s: str) -> str:
        t = s.strip()
        if t.startswith("```"):
            parts = t.split("\n", 1)
            t = parts[1] if len(parts) > 1 else t
            if t.lower().startswith("json\n"):
                parts = t.split("\n", 1)
                t = parts[1] if len(parts) > 1 else t
            if t.endswith("```"):
                t = t[:-3]
        return t.strip()
