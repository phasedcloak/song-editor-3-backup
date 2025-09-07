### iOS Song Editor – Full Implementation Specification

This document defines everything required to implement a self‑contained iOS app that replicates the current macOS Python app’s features, but using native iOS tech. It covers functional behavior, data/JSON schema, processing pipeline, UI/UX, performance, and implementation details suitable for a single‑shot implementation.

## 1. Goals and Scope
- Offline, on‑device processing of an audio file into:
  - Transcribed words with timing and confidences
  - Chords with timing
  - Melody notes with timing and pitch
- Source separation (vocals/instrumental) to improve transcription accuracy and derive chord track
- Editor UI with:
  - Lyrics editing pane with intelligent wrapping, rhyme analysis, syllable counts, and display modes (Enhanced/CCLI)
  - Chord and Melody panels
  - Status bar with processing progress and summary stats
- Export: CCLI text, JSON (master format), MIDI
- Persistence: open audio, auto‑load previously processed `*.song_data.json`

Out of scope in v1 (optional later):
- Batch processing of directories
- External plugins (e.g., Vamp/Chordino)
- Network/cloud dependency (operate fully offline)

## 2. Target Platform and Stack
- iOS 16+ (iPadOS supported; iPhone allowed but UI optimized for iPad)
- Language: Swift 5.9+
- UI: SwiftUI (preferred) + UIKit wrappers where needed (e.g., `UITextView` for rich editing)
- Audio: AVFoundation/AVAudioEngine for I/O; Accelerate/vDSP for DSP; CoreAudio if low‑level needed
- ML inference:
  - Transcription: Core ML Whisper (converted or use `whisper.cpp` Core ML variant); fp16; on ANE/GPU where possible
  - Source separation: Core ML MDX/Demucs variant (converted to Core ML) or “good enough” spectral masking model; fallback: skip separation
  - Melody: Basic Pitch converted to TFLite (via TensorFlow Lite Metal Delegate) or Core ML; fallback: CREPE TFLite
- Rendering/perf: Metal for potential DSP acceleration (optional)
- Export:
  - JSON: Foundation JSONEncoder/Decoder
  - MIDI: CoreMIDI or Swift packages (e.g., SwiftMIDI)
  - Text files: standard file APIs
- Storage: App sandbox `Documents/` for inputs/outputs; `FileProvider` + `UIDocumentPicker` for import/export

## 3. Top‑Level Architecture
- App Modules:
  - AudioProcessingService: load/denoise/normalize/separate
  - TranscriptionService: Whisper Core ML
  - ChordDetectionService: chromagram + template matching
  - MelodyExtractionService: Basic Pitch (TFLite/Core ML)
  - LyricsEngine: syllable count (CMU fallback), rhyme analysis (pronouncing-like logic)
  - UI: SwiftUI views (Lyrics, Chords, Melody) + side panels
  - Persistence: SongDataStore (read/write master JSON; auto‑load)
  - Exporters: CCLIExporter, MIDIExporter, JSONExporter
- Orchestration:
  - ProcessingPipeline (Combine/async/await): sequential stages with progress updates and cancellation
- Concurrency:
  - Heavy tasks on background queues; Core ML set to .highPrecision where needed; expose progress in UI

## 4. Data Model and JSON Specification

Master JSON file `SongData` (UTF‑8, LF line endings). Fields marked “required” must be present.

### 4.1 JSON Schema (informal)

- SongData (object, required):
  - metadata (object, required):
    - version (string, default "3.0.0")
    - created_at (ISO 8601 string)
    - source_audio (string, absolute or app-relative path)
    - processing_tool (string, e.g., "Song Editor iOS")
    - transcription (object):
      - engine (string, e.g., "whisper-coreml")
      - model (string, e.g., "tiny/base/small/…")
      - alternatives_count (int)
    - audio_processing (object):
      - use_demucs (bool)
      - demucs_model (string, e.g., "htdemucs"/"bag_of_models")
      - denoise (bool)
      - normalize (bool)
  - audio_analysis (object, optional):
    - duration (float seconds)
    - sample_rate (int)
    - channels (int)
    - audio_levels (object):
      - rms (float)
      - rms_db (float)
      - peak (float)
      - peak_db (float)
      - dynamic_range_db (float)
      - crest_factor (float)
    - tempo (float bpm) [required by product]
    - key (object):
      - key (string, e.g., "C", "D#")
      - mode (string, "major"/"minor")
      - confidence (float)
  - words (array<Word>, required)
  - chords (array<Chord>, required)
  - notes (array<Note>, required)
  - segments (array<object>, optional, reserved)

- Word (object, required):
  - text (string, required)
  - start (float seconds, required)
  - end (float seconds, required)
  - confidence (float 0..1, required)
  - alternatives (array<string>, optional)
  - chord (string, optional)  // inline chord tag if set
  - line_break (bool, optional; default false)  // newline marker after this word

- Chord (object, required):
  - symbol (string, required) // e.g., "G", "Am7", "D/F#"
  - start (float, required)
  - end (float, required)
  - root (string, optional)
  - quality (string, optional)
  - bass (string, optional)
  - duration (float, optional)
  - confidence (float, optional)
  - detection_method (string, optional)

- Note (object, required):
  - pitch_midi (int, required)
  - start (float, required)
  - end (float, required)
  - pitch_name (string, optional) // e.g., "C#4"
  - duration (float, optional)
  - velocity (int 0..127, optional)
  - confidence (float, optional)
  - detection_method (string, optional)

### 4.2 JSON Examples
- Provide minimal and full examples with 2 lines of words encoding `line_break: true` on last word of each line.
- Ensure idempotent save/load (round‑trips preserve line breaks).

## 5. Processing Pipeline Behavior

### 5.1 Load
- Import audio via file picker or “Open In…”
- Convert to mono float32 44.1kHz for downstream ML (keep stereo for separation if needed)
- Store original metadata (sampleRate, duration, channels)

### 5.2 Denoise
- Optional light stationary noise reduction (vDSP spectral gate or simplified Wiener)
- Parameter: `denoise_strength` (0..1); default 0.5

### 5.3 Normalize
- Target LUFS −23.0; measure integrated loudness (EBU R128 approximation)
- Apply linear gain with headroom (cap at 0.95 peak)

### 5.4 Source Separation
- Preferred: Core ML model approximating HTDemucs/MDX (4 stems: drums, bass, other, vocals)
- Minimal viable: two stems (vocals/other) separation model
- Use Metal/ANE; fallback: skip separation (use original for both tracks)
- Save separated WAVs into `Documents/separated/<baseName>/`

### 5.5 Transcription (Whisper)
- Use Core ML Whisper (tiny/base/small depending on device)
- Transcribe from vocals stem if available; else original
- Produce segments and words with start/end and probability
- Filter words by confidence threshold (default 0.5–0.7)
- Debounce remove zero/negative duration and repeated tail duplicates

### 5.6 Chord Detection
- Compute chromagram (CQT) on “other” stem or original
- Template matching for major/minor triads (extendable to sevenths)
- Per-frame best label; median filtering; merge consecutive identical labels
- Drop segments < 250 ms by default (configurable)
- Output `Chord` list with start/end/symbol and confidence

### 5.7 Melody Extraction
- Basic Pitch: run Core ML/TFLite model on vocals; extract notes
- Post‑filter by min duration (default 0.1s), pitch range (C2..C6), confidence
- Map Hz→MIDI, derive note names

### 5.8 Tempo/Key
- Tempo via librosa‑YIN equivalent: autocorrelation/tempo estimation; return bpm
- Key via mean chroma profile; major/minor likelihood; key and confidence

### 5.9 Assemble SongData
- Use `line_break` on last word per visual line; persist

## 6. UI/UX Behavior

### 6.1 Main Layout
- Tabs: Lyrics (default), Chords, Melody
- Left control panel:
  - File info: file name, duration
  - Processing options: Whisper model size; Chord method (Chromagram – default); Melody (Basic Pitch)
  - Use Separation toggle (default on)
  - Save Intermediates toggle (optional)
  - Song Information (Title/Artist/Album/Genre)
  - Statistics (Words/Chords/Notes)
- Status bar: progress and completion summary (Words/Chords/Notes)

### 6.2 Lyrics Tab
- Three‑pane: Syllables (left narrow), Editor (center), Rhymes (right)
- Editor display modes:
  - CCLI (default): `[Chord]word` inline; show chord only when it changes along a line
  - Enhanced: chords appended to words as `[C]`, same grouping logic
- Intelligent wrapping:
  - Internal wrapper: wrap width = editor content width (viewport − 2×padding − 2×document margins − vertical scrollbar)
  - Do not rely on UITextView auto wrap for logic; line breaks in data preserve when saving via `line_break`
- Syllable panel:
  - One label per visual line; height equals editor line spacing; aligned to editor content top
  - Scroll sync proportionally with editor
- Rhyme panel:
  - “Select a word” single row
  - “Perfect Rhymes:” and “Near Rhymes:” headings (12pt)
  - Text areas list words, read‑only
- Rhyme analysis:
  - Perfect: group by rhyme key (CMU’s rhyming part approximation)
  - Near: last vowel match ignoring stress; exclude perfect overlaps
  - Apply colors (bold for perfect groups)
- Syllable counting:
  - Use CMU dictionary (embedded resource) when available; fallback vowel grouping heuristic
- Double‑tap word: play its audio segment from vocals stem

### 6.3 Chords Tab
- Timeline list of chords with start times
- Simple transport to audition segments (play from chord start)

### 6.4 Melody Tab
- Piano roll visualization or list of notes
- Playback preview of selected note range

### 6.5 File Loading/Existing Data
- On selecting audio: if `<name>.song_data.json` exists alongside, auto‑load it (no modal), populate panels and status bar summary

### 6.6 Exports
- CCLI text to file: one line per lyric line with inline chord tags; no duplicate chord when same as previous
- MIDI: lyrics to syllabic track (optional), chords as markers, melody as note events
- JSON: master `SongData` spec above

### 6.7 Accessibility
- Dynamic Type scaling
- VoiceOver labels for key controls
- High‑contrast color option for rhymes

### 6.8 Internationalization
- Strings localized; English default
- Whisper language auto or manual select (future)

## 7. Detailed Module Specs

### 7.1 AudioProcessingService
- API:
  - func process(audioURL: URL, options: AudioProcessingOptions) async throws -> ProcessedAudio
- Options: useSeparation (Bool), saveIntermediates (Bool), demucsModel (String)
- Output: ProcessedAudio { monoBuffer, sampleRate, duration, stems: [String: URL], analysis: AudioAnalysis, timings: StageTimings }
- Timings: demucs, write_sep, tempo, key, intermed_total

### 7.2 TranscriptionService
- API:
  - func transcribe(audioURLOrBuffer, modelSize) async throws -> [Word]
- Filters out zero/neg duration, deduplicates end repeats

### 7.3 ChordDetectionService
- API:
  - func detectChords(audioURL, config) async throws -> [Chord]
- Steps: CQT chromagram, template match, median filter, merge, trim short

### 7.4 MelodyExtractionService
- API:
  - func extractMelody(audioURL, config) async throws -> [Note]
- Steps: model inference, post‑filter

### 7.5 LyricsEngine
- func applyCCLI(words: [Word], chords: [Chord]) -> [DisplayLine]
  - Inline chords only when changes; per line
- func syllableCount(lineText: String) -> Int
- func rhymeGroups(words: [String]) -> (perfectGroups: [[String]], nearGroups: [[String]])

### 7.6 Persistence
- SongDataStore:
  - load(from: URL) throws -> SongData
  - save(_ data: SongData, to: URL) throws
- Auto‑round‑trip `line_break`

### 7.7 Exporters
- CCLIExporter.save(SongData, to: URL)
- MIDIExporter.save(SongData, to: URL)
- JSONExporter.save(SongData, to: URL)

## 8. Performance and Resource Use
- Use Core ML compute units: .all (ANE/GPU/CPU); measure memory; chunk inference where needed
- Avoid copies; stream to disk for large stems
- Debounce UI re‑rendering; avoid heavy work on every keystroke
- Log wrap width, DPI, computed content width in DEBUG builds only

## 9. Error Handling and UX
- Processing progress updates per stage; cancel button disables new jobs until cancellation completes
- Friendly messages on model load failure; continue with degraded path (e.g., skip separation)
- Auto‑save `*.song_data.json` on completion; allow manual export via share sheet

## 10. Security and Privacy
- All processing on‑device; no network required
- Use app sandbox Documents; support delete/cache cleanup
- No PII collected

## 11. Testing
- Unit tests: syllable counting, rhyme grouping, chord merge logic, JSON round‑trip with `line_break`
- Integration tests: pipeline with short fixtures
- UI tests: wrapping, alignment, scroll sync
- Performance tests: 1‑minute and 8‑minute files

## 12. Build & Models
- Provide scripts/notebooks to convert:
  - Whisper model to Core ML (or ship prebuilt)
  - Basic Pitch to TFLite/Core ML
  - Separation model to Core ML
- App bundles models via .mlmodelc / .tflite in app; version pin in `metadata`

## 13. Future Enhancements
- Cloud/remote pipeline fallback
- Rich chord qualities (7ths, sus, add, slash)
- Multi‑language Whisper
- Real‑time monitoring mic input

## 14. Deliverables
- Xcode project with Swift/SwiftUI code
- ML models packaged
- Complete README (build steps, model conversion notes)
- Test suite and sample audio
- Sample `SongData` JSON files

---

### JSON Example (trimmed)

```json
{
  "metadata": {
    "version": "3.0.0",
    "created_at": "2025-09-06T21:59:00Z",
    "source_audio": "Documents/songs/test_short.wav",
    "processing_tool": "Song Editor iOS",
    "transcription": { "engine": "whisper-coreml", "model": "tiny", "alternatives_count": 5 },
    "audio_processing": { "use_demucs": true, "demucs_model": "htdemucs", "denoise": true, "normalize": true }
  },
  "audio_analysis": {
    "duration": 59.9,
    "sample_rate": 44100,
    "channels": 1,
    "audio_levels": { "rms": 0.08, "rms_db": -22.0, "peak": 0.92, "peak_db": -0.7, "dynamic_range_db": 21.3, "crest_factor": 11.5 },
    "tempo": 92.0,
    "key": { "key": "G", "mode": "major", "confidence": 0.71 }
  },
  "words": [
    { "text": "see", "start": 0.50, "end": 0.64, "confidence": 0.94 },
    { "text": "your", "start": 0.64, "end": 0.78, "confidence": 0.92 },
    { "text": "love", "start": 0.78, "end": 0.95, "confidence": 0.90, "chord": "G", "line_break": true },
    { "text": "it's", "start": 1.10, "end": 1.25, "confidence": 0.91 },
    { "text": "clear", "start": 1.25, "end": 1.45, "confidence": 0.89 },
    { "text": "as", "start": 1.45, "end": 1.53, "confidence": 0.88 },
    { "text": "day", "start": 1.53, "end": 1.75, "confidence": 0.90, "line_break": true }
  ],
  "chords": [
    { "symbol": "G", "start": 0.0, "end": 2.0, "confidence": 0.8 },
    { "symbol": "Em", "start": 2.0, "end": 3.0, "confidence": 0.7 }
  ],
  "notes": [
    { "pitch_midi": 67, "start": 0.60, "end": 0.95, "pitch_name": "G4", "velocity": 100, "confidence": 0.8, "detection_method": "basic_pitch" }
  ],
  "segments": []
}
```


