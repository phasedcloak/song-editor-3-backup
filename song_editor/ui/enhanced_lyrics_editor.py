import re
from typing import List, Dict
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel,
    QPushButton, QScrollArea, QSplitter, QCheckBox,
    QSizePolicy, QComboBox
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import (
    QTextCursor, QTextCharFormat, QColor, QFont, QPalette
)

import cmudict
import pronouncing

from ..models.lyrics import WordRow


@dataclass
class RhymeInfo:
    """Information about rhyming words"""
    word: str
    pronunciation: List[str]
    rhyme_type: str  # 'perfect', 'near', 'none'
    rhyme_group: str  # Group identifier for same color


class SyllableCounter:
    """Professional syllable counting using cmudict"""

    def __init__(self):
        self.cmu = cmudict.dict()
        self.cache = {}

    def count_syllables(self, word: str) -> int:
        """Count syllables in a word using cmudict"""
        if word in self.cache:
            return self.cache[word]

        # Clean the word
        clean_word = re.sub(r'[^\w\s]', '', word.lower())

        if clean_word in self.cmu:
            # Get the first pronunciation
            pronunciation = self.cmu[clean_word][0]
            # Count syllables (each vowel sound is a syllable)
            syllable_count = len([p for p in pronunciation if p[-1].isdigit()])
            self.cache[word] = syllable_count
            return syllable_count
        else:
            # Fallback: estimate syllables by counting vowel groups
            vowel_groups = len(re.findall(r'[aeiouy]+', clean_word))
            self.cache[word] = max(1, vowel_groups)
            return max(1, vowel_groups)


class RhymeAnalyzer:
    """Analyze rhyming patterns using pronouncing library"""

    def __init__(self):
        self.cache = {}

    def get_pronunciation(self, word: str) -> List[str]:
        """Get pronunciation for a word"""
        if word in self.cache:
            return self.cache[word]

        clean_word = re.sub(r'[^\w\s]', '', word.lower())
        pronunciation = pronouncing.phones_for_word(clean_word)

        if pronunciation:
            self.cache[word] = pronunciation[0]
            return pronunciation[0]
        else:
            # Fallback: return empty pronunciation
            self.cache[word] = ""
            return ""

    def rhyme_key(self, word: str) -> str:
        """Create a stable rhyme key for a word using CMU phones if available.
        Falls back to a simple textual heuristic to avoid heavy lookups.
        """
        try:
            clean_word = re.sub(r'[^A-Za-z]', '', word.lower())
            if not clean_word:
                return ""
            phones = pronouncing.phones_for_word(clean_word)
            if phones:
                try:
                    # Use pronouncing.rhyming_part if available
                    from pronouncing import rhyming_part
                    key = rhyming_part(phones[0])
                    return key or ""
                except Exception:
                    pass
            # Fallback: last stressed-ish vowel cluster + coda (very rough)
            vowels = "aeiouy"
            reversed_word = clean_word[::-1]
            cluster = []
            for ch in reversed_word:
                cluster.append(ch)
                if ch in vowels:
                    # take 3 letters of tail once we saw a vowel
                    if len(cluster) >= 3:
                        break
            return ''.join(cluster)[::-1]
        except Exception:
            return ""

    def near_rhyme_key(self, word: str) -> str:
        """Create a near-rhyme key using final vowel sound (ignoring stress)."""
        try:
            clean_word = re.sub(r'[^A-Za-z]', '', word.lower())
            if not clean_word:
                return ""

            # Try to get pronunciation first
            phones = pronouncing.phones_for_word(clean_word)
            if phones:
                # Extract the last vowel sound from pronunciation
                phone_list = phones[0].split()

                # Find the last vowel sound in the word
                for phone in reversed(phone_list):
                    if any(char.isdigit() for char in phone):  # Vowel sound
                        # Remove stress markers for comparison
                        vowel_clean = ''.join(c for c in phone if not c.isdigit())
                        return vowel_clean

            # Fallback: last vowel in the word
            vowels = "aeiouy"
            for i in range(len(clean_word) - 1, -1, -1):
                if clean_word[i] in vowels:
                    return clean_word[i]
            return ""
        except Exception:
            return ""

    def are_perfect_rhymes(self, word1: str, word2: str) -> bool:
        """Check if two words are perfect rhymes"""
        if word1 == word2:
            return False

        # Get rhymes for word1 and check if word2 is in the list
        rhymes_list = pronouncing.rhymes(word1)
        return word2.lower() in [r.lower() for r in rhymes_list]

    def are_near_rhymes(self, word1: str, word2: str) -> bool:
        """Check if two words are near rhymes (assonance) - simplified and more accurate"""
        if word1 == word2:
            return False

        # Get pronunciations
        pron1 = pronouncing.phones_for_word(word1)
        pron2 = pronouncing.phones_for_word(word2)

        if not pron1 or not pron2:
            return False

        # Use a simpler approach: check if they share the same final stressed vowel
        # This is more reliable than complex stress pattern matching
        try:
            # Get the rhyming parts
            rhyme1 = pronouncing.rhyming_part(pron1[0])
            rhyme2 = pronouncing.rhyming_part(pron2[0])

            # If they have the same rhyming part, they're perfect rhymes, not near rhymes
            if rhyme1 == rhyme2 and rhyme1:
                return False

            # For near rhymes, check if they end with similar vowel sounds
            # Extract the last vowel sound from each pronunciation
            phones1 = pron1[0].split()
            phones2 = pron2[0].split()

            last_vowel1 = None
            last_vowel2 = None

            # Find the last vowel sound in each word
            for phone in reversed(phones1):
                if any(char.isdigit() for char in phone):  # Vowel sound
                    last_vowel1 = phone
                    break

            for phone in reversed(phones2):
                if any(char.isdigit() for char in phone):  # Vowel sound
                    last_vowel2 = phone
                    break

            # Check if they have the same final vowel sound (ignoring stress)
            if last_vowel1 and last_vowel2:
                # Remove stress markers for comparison
                vowel1_clean = ''.join(c for c in last_vowel1 if not c.isdigit())
                vowel2_clean = ''.join(c for c in last_vowel2 if not c.isdigit())

                # Only consider them near rhymes if they have the same final vowel
                # AND they're not already perfect rhymes
                if vowel1_clean == vowel2_clean:
                    # Additional check: make sure they're not too similar
                    # If they share the same rhyme key, they're perfect rhymes, not near rhymes
                    key1 = self.rhyme_key(word1)
                    key2 = self.rhyme_key(word2)
                    if key1 == key2 and key1:
                        return False
                    return True

            return False

        except Exception:
            return False

    def find_rhymes(self, target_word: str, word_list: List[str]) -> Dict[str, List[str]]:
        """Find perfect and near rhymes for a target word"""
        perfect_rhymes = []
        near_rhymes = []

        for word in word_list:
            if word.lower() == target_word.lower():
                continue

            if self.are_perfect_rhymes(target_word, word):
                perfect_rhymes.append(word)
            elif self.are_near_rhymes(target_word, word):
                near_rhymes.append(word)

        return {
            'perfect': perfect_rhymes,
            'near': near_rhymes
        }

    def dict_perfect_rhymes(self, target_word: str) -> List[str]:
        """Return perfect rhymes from the CMU dict via pronouncing.rhymes"""
        clean_word = re.sub(r'[^\w\s]', '', target_word.lower())
        try:
            return pronouncing.rhymes(clean_word)
        except Exception:
            return []

    def dict_near_rhymes(self, target_word: str) -> List[str]:
        """Return near rhymes using stress pattern similarity from CMU dict"""
        clean_word = re.sub(r'[^\w\s]', '', target_word.lower())
        try:
            stresses_list = pronouncing.stresses_for_word(clean_word)
            if not stresses_list:
                return []
            stress = stresses_list[0]
            candidates = pronouncing.search_stresses(stress)
            perfect = set(w.lower() for w in pronouncing.rhymes(clean_word))
            result = []
            for w in candidates:
                wl = w.lower()
                if wl == clean_word:
                    continue
                if wl in perfect:
                    continue
                result.append(w)
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for w in result:
                if w.lower() in seen:
                    continue
                seen.add(w.lower())
                deduped.append(w)
            return deduped
        except Exception:
            return []


class AudioPlaybackThread(QThread):
    """Thread for playing audio segments"""

    playback_finished = Signal()

    def __init__(self, audio_path: str, start_time: float, duration: float):
        super().__init__()
        self.audio_path = audio_path
        self.start_time = start_time
        self.duration = duration
        self.player = None

    def run(self):
        """Play audio segment"""
        try:
            from ..core.audio_player import AudioPlayer
            self.player = AudioPlayer()
            self.player.load_audio(self.audio_path)
            self.player.play_segment(self.start_time, self.duration)

            # Wait for playback to finish
            import time
            time.sleep(self.duration)

        except Exception as e:
            print(f"Audio playback error: {e}")
        finally:
            if self.player:
                self.player.stop()
            self.playback_finished.emit()

    def stop_playback(self):
        """Stop playback"""
        if self.player:
            self.player.stop()
        self.playback_finished.emit()


class SyllablePanel(QWidget):
    """Left panel showing syllable counts for each line"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.syllable_counter = SyllableCounter()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header
        header = QLabel("Syllables")
        header.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #333;
                padding: 5px;
                background-color: #f0f0f0;
                border-radius: 3px;
            }
        """)
        layout.addWidget(header)

        # Spacer to align the top of counts with the top of the QTextEdit area (controls height)
        self.top_spacer = QWidget()
        self.top_spacer.setFixedHeight(0)
        layout.addWidget(self.top_spacer)

        # Scroll area for syllable counts (synchronized with text editor)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Hide scroll bar
        self.scroll_area.setAlignment(Qt.AlignTop)
        self.scroll_area.setMaximumWidth(100)

        # Container for syllable labels
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setSpacing(2)  # Small spacing for better visual separation
        self.container_layout.setContentsMargins(5, 0, 5, 5)
        # Ensure items stack from top to bottom and are aligned to top
        try:
            from PySide6.QtWidgets import QBoxLayout
            self.container_layout.setDirection(QBoxLayout.TopToBottom)
        except Exception:
            pass
        self.container_layout.setAlignment(Qt.AlignTop)

        self.scroll_area.setWidget(self.container)
        layout.addWidget(self.scroll_area)

        # Set fixed width (keep it compact)
        self.setMaximumWidth(100)
        self.setMinimumWidth(60)

    def update_counts(self, lyrics_text: str):
        """Update syllable counts for the lyrics"""
        # Clear existing labels
        for i in reversed(range(self.container_layout.count())):
            child = self.container_layout.itemAt(i)
            if child.widget():
                child.widget().deleteLater()

        if not lyrics_text.strip():
            return

        lines = lyrics_text.split('\n')

        for line in lines:
            if not line.strip():
                # Empty line - create a spacer to maintain alignment
                label = QLabel("")
                label.setFixedHeight(18)
                label.setFixedWidth(40)
                label.setStyleSheet("""
                    QLabel {
                        background-color: transparent;
                        border: none;
                        margin: 0px;
                    }
                """)
                self.container_layout.addWidget(label)
                continue

            # Count syllables in this line (exclude chord annotations)
            # Simple string replacement to remove chord annotations like [C], [Am], etc.
            line_without_chords = line
            while '[' in line_without_chords and ']' in line_without_chords:
                start = line_without_chords.find('[')
                end = line_without_chords.find(']', start)
                if end == -1:
                    break
                line_without_chords = line_without_chords[:start] + line_without_chords[end+1:]

            # Simple word extraction without regex
            words = []
            for word in line_without_chords.lower().split():
                # Clean word of punctuation
                clean_word = ''.join(c for c in word if c.isalpha())
                if clean_word and len(clean_word) > 0:
                    words.append(clean_word)

            # Filter out common chord names that might be mistaken for words
            chord_names = {'c', 'g', 'd', 'a', 'e', 'b', 'f', 'am', 'em', 'bm', 'dm', 'gm', 'cm', 'fm'}
            words = [word for word in words if word not in chord_names]
            total_syllables = sum(self.syllable_counter.count_syllables(word) for word in words)

            # Create label with height matching text editor line height
            label = QLabel(f"{total_syllables}")
            label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
            label.setStyleSheet("""
                QLabel {
                    background-color: #e8f4f8;
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    padding: 2px 6px;
                    font-weight: bold;
                    color: #333;
                    margin: 0px;
                    min-height: 18px;
                    max-height: 18px;
                }
            """)
            # Set a consistent height that matches typical text editor line height
            label.setFixedHeight(18)
            label.setFixedWidth(40)  # Fixed width for consistent alignment
            self.container_layout.addWidget(label)

        # Ensure scroll area shows from top and no extra stretch pushes content
        # Reset scroll to top
        if hasattr(self, 'scroll_area'):
            self.scroll_area.verticalScrollBar().setValue(0)

    def sync_syllable_scroll(self, value):
        """Synchronize syllable panel scrolling with text editor"""
        if hasattr(self, 'scroll_area'):
            self.scroll_area.verticalScrollBar().setValue(value)

    def set_top_offset(self, pixels: int):
        """Set a fixed spacer above the counts to align with the editor's controls row."""
        if hasattr(self, 'top_spacer'):
            self.top_spacer.setFixedHeight(max(0, pixels))


class RhymePanel(QWidget):
    """Right panel showing rhyming suggestions"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rhyme_analyzer = RhymeAnalyzer()
        self.current_word = ""
        self.all_words = []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header
        header = QLabel("Rhymes")
        header.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #333;
                padding: 5px;
                background-color: #f0f0f0;
                border-radius: 3px;
            }
        """)
        layout.addWidget(header)

        # Current word display
        self.current_word_label = QLabel("Select a word")
        self.current_word_label.setStyleSheet("""
            QLabel {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 3px;
                padding: 5px;
                font-weight: bold;
            }
        """)
        self.current_word_label.setWordWrap(True)
        layout.addWidget(self.current_word_label)

        # Perfect rhymes section
        perfect_header = QLabel("Perfect Rhymes:")
        perfect_header.setStyleSheet("font-weight: bold; color: #28a745;")
        layout.addWidget(perfect_header)

        self.perfect_rhymes_text = QTextEdit()
        self.perfect_rhymes_text.setMaximumHeight(100)
        self.perfect_rhymes_text.setReadOnly(True)
        self.perfect_rhymes_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.perfect_rhymes_text.setStyleSheet("""
            QTextEdit {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.perfect_rhymes_text)

        # Near rhymes section
        near_header = QLabel("Near Rhymes:")
        near_header.setStyleSheet("font-weight: bold; color: #ffc107;")
        layout.addWidget(near_header)

        self.near_rhymes_text = QTextEdit()
        self.near_rhymes_text.setMaximumHeight(100)
        self.near_rhymes_text.setReadOnly(True)
        self.near_rhymes_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.near_rhymes_text.setStyleSheet("""
            QTextEdit {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.near_rhymes_text)

        # Allow panel to expand with window width
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumWidth(150)

    def update_rhymes(self, target_word: str, all_words: List[str]):
        """Update rhyming suggestions for the target word"""
        self.current_word = target_word
        self.all_words = all_words

        if not target_word:
            self.current_word_label.setText("Select a word")
            self.perfect_rhymes_text.clear()
            self.near_rhymes_text.clear()
            return

        self.current_word_label.setText(f"Word: {target_word}")

        # Dictionary-based rhymes (CMU dict via pronouncing)
        perfect_rhymes = self.rhyme_analyzer.dict_perfect_rhymes(target_word)
        if perfect_rhymes:
            self.perfect_rhymes_text.setPlainText(', '.join(perfect_rhymes))
        else:
            self.perfect_rhymes_text.setPlainText("None found")

        # Dictionary-based near rhymes (stress-pattern similarity)
        near_rhymes = self.rhyme_analyzer.dict_near_rhymes(target_word)
        if near_rhymes:
            self.near_rhymes_text.setPlainText(', '.join(near_rhymes))
        else:
            self.near_rhymes_text.setPlainText("None found")


class EnhancedLyricsEditor(QWidget):
    """Enhanced lyrics editor with multi-line support, syllable counting, and rhyming"""

    lyrics_changed = Signal(str)
    play_audio_requested = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_path = None
        self.lyrics_data = []
        self.rhyme_analyzer = RhymeAnalyzer()
        self.syllable_counter = SyllableCounter()
        self.playback_thread = None
        self.color_mode = "confidence"  # "confidence" or "rhyme"
        self.rhyme_groups = {}
        self.near_rhyme_groups = {}
        self._rhyme_key_cache = {}
        self._near_key_cache = {}
        self._updating_text = False  # Flag to prevent recursion
        # Debounce timer for heavy analysis/formatting
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(250)
        self._debounce_timer.timeout.connect(self._analyze_and_color)
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Create splitter for three panels
        self.splitter = QSplitter(Qt.Horizontal)

        # Left panel: Syllable counts
        self.syllable_panel = SyllablePanel()
        self.splitter.addWidget(self.syllable_panel)

        # Center panel: Main lyrics editor
        self.lyrics_panel = self.create_lyrics_panel()
        self.splitter.addWidget(self.lyrics_panel)

        # Right panel: Rhyming suggestions
        self.rhyme_panel = RhymePanel()
        self.splitter.addWidget(self.rhyme_panel)

        # Set initial splitter proportions (center gets most space)
        self.splitter.setSizes([80, 500, 250])
        # Set stretch factors: center gets most space, sides are fixed
        self.splitter.setStretchFactor(0, 0)  # Left panel fixed width
        self.splitter.setStretchFactor(1, 1)  # Center panel stretches
        self.splitter.setStretchFactor(2, 0)  # Right panel fixed width
        # Prevent center pane from collapsing
        try:
            self.splitter.setCollapsible(0, True)
            self.splitter.setCollapsible(1, False)
            self.splitter.setCollapsible(2, True)
        except Exception:
            pass

        layout.addWidget(self.splitter)

        # Ensure all widgets are visible
        self.syllable_panel.show()
        self.lyrics_panel.show()
        self.rhyme_panel.show()
        self.splitter.show()

        # Connect text editor scrolling to syllable panel (after UI is set up)
        self.text_edit.verticalScrollBar().valueChanged.connect(self.on_text_scroll)

    def create_lyrics_panel(self):
        """Create the main lyrics editing panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)

        # Controls
        controls_widget = QWidget()
        controls_widget.setFixedHeight(50)  # Ensure controls have minimum height
        controls_widget.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
                padding: 5px;
            }
        """)
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(5, 5, 5, 5)

        # Color mode toggle
        self.color_mode_checkbox = QCheckBox("🎨 Color by Rhymes")
        self.color_mode_checkbox.setStyleSheet("font-weight: bold; color: #495057; font-size: 12px;")
        self.color_mode_checkbox.toggled.connect(self.on_color_mode_changed)
        controls_layout.addWidget(self.color_mode_checkbox)

        # Play button
        self.play_button = QPushButton("▶ Play")
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.play_button.clicked.connect(self.play_current_selection)
        controls_layout.addWidget(self.play_button)

        # Font controls
        controls_layout.addWidget(QLabel("Font:"))
        self.font_combo = QComboBox()
        self.font_combo.addItems([
            "Arial", "Helvetica", "Times New Roman", "Courier New",
            "Verdana", "Georgia", "Palatino"
        ])
        self.font_combo.setCurrentText("Arial")
        self.font_combo.currentTextChanged.connect(self.on_font_changed)
        controls_layout.addWidget(self.font_combo)

        controls_layout.addWidget(QLabel("Size:"))
        self.font_size_combo = QComboBox()
        self.font_size_combo.addItems(["10", "12", "14", "16", "18", "20", "24", "28", "32"])
        self.font_size_combo.setCurrentText("14")
        self.font_size_combo.currentTextChanged.connect(self.on_font_size_changed)
        controls_layout.addWidget(self.font_size_combo)

        controls_layout.addStretch()
        layout.addWidget(controls_widget)

        # No top offset; we align counts to top of editor area

        # Main text editor
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Enter lyrics here...")
        self.text_edit.textChanged.connect(self.on_text_changed)
        self.text_edit.mouseDoubleClickEvent = self.on_double_click
        # Ensure visible and readable text (explicit palette)
        try:
            pal = self.text_edit.palette()
            pal.setColor(QPalette.Base, QColor(255, 255, 255))
            pal.setColor(QPalette.Text, QColor(0, 0, 0))
            self.text_edit.setPalette(pal)
        except Exception:
            pass
        self.text_edit.setVisible(True)
        # Ensure the editor expands and has a reasonable minimum size
        try:
            self.text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception:
            pass
        self.text_edit.setMinimumSize(300, 200)
        # Set size policy to expand but not squeeze out controls
        self.text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Connect resize event to handle auto-wrapping
        self.text_edit.resizeEvent = self.on_text_edit_resize
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Arial', sans-serif;
                font-size: 14px;
                line-height: 1.5;
            }
            QTextEdit:focus {
                border: 2px solid #0078d4;
            }
        """)
        layout.addWidget(self.text_edit)

        return panel

    def set_audio_path(self, audio_path: str):
        """Set the audio file path for playback"""
        self.audio_path = audio_path

    def set_song_data(self, song_data):
        """Set song data and update display (compatible with main window)"""
        # Convert SongData.words to WordRow format expected by enhanced editor
        from ..models.lyrics import WordRow

        lyrics_data = []
        for word in song_data.words:
            word_row = WordRow(
                text=word.text,
                start=word.start,
                end=word.end,
                confidence=word.confidence,
                chord=getattr(word, 'chord', None),
                alt_text=word.alternatives[0] if word.alternatives else None
            )
            lyrics_data.append(word_row)

        self.set_lyrics_data(lyrics_data)

    def set_lyrics_data(self, lyrics_data: List[WordRow]):
        """Set lyrics data and update display"""
        self.lyrics_data = lyrics_data

        # Convert to text format
        text_lines = []
        current_line = []

        for word in lyrics_data:
            # Add chord if available
            word_text = word.text
            if word.chord:
                word_text += f"[{word.chord}]"

            current_line.append(word_text)

            # Check for line break (either from punctuation or stored line_break flag)
            should_break = (word.text.endswith(('.', '!', '?', ':', ';')) or
                            getattr(word, 'line_break', False))

            if should_break:
                text_lines.append(' '.join(current_line))
                current_line = []

        # Add any remaining words
        if current_line:
            text_lines.append(' '.join(current_line))

        # Set text in editor (prevent recursion)
        self._updating_text = True
        final_text = '\n'.join(text_lines)
        self.text_edit.setPlainText(final_text)
        self._updating_text = False

        # Update syllable counts
        self.syllable_panel.update_counts('\n'.join(text_lines))
        # Ensure syllable panel scrolls to top initially
        self.syllable_panel.sync_syllable_scroll(0)

        # Apply auto-wrapping after setting text (with a small delay)
        QTimer.singleShot(100, self.apply_auto_wrapping)

        # Analyze rhymes for coloring (debounced)
        self._debounce_timer.start(10)

        # Connect resize event to handle auto-wrapping after data is loaded
        self.text_edit.resizeEvent = self.on_text_edit_resize

    def on_text_changed(self):
        """Handle text changes"""
        # Prevent recursion when programmatically updating text
        if self._updating_text:
            return

        text = self.text_edit.toPlainText()
        self.lyrics_changed.emit(text)

        # Update syllable counts (without triggering text changes)
        self.syllable_panel.update_counts(text)
        # Keep scroll positions in sync proportionally
        self.syllable_panel.sync_syllable_scroll(self.text_edit.verticalScrollBar().value())

        # Debounce rhyme analysis + coloring
        self._debounce_timer.start(250)

    def _reset_formatting(self):
        """Reset all text formatting to default before applying new coloring."""
        cursor = self.text_edit.textCursor()
        cursor.beginEditBlock()
        cursor.select(QTextCursor.Document)
        default_format = QTextCharFormat()
        cursor.setCharFormat(default_format)
        cursor.clearSelection()
        cursor.endEditBlock()

    def _analyze_and_color(self):
        """Analyze rhymes and apply coloring according to current mode."""
        self.analyze_rhymes()
        self._reset_formatting()
        self.apply_coloring()

    def analyze_rhymes(self):
        """Analyze rhyming patterns using pronunciation-based grouping with fallbacks."""
        text = self.text_edit.toPlainText()
        # Remove chord annotations like [C]
        clean_text = text
        while '[' in clean_text and ']' in clean_text:
            start = clean_text.find('[')
            end = clean_text.find(']', start)
            if end == -1:
                break
            clean_text = clean_text[:start] + clean_text[end+1:]

        # Simple word extraction
        words = []
        for word in clean_text.lower().split():
            cleaned = ''.join(c for c in word if c.isalpha())
            if cleaned:
                words.append(cleaned)

        unique_words = list(dict.fromkeys(words))

        # Initialize groups
        self.rhyme_groups = {}
        self.near_rhyme_groups = {}

        # Use proper rhyme analysis - no arbitrary grouping
        remaining_words = unique_words

        # Build perfect rhyme groups by rhyme_key for remaining words
        key_to_words = {}
        for w in remaining_words:
            if w in self._rhyme_key_cache:
                key = self._rhyme_key_cache[w]
            else:
                key = self.rhyme_analyzer.rhyme_key(w)
                self._rhyme_key_cache[w] = key
            if not key:
                continue
            key_to_words.setdefault(key, []).append(w)

        group_id = 0  # Start from 0 since we removed manual groups
        for key, group_words in key_to_words.items():
            if len(group_words) < 2:
                continue
            group_id += 1
            group_name = f"group_{group_id}"
            for w in group_words:
                self.rhyme_groups[w] = group_name

        # Near rhyme groups by near_rhyme_key for remaining words
        near_key_to_words = {}
        for w in remaining_words:
            if w in self.rhyme_groups:  # Skip words already in perfect rhyme groups
                continue
            if w in self._near_key_cache:
                nkey = self._near_key_cache[w]
            else:
                nkey = self.rhyme_analyzer.near_rhyme_key(w)
                self._near_key_cache[w] = nkey
            if not nkey:
                continue
            near_key_to_words.setdefault(nkey, []).append(w)

        near_id = 0  # Start from 0 since we removed manual groups
        for nkey, group_words in near_key_to_words.items():
            if len(group_words) < 2:
                continue
            near_id += 1
            group_name = f"near_{near_id}"
            for w in group_words:
                self.near_rhyme_groups[w] = group_name

    def apply_coloring(self):
        """Apply color coding based on current mode"""
        if self.color_mode == "confidence":
            self.apply_confidence_coloring()
        else:
            self.apply_rhyme_coloring()

    def apply_confidence_coloring(self):
        """Apply confidence-based color coding"""
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.Start)

        for word_data in self.lyrics_data:
            # Calculate color based on confidence
            confidence = word_data.confidence
            red = int(255 * (1.0 - confidence))
            green = int(255 * confidence)
            color = QColor(red, green, 0)

            # Create format
            format = QTextCharFormat()
            format.setForeground(color)

            # Find and format the word
            word_text = word_data.text
            if word_data.chord:
                word_text += f"[{word_data.chord}]"

            # Search for the word and apply formatting
            search_cursor = self.text_edit.document().find(word_text, cursor)
            if not search_cursor.isNull():
                search_cursor.mergeCharFormat(format)

    def apply_rhyme_coloring(self):
        """Apply rhyme-based color coding. Perfect groups are bold; near groups not bold."""
        colors = [
            QColor(255, 0, 0), QColor(0, 128, 0), QColor(0, 0, 200), QColor(200, 120, 0),
            QColor(128, 0, 128), QColor(200, 0, 100), QColor(0, 160, 160), QColor(160, 160, 0),
        ]
        doc = self.text_edit.document()

        # First, set all words to black (default for non-rhyming words)
        black_fmt = QTextCharFormat()
        black_fmt.setForeground(QColor(0, 0, 0))
        black_fmt.setFontWeight(QFont.Normal)

        # Get all words from the text
        text = self.text_edit.toPlainText()
        all_words = []
        for line in text.split('\n'):
            for word in line.split():
                # Clean word of punctuation and brackets
                clean_word = ''.join(c for c in word if c.isalpha())
                if clean_word:
                    all_words.append(clean_word.lower())

        # Set all words to black first
        for word in set(all_words):
            cursor = doc.find(word)
            while not cursor.isNull():
                cursor.mergeCharFormat(black_fmt)
                cursor = doc.find(word, cursor)

        # Apply perfect rhyme groups (bold)
        group_to_words = {}
        for w, g in self.rhyme_groups.items():
            group_to_words.setdefault(g, []).append(w)

        for i, (group_name, words) in enumerate(group_to_words.items()):
            color = colors[i % len(colors)]
            fmt = QTextCharFormat()
            fmt.setForeground(color)
            fmt.setFontWeight(QFont.Bold)
            for w in words:
                cursor = doc.find(w)
                while not cursor.isNull():
                    cursor.mergeCharFormat(fmt)
                    cursor = doc.find(w, cursor)

        # Apply near rhyme groups (same palette, not bold)
        near_group_to_words = {}
        for w, g in self.near_rhyme_groups.items():
            near_group_to_words.setdefault(g, []).append(w)

        for i, (group_name, words) in enumerate(near_group_to_words.items()):
            color = colors[i % len(colors)]
            fmt = QTextCharFormat()
            fmt.setForeground(color)
            fmt.setFontWeight(QFont.Normal)
            for w in words:
                cursor = doc.find(w)
                while not cursor.isNull():
                    cursor.mergeCharFormat(fmt)
                    cursor = doc.find(w, cursor)

    def on_text_scroll(self, value):
        """Handle text editor scrolling and sync with syllable panel"""
        if hasattr(self, 'syllable_panel'):
            self.syllable_panel.sync_syllable_scroll(value)

    def on_color_mode_changed(self, checked: bool):
        """Handle color mode toggle"""
        self.color_mode = "rhyme" if checked else "confidence"
        self.apply_coloring()

    def on_double_click(self, event):
        """Handle double-click to play audio"""
        cursor = self.text_edit.cursorForPosition(event.pos())
        cursor.select(QTextCursor.WordUnderCursor)
        word = cursor.selectedText()

        # Find the word in lyrics data and play audio
        for word_data in self.lyrics_data:
            if word_data.text.lower() == word.lower():
                start_time = word_data.start
                end_time = word_data.end
                duration = end_time - start_time
                self.play_audio_requested.emit(start_time, duration)
                break

        # Also update rhyme panel
        self.update_rhyme_panel(word)

        # Call parent's double-click handler
        super().mouseDoubleClickEvent(event)

    def update_rhyme_panel(self, word: str):
        """Update the rhyme panel with suggestions for the selected word"""
        text = self.text_edit.toPlainText()
        all_words = re.findall(r'\b\w+\b', text.lower())
        self.rhyme_panel.update_rhymes(word, all_words)

    def play_current_selection(self):
        """Play audio for the currently selected text"""
        cursor = self.text_edit.textCursor()
        if cursor.hasSelection():
            selected_text = cursor.selectedText()
            # Find the corresponding audio segment
            for word_data in self.lyrics_data:
                if word_data.text in selected_text:
                    start_time = word_data.start
                    end_time = word_data.end
                    duration = end_time - start_time
                    self.play_audio_requested.emit(start_time, duration)
                    break

    def get_lyrics_text(self) -> str:
        """Get the current lyrics text"""
        return self.text_edit.toPlainText()

    def set_font(self, font: QFont):
        """Set font for the text editor"""
        self.text_edit.setFont(font)

    def on_font_changed(self, font_name: str):
        """Handle font family change"""
        self.apply_font_settings()

    def on_font_size_changed(self, font_size: str):
        """Handle font size change"""
        self.apply_font_settings()

    def apply_font_settings(self):
        """Apply current font settings"""
        font_name = self.font_combo.currentText()
        font_size = int(self.font_size_combo.currentText())

        # Create font
        font = QFont(font_name, font_size)
        self.text_edit.setFont(font)

        # Re-apply auto-wrapping after font change
        self.apply_auto_wrapping()

    def on_text_edit_resize(self, event):
        """Handle text editor resize to re-apply auto-wrapping"""
        # Call the original resize event
        super(QTextEdit, self.text_edit).resizeEvent(event)
        # Apply auto-wrapping after resize
        self.apply_auto_wrapping()

    def apply_auto_wrapping(self):
        """Automatically insert line breaks when text wraps"""
        if not self.lyrics_data:
            return

        # Get current text and document
        text = self.text_edit.toPlainText()

        # Don't apply auto-wrapping if there's no text yet
        if not text.strip():
            return

        # Get the width of the text editor (minus margins)
        editor_width = self.text_edit.viewport().width() - 20  # Account for margins

        # Don't apply if editor width is too small
        if editor_width < 100:
            return

        # Get font metrics for accurate width calculation
        font_metrics = self.text_edit.fontMetrics()

        # Process each line to check for wrapping
        lines = text.split('\n')
        new_lines = []

        for line in lines:
            if not line.strip():
                new_lines.append(line)
                continue

            # Check if this line would wrap
            words = line.split()
            if not words:
                new_lines.append(line)
                continue

            # Build line word by word to check width
            current_line = []
            current_width = 0

            for word in words:
                # Calculate actual word width using font metrics
                word_width = font_metrics.horizontalAdvance(word)
                space_width = font_metrics.horizontalAdvance(' ')

                if current_width + word_width > editor_width and current_line:
                    # This word would cause wrapping, insert line break before it
                    new_lines.append(' '.join(current_line))
                    current_line = [word]
                    current_width = word_width
                else:
                    current_line.append(word)
                    current_width += word_width + space_width

            # Add the last line
            if current_line:
                new_lines.append(' '.join(current_line))

        # Update the text if changes were made
        new_text = '\n'.join(new_lines)
        if new_text != text:
            # Prevent recursion
            self._updating_text = True
            self.text_edit.setPlainText(new_text)
            self._updating_text = False

            # Update the lyrics data with line break information
            self.update_lyrics_data_with_line_breaks(new_text)

    def update_lyrics_data_with_line_breaks(self, text: str):
        """Update lyrics data to include line break information"""
        lines = text.split('\n')
        word_index = 0

        for line in lines:
            words_in_line = line.split()
            for i, word in enumerate(words_in_line):
                if word_index < len(self.lyrics_data):
                    # Check if this word should have a line break after it
                    is_last_word_in_line = (i == len(words_in_line) - 1)
                    self.lyrics_data[word_index].line_break = is_last_word_in_line
                word_index += 1

    def merge_lines(self, line1_index: int, line2_index: int):
        """Merge two lines into one"""
        text = self.text_edit.toPlainText()
        lines = text.split('\n')

        if 0 <= line1_index < len(lines) and 0 <= line2_index < len(lines):
            # Merge the lines
            merged_line = lines[line1_index].strip() + ' ' + lines[line2_index].strip()

            # Remove the second line and update the first
            lines[line1_index] = merged_line
            lines.pop(line2_index)

            # Update text (prevent recursion)
            self._updating_text = True
            self.text_edit.setPlainText('\n'.join(lines))
            self._updating_text = False

            # Update syllable counts
            self.syllable_panel.update_counts('\n'.join(lines))

            # Re-analyze rhymes
            self.analyze_rhymes()
            self.apply_coloring()

    def split_line(self, line_index: int, word_index: int):
        """Split a line at a specific word"""
        text = self.text_edit.toPlainText()
        lines = text.split('\n')

        if 0 <= line_index < len(lines):
            line = lines[line_index]
            words = line.split()

            if 0 <= word_index < len(words):
                # Split the line at the word
                first_part = ' '.join(words[:word_index])
                second_part = ' '.join(words[word_index:])

                # Replace the line with two new lines
                lines[line_index] = first_part
                lines.insert(line_index + 1, second_part)

                # Update text (prevent recursion)
                self._updating_text = True
                self.text_edit.setPlainText('\n'.join(lines))
                self._updating_text = False

                # Update syllable counts
                self.syllable_panel.update_counts('\n'.join(lines))

                # Re-analyze rhymes
                self.analyze_rhymes()
                self.apply_coloring()

    def add_chord_to_word(self, word: str, chord: str):
        """Add or update a chord for a specific word"""
        text = self.text_edit.toPlainText()

        # Find the word and add/update chord
        import re
        pattern = rf'\b{re.escape(word)}\b(?:\[[^\]]*\])?'
        replacement = f"{word}[{chord}]"

        new_text = re.sub(pattern, replacement, text)

        if new_text != text:
            # Update text (prevent recursion)
            self._updating_text = True
            self.text_edit.setPlainText(new_text)
            self._updating_text = False

            # Update syllable counts
            self.syllable_panel.update_counts(new_text)

            # Re-analyze rhymes
            self.analyze_rhymes()
            self.apply_coloring()

    def remove_chord_from_word(self, word: str):
        """Remove chord from a specific word"""
        text = self.text_edit.toPlainText()

        # Find the word with chord and remove it
        import re
        pattern = rf'\b{re.escape(word)}\[([^\]]*)\]'
        replacement = word

        new_text = re.sub(pattern, replacement, text)

        if new_text != text:
            # Update text (prevent recursion)
            self._updating_text = True
            self.text_edit.setPlainText(new_text)
            self._updating_text = False

            # Update syllable counts
            self.syllable_panel.update_counts(new_text)

            # Re-analyze rhymes
            self.analyze_rhymes()
            self.apply_coloring()
