"""
Defines the UI components and logic for the Wordle game board.
"""
from enum import Enum, auto

from textual.containers import Container
from textual.widgets import Static
from textual.reactive import reactive
from textual.message import Message
from textual.app import ComposeResult
from textual import events
from weekend_wordle.config import GRAY, YELLOW, GREEN, APP_COLORS

# --- Constants and Enums ---
COLORS = (APP_COLORS['tile-gray'], APP_COLORS['tile-yellow'], APP_COLORS['tile-green'])
COLOR_CHARS = {"-": GRAY, "g": GREEN, "y": YELLOW}
PATTERN_MAP = {GRAY: 0, YELLOW: 1, GREEN: 2}

class GameState(Enum):
    INPUT_WORD = auto()
    INPUT_PATTERN = auto()
    GAME_OVER = auto()
    COMPUTING = auto() # State for when the backend is working

class LetterState(Enum):
    EMPTY = auto()
    RECOMMENDATION = auto()
    FILLED = auto()

# --- Custom Messages ---
class WordSubmitted(Message):
    """Posted when the user submits a word."""
    def __init__(self, word: str):
        super().__init__()
        self.word = word

class PatternSubmitted(Message):
    """Posted when the user submits a pattern."""
    def __init__(self, pattern: list[int]):
        super().__init__()
        self.pattern = pattern

# --- Widgets ---

class LetterSquare(Static):
    """A clickable, color-changing square for a single letter."""
    class Clicked(Message):
        def __init__(self, square: "LetterSquare"):
            super().__init__()
            self.square = square

    letter = reactive(" ")
    color_index = reactive(0)
    letter_state = reactive(LetterState.EMPTY)
    has_focus = reactive(False)

    def __init__(self, row: int, col: int):
        super().__init__()
        self.row = row
        self.col = col

    def watch_color_index(self, new_index: int) -> None:
        self.styles.animate("background", value=COLORS[new_index], duration=0.2)

    def watch_letter_state(self, new_state: LetterState) -> None:
        self.remove_class(*[s.name.lower() for s in LetterState])
        self.add_class(new_state.name.lower())

    def watch_has_focus(self, new_focus: bool) -> None:
        self.set_class(new_focus, "focused")

    def on_mount(self) -> None:
        self.watch_letter_state(self.letter_state)
        self.watch_color_index(self.color_index)

    def on_click(self) -> None:
        if self.parent.game_state == GameState.INPUT_PATTERN:
            self.post_message(self.Clicked(self))

    def render(self) -> str:
        if self.letter == " ":
            return " "
        return self.screen.text_processor.process(
            self.letter, self.content_size, self.letter_state
        )

class WordleBoard(Container):
    """A 6x5 grid container that holds game logic and state."""
    BORDER_TITLE = "Board"

    game_state = reactive(GameState.INPUT_WORD)
    current_row = 0
    focused_col = 0
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.grid: list[list[LetterSquare]] = [[] for _ in range(6)]
        self.current_word = ""
        self.suggested_word = ""

    def compose(self) -> ComposeResult:
        for row in range(6):
            for col in range(5):
                yield LetterSquare(row=row, col=col)

    def on_mount(self) -> None:
        for square in self.query(LetterSquare):
            self.grid[square.row].append(square)
        self.start_turn()

    # --- Public Methods (called by the screen) ---
    def update_suggestion(self, word: str) -> None:
        """Updates the suggested word to be displayed."""
        self.suggested_word = word.upper()
        if self.game_state == GameState.INPUT_WORD:
            self._update_recommendation_display()
    
    def handle_key(self, event: events.Key) -> None:
        if self.game_state == GameState.INPUT_WORD:
            self._handle_word_input(event)
        elif self.game_state == GameState.INPUT_PATTERN:
            self._handle_pattern_input(event)

    def undo_row(self) -> None:
        """Undoes the last action, either reverting a pattern input or a full guess."""
        # Case 1: A word was submitted, but not a pattern. Revert to word input on the same row.
        if self.game_state == GameState.INPUT_PATTERN:
            self.game_state = GameState.INPUT_WORD
            self.current_word = ""
            # Reset colors and remove focus from the pattern input.
            for square in self.grid[self.current_row]:
                square.color_index = 0
            if self.grid[self.current_row]:
                self.grid[self.current_row][self.focused_col].has_focus = False
            self.focused_col = 0
            # Restore the recommendation display for the now-empty row.
            self._update_recommendation_display()

        # Case 2: A full guess was made. Clear the previous row and the current suggestion row.
        elif self.current_row > 0:
            row_to_undo_idx = self.current_row - 1

            # Clear the contents of the row that was just completed.
            for square in self.grid[row_to_undo_idx]:
                square.letter = " "
                square.letter_state = LetterState.EMPTY
                square.color_index = 0

            # Also clear the current row, which is showing stale suggestions.
            if self.current_row < len(self.grid):
                for square in self.grid[self.current_row]:
                    square.letter = " "
                    square.letter_state = LetterState.EMPTY
                    square.color_index = 0
            
            # Reset the state to point to the row that was just undone.
            self.current_row = row_to_undo_idx
            self.game_state = GameState.INPUT_WORD
            self.current_word = ""

    def switch_to_pattern_input(self) -> None:
        """Called by the screen when a valid word is submitted."""
        self.game_state = GameState.INPUT_PATTERN
        for square in self.grid[self.current_row]:
            square.color_index = 0
        self.focused_col = 0
        if self.current_row < 6 and self.focused_col < 5:
            self.grid[self.current_row][self.focused_col].has_focus = True

    def end_turn(self, is_win: bool) -> None:
        """Finalizes the current row and sets state to COMPUTING, but does not advance the row."""
        if self.focused_col < 5:
             self.grid[self.current_row][self.focused_col].has_focus = False

        if is_win or self.current_row >= 5:
            self.game_state = GameState.GAME_OVER
        else:
            self.game_state = GameState.COMPUTING

    def advance_row_and_start_turn(self) -> None:
        """Advances the row counter and prepares the new row for input."""
        if self.game_state != GameState.GAME_OVER:
            self.current_row += 1
            self.start_turn()

    def start_turn(self) -> None:
        """Prepares the CURRENT row for word input and displays the suggestion."""
        if self.game_state != GameState.GAME_OVER:
            self.game_state = GameState.INPUT_WORD
            self.current_word = ""
            if self.current_row < 6:
                self._update_recommendation_display()

    # --- Internal Handlers ---
    def _handle_word_input(self, event: events.Key) -> None:
        if event.key == "tab":
            event.prevent_default()
            if self.suggested_word and self.suggested_word.startswith(self.current_word):
                for i in range(len(self.current_word), 5):
                    self.grid[self.current_row][i].letter = self.suggested_word[i]
                    self.grid[self.current_row][i].letter_state = LetterState.FILLED
                self.current_word = self.suggested_word
        elif event.key == "enter":
            if len(self.current_word) == 5 or len(self.current_word) == 0:
                self.post_message(WordSubmitted(self.current_word))
        elif event.key == "backspace":
            if self.current_word:
                col_to_clear = len(self.current_word) - 1
                self.grid[self.current_row][col_to_clear].letter = " "
                self.grid[self.current_row][col_to_clear].letter_state = LetterState.EMPTY
                self.current_word = self.current_word[:-1]
                self._update_recommendation_display()
        elif event.is_printable and len(self.current_word) < 5:
            if event.character and event.character.isalpha():
                char_upper = event.character.upper()
                col = len(self.current_word)
                square = self.grid[self.current_row][col]
                square.letter = char_upper
                square.letter_state = LetterState.FILLED
                self.current_word += char_upper
                self._update_recommendation_display()

    def _update_recommendation_display(self) -> None:
        prefix = self.current_word
        matches = self.suggested_word.startswith(prefix) if self.suggested_word else False
        for col in range(len(prefix), 5):
            square = self.grid[self.current_row][col]
            if matches:
                square.letter = self.suggested_word[col]
                square.letter_state = LetterState.RECOMMENDATION
            else:
                square.letter = " "
                square.letter_state = LetterState.EMPTY

    def _handle_pattern_input(self, event: events.Key) -> None:
        old_col = self.focused_col
        if event.key == "enter":
            pattern = [PATTERN_MAP[sq.color_index] for sq in self.grid[self.current_row]]
            self.post_message(PatternSubmitted(pattern))
        elif event.key == "right":
            self.focused_col = min(4, self.focused_col + 1)
        elif event.key == "left":
            self.focused_col = max(0, self.focused_col - 1)
        elif event.key == "backspace":
            # Just move left on backspace, don't change color
            self.focused_col = max(0, self.focused_col - 1)
        elif event.character and event.character.lower() in COLOR_CHARS:
            square = self.grid[self.current_row][self.focused_col]
            square.color_index = COLOR_CHARS[event.character.lower()]
            self.focused_col = min(4, self.focused_col + 1)
        
        if old_col != self.focused_col:
            self.grid[self.current_row][old_col].has_focus = False
            self.grid[self.current_row][self.focused_col].has_focus = True

    def on_letter_square_clicked(self, message: LetterSquare.Clicked) -> None:
        if (self.game_state == GameState.INPUT_PATTERN and
                message.square.row == self.current_row):
            square = message.square
            self.grid[self.current_row][self.focused_col].has_focus = False
            self.focused_col = square.col
            self.grid[self.current_row][self.focused_col].has_focus = True
            square.color_index = (square.color_index + 1) % len(COLORS)

