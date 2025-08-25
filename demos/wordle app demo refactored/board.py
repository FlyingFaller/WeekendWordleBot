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

# --- Constants and Enums ---
COLORS = ("#3a3a3c", "#bbaf30", "#16ac55")
COLOR_CHARS = {"-": 0, "g": 2, "y": 1}

class GameState(Enum):
    INPUT_WORD = auto()
    INPUT_PATTERN = auto()
    GAME_OVER = auto()

class LetterState(Enum):
    EMPTY = auto()
    RECOMMENDATION = auto()
    FILLED = auto()

# --- Widgets ---

class LetterSquare(Static):
    """A clickable, color-changing square for a single letter."""

    class Clicked(Message):
        def __init__(self, square: "LetterSquare"):
            self.square = square
            super().__init__()

    letter = reactive(" ")
    color_index = reactive(0)
    letter_state = reactive(LetterState.EMPTY)
    has_focus = reactive(False)

    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col
        super().__init__()

    def watch_color_index(self, new_index: int) -> None:
        self.styles.animate("background", value=COLORS[new_index], duration=0.2)

    def watch_letter_state(self, new_state: LetterState) -> None:
        self.remove_class(*[s.name.lower() for s in LetterState])
        self.add_class(new_state.name.lower())
        self.refresh(layout=True)

    def watch_has_focus(self, new_focus: bool) -> None:
        self.set_class(new_focus, "focused")

    def on_mount(self) -> None:
        self.watch_letter_state(self.letter_state)
        self.watch_color_index(self.color_index)

    def on_click(self) -> None:
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

    # --- Game State Properties ---
    game_state = reactive(GameState.INPUT_WORD)
    current_row = 0
    focused_col = 0
    suggested_word = "SLATE"
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.grid: list[list[LetterSquare]] = [[] for _ in range(6)]
        self.current_word = ""

    def compose(self) -> ComposeResult:
        """Creates the 6x5 grid of letter squares."""
        for row in range(6):
            for col in range(5):
                yield LetterSquare(row=row, col=col)

    def on_mount(self) -> None:
        """Populate the grid reference for easy access."""
        for square in self.query(LetterSquare):
            self.grid[square.row].append(square)
        self.start_turn()

    # --- Public Methods (called by the screen) ---
    def update_suggestion(self, word: str) -> None:
        """Updates the suggested word and refreshes the display."""
        self.suggested_word = word
        self._update_recommendation_display()

    def handle_key(self, event: events.Key) -> None:
        """Processes a key event from the parent screen."""
        if self.game_state == GameState.INPUT_WORD:
            self._handle_word_input(event)
        elif self.game_state == GameState.INPUT_PATTERN:
            self._handle_pattern_input(event)

    # --- Game Logic and State Transitions ---
    def start_turn(self) -> None:
        self.game_state = GameState.INPUT_WORD
        self.current_word = ""
        if self.current_row < 6:
            self._update_recommendation_display()

    def submit_word(self) -> None:
        self.game_state = GameState.INPUT_PATTERN
        for square in self.grid[self.current_row]:
            square.color_index = 0
        self.focused_col = 0
        self.grid[self.current_row][self.focused_col].has_focus = True

    def submit_pattern(self) -> None:
        self.grid[self.current_row][self.focused_col].has_focus = False
        is_win = all(sq.color_index == 2 for sq in self.grid[self.current_row])
        if is_win or self.current_row >= 5:
            self.game_state = GameState.GAME_OVER
            return
        
        self.current_row += 1
        self.start_turn()

    # --- Internal Input Handlers ---
    def _handle_word_input(self, event: events.Key) -> None:
        if event.key == "tab":
            event.prevent_default()
            if self.suggested_word.startswith(self.current_word):
                for i in range(len(self.current_word), 5):
                    self.grid[self.current_row][i].letter = self.suggested_word[i]
                    self.grid[self.current_row][i].letter_state = LetterState.FILLED
                self.current_word = self.suggested_word
        elif event.key == "enter":
            if len(self.current_word) == 5:
                self.submit_word()
        elif event.key == "backspace":
            if self.current_word:
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
        matches = self.suggested_word.startswith(prefix)
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
            self.submit_pattern()
        elif event.key == "right":
            self.focused_col = min(4, self.focused_col + 1)
        elif event.key == "left":
            self.focused_col = max(0, self.focused_col - 1)
        elif event.key == "backspace":
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
