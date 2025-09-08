from __future__ import annotations
from copy import deepcopy

from textual.containers import Container
from textual.widgets import Static
from textual.reactive import reactive
from textual.message import Message
from textual.app import ComposeResult

from weekend_wordle.config import APP_COLORS
# Import state definitions from the new central file
from .state import BoardState, GameState, LetterState, CHAR_TO_COLOR_INT

# --- Constants ---
COLORS = (APP_COLORS['tile-gray'], APP_COLORS['tile-yellow'], APP_COLORS['tile-green'])

# --- Widgets ---

class LetterSquare(Static):
    """A pure view component for a single letter. Its state is entirely controlled by the parent."""
    class Clicked(Message):
        def __init__(self, square: "LetterSquare"):
            super().__init__()
            self.square = square

    letter = reactive(" ")
    color_index = reactive(0)
    letter_state = reactive(LetterState.EMPTY)
    has_focus = reactive(False)

    ALLOW_SELECT = False

    def __init__(self, row: int, col: int):
        super().__init__()
        self.row = row
        self.col = col

    def watch_color_index(self, new_index: int):
        self.styles.animate("background", value=COLORS[new_index], duration=0.1)

    def watch_letter_state(self, new_state: LetterState):
        self.remove_class(*[s.name.lower() for s in LetterState])
        self.add_class(new_state.name.lower())

    def watch_has_focus(self, new_focus: bool):
        self.set_class(new_focus, "focused")

    def on_mount(self) -> None:
        self.watch_letter_state(self.letter_state)
        self.watch_color_index(self.color_index)

    def on_click(self) -> None:
        self.post_message(self.Clicked(self))

    def render(self) -> str:
        return self.screen.text_processor.process(
            self.letter, self.content_size, self.letter_state
        )

class WordleBoard(Container):
    """A pure 'view' component that renders a BoardState object and reports user interactions."""
    BORDER_TITLE = "Board"
    
    class CellClicked(Message):
        """Posted when the user clicks a letter square."""
        def __init__(self, row: int, col: int):
            super().__init__()
            self.row = row
            self.col = col

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.grid: list[list[LetterSquare]] = [[] for _ in range(6)]
        self._last_rendered_state: BoardState | None = None

    def compose(self) -> ComposeResult:
        """Creates the 6x5 grid of letter squares."""
        for row in range(6):
            for col in range(5):
                yield LetterSquare(row=row, col=col)

    def on_mount(self) -> None:
        """Populates the grid reference."""
        for square in self.query(LetterSquare):
            self.grid[square.row].append(square)

    def on_letter_square_clicked(self, message: LetterSquare.Clicked) -> None:
        """Reports a cell click to the parent screen without any internal logic."""
        self.post_message(self.CellClicked(message.square.row, message.square.col))

    def render_state(self, new_state: BoardState) -> None:
        """Intelligently updates the board's view to match the given state."""
        old_state = self._last_rendered_state or BoardState()

        for r in range(6):
            # Determine the state for the entire row
            is_active_row = (r == new_state.active_row)
            
            # Get data for completed rows
            completed_word, completed_pattern = None, None
            if r < len(new_state.guesses):
                completed_word, completed_pattern = new_state.guesses[r]
            
            old_completed_word, old_completed_pattern = None, None
            if r < len(old_state.guesses):
                old_completed_word, old_completed_pattern = old_state.guesses[r]

            # Determine if this row needs a full redraw based on mode/data changes
            needs_redraw = (
                is_active_row != (r == old_state.active_row) or
                (is_active_row and new_state.mode != old_state.mode) or
                completed_word != old_completed_word or
                completed_pattern != old_completed_pattern
            )
            
            for c in range(5):
                square = self.grid[r][c]
                
                # --- Determine new desired state for this square ---
                letter, color_idx, l_state, focus = " ", 0, LetterState.EMPTY, False

                if completed_word: # This is a completed row
                    letter = completed_word[c].upper()
                    color_idx = CHAR_TO_COLOR_INT.get(completed_pattern[c], 0)
                    l_state = LetterState.FILLED
                elif is_active_row and new_state.mode != GameState.IDLE:
                    if new_state.mode == GameState.INPUT_WORD:
                        if c < len(new_state.active_word):
                            letter = new_state.active_word[c]
                            l_state = LetterState.FILLED
                        elif new_state.suggestion.startswith(new_state.active_word):
                            letter = new_state.suggestion[c]
                            l_state = LetterState.RECOMMENDATION
                    elif new_state.mode == GameState.INPUT_PATTERN:
                        letter = new_state.active_word[c].upper()
                        l_state = LetterState.FILLED
                        color_idx = CHAR_TO_COLOR_INT.get(new_state.active_pattern[c], 0)
                        if c == new_state.focused_col:
                            focus = True
                
                # --- Compare to old state and update only if changed ---
                if needs_redraw or square.letter != letter: square.letter = letter
                if needs_redraw or square.color_index != color_idx: square.color_index = color_idx
                if needs_redraw or square.letter_state != l_state: square.letter_state = l_state
                if needs_redraw or square.has_focus != focus: square.has_focus = focus
        
        # Cache the new state for the next render call
        self._last_rendered_state = deepcopy(new_state)

