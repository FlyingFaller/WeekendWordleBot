"""
Defines the UI components for the Wordle game board.

- LetterSquare: A single, clickable tile representing a letter.
- WordleBoard: A 6x5 container that holds the LetterSquare widgets.
"""
from enum import Enum, auto

from textual.containers import Container
from textual.widgets import Static
from textual.reactive import reactive
from textual.message import Message
from textual.app import ComposeResult

# --- Constants and Enums ---

# Define the color cycle for the letter squares and their character representations
COLORS = ("#3a3a3c", "#bbaf30", "#16ac55")  # Gray, Yellow, Green
COLOR_CHARS = {"-": 0, "g": 2, "y": 1} # Key mapping for pattern input

class GameState(Enum):
    """Enumeration for the current game state."""
    INPUT_WORD = auto()
    INPUT_PATTERN = auto()
    GAME_OVER = auto()

class LetterState(Enum):
    """Enumeration for the state of a single letter tile."""
    EMPTY = auto()
    RECOMMENDATION = auto()
    FILLED = auto()

# --- Widgets ---

class LetterSquare(Static):
    """A clickable, color-changing square for a single letter."""

    class Clicked(Message):
        """Posted when a letter square is clicked."""
        def __init__(self, square: "LetterSquare"):
            self.square = square
            super().__init__()

    # --- Reactive properties to automatically update the UI ---
    letter = reactive(" ")
    color_index = reactive(0)
    letter_state = reactive(LetterState.EMPTY)
    has_focus = reactive(False)

    def __init__(self, row: int, col: int):
        """Initializes the LetterSquare with its grid position."""
        self.row = row
        self.col = col
        super().__init__()

    def watch_color_index(self, new_index: int) -> None:
        """Called when the 'color_index' reactive property changes."""
        self.styles.animate("background", value=COLORS[new_index], duration=0.2)

    def watch_letter_state(self, new_state: LetterState) -> None:
        """Called when the 'letter_state' reactive property changes."""
        self.remove_class(*[s.name.lower() for s in LetterState])
        self.add_class(new_state.name.lower())
        # Refresh the render to potentially change the font
        self.refresh(layout=True)

    def watch_has_focus(self, new_focus: bool) -> None:
        """Toggles the 'focused' CSS class based on the reactive property."""
        self.set_class(new_focus, "focused")

    def on_mount(self) -> None:
        """Set initial styling when the widget is added to the application."""
        self.watch_letter_state(self.letter_state)
        self.watch_color_index(self.color_index)

    def on_click(self) -> None:
        """Handles the click event by posting a message for the app to handle."""
        self.post_message(self.Clicked(self))

    # UPDATED: The render method now delegates to the current SCREEN's text processor
    def render(self) -> str:
        """
        Renders the letter by passing it to the correct text processor from the screen.
        """
        if self.letter == " ":
            return " "
            
        return self.screen.text_processor.process(
            self.letter, self.content_size, self.letter_state
        )


class WordleBoard(Container):
    """A 6x5 grid container that holds the LetterSquare widgets."""
    
    BORDER_TITLE = "Board"

    def compose(self) -> ComposeResult:
        """Creates the 6x5 grid of letter squares."""
        for row in range(6):
            for col in range(5):
                yield LetterSquare(row=row, col=col)
