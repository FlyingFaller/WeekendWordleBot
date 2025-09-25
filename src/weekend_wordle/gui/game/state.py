from dataclasses import dataclass, field
from enum import Enum, auto
from ...config import GRAY, YELLOW, GREEN

# --- Enums ---
class GameState(Enum):
    """Defines the primary UI state of the game."""
    INPUT_WORD = auto()
    INPUT_PATTERN = auto()
    IDLE = auto()
    GAME_OVER = auto()

class LetterState(Enum):
    """Defines the visual state of a single letter square."""
    EMPTY = auto()
    RECOMMENDATION = auto()
    FILLED = auto()

# --- State Object ---
@dataclass
class BoardState:
    """A single object describing the complete visual state of the Wordle board."""
    guesses: list[tuple[str, str]] = field(default_factory=list)
    active_row: int = 0
    active_word: str = ""
    active_pattern: str = ""
    suggestion: str = ""
    mode: GameState = GameState.INPUT_WORD
    focused_col: int = 0

# --- Centralized Mappings ---
COLOR_INT_TO_CHAR = {GRAY: "-", GREEN: "g", YELLOW: "y"}
CHAR_TO_COLOR_INT = {v: k for k, v in COLOR_INT_TO_CHAR.items()}
CHAR_CYCLE = {"-": "y", "y": "g", "g": "-"}

