"""
A standalone demo of a Wordle board UI element using Textual.

This script creates a 6x5 grid of letter squares that dynamically resizes
with the terminal window while maintaining the aspect ratio of the tiles.
Each letter is rendered as ASCII art using the pyfiglet library.

NOTE: You must install pyfiglet for this script to run:
pip install pyfiglet
"""
import random
import string
import pyfiglet
from textual.app import App, ComposeResult
from textual.containers import Container
# We are now using the Static widget for a sleeker, more semantically correct approach.
from textual.widgets import Static
from textual.reactive import reactive
from textual.geometry import Size

# Define the color cycle for the letter squares
COLORS = ("#3a3a3c", "#bbaf30", "#16ac55")  # Gray, Yellow, Green

class LetterSquare(Static):
    """A clickable, color-changing square for a single letter."""

    # By setting ALLOW_SELECT to False, we directly prevent text highlighting.
    ALLOW_SELECT = False

    # A reactive variable to track the current color state.
    # 0: Gray, 1: Yellow, 2: Green
    color_index = reactive(0)

    # Create a single Figlet instance to be reused for efficiency.
    figlet = pyfiglet.Figlet(font='bigfig')

    def __init__(self, letter: str = " "):
        """
        Initializes the LetterSquare.

        Args:
            letter: The letter to display in the square. Defaults to a space.
        """
        super().__init__()
        self.letter = letter
        # Render the ASCII art and strip only trailing newlines to preserve character shape.
        self.ascii_art = self.figlet.renderText(letter).rstrip('\n')

    def on_mount(self) -> None:
        """
        Set initial styling when the widget is added to the application.
        This ensures the square looks correct from the start.
        """
        self.styles.text_style = "bold"
        # Set the initial background color based on the default color_index.
        self.update_color()

    def watch_color_index(self, old_index: int, new_index: int) -> None:
        """
        This function is a "watcher" and is automatically called by Textual
        whenever the `color_index` reactive variable changes. It updates
        the background color of the square.
        """
        self.update_color()

    def update_color(self) -> None:
        """
        Updates the background color of the square based on the current index.
        We use animate to create a smooth transition between colors.
        """
        self.styles.animate("background", value=COLORS[self.color_index], duration=0.2)

    def on_click(self) -> None:
        """
        Handles the click event, cycling through the color states.
        The modulo operator (%) ensures the index wraps around to 0 after the last color.
        """
        self.color_index = (self.color_index + 1) % len(COLORS)

    def render(self) -> str:
        """
        Conditionally render either the ASCII art or a single character
        based on the available height of the tile.
        """
        # Check the available height inside the widget's content area.
        if self.content_size.height >= 5:
            return self.ascii_art
        else:
            return self.letter


class WordleBoard(Container):
    """A 6x5 grid container that holds the LetterSquare widgets."""

    def compose(self) -> ComposeResult:
        """Renders the 30 letter squares into the grid."""
        # Fill the board with squares containing random letters.
        for _ in range(30):
            random_letter = random.choice(string.ascii_uppercase)
            yield LetterSquare(random_letter)


class WordleBoardDemo(App):
    """A Textual application to demonstrate the Wordle board UI."""

    # Link to the external stylesheet
    CSS_PATH = "wordle_board_demo.tcss"

    # Define the aspect ratio for a single tile (width / height)
    TILE_ASPECT_RATIO = 2.0

    def compose(self) -> ComposeResult:
        """Create and return the main widget for the app."""
        yield WordleBoard()

    def on_mount(self) -> None:
        """
        Called when the app is first mounted. We use call_after_refresh
        to schedule the first resize after the initial screen paint,
        guaranteeing the layout is stable.
        """
        self.call_after_refresh(self.on_resize)

    def on_resize(self, event: object = None) -> None:
        """
        Handles the terminal resize event to dynamically adjust the board size.
        """
        # Get the available size of the screen
        available_size = self.screen.content_size

        # Guard against division by zero during initial setup
        if not available_size.height or not available_size.width:
            return

        # Define the grid dimensions and gutter size
        cols = 5
        rows = 6
        gutter = 1
        total_gutter_space = (cols - 1) * gutter

        # --- Calculate the best fit based on integer cell sizes ---

        # 1. Calculate the best size if height is the limiting factor
        cell_height_h = available_size.height // rows
        cell_width_h = int(cell_height_h * self.TILE_ASPECT_RATIO)
        new_width_from_height = (cell_width_h * cols) + total_gutter_space

        # 2. Calculate the best size if width is the limiting factor
        cell_width_w = (available_size.width - total_gutter_space) // cols
        cell_height_w = int(cell_width_w / self.TILE_ASPECT_RATIO)
        new_height_from_width = cell_height_w * rows

        # --- Choose the calculation that fits within the screen ---

        # If the height-limited calculation fits, it's the best option
        if new_width_from_height <= available_size.width and cell_height_h > 0:
            new_width = new_width_from_height
            new_height = cell_height_h * rows
        # Otherwise, use the width-limited calculation
        elif cell_width_w > 0:
            new_width = (cell_width_w * cols) + total_gutter_space
            new_height = new_height_from_width
        # If neither fits (very small screen), do nothing
        else:
            return

        # --- Enforce minimum readable size ---
        # Calculate the cell height from the final board height
        final_cell_height = new_height // rows
        if final_cell_height < 3:
            # If the calculated cell height is too small, fall back to a fixed minimum size.
            min_cell_height = 3
            min_cell_width = int(min_cell_height * self.TILE_ASPECT_RATIO)
            
            new_height = min_cell_height * rows
            new_width = (min_cell_width * cols) + total_gutter_space

        # Apply the new dimensions to the WordleBoard
        board = self.query_one(WordleBoard)
        board.styles.width = new_width
        board.styles.height = new_height


if __name__ == "__main__":
    app = WordleBoardDemo()
    app.run()
