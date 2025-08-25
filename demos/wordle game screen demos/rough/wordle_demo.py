import asyncio
import random
import string
from textual.app import App, ComposeResult
from textual.color import Color
from textual.containers import Container, Horizontal
from textual.css.query import NoMatches
from textual.events import Resize
from textual.widgets import DataTable, Footer, Header, Static, ProgressBar

class Letter(Static):
    """A single letter tile in the Wordle grid."""

    def on_mount(self) -> None:
        """Set the default background color on mount."""
        self.styles.background = "grey"
        # Also set the initial color here to override auto-contrast at startup
        self.styles.color = "white"

    def on_click(self) -> None:
        """Cycle through colors on click: grey -> yellow -> green -> grey."""
        current_color = self.styles.background

        if current_color == Color.parse("grey"):
            self.styles.background = "yellow"
        elif current_color == Color.parse("yellow"):
            self.styles.background = "green"
        else: # Covers green and any other state
            self.styles.background = "grey"
        
        # THE FIX: Re-apply the white color immediately after changing the background.
        # This overrides the automatic contrast adjustment.
        self.styles.color = "white"


class WordleRow(Container):
    """A row of letter tiles in the Wordle grid."""
    
    def compose(self) -> ComposeResult:
        for _ in range(5):
            yield Letter(" ")

class WordleGrid(Container):
    """The 6x5 Wordle grid."""
    
    def compose(self) -> ComposeResult:
        for _ in range(6):
            yield WordleRow()

class WordleDemoApp(App):
    """A Textual app to demonstrate a Wordle-style UI."""
    
    CSS_PATH = "wordle_demo.tcss"
    
    def __init__(self):
        super().__init__()
        self.current_row = 0
        self.current_col = 0
        self.suggestion = ""

    def compose(self) -> ComposeResult:
        """Create the layout of the app."""
        with Horizontal():
            with Container(id="left-panel"):
                yield DataTable(id="suggestion-table")
            with Container(id="right-panel"):
                yield WordleGrid()
        yield ProgressBar(total=100, id="progress-bar")
        yield Footer()

    async def on_mount(self) -> None:
        """Set up the app after mounting."""
        self.query_one(WordleGrid).border_title = "Wordle Grid"
        self.query_one("#left-panel").border_title = "Suggestions"
        
        table = self.query_one(DataTable)
        table.add_columns("Word", "Score")
        
        # Simulate loading suggestions
        await self.update_suggestions()
        
        # Start progress bar simulation
        self.set_interval(0.1, self.update_progress)

        # Set the initial size of the grid
        self._update_grid_layout()

    def _update_grid_layout(self) -> None:
        """Calculates and applies the optimal size for the Wordle grid letters."""
        try:
            # Query the container for the grid and the grid itself
            right_panel = self.query_one("#right-panel")
            grid = self.query_one(WordleGrid)
        except NoMatches:
            # This can happen if the widgets aren't mounted yet, so we exit early.
            return

        # Get the available space inside the container (accounting for padding/border)
        available_width = right_panel.content_size.width
        available_height = right_panel.content_size.height

        # --- Aspect Ratio Calculation ---
        # To make a cell look square in a terminal, its width needs to be roughly
        # double its height. Let's use a 2:1 width-to-height ratio.
        # Let 'h' be the letter height and 'w' be the letter width, so w = 2h.
        #
        # The grid has 5 letters and 4 gaps (1 cell each) horizontally.
        # Total width = 5 * w + 4  => 5 * (2h) + 4 => 10h + 4
        #
        # The grid has 6 letters and 5 gaps (1 cell each) vertically.
        # Total height = 6 * h + 5
        
        # Calculate the maximum possible height based on both width and height constraints.
        h_from_width = (available_width - 4) / 10
        h_from_height = (available_height - 5) / 6

        # To fit the grid, we must satisfy both constraints. We choose the smaller of the
        # two calculated heights and convert it to an integer.
        letter_height = int(min(h_from_width, h_from_height))

        # Ensure the letters have at least a minimum size to remain visible.
        if letter_height < 1:
            letter_height = 1
        
        # The width is double the height to create a square appearance.
        letter_width = letter_height * 2

        # Apply the newly calculated size to every letter in the grid.
        for letter in grid.query(Letter):
            letter.styles.width = letter_width
            letter.styles.height = letter_height

    def on_resize(self, event: Resize) -> None:
        """Handle the resize event for the app."""
        self._update_grid_layout()

    async def update_suggestions(self):
        """Simulate generating and displaying word suggestions."""
        table = self.query_one(DataTable)
        table.clear()
        
        suggestions = [
            ("".join(random.choices(string.ascii_uppercase, k=5)), f"{random.uniform(1, 5):.2f}")
            for _ in range(10)
        ]
        
        for word, score in suggestions:
            table.add_row(word, score)
            
        if suggestions:
            self.suggestion = suggestions[0][0]
            self.update_suggestion_in_grid()

    def update_suggestion_in_grid(self):
        """Display the top suggestion in the current row."""
        if self.current_row < 6:
            row = self.query(WordleRow)[self.current_row]
            letters = row.query(Letter)
            for i in range(5):
                if i < len(self.suggestion):
                    letters[i].update(self.suggestion[i])
                    # Force the color to be white
                    letters[i].styles.color = "white"
                else:
                    letters[i].update(" ")

    def update_progress(self) -> None:
        """Advance the progress bar."""
        progress_bar = self.query_one(ProgressBar)
        if progress_bar.progress < 100:
            progress_bar.advance(1)
        else:
            progress_bar.progress = 0
            
    async def on_key(self, event) -> None:
        """Handle key presses."""
        if 'a' <= event.key <= 'z' and self.current_col < 5:
            row = self.query(WordleRow)[self.current_row]
            letter_widget = row.query(Letter)[self.current_col]
            letter_widget.update(event.key.upper())
            # Force the color to be white
            letter_widget.styles.color = "white"
            self.current_col += 1
            
        elif event.key == "backspace" and self.current_col > 0:
            self.current_col -= 1
            row = self.query(WordleRow)[self.current_row]
            letter_widget = row.query(Letter)[self.current_col]
            letter_widget.update(" ")
            
        elif event.key == "enter" and self.current_col == 5:
            self.current_row += 1
            self.current_col = 0
            if self.current_row < 6:
                await self.update_suggestions()
                
        elif event.key == "tab" and self.suggestion:
            row = self.query(WordleRow)[self.current_row]
            letters = row.query(Letter)
            for i, char in enumerate(self.suggestion):
                letters[i].update(char.upper())
                # Force the color to be white
                letters[i].styles.color = "white"
            self.current_col = 5

if __name__ == "__main__":
    WordleDemoApp().run()
