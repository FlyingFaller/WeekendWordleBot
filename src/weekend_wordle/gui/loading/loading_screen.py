"""
Defines the LoadingScreen for the Wordle Solver application.

This screen provides visual feedback to the user while backend assets are
being loaded and processed.
"""
from textual.screen import Screen
from textual.widgets import Header, Footer, Static
from textual.app import ComposeResult
from weekend_wordle.gui.game.game_screen import GameScreen

class LoadingScreen(Screen):
    """A screen to display while the backend is loading data."""
    CSS_PATH = "loading_screen.tcss"

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header()
        yield Static("Loading backend data...", id="loading_text")
        yield Footer()

    def on_mount(self) -> None:
        """
        Called when the screen is mounted.
        
        Sets a timer to switch to the game screen. In a real scenario,
        this would be replaced by a worker that loads data and then
        switches the screen upon completion.
        """
        self.set_timer(2, self.start_game) # 2-second timer for demonstration

    def start_game(self) -> None:
        """
        Switches to the main game screen.
        
        We import the GameScreen here locally to avoid circular dependencies.
        """
        self.app.switch_screen(GameScreen())
