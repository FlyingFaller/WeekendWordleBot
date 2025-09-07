"""
Main application file for the Wordle Solver GUI.

This script sets up the Textual application and manages the different screens.
"""

from textual.app import App
from weekend_wordle.gui.startup.startup_screen import StartupScreen
from weekend_wordle.config import APP_COLORS

class WordleApp(App):
    """The main application class for the Wordle Solver."""

    # CSS_PATH = "wordle_app.tcss"
    
    # Define keybindings that work across all screens
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
    ]

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return APP_COLORS

    def on_mount(self) -> None:
        """Called when the app is first mounted."""
        # Start with the startup screen
        self.push_screen(StartupScreen())

def run_app():
    print('Starting!')
    app = WordleApp()
    app.run()

if __name__ == "__main__":
    run_app()
