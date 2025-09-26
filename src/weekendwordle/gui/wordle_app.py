"""
Main application file for the Wordle Solver GUI.

This script sets up the Textual application and manages the different screens.
"""

from textual.app import App
import argparse
from .startup.startup_screen import StartupScreen
from ..config import APP_COLORS, CONFIG_FILE

class WordleApp(App):
    """The main application class for the Wordle Solver."""

    # CSS_PATH = "wordle_app.tcss"
    
    # Define keybindings that work across all screens
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, config_path: str|None = CONFIG_FILE, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config_path = config_path

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return APP_COLORS

    def on_mount(self) -> None:
        """Called when the app is first mounted."""
        # Start with the startup screen
        self.push_screen(StartupScreen())

def run_app():
    parser = argparse.ArgumentParser(description="Wordle Solver TUI")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=CONFIG_FILE,
        help="Path to a custom configuration JSON file."
    )
    args = parser.parse_args()
    app = WordleApp(config_path=args.config)
    app.run()

if __name__ == "__main__":
    run_app()
