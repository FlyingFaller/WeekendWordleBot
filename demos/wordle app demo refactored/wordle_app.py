"""
Main application file for the Wordle Solver GUI.

This script sets up the Textual application and manages the different screens.
"""

from textual.app import App
from screens import StartupScreen, SettingsScreen

class WordleApp(App):
    """The main application class for the Wordle Solver."""

    CSS_PATH = "wordle_app.tcss"
    
    # Define keybindings that work across all screens
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+s", "toggle_settings", "Settings"),
    ]
    
    # Keep track of whether the settings screen is open
    show_settings_screen = False

    def on_mount(self) -> None:
        """Called when the app is first mounted."""
        # Start with the startup screen
        self.push_screen(StartupScreen())
        
    def action_toggle_settings(self) -> None:
        """Called when the user presses Ctrl+S."""
        if self.show_settings_screen:
            self.pop_screen()
            self.show_settings_screen = False
        else:
            self.push_screen(SettingsScreen())
            self.show_settings_screen = True


if __name__ == "__main__":
    app = WordleApp()
    app.run()
