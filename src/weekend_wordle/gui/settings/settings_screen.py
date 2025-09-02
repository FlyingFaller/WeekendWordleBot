from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Label

class SettingsScreen(Screen):
    """A placeholder screen for settings."""

    CSS_PATH = "settings_screen.tcss"

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Settings", id="title"),
            Label("This screen is a placeholder for future settings.", classes="subtitle"),
            Label("Press Ctrl+S to save and close.", classes="subtitle"),
            id="settings_dialog",
        )