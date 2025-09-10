from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Label
from textual import events
from textual_pyfiglet import FigletWidget

from weekend_wordle.gui.setup.setup_screen import SetupScreen

class StartupScreen(Screen):
    CSS_PATH = "startup_screen.tcss"

    """The first screen the user sees. Dismissed by any key press."""
    def compose(self) -> ComposeResult:
        yield Vertical(
            FigletWidget(
                "> Weekend Wordle", 
                font="georgia11", 
                justify="center",
                colors=["$gradient-start", "$gradient-end"], 
                horizontal=True
            ),
            Label("Press any key to start", classes="subtitle"),
            id="startup_dialog",
        )
    def on_key(self, event: events.Key) -> None:
        event.stop()
        self.app.switch_screen(SetupScreen())
