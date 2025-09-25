from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Label
from textual import events
from textual_pyfiglet import FigletWidget

# Import your new loading screen
from .transitional_loading_screen import TransitionalLoading


class StartupScreen(Screen):
    CSS_PATH = "startup_screen.tcss"

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
        """Handle a key press on the startup screen."""
        event.stop()
        # Simply switch to the loading screen. The worker will handle the rest.
        self.app.switch_screen(TransitionalLoading())