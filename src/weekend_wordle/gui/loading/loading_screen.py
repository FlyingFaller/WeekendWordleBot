"""
Defines the LoadingScreen for the Wordle Solver application.

This screen provides visual feedback to the user while backend assets are
being loaded and processed.
"""
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, ProgressBar, RichLog
from textual.worker import Worker, WorkerState

from weekend_wordle.gui.game.game_screen import GameScreen
from weekend_wordle.backend.messenger import TextualMessenger
from weekend_wordle.gui.game.progress_widget import TitledProgressBar
from weekend_wordle.backend.helpers import *
from weekend_wordle.backend.classifier import filter_words_by_probability, load_classifier, get_word_features

class LoadingScreen(Screen):
    """A screen to display while the backend is loading data."""
    CSS_PATH = "loading_screen.tcss"

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header()
        yield Static("Loading backend data...", id="loading_text")
        yield RichLog(id="log_output", highlight=True, markup=True)
        yield TitledProgressBar(id="progress_bar", total=100)
        yield Footer()

    def on_mount(self) -> None:
        """Called when the screen is mounted. Starts the backend worker."""
        # super().__init__()
        self.run_worker(self.load_backend_data, thread=True)
        # pass

    def load_backend_data(self) -> None:
        """
        This function is executed by the worker.
        It creates a messenger and passes it to the backend.
        """
        messenger = TextualMessenger()
        # return

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Called when the worker's state changes."""
        if event.state == WorkerState.SUCCESS:
            log = self.query_one(RichLog)
            log.write("\n[bold green]Loading complete! Starting game...[/bold green]")
            self.set_timer(1.5, self.start_game) # Short delay to show completion message
        elif event.state == WorkerState.ERROR:
            log = self.query_one(RichLog)
            log.write("\n[bold red]FATAL ERROR:[/bold red] Backend loading failed.")
            log.write(f"{event.worker.error}")


    def start_game(self) -> None:
        """Switches to the main game screen."""
        self.app.switch_screen(GameScreen())

    # # --- Message Handlers for TextualMessenger ---

    def on_textual_messenger_log(self, message: TextualMessenger.Log) -> None:
        """Write a log message to the RichLog."""
        self.query_one(RichLog).write(message.text)

    def on_textual_messenger_progress_start(
        self, message: TextualMessenger.ProgressStart
    ) -> None:
        """Reset the progress bar for a new task."""
        p_bar = self.query_one(ProgressBar)
        p_bar.total = message.total
        p_bar.progress = 0
        
        loading_text = self.query_one("#loading_text", Static)
        loading_text.update(message.description)

    def on_textual_messenger_progress_update(
        self, message: TextualMessenger.ProgressUpdate
    ) -> None:
        """Advance the progress bar."""
        self.query_one(ProgressBar).advance(message.advance)
