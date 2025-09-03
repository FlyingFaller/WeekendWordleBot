"""
Defines the LoadingScreen for the Wordle Solver application.

This screen provides visual feedback to the user while backend assets are
being loaded and processed.
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, ProgressBar, RichLog
from textual.worker import Worker, WorkerState
from textual.color import Gradient

from weekend_wordle.gui.game.game_screen import GameScreen
from weekend_wordle.backend.messenger import TextualMessenger
from weekend_wordle.gui.game.progress_widget import PatchedProgressBar
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

        gradient = Gradient.from_colors("#4795de", "#bb637a")
        yield PatchedProgressBar(
            gradient=gradient,
            show_time_elapsed=True, # Explicitly enable the new feature
        )
        
        yield Footer()

    def on_mount(self) -> None:
        """Called when the screen is mounted. Starts the backend worker."""
        self.run_worker(self.load_backend_data, thread=True)

    def load_backend_data(self) -> None:
        """
        This function is executed by the worker.
        It creates a messenger and passes it to the backend.
        """
        messenger = TextualMessenger()

        guess_cfg = self.config['guesses']
        guess_words = get_words(savefile          = get_abs_path(guess_cfg['savefile']),
                                url               = guess_cfg['url'],
                                refetch           = guess_cfg['refetch'],
                                save              = guess_cfg['save'],
                                include_uppercase = guess_cfg['include_uppercase'],
                                messenger         = messenger)
        
        answer_cfg = self.config['answers']
        answer_words = get_words(savefile          = get_abs_path(answer_cfg['savefile']),
                                 url               = answer_cfg['url'],
                                 refetch           = answer_cfg['refetch'],
                                 save              = answer_cfg['save'],
                                 include_uppercase = answer_cfg['include_uppercase'],
                                 messenger         = messenger)
        
        pattern_matrix_cfg = self.config['pattern_matrix']
        pattern_matrix = get_pattern_matrix(guesses   = guess_words,
                                            answers   = answer_words,
                                            savefile  = pattern_matrix_cfg['savefile'],
                                            recompute = pattern_matrix_cfg['refetch'],
                                            save      = pattern_matrix_cfg['save'],
                                            messenger = messenger)
        
        pattern_matrix = get_pattern_matrix(guesses   = guess_words,
                                            answers   = answer_words,
                                            savefile  = pattern_matrix_cfg['savefile'],
                                            recompute = pattern_matrix_cfg['refetch'],
                                            save      = pattern_matrix_cfg['save'],
                                            messenger = messenger)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Called when the worker's state changes."""
        if event.state == WorkerState.SUCCESS:
            log = self.query_one(RichLog)
            log.write("\n[bold green]Loading complete! Starting game...[/bold green]")
            # self.set_timer(1.5, self.start_game) # Short delay to show completion message
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
        p_bar = self.query_one(PatchedProgressBar)
        p_bar.total = message.total
        p_bar.progress = 0

        p_bar.border_title = message.description
        
        loading_text = self.query_one("#loading_text", Static)
        loading_text.update(message.description)

    def on_textual_messenger_progress_update(
        self, message: TextualMessenger.ProgressUpdate
    ) -> None:
        """Advance the progress bar."""
        self.query_one(PatchedProgressBar).advance(message.advance)
