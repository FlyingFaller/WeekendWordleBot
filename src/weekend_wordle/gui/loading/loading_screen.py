"""
Defines the LoadingScreen for the Wordle Solver application.

This screen provides visual feedback to the user while backend assets are
being loaded and processed.
"""
from functools import reduce
import numpy as np

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, RichLog
from textual.worker import Worker, WorkerState
from textual.color import Gradient
from textual import events

from weekend_wordle.gui.game.game_screen import GameScreen
from weekend_wordle.backend.messenger import TextualMessenger
from weekend_wordle.gui.game.progress_widget import PatchedProgressBar
from weekend_wordle.backend.helpers import *
from weekend_wordle.backend.classifier import filter_words_by_probability, load_classifier, get_word_features
from weekend_wordle.gui.setup.loading_widget import GetWordsWidget, ScrapeWordsWidget
from weekend_wordle.gui.setup.filter_widget import FilterSuffixWidget, FilterFrequencyWidget, FilterPOSWidget, FilterProbabilityWidget
from weekend_wordle.backend.core import WordleGame
from weekend_wordle.config import NPRUNE_GLOBAL_DEFAULT, NPRUNE_ANSWERS_DEFAULT, MAX_DEPTH_DEFAULT, APP_COLORS

class LoadingScreen(Screen):
    """A screen to display while the backend is loading data."""
    CSS_PATH = "loading_screen.tcss"

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.worker: Worker = None
        self.loading_error = False

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header()
        yield RichLog(id="log_output", highlight=False, markup=True)
        gradient = Gradient.from_colors(APP_COLORS["gradient-start"], APP_COLORS["gradient-end"])
        yield PatchedProgressBar(
            gradient=gradient,
            show_time_elapsed=True, # Explicitly enable the new feature
        )
        
        yield Footer()

    def on_mount(self) -> None:
        """Called when the screen is mounted. Starts the backend worker."""
        self.worker = self.run_worker(self.load_backend_data, thread=True, exit_on_error=False)

    def load_backend_data(self) -> WordleGame:
        """
        This function is executed by the worker.
        It creates a messenger and passes it to the backend.
        """
        messenger = TextualMessenger()
        guesses = get_words(**self.config['guesses'], messenger=messenger)
        answers = get_words(**self.config['answers'], messenger=messenger)
        pattern_matrix = get_pattern_matrix(guesses, answers, **self.config['pattern_matrix'], messenger=messenger)

        ### Load Classifier ###
        if self.config['classifier']:
            clss_cfg = self.config['classifier']
            positive_word_tuple = ()
            for positive_word_source in clss_cfg['positive_words']:
                if positive_word_source['type'] is GetWordsWidget:
                    positive_word_tuple += (get_words(**positive_word_source['contents'], messenger=messenger),)
                elif positive_word_source['type'] is ScrapeWordsWidget:
                    positive_word_tuple += (scrape_words(**positive_word_source['contents'], messenger=messenger),)

            positive_words = reduce(np.union1d, positive_word_tuple)
            word_features = get_word_features(all_words=guesses, **clss_cfg['word_features'], messenger=messenger)
            classifier_sort_func = load_classifier(word_features, 
                                                positive_words=positive_words, 
                                                all_words=guesses, 
                                                **clss_cfg['model'],
                                                messenger=messenger)
            
        ### Filter Answers ###
        filtered_answers = answers.copy()
        for filter_source in self.config['filters']:
            filter_type = filter_source['type']
            filter_contents = filter_source['contents']
            if filter_type is FilterSuffixWidget:
                filter_words = get_words(**filter_contents['get_words'], messenger=messenger)
                filtered_answers = filter_words_by_suffix(filtered_answers,
                                                        filter_words,
                                                        filter_contents['suffixes'],
                                                        messenger=messenger)
            elif filter_type is FilterFrequencyWidget:
                filtered_answers = filter_words_by_frequency(filtered_answers, 
                                                            filter_contents['min_freq'],
                                                            messenger=messenger)
            elif filter_type is FilterPOSWidget:
                filtered_answers = filter_words_by_POS(filtered_answers, 
                                                    **filter_contents,
                                                    messenger=messenger)
            elif filter_type is FilterProbabilityWidget and self.config['classifier']:
                filtered_answers = filter_words_by_probability(classifier_sort_func, 
                                                            filtered_answers,
                                                            filter_contents['threshold'],
                                                            messenger=messenger)
            game_obj = WordleGame(pattern_matrix,
                                  guesses,
                                  filtered_answers,
                                  nprune_global = NPRUNE_GLOBAL_DEFAULT,
                                  nprune_answers = NPRUNE_ANSWERS_DEFAULT,
                                  max_depth = MAX_DEPTH_DEFAULT,
                                  sort_func = classifier_sort_func if self.config['sort'] == 'Classifier' and self.config['classifier'] else None)
        
            return game_obj

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Called when the worker's state changes."""
        if event.state == WorkerState.SUCCESS:
            log = self.query_one(RichLog)
            log.write("\n[bold green]Loading complete! Starting game...[/bold green]")
            self.set_timer(1.0, self.start_game) # Short delay to show completion message


        elif event.state == WorkerState.ERROR:
            log = self.query_one(RichLog)
            log.write("\n[bold red]FATAL ERROR:[/bold red] Backend loading failed.\n")
            log.write(f"{event.worker.error}\n")
            log.write('Press any key to return to setup screen.')
            self.loading_error = True

    def on_key(self, event: events.Key) -> None:
        if self.loading_error:
            event.stop()
            self.app.pop_screen()

    def start_game(self) -> None:
        """Switches to the main game screen."""
        self.app.switch_screen(GameScreen(self.worker.result))

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

    def on_textual_messenger_progress_update(
        self, message: TextualMessenger.ProgressUpdate
    ) -> None:
        """Advance the progress bar."""
        self.query_one(PatchedProgressBar).advance(message.advance)

    def on_textual_messenger_progress_stop(
        self, messange: TextualMessenger.ProgressStop
    ) -> None:
        """Stop the progress bar and ensure it has finished."""
        p_bar = self.query_one(PatchedProgressBar)
        p_bar.progress = p_bar.total