from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Footer, Header, DataTable
from textual import events
from textual.color import Gradient
from textual.worker import Worker, WorkerState
import numpy as np

from dataclasses import replace

# Import BoardState and other components from the new state file
from .state import (
    BoardState, 
    GameState, 
    COLOR_INT_TO_CHAR, 
    CHAR_CYCLE
)
from .board_widget import WordleBoard
from .sidebar_widget import Sidebar, ResultsTable, StatsTable
from .progress_widget import PatchedProgressBar
from .text_processors import FigletProcessor
from .confirm_screen import ConfirmationDialog
from ..settings.settings_screen import SettingsScreen
from ...backend.helpers import int_to_pattern
from ...backend.core import WordleGame, InvalidWordError, InvalidPatternError

from ...config import APP_COLORS, NTHREADS, INITIAL_SUGGESTION

class GameScreen(Screen):
    """The main screen for the Wordle game, acting as the central controller."""

    CSS_PATH = "game_screen.tcss"
    BINDINGS = [("ctrl+s", "open_settings", "Settings"),
                ("ctrl+z", "undo_move", "Undo Move")]
    TILE_ASPECT_RATIO = 2

    def __init__(self, game_obj: WordleGame):
        super().__init__()
        self.text_processor = FigletProcessor()
        self.game_obj = game_obj
        self.game_number = None
        self.results_history = []

        # --- Single Source of Truth ---
        self.board_state = BoardState(suggestion=INITIAL_SUGGESTION[0])

        # Worker stuff
        self.worker: Worker|None = None
        self.progress_array: np.ndarray[np.int64]|None = None
        self.progress_timer = None

    def compose(self) -> ComposeResult:
        """Creates the layout of the application."""
        yield Header(show_clock=True)
        with Container(id="app_container"):
            yield Sidebar(id="sidebar_container")
            with Container(id="board_wrapper"):
                yield WordleBoard(id="wordle_board")
        gradient = Gradient.from_colors(APP_COLORS["gradient-start"], APP_COLORS["gradient-end"])
        progress_bar = PatchedProgressBar(gradient=gradient, show_time_elapsed=True)
        progress_bar.border_title = 'Computation Progress'
        yield progress_bar
        yield Footer()

    def on_mount(self) -> None:
        """Initializes the game screen and renders the initial board state."""
        self.call_after_refresh(self.on_resize)
        # Initial render
        self.query_one(WordleBoard).render_state(self.board_state)
        self.query_one(ResultsTable).update_data(self.game_obj, [INITIAL_SUGGESTION])

    # --- Centralized Input Handlers ---

    def on_key(self, event: events.Key) -> None:
        """Handles all keyboard input, modifying the central state and re-rendering."""
        old_state = self.board_state
        if old_state.mode in (GameState.IDLE, GameState.GAME_OVER):
            return

        if old_state.mode == GameState.INPUT_WORD:
            self._handle_word_input(event)
        elif old_state.mode == GameState.INPUT_PATTERN:
            self._handle_pattern_input(event)
        
        # After any state change, command the board to re-render. This is the
        # single source of rendering for all key-based actions.
        if old_state != self.board_state:
            self.query_one(WordleBoard).render_state(self.board_state)

    def on_wordle_board_cell_clicked(self, message: WordleBoard.CellClicked) -> None:
        """Handles a click on a letter square, modifying state and re-rendering."""
        state = self.board_state
        if state.mode == GameState.INPUT_PATTERN and message.row == state.active_row:
            # Cycle color
            pattern = list(state.active_pattern)
            current_color = pattern[message.col]
            pattern[message.col] = CHAR_CYCLE.get(current_color, "-")
            
            self.board_state = replace(state, 
                focused_col=message.col,
                active_pattern="".join(pattern)
            )
            self.query_one(WordleBoard).render_state(self.board_state)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Updates the suggestion in the central state when the user selects a word."""
        if self.board_state.mode == GameState.INPUT_WORD:
            try:
                new_suggestion = event.control.get_cell_at((event.cursor_row, 1))
                if self.board_state.suggestion != str(new_suggestion).upper():
                    self.board_state = replace(self.board_state, suggestion=str(new_suggestion).upper())
                    self.query_one(WordleBoard).render_state(self.board_state)
            except Exception:
                pass # Fails silently if cell doesn't exist

    # --- Game Logic and State Transitions ---

    def action_undo_move(self) -> None:
        """Handles the undo action by modifying the central state."""
        old_state = self.board_state
        state = self.board_state

        # Scenario 1: Reverting an unsubmitted pattern to re-enter the word.
        if state.mode == GameState.INPUT_PATTERN:
            self.board_state = replace(state, mode=GameState.INPUT_WORD, active_pattern="-" * 5, focused_col=0)
        
        # Scenario 2: Reverting a fully completed guess.
        elif state.mode in (GameState.INPUT_WORD, GameState.GAME_OVER) and self.game_obj.guesses_played:
            self._undo_completed_guess()
        
        if old_state != self.board_state:
            self.query_one(WordleBoard).render_state(self.board_state)

    def _undo_completed_guess(self) -> None:
        """Pops the last guess and restores the UI to its previous state."""
        # Store the guess we are about to undo.
        last_word: str = self.game_obj.guesses_played[-1]
        last_pattern_int = self.game_obj.patterns_seen[-1]
        pattern_list = int_to_pattern(last_pattern_int)
        last_pattern_str = "".join([COLOR_INT_TO_CHAR.get(c, "-") for c in pattern_list])

        # Pop the guess from the backend and the results from our history.
        self.game_obj.pop_last_guess()
        self.results_history.pop()

        # Update sidebar tables with the now-current results.
        previous_results = self.results_history[-1] if self.results_history else None
        self._update_sidebar_tables(previous_results, clear_stats=True)
        
        # Get the new backend state.
        status = self.game_obj.get_game_state()
        new_guesses = self._format_guesses(status['guesses_played'], status['patterns_seen'])
        
        suggestion = previous_results['recommendation'] if previous_results else INITIAL_SUGGESTION[0]

        # Update the board state to allow editing the just-popped guess.
        self.board_state = replace(self.board_state,
            mode=GameState.INPUT_PATTERN,
            guesses=new_guesses,
            active_row=len(new_guesses),
            active_word=last_word.upper(),
            active_pattern=last_pattern_str,
            focused_col=4,
            suggestion=suggestion
        )

    def _submit_word(self) -> None:
        """Validates the active word and transitions to pattern input mode."""
        state = self.board_state
        try:
            self.game_obj.validate_guess(state.active_word)
            self.board_state = replace(state,
                mode=GameState.INPUT_PATTERN,
                active_pattern="-" * 5,
                focused_col=0
            )
        except InvalidWordError as e:
            self.app.notify(str(e), title="Invalid Word", severity="error")

    def _submit_pattern(self) -> None:
        """Validates the active pattern and triggers the turn processing."""
        try:
            self.game_obj.validate_pattern(self.board_state.active_pattern)
            self._process_turn()
        except InvalidPatternError as e:
            self.app.notify(str(e), title="Invalid Pattern", severity="error")

    def _process_turn(self) -> None:
        """Orchestrates the entire sequence of events after a valid guess."""
        self.game_obj.make_guess(self.board_state.active_word, self.board_state.active_pattern)
        self.results_history.append(None)
        
        status = self.game_obj.get_game_state()
        new_guesses = self._format_guesses(status['guesses_played'], status['patterns_seen'])
        self.board_state = replace(self.board_state, guesses=new_guesses)

        if status['solved'] or status['failed'] or len(new_guesses) > 5:
            self.board_state = replace(self.board_state, mode=GameState.GAME_OVER)
            if status['solved']:
                self.app.notify("Congratulations, you solved it!", title="Solved!")
            elif status['failed']:
                self.app.notify("No possible answers remain.", title="Failed", severity="error")
            else:
                self.app.notify("Guesses exhausted.", title="Failed", severity="error")
        else:
            self._start_computation()

    def _start_computation(self) -> None:
        """Sets the UI to IDLE and starts the backend worker."""
        self.board_state = replace(self.board_state, mode=GameState.IDLE)
        self._set_ui_disable(True)

        progress_bar = self.query_one(PatchedProgressBar)
        progress_bar.progress = 0
        progress_bar.total = 0
        self.progress_array = np.zeros(NTHREADS + 1, dtype=np.float64)

        self.worker = self.run_worker(
            lambda: self.game_obj.compute_next_guess(self.progress_array),
            exclusive=True,
            thread=True,
            exit_on_error=False
        )
        self.progress_timer = self.set_interval(1 / 10, self._update_progress_bar)

    def _update_progress_bar(self) -> None:
        """Reads from the shared progress array and updates the UI."""
        if self.progress_array is not None:
            progress_bar = self.query_one(PatchedProgressBar)
            total = self.progress_array[-1]
            
            if progress_bar.total != total:
                progress_bar.total = total
                
            if total > 0:
                current_count = np.sum(self.progress_array[:-1])
                progress_bar.progress = min(current_count, total)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handles all worker outcomes, ensuring the UI is always cleaned up and consistent."""
        
        # Only act on the terminal states of the worker.
        if event.state not in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            return

        recommendation = ""
        should_advance_turn = False
        
        # --- Phase 1: Process the result and determine next action ---
        if event.state == WorkerState.SUCCESS:
            results: dict = event.worker.result
            self.results_history[-1] = results
            self._update_sidebar_tables(results)
            recommendation = results.get('recommendation') or ""
            should_advance_turn = True

        elif event.state in (WorkerState.ERROR, WorkerState.CANCELLED):
            if event.state == WorkerState.ERROR:
                self.notify(f"Computation failed. Undoing last move.", severity='error', title="Worker Error")
            else: # CANCELLED
                self.notify("Computation was cancelled. Undoing last move.", title="Worker Cancelled")
            
            # Automatically roll back the game state to a valid, interactive state.
            self._undo_completed_guess()

        # --- Phase 2: Finalize and clean up UI (for all terminal states) ---
        self._finalize_computation()
        
        # --- Phase 3: Prepare next turn (only on success) ---
        if should_advance_turn:
            next_active_row = self.board_state.active_row
            
            # This is the key condition: Only increment the row if a guess
            # has actually been submitted to the backend.
            if self.game_obj.guesses_played:
                next_active_row += 1

            self.board_state = replace(self.board_state,
                mode=GameState.INPUT_WORD,
                active_row=next_active_row, # Use the conditionally updated row
                active_word="",
                active_pattern="",
                suggestion=recommendation
            )
        
        # Finally, render the outcome of the entire operation.
        self.query_one(WordleBoard).render_state(self.board_state)

    def _finalize_computation(self) -> None:
        """Stops timers and re-enables the UI after computation ends."""
        if self.progress_timer:
            self.progress_timer.stop()
            self.progress_timer = None

        progress_bar = self.query_one(PatchedProgressBar)
        if progress_bar.progress < progress_bar.total:
            progress_bar.progress = progress_bar.total

        self._set_ui_disable(False)
        self.query_one(ResultsTable).query_one(DataTable).focus()

    def _update_sidebar_tables(self, results: dict | None, clear_stats: bool = False) -> None:
        """
        Updates the results and stats tables from a results dictionary.
        """
        results_table = self.query_one(ResultsTable)
        stats_table = self.query_one(StatsTable)
        
        if results:
            results_table.update_data(self.game_obj, results['sorted_results'])
        else:
            results_table.update_data(self.game_obj, [INITIAL_SUGGESTION])

        if not clear_stats and results:
            stats_table.update_data(results['event_counts'], self.game_obj)
        else:
            stats_table.query_one(DataTable).clear()

    # --- Internal Key Handlers ---

    def _handle_word_input(self, event: events.Key) -> None:
        """Logic for handling key presses in word input mode."""
        state = self.board_state
        if event.key == "enter":
            if len(state.active_word) == 5:
                self._submit_word()
            elif state.active_row == 0 and not state.active_word:
                event.stop()
                dialog = ConfirmationDialog("Warning: You are about to start a long, uninteruptable computation.")
                self.app.push_screen(dialog, self._start_initial_computation)

        elif event.key == "backspace" and state.active_word:
            self.board_state = replace(state, active_word=state.active_word[:-1])
        elif event.key == "tab":
            if state.suggestion.startswith(state.active_word):
                self.board_state = replace(state, active_word=state.suggestion)
        elif event.is_printable and event.character and len(state.active_word) < 5:
            if event.character.isalpha():
                self.board_state = replace(state, active_word=state.active_word + event.character.upper())
    
    def _handle_pattern_input(self, event: events.Key) -> None:
        """Logic for handling key presses in pattern input mode."""
        state = self.board_state
        if event.key == "enter":
            self._submit_pattern()
        elif event.key in ("right", "space"):
            self.board_state = replace(state, focused_col=min(4, state.focused_col + 1))
        elif event.key == "left":
            self.board_state = replace(state, focused_col=max(0, state.focused_col - 1))
        elif event.character and event.character.lower() in "gy-":
            pattern = list(state.active_pattern)
            pattern[state.focused_col] = event.character.lower()
            self.board_state = replace(state,
                active_pattern="".join(pattern),
                focused_col=min(4, state.focused_col + 1)
            )

    # --- Helper & UI Methods ---
    def _start_initial_computation(self, confirmed: bool) -> None:
        """Called when the confirmation dialog is dismissed."""
        if confirmed:
            self.results_history.append(None)
            self._start_computation()

    def _set_ui_disable(self, disabled: bool) -> None:
        """Disables or enables the UI elements visually."""
        self.query_one(WordleBoard).disabled = disabled
        self.query_one(Sidebar).disabled = disabled

    def _format_guesses(self, guesses: list[str], patterns: list[int]) -> list[tuple[str, str]]:
        """Translates backend guess/pattern data into a format for the BoardState."""
        board_data = []
        for word, p_int in zip(guesses, patterns):
            pattern_list = int_to_pattern(p_int)
            pattern_str = "".join([COLOR_INT_TO_CHAR.get(c, "-") for c in pattern_list])
            board_data.append((word, pattern_str))
        return board_data

    def on_resize(self, event: object = None) -> None:
        """Handles window resize events to keep the board centered and scaled."""
        app_container = self.query_one("#app_container")
        sidebar = self.query_one(Sidebar)
        progress_bar = self.query_one(PatchedProgressBar)
        results_table = sidebar.query_one(ResultsTable).query_one(DataTable)
        stats_table = sidebar.query_one(StatsTable).query_one(DataTable)

        sidebar_width = max(results_table.virtual_size.width, stats_table.virtual_size.width) + 10
        
        available_width = app_container.content_size.width - sidebar_width
        available_height = app_container.content_size.height - progress_bar.outer_size.height

        if not available_height or not available_width:
            return

        h_padding, v_padding = 2, 2
        padded_width = available_width - h_padding
        padded_height = available_height - v_padding

        if padded_width <= 0 or padded_height <= 0:
            return

        cols, rows, gutter = 5, 6, 1
        total_gutter_space = (cols - 1) * gutter
        cell_height_h = padded_height // rows
        cell_width_h = int(cell_height_h * self.TILE_ASPECT_RATIO)
        new_width_from_height = (cell_width_h * cols) + total_gutter_space
        cell_width_w = (padded_width - total_gutter_space) // cols
        cell_height_w = int(cell_width_w / self.TILE_ASPECT_RATIO)
        new_height_from_width = cell_height_w * rows

        if new_width_from_height <= padded_width and cell_height_h > 0:
            new_width, new_height = new_width_from_height, cell_height_h * rows
        elif cell_width_w > 0:
            new_width, new_height = (cell_width_w * cols) + total_gutter_space, new_height_from_width
        else:
            return

        final_cell_height = new_height // rows
        if final_cell_height < 3:
            min_cell_height = 3
            min_cell_width = int(min_cell_height * self.TILE_ASPECT_RATIO)
            new_height = min_cell_height * rows
            new_width = (min_cell_width * cols) + total_gutter_space

        board = self.query_one(WordleBoard)
        board.styles.width = new_width + h_padding
        board.styles.height = new_height + v_padding

    def action_open_settings(self) -> None:
        """Called when the user presses Ctrl+S to toggle the settings screen."""
        def settings_screen_callback(game_number: int|None) -> None:
            self.game_number = game_number
        self.app.push_screen(SettingsScreen(self.game_obj, self.game_number), settings_screen_callback)

