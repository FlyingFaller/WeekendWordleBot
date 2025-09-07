from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Footer, Header, DataTable
from textual import events
from textual.color import Gradient
from textual.worker import Worker, WorkerState
from textual.timer import Timer
import numpy as np

# Import refactored components
from weekend_wordle.config import APP_COLORS, NTHREADS, STARTING_GUESS, STARTING_GUESS_STATS
from weekend_wordle.gui.game.board_widget import WordleBoard, GameState, WordSubmitted, PatternSubmitted
from weekend_wordle.gui.game.sidebar_widget import Sidebar, ResultsTable, StatsTable
from weekend_wordle.gui.game.progress_widget import PatchedProgressBar
from weekend_wordle.gui.game.text_processors import TextProcessor, FigletProcessor
from weekend_wordle.backend.core import WordleGame, InvalidWordError, InvalidPatternError
from weekend_wordle.gui.settings.settings_screen import SettingsScreen

class GameScreen(Screen):
    """The main screen for the Wordle game, acting as an orchestrator."""

    # --- UI Settings ---
    use_figlet = True
    TILE_ASPECT_RATIO = 2.0
    CSS_PATH = "game_screen.tcss"
    BINDINGS = [("ctrl+s", "open_settings", "Settings"),
                ("ctrl+z", "undo_move", "Undo Move")]

    def __init__(self, game_obj: WordleGame):
        super().__init__()
        self.text_processor = FigletProcessor() if self.use_figlet else TextProcessor()
        self.game_obj = game_obj
        self.game_number = None
        self.last_guess = None
        self.worker: Worker = None
        self.progress_array: np.ndarray = None
        self.progress_timer: Timer = None
        self.results_history: list[dict] = []

    def compose(self) -> ComposeResult:
        """Create the layout of the application."""
        yield Header(show_clock=True)
        with Container(id="app_container"):
            yield Sidebar(id="sidebar_container")
            with Container(id="board_wrapper"):
                yield WordleBoard(id="wordle_board")

        gradient = Gradient.from_colors(APP_COLORS["gradient-start"], APP_COLORS["gradient-end"])
        progress_bar =  PatchedProgressBar(
            gradient=gradient,
            show_time_elapsed=True,
        )
        progress_bar.border_title = 'Computation Progress'
        yield progress_bar
        
        yield Footer()

    def on_mount(self) -> None:
        """Set up the initial state of the game."""
        self.call_after_refresh(self.on_resize)
        results_table = self.query_one(ResultsTable)
        board = self.query_one(WordleBoard)
        board.update_suggestion(STARTING_GUESS)
        results_table.query_one(DataTable).add_row("1", STARTING_GUESS, *STARTING_GUESS_STATS, "Default Suggestion")


    def set_ui_disabled(self, disabled: bool) -> None:
        """Disable or enable the main UI components."""
        self.query_one(WordleBoard).disabled = disabled
        self.query_one(Sidebar).disabled = disabled

    def compute_guess(self) -> None:
        """Disables UI and starts the worker to compute the next guess."""
        self.set_ui_disabled(True)

        # Make sure we reset this
        progress_bar = self.query_one(PatchedProgressBar)
        progress_bar.progress = 0

        self.progress_array = np.zeros(NTHREADS+1, dtype=np.float64)

        self.worker = self.run_worker(
            lambda: self.game_obj.compute_next_guess(self.progress_array), # Its so fucking stupid this works
            exclusive=True,
            thread=True
        )
        self.progress_timer = self.set_interval(1 / 10, self.update_progress_bar)

    def update_progress_bar(self) -> None:
        """Reads from the shared progress array and updates the UI."""
        if self.progress_array is not None:
            progress_bar = self.query_one(PatchedProgressBar)
            total = self.progress_array[-1]
            
            if progress_bar.total != total:
                progress_bar.total = total
                
            if total > 0:
                current_count = np.sum(self.progress_array[:-1])
                progress_bar.progress = min(current_count, total)

    def update_ui_with_results(self, results: dict) -> None:
        """Updates the sidebar tables and board suggestion with new data."""
        results_table = self.query_one(ResultsTable)
        stats_table = self.query_one(StatsTable)
        results_table.update_data(self.game_obj, results['sorted_results'])
        if self.game_obj.cache:
            stats_table.update_data(results['event_counts'], self.game_obj.cache)

        board = self.query_one(WordleBoard)
        board.update_suggestion(results['recommendation'])
        results_table.query_one(DataTable).focus()

    def on_worker_state_changed(self, event: Worker.StateChanged):
        """Handle the completion of the guess computation worker."""
        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR):
            if self.progress_timer:
                self.progress_timer.stop()

            progress_bar = self.query_one(PatchedProgressBar)
            progress_bar.progress = progress_bar.total

            self.set_ui_disabled(False)

        if event.state == WorkerState.SUCCESS:
            
            results: dict = event.worker.result
            if results and results['sorted_results']:
                self.results_history.append(results)
                self.update_ui_with_results(results)
            else:
                self.notify("No possible answers remaining!", title="Game Over", severity="error")

            # Update board with the new top suggestion and start the turn
            board = self.query_one(WordleBoard)
            board.update_suggestion(results['recommendation'])
            # Explicitly advance to the next row and prepare for input
            board = self.query_one(WordleBoard)
            if board.game_state == GameState.COMPUTING:
                board.advance_row_and_start_turn()
            else:
                board.start_turn()
            
        elif event.state == WorkerState.ERROR:
            self.notify(f"Computation failed: {event.worker.error}", severity='error', title="Worker Error")

    def on_word_submitted(self, message: WordSubmitted) -> None:
        """Handle the user submitting a word."""
        if self.worker and self.worker.state == WorkerState.RUNNING:
            return

        if self.game_obj.get_game_state()['nguesses'] == 0 and not message.word:
            self.compute_guess()
            return
            
        try:
            self.game_obj.validate_guess(message.word)
            self.last_guess = message.word
            self.query_one(WordleBoard).switch_to_pattern_input()
        except InvalidWordError as e:
            self.notify(str(e), title="Invalid Word", severity="error")


    def on_pattern_submitted(self, message: PatternSubmitted) -> None:
        """Handle the user submitting a pattern."""
        if self.worker and self.worker.state == WorkerState.RUNNING:
            return
            
        board = self.query_one(WordleBoard)
        try:
            pattern_int = self.game_obj.validate_pattern(message.pattern)
            self.game_obj.make_guess(self.last_guess, pattern_int)
            
            game_state = self.game_obj.get_game_state()
            is_win = game_state['solved']
            
            # Finalize the current row. This no longer advances the turn.
            board.end_turn(is_win=is_win)

            if is_win:
                self.notify("You solved it!", title="Congratulations!", severity="information")
            elif game_state['failed']:
                self.notify("No possible answers remaining.", title="Game Over", severity="error")
            else:
                self.compute_guess()

        except InvalidPatternError as e:
            self.notify(str(e), title="Invalid Pattern", severity="error")

    # --- Other Event Handlers ---
    def on_key(self, event: events.Key) -> None:
        if not self.query_one(WordleBoard).disabled:
            self.query_one(WordleBoard).handle_key(event)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        board = self.query_one(WordleBoard)
        if board.game_state == GameState.INPUT_WORD and not board.current_word:
            results_table = self.query_one(ResultsTable)
            table = results_table.query_one(DataTable)
            if event.cursor_row < table.row_count:
                try:
                    new_suggestion = table.get_cell_at((event.cursor_row, 1))
                    board.update_suggestion(new_suggestion)
                except Exception:
                    pass 

    def on_resize(self, event: object = None) -> None:
        """Handles window resize events to keep the board centered and scaled."""
        app_container = self.query_one("#app_container")
        sidebar = self.query_one(Sidebar)
        progress_bar = self.query_one(PatchedProgressBar)
        
        try:
            results_table = sidebar.query_one(ResultsTable).query_one(DataTable)
            stats_table = sidebar.query_one(StatsTable).query_one(DataTable)
            sidebar_width = max(results_table.virtual_size.width, stats_table.virtual_size.width) + 10
        except Exception:
            sidebar_width = 40

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
        
        cell_height_from_h = padded_height // rows
        cell_width_from_h = int(cell_height_from_h * self.TILE_ASPECT_RATIO)
        
        cell_width_from_w = (padded_width - total_gutter_space) // cols
        cell_height_from_w = int(cell_width_from_w / self.TILE_ASPECT_RATIO)

        if (cell_width_from_h * cols) + total_gutter_space <= padded_width:
            final_cell_width = cell_width_from_h
            final_cell_height = cell_height_from_h
        else:
            final_cell_width = cell_width_from_w
            final_cell_height = cell_height_from_w

        min_cell_height = 3
        if final_cell_height < min_cell_height:
            final_cell_height = min_cell_height
            final_cell_width = int(final_cell_height * self.TILE_ASPECT_RATIO)

        new_width = (final_cell_width * cols) + total_gutter_space
        new_height = final_cell_height * rows

        board = self.query_one(WordleBoard)
        board.styles.width = new_width + h_padding
        board.styles.height = new_height + v_padding

    def action_open_settings(self) -> None:
        self.app.push_screen(SettingsScreen(self.game_obj, self.game_number), lambda gn: setattr(self, 'game_number', gn))

    # def action_undo_move(self) -> None:
    #     if self.game_obj.get_game_state()['nguesses'] > 0:
    #         self.game_obj.pop_last_guess()
    #         self.query_one(WordleBoard).current_row -=1
    #         self.notify("Last move undone.")
    #         self.compute_guess()

    # def action_undo_move(self) -> None:
    #     """Undoes the last move, updating the backend and UI correctly."""
    #     board = self.query_one(WordleBoard)
    #     # Prevent undo if a worker is busy or no moves have been made.
    #     if (self.worker and self.worker.state == WorkerState.RUNNING) or self.game_obj.get_game_state()['nguesses'] == 0:
    #         return
        
    #     # 1. Update the backend state.
    #     self.game_obj.pop_last_guess()
    #     self.notify("Last move undone.")

    #     # 2. Update the visual state of the board.
    #     board.undo_row()

    #     # 3. Decide whether to re-compute suggestions or reset to the initial state.
    #     if self.game_obj.get_game_state()['nguesses'] > 0:
    #         # Re-compute suggestions for the new (previous) game state.
    #         # This is slow and should be updated to store previous results
    #         self.compute_guess()
    #     else:
    #         # Reset the entire UI to its initial state.
    #         self.set_ui_disabled(False)
    #         results_table_widget = self.query_one(ResultsTable)
    #         stats_table_widget = self.query_one(StatsTable)
    #         results_table = results_table_widget.query_one(DataTable)
    #         stats_table = stats_table_widget.query_one(DataTable)

    #         results_table.clear()
    #         stats_table.clear()

    #         board.update_suggestion(STARTING_GUESS)
    #         results_table.add_row("1", STARTING_GUESS, *STARTING_GUESS_STATS, "Default Suggestion")


    def action_undo_move(self) -> None:
        """Undoes the last committed action, either a word entry or a full guess."""
        board = self.query_one(WordleBoard)
        if self.worker and self.worker.state == WorkerState.RUNNING:
            return

        # Case 1: User has submitted a word and is in pattern input.
        # We only need to revert the board's state. No backend change is needed.
        if board.game_state == GameState.INPUT_PATTERN:
            board.undo_row()
            self.notify("Word entry reverted.")
            return

        # Case 2: User wants to undo a complete guess (word + pattern).
        # This is only possible if a guess has been made and its results are cached.
        if self.results_history:
            self.results_history.pop()
            self.game_obj.pop_last_guess()
            self.notify("Last move undone.")
            board.undo_row()

            if self.results_history:
                # Load the previous results from our history cache.
                previous_results = self.results_history[-1]
                self.update_ui_with_results(previous_results)
                board.start_turn()
            else:
                # We've undone all moves, so reset to the initial state.
                results_table_widget = self.query_one(ResultsTable)
                stats_table_widget = self.query_one(StatsTable)
                results_table = results_table_widget.query_one(DataTable)
                stats_table = stats_table_widget.query_one(DataTable)

                results_table.clear()
                stats_table.clear()

                board.update_suggestion(STARTING_GUESS)
                results_table.add_row("1", STARTING_GUESS, *STARTING_GUESS_STATS, "Default Suggestion")