from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Footer, Header, ProgressBar, DataTable
from textual import events
from textual.color import Gradient

from dataclasses import replace

# Import BoardState and other components from the new state file
from weekend_wordle.gui.game.state import (
    BoardState, 
    GameState, 
    COLOR_INT_TO_CHAR, 
    CHAR_CYCLE
)
from weekend_wordle.gui.game.board_widget import WordleBoard
from weekend_wordle.gui.game.sidebar_widget import Sidebar, ResultsTable, StatsTable
from weekend_wordle.gui.game.progress_widget import PatchedProgressBar
from weekend_wordle.gui.game.text_processors import FigletProcessor
from weekend_wordle.backend.core import WordleGame, InvalidWordError, InvalidPatternError
from weekend_wordle.gui.settings.settings_screen import SettingsScreen
from weekend_wordle.backend.helpers import int_to_pattern

from weekend_wordle.config import APP_COLORS

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

        # --- Single Source of Truth ---
        self.board_state = BoardState()

    def compose(self) -> ComposeResult:
        """Creates the layout of the application."""
        yield Header(show_clock=True)
        with Container(id="app_container"):
            yield Sidebar(id="sidebar_container")
            with Container(id="board_wrapper"):
                yield WordleBoard(id="wordle_board")
        gradient = Gradient.from_colors(APP_COLORS["gradient-start"], APP_COLORS["gradient-end"])
        yield PatchedProgressBar(gradient=gradient, show_time_elapsed=True)
        yield Footer()

    def on_mount(self) -> None:
        """Initializes the game screen and renders the initial board state."""
        self.set_interval(1 / 10, self.update_progress)
        self.call_after_refresh(self.on_resize)
        # Initial render
        self.query_one(WordleBoard).render_state(self.board_state)

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
        
        # After any state change, command the board to re-render
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
        # This works from the next turn's input mode OR from the game over screen.
        elif state.mode in (GameState.INPUT_WORD, GameState.GAME_OVER) and self.game_obj.guesses_played:
            # Store the guess we are about to undo.
            last_word: str = self.game_obj.guesses_played[-1]
            last_pattern_int = self.game_obj.patterns_seen[-1]
            pattern_list = int_to_pattern(last_pattern_int)
            last_pattern_str = "".join([COLOR_INT_TO_CHAR.get(c, "-") for c in pattern_list])

            # Pop the guess from the backend.
            self.game_obj.pop_last_guess()
            
            # Get the new state from the backend.
            status = self.game_obj.get_game_state()
            new_guesses = self._format_guesses(status['guesses_played'], status['patterns_seen'])
            
            # Update the board state to allow editing the just-popped guess.
            self.board_state = replace(state,
                mode=GameState.INPUT_PATTERN,
                guesses=new_guesses,
                active_row=len(new_guesses), # The new active row is the index of the popped guess
                active_word=last_word.upper(),
                active_pattern=last_pattern_str,
                focused_col=4
            )
        
        if old_state != self.board_state:
            self.query_one(WordleBoard).render_state(self.board_state)

    def _submit_word(self) -> None:
        """Validates and submits the active word."""
        state = self.board_state
        try:
            validated_word, _ = self.game_obj.validate_guess(state.active_word)
            self.board_state = replace(state,
                mode=GameState.INPUT_PATTERN,
                active_pattern="-" * 5, # Default to all gray
                focused_col=0
            )
        except InvalidWordError as e:
            self.app.notify(str(e), title="Invalid Word", severity="error")

    def _submit_pattern(self) -> None:
        """Validates the active pattern and triggers the guess."""
        try:
            self.game_obj.validate_pattern(self.board_state.active_pattern)
            self._make_guess()

        except InvalidPatternError as e:
            self.app.notify(str(e), title="Invalid Pattern", severity="error")

    def _make_guess(self) -> None:
        """Updates the backend, determines the next game state, and renders it."""
        state = self.board_state
        self.game_obj.make_guess(state.active_word, state.active_pattern)
        
        status = self.game_obj.get_game_state()
        new_guesses = self._format_guesses(status['guesses_played'], status['patterns_seen'])

        if status['solved'] or status['failed']:
            new_mode=GameState.GAME_OVER
            if status['solved']:
                self.app.notify("Congratulations, you solved it!", title="Solved!")
            else:
                self.app.notify("No possible answers remain.", title="Failed", severity="error")
        else:
            new_mode=GameState.IDLE
        
        self.board_state = replace(state, 
            mode=new_mode,
            guesses=new_guesses
        )
        self.query_one(WordleBoard).render_state(self.board_state)

        if new_mode == GameState.IDLE:
            self.set_timer(2.0, self._finish_turn_processing) ### LATER A CALL TO COMPUTE

    def _finish_turn_processing(self) -> None:
        """Transitions the game to the next state after the simulated delay."""
        ### POST COMPUTE CALL
        self.board_state = replace(self.board_state,
            mode=GameState.INPUT_WORD,
            active_row=self.board_state.active_row + 1,
            active_word="",
            active_pattern=""
        )
        self.query_one(WordleBoard).render_state(self.board_state)

    # --- Internal Key Handlers ---

    def _handle_word_input(self, event: events.Key) -> None:
        """Logic for handling key presses in word input mode."""
        state = self.board_state
        if event.key == "enter" and len(state.active_word) == 5:
            self._submit_word()
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

    def update_progress(self) -> None:
        """Dummy function to advance the progress bar."""
        progress_bar = self.query_one(ProgressBar)
        if progress_bar.progress < 100:
            progress_bar.advance(0.5)
        else:
            progress_bar.progress = 0

    def action_open_settings(self) -> None:
        """Called when the user presses Ctrl+S to toggle the settings screen."""
        def settings_screen_callback(game_number: int|None) -> None:
            self.game_number = game_number
        self.app.push_screen(SettingsScreen(self.game_obj, self.game_number), settings_screen_callback)

