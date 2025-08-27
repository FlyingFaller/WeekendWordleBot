"""
Defines the different screens for the Wordle application.
"""
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, ProgressBar, DataTable, Static, Label
from textual import events
from textual_pyfiglet import FigletWidget

# Import refactored components
from board import WordleBoard, GameState
from sidebar import Sidebar, ResultsTable, StatsTable
from progress import TitledProgressBar
from text_processors import TextProcessor, FigletProcessor
from setup_screen import SetupScreen

class StartupScreen(Screen):
    """The first screen the user sees. Dismissed by any key press."""
    def compose(self) -> ComposeResult:
        yield Vertical(
            FigletWidget(
                "> Weekend Wordle", font="georgia11", justify="center",
                colors=["#4795de", "#bb637a"], horizontal=True
            ),
            Label("Press any key to start", classes="subtitle"),
            id="startup-dialog",
        )
    def on_key(self, event: events.Key) -> None:
        event.prevent_default()
        self.app.switch_screen(SetupScreen())

class SettingsScreen(Screen):
    """A placeholder screen for settings."""
    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Settings", id="title"),
            Label("This screen is a placeholder for future settings.", classes="subtitle"),
            Label("Press Ctrl+S to save and close.", classes="subtitle"),
            id="settings-dialog",
        )

class GameScreen(Screen):
    """The main screen for the Wordle game, acting as an orchestrator."""

    # --- UI Settings ---
    use_figlet = True
    TILE_ASPECT_RATIO = 2.0

    def __init__(self):
        super().__init__()
        self.text_processor = FigletProcessor() if self.use_figlet else TextProcessor()

    def compose(self) -> ComposeResult:
        """Create the layout of the application."""
        yield Header(show_clock=True)
        with Container(id="app-container"):
            yield Sidebar(id="sidebar-container")
            with Container(id="board-wrapper"):
                yield WordleBoard(id="wordle-board")
        yield TitledProgressBar()
        yield Footer()

    def on_mount(self) -> None:
        """Start a dummy progress interval and trigger initial resize."""
        self.set_interval(1 / 10, self.update_progress)
        self.call_after_refresh(self.on_resize)

    # --- Event Handlers ---
    def on_key(self, event: events.Key) -> None:
        """Pass key events down to the WordleBoard to handle."""
        self.query_one(WordleBoard).handle_key(event)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Handle the user highlighting a new row in the suggestions table."""
        board = self.query_one(WordleBoard)
        if board.game_state == GameState.INPUT_WORD and not board.current_word:
            results_table = self.query_one(ResultsTable)
            new_suggestion = results_table.dummy_rows[event.cursor_row][1]
            board.update_suggestion(new_suggestion)

    # REMOVED: The watch_wordle_board_game_state method has been removed.

    def on_resize(self, event: object = None) -> None:
        """Handles window resize events to keep the board centered and scaled."""
        app_container = self.query_one("#app-container")
        sidebar = self.query_one(Sidebar)
        progress_bar = self.query_one(TitledProgressBar)
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
            progress_bar.advance(1)
        else:
            progress_bar.progress = 0
