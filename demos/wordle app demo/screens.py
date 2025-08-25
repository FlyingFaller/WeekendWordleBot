"""
Defines the different screens for the Wordle application.

- StartupScreen: The initial screen shown on launch.
- SettingsScreen: A placeholder for application settings.
- GameScreen: The main screen where the Wordle game is played.
"""

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, ProgressBar, DataTable, Static, Label
from textual import events
from textual_pyfiglet import FigletWidget

# Import components from other files
from board import WordleBoard, LetterSquare, GameState, LetterState, COLORS, COLOR_CHARS
from sidebar import Sidebar, StatsTable
from progress import TitledProgressBar
from stats_display import StatsDisplay
# Import the text processors
from text_processors import TextProcessor, FigletProcessor

class StartupScreen(Screen):
    """The first screen the user sees. Dismissed by any key press."""

    def compose(self) -> ComposeResult:
        """Render the startup screen."""
        yield Vertical(
            FigletWidget(
                "> Weekend Wordle",
                font="georgia11",
                justify="center",
                colors=["#4795de", "#bb637a"],
                horizontal=True
            ),
            Label("Press any key to start", classes="subtitle"),
            id="startup-dialog",
        )

    def on_key(self, event: events.Key) -> None:
        """Go to the main game screen when any key is pressed."""
        event.prevent_default()
        self.app.push_screen(GameScreen())

class SettingsScreen(Screen):
    """A placeholder screen for settings."""

    def compose(self) -> ComposeResult:
        """Render the settings screen."""
        yield Vertical(
            Label("Settings", id="title"),
            Label("This screen is a placeholder for future settings.", classes="subtitle"),
            Label("Press Ctrl+S to save and close.", classes="subtitle"),
            id="settings-dialog",
        )

class GameScreen(Screen):
    """The main screen for the Wordle game."""

    current_row = 0
    game_state = GameState.INPUT_WORD
    focused_col = 0
    suggested_word = "SLATE"
    use_figlet = True
    TILE_ASPECT_RATIO = 2.0

    def __init__(self):
        super().__init__()
        self.board: list[list[LetterSquare]] = [[] for _ in range(6)]
        self.current_word = ""
        if self.use_figlet:
            self.text_processor = FigletProcessor()
        else:
            self.text_processor = TextProcessor()

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
        """Called when the screen is first mounted."""
        for square in self.query(LetterSquare):
            self.board[square.row].append(square)
        self.start_turn()
        self.set_interval(1 / 10, self.update_progress)
        self.call_after_refresh(self.on_resize)

    def on_resize(self, event: object = None) -> None:
        """Handles window resize events to keep the board centered and scaled."""
        
        # --- UPDATED: Simplified and corrected layout logic ---
        
        # 1. Get references to all the layout components
        app_container = self.query_one("#app-container")
        sidebar = self.query_one(Sidebar)
        progress_bar = self.query_one(TitledProgressBar)
        stats_table = sidebar.query_one(StatsTable).query_one(DataTable)
        stats_display = sidebar.query_one(StatsDisplay).query_one(DataTable)

        # 2. Calculate the sidebar's required width dynamically
        # FIXED: Use virtual_size.width to get the ideal content width
        sidebar_width = max(stats_table.virtual_size.width, stats_display.virtual_size.width) + 10 # magic number do not change
        
        # 3. Calculate the true available space for the board
        available_width = app_container.content_size.width - sidebar_width
        # Subtract the progress bar's height (including its border)
        available_height = app_container.content_size.height - progress_bar.outer_size.height

        if not available_height or not available_width:
            return

        # 4. Perform the aspect ratio calculation as before
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

        # 5. Apply the final dimensions
        board = self.query_one(WordleBoard)
        board.styles.width = new_width + h_padding
        board.styles.height = new_height + v_padding

    # --- Game Logic (omitted for brevity, no changes) ---
    def start_turn(self) -> None:
        self.game_state = GameState.INPUT_WORD
        self.current_word = ""
        self.update_footer()
        if self.current_row < 6:
            self._update_recommendation_display()
    def submit_word(self) -> None:
        self.game_state = GameState.INPUT_PATTERN
        self.update_footer()
        for square in self.board[self.current_row]:
            square.color_index = 0
        self.focused_col = 0
        self.board[self.current_row][self.focused_col].has_focus = True
    def submit_pattern(self) -> None:
        self.board[self.current_row][self.focused_col].has_focus = False
        is_win = all(sq.color_index == 2 for sq in self.board[self.current_row])
        if is_win or self.current_row >= 5:
            self.game_state = GameState.GAME_OVER
            self.update_footer()
            return
        self.current_row += 1
        self.start_turn()
    def update_footer(self) -> None:
        footer = self.query_one(Footer)
        if self.game_state == GameState.INPUT_WORD:
            footer.key_text = "Tab: Autocomplete | Enter: Submit Word | Type to enter your guess."
        elif self.game_state == GameState.INPUT_PATTERN:
            footer.key_text = "Enter: Submit | Arrows: Navigate | Click or type pattern (g, y, -)."
        elif self.game_state == GameState.GAME_OVER:
            footer.key_text = "Game Over! Press 'Q' to Quit."
    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self.game_state == GameState.INPUT_WORD and not self.current_word:
            stats_table = self.query_one(StatsTable)
            self.suggested_word = stats_table.dummy_rows[event.cursor_row][1]
            self._update_recommendation_display()
    def on_key(self, event: events.Key) -> None:
        if self.game_state == GameState.INPUT_WORD:
            self.handle_word_input(event)
        elif self.game_state == GameState.INPUT_PATTERN:
            self.handle_pattern_input(event)
    def handle_word_input(self, event: events.Key) -> None:
        if event.key == "tab":
            event.prevent_default()
            if self.suggested_word.startswith(self.current_word):
                for i in range(len(self.current_word), 5):
                    self.board[self.current_row][i].letter = self.suggested_word[i]
                    self.board[self.current_row][i].letter_state = LetterState.FILLED
                self.current_word = self.suggested_word
        elif event.key == "enter":
            if len(self.current_word) == 5:
                self.submit_word()
        elif event.key == "backspace":
            if self.current_word:
                self.current_word = self.current_word[:-1]
                self._update_recommendation_display()
        elif event.is_printable and len(self.current_word) < 5:
            if event.character and event.character.isalpha():
                char_upper = event.character.upper()
                col = len(self.current_word)
                square = self.board[self.current_row][col]
                square.letter = char_upper
                square.letter_state = LetterState.FILLED
                self.current_word += char_upper
                self._update_recommendation_display()
    def _update_recommendation_display(self) -> None:
        recommendation = self.suggested_word
        prefix = self.current_word
        matches = recommendation.startswith(prefix)
        for col in range(len(prefix), 5):
            square = self.board[self.current_row][col]
            if matches:
                square.letter = recommendation[col]
                square.letter_state = LetterState.RECOMMENDATION
            else:
                square.letter = " "
                square.letter_state = LetterState.EMPTY
    def handle_pattern_input(self, event: events.Key) -> None:
        old_col = self.focused_col
        if event.key == "enter":
            self.submit_pattern()
        elif event.key == "right":
            self.focused_col = min(4, self.focused_col + 1)
        elif event.key == "left":
            self.focused_col = max(0, self.focused_col - 1)
        elif event.key == "backspace":
            self.focused_col = max(0, self.focused_col - 1)
        elif event.character and event.character.lower() in COLOR_CHARS:
            square = self.board[self.current_row][self.focused_col]
            square.color_index = COLOR_CHARS[event.character.lower()]
            self.focused_col = min(4, self.focused_col + 1)
        if old_col != self.focused_col:
            self.board[self.current_row][old_col].has_focus = False
            self.board[self.current_row][self.focused_col].has_focus = True
    def on_letter_square_clicked(self, message: LetterSquare.Clicked) -> None:
        if (self.game_state == GameState.INPUT_PATTERN and
                message.square.row == self.current_row):
            square = message.square
            self.board[self.current_row][self.focused_col].has_focus = False
            self.focused_col = square.col
            self.board[self.current_row][self.focused_col].has_focus = True
            square.color_index = (square.color_index + 1) % len(COLORS)
    def update_progress(self) -> None:
        progress_bar = self.query_one(ProgressBar)
        if progress_bar.progress < 100:
            progress_bar.advance(1)
        else:
            progress_bar.progress = 0
