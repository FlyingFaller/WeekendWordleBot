from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Container
from textual.screen import Screen
from textual.widgets import Label, Footer, Input, Button, RichLog
from textual.validation import Integer
from datetime import date
from ...backend.core import WordleGame

class SettingsScreen(Screen):
    """A screen for configuring application settings."""

    CSS_PATH = "settings_screen.tcss"
    BINDINGS = [("ctrl+s", "close_screen", "Save and Exit")]
    AUTO_FOCUS = ""

    def __init__(self, game_obj: WordleGame, game_number: int):
        super().__init__()
        self.game_obj = game_obj
        self.game_number: int | None = game_number
        self.discord_text: str|None = None

    def get_default_game_number(self) -> int:
        """Calculate the default game number based on the Wordle epoch."""
        epoch = date(2021, 6, 19)
        today = date.today()
        return (today - epoch).days

    def compose(self) -> ComposeResult:
        """Create the layout for the settings screen."""
        with Horizontal(id="settings_container"):
            left_pane = Vertical(id='left_pane')
            left_pane.border_title = "Game Settings"
            with left_pane:
                yield Label("Prune Globally By")
                yield Input(validators=[Integer(minimum=1)], id="nprune_global")

                yield Container(classes='spacer')

                yield Label("Prune Answers By")
                yield Input(validators=[Integer(minimum=0)], id="nprune_answers")

                yield Container(classes='spacer')

                yield Label("Max Depth")
                yield Input(validators=[Integer(minimum=1)], id="max_depth")
                
                yield Container(classes='spacer')

                yield Label("Game Number")
                yield Input(validators=[Integer(minimum=0)], id="game_number")

            with Container(id='right_pane'):
                copy_button =  Button("Copy", variant="primary", id="copy_button", compact=True)
                copy_button.can_focus = False
                yield copy_button
                discord = RichLog(highlight=True, 
                                    markup=True, 
                                    id="discord_output")
                discord.border_title = "Discord Copy"
                yield discord
        yield Footer()

    def on_mount(self) -> None:
        """Load current settings into inputs when the screen is mounted."""
        # Load game object settings
        self.query_one("#nprune_global", Input).value = str(getattr(self.game_obj, 'nprune_global', ''))
        self.query_one("#nprune_answers", Input).value = str(getattr(self.game_obj, 'nprune_answers', ''))
        self.query_one("#max_depth", Input).value = str(getattr(self.game_obj, 'max_depth', ''))

        # Set game number and update discord output
        game_num_input = self.query_one("#game_number", Input)
        default_game_num = self.game_number if self.game_number is not None else self.get_default_game_number()
        game_num_input.value = str(default_game_num)
        self.update_discord_output(game_num_input)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes to update the Discord output."""
        if event.input.id == "game_number":
            self.update_discord_output(event.input)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the copy button press event."""
        if event.button.id == "copy_button":
            if self.discord_text:
                self.app.copy_to_clipboard(self.discord_text)
                self.notify("Output copied to clipboard!")
            else:
                self.notify("Failed to copy!", severity='error')

    def update_discord_output(self, game_num_input: Input) -> None:
        """Fetch and display the discord printout for the given game number."""
        output_log = self.query_one(RichLog)
        if game_num_input.is_valid and game_num_input.value:
            game_number = int(game_num_input.value)
            self.game_number = game_number
            # Assuming get_discord_printout returns a string
            self.discord_text = self.game_obj.get_discord_printout(game_number=game_number)
            output_log.clear()
            output_log.write(self.discord_text)
        else:
            output_log.clear()
            output_log.notify("Please enter a valid game number.", severity="error")

    def action_close_screen(self) -> None:
        """Save settings and close the screen."""
        try:
            nprune_global_val = self.query_one("#nprune_global", Input).value
            nprune_answers_val = self.query_one("#nprune_answers", Input).value
            max_depth_val = self.query_one("#max_depth", Input).value

            if nprune_global_val:
                self.game_obj.nprune_global = int(nprune_global_val)
            if nprune_answers_val:
                self.game_obj.nprune_answers = int(nprune_answers_val)
            if max_depth_val:
                self.game_obj.max_depth = int(max_depth_val)
            
            self.notify("Settings saved!")
        except (ValueError, AttributeError) as e:
            self.notify(f"Error saving settings: {e}", severity="error")

        # self.app.pop_screen()
        self.dismiss(self.game_number)

