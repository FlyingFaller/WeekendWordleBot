from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Static
from textual.binding import Binding

class ConfirmationDialog(Screen):
    """A modal confirmation dialog screen."""
    BINDINGS = [
        ("left", "focus_previous", "Focus Previous"),
        ("right", "focus_next", "Focus Next")
    ]
    # This CSS will be applied to the screen. It makes it modal.
    CSS = """
    ConfirmationDialog {
        background: $screen-background 30%;
        align: center middle;
    }

    #dialog {
        width: auto;
        height: auto;
        background: $surface;
        border: tall $gradient-start;
        padding: 1 2;
    }

    #question {
        /* Ensure the question is centered and has space below it */
        width:auto;
        height: auto;
        max-width: 40;
        text-align: center;
    }

    #confrim_buttons {
        width: 100%;
        height: auto;
        align: center middle;
    }

    #confrim_buttons Button {
        margin: 0 1;
        margin-top: 1;
    }

    #confirm {
        background: $tile-green;
    }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        """Create the content of the dialog."""
        with Container(id='dialog'):
            yield Static(self.message, id="question")
            with Horizontal(id='confrim_buttons'):
                yield Button("Cancel", variant="error", id="cancel")
                yield Button("Confirm", variant="success", id="confirm")
        

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "confirm":
            # When dismissing, we can pass a result back.
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_focus_next(self):
        self.focus_next(selector=Button)

    def action_focus_previous(self):
        self.focus_previous(selector=Button)