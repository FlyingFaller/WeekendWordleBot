from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Label, Rule, Switch

class SettingsWidgetApp(App):
    """A Textual app to display a title, rule, and switch on one line."""

    # Use CSS to control the layout and appearance of the widgets.
    CSS = """
    Horizontal {
        /* This container will hold our widgets */
        height: auto;
        align: center middle; /* This vertically centers the items in the row */
        padding: 1 2;
    }

    Label {
        /* The title takes only the space it needs */
        width: auto;
    }

    Rule {
        /* The rule will expand to fill all available horizontal space */
        width: 1fr;
        margin: 0 2; /* Add horizontal margin for spacing */
    }
    
    Switch {
        /* The switch takes only the space it needs */
        width: auto;
        border: none;
        padding: 0;
    }
    """

    def compose(self) -> ComposeResult:
        """Yield the widgets that make up the app."""
        # The Horizontal container arranges its children on a single line.
        with Horizontal():
            yield Label("Your Setting")
            # The Rule widget creates the horizontal line.
            # 'dashed' style matches your '-----' request.
            yield Rule()
            # The Switch widget is a functional on/off toggle.
            yield Switch()
        with Horizontal():
            yield Rule()

if __name__ == "__main__":
    app = SettingsWidgetApp()
    app.run()