from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Collapsible, Label

# ==============================================================================
# New DeletableCollapsible Widget Implementation
# ==============================================================================

class DeletableCollapsible(Collapsible):
    """A Collapsible widget with a delete button in the title bar."""
    # Add new CSS to style the title bar and the delete button
    DEFAULT_CSS = Collapsible.DEFAULT_CSS + """
    DeletableCollapsible > #title-bar {
        height: auto;
    }

    DeletableCollapsible > #title-bar > #delete-button {
        dock: right;
        width: 3;
        height: 1;
        min-width: 3;
        border: none;
        margin: 0 1; /* Margin for spacing */
    }
    """
    class NoFocusButton(Button):
        """A button that cannot be focused."""
        can_focus = False # lowercase is correct, uppercase will do nothing.

    class Delete(Message):
        """Posted when the delete button is clicked.

        Can be handled using `on_deletable_collapsible_delete` in a parent widget.
        """
        def __init__(self, collapsible: "DeletableCollapsible") -> None:
            self.collapsible = collapsible
            super().__init__()

        @property
        def control(self) -> "DeletableCollapsible":
            """The `DeletableCollapsible` that was requested to be deleted."""
            return self.collapsible

    def compose(self) -> ComposeResult:
        """Overrides the original compose to add a title bar and delete button."""
        # Create a container for the title and the button
        with Container(id="title-bar"):
            # The original title widget is used here
            yield self._title
            # The new delete button
            yield self.NoFocusButton("X", id="delete-button", variant='error') 

        # The original contents container is used here
        with self.Contents():
            yield from self._contents_list

    def _on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the delete button being pressed."""
        # Ensure we are only responding to our delete button
        if event.button.id == "delete-button":
            # Stop the event from bubbling up further
            event.stop()
            # Post a message to be caught by the parent application
            self.post_message(self.Delete(self))


# ==============================================================================
# Example Application to Demonstrate the Widget
# ==============================================================================

class DeletableCollapsibleApp(App):
    """An example app to demonstrate the DeletableCollapsible widget."""

    CSS = """
    Screen {
        padding: 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Create the app layout and widgets."""
        with VerticalScroll():
            yield DeletableCollapsible(
                Label("This is the content of the 'General Load' section."),
                title="General Load",
            )
            yield DeletableCollapsible(
                Label("This is the content of the 'Scrape Words' section."),
                title="Scrape Words",
                collapsed=False, # This one starts expanded
            )
            yield DeletableCollapsible(
                Label("This is the content of the 'Another Section'."),
                title="Another Section",
            )

    def on_deletable_collapsible_delete(self, event: DeletableCollapsible.Delete) -> None:
        """Handle the delete message from the collapsible widget."""
        # Log the action to the textual console (press F12 to view)
        self.log(f"Removing collapsible: '{event.collapsible.title}'")
        # Remove the widget that sent the message
        event.collapsible.remove()

if __name__ == "__main__":
    app = DeletableCollapsibleApp()
    app.run()
