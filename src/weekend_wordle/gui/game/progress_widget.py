"""
Defines a custom, titled progress bar component.
"""
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import ProgressBar

class TitledProgressBar(Container):
    """A container that holds a ProgressBar and gives it a border title."""

    BORDER_TITLE = "Computation Progress"

    def compose(self) -> ComposeResult:
        """Create the layout."""
        # The actual ProgressBar widget goes inside this container.
        yield ProgressBar(total=100, id="progress")
