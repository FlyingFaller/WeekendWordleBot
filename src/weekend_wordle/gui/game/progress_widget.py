"""
Defines a custom, titled progress bar component.
"""
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import ProgressBar
from textual.color import Gradient

class TitledProgressBar(Container):
    """A container that holds a ProgressBar and gives it a border title."""
    def __init__(self, title: str = None, total: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_title = title
        self._total = total

    def compose(self) -> ComposeResult:
        """Create the layout."""
        # The actual ProgressBar widget goes inside this container.
        gradient = Gradient.from_colors("#4795de", "#bb637a")

        yield ProgressBar(total = self._total, id="progress", gradient=gradient)
