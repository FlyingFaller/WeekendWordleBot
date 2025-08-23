from textual.app import App, ComposeResult
from textual.widgets import ProgressBar


class MinimalProgressBarApp(App[None]):
    """A minimal, full-width progress bar example."""

    # Link to the external stylesheet.
    CSS_PATH = "progress_bar.tcss"

    def compose(self) -> ComposeResult:
        """Create the layout."""
        # Create a determinate progress bar.
        yield ProgressBar(total=100)

    def on_mount(self) -> None:
        """Called once the app is ready."""
        # Set a timer to call the `make_progress` method every 0.1 seconds.
        self.set_interval(1 / 10, self.make_progress)

    def make_progress(self) -> None:
        """Called by the timer to advance the progress bar."""
        # Get the ProgressBar widget.
        progress_bar = self.query_one(ProgressBar)
        
        # Advance the progress by 1 step.
        # If the progress is complete, reset it to 0.
        if progress_bar.progress < 100:
            progress_bar.advance(1)
        else:
            progress_bar.progress = 0


if __name__ == "__main__":
    MinimalProgressBarApp().run()
