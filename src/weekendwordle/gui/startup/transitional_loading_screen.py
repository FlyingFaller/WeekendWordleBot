import asyncio
from textual.screen import Screen
from textual.widgets import LoadingIndicator
from textual.containers import Vertical
from textual.app import ComposeResult

class TransitionalLoading(Screen):
    """A temporary loading screen that runs an import in a worker."""
    DEFAULT_CSS = Screen.DEFAULT_CSS + """
    Screen {
        background: $screen-background;
    }
    """

    def compose(self) -> ComposeResult:
        """Render a centred loading indicator."""
        with Vertical(id="loading-dialog"):
            yield LoadingIndicator()

    def on_mount(self) -> None:
        """Starts the worker to load the next screen."""
        self.run_worker(self.load_setup_screen, exclusive=True)

    def _import_setup_screen(self):
        """
        Helper method to contain the synchronous, blocking import.
        This will be run in a separate thread.
        """
        from ..setup.setup_screen import SetupScreen
        return SetupScreen

    async def load_setup_screen(self) -> None:
        """An async worker that imports the screen in a thread and then switches."""
        SetupScreen = await asyncio.to_thread(self._import_setup_screen)
        self.app.switch_screen(SetupScreen())