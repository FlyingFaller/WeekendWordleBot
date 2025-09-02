"""
Defines a messenger system to communicate from the backend to a UI.

This module provides a protocol (`UIMessenger`) and two implementations:
- ConsoleMessenger: For command-line output using print/tqdm.
- TextualMessenger: For sending messages to a Textual UI from a worker.
"""

from __future__ import annotations

from typing import Protocol
from tqdm import tqdm

from textual.worker import Worker, get_current_worker
from textual.message import Message


# 1. --- The Protocol (Interface) ---
# This defines the simplified methods for logging and progress bars.

class UIMessenger(Protocol):
    """Defines the interface for sending updates from the backend."""

    def log(self, message: str) -> None:
        """Logs a string message."""
        ...

    def start_progress(self, total: int, desc: str = "") -> None:
        """Starts/resets a progress bar with a new total and description."""
        ...

    def update_progress(self, advance: int = 1) -> None:
        """Advances the progress bar by a given amount."""
        ...

    def stop_progress(self) -> None:
        """Stops and cleans up the current progress bar."""
        ...


# 2. --- The Default Console Implementation ---
# This implementation manages a single tqdm instance.

class ConsoleMessenger:
    """A messenger that prints to the console and uses a tqdm progress bar."""
    def __init__(self):
        self.pbar: tqdm | None = None

    def log(self, message: str) -> None:
        """
        Prints a message. If a progress bar is active, uses its `write`
        method to avoid interfering with the bar's display.
        """
        if self.pbar:
            self.pbar.write(message)
        else:
            print(message)

    def start_progress(self, total: int, desc: str = "") -> None:
        """
        Closes any existing progress bar and starts a new one.
        """
        if self.pbar:
            self.pbar.close()
        self.pbar = tqdm(total=total, desc=desc)

    def update_progress(self, advance: int = 1) -> None:
        """Updates the active progress bar, if it exists."""
        if self.pbar:
            self.pbar.update(advance)

    def stop_progress(self) -> None:
        """Closes the active progress bar, if it exists."""
        if self.pbar:
            self.pbar.close()
            self.pbar = None


# 3. --- The Textual UI Implementation ---
# This posts messages to the Textual UI thread.

class TextualMessenger:
    """A messenger that posts messages to a Textual screen from a worker."""

    # --- Custom Messages for UI Communication ---
    class Log(Message):
        """Post a log message to the UI."""
        def __init__(self, text: str):
            self.text = text
            super().__init__()

    class ProgressStart(Message):
        """Reset and configure the progress bar for a new task."""
        def __init__(self, total: int, description: str):
            self.total = total
            self.description = description
            super().__init__()

    class ProgressUpdate(Message):
        """Advance the progress bar."""
        def __init__(self, advance: int = 1):
            self.advance = advance
            super().__init__()

    class ProgressStop(Message):
        """Signals that the current progress task is complete."""
        def __init__(self):
            super().__init__()

    # --- Messenger Implementation ---
    def __init__(self):
        try:
            # This must be instantiated within a running worker
            self._worker: Worker = get_current_worker()
        except RuntimeError as e:
            raise RuntimeError(
                "TextualMessenger can only be created inside a Textual worker."
            ) from e

    def post_message(self, message: Message) -> None:
        """Helper to post a message from the worker thread."""
        self._worker.post_message(message)

    def log(self, message: str) -> None:
        """Posts a Log message to the screen."""
        self.post_message(self.Log(message))

    def start_progress(self, total: int, desc: str = "") -> None:
        """Posts a ProgressStart message to the screen."""
        self.post_message(self.ProgressStart(total=total, description=desc))

    def update_progress(self, advance: int = 1) -> None:
        """Posts a ProgressUpdate message to the screen."""
        self.post_message(self.ProgressUpdate(advance=advance))

    def stop_progress(self) -> None:
        """Posts a ProgressStop message to the screen."""
        self.post_message(self.ProgressStop())

