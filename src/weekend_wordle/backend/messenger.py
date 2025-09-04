"""
Defines a messenger system to communicate from the backend to a UI.

This module provides a protocol (`UIMessenger`) and two implementations:
- ConsoleMessenger: For command-line output using print/tqdm.
- TextualMessenger: For sending messages to a Textual UI from a worker.

This version separates formatting from logging. The `log` method is a simple
pipe, while the `task_log` method handles all complex formatting for tasks.
"""

import time
from contextlib import contextmanager
from typing import Protocol, Literal, Iterator

from tqdm import tqdm
from textual.worker import Worker, get_current_worker
from textual.message import Message

# --- Type Definitions ---
LogLevel = Literal["INFO", "STEP", "SUCCESS", "WARNING", "ERROR", "NONE"]
TaskRole = Literal["header", "footer", "message"]

class UIMessenger(Protocol):
    """Defines the interface for sending updates from the backend."""

    def log(self, message: str, level: LogLevel = "NONE", prefix: str = "", suffix: str = "") -> None:
        """Logs a string message, with optional styling, prefix, and suffix."""
        ...

    def task_log(self, message: str, level: LogLevel, role: TaskRole = "message") -> None:
        """Formats and logs a message within the context of a task."""
        ...

    @contextmanager
    def task(self, description: str) -> Iterator[None]:
        """A context manager for grouping and timing a major task."""
        ...

    def start_progress(self, total: float, desc: str = "") -> None:
        """Starts/resets a progress bar with a new total and description."""
        ...

    def update_progress(self, advance: float = 1) -> None:
        """Advances the progress bar by a given amount."""
        ...

    def stop_progress(self) -> None:
        """Stops and cleans up the current progress bar."""
        ...

class ConsoleMessenger:
    """A messenger that prints to the console and uses a tqdm progress bar."""
    def __init__(self):
        self.pbar: tqdm | None = None
        self._indent_level = 0
        self._prefixes = {
            "INFO": "", "STEP": "", "SUCCESS": "",
            "WARNING": "Warning: ", "ERROR": "Error: ", "NONE": ""
        }

    def log(self, message: str, level: LogLevel = "NONE", prefix: str = "", suffix: str = "") -> None:
        """Logs a pre-formatted string directly to the console."""
        level_prefix = self._prefixes.get(level, "")
        output = f"{prefix}{level_prefix}{message}{suffix}"
        if self.pbar:
            self.pbar.write(output)
        else:
            print(output)

    def task_log(self, message: str, level: LogLevel, role: TaskRole = "message") -> None:
        """Formats a message with Rich markup and logs it."""
        indent_str = "│   "

        if role == "header":
            prefix = f"{indent_str*(self._indent_level-1)}╭── "
            self.log(message, level="STEP", prefix=prefix)
        elif role == "footer":
            prefix = f"{indent_str*(self._indent_level-1)}╰── "
            self.log(message, level=level, prefix=prefix)
        else: # role == "message"
            prefix = indent_str * self._indent_level
            self.log(message, level=level, prefix=prefix)

    @contextmanager
    def task(self, description: str) -> Iterator[None]:
        """Orchestrates a visually grouped task."""
        start_time = time.monotonic()
        self._indent_level += 1
        self.task_log(description, level="STEP", role="header")

        try:
            yield
            duration = time.monotonic() - start_time
            self.task_log(f"Success ({duration:.2f}s)", level="SUCCESS", role="footer")
            self._indent_level -= 1
            self.task_log("", level="NONE", role="message")

        except Exception as e:
            duration = time.monotonic() - start_time
            self.task_log(f"An exception occurred: {e}", level="ERROR", role="message")
            self.task_log(f"Failed ({duration:.2f}s)", level="ERROR", role="footer")
            self._indent_level -= 1
            self.task_log("", level="NONE", role="message")
            raise

    def start_progress(self, total: float, desc: str = "") -> None:
        """
        Closes any existing progress bar and starts a new one.
        """
        if self.pbar:
            self.pbar.close()
        self.pbar = tqdm(total=total, desc=desc)

    def update_progress(self, advance: float = 1) -> None:
        """Updates the active progress bar, if it exists."""
        if self.pbar:
            self.pbar.update(advance)

    def stop_progress(self) -> None:
        """Closes the active progress bar, if it exists."""
        if self.pbar:
            self.pbar.close()
            self.pbar = None

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
        def __init__(self, total: float, description: str):
            self.total = total
            self.description = description
            super().__init__()

    class ProgressUpdate(Message):
        """Advance the progress bar."""
        def __init__(self, advance: float = 1):
            self.advance = advance
            super().__init__()

    class ProgressStop(Message):
        """Signals that the current progress task is complete."""
        def __init__(self):
            super().__init__()

    # --- Messenger Implementation ---
    def __init__(self):
        try:
            self._worker: Worker = get_current_worker()
        except RuntimeError as e:
            raise RuntimeError("TextualMessenger can only be created inside a Textual worker.") from e
        self._indent_level = 0
        self._styles = {
            "INFO": "dim", "STEP": "bold", "SUCCESS": "bold green",
            "WARNING": "bold yellow", "ERROR": "bold red", "NONE": ""
        }

    def post_message(self, message: Message) -> None:
        """Helper to post a message from the worker thread."""
        self._worker._node.post_message(message)

    def log(self, message: str, level: LogLevel = "NONE", prefix: str = "", suffix: str = "") -> None:
        """Posts a message with optional styling, prefix, and suffix."""
        style = self._styles.get(level, "")
        
        if style:
            styled_message = f"[{style}]{message}[/]"
        else:
            styled_message = message
            
        final_message = f"{prefix}{styled_message}{suffix}"
        self.post_message(self.Log(final_message))

    def task_log(self, message: str, level: LogLevel, role: TaskRole = "message") -> None:
        """Formats a message with Rich markup and logs it."""
        indent_str = "[dim]│[/dim]   "

        if role == "header":
            prefix = f"{indent_str*(self._indent_level-1)}[dim]╭── [/dim]"
            self.log(message, level="STEP", prefix=prefix)
        elif role == "footer":
            prefix = f"{indent_str*(self._indent_level-1)}[dim]╰── [/dim]"
            self.log(message, level=level, prefix=prefix)
        else: # role == "message"
            prefix = indent_str * self._indent_level
            self.log(message, level=level, prefix=prefix)

    @contextmanager
    def task(self, description: str) -> Iterator[None]:
        """Orchestrates a visually grouped task."""
        start_time = time.monotonic()
        self._indent_level += 1
        self.task_log(description, level="STEP", role="header")

        try:
            yield
            duration = time.monotonic() - start_time
            self.task_log(f"Success ({duration:.2f}s)", level="SUCCESS", role="footer")
            self._indent_level -= 1
            self.task_log("", level="NONE", role="message")

        except Exception as e:
            duration = time.monotonic() - start_time
            self.task_log(f"An exception occurred: {e}", level="ERROR", role="message")
            self.task_log(f"Failed ({duration:.2f}s)", level="ERROR", role="footer")
            self._indent_level -= 1
            self.task_log("", level="NONE", role="message")
            raise

    def start_progress(self, total: float, desc: str = "") -> None:
        """Posts a ProgressStart message to the screen."""
        self.post_message(self.ProgressStart(total=total, description=desc))

    def update_progress(self, advance: float = 1) -> None:
        """Posts a ProgressUpdate message to the screen."""
        self.post_message(self.ProgressUpdate(advance=advance))

    def stop_progress(self) -> None:
        """Posts a ProgressStop message to the screen."""
        self.post_message(self.ProgressStop())

