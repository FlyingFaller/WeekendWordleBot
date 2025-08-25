"""
Defines the sidebar component for the Wordle Solver UI.

- StatsTable: A DataTable widget to display word suggestions and statistics.
- Sidebar: A container for the title and the StatsTable.
"""
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static
from textual.message import Message

from stats_display import StatsDisplay

class StatsTable(Static):
    """A DataTable widget to display word suggestions."""
    
    BORDER_TITLE = "Top Computer Answers"

    def __init__(self) -> None:
        super().__init__()
        self.dummy_rows = [
            ("1", "SLATE", "9.8"), ("2", "CRANE", "9.7"),
            ("3", "TRACE", "9.6"), ("4", "ROAST", "9.5"),
            ("5", "LATER", "9.4"), ("6", "ARISE", "9.3"),
            ("7", "IRATE", "9.2"), ("8", "STARE", "9.1"),
            ("9", "RAISE", "9.0"), ("10", "LEAST", "8.9"),
        ]

    def compose(self) -> ComposeResult:
        """Creates the table and its columns."""
        yield DataTable()

    def on_mount(self) -> None:
        """Adds columns and dummy data to the table."""
        table = self.query_one(DataTable)
        table.cursor_type = 'row'
        table.zebra_stripes = True
        table.add_columns("Rank", "Word", "Score")

        for row in self.dummy_rows:
            table.add_row(*row)
            
        table.move_cursor(row=0)

class Sidebar(Vertical):
    """The sidebar container widget."""

    def compose(self) -> ComposeResult:
        """Renders the sidebar's stats table."""
        yield StatsTable()
        yield StatsDisplay()
