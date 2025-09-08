"""
Defines the sidebar and its child components for the Wordle Solver UI.

- ResultsTable: A DataTable widget to display word suggestions and scores.
- StatsTable: A DataTable to display miscellaneous game statistics.
- Sidebar: A container for the ResultsTable and StatsTable.
"""
import random
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static

class ResultsTable(Static):
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

class StatsTable(Static):
    """A widget to display miscellaneous game statistics."""

    BORDER_TITLE = "Stats"

    EVENTS = [
        ('cache_hits', 'Cache hits'), ('entropy_skips', 'Entropy loop skips'),
        ('entropy_exits', 'Entropy loop exits'), ('winning_patterns', 'Winning patterns found'),
        ('low_pattern_counts', 'Low answer count patterns found'), ('recursions_queued', 'Recursions queued'),
        ('depth_limit', 'Depth limits reached while recursing'),
        ('mins_exceeded_simple', 'Min scores exceeded during simple calcs'),
        ('recursions_called', 'Recursions called'), ('mins_exceeded_recurse', 'Min scores exceeded during recursion'),
        ('mins_after_recurse', 'New min scores found after recursing'),
        ('mins_without_recurse', 'New min scores found without recursing'),
        ('leaf_calcs_complete', 'Leaf node calculations completed in full'),
    ]

    def compose(self) -> ComposeResult:
        """Creates the stats table."""
        yield DataTable()

    def on_mount(self) -> None:
        """Populates the stats table with dummy data."""
        table = self.query_one(DataTable)
        table.cursor_type = 'none'
        table.add_columns("Statistic", "Value")
        table.can_focus = False
        table.zebra_stripes = True

        for _, description in self.EVENTS:
            value = f"{random.randint(1000, 100000):,}"
            table.add_row(description, value)

        table.add_row("Cache entries", f"{random.randint(500, 2000):,}")
        table.add_row("Cache segments", f"{random.randint(5, 20):,}")

class Sidebar(Vertical):
    """The sidebar container widget."""

    def compose(self) -> ComposeResult:
        """Renders the sidebar's tables."""
        yield ResultsTable()
        yield StatsTable()