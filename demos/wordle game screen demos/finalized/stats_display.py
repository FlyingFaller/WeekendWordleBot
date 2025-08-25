"""
Defines the stats display widget for the lower-left corner of the UI.
"""
import random
from textual.app import ComposeResult
from textual.widgets import DataTable, Static

class StatsDisplay(Static):
    """A widget to display miscellaneous game statistics."""

    BORDER_TITLE = "Stats"

    # The list of events to display, taken from your example
    EVENTS = [
        ('cache_hits', 'Cache hits'),
        ('entropy_skips', 'Entropy loop skips'),
        ('entropy_exits', 'Entropy loop exits'),
        ('winning_patterns', 'Winning patterns found'),
        ('low_pattern_counts', 'Low answer count patterns found'),
        ('recursions_queued', 'Recursions queued'),
        ('depth_limit', 'Depth limits reached while recursing'),
        ('mins_exceeded_simple', 'Min scores exceeded during simple calcs'),
        ('recursions_called', 'Recursions called'),
        ('mins_exceeded_recurse', 'Min scores exceeded during recursion'),
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
        # FIXED: Disable the cursor to prevent highlighting and interaction
        table.cursor_type = 'none'
        table.add_columns("Statistic", "Value")
        table.can_focus = False
        table.zebra_stripes = True

        # Add the event stats with random numbers
        for _, description in self.EVENTS:
            value = f"{random.randint(1000, 100000):,}"
            table.add_row(description, value)

        # Add the cache stats with random numbers
        table.add_row("Cache entries", f"{random.randint(500, 2000):,}")
        table.add_row("Cache segments", f"{random.randint(5, 20):,}")