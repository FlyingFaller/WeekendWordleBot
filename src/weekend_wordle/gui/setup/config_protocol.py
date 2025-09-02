from typing import Protocol

class Configurable(Protocol):
    """A protocol for widgets that can provide configuration."""
    def get_config(self) -> dict[str]:
        """Returns a dictionary of the widget's configuration."""
        ...