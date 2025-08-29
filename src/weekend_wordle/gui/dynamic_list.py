from typing import Callable

from textual import events
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import (
    Button,
    Collapsible,
    Select,
    Static,
)

class HoverSelect(Select):
    """A Select widget that opens its dropdown on hover."""
    def on_enter(self, event: events.Enter) -> None:
        """Open the Select dropdown on hover."""
        if not self.expanded:
            self.action_show_overlay()

    def on_leave(self, event: events.Leave) -> None:
        """Close the Select dropdown when the mouse leaves."""
        if self.expanded and not isinstance(event.control, Static):
            self.expanded = False

class DeletableCollapsible(Collapsible):
    """A Collapsible widget with a delete button in the title bar."""
    # Add new CSS to style the title bar and the delete button
    DEFAULT_CSS = Collapsible.DEFAULT_CSS + """
    DeletableCollapsible > #title-bar {
        height: auto;
    }

    DeletableCollapsible > #title-bar > #delete-button {
        dock: right;
        width: 3;
        height: 1;
        min-width: 3;
        border: none;
        margin: 0 1; /* Margin for spacing */
    }
    """
    class NoFocusButton(Button):
        """A button that cannot be focused."""
        can_focus = False # lowercase is correct, uppercase will do nothing.

    class Delete(Message):
        """Posted when the delete button is clicked.

        Can be handled using `on_deletable_collapsible_delete` in a parent widget.
        """
        def __init__(self, collapsible: "DeletableCollapsible") -> None:
            self.collapsible = collapsible
            super().__init__()

        @property
        def control(self) -> "DeletableCollapsible":
            """The `DeletableCollapsible` that was requested to be deleted."""
            return self.collapsible

    def compose(self) -> ComposeResult:
        """Overrides the original compose to add a title bar and delete button."""
        # Create a container for the title and the button
        with Container(id="title-bar"):
            # The original title widget is used here
            yield self._title
            # The new delete button
            yield self.NoFocusButton("X", id="delete-button", variant='error') 

        # The original contents container is used here
        with self.Contents():
            yield from self._contents_list

    def _on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the delete button being pressed."""
        # Ensure we are only responding to our delete button
        if event.button.id == "delete-button":
            # Stop the event from bubbling up further
            event.stop()
            # Post a message to be caught by the parent application
            self.post_message(self.Delete(self))

class DynamicCollapsibleList(VerticalScroll):
    """A container for a dynamic list of DeletableCollapsible widgets."""

    def __init__(
        self,
        *,
        title: str,
        widget_constructors: dict[str, Callable[[], Widget]],
        default_widgets: list[tuple[str, Widget]] = [],
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.border_title = title
        self.widget_constructors = widget_constructors
        self.default_widgets = default_widgets

    def on_mount(self) -> None:
        """Called when the widget is mounted to populate default items."""
        for item_name, content_widget in self.default_widgets:
            self.add_item(item_name, content_widget)

    def compose(self) -> ComposeResult:
        """Creates the Select control for adding new items."""
        select_options = [(name, name) for name in self.widget_constructors.keys()]
        yield HoverSelect(
            options=select_options,
            prompt="Add new item...",
            allow_blank=True,
            id="add-item-select",
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handles when a Select option is chosen."""
        event.stop()
        if event.value != Select.BLANK:
            item_name = str(event.value)
            # Get the constructor from the dictionary
            constructor = self.widget_constructors[item_name]
            # Create an instance of the content widget
            content_widget = constructor()
            # Add the item
            self.add_item(item_name, content_widget)
        # Clear the select prompt
        event.control.clear()

    def add_item(self, item_name: str, content_widget: Widget) -> None:
        """Handles adding a new item"""
        # Remove widget instance's border
        content_widget.styles.border = ("none", "transparent")
        # Create and mount the new collapsible item
        new_item = DeletableCollapsible(content_widget, title=item_name)
        self.mount(new_item)
        new_item.scroll_visible()

    def on_deletable_collapsible_delete(self, event: DeletableCollapsible.Delete) -> None:
        """Handles the delete message from a child collapsible."""
        event.stop()
        self.log(f"Removing item: '{event.collapsible.title}'")
        event.collapsible.remove()

    def get_config(self) -> list[dict]:
            """
            Retrieves the configuration from all child widgets in the list.
            """
            config_list = []
            # Find all DeletableCollapsible children within this widget.
            for item in self.query(DeletableCollapsible):
                # The actual content widget (e.g., LoadingWidget) is inside the Contents container.
                # We can query for it. Since there's only one, we can grab the first result.
                content_widget = item.query_one(Widget)
                
                # Check if the content widget has a get_config method.
                if hasattr(content_widget, "get_config"):
                    # Call get_config to get the widget's specific data.
                    config = content_widget.get_config()
                    # Add the widget_name for context, using the collapsible's title.
                    config["widget_name"] = item.title
                    config_list.append(config)
            return config_list