from typing import Callable

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import (
    Button,
    Collapsible,
    Select,
    Static,
)

class NoFocusButton(Button):
    """A button that cannot be focused."""
    can_focus = False # lowercase is correct, uppercase will do nothing.

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
    DeletableCollapsible > #title_bar {
        height: auto;
    }

    DeletableCollapsible > #title_bar > #delete_button {
        dock: right;
        width: 3;
        height: 1;
        min-width: 3;
        border: none;
        margin: 0 1; /* Margin for spacing */
    }
    """

    class Delete(Message):
        """Posted when the delete button is clicked.

        Can be handled using `on_deletable_collapsible_delete` in a parent widget.
        """
        def __init__(self, collapsible: "DeletableCollapsible") -> None:
            super().__init__()
            self.collapsible = collapsible

        @property
        def control(self) -> "DeletableCollapsible":
            """The `DeletableCollapsible` that was requested to be deleted."""
            return self.collapsible

    def compose(self) -> ComposeResult:
        """Overrides the original compose to add a title bar and delete button."""
        # Create a container for the title and the button
        with Container(id="title_bar"):
            # The original title widget is used here
            yield self._title
            # The new delete button
            yield NoFocusButton("X", id="delete_button", variant='error') 

        # The original contents container is used here
        with self.Contents():
            yield from self._contents_list

    def _on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the delete button being pressed."""
        # Ensure we are only responding to our delete button
        if event.button.id == "delete_button":
            # Stop the event from bubbling up further
            event.stop()
            # Post a message to be caught by the parent application
            self.post_message(self.Delete(self))

class DynamicCollapsibleList(VerticalScroll):
    """A container for a dynamic list of DeletableCollapsible widgets."""

    def __init__(
        self,
        *,
        title: str = None,
        widget_constructors: dict[str, Callable[[], Widget]] = {},
        default_widgets: list[tuple[str, Widget]] = [],
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.border_title = title
        self.default_widgets = default_widgets
        self._full_widget_constructors = widget_constructors.copy()
        self._disabled_options: set[str] = set()


    def on_mount(self) -> None:
        """Called when the widget is mounted to populate default items."""
        for item_name, content_widget in self.default_widgets:
            self.add_item(item_name, content_widget)

    def _generate_select_options(self) -> list[tuple[Text | str, str]]:
        """Generates the list of options for the Select widget."""
        options = []
        for name in self._full_widget_constructors:
            if name in self._disabled_options:
                # If disabled, create a styled Text object
                prompt = Text(name, style="strike #888888")
                options.append((prompt, name))
            else:
                # Otherwise, use a simple string
                options.append((name, name))
        return options

    def compose(self) -> ComposeResult:
        """Creates the Select control for adding new items."""
        yield HoverSelect(
            options=self._generate_select_options(),
            prompt="Add new item...",
            allow_blank=True,
            id="add_item_select",
            compact=True
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handles when a Select option is chosen."""
        event.stop()

        if event.value != Select.BLANK:
            item_name = str(event.value)

            # Prevent adding the item if it's currently disabled.
            if item_name in self._disabled_options:
                self.log(f"Attempted to select disabled item: '{item_name}'")
            else:
                constructor = self._full_widget_constructors[item_name]
                content_widget = constructor()
                self.add_item(item_name, content_widget)

        # Always clear the select prompt after a selection attempt.
        event.control.clear()


    def add_item(self, item_name: str, content_widget: Widget) -> None:
        """Handles adding a new item"""
        # Remove widget instance's border
        content_widget.styles.border = ("none", "transparent")
        # Create and mount the new collapsible item
        new_item = DeletableCollapsible(content_widget, title=item_name)
        self.mount(new_item)

    def on_deletable_collapsible_delete(self, event: DeletableCollapsible.Delete) -> None:
        """Handles the delete message from a child collapsible."""
        event.stop()
        self.log(f"Removing item: '{event.collapsible.title}'")
        event.collapsible.remove()

    def _update_select_options(self) -> None:
        """Helper method to refresh the options in the Select widget."""
        select_widget = self.query_one("#add_item_select", HoverSelect)
        select_widget.set_options(self._generate_select_options())

    def update_classifier_dependency(self, classifier_enabled: bool) -> None:
        """
        Updates available filters based on the classifier's state.
        Disables the 'Classifier Probability Filter' if the classifier is disabled.
        """
        filter_name = "Classifier Probability Filter"

        if classifier_enabled:
            # Mark the option as enabled by removing it from the disabled set.
            self._disabled_options.discard(filter_name)
        else:
            # Mark the option as disabled.
            self._disabled_options.add(filter_name)
            
            # Also, find and remove any existing instance of this filter from the list.
            for item in self.query(DeletableCollapsible):
                if item.title == filter_name:
                    item.remove()
                    self.log(f"Automatically removed '{filter_name}' as classifier is disabled.")
        
        # Refresh the options displayed in the dropdown to reflect the new state.
        self._update_select_options()

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
    
class SimpleDynamicListWidget(Vertical):
    """A widget that dynamically manages a list of other widgets."""

    def __init__(
        self,
        item_factory,
        items: list | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.item_factory = item_factory
        self._items = items or []

    def compose(self) -> ComposeResult:
        """Create child widgets for the dynamic list."""
        for item_data in self._items:
            yield self.item_factory(item_data)

    def get_items(self) -> list[Widget]:
        """Returns the list of item widgets."""
        # We need to filter out any potential non-item widgets if the list gets complex
        return [
            widget for widget in self.children if isinstance(widget, Widget)
        ]