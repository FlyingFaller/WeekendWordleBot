from __future__ import annotations
import uuid
from typing import Callable

from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal, Container
from textual.message import Message
from textual.widget import Widget
from textual import events
from textual.widgets import (
    Collapsible,
    Static,
    Button,
    Footer,
    Input,
    Switch,
    Select,
)

class LoadingWidget(Container):
    """A generic, extensible widget for configuring data loading."""

    def __init__(
        self,
        title: str,
        savefile_path: str = "",
        url: str = "",
        refetch: bool = False,
        save: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.border_title = title
        self._savefile_path = savefile_path
        self._url = url
        self._refetch = refetch
        self._save = save

    def compose_inputs(self) -> ComposeResult:
        """Yields the input field widgets. Can be overridden by subclasses."""
        with Horizontal(classes="input-row"):
            yield Static("Save File Path:")
            yield Input(
                value=self._savefile_path,
                placeholder="e.g., data/my_words.txt",
                id="savefile",
                compact=True
            )
        with Horizontal(classes="input-row"):
            yield Static("URL:")
            yield Input(value=self._url,
                        placeholder="e.g., https://...",
                        id="url",
                        compact=True)

    def compose_switches(self) -> ComposeResult:
        """Yields the switch widgets. Can be overridden by subclasses."""
        with Horizontal(classes="switch-group"):
            yield Switch(value=self._refetch, id="refetch")
            yield Static("Refetch/Recompute")
        with Horizontal(classes="switch-group"):
            yield Switch(value=self._save, id="save")
            yield Static("Save to File")

    def compose(self) -> ComposeResult:
        """Create child widgets for the loading configuration."""
        yield from self.compose_inputs()
        with Container(classes="switch-container"):
            yield from self.compose_switches()

    def get_config(self) -> dict:
        """Returns the current configuration from the UI widgets."""
        return {
            "savefile": self.query_one("#savefile", Input).value,
            "url": self.query_one("#url", Input).value,
            "refetch": self.query_one("#refetch", Switch).value,
            "save": self.query_one("#save", Switch).value,
        }


class GetWordsWidget(LoadingWidget):
    """A specialized widget for loading word lists from files or URLs."""

    def __init__(
        self,
        include_uppercase: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._include_uppercase = include_uppercase

    def compose_switches(self) -> ComposeResult:
        """Yields the base switches and the new 'include_uppercase' switch."""
        yield from super().compose_switches()
        with Horizontal(classes="switch-group"):
            yield Switch(value=self._include_uppercase, id="include_uppercase")
            yield Static("Incl. Uppercase")

    def get_config(self) -> dict:
        """Returns the current configuration, including the extra switch."""
        config = super().get_config()
        config["include_uppercase"] = self.query_one("#include_uppercase", Switch).value
        return config

class ScrapeWordsWidget(LoadingWidget):
    """A specialized widget for scraping words from a website."""

    def __init__(self, header: str = "All Wordle answers,h2", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._header = header

    def compose_inputs(self) -> ComposeResult:
        """Yields base inputs and the new 'header' input."""
        yield from super().compose_inputs()
        with Horizontal(classes="input-row"):
            yield Static("Header (text,tag):")
            yield Input(value=self._header, id="header", compact=True)

    def get_config(self) -> dict:
        """Returns the config, including the header."""
        config = super().get_config()
        header_text = self.query_one("#header", Input).value
        config["header"] = tuple(header_text.split(','))
        return config

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

class CollapsibleWithClose(Collapsible):
    """
    A Collapsible widget that manages its own close button as an overlay.
    """

    class Toggled(Message):
        """Posted when the collapsible is toggled."""

    def __init__(self, content_widget: Widget, *, title: str, **kwargs):
        """
        Initializes the widget with a specific content widget.
        """
        # Pass the content widget as a child to the parent Collapsible
        super().__init__(content_widget, title=title, **kwargs)
        # Keep a direct reference to the content widget
        self.content_widget = content_widget
        # Generate a unique ID for the button to avoid conflicts.
        self._close_button_id = f"close-button-{uuid.uuid4().hex}"
        # Create the button that will be overlaid using our non-focusable class.
        self._close_button = NoFocusButton("x", id=self._close_button_id, variant="error")
        # Add a CSS class so we can style all close buttons consistently.
        self._close_button.add_class("close-button")
        # Add a reference from the button back to this widget instance.
        self._close_button._owner_widget = self


    def _watch_collapsed(self, collapsed: bool) -> None:
        """
        Called when the collapsed state changes.
        We override this to post a message so the parent can react.
        """
        super()._watch_collapsed(collapsed)
        self.post_message(self.Toggled())

    def _reposition_close_button(self) -> None:
        """
        Calculates and sets the position of the close button.
        The position is now relative to the parent CollapsibleList.
        """
        if not self.is_mounted or not self.parent:
            return
        
        # Use the parent's content_region to correctly calculate offsets
        # when the parent has a border and/or padding.
        parent_content_region = self.parent.content_region

        # The button's X offset is relative to the parent's content area.
        offset_x = parent_content_region.width - 4

        # The button's Y offset is the child's Y position relative to the
        # parent's content area, plus 1 for the child's own top padding.
        offset_y = (self.region.y - parent_content_region.y) + 1

        self._close_button.styles.offset = (offset_x, offset_y)

    def on_mount(self) -> None:
        """
        Called when the widget is mounted.
        We mount the button to our parent (the list) and schedule its repositioning.
        """
        self.parent.mount(self._close_button)
        self.call_later(self._reposition_close_button)

    def on_resize(self) -> None:
        """
        Called when the widget is resized.
        We must reposition the button whenever the parent's size changes.
        """
        self._reposition_close_button()

    def remove(self) -> None:
        """
        Overrides the default remove method to also remove the button.
        """
        self._close_button.remove()
        super().remove()

class CollapsibleList(Vertical):
    """A container for a vertical list of CollapsibleWithClose widgets."""

    def __init__(self, *, title: str, widget_constructors: dict[str, Callable[[], Widget]], **kwargs) -> None:
        super().__init__(**kwargs)
        self.widget_constructors = widget_constructors
        self.border_title = title

    def compose(self) -> ComposeResult:
        """Creates the Select control and a container for the list items."""
        select_options = [(name, name) for name in self.widget_constructors.keys()]
        # Use our new HoverSelect widget.
        yield HoverSelect(
            options=select_options,
            prompt="Add new item...",
            allow_blank=True,
            id="add-item-select",
        )

    def add_item(self, title: str, content_widget: Widget) -> CollapsibleWithClose:
        """Adds a new collapsible item to the list, wrapping the content widget."""
        # Programmatically remove the border by setting its style to a tuple.
        # This satisfies the style parser and correctly removes the border.
        content_widget.styles.border = ("none", "transparent")
        
        new_item = CollapsibleWithClose(
            content_widget,
            title=title,
            collapsed=True,
        )
        self.mount(new_item)
        self.call_later(self._update_all_buttons)
        return new_item

    def get_config(self) -> list[dict]:
        """
        Retrieves the configuration from all child widgets in the list.
        """
        config_list = []
        # Find all CollapsibleWithClose children.
        for item in self.query(CollapsibleWithClose):
            # Get the inner content widget (e.g., LoadingWidget).
            content_widget = item.content_widget
            # Check if it has a get_config method.
            if hasattr(content_widget, "get_config"):
                # Call get_config to get the widget's specific data.
                config = content_widget.get_config()
                # Add the widget_type for context.
                config["widget_type"] = item.title
                config_list.append(config)
        return config_list

    def on_resize(self) -> None:
        """
        When the list itself resizes (e.g. after a child toggles),
        this is a reliable trigger to update all button positions.
        """
        self.call_later(self._update_all_buttons)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handles button presses for buttons mounted inside this list."""
        if hasattr(event.button, "_owner_widget"):
            owner_widget = event.button._owner_widget
            owner_widget.remove()
            self.call_later(self._update_all_buttons)
            event.stop()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handles adding a new item when a Select option is chosen."""
        event.stop()
        if event.value == Select.BLANK:
            return
        
        if str(event.value) in self.widget_constructors:
            constructor = self.widget_constructors[event.value]
            new_widget = constructor()
            new_item = self.add_item(title=str(event.value), content_widget=new_widget)
            new_item.scroll_visible()

        event.control.clear()


    def _update_all_buttons(self) -> None:
        """
        Iterate through all children and tell them to reposition their buttons.
        """
        for child in self.query(CollapsibleWithClose):
            child._reposition_close_button()

class CloseableCollapsibleApp(App):
    """
    A Textual app to demonstrate a list of CollapsibleWithClose widgets.
    """
    CSS_PATH = "collapsible_demo.tcss"

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        # Define the available widgets that can be added to the list.
        widget_constructors: dict[str, Callable[[], Widget]] = {
            "General Load": lambda: LoadingWidget(title="File Source"),
            "Get Words": lambda: GetWordsWidget(title="Get Words Config"),
            "Scrape Words": lambda: ScrapeWordsWidget(title="Scrape Words Config"),
        }
        # Pass the configuration to our self-contained list widget.
        yield CollapsibleList(
            title="Dynamic Widget List",
            widget_constructors=widget_constructors
        )
        # Add a button to test the get_config method.
        yield Button("Print Config to Log", id="print-config")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the test button press."""
        if event.button.id == "print-config":
            config = self.query_one(CollapsibleList).get_config()
            self.log("--- WIDGET CONFIGURATION ---")
            self.log(config)
            self.log("--------------------------")


if __name__ == "__main__":
    app = CloseableCollapsibleApp()
    app.run()
