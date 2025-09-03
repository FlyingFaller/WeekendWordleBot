from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Input, Static, Switch

from weekend_wordle.gui.setup.dynamic_list_widget import SimpleDynamicListWidget, NoFocusButton
from weekend_wordle.gui.setup.loading_widget import GetWordsWidget, LoadingWidget


class _SuffixRuleWidget(Vertical):
    """A widget for a single suffix rule, with inputs for suffix and exceptions."""

    def __init__(self, rule: str | tuple[str, ...] = "") -> None:
        super().__init__()
        if isinstance(rule, tuple):
            self.suffix = rule[0]
            # Join all exception elements with a comma for display
            self.exception = ", ".join(rule[1:])
        else:
            self.suffix = rule
            self.exception = ""

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static("Suffix:")
            yield Input(
                value=self.suffix,
                placeholder="e.g., s",
                id="suffix_input",
                compact=True,
            )
            yield Static("Exceptions (comma-sep):")
            yield Input(
                value=self.exception,
                placeholder="e.g., s, r",
                id="exception_input",
                compact=True,
            )
            yield NoFocusButton("Remove", id="remove_button", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the remove button press to remove the widget itself."""
        if event.button.id == "remove_button":
            self.remove()
            event.stop()

    def get_rule(self) -> str | tuple[str, ...] | None:
        """Returns the suffix rule from the inputs."""
        suffix = self.query_one("#suffix_input", Input).value.strip()
        exceptions_str = self.query_one("#exception_input", Input).value.strip()

        if not suffix:
            return None

        if exceptions_str:
            # Split the comma-separated string back into a tuple of exceptions
            exceptions = tuple(e.strip() for e in exceptions_str.split(","))
            return (suffix,) + exceptions
        return suffix


class FilterSuffixWidget(GetWordsWidget):
    """A widget to filter words by suffixes with a dynamic list of rules."""

    def __init__(
        self, suffixes: list[str | tuple[str, ...]] = [], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._suffixes = suffixes

    def compose(self) -> ComposeResult:
        """Yields the base inputs, switches, and the dynamic list for suffix rules."""
        yield Static(
            "Source for Suffix Validation Words", classes="widget-title"
        )
        yield from super().compose_inputs()
        with Container(classes="switch-container"):
            yield from super().compose_switches()

        with Horizontal(classes="list-header"):
            yield Static(classes="header-spacer") # The new invisible spacer
            yield Static("Suffix Filtering Rules", classes="widget-title")
            yield NoFocusButton("Add New", id="add_suffix_rule_button")

        yield SimpleDynamicListWidget(
            item_factory=_SuffixRuleWidget, items=self._suffixes, id="suffix_list"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button press to add a new suffix rule."""
        if event.button.id == "add_suffix_rule_button":
            list_widget = self.query_one("#suffix_list", SimpleDynamicListWidget)
            new_rule = _SuffixRuleWidget()
            list_widget.mount(new_rule)
            new_rule.scroll_visible()
            event.stop()

    def get_config(self) -> dict:
        """Returns the config, including the suffixes from the dynamic list."""
        config = {'get_words': super().get_config()}
        suffix_list = self.query_one("#suffix_list", SimpleDynamicListWidget)
        rules = []
        for rule_widget in suffix_list.query(_SuffixRuleWidget):
            rule = rule_widget.get_rule()
            if rule:
                rules.append(rule)
        config["suffixes"] = rules
        return config


class FilterFrequencyWidget(LoadingWidget):
    """A widget to filter words by minimum frequency."""

    def __init__(self, min_freq: float = 1e-7, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._min_freq = min_freq

    def compose_inputs(self) -> ComposeResult:
        """Yields the input field for min_freq."""
        with Horizontal(classes="input-row"):
            yield Static("Min Frequency:")
            yield Input(
                value=str(self._min_freq),
                id="min_freq",
                compact=True,
            )

    def compose_switches(self) -> ComposeResult:
        """This widget has no switches."""
        yield from ()

    def get_config(self) -> dict:
        """Returns the current configuration from the UI widgets."""
        return {"min_freq": float(self.query_one("#min_freq", Input).value)}


class FilterPOSWidget(LoadingWidget):
    """A widget to filter words by part-of-speech tags."""

    def __init__(
        self,
        tags: list[str] = ["NNS", "VBD", "VBN"],
        download: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tags = tags
        self._download = download

    def compose_inputs(self) -> ComposeResult:
        """Yields the input field for POS tags."""
        with Horizontal(classes="input-row"):
            yield Static("POS Tags (comma-sep):")
            yield Input(
                value=", ".join(self._tags),
                placeholder="e.g., NNS, VBD, VBN",
                id="pos_tags",
                compact=True,
            )

    def compose_switches(self) -> ComposeResult:
        """Yields the switch for downloading the model."""
        with Horizontal(classes="switch-group"):
            yield Switch(value=self._download, id="download")
            yield Static("Download Model")

    def get_config(self) -> dict:
        """Returns the current configuration from the UI widgets."""
        tags_str = self.query_one("#pos_tags", Input).value
        return {
            "tags": [tag.strip() for tag in tags_str.split(",")],
            "download": self.query_one("#download", Switch).value,
        }


class FilterProbabilityWidget(LoadingWidget):
    """A widget to filter words by a probability threshold."""

    def __init__(self, threshold: float = 0.07, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._threshold = threshold

    def compose_inputs(self) -> ComposeResult:
        """Yields the input field for the probability threshold."""
        with Horizontal(classes="input-row"):
            yield Static("Probability Threshold:")
            yield Input(
                value=str(self._threshold),
                id="threshold",
                compact=True,
            )

    def compose_switches(self) -> ComposeResult:
        """This widget has no switches."""
        yield from ()

    def get_config(self) -> dict:
        """Returns the current configuration from the UI widgets."""
        return {"threshold": float(self.query_one("#threshold", Input).value)}