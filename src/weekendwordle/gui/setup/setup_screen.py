"""
Defines the SetupScreen and its components for the Wordle Solver application.

This screen is intended to gather configuration from the user before loading
the main game.
"""
import json
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.message import Message
from textual.widget import Widget
from textual.widgets import (Header, 
                             Footer, 
                             Static,
                             Switch, 
                             Rule,
                             Label,
                             Collapsible,
                             RadioSet,
                             RadioButton)

from .dynamic_list_widget import DynamicCollapsibleList
from .loading_widget import (GetWordsWidget, 
                             ScrapeWordsWidget, 
                             GetWordFeaturesWidget, 
                             LoadModelWidget,
                             GetPatternMatrixWidget)
from .filter_widget import (FilterSuffixWidget,
                            FilterFrequencyWidget,
                            FilterPOSWidget,
                            FilterProbabilityWidget)
from ..loading.loading_screen import LoadingScreen
from ...backend.helpers import get_abs_path

# --- WIDGET DEFINITIONS ---
# All widget classes must be defined *before* they are registered.

class ClassifierSection(Container):
    """A widget for configuring the entire classifier training pipeline."""

    class ClassifierStateChanged(Message):
        def __init__(self, enabled: bool) -> None:
            super().__init__()
            self.enabled = enabled

    def __init__(
        self, 
        default_state: bool = True, 
        collapse_on_disable: bool = True,
        sections: dict = {},
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._default_state = default_state
        self._collapse_on_disable = collapse_on_disable
        self._sections_config = sections

    def compose(self) -> ComposeResult:
        """Create the child widgets for the classifier section from config."""
        with Horizontal(classes="title-bar"):
            yield Label("Load Optional Classifier")
            yield Rule()
            yield Switch(id="enable_switch", value=self._default_state)
        
        # Build children from the provided section configs
        yield build_widget_from_config(self._sections_config['positive_words'], id="positive_words_list")
        yield build_widget_from_config(self._sections_config['word_features'], id="word_features")
        yield build_widget_from_config(self._sections_config['load_model'], id="load_model")

        yield Rule(classes="bottom-bar", id="bottom_rule")

    def on_mount(self) -> None:
        self.toggle_widgets(self.query_one("#enable_switch", Switch).value)

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "enable_switch":
            event.stop()
            self.toggle_widgets(event.value)

    def toggle_widgets(self, enabled: bool) -> None:
        widgets_to_toggle = [
            self.query_one("#positive_words_list"),
            self.query_one("#word_features"),
            self.query_one("#load_model"),
        ]
        if self._collapse_on_disable:
            for widget in widgets_to_toggle:
                widget.display = enabled
        else:            
            for widget in widgets_to_toggle:
                widget.disabled = not enabled
            if not enabled:
                for widget in widgets_to_toggle:
                    for collapsible in widget.query(Collapsible):
                        collapsible.collapsed = True
        self.post_message(self.ClassifierStateChanged(enabled))

    def get_config(self) -> dict | None:
        if not self.query_one("#enable_switch", Switch).value:
            return None
        return {
            "positive_words": self.query_one("#positive_words_list").get_config(),
            "word_features": self.query_one("#word_features").get_config(),
            "model": self.query_one("#load_model").get_config(),
        }

class AnswerSortWidget(Static):
    """A widget to select the sorting method for answers."""
    class CustomRadioButton(RadioButton):
        BUTTON_INNER = '\u25FC'

    def __init__(self, title: str = None,default_selection: str = 'classifier', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_title = title
        self._default_selection = default_selection
        
    def compose(self) -> ComposeResult:
        with RadioSet():
            yield self.CustomRadioButton("Word Frequency", value=self._default_selection=='frequency', id="word_frequency")
            yield self.CustomRadioButton("Classifier Probability", value=self._default_selection=='classifier', id="classifier_probability")

    def update_classifier_dependency(self, classifier_enabled: bool) -> None:
        classifier_button = self.query_one("#classifier_probability", RadioButton)
        classifier_button.disabled = not classifier_enabled
        if not classifier_enabled:
            radio_set = self.query_one(RadioSet)
            if radio_set.pressed_button and radio_set.pressed_button.id == "classifier_probability":
                word_freq_button = self.query_one("#word_frequency", RadioButton)
                word_freq_button.value = True

    def get_config(self) -> str:
        radio_set = self.query_one(RadioSet)
        if radio_set.pressed_button and radio_set.pressed_button.id == "classifier_probability":
            return "Classifier"
        return "Frequency"

# --- WIDGET REGISTRY & BUILDER ---
# This section makes the config-driven UI possible.
# It is placed here, after all custom widget classes are defined.

WIDGET_REGISTRY = {
    "GetWordsWidget"         : GetWordsWidget,
    "ScrapeWordsWidget"      : ScrapeWordsWidget,
    "GetWordFeaturesWidget"  : GetWordFeaturesWidget,
    "LoadModelWidget"        : LoadModelWidget,
    "GetPatternMatrixWidget" : GetPatternMatrixWidget,
    "ClassifierSection"      : ClassifierSection,
    "DynamicCollapsibleList" : DynamicCollapsibleList,
    "AnswerSortWidget"       : AnswerSortWidget,
    "FilterSuffixWidget"     : FilterSuffixWidget,
    "FilterFrequencyWidget"  : FilterFrequencyWidget,
    "FilterPOSWidget"        : FilterPOSWidget,
    "FilterProbabilityWidget": FilterProbabilityWidget,
}

def build_widget_from_config(config: dict[str, Any], id: str | None = None) -> Widget:
    """
    Builds a widget instance from a configuration dictionary.
    """
    widget_class_name = config["class"]
    widget_factory = WIDGET_REGISTRY[widget_class_name]

    kwargs: dict = config.get("backend_params", {}).copy()
    gui_params: dict = config.get("gui_params", {}).copy()

    if "_arg_map" in gui_params:
        for gui_arg, backend_arg in gui_params["_arg_map"].items():
            if backend_arg in kwargs:
                kwargs[gui_arg] = kwargs.pop(backend_arg)
        del gui_params["_arg_map"]

    kwargs.update(gui_params)
    
    # Pass the ID to the widget constructor if it's provided
    if id:
        kwargs['id'] = id

    if widget_class_name == "DynamicCollapsibleList":
        # Look for the new top-level keys in the main config object
        constructors_config = config.get("constructors", {})
        items_config = config.get("items", [])
        
        constructors = {name: lambda cls=WIDGET_REGISTRY[c_name]: cls() 
                        for name, c_name in constructors_config.items()}

        items = []
        for widget_conf in items_config:
            title = widget_conf["gui_params"].pop("_title_override")
            content_widget = build_widget_from_config(widget_conf)
            items.append((title, content_widget))
        
        # Add the processed items to the kwargs for the widget's __init__
        kwargs["widget_constructors"] = constructors
        kwargs["default_widgets"] = items

    if widget_class_name == "ClassifierSection":
        kwargs['sections'] = {
            'positive_words': config['positive_words'],
            'word_features': config['word_features'],
            'load_model': config['load_model']
        }

    return widget_factory(**kwargs)

# --- MAIN SCREEN ---

class SetupScreen(Screen):
    """A screen to configure the Wordle solver setup."""
    AUTO_FOCUS = ""
    CSS_PATH = "setup_screen.tcss"
    BINDINGS = [("enter", "confirm_setup", "Confirm Setup")]

    def __init__(self) -> None:
        super().__init__()
        self._ready_to_confirm = False
        self.config_data = self.app.config_data

    def on_mount(self) -> None:
        self.call_after_refresh(self._unlock_confirm)

    def _unlock_confirm(self) -> None:
        self._ready_to_confirm = True

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Static("Solver Configuration", id="main_title")

            with Horizontal(classes="section-header"):
                yield Label("Mandatory Settings")
                yield Rule()

            yield build_widget_from_config(self.config_data['guesses'], id="get_guesses")
            yield build_widget_from_config(self.config_data['answers'], id="get_answers")
            yield build_widget_from_config(self.config_data['pattern_matrix'], id='get_pattern_matrix')

            yield build_widget_from_config(self.config_data['classifier'], id="classifier_section")

            with Horizontal(classes="section-header"):
                yield Label("Apply Optional Filters to Answer Set")
                yield Rule()

            yield build_widget_from_config(self.config_data['answer_filters'], id="answer_filters_list")

            with Horizontal(classes="section-header"):
                yield Label("Select Optional Answer Sort")
                yield Rule()

            yield build_widget_from_config(self.config_data['answer_sort'], id="answer_sort")
        yield Footer()

    def on_classifier_section_classifier_state_changed(
        self, message: ClassifierSection.ClassifierStateChanged
    ) -> None:
        answer_sort_widget = self.query_one(AnswerSortWidget)
        answer_sort_widget.update_classifier_dependency(message.enabled)
        filter_list = self.query_one("#answer_filters_list", DynamicCollapsibleList)
        filter_list.update_classifier_dependency(message.enabled)

    def action_confirm_setup(self) -> None:
        if not self._ready_to_confirm:
            return
        
        config = {
            'guesses': self.query_one('#get_guesses').get_config(),
            'answers': self.query_one('#get_answers').get_config(),
            'pattern_matrix': self.query_one('#get_pattern_matrix').get_config(),
            'classifier': self.query_one('#classifier_section').get_config(),
            'filters': self.query_one('#answer_filters_list').get_config(),
            'sort': self.query_one('#answer_sort').get_config(),
        }
        self.app.push_screen(LoadingScreen(config))

