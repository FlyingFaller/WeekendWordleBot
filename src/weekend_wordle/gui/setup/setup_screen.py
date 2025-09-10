"""
Defines the SetupScreen and its components for the Wordle Solver application.

This screen is intended to gather configuration from the user before loading
the main game.
"""
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.message import Message
from textual.widgets import (Header, 
                             Footer, 
                             Static,
                             Switch, 
                             Rule,
                             Label,
                             Collapsible,
                             RadioSet,
                             RadioButton)

from weekend_wordle.gui.loading.loading_screen import LoadingScreen
from weekend_wordle.gui.setup.dynamic_list_widget import DynamicCollapsibleList
from weekend_wordle.gui.setup.loading_widget import (GetWordsWidget, 
                                                     ScrapeWordsWidget, 
                                                     GetWordFeaturesWidget, 
                                                     LoadModelWidget,
                                                     GetPatternMatrixWidget)

from weekend_wordle.gui.setup.filter_widget import (FilterSuffixWidget,
                                                    FilterFrequencyWidget,
                                                    FilterPOSWidget,
                                                    FilterProbabilityWidget)
from weekend_wordle.config import *

class ClassifierSection(Container):
    """A widget for configuring the entire classifier training pipeline."""

    class ClassifierStateChanged(Message):
        """Posted when the classifier section is enabled or disabled."""
        def __init__(self, enabled: bool) -> None:
            super().__init__()
            self.enabled = enabled

    def __init__(self, 
                 default_state: bool = True, 
                 collapse_on_disable: bool = True,
                 positive_words_defaults: dict = {},
                 word_features_defaults: dict = {},
                 load_model_defaults: dict = {},
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(*args, **kwargs)
        self._default_state           = default_state
        self._collapse_on_disable     = collapse_on_disable
        self._positive_words_defaults = positive_words_defaults
        self._word_features_defaults  = word_features_defaults
        self._load_model_defaults     = load_model_defaults

    def compose(self) -> ComposeResult:
        """Create the child widgets for the classifier section."""
        
        # Add a specific class to the title bar's container
        with Horizontal(classes="title-bar"):
            yield Label("Load Optional Classifier")
            yield Rule()
            yield Switch(id="enable_switch", value=self._default_state)
        
        yield DynamicCollapsibleList(title='Positive Words', **self._positive_words_defaults, id="positive_words_list")
        yield GetWordFeaturesWidget(title='Word Features',**self._word_features_defaults, id="word_features")
        yield LoadModelWidget(title='Load Model', **self._load_model_defaults, id="load_model")

        yield Rule(classes="bottom-bar", id="bottom_rule")

    def on_mount(self) -> None:
        """Called when the widget is first mounted to set initial state."""
        # Ensure the initial visibility matches the switch's default value.
        self.toggle_widgets(self.query_one("#enable_switch", Switch).value)

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Called when the 'Enable' switch is toggled."""
        # We only care about our specific switch.
        if event.switch.id == "enable_switch":
            # Stop the event from bubbling up to other widgets.
            event.stop()
            # Call the helper method to do the actual work.
            self.toggle_widgets(event.value)

    def toggle_widgets(self, enabled: bool) -> None:
        """A helper method to show or hide the classifier configuration widgets."""
        # Find all the widgets we want to control.
        widgets_to_toggle = [
            self.query_one("#positive_words_list"),
            self.query_one("#word_features"),
            self.query_one("#load_model"),
        ]
        if self._collapse_on_disable:
            # Loop through them and set their visibility.
            # The 'display' property is the most effective way to hide widgets.
            for widget in widgets_to_toggle:
                widget.display = enabled
        else:            
            # Loop through the main widgets and set their disabled state.
            # If the switch is enabled, `disabled` is False.
            # If the switch is disabled, `disabled` is True.
            for widget in widgets_to_toggle:
                widget.disabled = not enabled

            # If the section is being disabled, find and collapse all
            # Collapsible widgets within the main containers.
            if not enabled:
                for widget in widgets_to_toggle:
                    # The .query() method finds all descendant widgets of a certain type.
                    for collapsible in widget.query(Collapsible):
                        collapsible.collapsed = True

        self.post_message(self.ClassifierStateChanged(enabled))

    def get_config(self) -> dict | None:
        """
        Retrieves the configuration for the classifier pipeline.
        """
        # Query for the switch by its ID and check its boolean value
        if not self.query_one("#enable_switch", Switch).value:
            return None

        # If enabled, gather the config from the child widgets
        return {
            "positive_words": self.query_one("#positive_words_list").get_config(),
            "word_features": self.query_one("#word_features").get_config(),
            "model": self.query_one("#load_model").get_config(),
        }

class AnswerSortWidget(Static):
    """A widget to select the sorting method for answers."""
    class CustomRadioButton(RadioButton):
        BUTTON_INNER = '\u25FC'

    def __init__(self, title: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_title = title
        
    def compose(self) -> ComposeResult:
        """Create the child widgets for the answer sort widget."""
        with RadioSet():
            yield self.CustomRadioButton("Word Frequency", id="word_frequency")
            yield self.CustomRadioButton("Classifier Probability", value=True, id="classifier_probability")

    def update_classifier_dependency(self, classifier_enabled: bool) -> None:
        """Disables and resets the classifier sort option based on classifier state."""
        classifier_button = self.query_one("#classifier_probability", RadioButton)
        classifier_button.disabled = not classifier_enabled

        # If the classifier was disabled, check if we need to reset the selection.
        if not classifier_enabled:
            radio_set = self.query_one(RadioSet)
            # If the (now disabled) classifier button is still pressed...
            if radio_set.pressed_button and radio_set.pressed_button.id == "classifier_probability":
                # ...then switch the selection to the word frequency button.
                word_freq_button = self.query_one("#word_frequency", RadioButton)
                word_freq_button.value = True

    def get_config(self) -> str:
        """Returns the selected sort method as a string."""
        radio_set = self.query_one(RadioSet)
        # Check which button is pressed and return the corresponding value
        if radio_set.pressed_button and radio_set.pressed_button.id == "classifier_probability":
            return "Classifier"
        return "Frequency"

class SetupScreen(Screen):
    """A screen to configure the Wordle solver setup."""

    AUTO_FOCUS = ""
    CSS_PATH = "setup_screen.tcss"

    BINDINGS = [
        ("enter", "confirm_setup", "Confirm Setup"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._ready_to_confirm = False

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.call_after_refresh(self._unlock_confirm)

    def _unlock_confirm(self) -> None:
        """Callback to enable the confirm action."""
        self._ready_to_confirm = True

    def compose(self) -> ComposeResult:
        # """Create child widgets for the screen."""
        yield Header()

        with VerticalScroll():
            yield Static("Solver Configuration", id="main_title")

            with Horizontal(classes="section-header"):
                yield Label("Mandatory Settings")
                yield Rule()

            yield GetWordsWidget(
                title="Guesses",
                savefile_path=DATA_ROOT+VALID_GUESSES_FILE,
                url=VALID_GUESSES_URL,
                id="get_guesses"
            )
            yield GetWordsWidget(
                title="Answers",
                savefile_path=DATA_ROOT+VALID_GUESSES_FILE,
                url=VALID_GUESSES_URL,
                id="get_answers"
            )
            yield GetPatternMatrixWidget(
                title="Pattern Matrix",
                savefile_path=DATA_ROOT+PATTERN_MATRIX_FILE,
                id='get_pattern_matrix'
            )

            positive_words_defaults = {
                'widget_constructors': {
                    'Get Words': lambda: GetWordsWidget(),
                    'Scrape Words': lambda: ScrapeWordsWidget()
                },
                'default_widgets': 
                    [('Original Answers', GetWordsWidget(savefile_path=DATA_ROOT+ORIGINAL_ANSWERS_FILE, url=ORIGINAL_ANSWERS_URL, )),
                    ('Past Answers', ScrapeWordsWidget(savefile_path=DATA_ROOT+PAST_ANSWERS_FILE, url=PAST_ANSWERS_URL, refetch=True))]
            }
            word_features_defaults = {
                'savefile_path': DATA_ROOT + WORD_FEATURES_FILE
            }
            load_model_defaults = {
                'savefile_path': DATA_ROOT + CLASSIFIER_MODEL_FILE
            }
            yield ClassifierSection(collapse_on_disable=False, 
                                    positive_words_defaults=positive_words_defaults,
                                    word_features_defaults=word_features_defaults,
                                    load_model_defaults=load_model_defaults, 
                                    id="classifier_section")

            with Horizontal(classes="section-header"):
                yield Label("Apply Optional Filters to Answer Set")
                yield Rule()

            filter_constructors = {'Suffix Filter': lambda: FilterSuffixWidget(suffixes=[('s', 's'), ('d', 'r', 'w', 'n'), 'es', 'ed'],
                                                                               savefile_path=DATA_ROOT+ENGLISH_DICTIONARY_FILE),
                                   'Frequency Filter': lambda: FilterFrequencyWidget(),
                                   'POS Filter': lambda: FilterPOSWidget(),
                                   'Classifier Probability Filter': lambda: FilterProbabilityWidget()}
            
            default_filters = [('Classifier Probability Filter', FilterProbabilityWidget())]

            yield DynamicCollapsibleList(widget_constructors=filter_constructors,
                                         default_widgets=default_filters,
                                         id="answer_filters_list")

            with Horizontal(classes="section-header"):
                yield Label("Select Optional Answer Sort")
                yield Rule()

            yield AnswerSortWidget(id="answer_sort")

        yield Footer()

    def on_classifier_section_classifier_state_changed(
        self, message: ClassifierSection.ClassifierStateChanged
    ) -> None:
        """A message handler to update widgets when the classifier is toggled."""
        # Update the sort widget
        answer_sort_widget = self.query_one(AnswerSortWidget)
        answer_sort_widget.update_classifier_dependency(message.enabled)

        # Update the filter list widget
        filter_list = self.query_one("#answer_filters_list", DynamicCollapsibleList)
        filter_list.update_classifier_dependency(message.enabled)


    def action_confirm_setup(self) -> None:
        """
        Called when the user presses Enter.
        Switches to the loading screen.
        """

        if not self._ready_to_confirm:
            return
        
        config = {}
        config['guesses']        = self.query_one('#get_guesses').get_config()
        config['answers']        = self.query_one('#get_answers').get_config()
        config['pattern_matrix'] = self.query_one('#get_pattern_matrix').get_config()
        config['classifier']     = self.query_one('#classifier_section').get_config()
        config['filters']        = self.query_one('#answer_filters_list').get_config()
        config['sort']           = self.query_one('#answer_sort').get_config()

        self.app.push_screen(LoadingScreen(config))
