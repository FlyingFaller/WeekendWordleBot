"""
Defines the SetupScreen and its components for the Wordle Solver application.

This screen is intended to gather configuration from the user before loading
the main game.
"""
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (Header, 
                             Footer, 
                             Static,
                             Switch, 
                             Rule,
                             Label)

from weekend_wordle.gui.loading.loading_screen import LoadingScreen
from weekend_wordle.gui.setup.dynamic_list_widget import DynamicCollapsibleList
from weekend_wordle.gui.setup.loading_widget import (GetWordsWidget, 
                                  ScrapeWordsWidget, 
                                  GetWordFeaturesWidget, 
                                  LoadClassifierWidget,
                                  GetPatternMatrixWidget)

from weekend_wordle.backend.helpers import (ORIGINAL_ANSWERS_FILE,
                                            ORIGINAL_ANSWERS_URL,
                                            PAST_ANSWERS_FILE,
                                            PAST_ANSWERS_URL,
                                            VALID_GUESSES_FILE,
                                            VALID_GUESSES_URL,
                                            DEFAULT_PATTERN_MATRIX_FILE)


class ClassifierSection(Container):
    """A widget for configuring the entire classifier training pipeline."""

    def compose(self) -> ComposeResult:
        """Create the child widgets for the classifier section."""
        
        # Add a specific class to the title bar's container
        with Horizontal(classes="title-bar"):
            yield Label("Load Optional Classifier")
            yield Rule()
            yield Switch(id="enable-switch")
        
        # The rest of the content widgets...
        widget_constructors = {
            'Get Words': lambda: GetWordsWidget(title='Get Words'),
            'Scrape Words': lambda: ScrapeWordsWidget(title='Scrape Words')
        }
        default_widgets = [('Original Answers', GetWordsWidget(title='Get Words', savefile_path=ORIGINAL_ANSWERS_FILE, url=ORIGINAL_ANSWERS_URL, )),
                           ('Past Answers', ScrapeWordsWidget(title='Scrape Words', savefile_path=PAST_ANSWERS_FILE, url=PAST_ANSWERS_URL, refetch=True))]
        yield DynamicCollapsibleList(title='Positive Words', 
                                     widget_constructors=widget_constructors,
                                     default_widgets=default_widgets,
                                     id="positive-words-list")
        yield GetWordFeaturesWidget(title='Word Features',id="word-features")
        yield LoadClassifierWidget(title='Load Classifier', id="load-classifier")

        yield Rule(classes="bottom-bar", id="bottom-rule")

    def on_mount(self) -> None:
        """Called when the widget is first mounted to set initial state."""
        # Ensure the initial visibility matches the switch's default value.
        self.toggle_widgets(self.query_one("#enable-switch", Switch).value)

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Called when the 'Enable' switch is toggled."""
        # We only care about our specific switch.
        if event.switch.id == "enable-switch":
            # Stop the event from bubbling up to other widgets.
            event.stop()
            # Call the helper method to do the actual work.
            self.toggle_widgets(event.value)

    def toggle_widgets(self, enabled: bool) -> None:
        """A helper method to show or hide the classifier configuration widgets."""
        # Find all the widgets we want to control.
        widgets_to_toggle = [
            self.query_one("#positive-words-list"),
            self.query_one("#word-features"),
            self.query_one("#load-classifier"),
            # self.query_one("#bottom-rule"),
        ]
        
        # Loop through them and set their visibility.
        # The 'display' property is the most effective way to hide widgets.
        for widget in widgets_to_toggle:
            widget.display = enabled

    # def toggle_widgets(self, enabled: bool) -> None:
    #     """
    #     A helper method to enable/disable the classifier configuration widgets
    #     and collapse any inner Collapsible widgets when disabling.
    #     """
    #     # We don't need to toggle the Rule, so it's removed from this list.
    #     widgets_to_toggle = [
    #         self.query_one("#positive-words-list"),
    #         self.query_one("#word-features"),
    #         self.query_one("#load-classifier"),
    #     ]
        
    #     # Loop through the main widgets and set their disabled state.
    #     # If the switch is enabled, `disabled` is False.
    #     # If the switch is disabled, `disabled` is True.
    #     for widget in widgets_to_toggle:
    #         widget.disabled = not enabled

    #     # If the section is being disabled, find and collapse all
    #     # Collapsible widgets within the main containers.
    #     if not enabled:
    #         for widget in widgets_to_toggle:
    #             # The .query() method finds all descendant widgets of a certain type.
    #             for collapsible in widget.query(Collapsible):
    #                 collapsible.collapsed = True

class SetupScreen(Screen):
    """A screen to configure the Wordle solver setup."""

    AUTO_FOCUS = " "
    CSS_PATH = "setup_screen.tcss"

    BINDINGS = [
        ("enter", "confirm_setup", "Confirm Setup"),
    ]

    def compose(self) -> ComposeResult:
        """Create child widgets for the screen."""
        yield Header()

        with VerticalScroll():
            yield Static("Solver Configuration", id="main_title")

            with Horizontal(classes="section-header"):
                yield Label("Mandatory Settings")
                yield Rule()

            yield GetWordsWidget(
                title="Guesses",
                savefile_path=VALID_GUESSES_FILE,
                url=VALID_GUESSES_URL,
            )
            yield GetWordsWidget(
                title="Answers",
                savefile_path=VALID_GUESSES_FILE,
                url=VALID_GUESSES_URL,
            )
            yield GetPatternMatrixWidget(
                title="Pattern Matrix",
                savefile_path=DEFAULT_PATTERN_MATRIX_FILE,
            )
            # After this point we have:
            # Toggleable Load Classifier Widget:
            #   Load Positive Words:
            #       Dynamic ListView widget of Collapsibles with option to contain either:
            #           GetWordsWidget and/or ScrapeWordsWidget
            #   GetWordFeaturesWidget
            #   LoadClassifierWidget
            #   
            # FilterWordsWidget:
            #   Dynamic ListView widget of Collapsibles with options to contain:
            #       FilterFrequencyWidget and/or FilterSuffixWidget and/or FilterClassifierWidget (if classifier is enabled/loaded)

            # # Example usage not part of final:
            # default_widgets = [('Default General Load', LoadingWidget(title='')),
            #                 ('Default Get Words', GetWordsWidget(title='')),
            #                 ('Default Scrape Words', ScrapeWordsWidget(title=''))]

            # widget_constructors: dict[str, Callable[[], Widget]] = {
            #     "General Load": lambda: LoadingWidget(title="File Source"),
            #     "Get Words": lambda: GetWordsWidget(title="Get Words Config"),
            #     "Scrape Words": lambda: ScrapeWordsWidget(title="Scrape Words Config"),
            # }

            # yield Static("─" * 100, classes="separator")
            yield ClassifierSection()
            # yield Static("─" * 100, classes="separator")

            # yield ScrapeWordsWidget(
            #     title="Past Answers (Scraper)",
            #     savefile_path="data/past_answers.txt",
            #     url="https://www.rockpapershotgun.com/wordle-past-answers",
            # )
            # yield GetWordFeaturesWidget(
            #     title="Word Features",
            #     savefile_path="data/word_features.pkl",
            # )
            # yield LoadClassifierWidget(
            #     title="Wordle Classifier",
            #     savefile_path="data/wordle_classifier.pkl",
            # )
        yield Footer()

    def action_confirm_setup(self) -> None:
        """
        Called when the user presses Enter.
        Switches to the loading screen.
        """
        self.app.switch_screen(LoadingScreen())
