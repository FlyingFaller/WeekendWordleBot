
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (Static, 
                             Input, 
                             Switch, 
                             Collapsible, 
                             Checkbox)

from weekend_wordle.backend.classifier import DEFAULT_CONFIG
from weekend_wordle.gui.setup.config_protocol import Configurable

class LoadingWidget(Container):
    """A generic, extensible widget for configuring data loading."""

    def __init__(
        self,
        title: str = None,
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


class GetPatternMatrixWidget(LoadingWidget):
    """A specialized widget for loading the pattern matrix."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compose_inputs(self) -> ComposeResult:
        """Yields only the savefile input, as no URL is needed."""
        with Horizontal(classes="input-row"):
            yield Static("Save File Path:")
            yield Input(
                value=self._savefile_path,
                placeholder="e.g., data/pattern_matrix.npy",
                id="savefile",
                compact=True
            )
    
    def get_config(self) -> dict:
        """Returns the config, excluding the URL which is not present."""
        return {
            "savefile": self.query_one("#savefile", Input).value,
            "recompute": self.query_one("#refetch", Switch).value,
            "save": self.query_one("#save", Switch).value,
        }


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


class GetWordFeaturesWidget(GetPatternMatrixWidget):
    """A specialized widget for loading word features."""

    def __init__(self, model_name: str = "en_core_web_lg", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_name = model_name

    def compose_inputs(self) -> ComposeResult:
        """Yields base inputs and the new 'model_name' input."""
        yield from super().compose_inputs()
        with Horizontal(classes="input-row"):
            yield Static("Spacy Model Name:")
            yield Input(value=self._model_name, id="model_name", compact=True)

    def get_config(self) -> dict:
        """Returns the config, including the model name."""
        config = super().get_config()
        config["model_name"] = self.query_one("#model_name", Input).value
        return config


class ExplicitFeatureWidget(Container):
    """A widget for a single feature with a checkbox and a weight input."""
    def __init__(self, name: str, weight: float, enabled: bool = True):
        super().__init__()
        self.feature_name = name
        self.feature_weight = weight
        self.feature_enabled = enabled

    def compose(self) -> ComposeResult:
        yield Checkbox(self.feature_name, value=self.feature_enabled, id="enabled_checkbox", compact=True)
        yield Input(value=str(self.feature_weight), id="weight_input", compact=True)


class LoadModelWidget(GetPatternMatrixWidget):
    """A specialized widget for loading the word classifier with advanced options."""

    def __init__(self, config: dict = DEFAULT_CONFIG, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config

    def compose(self) -> ComposeResult:
        """Yields base inputs and a collapsible advanced section."""
        yield from super().compose_inputs()
        with Container(classes="switch-container"):
            yield from super().compose_switches()
        
        with Collapsible(title="Advanced Training Options"):
            with Horizontal(classes="column-container"):
                with Vertical(classes="column"):
                    with Horizontal(classes="input-row"):
                        yield Static("Spy Rate:")
                        yield Input(value=str(self._config['spy_rate']), id="spy_rate", compact=True)
                    with Horizontal(classes="input-row"):
                        yield Static("Max Iterations:")
                        yield Input(value=str(self._config['max_iterations']), id="max_iterations", compact=True)
                    with Horizontal(classes="input-row"):
                        yield Static("Convergence Tolerance:")
                        yield Input(value=str(self._config['convergence_tolerance']), id="convergence_tolerance", compact=True)
                with Vertical(classes="column"):
                    with Horizontal(classes="input-row"):
                        yield Static("Random Seed:")
                        yield Input(placeholder="Blank for None", value=str(self._config.get('random_seed') or ''), id="random_seed", compact=True)
                    with Horizontal(classes="input-row"):
                        yield Static("Evaluation Threshold:")
                        yield Input(value=str(self._config['evaluation_threshold']), id="evaluation_threshold", compact=True)
                    with Horizontal(classes="switch-group"):
                        yield Switch(value=self._config['use_vectors'], id="use_vectors")
                        yield Static("Use Word Vectors")
            
            yield Static("\nExplicit Feature Weights:", classes="feature-header")
            
            with Horizontal(classes="column-container", id="feature_list"):
                feature_items = list(self._config['explicit_features'].items())
                with Vertical(classes="column"):
                    for name, weight in feature_items[:5]:
                        yield ExplicitFeatureWidget(name=name, weight=weight)
                with Vertical(classes="column"):
                    for name, weight in feature_items[5:]:
                        yield ExplicitFeatureWidget(name=name, weight=weight)

    def get_config(self) -> dict:
        """Returns the full, nested config dictionary from all UI elements."""
        base_config = super().get_config()
        
        explicit_features = {}
        for feature_widget in self.query(ExplicitFeatureWidget):
            if feature_widget.query_one("#enabled_checkbox", Checkbox).value:
                name = feature_widget.feature_name
                weight_str = feature_widget.query_one("#weight_input", Input).value
                try:
                    explicit_features[name] = float(weight_str)
                except ValueError:
                    explicit_features[name] = 1.0

        seed_str = self.query_one("#random_seed", Input).value
        random_seed = int(seed_str) if seed_str else None

        base_config['config'] = {
            'use_vectors': self.query_one("#use_vectors", Switch).value,
            'spy_rate': float(self.query_one("#spy_rate", Input).value),
            'max_iterations': int(self.query_one("#max_iterations", Input).value),
            'convergence_tolerance': float(self.query_one("#convergence_tolerance", Input).value),
            'random_seed': random_seed,
            'evaluation_threshold': float(self.query_one("#evaluation_threshold", Input).value),
            'explicit_features': explicit_features,
        }
        return base_config