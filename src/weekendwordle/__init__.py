import importlib

# A mapping of the names we want to expose to the module they live in.
# This dictionary will be used by __getattr__ to perform the import on demand.
_LAZY_IMPORTS = {
    # from .backend.atomic_ops
    "atomic_add": ".backend.atomic_ops",
    "atomic_sub": ".backend.atomic_ops",
    "atomic_cas": ".backend.atomic_ops",

    # from .backend.cache
    "AtomicInt": ".backend.cache",
    "Cache"    : ".backend.cache",

    # from .backend.classifier
    "compute_word_features"      : ".backend.classifier",
    "get_word_features"          : ".backend.classifier",
    "load_classifier"            : ".backend.classifier",
    "filter_words_by_probability": ".backend.classifier",

    # from .backend.core
    "WordleGame"         : ".backend.core",
    "InvalidWordError"   : ".backend.core",
    "InvalidPatternError": ".backend.core",

    # from .backend.helpers
    "get_pattern"              : ".backend.helpers",
    "pattern_str_to_int"       : ".backend.helpers",
    "pattern_to_int"           : ".backend.helpers",
    "int_to_pattern"           : ".backend.helpers",
    "precompute_pattern_matrix": ".backend.helpers",
    "get_pattern_matrix"       : ".backend.helpers",
    "get_words"                : ".backend.helpers",
    "scrape_words"             : ".backend.helpers",
    "get_word_freqs"           : ".backend.helpers",
    "get_minimum_freq"         : ".backend.helpers",
    "filter_words_by_frequency": ".backend.helpers",
    "PNR_hash"                 : ".backend.helpers",
    "FNV_hash"                 : ".backend.helpers",
    "python_hash"              : ".backend.helpers",
    "robust_mixing_hash"       : ".backend.helpers",
    "blake2b_hash"             : ".backend.helpers",
    "filter_words_by_length"   : ".backend.helpers",
    "filter_words_by_POS"      : ".backend.helpers",
    "filter_words_by_suffix"   : ".backend.helpers",
    "get_abs_path"             : ".backend.helpers",

    # from .backend.messenger
    "UIMessenger"     : ".backend.messenger",
    "ConsoleMessenger": ".backend.messenger",
    "TextualMessenger": ".backend.messenger",

    # from .backend.tests
    "simulate_game": ".backend.tests",
    "benchmark"    : ".backend.tests",

    # from .backend.cli_app
    "run_cli": ".backend.cli_app",

    # from .gui.wordle_app 
    "run_gui": ".gui.wordle_app",
}

# --- Handling the wildcard import from .config ---
# Wildcard imports are eager by nature. To make them lazy, we must dynamically
# discover the names in the config module and add them to our lazy map.
try:
    _config_module = importlib.import_module(".config", __name__)
    # Use __all__ if defined, otherwise get all public names.
    _config_names = getattr(_config_module, '__all__', [
        name for name in dir(_config_module) if not name.startswith('_')
    ])
    for _name in _config_names:
        _LAZY_IMPORTS[_name] = ".config"
    del _config_module, _config_names, _name
except ImportError:
    # Handle case where config might not exist
    pass
# ----------------------------------------------------


def __getattr__(name: str):
    """
    PEP 562: Lazily import attributes from this package.
    This function is called by Python when an attribute `name` is accessed
    on this module but is not found in the module's dictionary.
    """
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __name__)
        attr = getattr(module, name)

        # Cache the imported attribute in the module's globals() dictionary.
        # This ensures that __getattr__ is only called once for each attribute.
        globals()[name] = attr

        return attr

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    """
    Optional: This improves autocompletion for tools like IPython/Jupyter.
    It tells them what names are available in this module.
    """
    # Combine the names already loaded in globals with the lazy-loaded names.
    return list(globals().keys()) + list(_LAZY_IMPORTS.keys())