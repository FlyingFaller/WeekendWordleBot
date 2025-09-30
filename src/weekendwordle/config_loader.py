import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from .config import CONFIG_FILE, PROJECT_ROOT, REQUIRED_SCHEMA

# --- Helper Functions ---

def get_abs_path(usr_path_str: str, root_path: Path = PROJECT_ROOT) -> Path:
    user_path = Path(usr_path_str)
    return user_path if user_path.is_absolute() else root_path / user_path

class ConfigError(Exception):
    """Custom exception for configuration loading or validation errors."""
    pass

def deep_merge(source: dict, destination: dict) -> dict:
    """Recursively merges a source dictionary into a destination dictionary."""
    for key, value in source.items():
        if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
            destination[key] = deep_merge(value, destination[key])
        else:
            destination[key] = value
    return destination

def _parse_cli_value(value: Any) -> Any:
    """
    Intelligently parses a string from the CLI into a Python type.
    Handles None, bool, int, float, lists, and falls back to string.
    Strips common container characters from list-like strings.
    """
    if not isinstance(value, str):
        return value # Pass through non-strings
        
    stripped_val = value.strip()

    # If the value is wrapped in list-like containers, strip them first.
    if (stripped_val.startswith('[') and stripped_val.endswith(']')) or \
       (stripped_val.startswith('(') and stripped_val.endswith(')')) or \
       (stripped_val.startswith('{') and stripped_val.endswith('}')):
        inner_val = stripped_val[1:-1]
    else:
        inner_val = stripped_val

    # Check if the inner content represents a list (comma-separated)
    if ',' in inner_val:
        # Recursively parse each element of the comma-separated list
        return [_parse_cli_value(item) for item in inner_val.split(',')]
    
    # If not a list, parse as a single primitive value
    val_lower = inner_val.lower()
    if val_lower in ['none', 'null']:
        return None
    if val_lower == 'true':
        return True
    if val_lower == 'false':
        return False
    if inner_val.isdigit():
        return int(inner_val)
    try:
        return float(inner_val)
    except ValueError:
        # Return the processed inner string if it's not a known type
        return inner_val

def set_nested_value(d: dict, key_path: str, value: Any):
    """Sets a value in a nested dictionary using a dot-separated key path."""
    keys = key_path.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = _parse_cli_value(value)

def get_nested_value(d: dict, key_path: str) -> Any:
    """Gets a value from a nested dictionary using a dot-separated key path."""
    keys = key_path.split('.')
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return None # Path does not exist
    return d

# --- New Validation Logic ---

def validate_config(
    config: Dict[str, Any],
    schema: Dict[str, Any],
    root_config: Dict[str, Any],
    no_gui: bool,
    path: str = ""
) -> List[str]:
    """Recursively validates a configuration dict against the new schema."""
    errors = []
    
    for key, rules in schema.items():
        current_path = f"{path}.{key}" if path else key
        
        # 1. CHECK FOR KEY EXISTENCE (Required by default)
        if key not in config:
            is_optional = rules.get("optional", False)
            is_gui_only = rules.get("gui_only", False) and no_gui
            
            required_if = rules.get("required_if")
            condition_met = False
            if required_if:
                dep_path, dep_val = required_if
                actual_val = get_nested_value(root_config, dep_path)
                condition_met = (actual_val == dep_val)

            if not (is_optional or is_gui_only or (required_if and not condition_met)):
                errors.append(f"Schema Error: Missing required key '{current_path}'")
            continue # Nothing more to check for a missing key

        actual_value = config[key]
        
        # 2. VALIDATE TYPE
        expected_types = rules.get("type")
        if expected_types and not isinstance(actual_value, expected_types):
            type_names = getattr(expected_types, '__name__', str(expected_types))
            errors.append(
                f"Schema Error: Key '{current_path}' has wrong type. "
                f"Expected {type_names}, but got {type(actual_value).__name__}."
            )
            continue # Don't check value/schema if type is wrong

        # 3. VALIDATE SPECIFIC VALUES
        valid_values = rules.get("valid_values")
        if valid_values is not None:
            allowed = valid_values if isinstance(valid_values, list) else [valid_values]
            if actual_value not in allowed:
                errors.append(
                    f"Schema Error: Key '{current_path}' has wrong value. "
                    f"Expected one of {allowed}, but got '{actual_value}'."
                )

        # 4. RECURSE FOR NESTED SCHEMAS
        nested_schema = rules.get("schema")
        if nested_schema:
            if isinstance(actual_value, dict):
                errors.extend(validate_config(actual_value, nested_schema, root_config, no_gui, path=current_path))
            elif isinstance(actual_value, list) and isinstance(rules.get('type'), type) and issubclass(rules['type'], list):
                for i, item in enumerate(actual_value):
                    if isinstance(item, dict):
                        errors.extend(validate_config(item, nested_schema, root_config, no_gui, path=f"{current_path}[{i}]"))
                    else:
                        errors.append(f"Schema Error: Expected a list of dictionaries for '{current_path}', but found an item of type {type(item).__name__}.")

    return errors

# --- Main Loading Function ---

def parse_cli_args(argv: list[str] | None = None):
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Wordle Solver Configuration Loader")
    parser.add_argument("-c", "--config", type=Path, help="Path to a custom configuration JSON file.")
    parser.add_argument('--set', nargs=2, action='append', metavar=('KEY', 'VALUE'), help="Override a config value using dot notation.")
    return parser.parse_args(argv)

def load_config(argv: list[str] | None = None, no_gui: bool = False) -> dict:
    """
    Loads, merges, and validates configuration for GUI or CLI contexts.
    Raises ConfigError if loading or validation fails.
    """
    # 1. Load Default Config
    config_file = get_abs_path(CONFIG_FILE)
    if not config_file.exists():
        raise ConfigError(f"Fatal Error: Default config file not found at {config_file}")
    with open(config_file) as f:
        final_config = json.load(f)

    # 2. Merge Custom Config and CLI Args
    args = parse_cli_args(argv)
    if args.config:
        custom_config_path = get_abs_path(args.config)
        if custom_config_path.exists():
            try:
                with open(custom_config_path) as f:
                    custom_data = json.load(f)
                final_config = deep_merge(source=custom_data, destination=final_config)
            except json.JSONDecodeError as e:
                raise ConfigError(f"Error parsing custom config file at '{custom_config_path}': {e}")
        else:
            print(f"Warning: Custom config file not found at {custom_config_path}", file=sys.stderr)

    if args.set:
        for key, value in args.set:
            set_nested_value(final_config, key, value)
    
    # 3. Validate the Final Configuration
    validation_errors = validate_config(final_config, REQUIRED_SCHEMA, root_config=final_config, no_gui=no_gui)
    if validation_errors:
        header = "Configuration validation failed with the following errors:"
        full_error_message = "\n".join([header] + [f"  - {e}" for e in validation_errors])
        raise ConfigError(full_error_message)

    return final_config


# """
# Translates the generic, validated configuration into context-specific formats
# for the GUI (widget construction) and the backend (data loading).
# """
# from typing import Any, Dict, List

# # --- WIDGET CLASSNAME MAPPING (for backend translation) ---
# # This avoids needing to import the actual widget classes in the translator.
# WIDGET_CLASS_MAP = {
#     "GetWordsWidget": "GetWordsWidget",
#     "ScrapeWordsWidget": "ScrapeWordsWidget",
#     "FilterSuffixWidget": "FilterSuffixWidget",
#     "FilterFrequencyWidget": "FilterFrequencyWidget",
#     "FilterPOSWidget": "FilterPOSWidget",
#     "FilterProbabilityWidget": "FilterProbabilityWidget",
# }


def translate_for_backend(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translates the full config into a simple dictionary of backend parameters,
    similar to the output of the original SetupScreen.
    """
    backend_config = {}

    # Translate simple sections
    backend_config['game_settings'] = config['game_settings']
    backend_config['guesses'] = config['guesses']['backend_params']
    backend_config['answers'] = config['answers']['backend_params']
    backend_config['pattern_matrix'] = config['pattern_matrix']['backend_params']
    backend_config['sort'] = config['answer_sort']['backend_params'].get('default_selection', 'frequency')

    # Translate classifier
    classifier_cfg = config.get('classifier')
    if classifier_cfg and classifier_cfg.get('enabled'):
        positive_words = []
        for item in classifier_cfg['positive_words']['items']:
            positive_words.append({
                'type': item['class'],
                'contents': item['backend_params']
            })
        
        backend_config['classifier'] = {
            'positive_words': positive_words,
            'word_features': classifier_cfg['word_features']['backend_params'],
            'model': classifier_cfg['load_model']['backend_params']
        }
    else:
        backend_config['classifier'] = None

    # Translate filters
    filters_cfg = config.get('answer_filters', {})
    backend_filters = []
    for item in filters_cfg.get('items', []):
        backend_filters.append({
            'type': item['class'],
            'contents': item['backend_params']
        })
    backend_config['filters'] = backend_filters
    
    return backend_config


def _translate_section_for_gui(section_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively processes a single section of the config for the GUI."""
    if not isinstance(section_config, dict):
        return section_config

    # Handle argument mapping between backend and GUI params
    if 'gui_params' in section_config and '_arg_map' in section_config['gui_params']:
        arg_map = section_config['gui_params'].pop('_arg_map')
        for gui_arg, backend_arg in arg_map.items():
            if backend_arg in section_config.get('backend_params', {}):
                value = section_config['backend_params'].pop(backend_arg)
                section_config['backend_params'][gui_arg] = value
    
    # Recursively translate children/items
    if 'items' in section_config:
        section_config['items'] = [_translate_section_for_gui(item) for item in section_config['items']]
    
    for key, value in section_config.items():
        if isinstance(value, dict) and 'class' in value:
             section_config[key] = _translate_section_for_gui(value)

    return section_config


def translate_for_gui(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translates the generic config into a format optimized for the
    build_widget_from_config function, handling all special cases.
    """
    gui_config = config.copy()

    # --- Translate ClassifierSection ---
    if 'classifier' in gui_config and 'enabled' in gui_config['classifier']:
        classifier_cfg = gui_config['classifier']
        # Create gui_params if it doesn't exist
        if 'gui_params' not in classifier_cfg:
            classifier_cfg['gui_params'] = {}
            
        classifier_cfg['gui_params']['sections'] = {
            'positive_words': classifier_cfg.pop('positive_words'),
            'word_features': classifier_cfg.pop('word_features'),
            'load_model': classifier_cfg.pop('load_model')
        }
        classifier_cfg['gui_params']['default_state'] = classifier_cfg.pop('enabled')

    # Perform recursive translation for arg maps and other transformations
    for key, section in gui_config.items():
        gui_config[key] = _translate_section_for_gui(section)
            
    return gui_config