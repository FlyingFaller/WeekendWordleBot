import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List
from .config import CONFIG_FILE, PROJECT_ROOT, REQUIRED_SCHEMA

# --- Path Loader Taken from helpers.py ---

def get_abs_path(usr_path_str: str, root_path: Path = PROJECT_ROOT) -> Path:
    user_path = Path(usr_path_str)

    if user_path.is_absolute():
        # If it's absolute, use it directly.
        return user_path
    else:
        # If it's relative, assume it's relative to the project root.
        return root_path / user_path

# --- Custom Exception for Configuration ---

class ConfigError(Exception):
    """Custom exception for configuration loading or validation errors."""
    pass

# --- CONFIGURATION SCHEMA & VALIDATION ---

def validate_config_schema(config: Dict[str, Any], schema: Dict[str, Any] = REQUIRED_SCHEMA, path: str = "") -> List[str]:
    """
    Recursively validates a configuration dict against a schema using a generalized tuple format:
    (type1, type2, ..., required_value [optional])
    """
    errors: List[str] = []
    for key, expected in schema.items():
        current_path = f"{path}.{key}" if path else key
        
        if key not in config:
            errors.append(f"Schema Error: Missing required key '{current_path}'")
            continue

        actual_value = config[key]

        if isinstance(expected, tuple):
            allowed_types = list(expected)
            required_value = None
            has_required_value = False

            # Check if the last element is a required value, not a type
            if expected and not isinstance(expected[-1], type):
                required_value = allowed_types.pop()
                has_required_value = True
            
            # Perform type check
            if type(actual_value) not in allowed_types:
                type_names = ", ".join(t.__name__ for t in allowed_types)
                errors.append(f"Schema Error: Key '{current_path}' has wrong type. "
                              f"Expected one of ({type_names}), but got {type(actual_value).__name__}.")
                continue # No need to check value if type is wrong
            
            # Perform value check if one was specified
            if has_required_value and actual_value != required_value:
                errors.append(f"Schema Error: Key '{current_path}' has wrong value. "
                              f"Expected '{required_value}', but got '{actual_value}'.")

        elif isinstance(expected, type):
            if not isinstance(actual_value, expected):
                errors.append(f"Schema Error: Key '{current_path}' has wrong type. "
                              f"Expected {expected.__name__}, but got {type(actual_value).__name__}.")
        elif isinstance(expected, dict):
            if not isinstance(actual_value, dict):
                 errors.append(f"Schema Error: Key '{current_path}' should be a dictionary, "
                               f"but got {type(actual_value).__name__}.")
            else:
                nested_errors = validate_config_schema(actual_value, expected, path=current_path)
                errors.extend(nested_errors)
    return errors

# --- CONFIGURATION LOADING & MERGING ---

def deep_merge(source: dict, destination: dict) -> dict:
    """
    Recursively merges a source dictionary into a destination dictionary.
    """
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


def set_nested_value(d: dict, key_path: str, value: str):
    """
    Sets a value in a nested dictionary using a dot-separated key path.
    The input `value` is a string from the command line which is intelligently parsed.
    """
    keys = key_path.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    
    # The entire parsing logic is now handled by the robust _parse_cli_value function
    d[keys[-1]] = _parse_cli_value(value)


def parse_cli_args(argv: list[str] | None = None):
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Wordle Solver Configuration Loader")
    parser.add_argument("-c", "--config", type=Path, help="Path to a custom configuration JSON file.")
    parser.add_argument('--set', nargs=2, action='append', metavar=('KEY', 'VALUE'), help="Override a config value using dot notation.")
    return parser.parse_args(argv)


def load_config(argv: list[str] | None = None) -> dict:
    """
    Loads, merges, and validates configuration from multiple sources.
    Raises ConfigError if loading or validation fails.
    """
    config_file = get_abs_path(CONFIG_FILE)
    if not config_file.exists():
        raise ConfigError(f"Fatal Error: Default config file not found at {config_file}")
        
    with open(config_file) as f:
        final_config = json.load(f)

    args = parse_cli_args(argv)

    if args.config:
        config = get_abs_path(args.config)
        if config.exists():
            try:
                with open(config) as f:
                    custom_data = json.load(f)
                final_config = deep_merge(source=custom_data, destination=final_config)
            except json.JSONDecodeError as e:
                raise ConfigError(f"Error parsing custom config file at '{config}': {e}")
        else:
            # This is a warning, not a fatal error.
            print(f"Warning: Custom config file not found at {config}", file=sys.stderr)

    if args.set:
        for key, value in args.set:
            set_nested_value(final_config, key, value)
    
    # Validate the final configuration and raise an exception if it fails
    validation_errors = validate_config_schema(final_config)
    if validation_errors:
        header = "Configuration validation failed with the following errors:"
        full_error_message = "\n".join([header] + [f"  - {e}" for e in validation_errors])
        raise ConfigError(full_error_message)

    return final_config

