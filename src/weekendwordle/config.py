from pathlib import Path
from numba import get_num_threads
import os

CONFIG_FILE = "config.json"

USER =  os.getlogin( )

PROJECT_ROOT = Path(__file__).parent

NTHREADS = get_num_threads()

GREEN  = 2
YELLOW = 1
GRAY   = 0

CLASSIFIER_CONFIG = {
    'use_vectors'          : True,
    'spy_rate'             : 0.15,
    'max_iterations'       : 1000,
    'convergence_tolerance': 1e-2,
    'random_seed'          : None,
    'evaluation_threshold' : 0.07,
    'explicit_features'    : {
        'frequency'          : 1.0,
        'is_regular_plural'  : 1.0,
        'is_irregular_plural': 1.0,
        'is_past_tense'      : 1.0,
        'is_adjective'       : 1.0,
        'is_proper_noun'     : 1.0,
        'is_gerund'          : 1.0,
        'vowel_count'        : 1.0,
        'has_double_letter'  : 1.0
    }
}

EVENTS = [
    ('cache_hits', 'Cache hits'),
    ('entropy_skips', 'Entropy loop skips'),
    ('entropy_exits', 'Entropy loop exits'),
    ('winning_patterns', 'Winning patterns found'),
    ('low_pattern_counts', 'Low answer count patterns found'),
    ('recursions_queued', 'Recursions queued'),
    ('depth_limit', 'Depth limits reached while recursing'),
    ('mins_exceeded_simple', 'Min scores exceeded during simple calcs'),
    ('recursions_called', 'Recursions called'),
    ('mins_exceeded_recurse', 'Min scores exceeded during recursion'),
    ('mins_after_recurse', 'New min scores found after recursing'),
    ('mins_without_recurse', 'New min scores found without recursing'),
    ('leaf_calcs_complete', 'Leaf node calculations completed in full'),
]

APP_COLORS = {
    'gradient-start'        : '#4795de',
    'gradient-end'          : '#bb637a',
    'tile-green'            : '#16ac55',
    'tile-yellow'           : '#bbaf30',
    'tile-gray'             : '#3a3a3c',
    'yellow-highlight'      : '#FFFF00',
    'screen-background'     : '#121213',
    'standard-gray'         : '#808080',
    'progress-indeterminate': '#4795de',
    'progress-complete'     : '#16ac55',
    'widget-dark'           : '#202020',
    'widget-medium'         : '#202020',
    'widget-bright'         : '#202020',
    'widget-input-dark'     : '#2C2C2C',
    'widget-input-medium'   : '#3F3F3F',
    'widget-input-bright'   : '#4F4F4F'
}

NPRUNE_GLOBAL_DEFAULT  = 15
NPRUNE_ANSWERS_DEFAULT = 15
MAX_DEPTH_DEFAULT      = 10
INITIAL_SUGGESTION = ("TARSE", 15_005)

VALID_GUESSES_URL       = "https://gist.github.com/dracos/dd0668f281e685bad51479e5acaadb93/raw/6bfa15d263d6d5b63840a8e5b64e04b382fdb079/valid-wordle-words.txt"
VALID_GUESSES_FILE      = "data/valid_guesses.txt"
ORIGINAL_ANSWERS_URL    = "https://gist.github.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b/raw/c46f451920d5cf6326d550fb2d6abb1642717852/wordle-answers-alphabetical.txt"
ORIGINAL_ANSWERS_FILE   = "data/original_answers.txt"
PAST_ANSWERS_FILE       = "data/past_answers.txt"
PAST_ANSWERS_URL        = "https://www.rockpapershotgun.com/wordle-past-answers"
ENGLISH_DICTIONARY_FILE = "data/en_US-large.txt"
PATTERN_MATRIX_FILE     = "data/pattern_matrix.npy"
WORD_FEATURES_FILE      = "data/word_features.pkl"
CLASSIFIER_MODEL_FILE   = "data/trained_classifier.pkl"

REQUIRED_SCHEMA = {
    "game_settings": {
        "type": dict,
        "schema": {
            "nprune_global": {"type": int},
            "nprune_answers": {"type": int},
            "max_depth": {"type": int},
            "game_number": {"type": (int, type(None)), "optional": True},
            "initial_suggestion": {"type": list},
        }
    },
    "guesses": {
        "type": dict,
        "schema": {
            "class": {"type": str, "valid_values": ["GetWordsWidget", "ScrapeWordsWidget"]},
            "backend_params": {"type": dict},
            "gui_params": {"type": dict, "gui_only": True},
        }
    },
    "answers": {
        "type": dict,
        "schema": {
            "class": {"type": str, "valid_values": ["GetWordsWidget", "ScrapeWordsWidget"]},
            "backend_params": {"type": dict},
            "gui_params": {"type": dict, "gui_only": True},
        }
    },
    "pattern_matrix": {
        "type": dict,
        "schema": {
            "class": {"type": str, "valid_values": "GetPatternMatrixWidget"},
            "backend_params": {"type": dict},
            "gui_params": {"type": dict, "gui_only": True},
        }
    },
    "classifier": {
        "type": dict,
        "schema": {
            "enabled": {"type": bool},
            "class": {"type": str, "valid_values": "ClassifierSection", "gui_only": True},
            "gui_params": {"type": dict, "gui_only": True},
            "positive_words": {
                "type": dict,
                "required_if": ("classifier.enabled", True),
                "schema": {
                    "class": {"type": str, "valid_values": "DynamicCollapsibleList", "gui_only": True},
                    "items": {
                        "type": list,
                        "schema": {
                            "class": {"type": str, "valid_values": ["GetWordsWidget", "ScrapeWordsWidget"]},
                            "backend_params": {"type": dict},
                            "gui_params": {"type": dict, "gui_only": True}
                        }
                    },
                    "constructors": {"type": dict, "gui_only": True},
                    "gui_params": {"type": dict, "gui_only": True}
                }
            },
            "word_features": {
                "type": dict,
                "required_if": ("classifier.enabled", True),
                "schema": {
                    "class": {"type": str, "valid_values": "GetWordFeaturesWidget", "gui_only": True},
                    "backend_params": {"type": dict},
                    "gui_params": {"type": dict, "gui_only": True}
                }
            },
            "load_model": {
                "type": dict,
                "required_if": ("classifier.enabled", True),
                "schema": {
                    "class": {"type": str, "valid_values": "LoadModelWidget", "gui_only": True},
                    "backend_params": {"type": dict},
                    "gui_params": {"type": dict, "gui_only": True}
                }
            },
        }
    },
    "answer_filters": {
        "type": dict,
        "schema": {
            "class": {"type": str, "valid_values": "DynamicCollapsibleList", "gui_only": True},
            "items": {
                "type": list,
                "schema": { # Schema for each object within the 'items' list
                    "class": {
                        "type": str,
                        "valid_values": [
                            "FilterProbabilityWidget",
                            "FilterSuffixWidget",
                            "FilterFrequencyWidget",
                            "FilterPOSWidget"
                        ]
                    },
                    "backend_params": {"type": dict},
                    "gui_params": {"type": dict, "gui_only": True}
                }
            },
            "constructors": {"type": dict, "gui_only": True},
            "gui_params": {"type": dict, "gui_only": True}
        }
    },
    "answer_sort": {
        "type": dict,
        "schema": {
            "class": {"type": str, "valid_values": "AnswerSortWidget", "gui_only": True},
            "backend_params": {"type": dict},
            "gui_params": {"type": dict, "gui_only": True},
        }
    }
}