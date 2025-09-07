from pathlib import Path
from numba import get_num_threads

# Potentially temporary root path placement
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = 'C:/Users/PELICAN-5/Documents/WeekendWordleBot/'

NTHREADS = get_num_threads()

STARTING_GUESS = "TALES"
STARTING_GUESS_STATS = ('-', '-')

GREEN = 2
YELLOW = 1
GRAY = 0

VALID_GUESSES_URL = "https://gist.github.com/dracos/dd0668f281e685bad51479e5acaadb93/raw/6bfa15d263d6d5b63840a8e5b64e04b382fdb079/valid-wordle-words.txt"
VALID_GUESSES_FILE = "data/valid_guesses.txt"
ORIGINAL_ANSWERS_URL = "https://gist.github.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b/raw/c46f451920d5cf6326d550fb2d6abb1642717852/wordle-answers-alphabetical.txt"
ORIGINAL_ANSWERS_FILE = "data/original_answers.txt"
PAST_ANSWERS_FILE = 'data/past_answers.txt'
PAST_ANSWERS_URL = 'https://www.rockpapershotgun.com/wordle-past-answers'
ENGLISH_DICTIONARY_FILE = "data/en_US-large.txt"
PATTERN_MATRIX_FILE = "data/pattern_matrix.npy"
WORD_FEATURES_FILE = "data/word_features.pkl"
CLASSIFIER_MODEL_FILE = "data/wordle_classifier.pkl"

CLASSIFIER_CONFIG = {
    'use_vectors': True,
    'spy_rate': 0.15,
    'max_iterations': 1000,
    'convergence_tolerance': 1e-2,
    'random_seed': None,
    'evaluation_threshold': 0.07,
    'explicit_features': {
        'frequency': 1.0, 
        'is_regular_plural': 1.0, 
        'is_irregular_plural': 1.0,
        'is_past_tense': 1.0, 
        'is_adjective': 1.0,
        'is_proper_noun': 1.0, 
        'is_gerund': 1.0, 
        'vowel_count': 1.0, 
        'has_double_letter': 1.0
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

NPRUNE_GLOBAL_DEFAULT = 15
NPRUNE_ANSWERS_DEFAULT = 15
MAX_DEPTH_DEFAULT = 10

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