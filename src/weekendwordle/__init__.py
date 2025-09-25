# all convience imports from backend
from .backend.atomic_ops import (
    atomic_add, atomic_sub, atomic_cas
)
from .backend.cache import (
    AtomicInt, Cache
)
from .backend.classifier import (
    compute_word_features,
    get_word_features, 
    load_classifier, 
    filter_words_by_probability
)
from .backend.core import (
    WordleGame, InvalidWordError, InvalidPatternError
)
from .backend.helpers import (
    get_pattern,
    pattern_str_to_int,
    pattern_to_int,
    int_to_pattern,
    precompute_pattern_matrix,
    get_pattern_matrix,
    get_words,
    scrape_words,
    get_word_freqs,
    get_minimum_freq,
    filter_words_by_frequency,
    PNR_hash, FNV_hash, python_hash, 
    robust_mixing_hash, blake2b_hash,
    filter_words_by_length,
    filter_words_by_POS,
    filter_words_by_suffix,
    get_abs_path
)
from .backend.messenger import (
    UIMessenger, ConsoleMessenger, TextualMessenger
)
from .backend.tests import (
    simulate_game, benchmark
)

# import everything from the config
from .config import *