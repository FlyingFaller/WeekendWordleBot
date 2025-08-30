import os
import requests
import numpy as np
from tqdm import tqdm
import multiprocessing
import threading
from numba import njit
import wordfreq
import nltk
import hashlib
from weekend_wordle.backend.cache import Cache
from bs4 import BeautifulSoup
from numba.types import int64
from numba import njit
from numba.experimental import jitclass
import time

### DEFAULTS ###
VALID_GUESSES_URL = "https://gist.github.com/dracos/dd0668f281e685bad51479e5acaadb93/raw/6bfa15d263d6d5b63840a8e5b64e04b382fdb079/valid-wordle-words.txt"
VALID_GUESSES_FILE = "data/valid_guesses.txt"
ORIGINAL_ANSWERS_URL = "https://gist.github.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b/raw/c46f451920d5cf6326d550fb2d6abb1642717852/wordle-answers-alphabetical.txt"
ORIGINAL_ANSWERS_FILE = "data/original_answers.txt"
PAST_ANSWERS_FILE = 'data/past_answers.txt'
PAST_ANSWERS_URL = 'https://www.rockpapershotgun.com/wordle-past-answers'
ENGLISH_DICTIONARY_FILE = "data/en_US-large.txt"
DEFAULT_PATTERN_MATRIX_FILE = "data/pattern_matrix.npy"
GREEN = 2
YELLOW = 1
GRAY = 0

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

### FUNCTIONS ###
def get_pattern(guess: str, answer: str) -> list[int]:
    """Calculates the wordle pattern for guess word and answer word."""
    pattern = [GRAY]*5
    letter_count = {}
    # Green pass
    for i in range(5):
        if answer[i] == guess[i]:
            # Set equal to 3 if green
            pattern[i] = GREEN
        else:
            # If not green we keep a running tally of all unique non-green letters
            letter_count[str(answer[i])] = letter_count.get(str(answer[i]), 0) + 1

    # Yellow pass
    for i in range(5):
        if (pattern[i] != GREEN) and letter_count.get(str(guess[i]), 0) > 0:
            pattern[i] = YELLOW
            letter_count[str(guess[i])] -= 1

    return pattern

def pattern_str_to_int(pattern: str) -> int:
    pattern_list = []
    for c in pattern.upper():
        match c:
            case "G": pattern_list.append(GREEN)
            case "Y": pattern_list.append(YELLOW)
            case _: pattern_list.append(GRAY)

    return pattern_to_int(pattern_list)

def pattern_to_int(pattern: list[int]) -> int:
    """Converts a pattern list represeting a wordle pattern to a unique int"""
    ret_int = 0
    for i in range(5):
        ret_int += (3**i)*pattern[i]
    return ret_int

def int_to_pattern(num: int) -> list[int]:
    """Converts an int back to its pattern list"""
    pattern = 5*[GRAY]
    for i in range(4, -1, -1):
        pattern[i], num = divmod(num, 3**i)
    return pattern

def compute_pattern_row(args):
    """Worker function to compute a single row of the pattern matrix"""
    guess_word, answers = args
    row = np.zeros(len(answers), dtype=np.uint8)
    for j, answer_word in enumerate(answers):
        row[j] = pattern_to_int(get_pattern(guess_word, answer_word))
    return row

def precompute_pattern_matrix(guesses:np.ndarray[str], answers: np.ndarray[str]) -> np.ndarray[int]:
    """Generates the pattern matrix from word list to make searching efficient later"""
    nguesses = len(guesses)
    worker_args = [(guesses[i], answers) for i in range(nguesses)]
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(compute_pattern_row, worker_args), total=nguesses, desc="Building Pattern Matrix"))
    pattern_matrix = np.vstack(results)
    return pattern_matrix

def get_pattern_matrix(guesses:np.ndarray[str], answers: np.ndarray[str], savefile=DEFAULT_PATTERN_MATRIX_FILE, recompute=False, save=True) -> np.ndarray[str]:
    """Retrieves the pattern matrix from file if it exists otherwise it generates it and saves it to file."""
    if not recompute:
        if os.path.exists(savefile):
            print("Fetching pattern matrix from file")
            pattern_matrix = np.load(savefile)
            return pattern_matrix
    print("No pattern matrix file found or recompute requested")
    pattern_matrix = precompute_pattern_matrix(guesses, answers)
    if save:
        print("Saving pattern matrix to file")
        np.save(savefile, pattern_matrix)
    return pattern_matrix

def get_words(savefile=VALID_GUESSES_FILE, url=VALID_GUESSES_URL, refetch=False, save=True, include_uppercase=False) -> np.ndarray[str]:
    """
    Retrieves the word list, filtering for lowercase a-z words.
    It fetches from a local file if it exists, otherwise from a URL.
    """
    # --- Path 1: Reading from local file ---
    if not refetch and os.path.exists(savefile):
        print("Fetching words from file")
        with open(savefile, 'r') as f:
            # Filter for non-empty, all-lowercase, a-z only words.
            words = [
                line.strip().lower() for line in f 
                if line.strip() and line.strip().isascii() and (line.strip().islower() or include_uppercase)
            ]
            return np.array(words, dtype=str)

    # --- Path 2: Fetching from the web ---
    print("No word list exists or refetching requested, fetching from the web")
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes

        # Filter the freshly downloaded list.
        all_words = response.text.splitlines()
        filtered_words = [
            word for word in all_words 
            if word and word.isascii() and word.islower()
        ]

        if save:
            print(f"Saving {len(filtered_words)} filtered words to file")
            # Save the *filtered* list, with each word on a new line.
            with open(savefile, 'w') as f:
                f.write('\n'.join(filtered_words))
        
        return np.array(filtered_words, dtype=str)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the word list: {e}")
        return np.array([], dtype=str)
    
def scrape_words(savefile: str | None = PAST_ANSWERS_FILE, 
                 url: str | None = PAST_ANSWERS_URL, 
                 refetch: bool = False, 
                 save: bool = True,
                 header: tuple[str, str] = ('All Wordle answers', 'h2')) -> np.ndarray:
    
    if not refetch and os.path.exists(savefile):
        print("Fetching words from file")
        with open(savefile, 'r') as f:
            # Filter for non-empty, all-lowercase, a-z only words.
            words = [
                line.strip().lower() for line in f 
                if line.strip() and line.strip().isascii()
            ]
            return np.array(words, dtype=str)
    
    print("No word list exists or refetching requested, fetching from the web")     
    if not url:
        raise ValueError("A URL must be provided to scrape words when no valid savefile is found.")
        
    words = []
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the <h2> tag that acts as an anchor for our data
        header_tag = soup.find(header[1], string=header[0])
        if not header_tag:
            print(f"Error: Could not find the {header[0]} header on the page.")
            return np.array([], dtype=str)

        # Find the <ul> tag immediately following the header
        word_list_ul = header_tag.find_next_sibling('ul')
        if not word_list_ul:
            print("Error: Could not find the word list (<ul>) after the header.")
            return np.array([], dtype=str)

        # Extract all list items and validate them
        list_items = word_list_ul.find_all('li')
        raw_words = [li.get_text(strip=True).upper() for li in list_items if li.get_text(strip=True)]
        words = [word.lower() for word in raw_words if len(word) == 5 and word.isalpha()]

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the web request: {e}")
        return np.array([], dtype=str)
    except Exception as e:
        print(f"An unexpected error occurred during scraping: {e}")
        return np.array([], dtype=str)

    # --- Case 3: Save the scraped words to a file ---
    if save and savefile and words:
        print(f"Saving {len(words)} words to {savefile}")
        try:
            with open(savefile, 'w') as f:
                for word in words:
                    f.write(f"{word}\n")
        except Exception as e:
            print(f"Error saving words to file: {e}")
            
    return np.array(words, dtype=str)

def get_word_freqs(words: np.ndarray[str]) -> np.ndarray[float]:
    frequencies = np.zeros(len(words))
    for i, word in enumerate(words):
        frequencies[i] = wordfreq.word_frequency(word.lower(), 'en')
    return frequencies

def get_minimum_freq(words: np.ndarray[str]) -> tuple[float, int, str]:
    frequencies = get_word_freqs(words)
    word_idx = np.argmin(frequencies)
    return (np.min(frequencies), word_idx, words[word_idx])

def filter_words_by_occurance(words: np.ndarray, min_freq: float = 1e-7) -> np.ndarray:
    common_words = []
    print(f"Filtering {len(words)} words with a minimum frequency of {min_freq}...")
    
    # Using tqdm to create a progress bar
    for word in tqdm(words, desc="Analyzing word frequency"):
        # 'word_frequency' returns the frequency of the word in English ('en').
        # If the word is not found, it returns 0.
        frequency = wordfreq.word_frequency(word.lower(), 'en')
        if frequency >= min_freq:
            common_words.append(word)
    return np.array(common_words)

@njit(cache=True)
def PNR_hash(arr: np.ndarray) -> np.int64:
    P = np.int64(31)
    M = np.int64(10**9 + 7)
    hash_value = np.int64(0)
    for x in arr:
        hash_value = (hash_value * P + x) % M
    return hash_value

@njit(cache=True)
def FNV_hash(arr: np.ndarray) -> np.uint64:
    """
    Computes a hash for a 1D NumPy array of int64 integers.

    This implementation is a variation of the FNV-1a hash algorithm, adapted
    for a sequence of 64-bit integers.
    """
    h = np.uint64(14695981039346656037)  # FNV_offset_basis for 64-bit
    for x in arr:
        h = h ^ np.uint64(x)
        h = h * np.uint64(1099511628211)  # FNV_prime for 64-bit
    return h

def python_hash(arr: np.ndarray) -> np.int64:
    return hash(arr.tobytes())

@njit(cache=True)
def robust_mixing_hash(arr: np.ndarray) -> np.int64:
    """
    A robust, njit-compatible hash function for a 1D NumPy array of int64s.

    This algorithm uses strong bit-mixing principles inspired by MurmurHash3's
    finalizer to provide excellent distribution and collision resistance,
    making it much more robust than FNV-1a for difficult datasets.
    """
    h = np.uint64(len(arr)) # Start with the length as a seed

    for x in arr:
        # Incorporate each element
        k = np.uint64(x)

        # Mixing constants - chosen for their properties in creating good bit dispersion
        k *= np.uint64(0xff51afd7ed558ccd)
        k ^= k >> np.uint64(33)
        k *= np.uint64(0xc4ceb9fe1a85ec53)
        k ^= k >> np.uint64(33)

        # Mix it into the main hash value
        h ^= k
        # Rotate left by 27 bits - ensures bits from different positions interact
        h = (h << np.uint64(27)) | (h >> np.uint64(37))
        h = h * np.uint64(5) + np.uint64(0x52dce729)

    # Final mixing function (aka "finalizer")
    # This is crucial for breaking up final patterns.
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xff51afd7ed558ccd)
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xc4ceb9fe1a85ec53)
    h ^= h >> np.uint64(33)

    return np.int64(h)


def blake2b(arr: np.ndarray) -> str:
  """
  Calculates a secure 128-bit (16-byte) BLAKE2b hash for a NumPy array.
  """
  # digest_size=16 specifies a 128-bit output
  hasher = hashlib.blake2b(digest_size=16)
  hasher.update(np.ascontiguousarray(arr).tobytes())
  return hasher.hexdigest()

def get_nltk_words(download=False) -> np.ndarray[str]:
    if download:
        nltk.download('words')
    return nltk.corpus.words.words()

def filter_words_by_length(words, length) -> np.ndarray[str]:
    return np.array([word.lower() for word in words if len(word) == length])

def filter_words_by_POS(input_words, tags=['NNS', 'VBD', 'VBN'], download=False) -> np.ndarray[str]:
    if download:
        nltk.download('averaged_perceptron_tagger')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger_eng')
    
    tagged_words = nltk.pos_tag(input_words)
    exclude_tags = set(tags)

    # Keep a word only if its tag is NOT in our exclusion set
    filtered_list = [word for word, tag in tagged_words if tag not in exclude_tags]

    return np.array(filtered_list)

def filter_words_by_suffix(
    input_words: np.ndarray[str],
    filter_words: np.ndarray[str],
    suffixes: list[str | tuple[str, ...]] = []
) -> np.ndarray[str]:
    """
    Filters 5-letter words that are shorter words with a suffix.

    This function can handle simple suffix rules (e.g., 's') and complex
    rules with exceptions (e.g., ('d', 'r')), where words ending in 'd'
    are not filtered if the 'd' is preceded by an 'r'.

    Args:
        input_words: A NumPy array of 5-letter words to be filtered.
        filter_words: A NumPy array of all valid words to check stems against.
        suffixes: A list of rules. Each rule can be:
            - A str (e.g., 'es') for a simple suffix.
            - A tuple (e.g., ('s', 's')) where the first item is the
              suffix and the following items are preceding character exceptions.

    Returns:
        A NumPy array of words with the filtered words removed.
    """
    if not suffixes:
        return input_words

    # Pre-filter the dictionary for 3 and 4-letter words for efficiency
    words3 = filter_words_by_length(filter_words, 3)
    words4 = filter_words_by_length(filter_words, 4)

    masks_to_remove = []
    for rule in suffixes:
        # 1. Unpack the rule into suffix and exceptions
        if isinstance(rule, tuple):
            if not rule: continue # Skip empty tuple
            suffix, *exceptions = rule
        else:
            suffix = rule
            exceptions = []
        
        # Determine the stem length and which word list to use
        stem_len = 5 - len(suffix)
        if stem_len == 4:
            valid_stems = words4
        elif stem_len == 3:
            valid_stems = words3
        else:
            continue # Skip suffixes of invalid length

        # 2. Create a base mask for words that are candidates for removal
        #    - Condition 1: Word ends with the suffix.
        #    - Condition 2: The stem (word without suffix) is a valid word.
        potential_removal_mask = np.logical_and(
            np.char.endswith(input_words, suffix.lower()),
            np.isin([word[:stem_len] for word in input_words], valid_stems)
        )

        # 3. If there are exceptions, create a mask to prevent removal
        if exceptions:
            # Get the character that precedes the suffix for each word
            preceding_chars = np.array([word[-len(suffix)-1] for word in input_words])
            # Create a mask that is True for words that meet the exception criteria
            exception_mask = np.isin(preceding_chars, exceptions)
            
            # A word should be removed only if it's a potential candidate
            # AND it does NOT meet the exception criteria.
            final_mask_for_rule = np.logical_and(potential_removal_mask, ~exception_mask)
        else:
            # If no exceptions, all potential candidates are marked for removal
            final_mask_for_rule = potential_removal_mask
        
        masks_to_remove.append(final_mask_for_rule)

    # Combine all removal masks. A word is removed if it matches ANY rule.
    if not masks_to_remove:
        return input_words
        
    composite_removal_mask = np.logical_or.reduce(masks_to_remove)

    return input_words[~composite_removal_mask]

def print_stats(event_counts, cache: Cache):
    """
    Prints formatted statistics from an EventCounter object.
    It directly uses the EVENTS list defined in this same module.
    """
    padding = 45

    print(f"\nStats:")
    for name, description in EVENTS:
        value = getattr(event_counts, name)
        print(f"{description:.<{padding}}{value:,}")

    print(f"{'Cache entries':.<{padding}}{cache.nentries():,}")
    print(f"{'Cache segments':.<{padding}}{cache.nsegments():,}")

def build_event_counter_class():
    """
    Dynamically builds the EventCounter jitclass as a string and executes it.
    This provides the maintainability of being driven by the EVENTS list while
    satisfying Numba's need for a static class definition.
    """
    # Start the class definition string
    class_def = """
@jitclass([('counts', int64[:])])
class EventCounter:
    def __init__(self):
        self.counts = np.zeros(NEVENTS, dtype=np.int64)

    def merge(self, other_counters):
        for counter in other_counters:
            self.counts += counter.counts

    @staticmethod
    def spawn(n: int):
        lst = []
        for _ in range(n):
            lst.append(EventCounter())
        return lst     
"""

    # Add incrementer methods from the EVENTS list
    for i, (name, _) in enumerate(EVENTS):
        class_def += f"""
    def inc_{name}(self):
        self.counts[{i}] += 1
"""

    # Add getter properties from the EVENTS list
    for i, (name, _) in enumerate(EVENTS):
        class_def += f"""
    @property
    def {name}(self):
        return self.counts[{i}]
"""
    return class_def

event_counter_class_string = build_event_counter_class()
exec_scope = {
    'jitclass': jitclass,
    'int64': int64,
    'np': np,
    'NEVENTS': len(EVENTS)
}
exec(event_counter_class_string, exec_scope)
EventCounter = exec_scope['EventCounter']

def solver_progress_bar(progress_array: np.ndarray[np.float64],
                        pbar: tqdm,
                        stop_event: threading.Event,
                        refresh=0.25):
    """
    Monitors the progress array and updates the tqdm bar.
    Exits when stop_event is set.
    """
    while not stop_event.is_set():
        total = progress_array[-1]
        
        if total > 0:
            if pbar.total != total:
                pbar.total = total

            current_count = np.sum(progress_array[:-1])
            current_count = min(current_count, total)
            pbar.n = current_count
            pbar.refresh()
        
        time.sleep(refresh)

    # Ensure the bar is at 100% when the process is finished.
    total = progress_array[-1] if progress_array[-1] > 0 else 1.0
    if pbar.total != total:
        pbar.total = total
    pbar.n = total
    pbar.refresh()
    pbar.close()