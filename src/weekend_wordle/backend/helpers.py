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
from bs4 import BeautifulSoup
from numba.types import int64
from numba import njit
from numba.experimental import jitclass
import time
from pathlib import Path

from weekend_wordle.backend.cache import Cache
from weekend_wordle.backend.messenger import UIMessenger, ConsoleMessenger

from weekend_wordle.config import *

### FUNCTIONS ###
def get_messenger(messenger: UIMessenger = None) -> UIMessenger:
    return messenger if messenger is not None else ConsoleMessenger()
    
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

def precompute_pattern_matrix(
    guesses: np.ndarray[str],
    answers: np.ndarray[str],
    messenger: UIMessenger
    ) -> np.ndarray[int]:
    """Generates the pattern matrix from word list efficiently."""
    nguesses = len(guesses)
    worker_args = [(guesses[i], answers) for i in range(nguesses)]
    
    # This log is now correctly indented as part of the parent task.
    messenger.task_log("Starting parallel computation...", level="STEP")
    messenger.start_progress(total=nguesses, desc="Building Pattern Matrix")
    
    results = []
    # NOTE: Multiprocessing pools can have issues with some complex objects.
    # If the messenger object causes pickling errors, it may need to be handled differently.
    with multiprocessing.Pool() as pool:
        for result_row in pool.imap(compute_pattern_row, worker_args):
            results.append(result_row)
            messenger.update_progress()
            
    messenger.stop_progress()
    # Use INFO as this is a step, not the final success message of the whole task.
    messenger.task_log("Computation complete.", level="INFO")
    
    return np.vstack(results)

def get_pattern_matrix(guesses:np.ndarray[str], 
                       answers: np.ndarray[str], 
                       savefile: str = PATTERN_MATRIX_FILE, 
                       recompute: bool = False, 
                       save: bool = True,
                       messenger: UIMessenger = None) -> np.ndarray:
    """Retrieves the pattern matrix from file if it exists, otherwise generates and saves it."""
    messenger = get_messenger(messenger)

    with messenger.task("Acquiring pattern matrix"):
        # --- Path 2: Load from local file ---
        if not recompute and os.path.exists(savefile):
            messenger.task_log("Found local file. Loading matrix...", level="INFO")
            pattern_matrix = np.load(savefile)
            messenger.task_log(f"Loaded matrix with shape {pattern_matrix.shape}.", level="INFO")
            return pattern_matrix
    
        # --- Path 1: Compute if recompute is forced or file doesn't exist ---
        if recompute:
            messenger.task_log("Recompute requested. Starting new computation...", level="INFO")
        else:
            messenger.task_log("Local file not found. Starting new computation...", level="INFO")
        
        pattern_matrix = precompute_pattern_matrix(guesses, answers, messenger)

        if save:                
            messenger.task_log(f"Saving to {savefile}...", level="INFO")
            np.save(savefile, pattern_matrix)
            messenger.task_log("Save complete.", level="INFO")
        return pattern_matrix

def get_words(savefile               = VALID_GUESSES_FILE, 
              url                    = VALID_GUESSES_URL,
              refetch                = False,
              save                   = True,
              include_uppercase      = False,
              messenger: UIMessenger = None) -> np.ndarray[str]:
    """
    Retrieves the word list, filtering for lowercase a-z words.
    It fetches from a local file if it exists, otherwise from a URL.
    """
    messenger = get_messenger(messenger)
    
    with messenger.task(f"Acquiring word list"):

        # --- Path 1: Fetch from the web if refetch is forced or file doesn't exist ---
        if not refetch and os.path.exists(savefile):
            messenger.task_log("Found local file. Loading words...", level="INFO")
            with open(savefile, 'r') as f:
                words = [
                    line.strip().lower() for line in f 
                    if line.strip() and line.strip().isascii() and (line.strip().islower() or include_uppercase)
                ]
            messenger.task_log(f"Found {len(words)} words.", level="INFO")
            return np.array(words, dtype=str)

        # --- Path 2: Fetch from the web if refetch is forced or file doesn't exist ---
        if refetch:
            messenger.task_log("Refetch requested. Fetching from the web...", level="INFO")
        else:
            messenger.task_log("Local file not found. Fetching from the web...", level="INFO")
        
        # The messenger's task context manager will handle any exceptions here.
        response = requests.get(url)
        response.raise_for_status()

        all_words = response.text.splitlines()
        filtered_words = [
            word for word in all_words 
            if word and word.isascii() and word.islower()
        ]
        messenger.task_log(f"Downloaded {len(filtered_words)} words.", level="INFO")

        if save:
            messenger.task_log(f"Saving to {savefile}...", level="INFO")
            with open(savefile, 'w') as f:
                f.write('\n'.join(filtered_words))
            messenger.task_log("Save complete.", level="INFO")
        
        return np.array(filtered_words, dtype=str)
    
def scrape_words(savefile: str | None = PAST_ANSWERS_FILE, 
                 url: str | None = PAST_ANSWERS_URL, 
                 refetch: bool = False, 
                 save: bool = True,
                 header: tuple[str, str] = ('All Wordle answers', 'h2'),
                 messenger: UIMessenger = None) -> np.ndarray:
    """
    Scrapes a list of 5-letter words from a given URL, or loads them from a local file.
    """
    messenger = get_messenger(messenger)
    
    with messenger.task("Aquiring scraped words"):

        # --- Path 1: Load from local file if it exists and refetch is false ---
        if not refetch and os.path.exists(savefile):
            messenger.task_log("Found local file. Loading words...", level="INFO")
            with open(savefile, 'r') as f:
                words = [
                    line.strip().lower() for line in f 
                    if line.strip() and line.strip().isascii()
                ]
            messenger.task_log(f"Found {len(words)} words.", level="INFO")
            return np.array(words, dtype=str)

        # --- Path 2: Scrape from the web ---
        if refetch:
            messenger.task_log("Refetch requested. Scraping from the web...", level="INFO")
        else:
            messenger.task_log("Local file not found. Scraping from the web...", level="INFO")

        if not url:
            raise ValueError("A URL must be provided to scrape words when no local file is available.")

        # Let the task context manager handle request exceptions
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        messenger.task_log("Web page retrieved successfully.", level="INFO")

        soup = BeautifulSoup(response.text, 'html.parser')

        header_tag = soup.find(header[1], string=header[0])
        if not header_tag:
            raise ValueError(f"Could not find the '{header[0]}' header on the page.")

        word_list_ul = header_tag.find_next_sibling('ul')
        if not word_list_ul:
            raise ValueError("Could not find the word list (<ul>) after the header.")

        messenger.task_log("Found word list. Extracting and validating words...", level="INFO")
        list_items = word_list_ul.find_all('li')
        raw_words = [li.get_text(strip=True).upper() for li in list_items if li.get_text(strip=True)]
        words = [word.lower() for word in raw_words if len(word) == 5 and word.isalpha()]
        messenger.task_log(f"Extracted {len(words)} valid words.", level="INFO")

        if save and savefile:
            messenger.task_log(f"Saving to {savefile}...", level="INFO")
            with open(savefile, 'w') as f:
                f.write('\n'.join(words))
            messenger.task_log("Save complete.", level="INFO")

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

def filter_words_by_frequency(words: np.ndarray, 
                              min_freq: float = 1e-7,
                              messenger: UIMessenger = None) -> np.ndarray:
    messenger = get_messenger(messenger)
    initial_count = len(words)

    with messenger.task(f"Filtering by frequency (min: {min_freq})"):
        common_words = []
        messenger.start_progress(total=initial_count, desc="Analyzing Word Frequency")
        for word in words:
            frequency = wordfreq.word_frequency(word.lower(), 'en')
            if frequency >= min_freq:
                common_words.append(word)
            messenger.update_progress()
        messenger.stop_progress()
        
        filtered_words = np.array(common_words)
        final_count = len(filtered_words)
        messenger.task_log(f"Filtered from {initial_count} ⟶  {final_count} words.", level="INFO")
        return filtered_words

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

def filter_words_by_length(words: np.ndarray[str], 
                           length: int, 
                           messenger: UIMessenger = None) -> np.ndarray[str]:
    messenger = get_messenger(messenger)
    initial_count = len(words)

    with messenger.task(f"Filtering by length (== {length})"):
        filtered_list = [word.lower() for word in words if len(word) == length]
        filtered_words = np.array(filtered_list)
        final_count = len(filtered_words)
        messenger.task_log(f"Filtered from {initial_count} ⟶  {final_count} words.", level="INFO")
        return filtered_words

def filter_words_by_POS(input_words: np.ndarray[str], 
                        tags: list[str] = ['NNS', 'VBD', 'VBN'], 
                        download: bool = False, 
                        messenger: UIMessenger = None) -> np.ndarray[str]:
    messenger = get_messenger(messenger)
    initial_count = len(input_words)

    with messenger.task(f"Filtering by Part-of-Speech (excluding {tags})"):
        if download:
            messenger.task_log('Downloading NLTK dependencies...', level="INFO")
            # Download quietly to avoid cluttering our log
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        
        messenger.task_log("Tagging words...", level="INFO")
        tagged_words = nltk.pos_tag(input_words)
        exclude_tags = set(tags)

        filtered_list = [word for word, tag in tagged_words if tag not in exclude_tags]
        
        filtered_words = np.array(filtered_list)
        final_count = len(filtered_words)
        messenger.task_log(f"Filtered from {initial_count} ⟶  {final_count} words.", level="INFO")
        return filtered_words

def filter_words_by_suffix(
    input_words: np.ndarray[str],
    filter_words: np.ndarray[str],
    suffixes: list[str | tuple[str, ...]] = [],
    messenger: UIMessenger = None) -> np.ndarray[str]:
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
    messenger = get_messenger(messenger)
    initial_count = len(input_words)

    with messenger.task("Filtering by suffixes"):
        if not suffixes:
            messenger.task_log("No suffixes provided, returning.", level="INFO")
            return input_words

        # These sub-tasks will now appear as nested, logged operations.
        words3 = filter_words_by_length(filter_words, 3, messenger)
        words4 = filter_words_by_length(filter_words, 4, messenger)

        messenger.start_progress(total=len(suffixes), desc='Evaluating Suffixes')
        masks_to_remove = []
        for rule in suffixes:
            messenger.update_progress()
            if isinstance(rule, tuple):
                if not rule: continue
                suffix, *exceptions = rule
            else:
                suffix = rule
                exceptions = []
            
            stem_len = 5 - len(suffix)
            if stem_len == 4:
                valid_stems = words4
            elif stem_len == 3:
                valid_stems = words3
            else:
                continue

            potential_removal_mask = np.logical_and(
                np.char.endswith(input_words, suffix.lower()),
                np.isin([word[:stem_len] for word in input_words], valid_stems)
            )

            if exceptions:
                preceding_chars = np.array([word[-len(suffix)-1] for word in input_words])
                exception_mask = np.isin(preceding_chars, exceptions)
                final_mask_for_rule = np.logical_and(potential_removal_mask, ~exception_mask)
            else:
                final_mask_for_rule = potential_removal_mask
            
            masks_to_remove.append(final_mask_for_rule)
            
        messenger.stop_progress()

        if not masks_to_remove:
            messenger.task_log("No matching words found, returning.", level="INFO")
            return input_words
            
        composite_removal_mask = np.logical_or.reduce(masks_to_remove)
        
        filtered_words = input_words[~composite_removal_mask]
        final_count = len(filtered_words)
        messenger.task_log(f"Filtered from {initial_count} ⟶  {final_count} words.", level="INFO")
        return filtered_words

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

def get_abs_path(usr_path_str: str, root_path: Path = PROJECT_ROOT) -> Path:
    user_path = Path(usr_path_str)

    if user_path.is_absolute():
        # If it's absolute, use it directly.
        return user_path
    else:
        # If it's relative, assume it's relative to the project root.
        return root_path / user_path