import os
import requests
import numpy as np
from tqdm import tqdm
import multiprocessing
from numba import njit
import wordfreq
import nltk
import hashlib

VALID_GUESSES_URL = "https://gist.github.com/dracos/dd0668f281e685bad51479e5acaadb93/raw/6bfa15d263d6d5b63840a8e5b64e04b382fdb079/valid-wordle-words.txt"
VALID_GUESSES_FILE = "data/valid_guesses.txt"
ORIGINAL_ANSWERS_URL = "https://gist.github.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b/raw/c46f451920d5cf6326d550fb2d6abb1642717852/wordle-answers-alphabetical.txt"
ORIGINAL_ANSWERS_FILE = "data/original_answers.txt"
ENGLISH_DICTIONARY_FILE = "data/en_US-large.txt"
DEFAULT_PATTERN_MATRIX_FILE = "data/pattern_matrix.npy"
GREEN = 2
YELLOW = 1
GRAY = 0

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

# def get_words(savefile=VALID_WORDS_FILE, url=VALID_WORDS_URL, refetch=False, save=True) -> np.ndarray[str]:
#     """Retrieves the word list from file it it exists otherwise it fetches from url."""
#     if not refetch:
#         if os.path.exists(savefile):
#             print("Fetching all valid words from file")
#             with open(savefile, 'r') as f:
#                 # The last line might be empty, so we filter it out.
#                 return np.array([line.strip() for line in f if line.strip()], dtype=str)
#     print("No word list exists or refetching requested, fetching from the web")
#     try:
#         response = requests.get(url)
#         # Raises an HTTPError if the HTTP request returned an unsuccessful status code.
#         response.raise_for_status()
#         words_text = response.text
#         if save:
#             print("Saving word list to file")
#             with open(savefile, 'w') as f:
#                 f.write(words_text)
#         # The last line might be empty, so we filter it out.
#         return np.array([word for word in words_text.splitlines() if word])
#     except requests.exceptions.RequestException as e:
#         print(f"Error downloading the word list: {e}")
#         return np.array([], dtype=str)

def get_words(savefile=VALID_GUESSES_FILE, url=VALID_GUESSES_URL, refetch=False, save=True) -> np.ndarray[str]:
    """
    Retrieves the word list, filtering for lowercase a-z words.
    It fetches from a local file if it exists, otherwise from a URL.
    """
    # --- Path 1: Reading from local file ---
    if not refetch and os.path.exists(savefile):
        print("Fetching and filtering words from file")
        with open(savefile, 'r') as f:
            # Filter for non-empty, all-lowercase, a-z only words.
            words = [
                line.strip() for line in f 
                if line.strip() and line.strip().isascii() and line.strip().islower()
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
def FNV_hash(arr: np.ndarray) -> np.int64:
    """
    Computes a hash for a 1D NumPy array of int64 integers.

    This implementation is a variation of the FNV-1a hash algorithm, adapted
    for a sequence of 64-bit integers.
    """
    h = np.uint64(14695981039346656037)  # FNV_offset_basis for 64-bit
    for x in arr:
        h = h ^ np.uint64(x)
        h = h * np.uint64(1099511628211)  # FNV_prime for 64-bit
    return np.int64(h)

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

def filter_words_by_suffix(input_words, filter_words, suffixes=[], min_freq=0) -> np.ndarray[str]:
    if len(suffixes) == 0: 
        return input_words

    words3 = filter_words_by_length(filter_words, 3)
    words4 = filter_words_by_length(filter_words, 4)
    masks = []
    for suffix in suffixes:
        match len(suffix):
            case 1: filter_words = words4
            case 2: filter_words = words3
        mask = np.logical_and(
            np.char.endswith(input_words, suffix.lower()), 
            np.isin([s[:5-len(suffix)] for s in input_words], filter_words)
            )
        masks.append(mask)
    composite_mask = np.logical_or.reduce(masks)
    return input_words[~composite_mask]

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