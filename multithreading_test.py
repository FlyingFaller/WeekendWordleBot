import os
import requests
import numpy as np
from tqdm import tqdm
import time
import multiprocessing

VALID_WORDS_URL = "https://gist.github.com/dracos/dd0668f281e685bad51479e5acaadb93/raw/6bfa15d263d6d5b64e04b382fdb079/valid-wordle-words.txt"
VALID_WORDS_FILE = "wordle_words.txt"
PATTERN_MATRIX_FILE = "pattern_matrix.npy"
GREEN = 2
YELLOW = 1
GRAY = 0

# --- Configuration ---
NPRUNE = 10  # How many branches to explore at each level of recursion
NDEPTH = 2  # How many guesses to look ahead

def get_pattern(guess: str, answer: str) -> list[int]:
    """Calculates the 5-letter pattern for a given guess and answer."""
    pattern = [GRAY] * 5
    letter_count = {}
    used_in_guess = [False] * 5

    # First pass for GREEN letters
    for i in range(5):
        if guess[i] == answer[i]:
            pattern[i] = GREEN
            used_in_guess[i] = True
        else:
            letter_count[answer[i]] = letter_count.get(answer[i], 0) + 1

    # Second pass for YELLOW letters
    for i in range(5):
        if not used_in_guess[i] and letter_count.get(guess[i], 0) > 0:
            pattern[i] = YELLOW
            letter_count[guess[i]] -= 1

    return pattern

def pattern_to_int(pattern: list[int]) -> int:
    """Converts a pattern list (e.g., [2, 1, 0, 0, 0]) to a unique integer."""
    ret_int = 0
    # Using base 3 representation for the pattern
    for i in range(5):
        ret_int += (3**i) * pattern[i]
    return ret_int

def int_to_pattern(num: int) -> list[int]:
    """Converts an integer back to its 5-letter pattern representation."""
    pattern = [GRAY] * 5
    # Since we used little-endian for conversion, we reverse for reconstruction
    for i in range(5):
        num, remainder = divmod(num, 3)
        pattern[i] = remainder
    return pattern


def precompute_pattern_matrix(guesses: np.ndarray, answers: np.ndarray):
    """Generates and saves the pattern matrix if it doesn't exist."""
    nguesses = len(guesses)
    nanswers = len(answers)
    pattern_matrix = np.zeros((nguesses, nanswers), dtype=np.int16)
    print("Precomputing pattern matrix. This may take a while...")
    for i in tqdm(range(nguesses)):
        for j in range(nanswers):
            pattern_matrix[i, j] = pattern_to_int(get_pattern(guesses[i], answers[j]))
    return pattern_matrix

def get_pattern_matrix(guesses: np.ndarray, answers: np.ndarray) -> np.ndarray:
    """Loads the pattern matrix from file or computes it if not found."""
    if os.path.exists(PATTERN_MATRIX_FILE):
        print("Loading pattern matrix from file...")
        return np.load(PATTERN_MATRIX_FILE)
    else:
        pattern_matrix = precompute_pattern_matrix(guesses, answers)
        print("Saving pattern matrix to file...")
        np.save(PATTERN_MATRIX_FILE, pattern_matrix)
        return pattern_matrix

def get_valid_words() -> np.ndarray:
    """Downloads or loads the list of valid Wordle words."""
    if os.path.exists(VALID_WORDS_FILE):
        with open(VALID_WORDS_FILE, 'r') as f:
            return np.array([line.strip() for line in f if line.strip()], dtype=str)
    else:
        print("Downloading word list...")
        try:
            response = requests.get(VALID_WORDS_URL)
            response.raise_for_status()
            words_text = response.text
            with open(VALID_WORDS_FILE, 'w') as f:
                f.write(words_text)
            return np.array([word for word in words_text.splitlines() if word], dtype=str)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the word list: {e}")
            return np.array([], dtype=str)

def compute_entropies(pattern_matrix: np.ndarray) -> np.ndarray:
    """Calculates the Shannon entropy for each guess based on pattern distributions."""
    nguesses, nanswers = pattern_matrix.shape
    if nanswers == 0:
        return np.zeros(nguesses)
        
    entropies = np.zeros(nguesses)
    for i in range(nguesses):
        _, counts = np.unique(pattern_matrix[i], return_counts=True)
        probabilities = counts / nanswers
        entropies[i] = -np.sum(probabilities * np.log2(probabilities))
    return entropies

def find_best_guess_recursive(
    full_pattern_matrix: np.ndarray,
    guess_indices: np.ndarray,
    answer_indices: np.ndarray,
    depth: int,
    max_depth: int
) -> float:
    """
    Recursively finds the best *future* entropy from a given game state.
    This is the core recursive engine.
    """
    # --- BASE CASE ---
    if depth == max_depth or len(answer_indices) <= 2:
        return 0.0

    # --- Step 1: Compute entropies for the current sub-problem ---
    sub_matrix = full_pattern_matrix[np.ix_(guess_indices, answer_indices)]
    initial_entropies = compute_entropies(sub_matrix)

    # --- Step 2: Prune to the top candidates ---
    # We only need to explore the very best branch from this point forward.
    best_sub_idx = np.argmax(initial_entropies)
    
    # --- Step 3: Recurse down the best path ---
    current_guess_idx = guess_indices[best_sub_idx]
    
    patterns, counts = np.unique(full_pattern_matrix[current_guess_idx, answer_indices], return_counts=True)
    probabilities = counts / len(answer_indices)
    
    expected_future_entropy = 0.0
    for i, pattern_int in enumerate(patterns):
        next_answer_indices = answer_indices[full_pattern_matrix[current_guess_idx, answer_indices] == pattern_int]
        next_guess_indices = np.delete(guess_indices, np.where(guess_indices == current_guess_idx))

        # --- RECURSIVE CALL ---
        max_future_entropy = find_best_guess_recursive(
            full_pattern_matrix, next_guess_indices, next_answer_indices, depth + 1, max_depth
        )
        expected_future_entropy += probabilities[i] * max_future_entropy

    # The value of this state is its best initial entropy plus the expected future entropy
    return initial_entropies[best_sub_idx] + expected_future_entropy


def analyze_candidate_worker(args):
    """
    A wrapper function for our recursive search, designed to be called by a multiprocessing Pool.
    It takes a single tuple of arguments to facilitate `pool.map`.
    """
    candidate_idx, initial_entropy, full_pattern_matrix, guess_indices, answer_indices, max_depth = args
    
    # This worker calculates the total expected entropy for ONE top-level candidate word.
    total_expected_entropy = initial_entropy
    
    patterns, counts = np.unique(full_pattern_matrix[candidate_idx, answer_indices], return_counts=True)
    probabilities = counts / len(answer_indices)

    expected_future_entropy = 0.0
    for i, pattern_int in enumerate(patterns):
        # Find the answers that would produce this specific pattern
        next_answer_indices = answer_indices[full_pattern_matrix[candidate_idx, answer_indices] == pattern_int]
        
        # The next set of guesses excludes the one we just "made"
        next_guess_indices = np.delete(guess_indices, np.where(guess_indices == candidate_idx))

        # --- RECURSIVE CALL ---
        # Find the best entropy we can get from the *next* state
        max_future_entropy = find_best_guess_recursive(
            full_pattern_matrix,
            next_guess_indices,
            next_answer_indices,
            depth=1, # Starting the recursive search at depth 1
            max_depth=max_depth
        )
        # Weight this future entropy by the probability of this pattern occurring
        expected_future_entropy += probabilities[i] * max_future_entropy

    total_expected_entropy += expected_future_entropy
    
    # Return the result for this one candidate
    return total_expected_entropy, candidate_idx


if __name__ == "__main__":
    words = get_valid_words()
    # Ensure the matrix uses a smaller integer type to save memory
    pattern_matrix = get_pattern_matrix(words, words).astype(np.int16)

    # Initial state: all words are possible guesses and answers
    initial_guess_indices = np.arange(len(words))
    initial_answer_indices = np.arange(len(words))

    print(f"Starting search with depth={NDEPTH} and prune_factor={NPRUNE}...")
    start_time = time.time()
    
    # --- Step 1: Find the top-level candidates (this is done on the main process) ---
    sub_matrix = pattern_matrix[np.ix_(initial_guess_indices, initial_answer_indices)]
    initial_entropies = compute_entropies(sub_matrix)
    
    num_candidates = min(NPRUNE, len(initial_guess_indices))
    candidate_indices = np.argsort(initial_entropies)[-num_candidates:]
    candidate_entropies = initial_entropies[candidate_indices]

    # --- Step 2: Prepare arguments for the parallel workers ---
    # Each worker will analyze one candidate word.
    worker_args = []
    for i in range(num_candidates):
        candidate_idx = candidate_indices[i]
        entropy = candidate_entropies[i]
        worker_args.append(
            (candidate_idx, entropy, pattern_matrix, initial_guess_indices, initial_answer_indices, NDEPTH)
        )

    # --- Step 3: Create a multiprocessing Pool and distribute the work ---
    print(f"Dispatching {len(worker_args)} candidates to worker processes...")
    # NOTE: We removed `multiprocessing.set_start_method('fork')` because it's
    # not supported on Windows. The library will automatically use the correct
    # default method for the host OS ('spawn' on Windows, 'fork' on Linux/macOS).
    with multiprocessing.Pool() as pool:
        # Use tqdm to show a progress bar for the parallel computation
        results = list(tqdm(pool.imap(analyze_candidate_worker, worker_args), total=len(worker_args)))

    # --- Step 4: Find the best result from all the workers ---
    best_total_entropy, best_word_idx = max(results, key=lambda item: item[0])
    
    end_time = time.time()
    
    if best_word_idx != -1:
        best_word = words[best_word_idx]
        print("\n--- Search Complete ---")
        print(f"Best opening word: '{best_word.upper()}'")
        print(f"Total Expected Entropy: {best_total_entropy:.4f} bits")
        print(f"Calculation time: {end_time - start_time:.2f} seconds")
    else:
        print("Could not determine a best word.")