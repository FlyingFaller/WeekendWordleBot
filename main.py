import os
import requests
import numpy as np
from tqdm import tqdm
import random
import multiprocessing
import time
from numba import njit, prange

VALID_WORDS_URL = "https://gist.github.com/dracos/dd0668f281e685bad51479e5acaadb93/raw/6bfa15d263d6d5b63840a8e5b64e04b382fdb079/valid-wordle-words.txt"
VALID_WORDS_FILE = "wordle_words.txt"
PATTERN_MATRIX_FILE = "pattern_matrix.npy"
GREEN = 2
YELLOW = 1
GRAY = 0

NPRUNE = 250
NDEPTH = 2

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
    row = np.zeros(len(answers), dtype=np.int16)
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

def get_pattern_matrix(guesses:np.ndarray[str], answers: np.ndarray[str], recompute=False) -> np.ndarray[str]:
    """Retrieves the pattern matrix from file if it exists otherwise it generates it and saves it to file."""
    if os.path.exists(PATTERN_MATRIX_FILE) and not recompute:
        print("Fetching pattern matrix from file")
        pattern_matrix = np.load(PATTERN_MATRIX_FILE)
    else:
        print("No pattern matrix file found, recomputing")
        pattern_matrix = precompute_pattern_matrix(guesses, answers)
        np.save(PATTERN_MATRIX_FILE, pattern_matrix)
    return pattern_matrix

def get_valid_words(refetch=False) -> np.ndarray[str]:
    """Retrieves the word list from file it it exists otherwise it fetches and saves it to file."""
    if os.path.exists(VALID_WORDS_FILE) and not refetch:
        print("Fetching all valid words from file")
        with open(VALID_WORDS_FILE, 'r') as f:
            # The last line might be empty, so we filter it out.
            return np.array([line.strip() for line in f if line.strip()], dtype=str)
    else:
        print("No word list exists, fetching from the web")
        try:
            response = requests.get(VALID_WORDS_URL)
            # Raises an HTTPError if the HTTP request returned an unsuccessful status code.
            response.raise_for_status()
            words_text = response.text
            with open(VALID_WORDS_FILE, 'w') as f:
                f.write(words_text)
            # The last line might be empty, so we filter it out.
            return [word for word in words_text.splitlines() if word]
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the word list: {e}")
            return np.array([], dtype=str)

def compute_entropies(pattern_matrix: np.ndarray) -> np.ndarray:
    """Calculates the expected Shannon entropy for each guess considering all remaining answers."""
    nguesses, nanswers = pattern_matrix.shape

    if nanswers == 0: # If this triggers we have eliminated all possible answers...
        return np.zeros(nguesses)

    entropy_vals = np.zeros(nguesses)
    for i in range(nguesses):
        patterns, counts = np.unique(pattern_matrix[i], return_counts=True)
        px = counts/nanswers

        entropy = -np.sum(px*np.log2(px))
        entropy_vals[i] = entropy
    return entropy_vals

@njit(parallel=True, cache=True)
def fast_entropies(pattern_matrix: np.ndarray, guess_indicies: np.ndarray, answer_indicies: np.ndarray) -> np.ndarray:
    """Faster implementation of compute_entropies that takes the full pattern_matrix"""
    nguesses = len(guess_indicies)
    nanswers = len(answer_indicies)

    entropy_vals = np.zeros(nguesses, dtype=np.float64)
    if nanswers == 0: 
        return entropy_vals
    
    # precalculation
    log2_nanswers = np.log2(nanswers)

    for i in prange(nguesses):
        guess_idx = guess_indicies[i] # global index for current guess
        pattern_row = pattern_matrix[guess_idx, answer_indicies] # get all patterns for the current guess
        counts = np.bincount(pattern_row)
        counts = counts[counts > 0]
        sum_c_log2_c = np.sum(counts * np.log2(counts))
        entropy_vals[i] = log2_nanswers - (sum_c_log2_c / nanswers)
    return entropy_vals

@njit()
def recursive_search(
    full_pattern_matrix: np.ndarray,
    guess_indicies: np.ndarray,
    answer_indicies: np.ndarray,
    current_depth: int
    ) -> tuple[float, int]:
    nanswers = len(answer_indicies)

    H_array = fast_entropies(full_pattern_matrix, guess_indicies, answer_indicies)

    top_H_indicies = np.flip(np.argsort(H_array))[0:NPRUNE] # Sort entropies best to worst and get the top indicies into pruned guess list
    top_H_values = H_array[top_H_indicies] # Get values of top entropies

    if current_depth == 1: # If last search
        return (top_H_values[0], guess_indicies[top_H_indicies[0]]) # Return the max entropy and associated global guess index

    for i in range(0, NPRUNE):
        guess_idx_local = top_H_indicies[i] # index of guess into current pruned guess list
        guess_idx_global = guess_indicies[guess_idx_local] # retrieve the real index
        pattern_matrix_row = full_pattern_matrix[guess_idx_global, answer_indicies] # row pruned pattern matrix for current word

        # do this instead of np.unique for speed and numba compatability
        all_counts = np.bincount(pattern_matrix_row, minlength=243)
        patterns = np.nonzero(all_counts)[0]
        counts = all_counts[patterns]

        probabilities = counts/nanswers # probability of falling into each pattern bucket

        weighted_H = 0.0
        for (px, pattern) in zip(probabilities, patterns):
            next_answer_indicies = answer_indicies[pattern_matrix_row == pattern] # Prune the answer indicies we are considering

            if len(next_answer_indicies) <= 1: # If our solution space shinks to 1 we have solved the wordle and there is no more information to get
                next_H = 0.0
            else:
                next_guess_indicies = np.delete(guess_indicies, guess_idx_local) # Get rid of current guess for next round

                next_H, next_guess_idx = recursive_search(full_pattern_matrix, next_guess_indicies, next_answer_indicies, current_depth-1)
            weighted_H += px*next_H

        top_H_values[i] += weighted_H

    best_local_index = top_H_indicies[np.argmax(top_H_values)]
    return (np.max(top_H_values), guess_indicies[best_local_index])

def logged_recursive_search(
    full_pattern_matrix: np.ndarray,
    guess_indicies: np.ndarray,
    answer_indicies: np.ndarray,
    depth: int
    ) -> tuple[float, int]:
    """
    Python-level wrapper to show a smooth progress bar for the top-level search.
    """
    nanswers = len(answer_indicies)
    H_array = fast_entropies(full_pattern_matrix, guess_indicies, answer_indicies)
    top_H_indicies = np.flip(np.argsort(H_array))[0:NPRUNE]
    top_H_values = H_array[top_H_indicies]

    if depth == 1:
        return (top_H_values[0], guess_indicies[top_H_indicies[0]])

    # First, calculate the total number of patterns we will explore to set up the progress bar.
    # We also store the results to avoid redundant calculations.
    total_patterns_to_explore = 0
    precalculated_data = []
    for i in range(NPRUNE):
        guess_idx_global = guess_indicies[top_H_indicies[i]]
        pattern_matrix_row = full_pattern_matrix[guess_idx_global, answer_indicies]
        all_counts = np.bincount(pattern_matrix_row, minlength=243)
        patterns = np.nonzero(all_counts)[0]
        counts = all_counts[patterns]
        probabilities = counts / nanswers
        
        total_patterns_to_explore += len(patterns)
        precalculated_data.append((pattern_matrix_row, patterns, probabilities))

    # --- Main Execution Loop with Smooth Progress Bar ---
    with tqdm(total=total_patterns_to_explore, desc=f"Checking top {NPRUNE} words to depth {depth}") as pbar:
        for i in range(NPRUNE):
            guess_idx_local = top_H_indicies[i]
            # Retrieve the pre-calculated data for this word
            pattern_matrix_row, patterns, probabilities = precalculated_data[i]
            
            weighted_H = 0.0
            for (px, pattern) in zip(probabilities, patterns):
                next_answer_indicies = answer_indicies[pattern_matrix_row == pattern]
                if len(next_answer_indicies) <= 1:
                    next_H = 0.0
                else:
                    next_guess_indicies = np.delete(guess_indicies, guess_idx_local)
                    # Call the fast, compiled function for the deeper search
                    next_H, _ = recursive_search(full_pattern_matrix, next_guess_indicies, next_answer_indicies, depth - 1)
                
                weighted_H += px * next_H
                pbar.update(1) # Update the progress bar for each pattern explored
            
            top_H_values[i] += weighted_H

    best_local_index = top_H_indicies[np.argmax(top_H_values)]
    return (np.max(top_H_values), guess_indicies[best_local_index])


if __name__ == "__main__":
    words = get_valid_words(refetch=False) # Get all possible words
    pattern_matrix = get_pattern_matrix(words, words, recompute=False) # Get full pattern matrix (all guesses x all answers)

    nwords = len(words)
    guess_indicies = np.arange(0, nwords)
    answer_indicies = np.arange(0, nwords)

    start_time = time.time()
    H_top, guess_idx = logged_recursive_search(pattern_matrix, guess_indicies, answer_indicies, NDEPTH)

    print(f'The word {words[guess_idx].upper()} with {H_top:.3f} bits was found in {time.time()-start_time:.1f} sec.')
