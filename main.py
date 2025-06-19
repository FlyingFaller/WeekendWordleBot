import os
import requests
import numpy as np
from tqdm import tqdm
import random
import multiprocessing
import time
from numba import njit, prange
import wordfreq

VALID_WORDS_URL = "https://gist.github.com/dracos/dd0668f281e685bad51479e5acaadb93/raw/6bfa15d263d6d5b63840a8e5b64e04b382fdb079/valid-wordle-words.txt"
VALID_WORDS_FILE = "valid_words.txt"
PATTERN_MATRIX_FILE = "pattern_matrix.npy"
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

def get_pattern_matrix(guesses:np.ndarray[str], answers: np.ndarray[str], savefile=PATTERN_MATRIX_FILE, recompute=False, save=True) -> np.ndarray[str]:
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

def get_words(savefile=VALID_WORDS_FILE, url=VALID_WORDS_URL, refetch=False, save=True) -> np.ndarray[str]:
    """Retrieves the word list from file it it exists otherwise it fetches from url."""
    if not refetch:
        if os.path.exists(savefile):
            print("Fetching all valid words from file")
            with open(savefile, 'r') as f:
                # The last line might be empty, so we filter it out.
                return np.array([line.strip() for line in f if line.strip()], dtype=str)
    print("No word list exists or refetching requested, fetching from the web")
    try:
        response = requests.get(url)
        # Raises an HTTPError if the HTTP request returned an unsuccessful status code.
        response.raise_for_status()
        words_text = response.text
        if save:
            print("Saving word list to file")
            with open(savefile, 'w') as f:
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
    current_depth: int,
    nprune: int
    ) -> tuple[float, int]:
    nanswers = len(answer_indicies)

    if nanswers == 1:
        return 0.0, guess_indicies[np.argmax(full_pattern_matrix[guess_indicies, answer_indicies[0]])] # Get the guess indicie corresponding to the remaining answer
    elif nanswers == 2:
        return 1.0, guess_indicies[np.argmax(full_pattern_matrix[guess_indicies, answer_indicies[0]])] # Get one of the possible answers and return its correspoinding guess indicie

    H_array = fast_entropies(full_pattern_matrix, guess_indicies, answer_indicies)

    H_array = fast_entropies(full_pattern_matrix, guess_indicies, answer_indicies)
    H_nonzero_indicies = np.where(H_array > 0.1)[0] # Find only the location of nonzero values
    H_nonzero_values = H_array[H_nonzero_indicies] # Get the nonzero values
    sorted_H_indicies = np.argsort(H_nonzero_values)[::-1] # indicies local to our sub list
    top_H_indicies = H_nonzero_indicies[sorted_H_indicies[:min(nprune, len(H_nonzero_indicies))]]
    top_H_values = H_array[top_H_indicies]

    if current_depth == 1: # If last search
        return (top_H_values[0], guess_indicies[top_H_indicies[0]]) # Return the max entropy and associated global guess index

    for i in range(0, len(top_H_indicies)):
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

            next_guess_indicies = np.delete(guess_indicies, guess_idx_local) # Get rid of current guess for next round
            next_H, next_guess_idx = recursive_search(full_pattern_matrix, next_guess_indicies, next_answer_indicies, current_depth-1, nprune)
            weighted_H += px*next_H

        top_H_values[i] += weighted_H

    best_local_index = top_H_indicies[np.argmax(top_H_values)]
    return (np.max(top_H_values), guess_indicies[best_local_index])

def logged_recursive_search(
    full_pattern_matrix: np.ndarray,
    guess_indicies: np.ndarray,
    answer_indicies: np.ndarray,
    depth: int,
    nprune: int
    ) -> list[tuple[float, int]]:
    """
    Python-level wrapper to show a smooth progress bar for the top-level search.
    """
    nanswers = len(answer_indicies)
    all_green_pattern = pattern_to_int(5*[GREEN])

    pruned_pattern_matrix = full_pattern_matrix[guess_indicies]
    if nanswers == 1:
        pattern_row = pruned_pattern_matrix[:, answer_indicies[0]].flatten()
        return [(0.0, guess_indicies[np.argmax(pattern_row == all_green_pattern)])] # Get the guess indicie corresponding to the remaining answer
    elif nanswers == 2:
        pattern_row_1 = pruned_pattern_matrix[: answer_indicies[0]].flatten()
        pattern_row_2 = pruned_pattern_matrix[: answer_indicies[1]].flatten()
        word_1 = (1.0, guess_indicies[np.argmax(pattern_row_1 == all_green_pattern)])
        word_2 = (1.0, guess_indicies[np.argmax(pattern_row_2 == all_green_pattern)])
        return [word_1, word_2]

    H_array = fast_entropies(full_pattern_matrix, guess_indicies, answer_indicies)
    H_nonzero_indicies = np.where(H_array > 0.1)[0] # Find only the location of nonzero values
    H_nonzero_values = H_array[H_nonzero_indicies] # Get the nonzero values
    sorted_H_indicies = np.argsort(H_nonzero_values)[::-1] # indicies local to our sub list
    top_H_indicies = H_nonzero_indicies[sorted_H_indicies[:min(nprune, len(H_nonzero_indicies))]]
    top_H_values = H_array[top_H_indicies]

    if depth == 1:
        best_global_indices = guess_indicies[top_H_indicies]
        return list(zip(top_H_values, best_global_indices))
    
    # First, calculate the total number of patterns we will explore to set up the progress bar.
    # We also store the results to avoid redundant calculations.
    total_patterns_to_explore = 0
    precalculated_data = []
    for i in range(len(top_H_indicies)):
        guess_idx_global = guess_indicies[top_H_indicies[i]]
        pattern_matrix_row = full_pattern_matrix[guess_idx_global, answer_indicies]
        all_counts = np.bincount(pattern_matrix_row, minlength=243)
        patterns = np.nonzero(all_counts)[0]
        counts = all_counts[patterns]
        probabilities = counts / nanswers
        
        total_patterns_to_explore += len(patterns)
        precalculated_data.append((pattern_matrix_row, patterns, probabilities))

    # --- Main Execution Loop with Smooth Progress Bar ---
    with tqdm(total=total_patterns_to_explore, desc=f"Checking top {nprune} words to depth {depth}") as pbar:
        for i in range(len(top_H_indicies)):
            guess_idx_local = top_H_indicies[i]
            # Retrieve the pre-calculated data for this word
            pattern_matrix_row, patterns, probabilities = precalculated_data[i]
            
            weighted_H = 0.0
            for (px, pattern) in zip(probabilities, patterns):
                next_answer_indicies = answer_indicies[pattern_matrix_row == pattern]
                next_guess_indicies = np.delete(guess_indicies, guess_idx_local)
                # Call the fast, compiled function for the deeper search
                next_H, _ = recursive_search(full_pattern_matrix, next_guess_indicies, next_answer_indicies, depth - 1, nprune)
                
                weighted_H += px * next_H
                pbar.update(1) # Update the progress bar for each pattern explored
            
            top_H_values[i] += weighted_H

    sorted_final_indices = np.argsort(top_H_values)[::-1]
    best_local_indices = top_H_indicies[sorted_final_indices]
    best_scores = top_H_values[sorted_final_indices]
    best_global_indices = guess_indicies[best_local_indices]
    return list(zip(best_scores, best_global_indices))

def play_wordle(guesses: np.ndarray[str], answers: np.ndarray[str], pattern_matrix: np.ndarray[int], search_depth=3, nprune=50, starting_guess: str = None):
    guess_indicies = np.arange(0, len(guesses))
    answer_indicies = np.arange(0, len(answers))
    for i in range(6):
        print(f'\nRound {i+1}')
        print(f'The current solution space has {len(answer_indicies)} words.')

        if i != 0 or starting_guess is None: # Do a search if not round one or no starting_guess was provided
            depth = min(search_depth, 6-i)

            top_words_results = logged_recursive_search(pattern_matrix, guess_indicies, answer_indicies, depth, nprune)

            if not top_words_results:
                print("Could not determine a best word. The remaining words might be the only option.")
                if len(answer_indicies) == 1:
                    print(f"The only remaining word is {answers[answer_indicies[0]].upper()}")
                return

            print("\nTop 5 recommended words to play:")
            # Take the top 5 from the results
            for j, (score, guess_idx) in enumerate(top_words_results[:5]):
                guess_word: str = guesses[guess_idx]
                # Calculate immediate entropy for display, as it's more intuitive
                if len(answer_indicies) == 1:
                    immediate_entropy = 0.0
                else:
                    immediate_entropy = fast_entropies(pattern_matrix, np.array([guess_idx]), answer_indicies)[0]
                print(f"  {j+1}. {guess_word.upper()} (Score: {score:.3f}, Info: {immediate_entropy:.3f} bits, Words after play: {int(len(answer_indicies)//(2**(immediate_entropy)))})")

            word_played = input("Word played: ").lower()
        else:
            print(f"Word played: {starting_guess.upper()}")
            word_played = starting_guess

        word_idx = np.where(guesses == word_played)[0]

        pattern = input("Pattern seeen: ")
        pattern_list = []
        for c in pattern.upper():
            match c:
                case "G": pattern_list.append(GREEN)
                case "Y": pattern_list.append(YELLOW)
                case _: pattern_list.append(GRAY)
        
        if pattern_list == 5*[GREEN]:
            print(f"Solution found in {i+1} guesses. Word was {guess_word.upper()}.")
            return

        pattern_int = pattern_to_int(pattern_list)
        pattern_matrix_row = pattern_matrix[word_idx, answer_indicies]
        answer_indicies = answer_indicies[pattern_matrix_row == pattern_int]
        guess_indicies = np.delete(guess_indicies, np.isin(guess_indicies, word_idx))
    print('Failed to find a solution.')
    return

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

if __name__ == "__main__":
    guesses = get_words(refetch=False) # Get all possible words
    answers = filter_words_by_occurance(guesses)
    pattern_matrix = get_pattern_matrix(guesses, answers, recompute=True, save=False)

    play_wordle(guesses, answers, pattern_matrix, search_depth=1, nprune=250, starting_guess='salet')