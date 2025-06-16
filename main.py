import os
import requests
import numpy as np
from tqdm import tqdm
import random

VALID_WORDS_URL = "https://gist.github.com/dracos/dd0668f281e685bad51479e5acaadb93/raw/6bfa15d263d6d5b63840a8e5b64e04b382fdb079/valid-wordle-words.txt"
VALID_WORDS_FILE = "wordle_words.txt"
PATTERN_MATRIX_FILE = "pattern_matrix.npy"
GREEN = 2
YELLOW = 1
GRAY = 0
NPRUNE = 5
NDEPTH = 2

def precompute_pattern_matrix(guesses:np.ndarray[str], answers: np.ndarray[str]):
    nguesses = len(guesses)
    nanswers = len(answers)
    pattern_matrix = np.zeros((nguesses, nanswers))
    for i in tqdm(range(nguesses)):
        for j in range(nanswers):
            guess = guesses[i]
            answer = answers[j]
            pattern_matrix[i][j] = pattern_to_int(get_pattern(guess, answer))
    return pattern_matrix

def get_pattern_matrix(guesses:np.ndarray[str], answers: np.ndarray[str]) -> np.ndarray[str]:
    if os.path.exists(PATTERN_MATRIX_FILE):
        print("Fetching pattern matrix from file")
        pattern_matrix = np.load(PATTERN_MATRIX_FILE)
    else:
        print("No pattern matrix file found, recomputing")
        pattern_matrix = precompute_pattern_matrix(guesses, answers)
        np.save(PATTERN_MATRIX_FILE, pattern_matrix)
    return pattern_matrix

def get_pattern(guess: str, answer: str) -> list[int]:
    pattern = [GRAY]*5
    letter_count = {}

    # Green pass
    for i in range(5):
        if answer[i] == guess[i]:
            # Set equal to 3 if green
            pattern[i] = GREEN
        else:
            # If not green we keep a running tally of all unique non-green letters
            letter_count[answer[i]] = letter_count.get(answer[i], 0) + 1

    # Yellow pass
    for i in range(5):
        if (pattern[i] != GREEN) and letter_count.get(guess[i], 0) > 0:
            pattern[i] = YELLOW
            letter_count[guess[i]] -= 1

    return pattern

def pattern_to_int(pattern: list[int]) -> int:
    ret_int = 0
    for i in range(5):
        ret_int += (3**i)*pattern[i]
    return ret_int

def int_to_pattern(num: int) -> list[int]:
    pattern = 5*[GRAY]
    for i in range(4, -1, -1):
        pattern[i], num = divmod(num, 3**i)
    return pattern

def get_valid_words() -> np.ndarray[str]:
    if os.path.exists(VALID_WORDS_FILE):
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
    nguesses, nanswers = pattern_matrix.shape
    entropy_vals = np.zeros(nguesses)
    for i in range(nguesses):
        pattern, count = np.unique(pattern_matrix[i], return_counts=True)
        px = count/nanswers
        entropy = np.sum(-px*np.log2(px))
        entropy_vals[i] = entropy
    return entropy_vals

def prune_pattern_matrix(pattern_matrix: np.ndarray, 
                         guess_idx: int, 
                         pattern_int: int,
                         prune_guesses: bool = False) -> np.ndarray:
    a = pattern_matrix[:, pattern_matrix[guess_idx] == pattern_int]
    if prune_guesses:
        guesses_idx = np.where(pattern_matrix[guess_idx] == pattern_int)
        filter_idx = np.setdiff1d(guesses_idx, guess_idx)
        b = a[filter_idx]
    else:
        b = np.delete(b, guess_idx, axis=0)
    return b

def get_word_idx(words_arr: np.ndarray[str], word: str) -> int:
    return np.where(words_arr == word)[0][0]

if __name__ == "__main__":
    words = get_valid_words() # Get all possible words
    pattern_matrix = get_pattern_matrix(words, words) # Get full pattern matrix (all guesses x all answers)
    entropies = compute_entropies(pattern_matrix) # Compute expected entropy for all guesses
    # pruned_entropies_idx = np.argsort(entropies)[-NPRUNE:] # Only want to look further at top NPRUNE guesses
    # pruned_entropies = entropies[pruned_entropies_idx] # So we get their indicies in the word list and their values 

    # for i in tqdm(range(NPRUNE)): # Loop through each candidate starting word
    #     guess_idx = pruned_entropies_idx[i] # Get candidate guess 'global' index
    #     nguesses, nanswers = pattern_matrix.shape 
    #     patterns, counts = np.unique(pattern_matrix[guess_idx], return_counts=True) # Get unique patterns for candidate guess
    #     px = counts/nanswers # Probability of the answer being a member of each pattern
    #     for j, pattern in enumerate(patterns): # Loop through each pattern 'bucket'
    #         pruned_pattern_matrix = prune_pattern_matrix(pattern_matrix, guess_idx, pattern) # Look only at answers that are in this pattern, exclude our first guess from consideration
    #         bucket_entropies = compute_entropies(pruned_pattern_matrix) # Get entropies for remaining guesses expected from remaining answers 
    #         pruned_entropies[i] += np.max(bucket_entropies)*px[j] # Assume we will use the max for that pattern, add its entropy times the probability we will see that bucket to the game total.

    # best_word = words[pruned_entropies_idx[np.argmax(pruned_entropies)]]
    # max_entropy = np.max(pruned_entropies)
    # print(f'The best word at depth 2 is \'{best_word}\' with an entropy of {max_entropy:.3f} bits')

    # Quick real world test
    for i in range(6): # Loop through the rounds
        max_entropy_idx = np.flip(np.argsort(entropies))[0:5]
        max_entropies = entropies[max_entropy_idx]
        candidate_words = words[max_entropy_idx]

        nguesses, nanswers = pattern_matrix.shape
        print(f'{pattern_matrix.shape = }')
        print(f'\nThe current solution space is {nanswers} words')
        print('Best words:')
        for (word, entropy) in zip(candidate_words, max_entropies):
            print(f'{word.upper()}: {entropy:.5f} bits')

        played_word  = input("Word played:  ").lower()
        pattern_word = input("Pattern seen: ").lower()
        guess_idx = get_word_idx(words, played_word)

        pattern = []
        for char in pattern_word:
            match char:
                case "g": pattern.append(GREEN)
                case "y": pattern.append(YELLOW)
                case _:   pattern.append(GRAY)

        if pattern == 5*[GREEN]:
            print(f"Sucess! The word was {played_word.upper()}.")
            break

        pattern_int = pattern_to_int(pattern)
        guesses_idx = np.where(pattern_matrix[guess_idx] == pattern_int)
        filter_idx = np.setdiff1d(guesses_idx, guess_idx)
        words = words[filter_idx]
        # words = np.delete(words[pattern_matrix[guess_idx] == pattern_int])
        pattern_matrix = prune_pattern_matrix(pattern_matrix, guess_idx, pattern_int, True)

        # words = np.delete(words, guess_idx)
        entropies = compute_entropies(pattern_matrix)


    # Algorithm
    # Get top words from pattern matrix
    # For each word:
        # Get unique patterns
        # For each unique pattern
            # Reduce answer space by words in that pattern bucket
            # Reduce guess space by first guess
            # Get top words from reduced pattern matrix
            # Recurse to top loop