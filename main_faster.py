import os
import requests
import numpy as np
from tqdm import tqdm
import random
import multiprocessing
import time
from numba import njit, prange, float64, int64
from numba.core import types
from numba.typed import Dict
import wordfreq

VALID_WORDS_URL = "https://gist.github.com/dracos/dd0668f281e685bad51479e5acaadb93/raw/6bfa15d263d6d5b63840a8e5b64e04b382fdb079/valid-wordle-words.txt"
ORIGINAL_ANSWER_URL = "https://gist.github.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b/raw/c46f451920d5cf6326d550fb2d6abb1642717852/wordle-answers-alphabetical.txt"
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

@njit(cache=True)
def recursive_engine(pattern_matrix: np.ndarray, 
                     nguesses: int,
                     ans_idxs: np.ndarray, 
                     depth: int, 
                     nprune: int,
                     global_cache: dict,
                     local_cache: dict,
                     event_counter: np.ndarray) -> float:
    # the question this should answer is, on average, how many words remain in the answer set after playing
    # N (depth) guesses 
    ### CACHE LOOKUP ###
    # key = (PNR_hash(ans_idxs), depth)
    key = (FNV_hash(ans_idxs), depth)
    if key in local_cache:
        event_counter[6] += 1  # Increment the "local cache hit" counter
        return local_cache[key]
    if key in global_cache:
        event_counter[5] += 1  # Increment the "global cache hit" counter
        return global_cache[key]

    nanswers = len(ans_idxs)

    ### COMPILE NEXT GUESSES ###
    pattern_data = []
    log2_nanswers = np.log2(nanswers)
    entropy_vals = np.zeros(nguesses, dtype=np.float64)
    pattern_columns = pattern_matrix[:, ans_idxs] # grab these to avoid repeated slicing
    for i in range(nguesses):
        pattern_row = pattern_columns[i]
        all_pcounts = np.bincount(pattern_row, minlength=243)
        patterns = np.nonzero(all_pcounts)[0] # this works beceause we've forced there to be 243 bins so idx == pattern int

        if len(patterns) < 2: # If this word generates one pattern we cannot gain any info from it, just leave the entropy at zero and skip the rest
            event_counter[0] += 1
            pattern_data.append(None)
            continue

        pcounts = all_pcounts[patterns]

        # More aggressively finds cases where the word is good enough to reduce the search space to zero by end of guesses
        if np.max(pcounts) <= depth:
            event_counter[1] += 1
            local_cache[key] = 0.0
            return 0.0

        pattern_data.append((patterns, pcounts))

        sum_c_log2_c = np.sum(pcounts * np.log2(pcounts))
        entropy_vals[i] = log2_nanswers - (sum_c_log2_c / nanswers)

    ### EVALUTE CANDIDATE GUESSES ###
    candidate_idxs = np.argsort(entropy_vals)[-nprune:]
    candidate_scores = np.zeros(nprune, dtype=np.float64) # probable number of answers left after depth guesses
    for i in range(nprune): # Make one of the top guesses
        score = 0.0
        candidate_idx = candidate_idxs[i]
        pattern_row = pattern_columns[candidate_idx]
        patterns, pcounts = pattern_data[candidate_idx]
        for (pattern, count) in zip(patterns, pcounts): 
            if depth == 1 and count > 1: # Final search
                event_counter[4] += 1
                score += count*np.log2(count)
            elif count > 2: # Depth > 1 && Words remaining as a result of this pattern are > 2
                event_counter[3] += 1
                next_ans_idxs = ans_idxs[pattern_row == pattern] # Remove non-matching answers from solution set
                score += count*recursive_engine(pattern_matrix, nguesses, next_ans_idxs, depth-1, nprune, global_cache, local_cache, event_counter)
            else:
                event_counter[2] += 1
                
        candidate_scores[i] = score/nanswers
    result = np.min(candidate_scores)
    local_cache[key] = result
    return result
    
@njit(cache=True, parallel=True)
def get_top_words(pattern_matrix: np.ndarray[int], 
                  guesses: np.ndarray[str], 
                  ans_idxs: np.ndarray[int], 
                  ans_to_gss_map: np.ndarray[int],
                  depth: int, 
                  nprune_global: int, 
                  nprune_answers: int,
                  global_cache: dict, 
                  local_caches: dict) -> tuple[np.ndarray[str], np.ndarray[float], np.ndarray[int]]:
    """This function should return the best words to play and a bunch of info"""
    # Compute top nprune words greedily
    # Create thread pool for searching down further in the top words
    # Somehow share cache between them
    # Evaluate top words (minimize remaining entropy)
    # Return ordered remaining entropy and words

    # Need to add endgame checks.

    ### SETUP ###
    nanswers = len(ans_idxs)
    nguesses = len(guesses)
    nthreads = len(local_caches)

    event_counter = np.zeros(8, dtype=np.int32)

    ### COMPILE NEXT GUESSES ###
    pattern_data = []
    log2_nanswers = np.log2(nanswers)
    entropy_vals = np.zeros(nguesses, dtype=np.float64)
    pattern_columns = pattern_matrix[:, ans_idxs] # grab these to avoid repeated slicing
    for i in range(nguesses):
        pattern_row = pattern_columns[i]
        all_pcounts = np.bincount(pattern_row, minlength=243)
        patterns = np.nonzero(all_pcounts)[0] # this works beceause we've forced there to be 243 bins so idx == pattern int

        if len(patterns) < 2: # If this word generates one pattern we cannot gain any info from it, just leave the entropy at zero and skip the rest
            event_counter[0] += 1
            pattern_data.append(None)
            continue

        pcounts = all_pcounts[patterns]
        pattern_data.append((patterns, pcounts))

        sum_c_log2_c = np.sum(pcounts * np.log2(pcounts))
        entropy_vals[i] = log2_nanswers - (sum_c_log2_c / nanswers)

    ### EVALUTE CANDIDATE GUESSES ###
    ans_gidxs = ans_to_gss_map[ans_idxs]
    ans_entropy_vals = entropy_vals[ans_gidxs]
    ans_candidate_idxs = ans_gidxs[np.argsort(ans_entropy_vals)[-nprune_answers:]]

    gss_candidate_idxs = np.argsort(entropy_vals)[-nprune_global:]

    candidate_idxs = np.union1d(gss_candidate_idxs, ans_candidate_idxs)
    ncandidates = len(candidate_idxs)
    candidate_scores = np.zeros(ncandidates, dtype=np.float64) # probable number of answers left after depth guesses

    ### BATCH PARALLEL SEARCH ###
    for batch_start in range(0, ncandidates, nthreads):
        batch_end = min(batch_start + nthreads, ncandidates)
        event_counter[7] += 1

        # for idx in prange(nprune): # Make one of the top guesses
        for i in prange(batch_end - batch_start):
            idx = i + batch_start
            score = 0.0
            candidate_idx = candidate_idxs[idx]
            pattern_row = pattern_columns[candidate_idx]
            patterns, pcounts = pattern_data[candidate_idx]
            for (pattern, count) in zip(patterns, pcounts): 
                if depth == 1 and count > 1: # Final search
                    event_counter[4] += 1
                    score += count*np.log2(count)
                elif count > 2: # Depth > 1 && Words remaining as a result of this pattern are > 2
                    event_counter[3] += 1
                    next_ans_idxs = ans_idxs[pattern_row == pattern] # Remove non-matching answers from solution set
                    score += count*recursive_engine(pattern_matrix, nguesses, next_ans_idxs, depth-1, nprune_global, global_cache, local_caches[i], event_counter)
                else:
                    event_counter[2] += 1
                    
            candidate_scores[idx] = score/nanswers

        for local_cache in local_caches:
            for key, value in local_cache.items():
                if key not in global_cache:
                    global_cache[key] = value

    return_lidxs = np.argsort(candidate_scores)
    return_gidxs = candidate_idxs[return_lidxs]
    return_scores = candidate_scores[return_lidxs]
    return_words = guesses[return_gidxs]
    return (return_words, return_scores, event_counter)

def play_wordle(pattern_matrix, guesses, answers, nprune_global, nprune_answers, starting_guess: str=None, batch_size=16, show_stats=False):
    # Cache construction
    global_cache = Dict.empty(
        key_type=types.Tuple((int64, int64)),
        value_type=float64
    )
    local_caches = [Dict.empty(key_type=types.Tuple((int64, int64)), value_type=float64) for _ in range(batch_size)]

    # indices stuff
    ans_to_gss_map = np.where(np.isin(guesses, answers))[0] # indices of answers in the guess list
    ans_idxs = np.arange(0, len(answers))

    # Gameplay loop
    for round_number in range(6):
        print(f"\n%%%%%%%%%% ROUND {round_number + 1} %%%%%%%%%%")

        ans_remaining = len(ans_idxs)
        print(f"Answers still remaining: {ans_remaining}\n")

        # Break early if we know the answer
        if ans_remaining == 1:
            print(f"\nSolution found in {round_number+1} guesses. Word is {answers[ans_idxs[0]].upper()}.")
            return
            
        # Normal guess generation or skip if starting_guess is set
        if round_number != 0 or starting_guess is None:

            # Search multiple depths till best words are found
            start_time = time.time()
            # for depth in range(1, max(2, 6 - round_number)):
            for depth in range(max(1, min(5 - round_number, 3)), 0, -1):
                print(f"Searching depth {depth}")
                words, rem_entropies, event_counter = get_top_words(pattern_matrix, 
                                                                    guesses, 
                                                                    ans_idxs, 
                                                                    ans_to_gss_map, 
                                                                    depth, 
                                                                    nprune_global, 
                                                                    nprune_answers,
                                                                    global_cache, 
                                                                    local_caches)
                if rem_entropies[0] > 0:
                    break

            end_time = time.time()
            
            # sort results
            results = list(zip(words, rem_entropies))
            current_answer_set = set(answers[ans_idxs])

            def sort_key(item):
                word = item[0]
                entropy = item[1]
                entropy_key = entropy
                answer_key = word not in current_answer_set
                frequency_key = -wordfreq.word_frequency(word, 'en')

                return (entropy_key, answer_key, frequency_key)
            
            sorted_results = sorted(results, key=sort_key)
            
            # show stats
            print(f"\nSearch completed in {end_time - start_time:.5f} seconds.")
            if show_stats:
                print(f"\nStats:")
                print(f"{'Entropy loop skips':.<40}{event_counter[0]}")
                print(f"{'Entropy loop returns':.<40}{event_counter[1]}")
                print(f"{'Low pattern count recursion skips':.<40}{event_counter[2]}")
                print(f"{'Recursions':.<40}{event_counter[3]}")
                print(f"{'Leaf node calculations':.<40}{event_counter[4]}")
                print(f"{"Global cache hits":.<40}{event_counter[5]}")
                print(f"{'Local cache hits':.<40}{event_counter[6]}")
                print(f"{'Batches':.<40}{event_counter[7]}")

            # show results
            print(f"\nThe best {len(words)} words after depth {depth} search are:")
            for i, (word, entropy) in enumerate(sorted_results):
                annotation = ""
                if word in current_answer_set:
                    annotation = "[Possible Answer]"
                    
                print(f"{i+1:>3}. {word.upper():<6} | Entropy: {entropy:.4f} {annotation}")

            # recommend a word
            recommendation = sorted_results[0][0]
            if depth < 2:
                for word, entropy in sorted_results:
                    if word in current_answer_set and entropy <= 0.5:
                        recommendation = word
                        break
            
            print(f"\nCOMPUTER RECOMMENDATION: {recommendation.upper()}")

            guess_played = input("\nGuess played: ").lower()
        else:
            print(f"Guess played: {starting_guess.upper()}")
            guess_played = starting_guess.lower()

        # Verify guess and pattern inputs
        guess_played_idx = np.where(guesses == guess_played)[0]
        while len(guess_played_idx) != 1:
            print("\nGuess not found in guess list, please re-enter.")
            guess_played = input("Guess played: ").lower()
            guess_played_idx = np.where(guesses == guess_played)[0]

        pattern_matrix_row = pattern_matrix[guess_played_idx, ans_idxs]
        while True:
            pattern = input("Pattern seen: ").upper()
            if len(pattern) != 5:
                print("\nInvalid pattern entry, please re-enter.")
                continue
                
            pattern_int = pattern_str_to_int(pattern)
            next_ans_idxs = ans_idxs[pattern_matrix_row == pattern_int]

            if len(next_ans_idxs) < 1:
                print("\nNo matching solutions found, please re-enter.")
                continue
            else:
                ans_idxs = next_ans_idxs
                break
        
        # check we didn't just solve it on the last guess
        if pattern_int == pattern_to_int(5*[GREEN]):
            print(f"\nSolution found in {round_number+1} guesses. Word is {guess_played.upper()}.")
            return

if __name__ == "__main__":

    guesses = get_words(refetch=False)
    original_answers = get_words(url=ORIGINAL_ANSWER_URL, refetch=True, save=False)
    freq, idx, word = get_minimum_freq(original_answers)
    # print(f"{word.upper()} has a minimum frequency of {freq}")
    answers = filter_words_by_occurance(guesses, min_freq=freq)
    print(f"Considering {len(answers)} answers with frequencies > {freq}")
    pattern_matrix = get_pattern_matrix(guesses, answers, savefile='filtered_pattern_matrix.npy', recompute=True, save=False)

    play_wordle(pattern_matrix, guesses, answers, nprune_global=30, nprune_answers=1, starting_guess=None, show_stats=True)