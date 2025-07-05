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
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import plotly.io as pio

VALID_WORDS_URL = "https://gist.github.com/dracos/dd0668f281e685bad51479e5acaadb93/raw/6bfa15d263d6d5b63840a8e5b64e04b382fdb079/valid-wordle-words.txt"
ORIGINAL_ANSWER_URL = "https://gist.github.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b/raw/c46f451920d5cf6326d550fb2d6abb1642717852/wordle-answers-alphabetical.txt"
VALID_WORDS_FILE = "data/valid_guesses.txt"
PATTERN_MATRIX_FILE = "data/pattern_matrix.npy"
GREEN = 2
YELLOW = 1
GRAY = 0

class InvalidWordError(ValueError):
    """Raised when the guessed word is not in the allowed word list."""
    def __init__(self, word: str):
        self.word = word
        super().__init__(f"The word '{word}' is not a valid guess.")

class InvalidPatternError(ValueError):
    """Raised when the feedback pattern is invalid."""
    def __init__(self, pattern: any, reason: str):
        self.pattern = pattern
        self.reason = reason
        super().__init__(f"Invalid pattern '{pattern}': {reason}")

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
def recursive_root(pattern_matrix: np.ndarray[int], 
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

class wordle_game:
    def __init__(self, pattern_matrix, guesses, answers, nprune_global, nprune_answers, batch_size=16):
        self.pattern_matrix = pattern_matrix
        self.guess_set = guesses
        self.answer_set = answers
        self.nprune_global = nprune_global
        self.nprune_answers = nprune_answers
        self.batch_size = batch_size
        self.global_cache = Dict.empty(key_type=types.Tuple((int64, int64)), value_type=float64)
        self.local_caches = [Dict.empty(key_type=types.Tuple((int64, int64)), value_type=float64) for _ in range(batch_size)]
        self.ans_to_gss_map = np.where(np.isin(guesses, answers))[0]
        self.ans_idxs = np.arange(0, len(answers))
        self._old_ans_idxs = []
        self.solved = False
        self.failed = False
        self.guesses_played = []
        self.patterns_seen = []
        self.current_answer_set = self.answer_set[self.ans_idxs]

    def _sort_key(self, item):
        word = item[0]
        entropy = item[1]
        entropy_key = entropy
        answer_key = word not in set(self.current_answer_set)
        frequency_key = -wordfreq.word_frequency(word, 'en')

        return (entropy_key, answer_key, frequency_key)
        
    def validate_guess(self, guess: str) -> tuple[str, int]:
        """Checks if a word is valid. Raises InvalidWordError if not."""
        guess_lower = guess.lower()
        guess_idx = np.where(self.guess_set == guess_lower)[0]
        if len(guess_idx) != 1:
            raise InvalidWordError(guess)
        return (guess_lower, guess_idx)
    
    def validate_pattern(self, pattern: str|list[int]|int) -> int:
        """Checks if a pattern is valid. Raises InvalidPatternError if not."""
        if isinstance(pattern, str):
            pattern_str = pattern.upper()
            if len(pattern_str) != 5:
                raise InvalidPatternError(pattern, "pattern string must be 5 characters.")
            pattern_int = pattern_str_to_int(pattern_str)
        elif isinstance(pattern, int):
            if pattern > 242 or pattern < 0:
                raise InvalidPatternError(pattern, "pattern int must be in the range [0, 242].")
            pattern_int = pattern
        elif isinstance(pattern, list):
            if len(pattern) != 5:
                raise InvalidPatternError(pattern, "pattern list must have 5 elements.")
            pattern_int = pattern_to_int(pattern)
        else:
            raise NotImplementedError(f'Cannot handle patterns of type {type(pattern)}')
        return pattern_int

    def make_guess(self, guess: str, pattern: str|list[int]|int) -> None:
        guess_played, guess_played_idx = self.validate_guess(guess)
        pattern_int = self.validate_pattern(pattern)

        pattern_matrix_row = pattern_matrix[guess_played_idx, self.ans_idxs]
        next_ans_idxs = self.ans_idxs[pattern_matrix_row == pattern_int]        
        self.guesses_played.append(guess_played)
        self.patterns_seen.append(pattern_int)
        self._old_ans_idxs.append(self.ans_idxs)
        self.ans_idxs = next_ans_idxs
        self.update_game_state()

    def pop_last_guess(self):
        self.guesses_played.pop(-1)
        self.patterns_seen.pop(-1)
        self.ans_idxs = self._old_ans_idxs.pop(-1)
        self.update_game_state()

    def update_game_state(self):
        """Want to check if no possible answer remain or the answer has been played"""
        self.current_answer_set = self.answer_set[self.ans_idxs]
        if len(self.ans_idxs) < 1:
            self.failed = True
            self.solved = False
        elif self.patterns_seen[-1] == pattern_to_int(5*[GREEN]):
            self.solved = True
            self.failed = False
    
    def get_game_state(self) -> dict:
        self.update_game_state()
        answers_remaining = len(self.ans_idxs)
        nguesses = len(self.guesses_played)
        return {'answers_remaining': answers_remaining,
                'nguesses': nguesses,
                'guesses_played': self.guesses_played,
                'patterns_seen': self.patterns_seen,
                'solved': self.solved,
                'failed': self.failed}
    
    def get_discord_printout(self, game_number: int)->str:
        ret_str = f"## :robot: WeekendWordleBot #{game_number} :robot:\n"
        for i, (guess, pattern) in enumerate(zip(self.guesses_played, self.patterns_seen)):
            guess_idx = np.where(self.guess_set == guess.lower())[0]
            pattern_matrix_row = self.pattern_matrix[guess_idx, self._old_ans_idxs[i]]
            words_remaining = len(pattern_matrix_row)
            _, counts = np.unique_counts(pattern_matrix_row)
            pxs = counts/words_remaining
            expected_entropy = -np.sum(pxs*np.log2(pxs))
            pattern_out = ""
            for value in int_to_pattern(pattern):
                if value == GRAY:
                    pattern_out += ":black_large_square: "
                elif value == YELLOW:
                    pattern_out += ":yellow_square: "
                elif value == GREEN:
                    pattern_out += ":green_square: "
            expected_words_remaining = int(words_remaining//(2**expected_entropy))
            ret_str += f"{i+1}. ||**`{guess.upper()}`**||: `{words_remaining: >4}` ⟶ `{expected_words_remaining: >4}` Answers ({abs(expected_entropy):.2f} bits)\n"
            ret_str += pattern_out + "\n"
        
        return ret_str

    def compute_next_guess(self, 
                           recommendation_cuttoff=0.75, 
                           entropy_break = 0.0, 
                           max_depth=3,
                           verbose=False) -> dict:
        self.update_game_state()
        if self.solved or len(self.ans_idxs) == 1:
            solution = self.answer_set[self.ans_idxs[0]]
            return {'recommendation': solution, 
                    'sorted_results': [(solution, 0.0)], 
                    'solve_time': 0.0, 
                    'event_counts': np.zeros(8, dtype=np.int32),
                    'depth': 0}
        if self.failed:
            return {'recommendation': None, 
                    'sorted_results': [], 
                    'solve_time': 0.0, 
                    'event_counts': np.zeros(8, dtype=np.int32),
                    'depth': 0}
        
        start_time = time.time()
        t = max(max_depth - len(self.guesses_played), 1)
        # for depth in range(t, 0, -1):
        #     if verbose:
        #         print(f"Searching depth {depth}")
        #     recursive_results = recursive_root(self.pattern_matrix, 
        #                                        self.guess_set, 
        #                                        self.ans_idxs, 
        #                                        self.ans_to_gss_map, 
        #                                        depth, 
        #                                        self.nprune_global, 
        #                                        self.nprune_answers,
        #                                        self.global_cache, 
        #                                        self.local_caches)
        #     words, rem_entropies, event_counter = recursive_results
        #     if rem_entropies[0] > entropy_break:
        #         break
        results_history = []
        for depth in range(1, t+1, 1):
            if verbose:
                print(f"Searching depth {depth}")
            recursive_results = recursive_root(self.pattern_matrix, 
                                               self.guess_set, 
                                               self.ans_idxs, 
                                               self.ans_to_gss_map, 
                                               depth, 
                                               self.nprune_global, 
                                               self.nprune_answers,
                                               self.global_cache, 
                                               self.local_caches)
            words, entropies, event_counter = recursive_results
            if depth == 1:
                if entropies[0] <= entropy_break:
                    break
                else:
                    for word, entropy in zip(words, entropies):
                        if word in set(self.current_answer_set) and entropy <= recommendation_cuttoff:
                            break
            elif depth > 1 and entropies[0] <= entropy_break:
                depth -= 1
                words, entropies, event_counter = results_history[-1]
                break
            results_history.append(recursive_results)


        end_time = time.time()

        results = list(zip(words, entropies))
            
        sorted_results = sorted(results, key=self._sort_key)

        recommendation = sorted_results[0][0]
        if depth < 2:
            for word, entropy in sorted_results:
                if word in set(self.current_answer_set) and entropy <= recommendation_cuttoff:
                    recommendation = word
                    break

        return {'recommendation': recommendation, 
                'sorted_results': sorted_results,
                'solve_time': end_time - start_time,
                'event_counts': event_counter,
                'depth': depth}


def play_wordle(pattern_matrix, 
                guesses, 
                answers, 
                nprune_global = 15, 
                nprune_answers = 15, 
                starting_guess: str= "SALET", 
                batch_size=16, 
                show_stats=True,
                discord_printout=True):
    
    game_obj = wordle_game(pattern_matrix, guesses, answers, nprune_global, nprune_answers, batch_size)
    answers_remaining = len(answers)

    if discord_printout:
        game_number = input("Game number: ")

    # gameplay loop
    for round_number in range(6):
        print(f"\n%%%%%%%%%% ROUND {round_number + 1} %%%%%%%%%%")
        print(f"Answers still remaining: {answers_remaining}\n")

        # Get next-guess recommendations
        if round_number != 0 or starting_guess is None:
            results = game_obj.compute_next_guess(verbose=True)
            recommendation = results['recommendation']
            sorted_results = results['sorted_results']
            solve_time = results['solve_time']
            event_counts = results['event_counts']
            depth = results['depth']

            print(f"\nSearch completed in {solve_time:.5f} seconds.")
            if show_stats:
                print(f"\nStats:")
                print(f"{'Entropy loop skips':.<40}{event_counts[0]}")
                print(f"{'Entropy loop returns':.<40}{event_counts[1]}")
                print(f"{'Low pattern count recursion skips':.<40}{event_counts[2]}")
                print(f"{'Recursions':.<40}{event_counts[3]}")
                print(f"{'Leaf node calculations':.<40}{event_counts[4]}")
                print(f"{"Global cache hits":.<40}{event_counts[5]}")
                print(f"{'Local cache hits':.<40}{event_counts[6]}")
                print(f"{'Batches':.<40}{event_counts[7]}")
        
            print(f"\nThe best {len(sorted_results)} words after depth {depth} search are:")
            for i, (word, entropy) in enumerate(sorted_results):
                annotation = ""
                if word in set(game_obj.current_answer_set):
                    annotation = "[Possible Answer]"
                    
                print(f"{i+1:>3}. {word.upper():<6} | Entropy: {entropy:.4f} {annotation}")
            
            print(f"\nCOMPUTER RECOMMENDATION: {recommendation.upper()}")
            
        # Report guess
        if round_number != 0 or starting_guess is None:
            while(True):
                guess_played = input("\nGuess played: ").lower()
                try:
                    game_obj.validate_guess(guess_played)
                except InvalidWordError as e:
                    print(f"Error: {e}")
                    continue
                break
        else:
            print(f"Guess played: {starting_guess.upper()}")
            guess_played = starting_guess.lower()

        # Report pattern
        while(True):
            pattern_seen = input("Pattern seen: ").upper()
            try:
                game_obj.validate_pattern(pattern_seen)
            except InvalidPatternError as e:
                print(f"Error: {e}")
                continue
            break

        # Make guess
        game_obj.make_guess(guess_played, pattern_seen)

        # Check win conditions:
        game_state = game_obj.get_game_state()

        if game_state['solved']:
            print(f"\nSolution found in {game_state['nguesses']} guesses. Word was {game_state['guesses_played'][-1].upper()}.")
            if discord_printout:
                print("\nDiscord Copy-Paste:\n\n"+game_obj.get_discord_printout(game_number))
            return
        if game_state['failed']:
            print(f"\nAll answers eliminated. No solution found.")
            if discord_printout:
                print("\nDiscord Copy-Paste:\n\n"+game_obj.get_discord_printout(game_number))
            return
        
        answers_remaining = game_state['answers_remaining']

    print('Ran out of guesses!')
    if discord_printout:
        print("\nDiscord Copy-Paste:\n\n"+game_obj.get_discord_printout(game_number))
    return

def generate_algorithm_stats(pattern_matrix, 
                             guesses, 
                             solver_answers, 
                             test_answers, 
                             nprune_global, 
                             nprune_answers,
                             ngames,
                             max_guesses,
                             recommendation_cutoff=0.70, 
                             entropy_break = 0, 
                             max_depth=3,
                             seed = None,
                             starting_guess=None, 
                             batch_size=16, 
                             plot=False,
                             real_time=False) -> dict:
    start_time = time.time()

    def init_plot():
        plt.ion() # Turn on interactive mode
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        # Define histogram bins to keep the x-axis stable.
        bins = np.arange(-1.5, max_guesses + 2.5, 1)
        return fig, axs, bins

    if plot and real_time:
        fig, axs, bins = init_plot()

    if seed is not None:
        random.seed(seed)

    if ngames is None or ngames == -1:
        ngames = len(test_answers)
    else:
        ngames = min(len(test_answers), ngames)

    game_logs = []
    game_stats = np.zeros(ngames, dtype=np.int8)

    # Play all the games
    for game_idx in tqdm(range(ngames), desc="Running simulation"):
        real_answer_idx = random.randint(0, len(test_answers)-1)
        real_answer = test_answers[real_answer_idx]
        test_answers = np.delete(test_answers, real_answer_idx)
        game_obj = wordle_game(pattern_matrix, guesses, solver_answers, nprune_global, nprune_answers, batch_size)

        solve_times = []
        event_counts = []
        depths = []
        nsolves = 0

        for round_number in range(max_guesses):
            if round_number != 0 or starting_guess is None:
                results = game_obj.compute_next_guess(recommendation_cutoff, entropy_break, max_depth, verbose=False)
                recommendation = results['recommendation']
                solve_times.append(results['solve_time'])
                event_counts.append(results['event_counts'])
                depths.append(results['depth'])
                nsolves += 1
                
            # Make guess
            if round_number != 0 or starting_guess is None:
                guess_played = recommendation.lower()
            else:
                guess_played = starting_guess.lower()

            # Get pattern
            pattern_seen = get_pattern(guess_played, real_answer)

            # Make guess
            game_obj.make_guess(guess_played, pattern_seen)

            # Check win conditions:
            game_state = game_obj.get_game_state()

            if game_state['solved']:
                game_stats[game_idx] = game_state['nguesses']
                break
            if game_state['failed']:
                game_stats[game_idx] = -1
                break

        if not game_state['solved'] and not game_state['failed']:
            game_stats[game_idx] = -1

        game_log = {**game_state, 'solve_times': solve_times, 'event_counts': event_counts, 'depths': depths, 'nsolves': nsolves}
        game_logs.append(game_log)
        
        end_time = time.time()

        if plot and (real_time or game_idx==ngames-1):
            if not real_time:
                fig, axs, bins = init_plot()

            axs[0].clear() # Clear previous histogram
            axs[1].clear()
            
            # We only plot non-zero stats (games that have finished)
            valid_stats = game_stats[game_stats != 0]
            
            if len(valid_stats) > 0:
                #  ax.hist(valid_stats, bins=bins, rwidth=0.8, color='dodgerblue', edgecolor='black')
                 axs[0].hist(valid_stats, bins=bins, rwidth=0.8)

            # --- MODIFIED: Calculate and plot average line ---
            successful_stats = valid_stats[valid_stats > 0]

            if len(successful_stats) > 0:
                cum_stats = np.cumsum(successful_stats)
                games_played = np.arange(1, len(successful_stats)+1)
                cum_avg = cum_stats/games_played

                avg_guesses = np.mean(successful_stats)
                axs[0].axvline(avg_guesses, color='red', linestyle='--', linewidth=1, label=f'Average: {avg_guesses:.5f}')
                axs[0].legend() # Display the legend for the vline
                
                axs[1].plot(games_played, cum_avg)

            # Consistently set labels and title
            axs[0].set_title(f'Distribution of Guesses After {game_idx + 1}/{ngames} Games')
            axs[0].set_xlabel('Number of Guesses to Solve')
            axs[0].set_ylabel('Frequency')
            
            axs[1].set_xlabel('Games Played')
            axs[1].set_ylabel('Average Number of Guesses')

            # --- MODIFIED: Set custom x-axis ticks and labels ---
            # Define ticks to show: -1 (for DNF), and 1 up to max_guesses
            ticks_to_show = np.arange(1, max_guesses + 1)
            all_ticks = np.insert(ticks_to_show, 0, -1)
            axs[0].set_xticks(all_ticks)
            
            # Create labels, replacing -1 with 'DNF'
            tick_labels = [str(t) for t in all_ticks]
            tick_labels[0] = 'DNF'
            axs[0].set_xticklabels(tick_labels)
            
            axs[0].grid(axis='y', alpha=0.75)
            axs[1].grid(alpha=0.75)

            plt.pause(0.01) # Pause to update the plot

    if plot:
        plt.ioff() # Turn off interactive mode
        plt.show() # Keep the final plot window open

    print(f"Results after {ngames} solves ({end_time - start_time:.3f} sec):")
    print(f"Recommendation entropy cutoff of: {recommendation_cutoff}, search break of {entropy_break}, and a max depth of: {max_depth}:")
    print(f"Starting guess of: {starting_guess}")
    print(f"Average score: {np.average(game_stats[game_stats > 0]):.5f}")
    print(f"Number of failed solves: {len(game_stats[game_stats == -1])}")
    print(f"Seed used: {seed}")

    return {"game_logs": game_logs, "game_stats": game_stats}

if __name__ == "__main__":
    ### LOAD STUFF ###
    guesses = get_words(refetch=False)
    original_answers = get_words(url=ORIGINAL_ANSWER_URL, refetch=True, save=False)
    freq, idx, word = get_minimum_freq(original_answers)
    # print(f"{word.upper()} has a minimum frequency of {freq}")
    answers = filter_words_by_occurance(guesses, min_freq=freq)
    print(f"Considering {len(answers)} answers with frequencies > {freq}")
    pattern_matrix = get_pattern_matrix(guesses, answers, savefile='9151_pattern_matrix.npy', recompute=False, save=True)

    ### PLAY THE REAL GAME ###
    play_wordle(pattern_matrix, guesses, answers, nprune_global=15, nprune_answers=15, starting_guess="SALET", show_stats=True, discord_printout=True)

    ### SIMULATE GAMES WITH DIFFERENT ALGORITHMS ###
    # generate_algorithm_stats(pattern_matrix, 
    #                          guesses, 
    #                          answers, 
    #                          original_answers, 
    #                          nprune_global=15, 
    #                          nprune_answers=15, 
    #                          ngames=750, 
    #                          max_guesses=6,
    #                          recommendation_cutoff=0.7, 
    #                          entropy_break = 0, 
    #                          max_depth=3,
    #                          seed=5, 
    #                          starting_guess='SALET', 
    #                          plot=True,
    #                          real_time=False)

    ### PLOT BEST GUESSES AT DIFFERENT DEPTHS ###
    # game_obj = wordle_game(pattern_matrix, guesses, answers, 25, 1)
    # result_dict = {}
    # for depth in tqdm(range(1, 6), desc='Searching all depths'):
    #     results = recursive_root(game_obj.pattern_matrix, 
    #                             game_obj.guess_set,
    #                             game_obj.ans_idxs,
    #                             game_obj.ans_to_gss_map,
    #                             depth,
    #                             game_obj.nprune_global,
    #                             game_obj.nprune_answers,
    #                             game_obj.global_cache,
    #                             game_obj.local_caches)
    #     words, entropies, _ = results
    #     for (word, entropy) in zip(words, entropies):
    #         if depth == 1:
    #             result_dict[word] = [entropy]
    #         else:
    #             result_dict[word].append(entropy)

    # pio.renderers.default = 'browser'
    # fig = go.Figure()

    # for word, entropies in result_dict.items():
    #     fig.add_trace(go.Scatter(x=np.arange(1, 6).tolist(), 
    #                              y=entropies,
    #                              mode='lines+markers',
    #                              name=word.upper()))

    # fig.update_layout(
    #     title="Wordle Strategy Entropies",
    #     xaxis_title="Search Depth",
    #     yaxis_title="Entropy",
    #     legend_title="Initial Guesses"
    # )

    # fig.show()