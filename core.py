import numpy as np
import time
import wordfreq
from helpers import *
from engine import *
from typing import Callable

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

class wordle_game:
    def __init__(self, pattern_matrix: np.ndarray[np.uint8], 
                 guesses: np.ndarray[str], 
                 answers: np.ndarray[str], 
                 nprune_global: int, 
                 nprune_answers: int, 
                 max_depth: int = 6, 
                 cache: Cache|int = 100_000,
                 sort_func: Callable = None): # Still need to get a better automatic estimate for this
    
        if not set(answers).issubset(set(guesses)):
            raise Exception(f'Guess set must completely contain all answers in answer set.')

        nguesses = len(guesses)
        nanswers = len(answers)
        if (nguesses, nanswers) == pattern_matrix.shape:
            # We got a reduced pattern matrix, guess set is larger than answer set
            self.guess_set = guesses
            self.answer_set = answers
        elif (nguesses, nguesses) == pattern_matrix.shape:
            # We got the full pattern matrix, guess and answer set are identical
            self.guess_set = guesses
            self.answer_set = guesses
        else:
            raise Exception(f'Pattern matrix must be shape (guesses, answers) or (guesses, guesses). Got shape {pattern_matrix.shape}, expected {(nguesses, nanswers)} or {(nguesses, nguesses)}.')

        self.ans_idxs = [np.where(np.isin(self.answer_set, answers))[0]] # Creates a mapping between answer_set being used and the answers provided which may be a subset
        self.ans_to_gss_map = np.where(np.isin(self.guess_set, self.answer_set))[0]

        # Set true if answers passed is smaller than answer_set being used
        self.reduced_answer_set = [len(self.answer_set) > len(answers)]

        self.pattern_matrix = pattern_matrix
        self.current_answer_set = self.answer_set[self.ans_idxs[-1]]
        
        self.nprune_global = nprune_global
        self.nprune_answers = nprune_answers
        self.max_depth = max_depth

        if isinstance(cache, int):
            self.cache = Cache(cache)
        elif isinstance(cache, Cache):
            self.cache = cache
        else:
            raise Exception(f'Wrong type for cache. Expected int or Cache and got {type(cache)}.')

        self.guesses_played = []
        self.patterns_seen = []
        
        self.solved = False
        self.failed = False

        if sort_func is None:
            self.sort_func = lambda word: wordfreq.word_frequency(word, 'en')
        else:
            self.sort_func = sort_func

    def _sort_key(self, item):
        word = item[0]
        score = item[1]
        score_key = score
        answer_key = word not in set(self.current_answer_set)
        func_key = -self.sort_func(word)

        return (score_key, answer_key, func_key)
        
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

        current_ans_idxs = self.ans_idxs[-1]
        pattern_matrix_row = self.pattern_matrix[guess_played_idx, current_ans_idxs]
        next_ans_idxs = current_ans_idxs[pattern_matrix_row == pattern_int]

        self.guesses_played.append(guess_played)
        self.patterns_seen.append(pattern_int)
        self.reduced_answer_set.append(self.reduced_answer_set[-1]) # This will be updated in in update_game_state()
        self.ans_idxs.append(next_ans_idxs)

        self.update_game_state()

    def pop_last_guess(self):
        # Now broken due to reduced answer sets being possible
        if len(self.guesses_played) > 0:
            self.guesses_played.pop(-1)
            self.patterns_seen.pop(-1)
            self.reduced_answer_set.pop(-1)
            self.ans_idxs.pop(-1)
            self.update_game_state()

    def regenerate_answer_idxs(self):
        new_ans_idxs = np.arange(0, len(self.answer_set))
        for (guess, pattern_int) in zip(self.guesses_played, self.patterns_seen):
            guess_idx = np.where(self.guess_set == guess.lower())[0]
            pattern_matrix_row = self.pattern_matrix[guess_idx, new_ans_idxs]
            new_ans_idxs = new_ans_idxs[pattern_matrix_row == pattern_int]
        self.ans_idxs[-1] = new_ans_idxs

    def update_game_state(self):
        """Want to check if no possible answer remain or the answer has been played"""
        current_ans_idxs = self.ans_idxs[-1]
        self.current_answer_set = self.answer_set[current_ans_idxs]
        if len(current_ans_idxs) < 1:
            if self.reduced_answer_set[-1]:
                # We need to expand the answer set and try to continue
                self.regenerate_answer_idxs()
                self.reduced_answer_set[-1] = False
                self.update_game_state() # Call this again to check if we still have 0 possible answers
            else:
                self.failed = True
                self.solved = False
        elif len(self.patterns_seen) > 0: 
            if self.patterns_seen[-1] == pattern_to_int(5*[GREEN]):
                self.solved = True
                self.failed = False
    
    def get_game_state(self) -> dict:
        answers_remaining = len(self.ans_idxs[-1])
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
            pattern_matrix_row = self.pattern_matrix[guess_idx, self.ans_idxs[i]]
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

    def compute_next_guess(self, progress_array: np.ndarray[np.float64] = None) -> dict:
        current_ans_idxs = self.ans_idxs[-1]
        if self.solved or len(current_ans_idxs) == 1:
            solution = self.answer_set[current_ans_idxs[0]]
            return {'recommendation': solution, 
                    'sorted_results': [(solution, 1)], 
                    'solve_time': 0.0, 
                    'event_counts': EventCounter()}
        if self.failed:
            return {'recommendation': None, 
                    'sorted_results': [], 
                    'solve_time': 0.0, 
                    'event_counts': EventCounter()}
        
        start_time = time.time()
        recursive_results = recursive_root(self.pattern_matrix, 
                                           self.guess_set, 
                                           current_ans_idxs, 
                                           self.ans_to_gss_map, 
                                           self.nprune_global, 
                                           self.nprune_answers,
                                           self.max_depth,
                                           self.cache,
                                           progress_array)
        words, scores, event_counter = recursive_results
        end_time = time.time()

        results = list(zip(words, scores))
            
        sorted_results = sorted(results, key=self._sort_key)

        recommendation = sorted_results[0][0]

        return {'recommendation': recommendation, 
                'sorted_results': sorted_results,
                'solve_time': end_time - start_time,
                'event_counts': event_counter}