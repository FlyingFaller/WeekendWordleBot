from helpers import *
from tests import *
from core import *
from engine import *
from game_runtime import *

if __name__ == "__main__":
    ### LOAD STUFF ###
    guesses = get_words(refetch=False)
    original_answers = get_words(url=ORIGINAL_ANSWERS_URL, refetch=True, save=False)
    freq, idx, word = get_minimum_freq(original_answers)
    all_words = get_words("data/en_US-large.txt")
    answers = filter_words_by_suffix(guesses, all_words, suffixes=['s','es', 'ed'], min_freq=0)
    answers = filter_words_by_occurance(answers, min_freq=freq)
    # print(f"{len(answers) = }")
    # guess_set = set(answers)
    # guess_set = set(filter_words_by_length(all_words, 5))
    # for answer in original_answers:
    #     if answer not in guess_set:
    #         print(answer)

    # print(f"Reduction: {len(guesses) - len(answers)}")
    pattern_matrix = get_pattern_matrix(guesses, answers, savefile="data/temp_pattern_matrix_again.npy")

    ### PLAY THE REAL GAME ###
    play_wordle(pattern_matrix, guesses, answers, nprune_global=25, nprune_answers=25, starting_guess=None, show_stats=True, discord_printout=True)
