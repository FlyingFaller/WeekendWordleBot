from helpers import *
from tests import *
from core import *
from engine import *
from game_runtime import *

if __name__ == "__main__":
    ### PLAY THE GAME BUT SAFE THIS TIME ###
    guesses = get_words(refetch=False)
    answers = get_words(refetch=False)
    pattern_matrix = get_pattern_matrix(guesses, answers, savefile = "data/full_pattern_matrix.npy")

    original_answers = get_words(url=ORIGINAL_ANSWERS_URL, savefile='data/original_answers.txt', refetch=False, save=True) # These are the ~2400 answer words from the original wordle game (pre NYT)
    freq, idx, word = get_minimum_freq(original_answers)
    all_5letter_words = get_words("data/en_US-large.txt")
    reduced_answers = filter_words_by_suffix(answers, all_5letter_words, suffixes=[('s', 's'), ('d', 'r', 'w', 'n'), 'es', 'ed']) # This gets rid of suffixes that should not occur (plurals/past tense)
    # reduced_answers = filter_words_by_POS(answers, download=True)
    reduced_answers = filter_words_by_occurance(reduced_answers, min_freq=4*freq) # This gets rid of words that are not common enough. I play with the min_freq value.  

    # This just tells you which words it has accidentally elimated from the OG answer set
    reduced_answers_set = set(reduced_answers)
    not_in_set = 0
    print('Exluded words:')
    for answer in original_answers:
        if answer not in reduced_answers_set:
            print(answer)
            not_in_set += 1

    print(f'Reduction from {len(answers)} to {len(reduced_answers)} ({len(reduced_answers)-len(answers)}). {not_in_set} words not in answer set.')

    previous_answers = scrape_words()
    print(previous_answers)
    print(len(previous_answers))
    # This plays the game
    # play_wordle(pattern_matrix, 
    #             guesses, 
    #             reduced_answers, 
    #             nprune_global=50, 
    #             nprune_answers=50, 
    #             starting_guess="SALET", 
    #             show_stats=True, 
    #             discord_printout=True,
    #             max_guesses = 10)