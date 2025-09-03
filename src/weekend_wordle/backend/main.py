from weekend_wordle.backend.helpers import *
from weekend_wordle.backend.tests import *
from weekend_wordle.backend.core import *
from weekend_wordle.backend.engine import *
from weekend_wordle.backend.game_runtime import *
from weekend_wordle.backend.classifier import *

if __name__ == "__main__":
    import json
    ### PLAY THE GAME BUT SAFE THIS TIME ###
    # guesses = get_words(refetch=False)
    # answers = get_words(refetch=False)
    # pattern_matrix = get_pattern_matrix(guesses, answers, savefile = "data/full_pattern_matrix.npy")

    # original_answers = get_words(url=ORIGINAL_ANSWERS_URL, savefile='data/original_answers.txt', refetch=False, save=True) # These are the ~2400 answer words from the original wordle game (pre NYT)
    # freq, idx, word = get_minimum_freq(original_answers)
    # all_5letter_words = get_words("data/en_US-large.txt")
    # reduced_answers = filter_words_by_suffix(answers, all_5letter_words, suffixes=[('s', 's'), ('d', 'r', 'w', 'n'), 'es', 'ed']) # This gets rid of suffixes that should not occur (plurals/past tense)
    # # reduced_answers = filter_words_by_POS(answers, download=True)
    # reduced_answers = filter_words_by_occurance(reduced_answers, min_freq=4*freq) # This gets rid of words that are not common enough. I play with the min_freq value.  

    # # This just tells you which words it has accidentally elimated from the OG answer set
    # reduced_answers_set = set(reduced_answers)
    # not_in_set = 0
    # print('Exluded words:')
    # for answer in original_answers:
    #     if answer not in reduced_answers_set:
    #         print(answer)
    #         not_in_set += 1

    # print(f'Reduction from {len(answers)} to {len(reduced_answers)} ({len(reduced_answers)-len(answers)}). {not_in_set} words not in answer set.')

    # previous_answers = scrape_words()
    # print(previous_answers)
    # print(len(previous_answers))
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
    
    # Mandatory loads
    guesses = get_words(refetch=False)
    answers = get_words(refetch=False)
    pattern_matrix = get_pattern_matrix(guesses, answers, savefile = "data/full_pattern_matrix.npy")

    # Classifier Features
    ## Positive word examples ##
    past_answers = scrape_words(refetch=True, save=False)
    original_answers = get_words(refetch=False, savefile=ORIGINAL_ANSWERS_FILE)
    positive_words = np.union1d(past_answers, original_answers)

    word_features = get_word_features(all_words=guesses)
    prediction_func = load_classifier(word_features, positive_words=positive_words, all_words=guesses, recompute=False)

    # Optional filter
    reduced_answers = filter_words_by_probability(prediction_func, guesses)
    play_wordle(
        pattern_matrix, 
        guesses, 
        reduced_answers, 
        nprune_global=10, 
        nprune_answers=10, 
        starting_guess="TALES", 
        show_stats=True, 
        discord_printout=True,
        max_guesses = 10,
        sort_func=prediction_func)
    
    # benchmark(pattern_matrix,
    #           guesses,
    #           reduced_answers,
    #           reduced_answers,
    #           nprune_global=25,
    #           nprune_answers=25,
    #           ngames=10,
    #           init_guess="TALES",
    #           max_depth=8,
    #           max_guesses=6,
    #           segment_size=1_000_000,
    #           reuse_cache=True,
    #           sort_func = prediction_func,
    #           seed=None,
    #           plot='post')

    # results = size_cache(pattern_matrix,
    #                      guesses,
    #                      answers,
    #                      games_per_prune = 1,
    #                      nprune_list = [5, 10, 30, 40],
    #                      init_guess = None,
    #                      max_depth = 7,
    #                      max_guesses = 1,
    #                      sort_func = prediction_func,
    #                      plot=False)
    
    # with open('cache_data.json', 'w') as f:
    #     json.dump(results,f)