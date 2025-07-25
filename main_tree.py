from helpers import *
from tests import *
from core import *
from engine import *
from game_runtime import *
from queue_engine import *

if __name__ == "__main__":
    ### LOAD STUFF ###
    # guesses = get_words(refetch=False)
    # original_answers = get_words(url=ORIGINAL_ANSWERS_URL, refetch=True, save=False)
    # freq, idx, word = get_minimum_freq(original_answers)
    # all_5letter_words = get_words("data/en_US-large.txt")
    # answers = filter_words_by_suffix(guesses, all_5letter_words, suffixes=['s','es', 'ed'], min_freq=0)
    # answers = filter_words_by_occurance(answers, min_freq=freq)
    # old_answer_set = answers
    # print(f"{len(answers) = }")""
    # guess_set = set(answers)
    # guess_set = set(filter_words_by_length(all_words, 5))
    # for answer in original_answers:
    #     if answer not in guess_set:
    #         print(answer)

    # print(f"Reduction: {len(guesses) - len(answers)}")

    # pattern_matrix = get_pattern_matrix(guesses, answers, savefile="data/temp_pattern_matrix_again.npy")

    ### PLAY THE REAL GAME ###
    # play_wordle(pattern_matrix, guesses, answers, nprune_global=25, nprune_answers=25, starting_guess="SALET", show_stats=True, discord_printout=True)

    ### PLAY THE GAME BUT SAFE THIS TIME ###
    guesses = get_words(refetch=False)
    answers = get_words(refetch=False)
    pattern_matrix = get_pattern_matrix(guesses, answers, savefile = "data/full_pattern_matrix.npy")

    original_answers = get_words(url=ORIGINAL_ANSWERS_URL, refetch=True, save=False)
    freq, idx, word = get_minimum_freq(original_answers)
    all_5letter_words = get_words("data/en_US-large.txt")
    reduced_answers = filter_words_by_suffix(answers, all_5letter_words, suffixes=['s','es', 'ed', 'd'])
    reduced_answers = filter_words_by_occurance(reduced_answers, min_freq=freq*2)

    not_in_set = 0
    print('Exluded words:')
    for answer in original_answers:
        if answer not in set(reduced_answers):
            print(answer)
            not_in_set += 1

    print(f'Reduction from {len(answers)} to {len(reduced_answers)}. {not_in_set} words not in answer set.')

    ans_idxs = np.where(np.isin(answers, reduced_answers))[0]
    play_wordle(pattern_matrix, 
                guesses, 
                reduced_answers, 
                nprune_global=50, 
                nprune_answers=50, 
                starting_guess="SALET", 
                show_stats=True, 
                discord_printout=True,
                max_guesses = 10)



    # final_result = run_solver(pattern_matrix, np.arange(0, len(answers)), len(guesses), 6, 6, 16)

    ### BENCHMARK ###
    # return_dict = benchmark_algorithm(pattern_matrix, guesses, answers, original_answers, 25, 25, 10, 6, 5, "SALET", plot=True)
    # failed_words = []
    # for i, score in enumerate(return_dict['game_stats']):
    #     if score == -1:
    #         failed_words.append(return_dict['game_answers'][i])
    # answer_set = set(answers)
    # for failed_word in failed_words:
    #     if failed_word not in answer_set:
    #         print(f"{failed_word.upper()} was an answer but was not in answer set.")
    #     else:
    #         print(f"{failed_word.upper()} failed despite being in the answer set.")

    # total_event_counts = np.zeros(9, dtype=np.int64)
    # for game_log in return_dict['game_logs']:
    #     for event_count in game_log['event_counts']:
    #         total_event_counts += event_count

    # print(f"\nStats:")
    # print(f"{'Entropy loop skips':.<40}{total_event_counts[0]}")
    # print(f"{'Entropy loop returns':.<40}{total_event_counts[1]}")
    # print(f"{'Solution pattern skips':.<40}{total_event_counts[2]}")
    # print(f"{'Recursions':.<40}{total_event_counts[3]}")
    # print(f"{'Small solution space skips':.<40}{total_event_counts[4]}")
    # print(f"{"Global cache hits":.<40}{total_event_counts[5]}")
    # print(f"{'Local cache hits':.<40}{total_event_counts[6]}")
    # print(f"{'Batches':.<40}{total_event_counts[7]}")
    # print(f"{'Max depth exceeded':.<40}{total_event_counts[8]}")
