from weekendwordle.backend.helpers import *
from weekendwordle.backend.tests import *
from weekendwordle.backend.core import *
from weekendwordle.backend.engine import *
from weekendwordle.backend.game_runtime import *
from weekendwordle.backend.classifier import *
from weekendwordle.config import *

def run() -> None:
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
    prediction_func = load_classifier(word_features, positive_words=positive_words, all_words=guesses, recompute=True)

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
    
    
if __name__ == "__main__":
    run()