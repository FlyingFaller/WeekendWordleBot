from functools import reduce
import threading
import shlex
from tqdm import tqdm
import numpy as np
from datetime import date
from dataclasses import dataclass, field
from typing import Any
from pyfiglet import Figlet
import pyperclip
from ..config_loader import load_config, translate_for_backend
from .messenger import UIMessenger, ConsoleMessenger
from .helpers import (get_words, 
                      get_pattern_matrix, 
                      scrape_words, 
                      filter_words_by_suffix,
                      filter_words_by_POS,
                      filter_words_by_frequency,
                      int_to_pattern,
                      solver_progress_bar) 
from .classifier import filter_words_by_probability, load_classifier, get_word_features
from .core import WordleGame, InvalidWordError, InvalidPatternError
from .cache import Cache
from ..gui.game.state import GameState
from ..config import GRAY, YELLOW, GREEN, NTHREADS, EVENTS, FIGLET_FONT

@dataclass
class CliContext:
    """Stores the runtime state of the CLI application."""
    game_state: GameState = GameState.INPUT_WORD
    last_guess: str | None = None
    game_number: int = 0
    initial_suggestion: str = ""
    results_history: list[dict[str, Any]] = field(default_factory=list)

def run_cli():
    """Initializes the application and runs the main command loop."""
    config = translate_for_backend(load_config(no_gui=True))
    messenger = ConsoleMessenger()

    f = Figlet(font=FIGLET_FONT)
    messenger.log(f.renderText("Weekend Wordle"))

    messenger.log('Loading gamed data...')
    game_obj = load_backend_data(config, messenger)
    
    initial_suggestion_cfg = config['game_settings']['initial_suggestion']
    game_number = config['game_settings'].get('game_number', None)
    if not game_number:
        game_number = _get_default_game_number()
    
    context = CliContext(
        game_number=game_number,
        initial_suggestion=initial_suggestion_cfg[0]
    )

    command_handlers = {
        '/g': _handle_guess,   '/guess'  : _handle_guess,
        '/p': _handle_pattern, '/pattern': _handle_pattern,
        '/u': _handle_undo,    '/undo'   : _handle_undo,
        '/s': _handle_set,     '/set'    : _handle_set,
        '/d': _handle_discord, '/discord': _handle_discord,
        '/h': _handle_help,    '/help'   : _handle_help,
    }
    
    state_prompts = {
        GameState.INPUT_WORD   : "\n(WORD) >> ",
        GameState.INPUT_PATTERN: "\n(PATTERN) >> ",
        GameState.IDLE         : "\n(COMPUTING) >> ",
        GameState.GAME_OVER    : "\n(GAME OVER) >> "
    }

    messenger.log("Welcome to WeekendWordle CLI!")
    messenger.log("Use '/help' for a list of commands.")
    _print_board(game_obj, messenger)
    messenger.log(f"Initial suggestion is '{context.initial_suggestion.upper()}'.")

    while True:
        try:
            prompt = state_prompts.get(context.game_state, ">> ")
            raw_input = input(prompt)
            if not raw_input.strip():
                continue
            
            parts = shlex.split(raw_input)
            command = parts[0].lower()
            args = parts[1:]

            if command in ('/q', '/quit'):
                messenger.log("\nExiting game. Goodbye!\n")
                break
            
            handler = command_handlers.get(command)
            if handler:
                handler(args, game_obj, messenger, context)
            else:
                messenger.log(f"Unknown command: '{command}'. Type /help for options.")

        except (KeyboardInterrupt, EOFError):
            messenger.log("\nExiting game. Goodbye!\n")
            break
        except Exception as e:
            messenger.log(f"An unexpected error occurred: {e}")

def _handle_guess(args, game_obj: WordleGame, messenger: UIMessenger, context: CliContext):
    """Handles the /guess command."""
    if context.game_state != GameState.INPUT_WORD:
        messenger.log("Error: A word has already been guessed. Please enter a pattern with /p or /u to undo the guess.")
        return

    # Try initial computation
    if not args:
        if not game_obj.guesses_played and not context.results_history:
             messenger.log("Computing initial best guesses...")
             results = _compute_next_guess(game_obj, messenger, context)
             if results:
                context.results_history.append(results)
             return
        else:
            messenger.log("Error: Guess command requires a word mid-game.")
            return
    
    word = args[0].lower()
    if word == 'rec':
        last_results = context.results_history[-1] if context.results_history else None
        rec = last_results.get('recommendation') if last_results else context.initial_suggestion
        if not rec:
            messenger.log("Error: No recommendation available to guess.")
            return
        word_to_guess = rec
    else:
        word_to_guess = word

    try:
        game_obj.validate_guess(word_to_guess)
        context.last_guess = word_to_guess
        context.game_state = GameState.INPUT_PATTERN
        messenger.log(f"Guess set to '{context.last_guess.upper()}'. Enter pattern with /p PATTERN.")
    except InvalidWordError as e:
        messenger.log(f"Error: {e}")
        context.last_guess = None

def _handle_pattern(args, game_obj: WordleGame, messenger: UIMessenger, context: CliContext):
    """Handles the /pattern command."""
    if context.game_state != GameState.INPUT_PATTERN:
        messenger.log("Error: No word has been guessed. Please enter a guess with /g first.")
        return
    if not context.last_guess:
        messenger.log("Error: Internal state error. No last guess recorded. Use /u to undo.")
        return
    if not args:
        messenger.log("Error: Pattern command requires a pattern (e.g., /p --g-y).")
        return
    
    try:
        game_obj.validate_pattern(args[0].lower())
        
        game_obj.make_guess(context.last_guess, args[0].lower())
        context.last_guess = None
        context.game_state = GameState.IDLE

        _print_board(game_obj, messenger)
        game_state = game_obj.get_game_state()

        if game_state['solved'] or game_state['failed'] or len(game_obj.guesses_played) >= 6:
            context.game_state = GameState.GAME_OVER
            if game_state['solved']:
                messenger.log(f"Solved! The word was {game_state['guesses_played'][-1].upper()}.")
                # messenger.log(game_obj.get_discord_printout(context.game_number))
            elif game_state['failed']:
                messenger.log("Failed: All words eliminated.")
            else:
                messenger.log("Failed: Ran out of guesses.")
            return

        messenger.log("Computing next optimal guess...")
        results = _compute_next_guess(game_obj, messenger, context)
        if results:
            context.results_history.append(results)
        context.game_state = GameState.INPUT_WORD

    except (InvalidPatternError, ValueError) as e:
        messenger.log(f"Error: {e}")


def _handle_undo(args, game_obj: WordleGame, messenger: UIMessenger, context: CliContext):
    """Handles the /undo command."""
    if not args:
        nundo = 1
    else:
        try:
            nundo = int(args[0])
        except ValueError as e:
            messenger.log(f"Error: {e}")
            return
    
    for _ in range(nundo):
        # Scenario 1: User has typed a guess, is now in INPUT_PATTERN mode, and wants to undo the guess.
        if context.game_state == GameState.INPUT_PATTERN:
            messenger.log(f"Undoing guess '{context.last_guess.upper()}'.\n")
            context.last_guess = None
            context.game_state = GameState.INPUT_WORD
            
            _print_board(game_obj, messenger)

            # messenger.log("State reverted to word input. Displaying previous recommendations:")
            last_results = context.results_history[-1] if context.results_history else None
            _print_results(last_results, game_obj, messenger, context)

        # Scenario 2: User has submitted a full guess + pattern, is now in INPUT_WORD mode, and wants to undo the whole turn.
        elif context.game_state == GameState.INPUT_WORD or context.game_state == GameState.GAME_OVER:
            if not game_obj.guesses_played:
                messenger.log("Error: Nothing to undo.")
                return

            undone_guess = game_obj.guesses_played[-1]
            game_obj.pop_last_guess()
            
            # Pop the results that were generated from the move we are now undoing.
            if context.results_history:
                context.results_history.pop()

            context.last_guess = undone_guess
            context.game_state = GameState.INPUT_PATTERN
            
            messenger.log("Undoing pattern.\n")
            messenger.log(f"State reverted to pattern input for word '{undone_guess.upper()}'.")
        
        else: # Should cover IDLE, GAME_OVER states
            messenger.log("Error: Cannot undo at this time.")

def _handle_set(args, game_obj: WordleGame, messenger: UIMessenger, context: CliContext):
    """Handles the /set command."""
    if len(args) > 0 and args[0].lower() == 'list':
        messenger.log("\nCurrent Settings:")
        padding = 16
        messenger.log(f"{'nprune_global':<{padding}}: {game_obj.nprune_global}")
        messenger.log(f"{'nprune_answers':<{padding}}: {game_obj.nprune_answers}")
        messenger.log(f"{'max_depth':<{padding}}: {game_obj.max_depth}")
        messenger.log(f"{'game_number':<{padding}}: {context.game_number}")
        return

    if len(args) == 0 or len(args) % 2 != 0:
        messenger.log("Usage: /set <list | KEY VALUE [KEY2 VALUE2 ...]>")
        return

    # Iterate over the arguments in key-value pairs
    for i in range(0, len(args), 2):
        setting, value = args[i].lower(), args[i+1]
        
        if hasattr(game_obj, setting):
            try:
                # Special handling for max_depth which can be None
                if setting == 'max_depth' and value.lower() == 'none':
                    setattr(game_obj, setting, None)
                    messenger.log(f"Set '{setting}' to 'None'.")
                    continue
                
                current_val = getattr(game_obj, setting)
                # Attempt to cast to the same type as the existing value
                new_val = type(current_val)(value) if current_val is not None else int(value)
                setattr(game_obj, setting, new_val)
                messenger.log(f"Set '{setting}' to '{new_val}'.")
            except (ValueError, TypeError):
                messenger.log(f"Invalid value '{value}' for setting '{setting}'. Skipping.")
        elif setting == 'game_number':
            try:
                context.game_number = int(value)
                messenger.log(f"Set 'game_number' to '{context.game_number}'.")
            except ValueError:
                messenger.log("Invalid value for game_number, must be an integer. Skipping.")
        else:
            messenger.log(f"Unknown setting: '{setting}'. Skipping.")


def _handle_discord(args, game_obj: WordleGame, messenger: UIMessenger, context: CliContext):
    """Handles the /discord command."""
    text = game_obj.get_discord_printout(context.game_number)
    messenger.log("\n" + text)
    pyperclip.copy(text)
    messenger.log("Output copied to clipboard.")

def _handle_help(args, game_obj: WordleGame, messenger: UIMessenger, context: CliContext):
    """Handles the /help command."""
    messenger.log("\nAvailable commands:")
    messenger.log("  /guess WORD,  /g WORD    - Make a word guess (e.g., /g slate).")
    messenger.log("  /guess rec,   /g rec     - Guess the computer's recommendation.")
    messenger.log("  /guess,       /g         - (First turn only) Compute best initial words.")
    messenger.log("  /pattern PAT, /p PAT     - Enter pattern for last guess (e.g., /p --g-y).")
    messenger.log("  /undo,        /u         - Undo the last guess or pattern entry.")
    messenger.log("  /discord,     /d         - Show Discord-formatted summary.")
    messenger.log("  /set SET VAL, /s SET VAL - Change a setting (e.g., /s nprune_global 50).")
    messenger.log("  /quit,        /q         - Exit the application.")
    messenger.log("  /help,        /h         - Show this help message.")

def _print_results(results: dict | None, game_obj: WordleGame, messenger: UIMessenger, context: CliContext):
    """Prints the results from a pre-computed dictionary without running computation."""
    if not results:
        if context and not game_obj.guesses_played:
             messenger.log(f"Initial suggestion is '{context.initial_suggestion.upper()}'.")
        else:
            messenger.log("No results to display.")
        return


    sorted_results = results.get('sorted_results')
    answers_remaining = game_obj.get_game_state()['answers_remaining']

    if not sorted_results:
        messenger.log("No recommendations found.")
        return

    max_len = max(len(str(sorted_results[-1][1])), 5) if sorted_results else 5
    messenger.log(f"\nThe best {len(sorted_results)} recommendations for {answers_remaining} possible answer(s):")
    messenger.log(f"\n###. WORDS | Avrg.  | {f'Total': ^{max_len}} | Notes |")
    messenger.log(f"-----------|--------|-{'-'*max_len}-|-------|")
    
    for i, (word, score) in enumerate(sorted_results):
        annotation = "*" if word in game_obj.current_answer_set else " "
        avg_score = score / answers_remaining if answers_remaining > 0 else 0
        messenger.log(f"{i+1:>3}. {word.upper():<5} | {avg_score:.4f} | {score: ^{max_len}} |   {annotation}   |")
    
    recommendation = results.get('recommendation')
    if recommendation:
        messenger.log(f"\n=> Top Recommendation: {recommendation.upper()}")

def _get_default_game_number() -> int:
    """Calculate the default game number based on the Wordle epoch."""
    epoch = date(2021, 6, 19)
    today = date.today()
    return (today - epoch).days

def _print_stats(event_counts, cache: Cache, messenger: UIMessenger):
    padding = 45

    messenger.log(f"\nStats:")
    for name, description in EVENTS:
        value = getattr(event_counts, name)
        messenger.log(f"{description:.<{padding}}{value:,}")

    messenger.log(f"{'Cache entries':.<{padding}}{cache.nentries():,}")
    messenger.log(f"{'Cache segments':.<{padding}}{cache.nsegments():,}")

def _compute_next_guess(game_obj: WordleGame, messenger: UIMessenger, context: CliContext) -> dict | None:
    """Runs the backend computation and prints the results."""
    # A simple progress array is required by the backend function.
    progress_array = np.zeros(NTHREADS+1, dtype=np.float64) 
    progress_format = '{l_bar}{bar}| {n:.1f}/{total_fmt} [{elapsed}<{remaining}]'
    pbar = tqdm(total=0, desc="Evaluating candidates", bar_format=progress_format)
    stop_event = threading.Event() # Create the event
    monitor = threading.Thread(target=solver_progress_bar, 
                                args=(progress_array, pbar, stop_event)) # Pass it here
    monitor.start()
    try:
        results = game_obj.compute_next_guess(progress_array)
    finally:
        stop_event.set() # Signal the monitor thread to exit its loop
        monitor.join()   # Wait for the monitor thread to terminate cleanly

    solve_time = results.get('solve_time', 0.0)
    messenger.log(f"Search completed in {solve_time:.4f} seconds.")

    if results.get('event_counts'):
        _print_stats(results['event_counts'], game_obj.cache, messenger)

    _print_results(results, game_obj, messenger, context)
    
    return results

def _print_board(game_obj: WordleGame, messenger: UIMessenger):
    """Prints the emoji representation of the game board."""
    COLOR_MAP = {
        GRAY  : "🟦",
        YELLOW: "🟨",
        GREEN : "🟩"
    }
    EMPTY_ROW = "🟦" * 5

    lines = []
    # Print completed rows
    for p_int in game_obj.patterns_seen:
        pattern_list = int_to_pattern(p_int)
        emoji_line = "".join([COLOR_MAP.get(c, "?") for c in pattern_list])
        lines.append(emoji_line)
    
    # Print empty rows
    num_guesses_made = len(lines)
    for i in range(6 - num_guesses_made):
        if i == 0:
            lines.append(">"+EMPTY_ROW)
        else:
            lines.append(EMPTY_ROW)
        
    messenger.log(f"\n- Round {num_guesses_made+1} -\n" + "\n".join(lines) + "\n-----------\n")

def load_backend_data(config: dict, messenger: UIMessenger) -> WordleGame:
    guesses = get_words(**config['guesses'], messenger=messenger)
    answers = get_words(**config['answers'], messenger=messenger)
    pattern_matrix = get_pattern_matrix(guesses, answers, **config['pattern_matrix'], messenger=messenger)

    ### Load Classifier ###
    if config['classifier']:
        clss_cfg = config['classifier']
        positive_word_tuple = ()
        for positive_word_source in clss_cfg['positive_words']:
            if positive_word_source['type'] == "GetWordsWidget":
                positive_word_tuple += (get_words(**positive_word_source['contents'], messenger=messenger),)
            elif positive_word_source['type'] == "ScrapeWordsWidget":
                positive_word_tuple += (scrape_words(**positive_word_source['contents'], messenger=messenger),)

        positive_words = reduce(np.union1d, positive_word_tuple)
        word_features = get_word_features(all_words=guesses, **clss_cfg['word_features'], messenger=messenger)
        classifier_sort_func = load_classifier(word_features, 
                                               positive_words=positive_words, 
                                               all_words=guesses, 
                                               **clss_cfg['model'],
                                               messenger=messenger)
        
    ### Filter Answers ###
    filtered_answers = answers.copy()
    for filter_source in config['filters']:
        filter_type = filter_source['type']
        filter_contents = filter_source['contents']
        if filter_type == "FilterSuffixWidget":
            filter_words = get_words(**filter_contents['get_words'], messenger=messenger)
            filtered_answers = filter_words_by_suffix(filtered_answers,
                                                    filter_words,
                                                    filter_contents['suffixes'],
                                                    messenger=messenger)
        elif filter_type == "FilterFrequencyWidget":
            filtered_answers = filter_words_by_frequency(filtered_answers, 
                                                        filter_contents['min_freq'],
                                                        messenger=messenger)
        elif filter_type == "FilterPOSWidget":
            filtered_answers = filter_words_by_POS(filtered_answers, 
                                                **filter_contents,
                                                messenger=messenger)
        elif filter_type == "FilterProbabilityWidget" and config['classifier']:
            filtered_answers = filter_words_by_probability(classifier_sort_func, 
                                                        filtered_answers,
                                                        filter_contents['threshold'],
                                                        messenger=messenger)
            
    game_settings: dict = config['game_settings']
    game_obj = WordleGame(pattern_matrix,
                          guesses,
                          filtered_answers,
                          nprune_global = game_settings['nprune_global'],
                          nprune_answers = game_settings['nprune_answers'],
                          max_depth = game_settings['max_depth'],
                          sort_func = classifier_sort_func if config['sort'] == 'Classifier' and config['classifier'] else None)
    return game_obj

if __name__ == "__main__":
    run_cli()