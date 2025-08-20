from helpers import *
from tests import *
from core import *
from engine import *
from typing import Callable

def play_wordle(pattern_matrix: np.ndarray[np.uint8], 
                guesses: np.ndarray[str], 
                answers: np.ndarray[str], 
                nprune_global: int = 15, 
                nprune_answers: int = 15, 
                starting_guess: str= "SALET", 
                show_stats: bool =True,
                discord_printout: bool =True,
                max_guesses: int = 6,
                sort_func: Callable = None):
    
    game_obj = wordle_game(pattern_matrix, guesses, answers, nprune_global, nprune_answers, sort_func = sort_func)
    answers_remaining = len(answers)

    if discord_printout:
        game_number = input("Game number: ")

    # gameplay loop
    for round_number in range(max_guesses):
        print(f"\n%%%%%%%%%% ROUND {round_number + 1} %%%%%%%%%%")
        print(f"Answers still remaining: {answers_remaining}\n")

        # Get next-guess recommendations
        if round_number != 0 or starting_guess is None:
            nthreads = get_num_threads()
            progress_array = np.zeros(nthreads + 1, dtype=np.float64)
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

            recommendation = results['recommendation']
            sorted_results = results['sorted_results']
            solve_time = results['solve_time']
            event_counts = results['event_counts']

            print(f"\nSearch completed in {solve_time:.5f} seconds.")
            if show_stats:
                cache = game_obj.cache
                print_stats(event_counts, cache)

            max_len = len(str(sorted_results[-1][1]))
            print(f"\nThe best {len(sorted_results)} words are...")
            print(f"\n###. WORDS | Avrg.  | Total")
            for i, (word, score) in enumerate(sorted_results):
                annotation = ""
                if word in set(game_obj.current_answer_set):
                    annotation = "[Possible Answer]"
                    
                print(f"{i+1:>3}. {word.upper():<5} | {score/answers_remaining:.4f} | {score: <{max_len}} {annotation}")
            
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