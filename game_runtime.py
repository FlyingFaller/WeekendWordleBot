from helpers import *
from tests import *
from core import *
from engine import *

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
            results = game_obj.compute_next_guess()
            recommendation = results['recommendation']
            sorted_results = results['sorted_results']
            solve_time = results['solve_time']
            event_counts = results['event_counts']

            print(f"\nSearch completed in {solve_time:.5f} seconds.")
            if show_stats:
                print(f"\nStats:")
                print(f"{'Entropy loop skips':.<40}{event_counts[0]}")
                print(f"{'Entropy loop returns':.<40}{event_counts[1]}")
                print(f"{'Solution pattern skips':.<40}{event_counts[2]}")
                print(f"{'Recursions':.<40}{event_counts[3]}")
                print(f"{'Small solution space skips':.<40}{event_counts[4]}")
                print(f"{"Global cache hits":.<40}{event_counts[5]}")
                print(f"{'Local cache hits':.<40}{event_counts[6]}")
                print(f"{'Batches':.<40}{event_counts[7]}")
                print(f"{'Max depth exceeded':.<40}{event_counts[8]}")
        
            print(f"\nThe best {len(sorted_results)} words:")
            for i, (word, score) in enumerate(sorted_results):
                annotation = ""
                if word in set(game_obj.current_answer_set):
                    annotation = "[Possible Answer]"
                    
                print(f"{i+1:>3}. {word.upper():<6} | Exected Guesses: {score:.4f} {annotation}")
            
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