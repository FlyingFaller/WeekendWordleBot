from helpers import *
from core import *
from engine import *

import numpy as np
from tqdm import tqdm
import random
import time
import matplotlib.pyplot as plt

import itertools
import math

def benchmark_algorithm(pattern_matrix, 
                        guesses, 
                        solver_answers, 
                        test_answers, 
                        nprune_global, 
                        nprune_answers,
                        ngames,
                        max_guesses,
                        seed = None,
                        starting_guess=None, 
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
    game_answers = []
    game_stats = np.zeros(ngames, dtype=np.int8)

    # Play all the games
    for game_idx in tqdm(range(ngames), desc="Running simulation"):
        real_answer_idx = random.randint(0, len(test_answers)-1)
        real_answer = test_answers[real_answer_idx]
        game_answers.append(real_answer)
        test_answers = np.delete(test_answers, real_answer_idx)
        game_obj = wordle_game(pattern_matrix, guesses, solver_answers, nprune_global, nprune_answers)

        solve_times = []
        event_counts = []
        nsolves = 0

        for round_number in range(max_guesses):
            if round_number != 0 or starting_guess is None:
                results = game_obj.compute_next_guess()
                recommendation = results['recommendation']
                solve_times.append(results['solve_time'])
                event_counts.append(results['event_counts'])
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

        game_log = {**game_state, 'solve_times': solve_times, 'event_counts': event_counts, 'nsolves': nsolves}
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
    print(f"Starting guess of: {starting_guess}")
    print(f"Average score: {np.average(game_stats[game_stats > 0]):.5f}")
    print(f"Number of failed solves: {len(game_stats[game_stats == -1])}")
    print(f"Seed used: {seed}")

    return {"game_logs": game_logs, "game_stats": game_stats, "game_answers": game_answers}

def check_pattern_uniqueness(
    pattern_matrix: np.ndarray,
    guesses: np.ndarray,
    answers: np.ndarray
) -> None | tuple[int, int, int]:

    # Get the total number of possible answers from the matrix shape.
    num_answers = len(answers)
    ans_to_gss_map = np.where(np.isin(guesses, answers))[0]

    # Calculate the total number of combinations to set up the progress bar.
    total_combinations = math.comb(num_answers, 3)

    # Generate all unique combinations of 3 answer indices from the list of all answers.
    # Wrap the iterator with tqdm for a progress bar.
    combinations_iterator = itertools.combinations(range(num_answers), 3)
    pbar_desc = "Checking Answer Combinations"
    for answer_combo in tqdm(combinations_iterator, total=total_combinations, desc=pbar_desc):
        c1, c2, c3 = answer_combo
        # c1, c2 = answer_combo

        # According to the problem description, for each answer combination,
        # we only need to check the three guesses that correspond to those
        # answers via the ans_to_gss_map.
        guess_idxs_to_check = [
            ans_to_gss_map[c1],
            ans_to_gss_map[c2],
            ans_to_gss_map[c3]
        ]

        success = False
        # Iterate through the three designated guesses to see if any of them work.
        for guess_idx in guess_idxs_to_check:
            # Retrieve the patterns for the current guess against the three answers.
            p1 = pattern_matrix[guess_idx, c1],
            p2 = pattern_matrix[guess_idx, c2],
            p3 = pattern_matrix[guess_idx, c3]

            # A guess "works" if it produces three unique patterns.
            # We can check for uniqueness by converting the list to a set and
            # checking its length.
            if p1 != p2 and p1 != p3 and p2 != p3:
            # if p1 != p2:
                success = True
                # This guess successfully distinguishes the answers, so we can
                # stop checking guesses for this combination and move to the next.
                break

        # If we looped through all three designated guesses and none produced
        # three unique patterns, this is a failing combination.
        if not success:
            # Immediately return the failing combination as per the requirement.
            return answer_combo

    # If the loop completes, it means every combination had at least one
    # distinguishing guess.
    return None
