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
from typing import Callable
import plotly.graph_objects as go
import plotly.io as pio

def simulate_game(pattern_matrix: np.ndarray[np.uint8],
                  guesses       : np.ndarray[str],
                  answers       : np.ndarray[str],
                  secret        : str,
                  nprune_global : int,
                  nprune_answers: int,
                  init_guess    : str|None       = None,
                  max_depth     : int            = 6,
                  max_guesses   : int            = 6,
                  cache         : int|Cache|None = None,
                  sort_func     : Callable|None  = None) -> dict: 
    
    game_obj = wordle_game(pattern_matrix,
                           guesses,
                           answers,
                           nprune_global,
                           nprune_answers,
                           max_depth,
                           cache,
                           sort_func)
    # for each round:
    # cache segments
    # cache entries
    # results dict: recommendation, sorted_results, solve_time, event_counter
    # game state dict: answers_remaining, nguesses, guesses_played, patterns_seen, solved, failed
    game_stats = {'secret': secret,
                  'success': False, 
                  'all_eliminated': False,
                  'round_stats': []}
    for round_number in range(max_guesses):
        if round_number != 0 or init_guess is None:
            results = game_obj.compute_next_guess()
            recommendation: str = results['recommendation']
            guess_played = recommendation.lower()
        else:
            results = {'recommendation': None, 
                       'sorted_results': [],
                       'solve_time': 0.0,
                       'event_counts': EventCounter()}
            guess_played = init_guess.lower()

        pattern_seen = get_pattern(guess_played, secret)
        game_obj.make_guess(guess_played, pattern_seen)
        game_state = game_obj.get_game_state()

        round_stats = {**results, 
                       **game_state, 
                       'cache_segments': game_obj.cache.nsegments(),
                       'cache_entries': game_obj.cache.nentries()}
        game_stats['round_stats'].append(round_stats)

        if game_state['solved']:
            game_stats['success'] = True
            break
        if game_state['failed']:
            game_stats['all_eliminated'] = True
            break
    
    # Round ending stats (i.e. cummulative stats)
    game_stats['nguesses']       = game_state['nguesses']
    game_stats['guesses_played'] = game_state['guesses_played']
    game_stats['patterns_seen']  = game_state['patterns_seen']
    game_stats['cache_segments'] = game_obj.cache.nsegments()
    game_stats['cache_entries']  = game_obj.cache.nentries()
    return game_stats

def benchmark(pattern_matrix: np.ndarray[np.uint8],
              guesses       : np.ndarray[str],
              answers       : np.ndarray[str],
              secrets       : np.ndarray[str],
              nprune_global : int,
              nprune_answers: int,
              ngames        : int|None                           = -1,
              init_guess    : str|None                           = None,
              max_depth     : int                                = 6,
              max_guesses   : int                                = 6,
              segment_size  : int|None                           = None,
              reuse_cache   : bool                               = False,
              sort_func     : Callable|None                      = None,
              seed          : int|float|str|bytes|bytearray|None = None,
              plot          : str|None                           = None,
              verbose       : bool                               = True): 
    
    start_time = time.time()
    
    # init plot for live display    
    if plot == 'live':
        fig, axs, bins = _init_plot(max_guesses)

    # seed rng
    random.seed(seed)

    # set ngames to play
    if ngames is None or ngames <= 0:
        ngames = len(secrets)
    else:
        ngames = min(len(secrets), ngames)

    # set cache
    if reuse_cache:
        cache = Cache(segment_size)
    else:
        cache = segment_size

    # where all game_stats will end up
    stats = [] # This is the main stat collection
    plotting_stats =[]

    for game_idx in tqdm(range(ngames), desc="Running simulations"):
        secret_idx = random.randint(0, len(secrets)-1)
        secret = secrets[secret_idx] # select secret
        secrets = np.delete(secrets, secret_idx) # remove secret so it's not picked again
        game_stats = simulate_game(pattern_matrix,
                                   guesses,
                                   answers,
                                   secret,
                                   nprune_global,
                                   nprune_answers,
                                   init_guess,
                                   max_depth,
                                   max_guesses,
                                   cache,
                                   sort_func)
        stats.append(game_stats)
        if game_stats['success']:
            plotting_stats.append(game_stats['nguesses'])
        else:
            plotting_stats.append(-1)

        ### PLOTTING ###
        if plot == 'live' or (plot == 'post' and game_idx==ngames-1):
            if plot == 'post':
                fig, axs, bins = _init_plot(max_guesses)
            _update_plot(axs, bins, plotting_stats, max_guesses, ngames, game_idx)


    end_time = time.time()

    if plot == 'live' or plot == 'post':
        _pause_plot()

    if verbose:
        print(f"Results after {ngames} solves ({end_time - start_time:.3f} sec):")
        print(f"Starting guess of: {init_guess}")
        print(f"Average score: {np.average(list(filter(lambda x: x > 0, plotting_stats))):.5f}")
        print(f"Number of failed solves: {len(list(filter(lambda x: x < 0, plotting_stats)))}")
        print(f"Seed used: {seed}")

    return stats

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

def size_cache(pattern_matrix : np.ndarray[np.uint8], 
               guesses        : np.ndarray[str],
               answers        : np.ndarray[str],
               games_per_prune: int,
               nprune_list    : list[int]|np.ndarray[int],
               init_guess     : str|None                           = None,
               max_depth      : int                                = 6,
               max_guesses    : int                                = 6,
               segment_size   : int|None                           = None,
               sort_func      : Callable|None                      = None,
               seed           : int|float|str|bytes|bytearray|None = None,
               plot           : bool                               = True): 
    # tuples[nprune, nanswers, cache_entries]
    cache_data = []
    for nprune in nprune_list:
        stats = benchmark(pattern_matrix = pattern_matrix,
                          guesses        = guesses,
                          answers        = answers,
                          secrets        = answers,
                          nprune_global  = nprune,
                          nprune_answers = 0,
                          ngames         = games_per_prune,
                          init_guess     = init_guess,
                          max_depth      = max_depth,
                          max_guesses    = max_guesses,
                          segment_size   = segment_size,
                          sort_func      = sort_func,
                          seed           = seed,
                          verbose        = False)
        
        for game_stats in stats:
            old_cache_entries = 0
            for round_stats in game_stats['round_stats']:
                nanswers = round_stats['answers_remaining']
                cache_entries = round_stats['cache_entries']
                old_cache_entries = cache_entries
                new_cache_entries = cache_entries - old_cache_entries
                cache_hits = getattr(round_stats['event_counts'], 'cache_hits')
                cache_data.append((nprune, nanswers, new_cache_entries + cache_hits))

    if plot:
        plot_cache_data(cache_data)
    return cache_data

def _init_plot(max_guesses):
    plt.ion() # Turn on interactive mode
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    # Define histogram bins to keep the x-axis stable.
    bins = np.arange(-1.5, max_guesses + 2.5, 1)
    return fig, axs, bins

def _update_plot(axs, bins, plotting_stats, max_guesses, ngames, game_idx):
    axs[0].clear() # Clear previous histogram
    axs[1].clear()

    axs[0].hist(plotting_stats, bins=bins, rwidth=0.8)

    successful_stats = list(filter(lambda x: x > 0, plotting_stats))

    if len(successful_stats) > 0:
        cum_stats = np.cumsum(successful_stats)
        games_played = np.arange(1, len(successful_stats)+1)
        cum_avg = cum_stats/games_played

        axs[0].axvline(cum_avg[-1], color='red', linestyle='--', linewidth=1, label=f'Average: {cum_avg[-1]:.5f}')
        axs[0].legend() # Display the legend for the vline
        
        axs[1].plot(games_played, cum_avg)

    # Consistently set labels and title
    axs[0].set_title(f'Distribution of Guesses After {game_idx + 1}/{ngames} Games')
    axs[0].set_xlabel('Number of Guesses to Solve')
    axs[0].set_ylabel('Frequency')
    
    axs[1].set_xlabel('Games Played')
    axs[1].set_ylabel('Average Number of Guesses')

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

def _pause_plot():
    plt.ioff() # Turn off interactive mode
    plt.show() # Keep the final plot window open

def plot_cache_data(data_tuples):
    nprunes = np.array([d[0] for d in data_tuples])
    nanswers = np.array([d[1] for d in data_tuples])
    cache_entries = np.array([d[2] for d in data_tuples])

    # Create a scatter plot of the original data points
    scatter_trace = go.Scatter3d(
        x=nprunes, y=nanswers, z=cache_entries,
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Actual Data'
    )

    # Combine traces and define layout
    fig = go.Figure(data=[scatter_trace])
    fig.update_layout(
        title='3D Scatter Plot of Raw Data',
        scene=dict(
            xaxis_title='nprune',
            yaxis_title='nanswers',
            zaxis_title='cache_entries'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show(renderer='browser')