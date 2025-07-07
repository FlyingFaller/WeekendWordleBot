import numpy as np
from numba import njit, prange
from helpers import *

@njit(cache=False)
def recursive_engine(pattern_matrix: np.ndarray, 
                     nguesses: int,
                     ans_idxs: np.ndarray, 
                     nprune: int,
                     max_depth: int,
                     current_depth: int,
                     global_cache: dict,
                     local_cache: dict,
                     event_counter: np.ndarray) -> float:
    # the question this should answer is, on average, how guesses will the game take to complete from this state
    # N (depth) guesses 
    ### CACHE LOOKUP ###
    # key = (PNR_hash(ans_idxs), depth)
    key = FNV_hash(ans_idxs)
    if key in local_cache:
        event_counter[6] += 1  # Increment the "local cache hit" counter
        return local_cache[key]
    if key in global_cache:
        event_counter[5] += 1  # Increment the "global cache hit" counter
        return global_cache[key]

    nanswers = len(ans_idxs)

    ### COMPILE NEXT GUESSES ###
    pattern_data = []
    log2_nanswers = np.log2(nanswers)
    entropy_vals = np.zeros(nguesses, dtype=np.float64)
    pattern_columns = pattern_matrix[:, ans_idxs] # grab these to avoid repeated slicing
    for i in range(nguesses):
        pattern_row = pattern_columns[i]
        all_pcounts = np.bincount(pattern_row, minlength=243)
        patterns = np.nonzero(all_pcounts)[0] # this works beceause we've forced there to be 243 bins so idx == pattern int

        # If this word generates one pattern we cannot gain any info from it, just leave the entropy at zero and skip the rest
        if len(patterns) < 2:
            event_counter[0] += 1
            pattern_data.append(None)
            continue

        # best possible word: chance of this guess being the answer or worst case will eliminate all others after play
        if all_pcounts[-1] > 1 and len(patterns) >= nanswers:
            event_counter[1] += 1
            score = 1 + (nanswers-1)/nanswers # this guess plus chance another guess will be needed
            # local_cache[key] = score
            global_cache[key] = score
            return score

        pcounts = all_pcounts[patterns]
        pattern_data.append((patterns, pcounts))

        sum_c_log2_c = np.sum(pcounts * np.log2(pcounts))
        entropy_vals[i] = log2_nanswers - (sum_c_log2_c / nanswers)

    ### EVALUTE CANDIDATE GUESSES ###
    candidate_idxs = np.argsort(entropy_vals)[-nprune:]
    candidate_scores = np.zeros(nprune, dtype=np.float64) # probable number of answers left after depth guesses
    for i in range(nprune): # Make one of the top guesses
        score = 0.0
        candidate_idx = candidate_idxs[i]
        pattern_row = pattern_columns[candidate_idx]
        patterns, pcounts = pattern_data[candidate_idx]

        # loop through resulting patterns and figure out how many additional guesses are needed
        for (pattern, count) in zip(patterns, pcounts): 
            if pattern == 242:
                # if this pattern solves the game we don't need additional guesses...
                event_counter[2] += 1
            elif count < 3:
                event_counter[4] += 1
                # if count is 1 only 1 additional guess would be needed so return 1
                # if count is 2 then 50% 1 additional and 50% 2 additional so return 1.5
                # if count is 3 (we have to check this is garunteed) then 33% 1 additional and 66% 2 additional so return 1.66

                score += 2*count - 1 # px * (1 + (count-1)/count) = count/nanswers*(1 + (count-1)/count) = (1/nanswers)*(2*count - 1)
            elif current_depth < max_depth:
                event_counter[3] += 1
                next_ans_idxs = ans_idxs[pattern_row == pattern] # Remove non-matching answers from solution set
                score += count*recursive_engine(pattern_matrix, 
                                                nguesses, 
                                                next_ans_idxs, 
                                                nprune, 
                                                max_depth, 
                                                current_depth+1, 
                                                global_cache, 
                                                local_cache, 
                                                event_counter)
            else:
                event_counter[8] += 1
                score += 1000 # prohibitively large number

        candidate_scores[i] = score/nanswers + 1 # This guess plus the expected future number of guesses
    result = np.min(candidate_scores)
    # local_cache[key] = result
    global_cache[key] = score
    return result
    
@njit(cache=False, parallel=True)
def recursive_root(pattern_matrix: np.ndarray[int], 
                   guesses: np.ndarray[str], 
                   ans_idxs: np.ndarray[int], 
                   ans_to_gss_map: np.ndarray[int], 
                   nprune_global: int, 
                   nprune_answers: int,
                   max_depth: int,
                   global_cache: dict, 
                   local_caches: dict) -> tuple[np.ndarray[str], np.ndarray[float], np.ndarray[int]]:
    """This function should return the best words to play and a bunch of info"""
    # Compute top nprune words greedily
    # Create thread pool for searching down further in the top words
    # Somehow share cache between them
    # Evaluate top words (minimize remaining entropy)
    # Return ordered remaining entropy and words

    # Need to add endgame checks.

    ### SETUP ###
    nanswers = len(ans_idxs)
    nguesses = len(guesses)
    nthreads = len(local_caches)

    global_event_counter = np.zeros(9, dtype=np.int64)

    ### COMPILE NEXT GUESSES ###
    pattern_data = []
    log2_nanswers = np.log2(nanswers)
    entropy_vals = np.zeros(nguesses, dtype=np.float64)
    pattern_columns = pattern_matrix[:, ans_idxs] # grab these to avoid repeated slicing
    for i in range(nguesses):
        pattern_row = pattern_columns[i]
        all_pcounts = np.bincount(pattern_row, minlength=243)
        patterns = np.nonzero(all_pcounts)[0] # this works beceause we've forced there to be 243 bins so idx == pattern int

        if len(patterns) < 2: # If this word generates one pattern we cannot gain any info from it, just leave the entropy at zero and skip the rest
            global_event_counter[0] += 1 # Should be single threaded access
            pattern_data.append(None)
            continue

        pcounts = all_pcounts[patterns]
        pattern_data.append((patterns, pcounts))

        sum_c_log2_c = np.sum(pcounts * np.log2(pcounts))
        entropy_vals[i] = log2_nanswers - (sum_c_log2_c / nanswers)

    ### EVALUTE CANDIDATE GUESSES ###
    ans_gidxs = ans_to_gss_map[ans_idxs]
    ans_entropy_vals = entropy_vals[ans_gidxs]
    ans_candidate_idxs = ans_gidxs[np.argsort(ans_entropy_vals)[-nprune_answers:]]

    gss_candidate_idxs = np.argsort(entropy_vals)[-nprune_global:]

    candidate_idxs = np.union1d(gss_candidate_idxs, ans_candidate_idxs)
    ncandidates = len(candidate_idxs)
    candidate_scores = np.zeros(ncandidates, dtype=np.float64) # probable number of answers left after depth guesses

    ### BATCH PARALLEL SEARCH ###
    for batch_start in range(0, ncandidates, nthreads):
        batch_end = min(batch_start + nthreads, ncandidates)
        global_event_counter[7] += 1 # Is single threaded access
        # local_event_counters = [np.zeros(9, dtype=np.int64) for _ in range(nthreads)] # Should create 16 fresh new event counters every time
        local_event_counters = np.zeros((nthreads, 9), dtype=np.int64)

        # for idx in prange(nprune): # Make one of the top guesses
        for i in prange(batch_end - batch_start):
            idx = i + batch_start
            score = 0.0
            candidate_idx = candidate_idxs[idx]
            pattern_row = pattern_columns[candidate_idx]
            patterns, pcounts = pattern_data[candidate_idx]
            for (pattern, count) in zip(patterns, pcounts): 
                if pattern == 242:
                    # if this pattern solves the game we don't need additional guesses...
                    # score += 0
                    local_event_counters[i][2] += 1
                elif count < 3:
                    local_event_counters[i][4] += 1
                    # if count is 1 only 1 additional guess would be needed so return 1
                    # if count is 2 then 50% 1 additional and 50% 2 additional so return 1.5
                    # if count is 3 (we have to check this is garunteed) then 33% 1 additional and 66% 2 additional so return 1.66
                    # score += count*(1 + 1/count + (count-1)/count*2)
                    # score += count + 1 + (count - 1)*2
                    # score += 3*count - 1 
                    score += 2*count - 1 # px * (1 + (count-1)/count) = count/nanswers*(1 + (count-1)/count) = (1/nanswers)*(2*count - 1)
                else:
                    local_event_counters[i][3] += 1
                    next_ans_idxs = ans_idxs[pattern_row == pattern] # Remove non-matching answers from solution set
                    score += count*recursive_engine(pattern_matrix, 
                                                    nguesses, 
                                                    next_ans_idxs, 
                                                    nprune_global, 
                                                    max_depth, 
                                                    1, 
                                                    global_cache, 
                                                    local_caches[i], 
                                                    local_event_counters[i])
                    
            candidate_scores[idx] = score/nanswers + 1

        for local_cache in local_caches:
            for key, value in local_cache.items():
                if key not in global_cache:
                    global_cache[key] = value
        
        # Should do a single threaded sync of event counters
        for i in range(nthreads):
            global_event_counter += local_event_counters[i]

    return_lidxs = np.argsort(candidate_scores)
    return_gidxs = candidate_idxs[return_lidxs]
    return_scores = candidate_scores[return_lidxs]
    return_words = guesses[return_gidxs]
    return (return_words, return_scores, global_event_counter)