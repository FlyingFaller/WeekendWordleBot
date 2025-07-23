import numpy as np
from numba import njit, prange, get_num_threads, get_thread_id
from helpers import *
from cache import *
NEVENT_TYPES = 16

# Event counter map
# 0 : Cache hit
# 1 : Entropy loop skip
# 2 : Entropy loop early exit
# 3 : Winning pattern
# 4 : Low answer count pattern
# 5 : Recursion queued
# 6 : Depth limit reached while trying to recurse
# 7 : Min score exceeded during simple calculation
# 8 : Recursion called
# 9 : Min score exceeded during recusion
# 10: New min score found after recusions
# 11: New min score found without recusions
# 12: Leaf node calculations completed
# 13: 
# 14: 
# 15: 

@njit(cache=False)
def recursive_engine(pattern_matrix: np.ndarray, 
                     nguesses: int,
                     ans_idxs: np.ndarray, 
                     nprune: int,
                     max_depth: int,
                     current_depth: int,
                     cache: Cache,
                     event_counter: np.ndarray) -> float:
    # the question this should answer is, on average, how guesses will the game take to complete from this state
    # N (depth) guesses 
    ### CACHE LOOKUP ###
    key = FNV_hash(ans_idxs)
    if key == EMPTY_KEY: key = np.uint64(1)

    value, success = cache.get(key)
    if success:
        event_counter[0] += 1 # Cache hit
        return value

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
            event_counter[1] += 1
            pattern_data.append(None)
            continue

        # best possible word: chance of this guess being the answer or worst case will eliminate all others after play
        if all_pcounts[-1] > 1 and len(patterns) >= nanswers:
            event_counter[2] += 1
            score = 1 + (nanswers-1)/nanswers # this guess plus chance another guess will be needed
            cache.set(key, score)
            return score

        pcounts = all_pcounts[patterns]
        pattern_data.append((patterns, pcounts))

        sum_c_log2_c = np.sum(pcounts * np.log2(pcounts))
        entropy_vals[i] = log2_nanswers - (sum_c_log2_c / nanswers)

    ### EVALUTE CANDIDATE GUESSES ###
    candidate_idxs = np.argsort(entropy_vals)[-nprune:]
    min_partial_score = np.inf
    for i in range(nprune):
        partial_score = 0.0
        children = []

        candidate_idx = candidate_idxs[i]
        pattern_row = pattern_columns[candidate_idx]
        patterns, pcounts = pattern_data[candidate_idx]
        for (pattern, count) in zip(patterns, pcounts):
            if pattern == 242:
                # if this pattern solves the game we don't need additional guesses...
                event_counter[3] += 1
            elif count < 3:
                event_counter[4] += 1
                partial_score += 2*count - 1 # px * (1 + (count-1)/count) = count/nanswers*(1 + (count-1)/count) = (1/nanswers)*(2*count - 1)
            elif current_depth < max_depth:
                event_counter[5] += 1
                child_ans_idxs = ans_idxs[pattern_row == pattern] # Remove non-matching answers from solution set
                children.append((child_ans_idxs, count))
            else:
                event_counter[6] += 1
                partial_score += 1000 # prohibitively large number

            if partial_score >= min_partial_score:
                # this candidate is bad enough we can discontinue the entire search line
                event_counter[7] += 1
                break

        # if partial_score < min_partial_score:
        else: # at the end of our fast loop the partial score was still the best we've seen
            if children: # we must recurse to get a better number
                for child in children:
                    event_counter[8] += 1
                    child_ans_idxs, count = child
                    partial_score += count*recursive_engine(pattern_matrix, 
                                                            nguesses, 
                                                            child_ans_idxs, 
                                                            nprune, 
                                                            max_depth, 
                                                            current_depth+1, 
                                                            cache, 
                                                            event_counter)
                    if partial_score >= min_partial_score:
                        # recursion has increased the score enough that the word is no longer the best
                        event_counter[9] += 1
                        break
                else: # not executed if break is taken
                    event_counter[10] += 1
                    min_partial_score = partial_score

            else: # fully resolved candidate AND its the best we've seen
                event_counter[11] += 1
                min_partial_score = partial_score

    event_counter[12] += 1
    min_score = min_partial_score/nanswers + 1
    cache.set(key, min_score)
    return min_score

@njit(cache=False, parallel=True)
def recursive_root(pattern_matrix: np.ndarray[int], 
                   guesses: np.ndarray[str], 
                   ans_idxs: np.ndarray[int], 
                   ans_to_gss_map: np.ndarray[int], 
                   nprune_global: int, 
                   nprune_answers: int,
                   max_depth: int,
                   cache: Cache) -> tuple[np.ndarray[str], np.ndarray[float], np.ndarray[int]]:
    """This function should return the best words to play and a bunch of info"""

    ### SETUP ###
    nanswers = len(ans_idxs)
    nguesses = len(guesses)
    nthreads = get_num_threads()
    global_event_counter = np.zeros(NEVENT_TYPES, dtype=np.int64)

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
            global_event_counter[1] += 1 # Should be single threaded access
            pattern_data.append(None)
            continue

        pcounts = all_pcounts[patterns]
        pattern_data.append((patterns, pcounts))

        sum_c_log2_c = np.sum(pcounts * np.log2(pcounts))
        entropy_vals[i] = log2_nanswers - (sum_c_log2_c / nanswers)

    ### EVALUTE CANDIDATE GUESSES ###
    gss_candidate_idxs = np.argsort(entropy_vals)[-nprune_global:]

    if nprune_answers > 0:
        ans_gidxs = ans_to_gss_map[ans_idxs]
        ans_entropy_vals = entropy_vals[ans_gidxs]
        ans_candidate_idxs = ans_gidxs[np.argsort(ans_entropy_vals)[-nprune_answers:]]
        candidate_idxs = np.union1d(gss_candidate_idxs, ans_candidate_idxs)
    else:
        candidate_idxs = gss_candidate_idxs

    ncandidates = len(candidate_idxs)
    candidate_scores = np.zeros(ncandidates, dtype=np.float64) # probable number of answers left after depth guesses

    ### PARALLEL SEARCH ###
    local_event_counters = np.zeros((nthreads, NEVENT_TYPES), dtype=np.int64)
    for i in prange(0, ncandidates):
        partial_score = 0.0
        local_event_counter = local_event_counters[get_thread_id()]
        candidate_idx = candidate_idxs[i]
        pattern_row = pattern_columns[candidate_idx]
        patterns, pcounts = pattern_data[candidate_idx]
        for (pattern, count) in zip(patterns, pcounts): 
            if pattern == 242:
                # if this pattern solves the game we don't need additional guesses...
                local_event_counter[3] += 1
            elif count < 3:
                local_event_counter[4] += 1
                partial_score += 2*count - 1 # px * (1 + (count-1)/count) = count/nanswers*(1 + (count-1)/count) = (1/nanswers)*(2*count - 1)
            else:
                local_event_counter[8] += 1
                child_ans_idxs = ans_idxs[pattern_row == pattern] # Remove non-matching answers from solution set
                partial_score += count*recursive_engine(pattern_matrix, 
                                                        nguesses, 
                                                        child_ans_idxs, 
                                                        nprune_global, 
                                                        max_depth, 
                                                        1, 
                                                        cache, 
                                                        local_event_counter)
                
        candidate_scores[i] = partial_score/nanswers + 1

    for local_event_counter in local_event_counters:
        global_event_counter += local_event_counter

    return_lidxs = np.argsort(candidate_scores)
    return_gidxs = candidate_idxs[return_lidxs]
    return_scores = candidate_scores[return_lidxs]
    return_words = guesses[return_gidxs]
    return (return_words, return_scores, global_event_counter)