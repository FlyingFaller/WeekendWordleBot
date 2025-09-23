WeekendWordleBot is a work-in-progress bot to """optimally""" solve the NYT Wordle game I decided to make for fun one (three) weekened. It only has knowledge of the 14,855 valid guess words (kinda), not the much small solution set (which doesn't exist publically anymore). It's based in Shannon entropy/information theory and there are many explinations out there for similar bots see the excellent videos by 3b1b. 

---

## Background and Motivation

Approaches to developing a Wordle bot can be broadly categorized into two groups: those using **heuristic-based**, real-time computation and those that rely on a **precomputed** search of the entire solution space.

The precomputed approach can achieve provably-optimal performance. For example, the paper *An Exact and Interpretable Solution to Wordle* by Bertsimas and Paskov leveraged exact dynamic programming to fully map the game's decision tree. However, this optimality comes at a significant computational cost. According to the paper,

> "The current form of Wordle — with 6 rounds, 5 letter words, and a guess and solution space of sizes 10,657 and 2,315, respectively — took days to solve via an efficient C++ implementation of the algorithm, parallelized across a 64-core computer."

The primary disadvantage of these solve-once methods is their **brittleness**. If the list of valid guesses or possible answers changes, the entire multi-day computation must be redone. And the game has changed since that paper's publication. The valid guess space has expanded to **14,855 words**, and since the New York Times (NYT) acquired Wordle, the daily answer is chosen by an editor, not from a fixed, publicly known list.

This uncertainty creates several challenges for bot development:
1.  Relying on the pre-NYT answer list creates a significant risk of being unable to solve for newer, out-of-list words.
2.  Treating the entire valid guess list as the starting answer set is safe, but leaves lots of performance on the table.
3.  Without a definitive answer list, a provably-optimal strategy is impossible, diminishing the value of the immense precomputation required.

The Weekend Wordle (WW) project was undertaken to create a high-performance bot for the *current* version of Wordle by favoring a flexible, near-real-time approach over a rigid, precomputed one.

---

## A Hybrid Approach: Heuristic Pruning with Exhaustive Search

To avoid the pitfalls of precomputation, WW uses a real-time, hybrid strategy that combines the speed of heuristics with the accuracy of an exhaustive search.

### The Flaw in Purely Heuristic Scoring

Many heuristic bots score candidate guesses based on metrics like information entropy—prioritizing the word that provides the most information over one or more turns. However, maximizing information round over round is not equivalent to the real objective: finding the solution word in the fewest guesses. In every Wordle game, finding the solution is equivalent to accumulating enough information to reduce the search space to exactly one word: 13.8586 bits for an answer set of 14,855 words. Evaluting candidate guesses by their expected gained entropy over four or more rounds is unhelpful and many of the top guesses will all score the same, maximimal amount of gained information. In otherwords, there are many words which can lead to solutions before running out of guesses. On the other hand, evaluting guesses over fewer rounds only informs how likely the candidate is to lead to a solution at the evaluation depth and, cruitaially, not how quickly the word leads to solutions. 

### The Weekend Wordle Solution: Pruning vs. Scoring

A more direct method for scoring a candidate guess is to perform a deep search of the game tree that follows from it and calculate the average number of moves required to win. A lower average directly corresponds to a better guess. The challenge is that a complete, deep search for every possible guess is computationally infeasible.

The solution implemented in WW is to separate the tasks of **pruning** and **scoring**:
1.  **Pruning with a Heuristic:** In any given turn, a fast, greedy, depth-one entropy calculation is used to identify a small subset of the most promising candidate guesses. This step dramatically reduces the search space.
2.  **Scoring with a Search:** An exhaustive tree search is then performed *only* for this filtered set of promising candidates. The score for each candidate is the resulting average solution depth, the average number of guesses used, for all remaining possible answers.

This hybrid model reduces the search tree by orders of magnitude, making an exhaustive search of the most relevant branches possible in seconds to hours instead of days. While the initial entropy-based filter does not guarantee that the globally optimal move is always considered, the guiding principle is that the best guess will almost certainly rank among the top candidates. By using this filter to cast a sufficiently wide net, it is possible to explore the most critical branches of the game tree and achieve near-optimal performance without the brittleness and massive upfront cost of a full precomputation.

---

## Implementation Details and Optimizations

### Algorithm Overview

The algorithm is comprised of two stages: heuristic pruning and a depth-first recursive scoring. First, the set of all possible guesses is filtered using a fast, single-step entropy calculation. This pruning stage selects a small number of high-information candidate guesses, discarding the majority of likely-bad guesses. Second, each of the candidate guesses are evaluated using a recursive depth-first search to determine which move minimizes the total number of subsequent guesses required to guarantee a win. The candidate with the lowest total score is selected and returned. This is a high-level description of the core algorithm:

```
func recursive_engine(guess_set, answer_set, current_depth):

  // GREEDY ENTROPY CALCULATION
  // Initialize a variable to store entropy values
  entropies = []

  // Loop through each guess in the set of all guesses and calculate its greedy entropy
  for each guess in guess_set:
    // Get a list of all the patterns that are produced for the guess and each possible answer
    patterns = get_patterns_for_guess(guess, answer_set)

    // Count how many occurences of each pattern type there are i.e. how many answers would produce all grays, ect
    pattern_counts = count_unique_pattern_occurences(patterns)

    // Calculale the probability of seeing this pattern when playing this guess
    pattern_probabilities = pattern_counts/size(answer_set)

    // Calculate the expected information gained by playing this guess
    guess_entropy = -sum(pattern_probabilities * log2(pattern_probabilities))

    // Save the entropy along with the guess
    entropies.append((guess_entropy, guess))

  // Get a sorted list of the highest-entropy guess words
  sorted_entropies = sort_high_to_low(entropies)[:, 1]

  // Continue using only the top NPRUNE guesses
  candidate_guesses = sorted_entropies[0:NPRUNE]

  // DEPTH-FIRST SCORING
  // Initialize a variable to store scores for the candidate guesses
  scores = []

  // Loop through each candidate guess and score how many total guesses it takes to solve the puzzle
  for each guess in candidate_guesses:

    // Initialize a variable to store this guess' score
    guess_score = 0

    // Get a list of all the patterns that are produced for the guess and each possible answer; count their occurences
    patterns = get_patterns_for_guess(guess, answer_set)
    pattern_counts = count_unique_pattern_occurences(patterns)

    // Loop through the unique pattern 'buckets' we could fall into and the count of answers that produce the pattern
    for each (pattern, count) in (patterns, pattern_counts):
      // CASE 1: This guess is our answer and no more gusses are needed, add zero
      if pattern == all green:
        guess_score += 0

      // CASE 2: This guess leaves us with one or two possible answers and we can directly compute the number of extra guesses needed
      // If there is one possible answer remaining we need one additional guess
      // If there are two possible answers remaining we could find the solution in one guess half the time or two guesses the otherwise
      // We need three guesses in total between these cases
      elif count < 3:
        guess_score += 2*count - 1

      // CASE 3: There are too many possible answers remaining to directly calculate the number of additional guesses needed
      // Reduce the answer set and call recurse 
      elif current_depth < MAX_DEPTH:
        remaining_answers = get_answers_that_match_pattern(answer_set, guess, pattern)
        score += recusive_engine(guess_set, remaining_answers, current_depth + 1)

      // CASE 4: If we have recursed too deep we can punish this guess using a very large number
      else:
        guess_score += LARGE_NUMBER

    // Store the total score for this guess
    scores.append(guess_score)

  // The best guess will be the one that takes the fewest number of additional guesses to find every possible answer
  // For every answer in the answer set at least one guess must be made, even if that is the winning guess
  // To account for this the number of answers in the answer set must be added to the minimum score before returning
  return min(scores) + size(answer_set)
```

### Optimizations

In order to make the above algorithm be efficient at scale, great effort has been put into optimizing and parallelizing it. Probably the single largest performance boost comes from the way the code is run. Unlike traditional interpreted python, the `recusive_engine()` is JIT compiled directly into machine code using Numba. Further, the evaluation of candidate guesses, following the entropy calculation step, is fully parallelized using Numbas built-in parallel decorator argument. 
