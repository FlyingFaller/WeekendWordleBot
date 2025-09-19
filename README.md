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

Many heuristic bots score candidate guesses based on metrics like information entropy—prioritizing the word that provides the most information over one or more turns. However, maximizing information round over round is not equivalent to the real objective: finding the solution word in the fewest guesses. In every Wordle game you have to, by definition, accumulate enough information to reduce the search space to a single answer. Evaluting candidate guesses by their expected gained entropy over four or more rounds is unhelpful and many of the top guesses will all score the same, maximimal amount of gained information. In otherwords, there are many words which can lead to solutions before running out of guesses. On the other hand, evaluting guesses over fewer rounds only informs how likely the candidate is to lead to a solution at the evaluation depth and, cruitaially, not how quickly the word leads to solutions. 

### The Weekend Wordle Solution: Pruning vs. Scoring

A more direct method for scoring a candidate guess is to perform a deep search of the game tree that follows from it and calculate the average number of moves required to win. A lower average directly corresponds to a better guess. The challenge is that a complete, deep search for every possible guess is computationally infeasible.

The solution implemented in WW is to separate the tasks of **pruning** and **scoring**:
1.  **Pruning with a Heuristic:** In any given turn, a fast, greedy, depth-one entropy calculation is used to identify a small subset of the most promising candidate guesses. This step dramatically reduces the search space.
2.  **Scoring with a Search:** An exhaustive tree search is then performed *only* for this filtered set of promising candidates. The score for each candidate is the resulting average solution depth, the average number of guesses used, for all remaining possible answers.

This hybrid model reduces the search tree by orders of magnitude, making an exhaustive search of the most relevant branches possible in seconds to hours instead of days. While the initial entropy-based filter does not guarantee that the globally optimal move is always considered, the guiding principle is that the best guess will almost certainly rank among the top candidates. By using this filter to cast a sufficiently wide net, it is possible to explore the most critical branches of the game tree and achieve near-optimal performance without the brittleness and massive upfront cost of a full precomputation.
