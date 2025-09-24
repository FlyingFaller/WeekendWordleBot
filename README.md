<h3 align="center">
<img width="830" height="417" alt="WW_logo" src="https://github.com/user-attachments/assets/301ac5da-5a92-468a-89be-84db4fdb1185" />
</h3>

## A Near-Optimal, Modern Wordle Bot

WeekendWordle is a Python framework and TUI built to find near-optimal guesses for the modern, New York Times version of Wordle https://www.nytimes.com/games/wordle/index.html. It is designed to run on mid-end hardware and find guesses in as quick as subsecond time. WeekendWordle is built using a combination of hueristic pruning and an exhaustive depth-first tree search with pre-solve answer pruning accomplished by a Logistic Regression PU classifier. While there are many configuration options, the defaults achieve an average solve speed of 3.64 guesses [CHECK]. 

## Run Without a Permanent Installation

You can try the WeekendWordle without installing it using `pipx` or `uvx`. Note: using `pipx` requires installing SpaCy seperately, see [Installing SpaCy](#required-spacy-model).

**Using `uvx` (recommended):**
```bash
uvx --from weekendwordle weekendwordlebot
```
or for the CLI:
```bash
uvx --from weekendwordle weekendwordlebot_cli
```

**Using `pipx`:**
```bash
pipx run --spec weekendwordle weekendwordlebot
```
or for the CLI:
```bash
pipx run --spec weekendwordle weekendwordlebot_cli
```

## Installation

### Standard Installation from PyPI

To install WeekendWordle as a dependency run any of the following:

**Using `uv add` (recommended):**

```bash
uv add weekendwordle
```

**Using `uv pip install`:**

```bash
uv pip install weekendwordle
```

*(Note: This will not update project lock or pyproject.toml files)*

**Using `pip`:**

```bash
pip install weekendwordle
```

### Installation from GitHub

To add the latest development version from GitHub as a dependency:

**Using `uv` (recommended):**

```bash
uv add --git https://github.com/flyingfaller/WeekendWordleBot.git
```

**Using `pip`:**

```bash
pip install git+https://github.com/flyingfaller/WeekendWordleBot.git
```

### Installing For Development

If you want to contribute to the project, you can install it in an editable mode. First, clone the repository:

```bash
git clone https://github.com/flyingfaller/WeekendWordleBot.git
cd WeekendWordleBot
```

Then, install the package in editable mode:

**Using `uv` (recommended):**
```bash
uv pip install -e .
```

**Using `pip`:**
```bash
pip install -e .
```

### Required SpaCy Model

**Important:** If you install WeekendWordle using `pip` or `uv pip` and plan on using the PU classifier (on by default) then you may need to install the SpaCy model seperately:

**Using `uv`:**
```bash
uv add https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
```

**Using `pip`:**
```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
```

## Usage

After installing the WeekendWordle you can access the TUI, CLI, and underlying functionality.

**To run the terminal user interface (TUI) use:**

```bash
weekendwordlebot
```

**To run the command-line interface (CLI) use:**

```bash
weekendwordlebot_cli
```

**Example usage:**
```py
# This is some example code using WeekendWordle tbd
```

## Background and Motivation

Approaches to developing a Wordle bot can be broadly categorized into two groups: those using heuristic-based, real-time computation and those that rely on a precomputed search of the entire solution space.

The precomputed approach can achieve provably-optimal performance. For example, the paper *An Exact and Interpretable Solution to Wordle* by Bertsimas and Paskov leveraged exact dynamic programming to fully map the game's decision tree. However, this optimality comes at a significant computational cost. According to the paper,

> The current form of Wordle — with 6 rounds, 5 letter words, and a guess and solution space of sizes 10,657 and 2,315, respectively — took days to solve via an efficient C++ implementation of the algorithm, parallelized across a 64-core computer.

The primary disadvantage of these solve-once methods is their brittleness. If the list of valid guesses or possible answers changes, the entire multi-day computation must be redone. And the game has changed since that paper's publication. The valid guess space has expanded to 14,855 words, and since the New York Times (NYT) acquired Wordle, the daily answer is chosen by an editor, not from a fixed, publicly known list.

This uncertainty creates several challenges for bot development:
1.  Relying on the pre-NYT answer list creates a significant risk of being unable to solve for newer, out-of-list words.
2.  Treating the entire valid guess list as the starting answer set is safe, but leaves lots of performance on the table.
3.  Without a definitive answer list, a provably-optimal strategy is impossible, diminishing the value of the immense precomputation required.

The Weekend Wordle (WW) project was undertaken to create a high-performance bot for the *current* version of Wordle by favoring a flexible, near-real-time approach over a rigid, precomputed one.

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

    // Calculate the probability of seeing this pattern when playing this guess
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

### Implementation Optimizations

#### JIT Compilation and Parallelism:

* **Numba JIT**: The core `recursive_engine()` function is just-in-time (JIT) compiled directly into efficient machine code using Numba, eliminating Python's interpreter overhead.
* **Parallel Execution**: The root-level evaluation of candidate guesses is parallelized using Numba's `parallel` decorator argument. This workload is managed by a separate `recursive_root()` function, which also provides a more detailed return value than the `recursive_engine()`.

#### Pre-computation and Data Representation:

* **Pattern Matrix**: Rather than calculating patterns on the fly, all possible guess-answer outcomes are pre-computed and stored in a "pattern matrix." Each pattern is represented as an 8-bit unsigned integer (e.g., all-gray is `0x00`, and all-green is `0xF2`).
* **Integer Indexing**: The solver avoids slow string operations entirely. It works exclusively with integer indices that reference the rows (guesses) and columns (answers) of the pattern matrix. The current `answer_set` is maintained as an array of these integer indices.

#### Thread-Safe Memoization Cache:

A custom, high-performance cache stores the results of previously solved subproblems to avoid redundant computation. At each entry to `recursive_root()`, the cache is checked for a solution before any new calculations are performed.

* **Key-Value Storage**: The cache uses the FNV-1a hash of the current `answer_set` indices as its key and the resulting integer score as its value.
* **Collision Handling & Performance**: The cache maps a hash to an array index via modulo of the segment capacity (`index = hash % segment_capacity`). Collisions are resolved using linear probing. This approach is susceptible to 'blocking', the formation of large, contiguous data blocks that slow access, which is mitigated in two ways:
    1.  The FNV-1a hash function was chosen for its excellent key distribution, minimizing block formation.
    2.  The cache automatically expands when a segment reaches 80% capacity. This requires allocating new memory for new cache segments, which introduces a performance cost. However, the cache read/write time is of the order $O(n)$ where $n$ is the number of segments. As long as segments are reasonably sized, this performance loss is not noticeable in practice.
* **Numba-Compatible Thread Safety**: The cache is designed to be fully thread-safe, allowing a single, global instance to be shared across all parallel threads. This is achieved by defining custom atomic operations for the LLVM compiler via Numba. The cache's core functionality relies on atomic add/subtract (`fadd`, `fsub`) and compare-and-swap (`cmpxchg`) machine code instructions.

### Algorithm Optimizations

#### Entropy Calculation:

* **Shared Pattern Data**: The patterns and their corresponding counts (`pattern_counts`) are generated once per guess. This data is then stored and reused for both the entropy calculation and the subsequent candidate evaluation phase.
* **Early Exits & Loop Skips**: The entropy calculation loop for a guess can terminate early in two specific scenarios:
    1.  **Zero Information Guess**: If a guess results in only one unique pattern across all possible answers, it provides no information ($H=0$). The entropy calculation for this guess is immediately skipped.
    2.  **Optimal Guess**: If a guess is a potential answer and also partitions the remaining answer set perfectly (i.e., every answer produces a unique pattern), it is guaranteed to be an optimal move (winning in 1 or 2 turns). Its score is calculated directly, and `recursive_engine()` returns immediately, skipping all remaining entropy calculations and the entire candidate scoring phase.
* **Efficient Entropy Formulation**: The standard Shannon entropy formula, $H(X) = -\sum p_i \log_2(p_i)$, requires a division for each probability term ($p_i = n_i / N$). The implementation uses a mathematically equivalent but more efficient formulation:

    $$H(X) = \log_2(N) - \frac{1}{N} \sum n_i \log_2(n_i)$$
  
    This allows $\log_2(N)$ to be pre-calculated for the entire set of answers, minimizing floating-point operations within the sum. In this case $N = \text{number of answers}$ and $n_i = \text{pattern count}$.

#### Candidate Scoring:

* **Pruning with a Minimum Score**: Instead of calculating the full score for every candidate and then finding the minimum, the solver tracks the best score found so far (initialized to infinity). If a candidate's partially computed score already exceeds this known minimum, its evaluation is immediately aborted, pruning that entire search branch. 
* **Deferred Recursion**: Recursive calls are extremely expensive and are treated as a last resort.
    1.  First, the solver calculates the scores for all simple, non-recursive branches for a candidate guess. Any necessary recursive calls are added to a queue. This increases the chance the partial score will exceed the current minimum, thus avoiding the expensive recursive calls entirely.
    2.  If the candidate is still viable after all non-recursive branches are evaluated, the queued recursive calls are executed one by one, with the partial score checked against the minimum after each completion.

## Intelligent Answer Set Construction

The solver's efficiency is highly dependent on the size and quality of the initial answer set. While Wordle accepts a large vocabulary of guesses (14,855 words), the pool of possible answers is much smaller (2,315 words historically). A smaller, more probable starting answer set directly leads to faster solutions. Several strategies are available in WW to prune the full vocabulary into a minimal, high-quality answer set.

### Filtering Strategies

These methods offer simple, fast ways to reduce the word list based on historical patterns. All filtering methods can be used on their own or in conjunction to varrying degrees of success.

* **Frequency Filtering**: Historically, Wordle answers are common words. This filter removes rare words by checking their usage frequency in a large text corpus, using the `wordfreq` library. Words that fall below a certain frequency threshold are discarded.
* **Part-of-Speech (POS) Filtering**: Simple plurals (ending in 's') and past-tense verbs have historically been avoided as answers. This filter uses the natural language toolkit, `nltk`, to identify POS tags and remove certain tagged words (e.g. plurals `NNS` or past-tense verbs `VBD`).
* **Suffix Filtering**: An alternative to POS tagging, suffix filtering provides a simpler mechanism to filter words based on their endings (e.g., removing words that end in 's', unless preceeded by another 's').

### Spy-EM Classifier for Answer Prediction

While simple filters are effective, they are imprecise, either leaving too many unlikely words or removing valid exceptions. The most powerful method provided is a machine learning classifier trained to predict the probability that any given word is a plausible Wordle answer, effectively learning the preferences of the Wordle authors and editors. 

Training a standard classifier is challenging due to the lack of confirmed "negative" examples (words that will *never* be an answer). There are only positive examples (the past answers and pre-NYT answer list) and a large set of unlabeled words. This is known as a Positive-Unlabeled (PU) learning problem.

#### Word Features:

To solve this, a Logistic Regression model is trained on a wealth of word features including:
* **Word Vectors**: High-dimensional semantic embeddings from `spaCy`'s `en_core_web_lg` model.
* **Word Frequency**: The log-transformed frequency of the word from the `wordfreq` library.
* **POS Tags**: A suite of boolean features based on `spaCy` tags, such as `is_regular_plural`, `is_past_tense`, `is_adjective`, and `is_proper_noun`.
* **Orthographic Features**: Simple word characteristics like `vowel_count` and whether the word `has_double_letter`.

#### Training Process (Spy-EM):

The model is trained using the Spy Expectation-Maximization (Spy-EM) algorithm, an iterative process designed for PU learning which follows these steps:
1.  **Setup**: A small fraction of the positive examples (the "spies") are removed from the training set and mixed into the unlabeled data.
2.  **Initialization**: In the first iteration, known positive examples are given a sample weight of 1.0, and all unlabeled words are given a weight of 0.5.
3.  **M-Step**: A Logistic Regression model is trained on the positive and unlabeled data using the sample weights.
4.  **E-Step**: The model predicts the probability of being an answer for both the unlabeled set and the hidden "spies." The mean probability of the spies ($c$) is calculated.
5.  **Weight Update**: The weights for the unlabeled words are updated using the formula: $\text{new weights} = \text{probability} / c$. The spy probability $c$ acts as a calibration factor, estimating how a true positive appears to the model when it's not labeled.
6.  **Convergence**: The process repeats until the total change in weights between iterations (the L1 Norm) drops below a predefined `convergence_tolerance`.

#### Performance:

The resulting classifier is highly effective, achieving as high as a 99.4% recall on known answers (a 0.6% false negative rate). It produces a much more minimal and accurate answer set than the simpler filtering methods can achieve at a similar level of recall.
