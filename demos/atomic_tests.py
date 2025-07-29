import numpy as np
from numba import njit, prange, get_num_threads, get_thread_id

# Assuming numpy_atomic.py is in the same directory
from atomic_ops import atomic_add

# --- Problem Demonstration: Race Condition ---
# This function demonstrates what happens without atomic operations.
# Multiple threads will try to read and write to `result[0]` simultaneously.
# This leads to a "race condition," where updates are lost.
@njit(parallel=True)
def non_atomic_sum(n):
    """
    Sums numbers from 0 to n-1 in parallel and adds them to a single
    array element. This version is NOT thread-safe.
    """
    result = np.zeros(1, dtype=np.int64)
    for i in prange(n):
        # This is the race condition:
        # 1. Thread A reads result[0] (e.g., value is 100)
        # 2. Thread B reads result[0] (value is still 100)
        # 3. Thread A calculates 100 + i and writes it back.
        # 4. Thread B calculates 100 + j and writes it back, overwriting A's work.
        result[0] += 1
    return result[0]

# --- Solution: Using Custom Atomic Add ---
# This function uses the custom `atomic_add` to ensure thread safety.
@njit(parallel=True)
def safe_atomic_sum(n):
    """
    Sums numbers from 0 to n-1 in parallel using the custom atomic_add.
    This version IS thread-safe.
    """
    result = np.zeros(1, dtype=np.int64)

    # The `atomic_add` function returns the *original* value of the element
    # before the addition. In the original code, this return value was unused.
    # It appears a Numba optimization incorrectly removes the call to
    # `atomic_add` when the return value is not used, treating it as
    # dead code despite its side-effect of modifying the array.
    #
    # To work around this, we capture the return value in a dummy variable.
    # This signals to the compiler that the function call is necessary,
    # ensuring the atomic operation is performed.
    # dummy_returns = np.zeros(get_num_threads(), dtype=np.int64)
    for i in prange(n):
        # By assigning the return value, we ensure the call is not optimized away.
        # id = get_thread_id()
        dummy_val = atomic_add(result, 0, 1)
        # dummy_returns[id] = dummy_val

    return result


def run_demo():
    """
    Runs the demonstration and prints the results.
    """
    # The number of parallel additions to perform.
    iterations = 1_000_000
    print(f"Running demo with {iterations:,} iterations...\n")

    # --- Run the non-atomic version ---
    # We run it a few times to show the inconsistency of the result.
    print("--- Non-Atomic Sum (Incorrect) ---")
    for i in range(3):
        incorrect_result = non_atomic_sum(iterations)
        print(f"Run {i+1}: {incorrect_result:,}")
        if incorrect_result != iterations:
            print("(Result is incorrect due to race conditions!)")
        else:
            print('(The result happens to be correct!)')
    print("-" * 35)


    # --- Run the atomic version ---
    print("\n--- Atomic Sum (Correct) ---")
    correct_result = safe_atomic_sum(iterations)
    correct_result = correct_result[0]
    print(f"Result: {correct_result:,}")
    # print(f"Dummy returns: {dummy_returns}")
    if correct_result == iterations:
        print("(Result is correct!)")
    print("-" * 35)


if __name__ == "__main__":
    # Numba needs to compile the functions on the first run.
    # We call them once to warm up before the actual demo.
    print("Warming up JIT compiler...")
    safe_atomic_sum(1)
    non_atomic_sum(1)
    print("Warm-up complete.\n")

    run_demo()