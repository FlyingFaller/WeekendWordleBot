import numpy as np
from numba import njit, prange
from numba.experimental import jitclass
from numba.types import int64, float64, boolean
import time

# Import the atomics, including our new atomic_cas
from numpy_atomic import atomic_cas

# A sentinel value to indicate an empty slot in the keys array.
EMPTY_KEY = -1

# Define the specification for our jitclass
spec = [
    ('keys', int64[:]),
    ('values', float64[:]),
    ('capacity', int64),
]

@jitclass(spec)
class ThreadSafeDict:
    def __init__(self, capacity):
        self.capacity = capacity
        # Initialize the internal arrays
        self.keys = np.full(capacity, EMPTY_KEY, dtype=np.int64)
        self.values = np.zeros(capacity, dtype=np.float64)

    def _get_start_index(self, key: np.int64) -> np.int64:
        """A simple hash function."""
        return key % self.capacity

    def get(self, key: np.int64) -> tuple[float, bool]:
        """Gets a value from the cache. Returns the value and a success boolean."""
        start_index = self._get_start_index(key)
        for i in range(self.capacity):
            index = (start_index + i) % self.capacity
            # If we find the key, return its value
            if self.keys[index] == key:
                return self.values[index], True
            # If we hit an empty slot, the key isn't in the cache
            if self.keys[index] == EMPTY_KEY:
                return 0.0, False
        # The key was not found (and the cache might be full)
        return 0.0, False

    def set(self, key: np.int64, value: float) -> bool:
        """Sets a key-value pair in the cache atomically."""
        start_index = self._get_start_index(key)
        for i in range(self.capacity):
            index = (start_index + i) % self.capacity

            # If the key is already here, we don't need to do anything.
            # (A more complex implementation might update the value).
            if self.keys[index] == key:
                return True

            # If we find an empty slot, try to claim it with compare-and-swap.
            if self.keys[index] == EMPTY_KEY:
                # Atomically swap EMPTY_KEY with our key.
                original_value = atomic_cas(self.keys, index, EMPTY_KEY, key)
                
                # If the original value was EMPTY_KEY, we succeeded!
                if original_value == EMPTY_KEY:
                    # We have claimed the slot. Now we can safely write the value.
                    self.values[index] = value
                    return True# Exit successfully
                
                # If we failed, another thread claimed the slot just before us.
                # We check if that thread was for our key. If so, our work is done.
                if self.keys[index] == key:
                    return True
                
        # If the loop completes, the cache is full.        
        return False
        

# --- Demo of the jitclass Cache ---

@njit
def expensive_work(key):
    """A dummy function that simulates an expensive computation."""
    # The result is just some function of the key.
    return float(key) * np.random.randint(1, 11)

@njit(parallel=True)
def run_parallel_simulation(cache, items_to_process):
    """
    Simulates many parallel threads trying to access and compute values.
    The code is much cleaner now, just calling methods on the cache object.
    """
    for i in prange(items_to_process.shape[0]):
        key = items_to_process[i]
        
        # Check if the key exists
        value, success = cache.get(key)
        
        # If it doesn't exist, compute it and set it
        if not success:
            new_value = expensive_work(key)
            cache.set(key, new_value)

def main():
    """Sets up and runs the jitclass cache demonstration."""
    cache_capacity = 10_000_000 # Size of cache
    # We can now just instantiate our class
    cache = ThreadSafeDict(cache_capacity)

    unique_keys = 100_000_000_000 # Possible variety of keys the cache sees
    num_items = 9_000_000 # Number of entries in to the cache
    items_to_process = np.random.randint(1, unique_keys + 1, size=num_items, dtype=np.int64)

    print("Starting parallel simulation with the ThreadSafeDict jitclass.")
    
    start_time = time.time()
    run_parallel_simulation(cache, items_to_process)
    end_time = time.time()

    # Verify that each unique key was added to the cache exactly once
    keys_in_cache = set(cache.keys) - {EMPTY_KEY}
    
    print("--- Results ---")
    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
    print(f"Number of unique keys possible: {unique_keys:,}")
    print(f"Number of items sent to the cache: {num_items:,}")
    print(f"Number of items successfully stored in cache: {len(keys_in_cache):,}")

    if keys_in_cache >= set(items_to_process):
        print("\nSuccess! The jitclass cache worked correctly.")
    else:
        print("\nFailure. The cache did not store all items.")

if __name__ == "__main__":
    main()
