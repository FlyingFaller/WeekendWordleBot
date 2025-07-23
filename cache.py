import numpy as np
from numba.experimental import jitclass
from numba.types import int64, float64, ListType, uint64
from numba.typed import List as ListObj
from atomic_ops import atomic_cas, atomic_add, atomic_sub
import typing

if typing.TYPE_CHECKING:
    # Define simple, standard types for the IDE to prevent warnings.
    ListArrUint64 = typing.List[np.ndarray]
    ListArrFloat64 = typing.List[np.ndarray]
else:
    # This block is seen by Python/Numba at runtime.
    # Define the precise Numba types required for compilation.
    ListArrUint64 = ListType(uint64[:])
    ListArrFloat64 = ListType(float64[:])

# --- A Reusable Atomic Integer Class ---

@jitclass({'value': int64[:]})
class AtomicInt:
    """
    A jitclass that encapsulates a single integer and provides atomic
    operations on it, hiding the underlying length-1 array implementation.
    """
    def __init__(self, initial_value=0):
        # The internal representation is still a length-1 array
        self.value = np.array([initial_value], dtype=np.int64)

    def atomic_add(self, v):
        """
        Atomically adds `v` to the integer and returns the value *before*
        the addition. This is the raw behavior of our atomic_add.
        """
        return atomic_add(self.value, 0, v)
    
    def atomic_sub(self, v):
        return atomic_sub(self.value, 0, v)
    
    def atomic_cas(self, cmp, val):
        return atomic_cas(self.value, 0, cmp, val)
    
    def get(self):
        """Returns the current value of the integer."""
        return self.value[0]

    def set(self, v):
        """Sets the integer to a new value (this operation is not atomic)."""
        self.value[0] = v

# --- A List-of-Arrays based, Expanding, Thread-Safe Cache ---

EMPTY_KEY = np.uint64(0)

@jitclass
class Cache:
    segment_capacity: uint64
    full_threshold: float64
    growth_lock: AtomicInt
    key_segments: ListArrUint64
    value_segments: ListArrFloat64
    fill_count_segments: typing.List[AtomicInt]

    def __init__(self, segment_capacity):
        self.segment_capacity = segment_capacity
        self.full_threshold = 0.80*segment_capacity
        self.growth_lock = AtomicInt(0)

        self.key_segments = ListObj.empty_list(uint64[:])
        self.value_segments = ListObj.empty_list(float64[:])
        self.key_segments.append(np.full(segment_capacity, EMPTY_KEY, dtype=np.uint64))
        self.value_segments.append(np.zeros(segment_capacity, dtype=np.float64))

        fill_count_list: list = ListObj()
        fill_count_list.append(AtomicInt(0))
        self.fill_count_segments = fill_count_list

    def _acquire_lock(self):
        """Acquires a spinlock using the clean AtomicInt API."""
        # If growth_lock == 0 then we swap with 1 to aquire, return 0
        # If growth_lock == 1 the swap fails and the while attempts it again (spins)
        while self.growth_lock.atomic_cas(0, 1) == 1:
            pass
            
    def _release_lock(self):
        """Releases the lock using the clean AtomicInt API."""
        self.growth_lock.set(0)

    def _grow(self):
        """Appends new arrays to each list to create a new logical segment."""

        new_fill_count = AtomicInt(0)
        new_values = np.zeros(self.segment_capacity, dtype=np.float64)
        new_keys = np.full(self.segment_capacity, EMPTY_KEY, dtype=np.uint64)
        
        self.fill_count_segments.append(new_fill_count)
        self.value_segments.append(new_values)
        self.key_segments.append(new_keys)

    def _segment_get(self, segment_index, key):
        """Helper to get a key from a specific segment's arrays."""
        keys = self.key_segments[segment_index]
        values = self.value_segments[segment_index]
        
        probe_start = key % self.segment_capacity
        
        for i in range(self.segment_capacity):
            index = (probe_start + i) % self.segment_capacity
            
            if keys[index] == key:
                return values[index], True
            if keys[index] == EMPTY_KEY:
                return 0.0, False
        return 0.0, False

    def _segment_set(self, segment_index, key, value):
        """Helper to set a key in a specific segment's arrays."""
        fill_count: AtomicInt = self.fill_count_segments[segment_index]
        if fill_count.get() >= self.full_threshold:
            return False # Signal that this segment is full

        keys = self.key_segments[segment_index]
        values = self.value_segments[segment_index]
        probe_start = key % self.segment_capacity

        for i in range(self.segment_capacity):
            index = (probe_start + i) % self.segment_capacity

            if keys[index] == key:
                return True # Key already exists, potential to check for hash collision

            if keys[index] == EMPTY_KEY:
                original_value = atomic_cas(keys, index, EMPTY_KEY, key)
                if original_value == EMPTY_KEY:
                    values[index] = value
                    fill_count.atomic_add(1)
                    return True # Success
                if keys[index] == key:
                    return True
        return False

    def get(self, key):
        """Looks for a key by searching segments from newest to oldest."""
        num_segs = len(self.key_segments)
        for i in range(num_segs - 1, -1, -1):
            value, found = self._segment_get(i, key)
            if found:
                return value, True
        return 0.0, False

    def set(self, key, value):
        """Tries to set a value, triggering growth if necessary."""
        while True:
            last_segment_index = len(self.key_segments) - 1
            if self._segment_set(last_segment_index, key, value):
                return

            self._acquire_lock()
            try:
                # Double-check if another thread already grew the cache
                if last_segment_index == len(self.key_segments) - 1:
                    self._grow()
            finally:
                self._release_lock()