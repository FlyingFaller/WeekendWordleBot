################################################################################################################################################
# Inspired by https://github.com/arthurp who (presumably) implemented atomic_add, atomic_sub, atomic_min, and atomic_max                       #
# Sourced from https://github.com/KatanaGraph/katana/blob/4f418f0aeab539c05fd296f3b28c5b7616dc747f/python/katana/numba_support/numpy_atomic.py #
# BSD Licensed                                                                                                                                 #
# Copyright 2018 The University of Texas at Austin                                                                                             #
# See full license at LICENSES/ATOMIC_LICENSE                                                                                                  #    
################################################################################################################################################

from numba import types
from numba.core import cgutils
from numba.core.typing.arraydecl import get_array_index_type
from numba.extending import lower_builtin, type_callable
from numba.np.arrayobj import make_array, basic_indexing

# =================================================================================
# Generic Factory Functions for Read-Modify-Write (RMW)
# =================================================================================

def type_atomic_rmw(context):
    """
    A generic typing function for atomic read-modify-write operations.
    It ensures the array, index, and value are of compatible types.
    """
    def typer(ary, idx, val):
        out = get_array_index_type(ary, idx)
        if out is not None:
            res = out.result
            if context.can_convert(val, res):
                # The function returns the original value, which has the same
                # type as the array's elements.
                return res
    return typer

def make_atomic_rmw_lowerer(int_op, float_op):
    """
    A factory that creates a low-level implementation function for a
    given set of integer and floating-point atomic operations.
    
    Args:
        int_op (str): The name of the LLVM atomic operation for integers.
        float_op (str): The name of the LLVM atomic operation for floats.

    Returns:
        A function suitable for use with @lower_builtin.
    """
    def lowerer(context, builder, sig, args):
        aryty, idxty, valty = sig.args
        ary, idx, val = args

        # This boilerplate logic finds the memory address of ary[i]
        if isinstance(idxty, types.BaseTuple):
            indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
            index_types = idxty.types
        else:
            indices = (idx,)
            index_types = (idxty,)

        ary = make_array(aryty)(context, builder, ary)
        dataptr, shapes, _ = basic_indexing(
            context, builder, aryty, ary, index_types, indices, boundscheck=context.enable_boundscheck
        )
        if shapes:
            raise NotImplementedError("Atomic operations do not support slices or complex shapes")

        # Select the correct operation based on the array's dtype
        val = context.cast(builder, val, valty, aryty.dtype)
        op = None
        if isinstance(aryty.dtype, types.Integer):
            op = int_op
        elif isinstance(aryty.dtype, types.Float):
            op = float_op
        else:
            raise TypeError(f"Atomic operations not supported on array of type {aryty.dtype}")

        # Perform the atomic read-modify-write and return the original value
        dataval = context.get_value_as_data(builder, aryty.dtype, val)
        return builder.atomic_rmw(op, dataptr, dataval, "monotonic")

    return lowerer

# =================================================================================
# Specific Atomic Implementations
# =================================================================================

# --- Atomic Add ---
def atomic_add(ary, i, v):
    """Atomically performs `ary[i] += v` and returns the original value."""
    orig = ary[i]
    ary[i] += v
    return orig

# --- Atomic Sub ---
def atomic_sub(ary, i, v):
    """Atomically performs `ary[i] -= v` and returns the original value."""
    orig = ary[i]
    ary[i] -= v
    return orig

# --- Create and register the Numba implementations ---

# Use the generic typer for both add and sub
type_callable(atomic_add)(type_atomic_rmw)
type_callable(atomic_sub)(type_atomic_rmw)

# Create the specific lowerers using the factory
lower_atomic_add = make_atomic_rmw_lowerer(int_op="add", float_op="fadd")
lower_atomic_sub = make_atomic_rmw_lowerer(int_op="sub", float_op="fsub")

# Register the generated lowerers
lower_builtin(atomic_add, types.Buffer, types.Any, types.Any)(lower_atomic_add)
lower_builtin(atomic_sub, types.Buffer, types.Any, types.Any)(lower_atomic_sub)

# =================================================================================
# Standalone Atomic Compare and Swap (CAS)
# (This remains separate as it has a different underlying implementation)
# =================================================================================

def atomic_cas(ary, i, cmp, val):
    """Atomically performs a compare-and-swap, returning the original value."""
    orig = ary[i]
    if orig == cmp:
        ary[i] = val
    return orig

@type_callable(atomic_cas)
def type_atomic_cas(context):
    def typer(ary, idx, cmp, val):
        out = get_array_index_type(ary, idx)
        if out is not None:
            res = out.result
            if context.can_convert(cmp, res) and context.can_convert(val, res):
                return res
    return typer

@lower_builtin(atomic_cas, types.Buffer, types.Any, types.Any, types.Any)
def lower_atomic_cas(context, builder, sig, args):
    """Low-level implementation for atomic_cas using `cmpxchg`."""
    aryty, idxty, cmpty, valty = sig.args
    ary, idx, cmp, val = args

    if isinstance(idxty, types.BaseTuple):
        indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
        index_types = idxty.types
    else:
        indices = (idx,)
        index_types = (idxty,)

    ary = make_array(aryty)(context, builder, ary)
    dataptr, shapes, _ = basic_indexing(
        context, builder, aryty, ary, index_types, indices, boundscheck=context.enable_boundscheck
    )
    if shapes:
        raise NotImplementedError("atomic_cas does not support slices or complex shapes")

    cmp = context.cast(builder, cmp, cmpty, aryty.dtype)
    val = context.cast(builder, val, valty, aryty.dtype)
    
    cmp_dataval = context.get_value_as_data(builder, aryty.dtype, cmp)
    val_dataval = context.get_value_as_data(builder, aryty.dtype, val)

    res_pair = builder.cmpxchg(dataptr, cmp_dataval, val_dataval, "monotonic", "monotonic")
    original_value = builder.extract_value(res_pair, 0)

    return context.get_data_as_value(builder, aryty.dtype, original_value)