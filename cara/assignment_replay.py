"""Exact integer assignment objectives after one or two column replacements.

A family compiles one optimal matching and feasible dual. Each query starts
from that same state, resets the changed columns' duals, and completes at most
two shortest augmenting paths. Rectangular instances use zero padding. The
returned objective does not specify CARA's final physical tie mapping.
"""
from __future__ import annotations

import ctypes

import numpy as np


_FUNCTIONS = None
_METADATA = None
_LIMIT = np.iinfo(np.int64).max // 64


def prepare_assignment_replay():
    """Build/load the shipped C99 kernel through the shared temporary cache."""
    global _FUNCTIONS, _METADATA
    if _FUNCTIONS is None:
        from .native_runtime import load_kernel
        library, metadata = load_kernel("assignment")
        integer = np.ctypeslib.ndpointer(dtype=np.int64, flags=("C_CONTIGUOUS", "ALIGNED"))
        byte = np.ctypeslib.ndpointer(dtype=np.uint8, flags=("C_CONTIGUOUS", "ALIGNED"))
        compile_one = library.cara_column_compile
        compile_one.argtypes = [ctypes.c_int64, integer, integer, integer, byte, integer, integer]
        compile_one.restype = ctypes.c_int
        replay = library.cara_column_replay
        replay.argtypes = [ctypes.c_int64, integer, integer, ctypes.c_int64,
                          integer, integer, integer, integer, byte, integer, integer]
        replay.restype = ctypes.c_int
        _FUNCTIONS, _METADATA = (compile_one, replay), metadata
    return dict(_METADATA)


def _statistics(values):
    return dict(zip(("augmentations", "scanned_rows", "row_slots_scanned", "greedy_matches"),
                    map(int, values)))


class NativeTwoColumnReplay:
    """Immutable base objective and independent exact column reoptimizations.

Weights must be nonnegative integers. Counts outside the conservative int64
range raise ValueError before conversion; a checked native failure raises
ArithmeticError. Callers can then use their existing exact assignment solver.
An instance owns reusable scratch arrays and is intended for serial queries.
"""

    def __init__(self, weights):
        weights = np.asarray(weights)
        if (weights.ndim != 2 or not weights.size or
                not np.issubdtype(weights.dtype, np.integer) or np.any(weights < 0)):
            raise ValueError("Expected a nonempty nonnegative integer weight matrix")
        self.shape = weights.shape
        self.n = max(weights.shape)
        self._check_range(weights)
        self._weights = np.zeros((self.n, self.n), dtype=np.int64)
        self._weights[:weights.shape[0], :weights.shape[1]] = weights
        self._base_state = np.empty((4, self.n), dtype=np.int64)
        self._state = np.empty_like(self._base_state)
        self._work = np.empty((3, self.n), dtype=np.int64)
        self._used = np.empty(self.n, dtype=np.uint8)
        self._changed = np.zeros((2, self.n), dtype=np.int64)
        self._stats = np.zeros(4, dtype=np.int64)
        self._value = np.zeros(1, dtype=np.int64)
        if _FUNCTIONS is None:
            prepare_assignment_replay()
        status = _FUNCTIONS[0](self.n, self._weights, self._base_state,
                              self._work, self._used, self._stats, self._value)
        if status:
            raise ArithmeticError(f"Assignment compilation requires exact fallback (status {status})")
        self.base_retained = int(self._value[0])
        self.base_stats = _statistics(self._stats)
        self._base_state.setflags(write=False)
        self._weights.setflags(write=False)

    @property
    def weights(self):
        """Read-only square-padded base weights for exact fallback readers."""
        return self._weights

    def _check_range(self, weights):
        if self.n * max(1, int(weights.max(initial=0))) > _LIMIT:
            raise ValueError("Assignment weights exceed the checked integer range; use exact fallback")

    def replay(self, columns, new_weights):
        """Replace one/two original columns; return (optimum, work counters).

        ``new_weights`` has shape (number of changed columns, original row count).
        The base matrix/state remain unchanged, including after a failed call.
        """
        columns = np.asarray(columns)
        new_weights = np.asarray(new_weights)
        if (columns.ndim != 1 or not 1 <= len(columns) <= 2 or
                not np.issubdtype(columns.dtype, np.integer) or
                len(set(map(int, columns))) != len(columns) or
                np.any(columns < 0) or np.any(columns >= self.shape[1])):
            raise ValueError("One or two distinct original integer columns are required")
        if (new_weights.shape != (len(columns), self.shape[0]) or
                not np.issubdtype(new_weights.dtype, np.integer) or np.any(new_weights < 0)):
            raise ValueError("Expected a nonnegative integer vector for each changed column")
        self._check_range(new_weights)
        columns = np.ascontiguousarray(columns, dtype=np.int64)
        self._changed[:len(columns), :self.shape[0]] = new_weights
        status = _FUNCTIONS[1](self.n, self._weights, self._base_state, len(columns),
                              columns, self._changed, self._state, self._work,
                              self._used, self._stats, self._value)
        if status:
            raise ArithmeticError(f"Assignment replay requires exact fallback (status {status})")
        return int(self._value[0]), _statistics(self._stats)

