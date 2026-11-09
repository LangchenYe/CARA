"""Single-instance fixed-edge reduction as an experiment-only matching control.

The union-support decomposition and unchanged components are shared with the
component matcher. Each new count signature then applies the same exact edge
reduction to its actual varying matrix, allowing every column to be eliminated.
Its residual components are solved afresh by the common exact assignment
routine. No family-wide locks or dual state are retained between signatures.
"""
from contextlib import contextmanager
from time import process_time

import numpy as np

from cara.factorized_matching import FamilyMatching as ComponentMatching, _maximum
from cara.matching_core import peel_fixed_edges


def _add(stats, name, value=1):
    stats[name] = stats.get(name, 0) + value


def _entries(matrix):
    rows, columns = np.nonzero(matrix)
    return {(int(row), int(column)): int(matrix[row, column])
            for row, column in zip(rows, columns)}


def _static_maximum(size, entries, stats, started):
    """Reduce a square matrix with implicit zero entries, then solve exactly."""
    kernel = peel_fixed_edges(size, entries, entries, changed_columns=())
    owners, matrices, row_indices, column_indices = {}, [], [], []
    for rows, columns in kernel.components:
        if not rows or not columns:
            continue
        position = len(matrices)
        owners.update((row, position) for row in rows)
        row_indices.append({row: index for index, row in enumerate(rows)})
        column_indices.append({column: index for index, column in enumerate(columns)})
        matrices.append(np.zeros((len(rows), len(columns)), dtype=np.int64))
    for (row, column), value in kernel.upper.items():
        position = owners[row]
        matrices[position][row_indices[position][row], column_indices[position][column]] = value
    prepared = process_time()
    result = kernel.fixed_value + sum(_maximum(matrix) for matrix in matrices)
    solved = process_time()
    _add(stats, "matching_static_preparation_cpu_seconds", prepared - started)
    _add(stats, "matching_static_solve_cpu_seconds", solved - prepared)
    _add(stats, "matching_static_reductions")
    _add(stats, "matching_static_input_rows_sum", size)
    _add(stats, "matching_static_core_rows_sum", len(kernel.rows))
    stats["matching_static_core_rows_max"] = max(stats.get("matching_static_core_rows_max", 0), len(kernel.rows))
    _add(stats, "matching_static_locked_edges", len(kernel.fixed_edges))
    _add(stats, "matching_static_lock_rounds", kernel.rounds)
    _add(stats, "matching_component_solves", len(matrices))
    _add(stats, "matching_matrix_cells", sum(matrix.size for matrix in matrices))
    return result


class FamilyMatching(ComponentMatching):
    """Cache only unchanged support components; repeat reduction per signature."""

    def __init__(self, incumbent, seed, pair, mutable, symmetric, stats,
                 *, lock_edges=True, replay=True):
        started = process_time()
        super().__init__(incumbent, seed, pair, mutable, symmetric, stats)
        self._size = max(self.seed.shape)
        self._seed_entries = _entries(self.seed)
        self._base_entries = _entries(self.base)
        # Subsequent signatures update sparse counts rather than scan H^2 cells.
        del self.seed, self.base
        _add(stats, "matching_static_setup_cpu_seconds", process_time() - started)

    def retained(self, origins, left, right):
        started = process_time()
        entries = self._base_entries.copy()
        for old, first, second in zip(origins, left, right):
            row = self.rows[old]
            for column, value in zip(self.columns, (first, second)):
                if value:
                    edge = row, column
                    entries[edge] = entries.get(edge, 0) + int(value)
        value = _static_maximum(self._size, entries, self.stats, started)
        _add(self.stats, "matching_static_signature_reductions")
        return self.fixed_retained + value

    def seed_retained(self):
        value = _static_maximum(self._size, self._seed_entries, self.stats, process_time())
        _add(self.stats, "matching_static_seed_reductions")
        return self.fixed_retained + value


@contextmanager
def matching_control(enabled=False):
    """Temporarily install the static control in one serial benchmark process."""
    if not enabled:
        yield
        return
    from cara import compiled_matching
    previous = compiled_matching.FamilyMatching
    compiled_matching.FamilyMatching = FamilyMatching
    try:
        yield
    finally:
        compiled_matching.FamilyMatching = previous
