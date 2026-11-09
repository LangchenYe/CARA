"""Exact count matching on the support components of a residual family.

Rows are incumbent hosts and columns are seed groups. The union support of
every mask is compiled once. Components outside the two edited columns are
constant; only the affected component is solved for each count signature.
This module returns a count, never a physical host permutation.
"""
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment


def _maximum(matrix):
    """Maximum nonnegative rectangular matching weight, including zero edges."""
    rows, columns = matrix.shape
    if not rows or not columns:
        return 0
    if rows == 1 or columns == 1:
        return int(matrix.max(initial=0))
    if rows == columns == 2:
        return max(int(matrix[0, 0] + matrix[1, 1]),
                   int(matrix[0, 1] + matrix[1, 0]))
    row, column = linear_sum_assignment(matrix, maximize=True)
    return int(matrix[row, column].sum())


class FamilyMatching:
    """Compile the fixed support, then resolve changing overlap counts exactly."""

    def __init__(self, incumbent, seed, pair, mutable, symmetric, stats):
        self.stats = stats
        hosts = len(set(seed))
        parents = list(range(2 * hosts))

        def find(node):
            while node != parents[node]:
                parents[node] = parents[parents[node]]
                node = parents[node]
            return node

        def union(first, second):
            first, second = find(first), find(second)
            if first != second:
                parents[second] = first

        entries = Counter(zip(incumbent, seed))
        for old, new in entries:
            union(old, hosts + new)
        # Include both possible destinations, as well as the original seed.
        # A fixed symmetry representative is included even if it moves left.
        for vm in mutable:
            union(incumbent[vm], hosts + pair[0])
            union(incumbent[vm], hosts + pair[1])

        root = find(hosts + pair[0])
        if find(hosts + pair[1]) != root:
            raise ValueError("The edited columns must share a union component")
        active_rows = [old for old in range(hosts) if find(old) == root]
        active_columns = [new for new in range(hosts) if find(hosts + new) == root]
        row_index = {old: index for index, old in enumerate(active_rows)}
        column_index = {new: index for index, new in enumerate(active_columns)}
        self.rows = row_index
        self.columns = tuple(column_index[new] for new in pair)
        self.seed = np.zeros((len(active_rows), len(active_columns)), dtype=np.int64)
        fixed_components = {}
        for (old, new), count in entries.items():
            component = find(old)
            if component == root:
                self.seed[row_index[old], column_index[new]] = count
            else:
                fixed_components.setdefault(component, []).append((old, new, count))
        self.fixed_retained = 0
        for component in fixed_components.values():
            if len(component) == 1:
                self.fixed_retained += component[0][2]
                continue
            rows = {value: index for index, value in enumerate(sorted({r for r, _, _ in component}))}
            columns = {value: index for index, value in enumerate(sorted({c for _, c, _ in component}))}
            matrix = np.zeros((len(rows), len(columns)), dtype=np.int64)
            for old, new, count in component:
                matrix[rows[old], columns[new]] = count
            self.fixed_retained += _maximum(matrix)
            stats["matching_component_solves"] += 1
            stats["matching_matrix_cells"] += matrix.size
        self.base = self.seed.copy()
        for vm in mutable:
            self.base[row_index[incumbent[vm]], column_index[seed[vm]]] -= 1
        if symmetric:
            self.base[row_index[incumbent[mutable[0]]], self.columns[0]] += 1
        self.stats["matching_active_rows"] += len(active_rows)
        self.stats["matching_active_columns"] += len(active_columns)
        self.stats["matching_unchanged_components"] += len(fixed_components)

    def retained(self, origins, left, right):
        matrix = self.base.copy()
        for old, first, second in zip(origins, left, right):
            row = self.rows[old]
            matrix[row, self.columns[0]] += first
            matrix[row, self.columns[1]] += second
        self.stats["matching_component_solves"] += 1
        self.stats["matching_matrix_cells"] += matrix.size
        return self.fixed_retained + _maximum(matrix)

    def seed_retained(self):
        self.stats["matching_component_solves"] += 1
        self.stats["matching_matrix_cells"] += self.seed.size
        return self.fixed_retained + _maximum(self.seed)
