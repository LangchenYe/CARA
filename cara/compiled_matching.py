"""Compile persistent overlap edges and replay only changed assignment columns.

The logical result is the maximum retained count. Strict fixed-edge certificates
can also establish a unique physical permutation for retained actions. Every
bound covers the complete mask family and its original seed.
"""
from collections import Counter

import numpy as np

from .factorized_matching import _maximum
from .matching_core import Kernel, peel_fixed_edges, support_components


# Chosen on an independent integer-assignment calibration, shared by the
# compiled plan and its dynamic-assignment control. It changes only cost.
DIRECT_CORE_LIMIT = 64


def _add(stats, name, value=1):
    stats[name] = stats.get(name, 0) + int(value)


class _Variable:
    def __init__(self, rows, columns, entries, base, pair, stats, replay):
        self.rows = {row: index for index, row in enumerate(rows)}
        self.column_index = {column: index for index, column in enumerate(columns)}
        self.pair = tuple(pair)
        self.changed = tuple(column for column in pair if column in self.column_index)
        self.local_columns = np.asarray([self.column_index[column] for column in self.changed], dtype=np.int64)
        self.seed = np.zeros((len(rows), len(columns)), dtype=np.int64)
        for (row, column), value in entries.items():
            if row in self.rows and column in self.column_index:
                self.seed[self.rows[row], self.column_index[column]] = value
        self.base_columns = np.asarray([[base.get((row, column), 0) for row in rows]
                                        for column in self.changed], dtype=np.int64)
        self.stats = stats
        self.replay = bool(replay and max(self.seed.shape) > DIRECT_CORE_LIMIT)
        self.solver = None
        self.seed_value = None

    def _compile(self):
        if not self.replay or self.solver is not None:
            return
        try:
            from .assignment_replay import NativeTwoColumnReplay
            self.solver = NativeTwoColumnReplay(self.seed)
        except (ArithmeticError, ValueError, OSError):
            self.replay = False
            _add(self.stats, "matching_exact_fallbacks")
            return
        self.seed_value = self.solver.base_retained
        # The solver owns a snapshot; keep its original rectangle for fallback.
        self.seed = self.solver.weights[:self.seed.shape[0], :self.seed.shape[1]]
        _add(self.stats, "matching_dual_compilations")
        _add(self.stats, "matching_matrix_cells", self.solver.n ** 2)
        _add(self.stats, "matching_compile_augmentations", self.solver.base_stats["augmentations"])

    def retained(self, origins, left, right):
        changed = self.base_columns.copy()
        sides = {column: index for index, column in enumerate(self.changed)}
        for old, first, second in zip(origins, left, right):
            row = self.rows.get(old)
            if row is None:
                continue
            for column, value in zip(self.pair, (first, second)):
                if column in sides:
                    changed[sides[column], row] += value
        self._compile()
        if self.solver is not None:
            try:
                value, stats = self.solver.replay(self.local_columns, changed)
                _add(self.stats, "matching_component_solves")
                _add(self.stats, "matching_dual_replays")
                _add(self.stats, "matching_replay_augmentations", stats["augmentations"])
                _add(self.stats, "matching_replay_scanned_cells", stats["row_slots_scanned"])
                _add(self.stats, "matching_changed_entries", changed.size)
                return value
            except (ArithmeticError, ValueError, OSError):
                _add(self.stats, "matching_exact_fallbacks")
        matrix = self.seed.copy()
        matrix[:, self.local_columns] = changed.T
        _add(self.stats, "matching_component_solves")
        _add(self.stats, "matching_matrix_cells", matrix.size)
        return _maximum(matrix)

    def retained_seed(self):
        self._compile()
        if self.seed_value is None:
            self.seed_value = _maximum(self.seed)
            _add(self.stats, "matching_component_solves")
            _add(self.stats, "matching_matrix_cells", self.seed.size)
        return self.seed_value


class FamilyMatching:
    """Family-uniform fixed edges, residual components, and two-column replay."""

    def __init__(self, incumbent, seed, pair, mutable, symmetric, stats,
                 *, lock_edges=True, replay=True):
        hosts = len(set(seed))
        pair = tuple(pair)
        self._incumbent_plan = None
        if tuple(incumbent) == tuple(seed):
            # This structural case is already a two-row core. Its unchanged
            # singleton edges need neither a dominance scan nor a dual state.
            from .factorized_matching import FamilyMatching as Components
            self._incumbent_plan = Components(incumbent, seed, pair, mutable, symmetric, stats)
            _add(stats, "matching_core_rows", 2)
            _add(stats, "matching_trivial_fixed_edges", hosts - 2)
            return
        entries = Counter(zip(incumbent, seed))
        base = entries.copy()
        for vm in mutable:
            base[incumbent[vm], seed[vm]] -= 1
        if symmetric:
            base[incumbent[mutable[0]], pair[0]] += 1
        origins = Counter(incumbent[vm] for vm in mutable[int(symmetric):])
        upper = base.copy()
        for old, total in origins.items():
            for column in pair:
                upper[old, column] += total
        for edge, value in entries.items():
            upper[edge] = max(upper[edge], value)
        upper = {edge: int(value) for edge, value in upper.items() if value}
        fixed = {edge: value for edge, value in entries.items() if edge[1] not in pair}
        if lock_edges:
            kernel = peel_fixed_edges(hosts, upper, fixed, pair)
        else:
            indices = tuple(range(hosts))
            kernel = Kernel(indices, indices, upper, 0, (), 0,
                            support_components(indices, indices, upper), len(upper))
        self.kernel, self.stats = kernel, stats
        self.fixed_retained = kernel.fixed_value
        self.variables = []
        row_owner = {row: index for index, (rows, _) in enumerate(kernel.components) for row in rows}
        component_entries = [{} for _ in kernel.components]
        retained_columns = set(kernel.columns)
        for (row, column), value in entries.items():
            if row in row_owner and column in retained_columns and value:
                component_entries[row_owner[row]][row, column] = value
        for (rows, columns), local_entries in zip(kernel.components, component_entries):
            if not rows or not columns:
                continue
            if any(column in pair for column in columns):
                self.variables.append(_Variable(rows, columns, local_entries, base, pair, stats, replay))
                _add(stats, "matching_active_rows", len(rows))
                _add(stats, "matching_active_columns", len(columns))
            else:
                if len(rows) == len(columns) == 1:
                    self.fixed_retained += local_entries.get((rows[0], columns[0]), 0)
                    _add(stats, "matching_unchanged_components")
                    continue
                row_index = {row: index for index, row in enumerate(rows)}
                column_index = {column: index for index, column in enumerate(columns)}
                matrix = np.zeros((len(rows), len(columns)), dtype=np.int64)
                for (row, column), value in local_entries.items():
                    if row in row_index and column in column_index:
                        matrix[row_index[row], column_index[column]] = value
                self.fixed_retained += _maximum(matrix)
                _add(stats, "matching_component_solves")
                _add(stats, "matching_matrix_cells", matrix.size)
                _add(stats, "matching_unchanged_components")
        _add(stats, "matching_locked_edges", len(kernel.fixed_edges))
        _add(stats, "matching_lock_rounds", kernel.rounds)
        _add(stats, "matching_support_edges", kernel.original_support_edges)
        _add(stats, "matching_core_rows", len(kernel.rows))

    def retained(self, origins, left, right):
        if self._incumbent_plan is not None:
            return self._incumbent_plan.retained(origins, left, right)
        return self.fixed_retained + sum(model.retained(origins, left, right) for model in self.variables)

    def seed_retained(self):
        if self._incumbent_plan is not None:
            return self._incumbent_plan.seed_retained()
        return self.fixed_retained + sum(model.retained_seed() for model in self.variables)
