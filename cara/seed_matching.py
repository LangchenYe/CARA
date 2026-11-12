"""Reusable seed certificates for complete two-host matching families.

For seed edge (i,j) of weight w, 3*w >= s_i+t_j certifies that edge in every
family which leaves column j unchanged. Here s_i is incumbent occupancy and
t_j is the seed group's size. An edited column can redistribute only residents
outside any unchanged column, so its alternative row weight is at most s_i-w;
the unchanged column's other-row weight is at most t_j-w. A seed index stores
a disjoint set of these universal edges once. Each family restores only the
edges in its edited columns, then runs the tighter family reduction locally.
"""
from collections import Counter, OrderedDict
from collections.abc import Sequence
from itertools import islice

import numpy as np

from .compiled_matching import _Variable, _add
from .data import canonical_assignment, host_memberships
from .factorized_matching import _maximum
from .matching_core import Kernel, peel_fixed_edges


class _FixedEdges(Sequence):
    """Share seed edges without copying H records into every family."""

    def __init__(self, index, skipped_positions, local_edges, affected_rows):
        self.index = index
        self.seed_edges = index.fixed_edges
        self.skipped_positions = frozenset(skipped_positions)
        self.local_edges = local_edges
        self.affected_rows = frozenset(affected_rows)

    def __len__(self):
        return len(self.seed_edges) - len(self.skipped_positions) + len(self.local_edges)

    def __iter__(self):
        for position, edge in enumerate(self.seed_edges):
            if position in self.skipped_positions:
                continue
            row, column, weight, row_mass, _, round_number = edge
            # Only mutable origins can change a row. An unchanged column's
            # other-row maximum is reusable even for an affected origin.
            row_other = (row_mass if row in self.affected_rows
                         else self.index.row_alternatives[row])
            yield (row, column, weight, row_other,
                   self.index.column_alternatives[column], round_number)
        yield from self.local_edges

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self)[index]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return next(islice(iter(self), index, None))


class SeedMatchingIndex:
    """Immutable overlap structure and universal first-pass seed certificates."""

    def __init__(self, incumbent, seed):
        self.incumbent, self.seed = canonical_assignment(incumbent), canonical_assignment(seed)
        self.n_vms = len(self.seed)
        self.hosts = len(set(self.seed))
        if (not self.n_vms or self.n_vms > 2**50 or len(self.incumbent) != self.n_vms
                or len(set(self.incumbent)) != self.hosts):
            raise ValueError("A seed index requires complete equal-host assignments")
        self.seed_groups = host_memberships(self.seed)
        entries = Counter(zip(self.incumbent, self.seed))
        self.entries = dict(entries)
        self.row_edges = [dict() for _ in range(self.hosts)]
        row_sums, column_sums = [0] * self.hosts, [0] * self.hosts
        for (row, column), value in entries.items():
            self.row_edges[row][column] = value
            row_sums[row] += value
            column_sums[column] += value
        self.row_sums, self.column_sums = tuple(row_sums), tuple(column_sums)
        used_rows, used_columns, edges = set(), set(), []
        # The positive certificate graph is a union of stars. Any first-fit
        # maximal matching is maximum-cardinality, so sorting is unnecessary.
        for (row, column), weight in entries.items():
            if (3 * weight < row_sums[row] + column_sums[column]
                    or row in used_rows or column in used_columns):
                continue
            edges.append((row, column, weight,
                          row_sums[row] - weight, column_sums[column] - weight, 1))
            used_rows.add(row)
            used_columns.add(column)
        self.fixed_edges = tuple(edges)
        self.fixed_value = sum(edge[2] for edge in edges)
        self.fixed_columns = {edge[1]: position for position, edge in enumerate(edges)}
        self.row_alternatives = {row: max((value for other, value in self.row_edges[row].items()
                                          if other != column), default=0)
                                 for row, column, *_ in edges}
        self.column_alternatives = dict.fromkeys(self.fixed_columns, 0)
        fixed_rows = {column: row for row, column, *_ in edges}
        for (row, column), value in entries.items():
            if column in fixed_rows and row != fixed_rows[column]:
                self.column_alternatives[column] = max(self.column_alternatives[column], value)
        self.unmatched_rows = tuple(row for row in range(self.hosts) if row not in used_rows)
        self.unmatched_columns = tuple(column for column in range(self.hosts) if column not in used_columns)
        self.uses = 0

    def family(self, pair, mutable, symmetric, stats, *, replay=True):
        """Return a compiled-family compatible exact matching value processor."""
        pair, mutable = tuple(map(int, pair)), tuple(map(int, mutable))
        if (len(pair) != 2 or len(set(pair)) != 2
                or any(column < 0 or column >= self.hosts for column in pair)
                or not mutable or mutable != tuple(sorted(set(mutable)))
                or any(vm < 0 or vm >= self.n_vms or self.seed[vm] not in pair for vm in mutable)):
            raise ValueError("Expected two seed columns and ordered distinct mutable residents")
        expected_symmetric = len(mutable) == sum(self.column_sums[column] for column in pair)
        if bool(symmetric) != expected_symmetric:
            raise ValueError("The symmetry representative must match the family backgrounds")
        _add(stats, "matching_seed_compilations" if not self.uses else "matching_seed_reuses")
        if not self.uses:
            _add(stats, "matching_seed_index_entries", len(self.entries))
        self.uses += 1
        return IndexedFamilyMatching(self, pair, mutable, bool(symmetric), stats, replay=replay)


class IndexedFamilyMatching:
    """Family-local tightening after restoring the edited seed columns."""

    def __init__(self, index, pair, mutable, symmetric, stats, *, replay=True):
        self._incumbent_plan = None
        self.stats = stats
        restored = tuple(index.fixed_columns[column] for column in pair
                         if column in index.fixed_columns)
        rows = tuple(sorted((*index.unmatched_rows,
                             *(index.fixed_edges[pos][0] for pos in restored))))
        columns = tuple(sorted((*index.unmatched_columns,
                                *(index.fixed_edges[pos][1] for pos in restored))))
        if len(rows) != len(columns):
            raise ArithmeticError("Paired seed-edge restoration must leave a square core")
        row_index, column_index = ({row: pos for pos, row in enumerate(rows)},
                                   {column: pos for pos, column in enumerate(columns)})
        entries = {}
        inspected = 0
        row_support = sum(len(index.row_edges[row]) for row in rows)
        if row_support <= len(rows) * len(columns):
            for row in rows:
                for column, value in index.row_edges[row].items():
                    inspected += 1
                    if column in column_index:
                        entries[row, column] = value
        else:
            # A surviving row may touch many already certified columns. Avoid
            # scanning that global support when direct core lookups are cheaper.
            for row in rows:
                for column in columns:
                    inspected += 1
                    value = index.row_edges[row].get(column, 0)
                    if value:
                        entries[row, column] = value
        base = entries.copy()
        for vm in mutable:
            edge = index.incumbent[vm], index.seed[vm]
            if edge[0] in row_index:
                base[edge] = base.get(edge, 0) - 1
        if symmetric and index.incumbent[mutable[0]] in row_index:
            edge = index.incumbent[mutable[0]], pair[0]
            base[edge] = base.get(edge, 0) + 1
        origins = Counter(index.incumbent[vm] for vm in mutable[int(symmetric):])
        upper = base.copy()
        for row, total in origins.items():
            if row in row_index:
                for column in pair:
                    upper[row, column] = upper.get((row, column), 0) + total
        for edge, value in entries.items():
            upper[edge] = max(upper.get(edge, 0), value)
        local_upper = {(row_index[row], column_index[column]): value
                       for (row, column), value in upper.items() if value}
        local_fixed = {(row_index[row], column_index[column]): value
                       for (row, column), value in entries.items() if column not in pair}
        local_pair = tuple(column_index[column] for column in pair)
        tightened = peel_fixed_edges(len(rows), local_upper, local_fixed, local_pair)
        extra_edges = tuple((rows[row], columns[column], weight, row_other, column_other, rounds + 1)
                            for row, column, weight, row_other, column_other, rounds in tightened.fixed_edges)
        fixed_value = (index.fixed_value - sum(index.fixed_edges[pos][2] for pos in restored)
                       + tightened.fixed_value)
        # The global support count is available without rebuilding its entries.
        extra_support = {(row, column) for row in origins for column in pair
                         if not index.entries.get((row, column), 0)}
        if symmetric:
            edge = index.incumbent[mutable[0]], pair[0]
            if not index.entries.get(edge, 0):
                extra_support.add(edge)
        kernel = Kernel(tuple(rows[row] for row in tightened.rows),
                        tuple(columns[column] for column in tightened.columns),
                        {(rows[row], columns[column]): value
                         for (row, column), value in tightened.upper.items()}, fixed_value,
                        _FixedEdges(index, restored, extra_edges,
                                    (index.incumbent[vm] for vm in mutable)),
                        tightened.rounds + bool(len(index.fixed_edges) - len(restored)),
                        tuple((tuple(rows[row] for row in component_rows),
                               tuple(columns[column] for column in component_columns))
                              for component_rows, component_columns in tightened.components),
                        len(index.entries) + len(extra_support))
        self.kernel = kernel
        self.fixed_retained = kernel.fixed_value
        self.variables = []
        row_owner = {row: owner for owner, (rs, _) in enumerate(kernel.components) for row in rs}
        retained_columns = set(kernel.columns)
        component_entries = [{} for _ in kernel.components]
        for (row, column), value in entries.items():
            if row in row_owner and column in retained_columns and value:
                component_entries[row_owner[row]][row, column] = value
        for (rs, cs), values in zip(kernel.components, component_entries):
            if not rs or not cs:
                continue
            if any(column in pair for column in cs):
                self.variables.append(_Variable(rs, cs, values, base, pair, stats, replay))
                _add(stats, "matching_active_rows", len(rs))
                _add(stats, "matching_active_columns", len(cs))
            elif len(rs) == len(cs) == 1:
                self.fixed_retained += values.get((rs[0], cs[0]), 0)
                _add(stats, "matching_unchanged_components")
            else:
                rmap, cmap = ({row: pos for pos, row in enumerate(rs)},
                              {column: pos for pos, column in enumerate(cs)})
                matrix = np.zeros((len(rs), len(cs)), dtype=np.int64)
                for (row, column), value in values.items():
                    matrix[rmap[row], cmap[column]] = value
                self.fixed_retained += _maximum(matrix)
                _add(stats, "matching_component_solves")
                _add(stats, "matching_matrix_cells", matrix.size)
                _add(stats, "matching_unchanged_components")
        _add(stats, "matching_seed_fixed_edges", len(index.fixed_edges) - len(restored))
        _add(stats, "matching_seed_restored_edges", len(restored))
        _add(stats, "matching_family_entries_scanned", inspected)
        _add(stats, "matching_locked_edges", len(kernel.fixed_edges))
        _add(stats, "matching_lock_rounds", kernel.rounds)
        _add(stats, "matching_support_edges", kernel.original_support_edges)
        _add(stats, "matching_core_rows", len(kernel.rows))

    def retained(self, origins, left, right):
        return self.fixed_retained + sum(model.retained(origins, left, right) for model in self.variables)

    def seed_retained(self):
        return self.fixed_retained + sum(model.retained_seed() for model in self.variables)


class SeedMatchingCache:
    """Bounded reuse for immutable canonical seed objects in a query session.

    Object identities avoid re-hashing an N-label tuple for every family. The
    retained index keeps its seed alive, so an identity cannot be recycled while
    cached. Equal values in a new object may be recompiled without affecting any
    logical answer. The caller's canonical immutable objects are never mutated.
    """

    def __init__(self, incumbent, max_seeds=16):
        if int(max_seeds) != max_seeds or max_seeds < 1:
            raise ValueError("A positive seed-index cache capacity is required")
        self.incumbent = canonical_assignment(incumbent)
        self.max_seeds = int(max_seeds)
        self.indexes = OrderedDict()

    def get(self, seed):
        seed = canonical_assignment(seed)
        key = id(seed)
        index = self.indexes.get(key)
        if index is None:
            index = SeedMatchingIndex(self.incumbent, seed)
            self.indexes[key] = index
            if len(self.indexes) > self.max_seeds:
                self.indexes.popitem(last=False)
        elif index.seed is not seed:
            raise ArithmeticError("A live canonical seed identity was unexpectedly recycled")
        self.indexes.move_to_end(key)
        return index
