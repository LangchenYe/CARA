"""Global sparse assignment control with one compiled pattern per family.

The pattern contains the union of possible overlap edges and one private
dummy column per incumbent row.  Costs B - overlap and B for a dummy turn
maximum retained overlap into a full sparse assignment with positive costs.
Only the two edited columns change between signatures.  No connected
components or invariant matching objectives are computed or cached.
"""
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


class FamilyMatching:
    """Same count-only interface as cara.factorized_matching.FamilyMatching.

Inputs are the validated canonical family and count signatures supplied by
ResidualFamily, including relaxed signatures that put an unresolved workload
in both edited columns.  Physical matching and its tie rule remain outside
this class.  For this plan, matching_matrix_cells counts stored sparse entries
including dummies; matching_active_columns includes the H dummy columns.
"""

    def __init__(self, incumbent, seed, pair, mutable, symmetric, stats):
        self.stats = stats
        hosts = len(set(seed))
        self._shift = len(seed) + 1
        entries = Counter(zip(incumbent, seed))
        support = set(entries)
        origins = sorted({incumbent[vm] for vm in mutable})
        support.update((old, column) for old in origins for column in pair)
        support.update((old, hosts + old) for old in range(hosts))
        edges = sorted(support)
        positions = {edge: index for index, edge in enumerate(edges)}
        rows = np.fromiter((row for row, _ in edges), dtype=np.intp)
        columns = np.fromiter((column for _, column in edges), dtype=np.intp)
        seed_counts = np.fromiter((entries.get(edge, 0) for edge in edges), dtype=np.int64)
        base_counts = seed_counts.copy()
        for vm in mutable:
            base_counts[positions[incumbent[vm], seed[vm]]] -= 1
        if symmetric:
            base_counts[positions[incumbent[mutable[0]], pair[0]]] += 1

        indptr = np.empty(hosts + 1, dtype=np.intp)
        indptr[0] = 0
        np.cumsum(np.bincount(rows, minlength=hosts), out=indptr[1:])
        costs = (self._shift - base_counts).astype(np.float64)
        self._graph = csr_matrix((costs, columns, indptr), shape=(hosts, 2 * hosts))
        self._updates = {old: tuple(positions[old, column] for column in pair)
                         for old in origins}
        self._changed = np.fromiter((index for old in origins for index in self._updates[old]),
                                    dtype=np.intp)
        self._base_costs = costs[self._changed].copy()
        self._seed_costs = (self._shift - seed_counts[self._changed]).astype(np.float64)
        stats["matching_active_rows"] = stats.get("matching_active_rows", 0) + hosts
        stats["matching_active_columns"] = stats.get("matching_active_columns", 0) + 2 * hosts

    def retained(self, origins, left, right):
        data = self._graph.data
        data[self._changed] = self._base_costs
        for old, first, second in zip(origins, left, right):
            first_index, second_index = self._updates[old]
            data[first_index] -= first
            data[second_index] -= second
        return self._solve()

    def seed_retained(self):
        self._graph.data[self._changed] = self._seed_costs
        return self._solve()

    def _solve(self):
        row, column = min_weight_full_bipartite_matching(self._graph)
        # Recover individual integer overlaps, not HB minus a floating sum.
        # Thus a large common cost offset cannot cancel the retained count.
        selected = np.asarray(self._graph[row, column]).ravel().astype(np.int64)
        retained = int(np.sum(self._shift - selected, dtype=np.int64))
        self.stats["matching_sparse_solves"] = self.stats.get("matching_sparse_solves", 0) + 1
        self.stats["matching_matrix_cells"] = self.stats.get("matching_matrix_cells", 0) + self._graph.nnz
        return retained
