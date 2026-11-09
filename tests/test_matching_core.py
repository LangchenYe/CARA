"""Independent finite checks of family-wide matching-edge certificates."""
from collections import Counter
from itertools import permutations

import numpy as np

from cara.matching_core import peel_fixed_edges


def optimum(matrix):
    rows, columns = matrix.shape
    if not rows or not columns:
        return 0
    if rows <= columns:
        return max(sum(int(matrix[i, j]) for i, j in enumerate(order))
                   for order in permutations(range(columns), rows))
    return max(sum(int(matrix[i, j]) for j, i in enumerate(order))
               for order in permutations(range(rows), columns))


def sparse(matrix):
    return {(i, j): int(matrix[i, j]) for i in range(len(matrix))
            for j in range(matrix.shape[1]) if matrix[i, j]}


def check_kernel(matrix, kernel):
    reduced = matrix[np.ix_(kernel.rows, kernel.columns)]
    full = optimum(matrix)
    assert full == kernel.fixed_value + optimum(reduced)
    factored = sum(optimum(matrix[np.ix_(rows, columns)]) for rows, columns in kernel.components)
    assert full == kernel.fixed_value + factored
    assert len(kernel.rows) == len(kernel.columns)
    assert all(weight >= r_other + c_other for _, _, weight, r_other, c_other, _ in kernel.fixed_edges)


def test_all_binary_matrices_up_to_order_three():
    tied = rectangular = 0
    for hosts in range(1, 4):
        for bits in range(1 << (hosts * hosts)):
            matrix = np.array([(bits >> k) & 1 for k in range(hosts * hosts)], dtype=np.int64).reshape(hosts, hosts)
            edges = sparse(matrix)
            kernel = peel_fixed_edges(hosts, edges, edges, ())
            check_kernel(matrix, kernel)
            tied += sum(weight == r + c for _, _, weight, r, c, _ in kernel.fixed_edges)
            rectangular += sum(len(rows) != len(columns) for rows, columns in kernel.components)
    assert tied > 0
    assert rectangular > 0


def test_sequential_elimination_cascades_and_never_locks_changed_columns():
    matrix = 3 * np.eye(6, dtype=np.int64)
    matrix[np.arange(5), np.arange(1, 6)] = 2
    changed = (4, 5)
    edges = sparse(matrix)
    fixed = {edge: weight for edge, weight in edges.items() if edge[1] not in changed}
    kernel = peel_fixed_edges(6, edges, fixed, changed)
    check_kernel(matrix, kernel)
    assert kernel.rounds == 4
    assert kernel.rows == kernel.columns == (4, 5)
    assert all(column not in changed for _, column, *_ in kernel.fixed_edges)


def test_equal_bound_can_fix_an_optimum_without_preserving_every_tie():
    matrix = np.array([[2, 1, 0, 0], [1, 0, 0, 0],
                       [0, 0, 2, 1], [0, 0, 1, 2]], dtype=np.int64)
    edges = sparse(matrix)
    changed = (2, 3)
    fixed = {edge: weight for edge, weight in edges.items() if edge[1] not in changed}
    kernel = peel_fixed_edges(4, edges, fixed, changed)
    check_kernel(matrix, kernel)
    assert (0, 0) in {(row, column) for row, column, *_ in kernel.fixed_edges}
    # Both diagonal and crossed assignments on the first block have value two.
    assert int(matrix[0, 0] + matrix[1, 1]) == int(matrix[0, 1] + matrix[1, 0])


def test_whole_family_seed_and_relaxed_subtree_members():
    rng = np.random.default_rng(20261107)
    checked = 0
    for _ in range(120):
        hosts = int(rng.integers(2, 7))
        n = hosts * int(rng.integers(2, 5))
        incumbent = np.tile(np.arange(hosts), n // hosts)
        seed = incumbent.copy()
        seed[hosts:] = rng.integers(hosts, size=n-hosts)
        pair = tuple(sorted(map(int, rng.choice(hosts, size=2, replace=False))))
        candidates = np.flatnonzero(np.isin(seed, pair))
        width = min(len(candidates), int(rng.integers(1, 7)))
        mutable = tuple(sorted(map(int, rng.choice(candidates, size=width, replace=False))))
        symmetric = all(vm in mutable for vm in candidates)
        free = mutable[int(symmetric):]
        seed_counts = Counter(zip(incumbent, seed))
        base = seed_counts.copy()
        for vm in mutable:
            base[incumbent[vm], seed[vm]] -= 1
        if symmetric:
            base[incumbent[mutable[0]], pair[0]] += 1
        upper = base.copy()
        for vm in free:
            for column in pair:
                upper[incumbent[vm], column] += 1
        # Symmetry reduction need not contain the original seed.
        for edge, count in seed_counts.items():
            upper[edge] = max(upper[edge], count)
        fixed = {edge: count for edge, count in seed_counts.items() if edge[1] not in pair}
        kernel = peel_fixed_edges(hosts, upper, fixed, pair)
        seed_matrix = np.array([[seed_counts[i, j] for j in range(hosts)] for i in range(hosts)])
        check_kernel(seed_matrix, kernel)
        relaxed = base.copy()
        for vm in free:
            for column in pair:
                relaxed[incumbent[vm], column] += 1
        check_kernel(np.array([[relaxed[i, j] for j in range(hosts)] for i in range(hosts)]), kernel)
        for mask in range(1 << len(free)):
            assignment = seed.copy()
            if symmetric:
                assignment[mutable[0]] = pair[0]
            for bit, vm in enumerate(free):
                assignment[vm] = pair[(mask >> bit) & 1]
            if len(set(assignment)) != hosts:
                continue
            counts = Counter(zip(incumbent, assignment))
            matrix = np.array([[counts[i, j] for j in range(hosts)] for i in range(hosts)])
            check_kernel(matrix, kernel)
            checked += 1
    assert checked > 1000


def test_arbitrary_values_inside_two_column_envelopes():
    rng = np.random.default_rng(717)
    for hosts in range(2, 6):
        for _ in range(20):
            envelope = rng.integers(0, 9, size=(hosts, hosts))
            changed = tuple(sorted(map(int, rng.choice(hosts, 2, replace=False))))
            fixed = {edge: weight for edge, weight in sparse(envelope).items() if edge[1] not in changed}
            kernel = peel_fixed_edges(hosts, sparse(envelope), fixed, changed)
            for _ in range(4):
                matrix = envelope.copy()
                for column in changed:
                    matrix[:, column] = [rng.integers(int(value) + 1) for value in matrix[:, column]]
                check_kernel(matrix, kernel)
