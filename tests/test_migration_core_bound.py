"""Independent finite checks of the migration-dependent family core bound."""
from itertools import permutations

import numpy as np

from cara.matching_core import peel_fixed_edges


def _optimum(matrix):
    return max(sum(int(matrix[i, j]) for i, j in enumerate(order))
               for order in permutations(range(len(matrix))))


def _envelope(seed, edited):
    """Conservative row-budget envelope, independent of the family compiler."""
    upper = seed.copy()
    for row in range(len(seed)):
        budget = sum(int(seed[row, column]) for column in edited)
        for column in edited:
            upper[row, column] = budget
    return upper


def _sparse(matrix):
    return {(i, j): int(matrix[i, j]) for i in range(len(matrix))
            for j in range(len(matrix)) if matrix[i, j]}


def test_migration_bound_against_exhaustive_seed_alignment():
    rng = np.random.default_rng(173309)
    for hosts in range(2, 7):
        for _ in range(64):
            seed = rng.integers(0, 4, size=(hosts, hosts), dtype=np.int64)
            # Nonempty rows and columns correspond to complete host partitions.
            seed[np.arange(hosts), np.arange(hosts)] += 1
            edited = tuple(sorted(map(int, rng.choice(hosts, 2, replace=False))))
            upper = _envelope(seed, edited)
            edges = _sparse(upper)
            fixed = {edge: value for edge, value in _sparse(seed).items()
                     if edge[1] not in edited}
            kernel = peel_fixed_edges(hosts, edges, fixed, edited)
            movement = int(seed.sum()) - _optimum(seed)
            q = int(seed.sum(axis=1).min())
            bound = min(hosts, len(edited) + 3 * movement // (q + 1))
            assert len(kernel.rows) <= bound
            # Also check the occupancy-sensitive form, which can be tighter.
            costs = sorted(int(value) + 1 for value in seed.sum(axis=1))
            used, ambiguous = 0, 0
            for cost in costs:
                if used + cost > 3 * movement:
                    break
                used += cost
                ambiguous += 1
            assert len(kernel.rows) <= min(hosts, len(edited) + ambiguous)


def test_positive_certificate_graph_is_a_union_of_stars():
    rng = np.random.default_rng(173310)
    for hosts in range(2, 7):
        for _ in range(32):
            seed = rng.integers(0, 4, size=(hosts, hosts), dtype=np.int64)
            edited = (0, hosts - 1)
            upper = _envelope(seed, edited)
            certified = []
            for i in range(hosts):
                for j in range(hosts):
                    if j in edited or not seed[i, j]:
                        continue
                    row_other = max((int(upper[i, k]) for k in range(hosts) if k != j), default=0)
                    column_other = max((int(upper[k, j]) for k in range(hosts) if k != i), default=0)
                    if seed[i, j] >= row_other + column_other:
                        certified.append((i, j))
            row_degrees = [sum(i == r for i, _ in certified) for r in range(hosts)]
            column_degrees = [sum(j == c for _, j in certified) for c in range(hosts)]
            assert all(row_degrees[i] == 1 or column_degrees[j] == 1
                       for i, j in certified)


def test_coefficient_three_sharpness_construction():
    hosts, t = 8, 2
    seed = np.zeros((hosts, hosts), dtype=np.int64)
    seed[np.arange(hosts), np.arange(hosts)] = 2 * t - 1
    seed[np.arange(hosts), (np.arange(hosts) + 1) % hosts] = t
    edited = (0, 1)
    upper = _envelope(seed, edited)
    fixed = {edge: value for edge, value in _sparse(seed).items()
             if edge[1] not in edited}
    kernel = peel_fixed_edges(hosts, _sparse(upper), fixed, edited)
    movement = int(seed.sum()) - _optimum(seed)
    q = int(seed.sum(axis=1).min())
    assert movement == hosts * t
    assert q + 1 == 3 * t
    assert len(kernel.rows) == hosts
    assert kernel.fixed_edges == ()
