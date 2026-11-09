"""Independent combinatorial checks for count-only support factorization."""
from itertools import permutations

import numpy as np
import pytest

from cara.data import canonical_assignment
from cara.factorized_matching import _maximum
from cara.family_certificates import ResidualFamily


def brute_weight(matrix):
    """Enumerate every injective matching, without calling an assignment solver."""
    rows, columns = matrix.shape
    if rows > columns:
        matrix = matrix.T
        rows, columns = columns, rows
    return max((sum(int(matrix[r, c]) for r, c in enumerate(choice))
                for choice in permutations(range(columns), rows)), default=0)


def overlap(incumbent, labels, hosts):
    matrix = np.zeros((hosts, hosts), dtype=np.int64)
    for old, new in zip(incumbent, labels):
        matrix[old, new] += 1
    return matrix


def mask_labels(family, mask):
    """Expand masks directly, retaining empty physical columns for the oracle."""
    labels = list(family.seed)
    if family.symmetric:
        labels[family.mutable[0]] = family.pair[0]
    for index, vm in enumerate(family.free_vms):
        labels[vm] = family.pair[(mask >> index) & 1]
    return labels


def check_family(incumbent, seed, pair, mutable):
    factored = ResidualFamily(incumbent, seed, pair, mutable, factor_matching=True)
    dense = ResidualFamily(incumbent, seed, pair, mutable, factor_matching=False)
    n, h = factored.n_vms, factored.hosts
    expected = []
    for mask in range(factored.mask_count):
        matrix = overlap(factored.incumbent, mask_labels(factored, mask), h)
        count = n - brute_weight(matrix)
        expected.append(count)
        assert factored.movement_count(mask) == dense.movement_count(mask) == count
    assert factored.seed_movement_count() == dense.seed_movement_count() == (
        n - brute_weight(overlap(factored.incumbent, factored.seed, h)))
    span = 1
    while span <= factored.mask_count:
        for start in range(0, factored.mask_count, span):
            stop = start + span
            lower = factored.movement_lower_bound(start, stop)
            assert lower == dense.movement_lower_bound(start, stop)
            assert lower <= min(expected[start:stop])
        span *= 2
    return factored


@pytest.mark.parametrize('shape', [(0, 4), (1, 4), (4, 1), (2, 2), (2, 4), (4, 2), (3, 4), (4, 4)])
def test_rectangular_matching_matches_all_injections(shape):
    rng = np.random.default_rng(106)
    for _ in range(15):
        matrix = rng.integers(0, 10, size=shape)
        assert _maximum(matrix) == brute_weight(matrix)


@pytest.mark.parametrize('hosts', [2, 3, 4, 5])
def test_every_mask_and_dyadic_bound_against_all_host_permutations(hosts):
    rng = np.random.default_rng(500 + hosts)
    for _ in range(18):
        n = 4 * hosts
        incumbent = canonical_assignment(rng.permutation(np.repeat(np.arange(hosts), 4)))
        seed = canonical_assignment(rng.permutation(np.repeat(np.arange(hosts), 4)))
        pair = tuple(map(int, rng.choice(hosts, 2, replace=False)))
        eligible = [i for i, host in enumerate(seed) if host in pair]
        mutable = tuple(sorted(map(int, rng.choice(eligible, min(6, len(eligible)), replace=False))))
        check_family(incumbent, seed, pair, mutable)


def test_split_merge_support_with_rectangular_active_and_fixed_components():
    incumbent = (0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3)
    seed = (0, 0, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3)
    left = check_family(incumbent, seed, (0, 1), (0, 2))
    right = check_family(incumbent, seed, (2, 3), (3, 4, 9))
    assert left.stats['matching_active_rows'] == 1
    assert left.stats['matching_active_columns'] == 2
    assert right.stats['matching_active_rows'] == 3
    assert right.stats['matching_active_columns'] == 2


@pytest.mark.parametrize('pair', [(0, 1), (1, 0)])
def test_empty_background_symmetry_and_empty_leaf_counts(pair):
    incumbent = (0, 0, 0, 1, 1, 1, 2, 2)
    seed = (0, 1, 0, 1, 0, 1, 2, 2)
    family = check_family(incumbent, seed, pair, tuple(range(6)))
    assert family.symmetric
    assert family.mask_count == 32
    assert len(family.nonempty_masks()) == 31


def test_many_unchanged_hosts_reduce_solver_dimension_without_changing_count():
    incumbent = tuple(np.repeat(np.arange(40), 3))
    family = ResidualFamily(incumbent, incumbent, (5, 31), (15, 16, 93, 94), factor_matching=True)
    dense = ResidualFamily(incumbent, incumbent, (5, 31), (15, 16, 93, 94), factor_matching=False)
    assert [family.movement_count(mask) for mask in range(family.mask_count)] == [dense.movement_count(mask) for mask in range(dense.mask_count)]
    assert family.stats['matching_active_rows'] == family.stats['matching_active_columns'] == 2
    assert family.stats['matching_unchanged_components'] == 38
    assert family.stats['matching_matrix_cells'] < dense.stats['matching_matrix_cells']
