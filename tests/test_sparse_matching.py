"""Independent checks of the sparse family control, without timing runs."""
from collections import defaultdict
from itertools import permutations

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from cara.factorized_matching import FamilyMatching as FactoredMatching
from cara.family_certificates import ResidualFamily
from cara.sparse_matching import FamilyMatching as SparseMatching


def canonical(labels):
    mapping = {}
    return tuple(mapping.setdefault(int(label), len(mapping)) for label in labels)


def overlap(incumbent, labels):
    counts = np.zeros((len(set(incumbent)), len(set(incumbent))), dtype=np.int64)
    for old, new in zip(incumbent, labels):
        counts[old, new] += 1
    return counts


def dense_optimum(matrix):
    row, column = linear_sum_assignment(matrix, maximize=True)
    result = int(matrix[row, column].sum())
    if len(matrix) <= 5:
        brute = max(sum(int(matrix[row, column]) for row, column in enumerate(order))
                    for order in permutations(range(len(matrix))))
        assert result == brute
    return result


def expanded_query(incumbent, seed, pair, mutable, symmetric, start, span=1):
    """Build the defining overlap relation workload by workload.

    An unresolved low-order bit contributes once to each edited column.
    This does not use the compiled matrix, subtract a seed count, or call a
    ResidualFamily signature/bound routine.
    """
    free = mutable[int(symmetric):]
    bit_of = {vm: bit for bit, vm in enumerate(free)}
    origins = tuple(sorted({incumbent[vm] for vm in free}))
    left, right = defaultdict(int), defaultdict(int)
    unresolved = span.bit_length() - 1
    counts = np.zeros((len(set(seed)), len(set(seed))), dtype=np.int64)
    for vm, (old, new) in enumerate(zip(incumbent, seed)):
        if symmetric and vm == mutable[0]:
            destinations = (pair[0],)
        elif vm in bit_of:
            bit = bit_of[vm]
            sides = (0, 1) if bit < unresolved else ((start >> bit) & 1,)
            destinations = tuple(pair[side] for side in sides)
            for side in sides:
                (left if side == 0 else right)[old] += 1
        else:
            destinations = (new,)
        for destination in destinations:
            counts[old, destination] += 1
    return counts, origins, tuple(left[old] for old in origins), tuple(right[old] for old in origins)


def check_family(incumbent, seed, pair, mutable):
    symmetric = all(vm in mutable for vm, host in enumerate(seed) if host in pair)
    args = (incumbent, seed, pair, mutable, symmetric)
    stats = defaultdict(int)
    sparse = SparseMatching(*args, stats)
    factored = FactoredMatching(*args, defaultdict(int))
    expected_seed = dense_optimum(overlap(incumbent, seed))
    graph_id, data_id = id(sparse._graph), id(sparse._graph.data)
    indices, indptr = sparse._graph.indices.copy(), sparse._graph.indptr.copy()
    constant = ~np.isin(sparse._graph.indices, pair)
    constant_costs = sparse._graph.data[constant].copy()
    queries = 0
    count = 1 << (len(mutable) - int(symmetric))
    for span in (1 << exponent for exponent in range(count.bit_length())):
        for start in range(0, count, span):
            counts, origins, left, right = expanded_query(*args, start, span)
            expected = dense_optimum(counts)
            assert sparse.retained(origins, left, right) == factored.retained(origins, left, right) == expected
            queries += 1
            # Interleaving seed solves checks that only variable costs are reset.
            if queries % 5 == 0:
                assert sparse.seed_retained() == factored.seed_retained() == expected_seed
                queries += 1
            assert np.all(sparse._graph.data > 0)
            np.testing.assert_array_equal(sparse._graph.data[constant], constant_costs)
    assert sparse.seed_retained() == factored.seed_retained() == expected_seed
    queries += 1
    assert id(sparse._graph) == graph_id and id(sparse._graph.data) == data_id
    np.testing.assert_array_equal(sparse._graph.indices, indices)
    np.testing.assert_array_equal(sparse._graph.indptr, indptr)
    assert stats["matching_sparse_solves"] == queries
    assert stats["matching_matrix_cells"] == queries * sparse._graph.nnz
    assert stats["matching_component_solves"] == stats["matching_unchanged_components"] == 0
    return sparse


@pytest.mark.parametrize("pair", [(0, 1), (1, 0)])
def test_symmetric_family_empty_columns_and_seed_interleaving(pair):
    check_family((0, 0, 0, 1, 1, 1, 2, 2),
                 (0, 1, 0, 1, 0, 1, 2, 2), pair, tuple(range(6)))


def test_nonzero_support_without_a_perfect_real_matching():
    incumbent, seed = (0, 1, 2, 2), (0, 0, 1, 2)
    sparse = check_family(incumbent, seed, (1, 2), (2, 3))
    assert sparse.seed_retained() == 2
    assert sparse._graph.shape == (3, 6)


@pytest.mark.parametrize("pair,mutable", [((0, 1), (0, 2)), ((2, 3), (3, 4, 9))])
def test_rectangular_support_components_and_fixed_backgrounds(pair, mutable):
    check_family((0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3),
                 (0, 0, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3), pair, mutable)


@pytest.mark.parametrize("hosts", [2, 3, 5, 9])
def test_random_families_all_masks_and_relaxed_dyadic_queries(hosts):
    rng = np.random.default_rng(820 + hosts)
    for _ in range(4):
        incumbent = canonical(rng.permutation(np.repeat(np.arange(hosts), 3)))
        seed = canonical(rng.permutation(np.repeat(np.arange(hosts), 3)))
        pair = tuple(map(int, rng.choice(hosts, 2, replace=False)))
        available = [vm for vm, host in enumerate(seed) if host in pair]
        mutable = tuple(sorted(map(int, rng.choice(available, min(5, len(available)), replace=False))))
        check_family(incumbent, seed, pair, mutable)


def test_compiled_storage_is_sparse_with_many_unchanged_hosts():
    incumbent = tuple(map(int, np.repeat(np.arange(96), 3)))
    sparse = check_family(incumbent, incumbent, (5, 88), (15, 16, 264, 265))
    assert sparse._graph.nnz == 2 * 96 + 2
    assert sparse.stats["matching_active_rows"] == 96
    assert sparse.stats["matching_active_columns"] == 192


def test_residual_family_interface_including_signature_cache_and_bounds():
    incumbent = (0,) * 8 + (1,) * 8
    arguments = (incumbent, incumbent, (1, 0), (0, 1, 2, 3, 8, 9, 10, 11))
    sparse = ResidualFamily(*arguments)
    sparse._factor = SparseMatching(sparse.incumbent, sparse.seed, sparse.pair,
                                    sparse.mutable, sparse.symmetric, sparse.stats)
    dense = ResidualFamily(*arguments, factor_matching=False)
    for mask in range(sparse.mask_count):
        assert sparse.movement_count(mask) == dense.movement_count(mask)
    assert sparse.stats["migration_matching_calls"] == 25
    assert sparse.stats["migration_signature_hits"] == 231
    assert sparse.stats["matching_sparse_solves"] == 25
    for span in (1 << exponent for exponent in range(sparse.mask_count.bit_length())):
        for start in range(0, sparse.mask_count, span):
            assert sparse.movement_lower_bound(start, start + span) == dense.movement_lower_bound(start, start + span)
    assert sparse.seed_movement_count() == dense.seed_movement_count()
