"""Independent optima and primal-dual certificates for column reoptimization."""
from itertools import permutations, product

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from cara.assignment_replay import NativeTwoColumnReplay, prepare_assignment_replay


def brute(matrix):
    if matrix.shape[0] > matrix.shape[1]:
        matrix = matrix.T
    return max(sum(int(matrix[row, column]) for row, column in enumerate(order))
               for order in permutations(range(matrix.shape[1]), matrix.shape[0]))


def oracle(matrix):
    rows, columns = linear_sum_assignment(matrix, maximize=True)
    return sum(int(matrix[row, column]) for row, column in zip(rows, columns))


def certificate(plan, state, original, columns=None, changed=None):
    matrix = np.zeros((plan.n, plan.n), dtype=np.int64)
    matrix[:original.shape[0], :original.shape[1]] = original
    if columns is not None:
        matrix[:original.shape[0], columns] = np.asarray(changed).T
    u, v, row_to_column, column_to_row = state
    rows = np.arange(plan.n)
    assert np.all(u[:, None] + v[None, :] <= -matrix)
    np.testing.assert_array_equal(column_to_row[row_to_column], rows)
    np.testing.assert_array_equal(u + v[row_to_column], -matrix[rows, row_to_column])
    assert -sum(map(int, u)) - sum(map(int, v)) == sum(map(int, matrix[rows, row_to_column]))


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (4, 1), (2, 2), (2, 4), (4, 2), (3, 4), (4, 3), (5, 5)])
def test_rectangular_zero_padding_and_all_small_matching_oracles(shape):
    rng = np.random.default_rng(1301 + sum(shape))
    for trial in range(8):
        weights = rng.integers(0, 8, size=shape)
        if trial % 2:
            weights[rng.random(shape) < .75] = 0
        plan = NativeTwoColumnReplay(weights)
        assert plan.base_retained == brute(weights)
        certificate(plan, plan._base_state, weights)
        for number in range(1, min(2, shape[1]) + 1):
            columns = rng.choice(shape[1], number, replace=False)
            changed = rng.integers(0, 8, size=(number, shape[0]))
            candidate = weights.copy()
            candidate[:, columns] = changed.T
            retained, stats = plan.replay(columns, changed)
            assert retained == brute(candidate)
            assert stats["augmentations"] == number
            certificate(plan, plan._state, weights, columns, changed)


def test_all_binary_two_by_two_bases_and_updates_with_ties():
    matrices = [np.asarray(values, dtype=np.int64).reshape(2, 2)
                for values in product(range(2), repeat=4)]
    for weights in matrices:
        plan = NativeTwoColumnReplay(weights)
        for columns in ([0, 1], [1, 0]):
            for changed in matrices:
                candidate = weights.copy()
                candidate[:, columns] = changed.T
                retained, _ = plan.replay(columns, changed)
                assert retained == brute(candidate)
                certificate(plan, plan._state, weights, columns, changed)


@pytest.mark.parametrize("shape", [(17, 31), (31, 17), (32, 32), (65, 65)])
def test_dense_sparse_updates_and_repeated_seed_independence(shape):
    rng = np.random.default_rng(4401 + sum(shape))
    for sparse in (False, True):
        weights = rng.integers(0, 40, size=shape)
        if sparse:
            weights[rng.random(shape) < .9] = 0
        plan = NativeTwoColumnReplay(weights)
        frozen = plan._base_state.copy()
        original = weights.copy()
        assert plan.base_retained == oracle(weights)
        assert not plan.weights.flags.writeable
        np.testing.assert_array_equal(plan.weights[:shape[0], :shape[1]], weights)
        columns = rng.choice(shape[1], 2, replace=False)
        first = rng.integers(0, 40, size=(2, shape[0]))
        first_value = plan.replay(columns, first)[0]
        for step in range(12):
            selected = columns[:1] if step % 3 == 0 else columns
            changed = rng.integers(0, 40, size=(len(selected), shape[0]))
            candidate = weights.copy()
            candidate[:, selected] = changed.T
            assert plan.replay(selected, changed)[0] == oracle(candidate)
            certificate(plan, plan._state, weights, selected, changed)
        assert plan.replay(columns, first)[0] == first_value
        assert plan.replay(columns, weights[:, columns].T)[0] == plan.base_retained
        np.testing.assert_array_equal(plan._base_state, frozen)
        np.testing.assert_array_equal(weights, original)


def test_connected_cycle_does_not_require_disconnected_support():
    size = 127
    rows = np.arange(size)
    weights = np.zeros((size, size), dtype=np.int64)
    weights[rows, rows] = 7
    weights[rows, (rows + 1) % size] = 1
    plan = NativeTwoColumnReplay(weights)
    assert plan.base_stats["greedy_matches"] == size
    assert plan.base_stats["augmentations"] == 0
    columns = [0, 65]
    changed = weights[:, columns].T.copy()
    changed[0, 0] -= 4
    changed[1, 0] += 4
    candidate = weights.copy()
    candidate[:, columns] = changed.T
    actual, stats = plan.replay(columns, changed)
    assert actual == oracle(candidate)
    assert stats["augmentations"] == 2
    certificate(plan, plan._state, weights, columns, changed)


def test_integer_values_beyond_float_exact_range_and_checked_fallback():
    unit = 2**54
    weights = np.asarray([[unit, unit + 1], [unit + 1, unit]], dtype=np.int64)
    plan = NativeTwoColumnReplay(weights)
    assert plan.base_retained == brute(weights) == 2 * unit + 2
    changed = np.asarray([[unit + 3, unit], [unit, unit + 2]], dtype=np.int64)
    assert plan.replay([0, 1], changed)[0] == brute(changed.T)
    certificate(plan, plan._state, weights, [0, 1], changed)
    with pytest.raises(ValueError, match="integer range"):
        NativeTwoColumnReplay(np.full((2, 2), np.iinfo(np.int64).max, dtype=np.int64))
    with pytest.raises(ValueError, match="integer range"):
        plan.replay([0], np.full((1, 2), np.iinfo(np.uint64).max, dtype=np.uint64))
    assert plan.replay([0, 1], weights.T)[0] == plan.base_retained


@pytest.mark.parametrize("weights", [[], [[-1]], [[1.5]], np.zeros((0, 3), dtype=int), [1, 2]])
def test_invalid_initial_inputs(weights):
    with pytest.raises(ValueError):
        NativeTwoColumnReplay(weights)


@pytest.mark.parametrize("columns,changed", [
    ([0, 0], [[1, 2], [3, 4]]), ([0.5], [[1, 2]]), ([2], [[1, 2]]),
    ([-1], [[1, 2]]), ([], []), ([0], [[1.5, 2]]), ([0], [[-1, 2]]),
    ([0], [[1]]), ([[0]], [[1, 2]])])
def test_invalid_update_cannot_change_base(columns, changed):
    weights = np.asarray([[3, 1], [1, 3]])
    plan = NativeTwoColumnReplay(weights)
    frozen = plan._base_state.copy()
    with pytest.raises(ValueError):
        plan.replay(columns, changed)
    np.testing.assert_array_equal(plan._base_state, frozen)
    assert plan.replay([0, 1], weights.T)[0] == 6


def test_loader_reuses_shared_temporary_cache():
    metadata = prepare_assignment_replay()
    assert "cara-paired-kernels" in metadata["library"]
    assert "assignment-" in metadata["library"]
    assert prepare_assignment_replay() == metadata

