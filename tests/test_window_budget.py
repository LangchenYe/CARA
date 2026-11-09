"""Independent rational and stored-arithmetic checks for window allowances."""
from fractions import Fraction
import math

import numpy as np
import pytest

from cara.window_budget import window_raw_allowance_upper


U = Fraction(1, 2**53)
Z = Fraction(1, 2**1075)


def tree_error(count, bound):
    if count < 2:
        return Fraction()
    growth = 1 / (1 - (count - 1) * U)
    return (growth - 1) * count * bound + Z * (count - 1) * growth


def exact_budget(sizes, capacity, horizon, length, bound):
    cap, bound = Fraction(float(capacity)), Fraction(bound)
    hosts, terms = len(sizes), len(sizes) * horizon
    load = max(r * bound + tree_error(r, bound) for r in sizes)
    absolute = load / cap
    ratio = U * absolute + Z
    clip = (1 + U) * ratio + U * (absolute + 1) + Z
    cell = absolute + clip
    prefix = tree_error(length, cell)
    delta = 2 * (1 + U) * prefix + U * horizon * cell + Z
    total = hosts * delta + tree_error(hosts, horizon * cell + delta)
    indexed = (1 + U) * total / terms + U * cell + Z
    canonical = (1 + U) * tree_error(terms, cell) / terms + U * cell + Z
    return indexed + canonical


@pytest.mark.parametrize("scale", [2.**-1070, 2.**-400, 1., 2.**400])
def test_outward_allowance_majorizes_exact_rational_recurrence(scale):
    rng = np.random.default_rng(302731)
    for _ in range(12):
        sizes = tuple(map(int, rng.integers(1, 13, size=int(rng.integers(1, 7)))))
        horizon = int(rng.integers(1, 9))
        length = int(rng.integers(horizon, 513))
        cap = float(3. * scale)
        bound = Fraction(scale) * Fraction(5, 7)
        observed = window_raw_allowance_upper(sizes, cap, horizon, length, bound)
        assert math.isfinite(observed)
        assert Fraction(observed) >= exact_budget(sizes, cap, horizon, length, bound)


def test_count_limit_and_zero_demand_still_have_valid_outward_allowances():
    for length, bound in ((2**53, Fraction(1, 2**400)), (1, Fraction()), (9, Fraction())):
        observed = window_raw_allowance_upper((1,), 1., 1, length, bound)
        assert Fraction(observed) >= exact_budget((1,), 1., 1, length, bound)


def balanced8(values):
    first = (values[0] + values[1]) + (values[2] + values[3])
    second = (values[4] + values[5]) + (values[6] + values[7])
    return first + second


@pytest.mark.parametrize("capacity,bound", [(1.3, 1.), (1., 2.**53), (2.**-400, 2.**-400)])
def test_prefix_subtraction_encloses_original_stored_window_tree(capacity, bound):
    sizes, length, horizon = (2, 3, 5), 41, 8
    rng = np.random.default_rng(613833)
    values = rng.uniform(0., bound, (sum(sizes), length))
    raw_cells, resident = [], 0
    for size in sizes:
        load = np.zeros(length)
        for row in values[resident:resident + size]:load += row
        resident += size
        raw_cells.append(np.maximum(load / capacity - 1., 0.))
    raw_cells = np.asarray(raw_cells)
    prefix = np.zeros((len(sizes), length + 1))
    for hour in range(length):prefix[:, hour + 1] = prefix[:, hour] + raw_cells[:, hour]
    assert np.all(prefix[:, 1:] >= prefix[:, :-1])
    allowance = Fraction(window_raw_allowance_upper(sizes, capacity, horizon, length, bound))
    for start in range(length - horizon + 1):
        canonical = indexed = 0.
        for host in range(len(sizes)):
            canonical += balanced8(raw_cells[host, start:start + horizon])
            difference = prefix[host, start + horizon] - prefix[host, start]
            assert difference >= 0.
            indexed += difference
        canonical /= len(sizes) * horizon
        indexed /= len(sizes) * horizon
        assert abs(Fraction(canonical) - Fraction(indexed)) <= allowance


def test_prefix_range_failure_can_occur_when_a_single_exact_window_is_finite():
    capacity = 2.**-1000
    assert math.isfinite(1. / capacity - 1.)
    with pytest.raises(OverflowError):
        window_raw_allowance_upper((1,), capacity, 1, 2**30, 1.)


@pytest.mark.parametrize("sizes,capacity,horizon,length,bound", [
    ((), 1., 1, 1, 1.), ((0,), 1., 1, 1, 1.), ((1,), 0., 1, 1, 1.),
    ((1,), float("inf"), 1, 1, 1.), ((1,), 1., 2, 1, 1.),
    ((1,), 1., 1, 2**53 + 1, 1.), ((1,), 1., 1, 1, -1.),
    ((1,), 1., 1.5, 2, 1.), ((1, 1), 1., 2**53, 2**53, 1.)])
def test_unsupported_scalar_domains_are_rejected(sizes, capacity, horizon, length, bound):
    with pytest.raises(ValueError):
        window_raw_allowance_upper(sizes, capacity, horizon, length, bound)
