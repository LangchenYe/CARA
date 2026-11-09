from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from cara.migration_envelope import migration_cut_bound, stored_topology_gain_bound


def test_two_host_swap_and_pure_move():
    # One resident on either side stays, so both assignments use both hosts.
    assert migration_cut_bound((0, 0, 1, 1), (0, 1, 0, 1))["bound"] == 1
    assert migration_cut_bound((0, 0, 1, 1), (0, 1, 1, 1))["bound"] == 1


def test_box_extrema_equal_directed_cut():
    rng = np.random.default_rng(612)
    for _ in range(24):
        old = tuple(range(4)) + tuple(rng.integers(0, 4, 6))
        new = tuple(range(4)) + tuple(rng.integers(0, 4, 6))
        upper = tuple(Fraction(int(v), 8) for v in rng.integers(1, 8, 10))
        moved = [i for i, (u, v) in enumerate(zip(old, new)) if u != v]
        maximum = Fraction()
        for bits in product((0, 1), repeat=len(moved)):
            change = [Fraction()] * 4
            for i, bit in zip(moved, bits):
                change[old[i]] -= bit * upper[i]
                change[new[i]] += bit * upper[i]
            maximum = max(maximum, sum(map(lambda v: max(v, 0), change)))
        result = migration_cut_bound(old, new, upper)
        assert result["exact"] and result["bound"] == maximum
        assert migration_cut_bound(new, old, upper)["bound"] == maximum
        fallback = migration_cut_bound(old, new, upper, exact_component_limit=1)
        assert maximum <= fallback["bound"] <= result["total_moved_demand_bound"]


def test_component_additivity_and_raw_bound():
    old, new = (0, 0, 1, 1, 2, 2, 3, 3), (0, 1, 0, 1, 2, 3, 2, 3)
    result = migration_cut_bound(old, new, Fraction(1, 2))
    assert result["bound"] == 1 and len(result["components"]) == 2
    bound = stored_topology_gain_bound(old, new, 1, horizon=3, demand_upper=.5)
    assert bound > .25
    rng = np.random.default_rng(222)
    for _ in range(60):
        x = rng.uniform(0, .5, (8, 3))
        def raw(assignment):
            load = np.stack([x[np.array(assignment) == h].sum(axis=0) for h in range(4)])
            return np.maximum(load - 1, 0).mean()
        assert abs(raw(old) - raw(new)) <= bound


def test_invalid_inputs():
    with pytest.raises(ValueError):
        migration_cut_bound((0, 1), (0, 0))
    with pytest.raises(ValueError):
        migration_cut_bound((0, 1), (1, 0), -1)
    with pytest.raises(ValueError):
        migration_cut_bound((0, 1), (1, 0), exact_component_limit=0)
