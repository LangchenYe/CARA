"""Exact arithmetic checks for component-wise migration conservation."""
from fractions import Fraction
from itertools import product

import numpy as np

from cara.data import host_loads
from cara.migration_envelope import (
    migration_cut_bound, stored_action_gain_bounds, stored_topology_gain_bound,
)


def _exact_raw(assignment, demand, hosts, capacity=Fraction(1)):
    loads = [sum((x for h, x in zip(assignment, demand) if h == host), Fraction())
             for host in range(hosts)]
    return sum((max(x / capacity - 1, 0) for x in loads), Fraction()) / hosts


def test_component_bound_covers_all_three_host_graphs_and_demand_box_corners():
    edges = [(i, j, Fraction(k + 1, 4)) for k, (i, j) in enumerate(
        (pair for pair in product(range(3), repeat=2) if pair[0] != pair[1]))]
    for included in product((False, True), repeat=len(edges)):
        selected = [edge for edge, keep in zip(edges, included) if keep]
        old = tuple(range(3)) + tuple(i for i, _, _ in selected)
        new = tuple(range(3)) + tuple(j for _, j, _ in selected)
        upper = (Fraction(1),) * 3 + tuple(w for _, _, w in selected)
        bound = Fraction(stored_topology_gain_bound(old, new, 1, horizon=1,
                                                    demand_upper=upper))
        previous = Fraction(stored_topology_gain_bound(old, new, 1, horizon=1,
            demand_upper=upper, component_conservation=False))
        assert bound <= previous
        for corner in product((False, True), repeat=len(selected)):
            moved = tuple(w if high else Fraction()
                          for (_, _, w), high in zip(selected, corner))
            for stationary in ((Fraction(), Fraction(1, 2), Fraction(1)),
                               (Fraction(1),) * 3):
                demand = stationary + moved
                difference = abs(_exact_raw(old, demand, 3) - _exact_raw(new, demand, 3))
                assert difference <= bound


def test_disconnected_components_attain_different_branches_simultaneously():
    # Component {0,1} reaches its conservation cap; {2,3,4} reaches its cut
    # cap. Host 5 is unchanged. The two excess changes have the same sign.
    old = (0, 1, 1, 2, 3, 3, 4, 4, 5)
    new = (0, 1, 0, 2, 3, 2, 4, 2, 5)
    upper = tuple(map(Fraction, (1, 0, 10, 1, 0, .25, 0, .25, .5)))
    demand = tuple(map(Fraction, (1, 0, 1, 1, 0, .25, 0, .25, .5)))
    difference = abs(_exact_raw(old, demand, 6) - _exact_raw(new, demand, 6))
    assert difference == Fraction(1, 4)
    new_bound = stored_topology_gain_bound(old, new, 1, horizon=1, demand_upper=upper)
    old_bound = stored_topology_gain_bound(old, new, 1, horizon=1,
        demand_upper=upper, component_conservation=False)
    assert .25 <= new_bound < .250000000001
    assert 5 / 6 <= old_bound < 5 / 6 + 1e-12
    assert new_bound < old_bound / 3


def test_component_bound_covers_exact_differences_of_canonical_stored_losses():
    old = (0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4)
    new = (0, 0, 1, 1, 1, 0, 2, 2, 3, 3, 3, 2, 4)
    supports = np.array([.5, 1., 1., 1., .5, 1., 2., .5, 2., 1., 2., 2., 1.])
    rng = np.random.default_rng(29051)
    panels = rng.uniform(0, 1, (37, len(old), 7)) * supports[None, :, None]
    for capacity in (.5, 1., 3.75):
        options = dict(horizon=7, demand_upper=supports)
        bounds = stored_action_gain_bounds(old, new, capacity, **options)
        prior = stored_action_gain_bounds(old, new, capacity,
            component_conservation=False, **options)
        assert bounds["gain_bounds"][:3] == prior["gain_bounds"][:3]
        assert bounds["gain_bounds"][3] <= prior["gain_bounds"][3]
        losses = [np.maximum(host_loads(panels, placement) / capacity - 1., 0.).sum(axis=(0, 2)) / 35
                  for placement in (old, new)]
        for before, after in zip(*losses):
            assert abs(Fraction(float(before)) - Fraction(float(after))) <= Fraction(bounds["gain_bounds"][3])


def test_one_component_spanning_every_host_preserves_the_global_cap():
    old = (0, 0, 1, 1, 2, 2)
    new = (0, 1, 1, 2, 2, 0)
    graph = migration_cut_bound(old, new)
    assert len(graph["components"]) == 1
    for capacity in (.25, 1., 4.):
        assert stored_topology_gain_bound(old, new, capacity, horizon=3) == stored_topology_gain_bound(
            old, new, capacity, horizon=3, component_conservation=False)


def test_zero_support_migrations_need_only_the_stored_replay_allowance():
    old, new = (0, 0, 1, 1, 2), (0, 1, 0, 1, 2)
    supports = (1., 0., 0., 1., 1.)
    bound = stored_topology_gain_bound(old, new, 1., horizon=1, demand_upper=supports)
    assert 0 <= bound < 1e-12
    assert migration_cut_bound(old, new, supports)["components"] == []
