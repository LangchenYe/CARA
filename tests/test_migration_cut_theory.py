"""Independent exact oracles for the topology-sensitive migration bound."""
from fractions import Fraction
from itertools import product

import numpy as np

from cara.data import host_loads
from cara.migration_envelope import migration_cut_bound, stored_topology_gain_bound
from cara.refinement import migration_profile


def _cut_oracle(hosts, edges):
    """Enumerate host subsets without using the production graph routine."""
    return max(
        sum((weight for source, target, weight in edges
             if not side[source] and side[target]), Fraction())
        for side in product((False, True), repeat=hosts)
    )


def _load_change_oracle(hosts, edges):
    """Enumerate all corners of the independently bounded demand box."""
    best = Fraction()
    for corner in product((False, True), repeat=len(edges)):
        changes = [Fraction() for _ in range(hosts)]
        for included, (source, target, weight) in zip(corner, edges):
            demand = weight if included else Fraction()
            changes[source] -= demand
            changes[target] += demand
        best = max(best, sum((max(value, 0) for value in changes), Fraction()))
    return best


def test_directed_cut_equals_box_load_change_for_every_three_host_graph():
    possible = [(source, target, Fraction(index + 1, index + 2))
                for index, (source, target) in enumerate(
                    (pair for pair in product(range(3), repeat=2) if pair[0] != pair[1]))]
    for included in product((False, True), repeat=len(possible)):
        edges = [edge for edge, keep in zip(possible, included) if keep]
        cut = _cut_oracle(3, edges)
        assert cut == _load_change_oracle(3, edges)
        # Zero-demand anchors make every physical host nonempty in both
        # assignments; the remaining VMs encode precisely the enumerated graph.
        old = tuple(range(3)) + tuple(source for source, _, _ in edges)
        new = tuple(range(3)) + tuple(target for _, target, _ in edges)
        upper = (Fraction(),) * 3 + tuple(weight for _, _, weight in edges)
        actual = migration_cut_bound(old, new, upper)
        assert actual["exact"] and actual["bound"] == cut
        assert cut == _cut_oracle(3, [(target, source, weight)
                                     for source, target, weight in edges])
        pair_bound = sum((max(
            sum((weight for source, target, weight in edges
                 if (source, target) == (left, right)), Fraction()),
            sum((weight for source, target, weight in edges
                 if (source, target) == (right, left)), Fraction()))
            for left in range(3) for right in range(left + 1, 3)), Fraction())
        assert cut <= pair_bound <= sum((edge[2] for edge in edges), Fraction())
        conservative = migration_cut_bound(old, new, upper, exact_component_limit=1)
        assert cut <= conservative["bound"] <= pair_bound


def test_cut_is_additive_over_support_components_and_not_over_host_pairs():
    left = [(0, 1, Fraction(2)), (1, 2, Fraction(1)), (2, 0, Fraction(3))]
    right = [(3, 4, Fraction(1, 3)), (4, 3, Fraction(2, 3))]
    assert _cut_oracle(5, left + right) == (
        _cut_oracle(3, left) + _cut_oracle(5, right))
    cycle = [(0, 1, Fraction(1)), (1, 2, Fraction(1)), (2, 0, Fraction(1))]
    assert _cut_oracle(3, cycle) == 1
    assert sum((edge[2] for edge in cycle), Fraction()) == 3


def test_two_host_raw_envelope_is_attained_and_halves_the_count_bound():
    incumbent = (0, 0, 0, 1, 1, 1)
    candidate = (0, 0, 1, 0, 1, 1)
    panel = np.array([[[.5], [0.], [.5], [0.], [.5], [.5]]])
    movement = migration_profile(incumbent, candidate)
    assert movement["count"] == 2
    edges = [(incumbent[vm], movement["aligned_assignment"][vm], Fraction(1, 2))
             for vm in movement["moved_vms"]]
    cut = _cut_oracle(2, edges)
    assert cut == Fraction(1, 2)

    def exact_raw(assignment):
        loads = host_loads(panel, assignment)
        # Every operand is dyadic and exactly represented in this example.
        return sum((max(Fraction(float(value)) - 1, 0)
                    for value in loads.flat), Fraction()) / 2

    raw_change = abs(exact_raw(candidate) - exact_raw(incumbent))
    assert raw_change == cut / 2 == Fraction(1, 4)
    assert Fraction(movement["count"], 2) / 2 == 2 * raw_change


def test_stored_cut_bound_covers_exact_differences_of_stored_losses():
    incumbent, candidate = (0, 0, 0, 1, 1, 1), (0, 0, 1, 0, 1, 1)
    supports = (Fraction(1), Fraction(0), Fraction(1, 2),
                Fraction(1, 4), Fraction(3, 4), Fraction(1))
    rng = np.random.default_rng(299)
    panels = rng.uniform(0, 1, (19, 6, 7)) * np.array(supports, dtype=float)[None, :, None]
    for capacity in (.5, 1., 3.75):
        bound = Fraction(stored_topology_gain_bound(
            incumbent, candidate, capacity, horizon=7, demand_upper=supports))
        values = []
        for assignment in (incumbent, candidate):
            loads = host_loads(panels, assignment)
            values.append(np.maximum(loads / capacity - 1., 0.).sum(axis=(0, 2)) / 14)
        for reference, action in zip(*values):
            assert abs(Fraction(float(reference)) - Fraction(float(action))) <= bound


def test_capacity_only_bound_at_every_common_total_on_a_small_load_grid():
    for hosts in range(1, 5):
        by_total = {}
        for loads in product((Fraction(), Fraction(1, 2), Fraction(1), Fraction(2)), repeat=hosts):
            total = sum(loads, Fraction())
            raw = sum((max(load - 1, 0) for load in loads), Fraction()) / hosts
            by_total.setdefault(total, []).append(raw)
        for total, raws in by_total.items():
            refined = (min(total, hosts) - min(total, 1)) / hosts
            assert max(raws) - min(raws) <= refined <= Fraction(hosts - 1, hosts)


def test_combined_cut_and_capacity_bound_has_attained_examples_on_both_branches():
    for hosts in (2, 3, 5):
        for cut in (Fraction(), Fraction(1, 3), Fraction(hosts - 1, 2),
                    Fraction(hosts - 1), Fraction(2 * (hosts - 1))):
            old, new, upper, demand = [0], [0], [Fraction(1)], [Fraction(1)]
            edge_bound = cut / (hosts - 1)
            for source in range(1, hosts):
                old.extend((source, source))
                new.extend((source, 0))
                upper.extend((Fraction(), edge_bound))
                demand.extend((Fraction(), min(edge_bound, 1)))
            graph = migration_cut_bound(old, new, upper)
            assert graph["bound"] == cut
            assert migration_profile(old, new)["count"] == hosts - 1

            def raw(assignment):
                loads = [sum((value for host, value in zip(assignment, demand)
                              if host == h), Fraction()) for h in range(hosts)]
                return sum((max(load - 1, 0) for load in loads), Fraction()) / hosts

            assert abs(raw(old) - raw(new)) == min(cut / hosts, Fraction(hosts - 1, hosts))
