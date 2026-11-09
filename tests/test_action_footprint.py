from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from cara.certification import certify_fresh_mixture, certify_streaming_mixture
from cara.data import burdens, host_loads
from cara.migration_envelope import (
    migration_cut_bound, stored_action_gain_bounds, stored_topology_gain_bound,
)


def test_streaming_and_array_inspection_are_identical_and_read_only_prefix():
    rng = np.random.default_rng(721)
    reference = rng.uniform(.5, .9, (512, 4))
    candidate = reference - rng.uniform(.05, .2, reference.shape)
    options = dict(gain_bounds=(1,) * 4, noninferiority_margins=(.01, .01, 0, .001),
                   min_anchor_gain=.005, batch=16, metric_names=("J", "V", "R", "raw"))
    expected = certify_fresh_mixture(reference, candidate, **options)
    requested = []

    def reader(indices):
        requested.extend(indices.tolist())
        return reference[indices], candidate[indices]

    observed = certify_streaming_mixture(reader, len(reference), 4, **options)
    assert observed == expected
    assert observed["admitted"] and observed["rows"] < len(reference)
    assert requested == list(range(observed["rows"]))


def test_streaming_cap_and_malformed_reader():
    options = dict(gain_bounds=(1,), noninferiority_margins=(0,),
                   anchor=0, min_anchor_gain=0, batch=7)
    requested = []

    def reader(indices):
        requested.extend(indices.tolist())
        return np.zeros((len(indices), 1)), np.zeros((len(indices), 1))

    result = certify_streaming_mixture(reader, 19, 1, **options)
    assert not result["admitted"] and result["rows"] == 19
    assert requested == list(range(19))
    with pytest.raises(ValueError, match="wrong batch shape"):
        certify_streaming_mixture(lambda ids: (np.zeros((1, 1)), np.zeros((1, 1))),
                                  19, 1, **options)
    with pytest.raises(ValueError, match="positive integers"):
        certify_streaming_mixture(reader, 0, 1, **options)


def test_footprint_bounds_cover_stored_boolean_and_raw_differences():
    old = (0, 0, 1, 1, 2, 2, 3, 3)
    new = (0, 1, 0, 1, 2, 2, 3, 3)
    bounds = stored_action_gain_bounds(old, new, 1., horizon=3, demand_upper=1.)
    assert bounds["touched_hosts"] == [0, 1]
    assert .5 <= bounds["gain_bounds"][1] < .500000000000001
    assert bounds["gain_bounds"][1] == bounds["gain_bounds"][2]
    rng = np.random.default_rng(842)
    panel = rng.uniform(0, 1, (31, len(old), 3))
    # Include adversarial capacity boundaries, zeros and nextafter values.
    panel[:5] = np.array([0., .5, np.nextafter(.5, 0.), np.nextafter(.5, 1.), 1.])[:, None, None]
    profiles = [burdens(panel, assignment, 1.)[:, :3] for assignment in (old, new)]
    raw = []
    for assignment in (old, new):
        loads = host_loads(panel, assignment)
        raw.append(np.maximum(loads - 1., 0.).sum(axis=(0, 2)) / 12)
    profiles = [np.column_stack((value, excess)) for value, excess in zip(profiles, raw)]
    for e, bound in enumerate(bounds["gain_bounds"]):
        assert all(abs(Fraction(float(a)) - Fraction(float(b))) <= Fraction(bound)
                   for a, b in zip(profiles[0][:, e], profiles[1][:, e]))
    assert stored_action_gain_bounds(old, old, 1., horizon=3)["gain_bounds"] == (0.,) * 4


def test_capacity_branch_limits_raw_bound_when_migration_bound_is_large():
    old, new = (0, 0, 1, 1), (0, 1, 0, 1)
    bound = stored_topology_gain_bound(old, new, 1., horizon=1, demand_upper=10.)
    assert .5 <= bound < .50000000001


def test_balanced_cycle_large_component_bound_is_stronger_than_pair_count():
    hosts = 19
    old = tuple(range(hosts)) + tuple(range(hosts))
    new = tuple(range(hosts)) + tuple((h + 1) % hosts for h in range(hosts))
    supports = (0,) * hosts + (1,) * hosts
    result = migration_cut_bound(old, new, supports, exact_component_limit=1)
    assert not result["exact"] and result["bound"] == 9


def test_balance_fallback_is_safe_for_all_small_weighted_directed_graphs():
    edges = [(s, t, Fraction(i + 1, 8)) for i, (s, t) in
             enumerate((s, t) for s in range(3) for t in range(3) if s != t)]
    for included in product((False, True), repeat=len(edges)):
        selected = [edge for edge, keep in zip(edges, included) if keep]
        old = tuple(range(3)) + tuple(s for s, _, _ in selected)
        new = tuple(range(3)) + tuple(t for _, t, _ in selected)
        supports = (Fraction(),) * 3 + tuple(w for _, _, w in selected)
        exact = max(sum((w for s, t, w in selected if not side[s] and side[t]), Fraction())
                    for side in product((False, True), repeat=3))
        bounded = migration_cut_bound(old, new, supports, exact_component_limit=1)
        assert exact <= bounded["bound"] <= sum((w for _, _, w in selected), Fraction())
