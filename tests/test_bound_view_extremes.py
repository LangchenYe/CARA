"""Rational oracles for summary maintenance at extreme binary64 scales."""
from fractions import Fraction

import numpy as np

from cara.bound_views import BoundViewIndex
from cara.data import canonical_assignment, host_loads
from cara.query_engine import SharedQueryReplay
from cara.workload_views import host_projection


def test_summary_enclosures_survive_extreme_scales_and_slot_reuse():
    rng = np.random.default_rng(490391)
    hosts, scenes, horizon = 19, 3, 5
    seed = canonical_assignment(np.arange(hosts * 4) % hosts)
    exponents = np.array([-1074, -1073, -1022, -500, -54, 0, 1, 10, 500, 900])
    panel = np.ldexp(np.ones((scenes, len(seed), horizon)),
                     rng.choice(exponents, (scenes, len(seed), horizon)))
    reader = SharedQueryReplay(panel, seed, 1., max_views=2)
    index = BoundViewIndex(reader, seed)
    for step in range(4):
        loads = host_loads(panel, seed)
        projected = host_projection(loads, 1.)
        for pair in [(0, 1), (7, 18), (step + 2, step + 9)]:
            values = index.common(pair)
            common = [h for h in range(hosts) if h not in pair]
            assert np.array_equal(values["native_count"], (loads[common] > 1.).sum(axis=(0, 2)))
            assert np.array_equal(values["peak_max"], projected[6][common].max(axis=0))
            for scene in range(scenes):
                for name, data in zip(("frequency_lo", "excess_lo", "severity_lo"), projected[3:6]):
                    exact = sum((Fraction(float(v)) for v in data[common, scene]), Fraction(0))
                    assert Fraction(float(values[name][scene])) <= exact
                exact = sum((Fraction(float(v)) for v in projected[6][common, scene]), Fraction(0))
                assert Fraction(float(values["peak_hi"][scene])) >= exact
        revised = list(seed)
        source, destination = step, step + 1
        vm = next(vm for vm, host in enumerate(seed) if host == source)
        revised[vm] = destination
        seed = canonical_assignment(revised)
        index.update(seed)


def test_top_three_exclusion_with_zero_padding_and_tied_peaks():
    seed = canonical_assignment(np.arange(68) % 17)
    panel = np.zeros((3, 68, 5))
    panel[0, 0, 0] = 2.
    panel[0, 1, 0] = 3.
    panel[1, :4, 0] = 1.
    panel[2, :2, 0] = np.nextafter(0., 1.)
    reader = SharedQueryReplay(panel, seed, 1.)
    index = BoundViewIndex(reader, seed)
    result = index.common((0, 1))
    assert np.array_equal(result["peak_max"], [0., 1., 0.])
    changed = list(seed)
    changed[2] = 3
    index.update(canonical_assignment(changed))
    # The two excluded slots no longer need be the first two canonical slots.
    projection = host_projection(host_loads(panel, index.seed), 1.)
    result = index.common((0, 1))
    assert np.array_equal(result["peak_max"], projection[6][2:].max(axis=0))
