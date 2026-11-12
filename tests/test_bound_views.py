"""Independent projection and rational-sum oracles for certificate views."""
from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

from cara.bound_views import BoundViewIndex
from cara.data import canonical_assignment, host_loads
from cara.family_certificates import ResidualFamily
from cara.family_score_bounds import FamilyScoreContext
from cara.query_engine import SharedQueryReplay
from cara.refinement import PROFILE_NAMES, _relative_gains, panel_profile
from cara.workload_views import host_projection


@pytest.mark.parametrize("hosts,horizon", [(2, 1), (2, 3), (3, 8), (17, 5), (33, 17)])
def test_indexed_bounds_cover_every_descendant(hosts, horizon):
    rng = np.random.default_rng(8500 + hosts + horizon)
    seed = tuple(np.arange(hosts * 4) % hosts)
    panel = rng.uniform(.001, 1.5, (3, len(seed), horizon))
    capacity = 2.
    reader = SharedQueryReplay(panel, seed, capacity)
    reference = reader.profile(seed)
    context = FamilyScoreContext.from_reader(reader, reference, compiled_views=True)
    pair = (0, hosts - 1)
    mutable = tuple(sorted([i for i, h in enumerate(seed) if h in pair][:4]))
    family = ResidualFamily(seed, seed, pair, mutable)
    bound = context.for_family(seed, pair, mutable)
    ranges = [(start, start + span) for span in (1 << k for k in range(family.mask_count.bit_length()))
              for start in range(0, family.mask_count, span)]
    lower = bound.lower_profiles(ranges)
    upper = bound.upper_keys(ranges)
    for mask in family.nonempty_masks():
        profile = panel_profile(panel, family.assignment(mask), capacity)
        gain = _relative_gains(reference, profile, PROFILE_NAMES)
        key = tuple(sorted(int(np.floor(x * 2**40)) for x in gain)) + (0, 0.)
        for (start, stop), value, optimistic in zip(ranges, lower, upper):
            if start <= mask < stop and value is not None:
                assert np.all(value <= profile)
                assert optimistic >= key


def test_local_updates_and_two_exclusions_have_rational_enclosures():
    rng = np.random.default_rng(1957)
    hosts, scenes, horizon = 35, 4, 5
    seed = tuple(np.arange(hosts * 4) % hosts)
    panel = rng.uniform(.01, 2., (scenes, len(seed), horizon))
    reader = SharedQueryReplay(panel, seed, 2., max_views=3)
    index = BoundViewIndex(reader, seed)
    for step in range(9):
        loads = host_loads(panel, seed)
        projection = host_projection(loads, 2.)
        for pair in [(0, 1), (17, 34), (step % hosts, (step + 7) % hosts)]:
            result = index.common(pair)
            common = [h for h in range(hosts) if h not in pair]
            for name, values in zip(("active", "tolerant", "affected"), projection[:3]):
                assert np.array_equal(result[name], values[common].sum(axis=0))
            assert np.array_equal(result["native_count"], (loads[common] > 2.).sum(axis=(0, 2)))
            assert np.array_equal(result["peak_max"], projection[6][common].max(axis=0))
            for scene in range(scenes):
                for name, values in zip(("frequency_lo", "excess_lo", "severity_lo"), projection[3:6]):
                    exact = sum((Fraction(float(x)) for x in values[common, scene]), Fraction(0))
                    assert Fraction(float(result[name][scene])) <= exact
                exact_peak = sum((Fraction(float(x)) for x in projection[6][common, scene]), Fraction(0))
                assert Fraction(float(result["peak_hi"][scene])) >= exact_peak
        revised = list(seed)
        source, destination = step % hosts, (step + 1) % hosts
        vm = next(i for i, host in enumerate(seed) if host == source)
        revised[vm] = destination
        seed = canonical_assignment(revised)
        index.update(seed)
    assert index.stats["seed_updates"] == 9
    assert index.stats["changed_groups"] <= 18
    assert index.stored_bytes > 0


def test_previous_family_bounds_are_snapshots_after_seed_update():
    rng = np.random.default_rng(73)
    seed = tuple(np.arange(24) % 6)
    panel = rng.uniform(.01, .7, (3, 24, 8))
    reader = SharedQueryReplay(panel, seed, 1.)
    context = FamilyScoreContext.from_reader(reader, reader.profile(seed), compiled_views=True)
    original = context.for_family(seed, (0, 1), (0, 1, 6, 7))
    expected = original.upper_key(0, 16)
    revised = list(seed)
    revised[2] = 3
    revised = canonical_assignment(revised)
    context.for_family(revised, (0, 1), (0, 1, 6, 7))
    original._cache.clear()
    assert original.upper_key(0, 16) == expected


def test_idle_scene_preserves_exact_refinement_fallback():
    panel = np.zeros((2, 8, 3))
    seed = (0, 1) * 4
    reader = SharedQueryReplay(panel, seed, 1.)
    context = FamilyScoreContext.from_reader(reader, reader.profile(seed), compiled_views=True)
    bound = context.for_family(seed, (0, 1), (0, 1, 2, 3))
    assert not bound.available
    assert bound.upper_key(0, 16) is None
