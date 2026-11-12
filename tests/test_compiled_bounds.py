"""Independent native-profile and continuing-session certificate oracles."""
from collections import Counter

import numpy as np
import pytest

from benchmark_family_execution import PLANS, encoded
from benchmark_reference_oracle import CanonicalProfileOracle
from cara.aggregate_bounds import AggregateBounds
from cara.family_certificates import FamilyCertificates, MembershipIndex, ResidualFamily
from cara.family_score_bounds import FamilyScoreBounds, FamilyScoreContext
from cara.query_engine import ForecastViews, SharedQueryReplay
from cara.refinement import _ordered_records, migration_profile
from cara.subset_certificates import MutableSumIndex
from test_family_score_bounds import _key, _ranges


def verify_bounds(panel, seed, mutable, capacity, compiled, indexed, aggregate):
    family = ResidualFamily(seed, seed, (0, 1), mutable)
    oracle = CanonicalProfileOracle(panel, capacity)
    reference = oracle.profile(seed)
    full = FamilyScoreBounds(panel, seed, (0, 1), mutable, capacity, reference, compiled_bounds=compiled)
    if indexed:
        full.mutable_sum_index = MutableSumIndex(full.mutable_values, symmetric=full.symmetric, minimum_ranges=1)
    valid = family.nonempty_masks()
    profiles = {mask: oracle.profile(family.assignment(mask)) for mask in valid}
    movement = {mask: migration_profile(seed, family.assignment(mask))["count"] for mask in valid}
    target = AggregateBounds(full) if aggregate else full
    for ranges in (_ranges(family.mask_count), tuple((mask, mask + 1) for mask in range(family.mask_count))):
        target._cache.clear()
        if aggregate:
            target._cache_cells = 0
        minimum = [min((movement[mask] for mask in valid if start <= mask < stop), default=0)
                   for start, stop in ranges]
        lower, upper = target.lower_profiles(ranges), target.upper_keys(ranges, minimum, minimum)
        for (start, stop), value, key in zip(ranges, lower, upper):
            if value is None:
                assert key is None
                continue
            for mask in valid:
                if start <= mask < stop:
                    assert np.all(value <= profiles[mask])
                    assert _key(reference, profiles[mask], movement[mask]) <= key
    if compiled:
        prefix = "aggregate" if aggregate else "full"
        assert getattr(full, prefix + "_uniform_fast_nodes") == family.mask_count
        assert full.uniform_radius_compilations == 1
    return full, target


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize("symmetric,horizon,scale", [(False, 1, 2.**-100), (True, 3, 1.), (False, 8, 2.**200)])
def test_every_dyadic_descendant_and_both_sum_paths(compiled, indexed, aggregate, symmetric, horizon, scale):
    seed = (0, 1, 2) * 3
    panel = np.random.default_rng(301 + horizon).uniform(.01, 1.2, (3, 9, horizon)) * scale
    mutable = (0, 1, 3, 4, 6, 7) if symmetric else (0, 1, 3, 4)
    verify_bounds(panel, seed, mutable, scale, compiled, indexed, aggregate)


@pytest.mark.parametrize("aggregate", [False, True])
def test_subnormal_and_capacity_boundaries(aggregate):
    seed, mutable = (0, 1, 2) * 3, (0, 1, 3, 4)
    tiny = np.full((2, 9, 3), np.nextafter(0., 1.))
    verify_bounds(tiny, seed, mutable, 1., True, True, aggregate)
    tolerant = 1. + 1e-12
    row = np.asarray([np.nextafter(1., -np.inf), 1., np.nextafter(1., np.inf),
                      np.nextafter(tolerant, -np.inf), tolerant, np.nextafter(tolerant, np.inf), 0., 0.])
    panel = np.zeros((2, 9, 8))
    panel[:, 0], panel[:, 1], panel[:, 2] = row, row[::-1], .125
    verify_bounds(panel, seed, mutable, 1., True, True, aggregate)


def test_idle_and_overflowing_descendants_have_no_compiled_certificate():
    seed = (0, 1, 2) * 2
    panel = np.zeros((2, 6, 3))
    full = FamilyScoreBounds(panel, seed, (0, 1), (0, 1, 3, 4), 1.,
                             CanonicalProfileOracle(panel, 1.).profile(seed), compiled_bounds=True)
    assert not full.available
    assert AggregateBounds(full).upper_keys([(0, full.mask_count)]) == (None,)
    panel, seed = np.full((1, 4, 1), 7e307), (0, 1, 0, 1)
    full = FamilyScoreBounds(panel, seed, (0, 1), (0, 1), 1e308,
                             np.asarray([[1., 1., 1., .4, .4, 0.]]), compiled_bounds=True)
    assert full.available and not AggregateBounds(full).available
    assert full.upper_key(0, 1) is None
    assert full.full_uniform_fallback_nodes == 1


def session(exhaustive=False, compiled=False, staged=False):
    seed = (0, 1, 2) * 3
    panel = np.random.default_rng(631).uniform(.01, 1.2, (3, 9, 3))
    capacity, top = 1., 2
    oracle = CanonicalProfileOracle(panel, capacity)
    reference = oracle.profile(seed)
    reader = SharedQueryReplay(panel, seed, capacity)
    context = FamilyScoreContext.from_reader(reader, reference, compiled_views=True,
                                             compiled_subsets=True, compiled_bounds=compiled)
    forecast = ForecastViews(np.full((9, 3), .01), capacity, seed)
    index = MembershipIndex()
    seen, pool, snapshots, continuing, acquired = {index.assignment_key(seed)}, {}, [], [], []
    pilots, diagnostics = 0, []
    for pair in ((0, 1), (0, 1), (1, 2)):
        mutable = tuple(vm for vm, host in enumerate(seed) if host in pair)[:4]
        family = ResidualFamily(seed, seed, pair, mutable, matching_backend="compiled", unique_physical_min_hosts=2)
        cert = FamilyCertificates(family, forecast)
        passed = cert.forecast_masks(strategy="leaf")
        active = tuple(mask for mask in passed if family.identity(mask, index) not in seen)
        boundary = lambda: min(row["key"] for row in pool.values()) if len(pool) == top else None
        bounds = context.for_family(seed, pair, mutable)
        surviving = active if exhaustive else cert.prune(active, migration_limit=9, cutoff=boundary(),
            score_bounds=bounds, score_strategy="leaf", score_min_leaves=1, metric_cascade=True,
            staged_acquisition=staged, top_p=top)[0]
        for mask in passed:
            seen.add(family.identity(mask, index))
        pending = {}
        batches = (cert.score_batches(surviving, boundary, batch_size=2)
                   if cert.has_score_schedule and not exhaustive else (surviving,))
        for batch in batches:
            memberships = [family.memberships(mask) for mask in batch]
            acquired.extend(memberships)
            profiles = ([oracle.profile(family.assignment(mask)) for mask in batch] if exhaustive
                        else reader.profiles_groups(memberships))
            for mask, profile in zip(batch, profiles):
                assignment = family.assignment(mask)
                assert profile.tobytes() == oracle.profile(assignment).tobytes()
                moved = family.movement_count(mask)
                pool[assignment] = dict(assignment=assignment, key=_key(reference, profile, moved),
                    movement=dict(count=moved, cost=float(moved)), build_means=profile.mean(axis=0).tolist())
                pending[assignment] = mask
                pool = {row["assignment"]: row for row in _ordered_records(pool.values())[:top]}
        for assignment, record in pool.items():
            if "aligned_assignment" not in record["movement"]:
                physical = family.unique_physical_profile(pending[assignment], assignment)
                record["movement"] = physical if physical is not None else migration_profile(seed, assignment)
            assert encoded(record["movement"]) == encoded(migration_profile(seed, assignment))
        snapshots.append(encoded(_ordered_records(pool.values())))
        continuing.append(frozenset(tuple(index.groups[group] for group in key) for key in seen))
        pilots += cert.stats.get("staged_pilot_profiles", 0)
        diagnostics.append(bounds.diagnostics())
    return snapshots, continuing, acquired, pilots, diagnostics


@pytest.mark.parametrize("compiled,staged", [(False, False), (True, False), (False, True), (True, True)])
def test_retention_seen_physical_and_single_profile_acquisition(monkeypatch, compiled, staged):
    import cara.staged_acquisition as acquisition
    monkeypatch.setattr(acquisition, "MINIMUM_CANDIDATES", 0)
    expected = session(exhaustive=True)
    actual = session(compiled=compiled, staged=staged)
    assert actual[:2] == expected[:2]
    assert actual[1][0] == actual[1][1]
    assert max(Counter(actual[2]).values(), default=0) == 1
    assert actual[3] == (2 if staged else 0)
    assert all(value["uniform_radius_compilations"] <= 1 for value in actual[4])
    if compiled:
        assert sum(value["full_uniform_fast_nodes"] + value["aggregate_uniform_fast_nodes"] for value in actual[4]) > 0


def test_numerical_controls_form_the_declared_two_by_two():
    from reproduce import QUERY_PLANS, PACKED_PLANS
    names = ("recomputed_bounds", "full_uniform", "no_compiled_bounds", "indexed")
    flags = tuple((PLANS[name]["family_compiled_bounds"], PLANS[name]["family_staged_acquisition"]) for name in names)
    assert flags == ((False, False), (True, False), (False, True), (True, True))
    for name in ("family_only", "family_cost_gate"):
        assert not PLANS[name]["family_compiled_bounds"] and not PLANS[name]["family_staged_acquisition"]
    for plans in (QUERY_PLANS, PACKED_PLANS):
        assert {"no_compiled_bounds", "full_uniform", "recomputed_bounds"} <= set(plans)
