"""Independent exhaustive oracles for compact families and safe subtree cuts."""
from itertools import permutations

import numpy as np
import pytest

from cara.data import canonical_assignment, host_memberships
from cara.family_certificates import FamilyCertificates, MembershipIndex, ResidualFamily
from cara.query_engine import ForecastViews, SharedQueryReplay
from cara.refinement import migration_profile, panel_profile


def original_labels(seed, pair, mutable, mask):
    labels = np.asarray(seed).copy()
    fixed = [i for i in np.flatnonzero(np.isin(labels, pair)) if i not in mutable]
    symmetric = not fixed
    for j, vm in enumerate(mutable):
        side = 0 if symmetric and j == 0 else (mask >> (j - int(symmetric))) & 1
        labels[vm] = pair[side]
    return labels


def dyadic_ranges(count):
    width = count
    while width:
        yield from ((start, start + width) for start in range(0, count, width))
        width //= 2


@pytest.mark.parametrize("n,hosts", [(12, 3), (68, 3), (150, 16), (575, 30)])
def test_compact_masks_memberships_identities_and_unit_counts_match_full_oracle(n, hosts):
    rng = np.random.default_rng(73015 + n)
    incumbent = canonical_assignment(np.arange(n) % hosts)
    seed = canonical_assignment(rng.permutation(incumbent))
    pair = (0, hosts - 1)
    residents = np.flatnonzero(np.isin(seed, pair))
    mutable = tuple(map(int, residents[:min(8, len(residents))]))
    family = ResidualFamily(incumbent, seed, pair, mutable)
    index = MembershipIndex()
    for mask in range(family.mask_count):
        raw = original_labels(seed, pair, mutable, mask)
        expected = canonical_assignment(raw)
        assert family.assignment(mask) == expected
        if len(set(raw)) != hosts:
            continue
        assert family.memberships(mask) == host_memberships(expected)
        assert family.identity(mask, index) == index.assignment_key(expected)
        assert family.movement_count(mask) == migration_profile(incumbent, expected)["count"]
    assert family.nonempty_masks() == tuple(mask for mask in range(family.mask_count)
        if len(set(original_labels(seed, pair, mutable, mask))) == hosts)


def test_signature_reuse_and_relaxed_subtree_optimum_are_exact():
    incumbent = (0,) * 8 + (1,) * 8
    family = ResidualFamily(incumbent, incumbent, (0, 1), (0, 1, 2, 3, 8, 9, 10, 11))
    expected = []
    for mask in range(256):
        labels = original_labels(family.seed, family.pair, family.mutable, mask)
        expected.append(migration_profile(incumbent, labels)["count"])
        assert family.movement_count(mask) == expected[-1]
    assert family.stats["migration_matching_calls"] == 25
    assert family.stats["migration_signature_hits"] == 256 - 25
    for start, stop in dyadic_ranges(256):
        assert family.movement_lower_bound(start, stop) == min(expected[start:stop])

    # A non-incumbent seed and a reversed physical pair exercise column
    # permutations and every incumbent row, including empty-host completions.
    incumbent = (0, 0, 1, 1, 2, 2, 0, 1, 2)
    seed = canonical_assignment((0, 2, 1, 0, 2, 1, 0, 1, 2))
    mutable = tuple(i for i, host in enumerate(seed) if host in (2, 0))
    family = ResidualFamily(incumbent, seed, (2, 0), mutable)
    best = []
    for mask in range(family.mask_count):
        labels = original_labels(seed, family.pair, mutable, mask)
        best.append(min(sum(mapping[int(new)] != old for old, new in zip(incumbent, labels))
                        for mapping in permutations(range(3))))
    for start, stop in dyadic_ranges(family.mask_count):
        assert family.movement_lower_bound(start, stop) == min(best[start:stop])
        left, right, unresolved = family.forced_groups(start, stop)
        assert set(left).isdisjoint(right)
        assert set(left).isdisjoint(unresolved) and set(right).isdisjoint(unresolved)
        assert set(left) | set(right) | set(unresolved) == set(mutable)


def test_signature_ablation_keeps_per_mask_memoization_and_exact_counts():
    incumbent = (0,) * 8 + (1,) * 8
    arguments = (incumbent, incumbent, (0, 1), (0, 1, 2, 3, 8, 9, 10, 11))
    shared = ResidualFamily(*arguments)
    separate = ResidualFamily(*arguments, reuse_migration_signatures=False)
    for mask in range(256):
        expected = migration_profile(incumbent, original_labels(*arguments[1:], mask))["count"]
        assert shared.movement_count(mask) == separate.movement_count(mask) == expected
    assert shared.stats["migration_matching_calls"] == 25
    assert shared.stats["migration_signature_hits"] == 231
    assert separate.stats["migration_matching_calls"] == 256
    assert separate.stats["migration_signature_hits"] == 0
    for mask in reversed(range(256)):
        assert shared.movement_count(mask) == separate.movement_count(mask)
    assert shared.stats["migration_matching_calls"] == 25
    assert separate.stats["migration_matching_calls"] == 256
    assert shared.stats["migration_mask_cache_hits"] == separate.stats["migration_mask_cache_hits"] == 256
    assert shared.stats["migration_signature_hits"] == 231


@pytest.mark.parametrize("scenes,times", [(1, 1), (1, 8), (7, 1), (9, 8), (3, 65)])
def test_compact_profile_groups_preserve_every_canonical_bit(scenes, times):
    rng = np.random.default_rng(61192 + times)
    seed = (0,) * 5 + (1,) * 5 + (2,) * 4
    family = ResidualFamily(seed, seed, (0, 1), (0, 1, 2, 5, 6, 7))
    panel = rng.lognormal(-3., 3., (scenes, len(seed), times))
    panel[0, :, 0] = rng.choice([0., .1, np.nextafter(.1, 0.), np.nextafter(.1, 1.),
                                1., 2. ** -53, np.nextafter(0., 1.)], len(seed))
    masks = family.nonempty_masks()[::3]
    reader = SharedQueryReplay(panel, seed, 1., max_views=3, max_bytes=2000)
    actual = reader.profiles_groups([family.memberships(mask) for mask in masks])
    assert family.stats["label_materializations"] == 0
    for mask, profile in zip(masks, actual):
        oracle = original_labels(seed, family.pair, family.mutable, mask)
        assert profile.tobytes() == panel_profile(panel, oracle, 1.).tobytes()
    assert reader.profiles_groups([]).shape == (0, scenes, 6)


@pytest.mark.parametrize("times", [1, 8, 24])
@pytest.mark.parametrize("strategy", ["tree", "leaf"])
def test_forecast_certificates_preserve_predicate_at_boundaries(times, strategy):
    seed = (0,) * 6 + (1,) * 6 + (2,) * 4
    mutable = (0, 1, 2, 3, 6, 7, 8, 9)
    rng = np.random.default_rng(8172)
    base = rng.choice([0., .1, np.nextafter(.1, 0.), np.nextafter(.1, 1.), .2], (16, times))
    fixtures = [(base, .5, 1e-12), (base, .3, 0.),
                (np.full((16, times), np.nextafter(0., 1.)), 2e-323, 0.),
                (np.full((16, times), 1e308), 1.7e308, 0.)]
    for forecast, capacity, tolerance in fixtures:
        with np.errstate(over="ignore", invalid="ignore"):
            family = ResidualFamily(seed, seed, (0, 1), mutable)
            reader = ForecastViews(forecast, capacity, tolerance=tolerance, max_views=3)
            masks = family.nonempty_masks()
            expected = reader.groups_feasible_many([family.groups(mask) for mask in masks])
            fresh = ResidualFamily(seed, seed, (0, 1), mutable)
            certificates = FamilyCertificates(fresh, ForecastViews(forecast, capacity, tolerance=tolerance, max_views=3))
            witnesses = []
            actual = certificates.forecast_masks(observer=witnesses.append, strategy=strategy)
        assert actual == tuple(mask for mask, fits in zip(masks, expected) if fits)
        expected_map = dict(zip(masks, expected))
        assert all(not expected_map[mask] for row in witnesses for mask in row["active_masks"])
        assert fresh.stats["label_materializations"] == 0


@pytest.mark.parametrize("strategy", ["tree", "leaf"])
def test_forecast_rejection_precedes_changed_group_and_label_construction(strategy):
    seed = (0,) * 8 + (1,) * 8
    family = ResidualFamily(seed, seed, (0, 1), (0, 1, 2, 3, 8, 9, 10, 11))
    certificates = FamilyCertificates(family, ForecastViews(np.full((16, 8), .2), 1.))
    assert certificates.forecast_masks(strategy=strategy) == ()
    assert certificates.stats["forecast_excluded_masks"] > 0
    assert family.stats["group_pairs"] < family.mask_count
    assert family.stats["label_materializations"] == 0


def test_migration_cuts_preserve_active_order_and_compact_seen_identities():
    seed = (0,) * 8 + (1,) * 8
    family = ResidualFamily(seed, seed, (0, 1), (0, 1, 2, 3, 8, 9, 10, 11))
    certificates = FamilyCertificates(family, ForecastViews(np.full((16, 2), .01), 1.))
    active = certificates.forecast_masks()
    index = MembershipIndex()
    seen = {index.assignment_key(seed)}
    active = tuple(mask for mask in active if family.identity(mask, index) not in seen)
    retained, witnesses = certificates.prune(active, migration_limit=2, certify_score=False)
    expected = tuple(mask for mask in active if migration_profile(seed,
        original_labels(seed, family.pair, family.mutable, mask))["count"] <= 2)
    assert retained == expected
    excluded = {mask for witness in witnesses for mask in witness["active_masks"]}
    assert excluded == set(active) - set(expected)
    for mask in excluded:
        seen.add(family.identity(mask, index))
    for mask in excluded:
        labels = original_labels(seed, family.pair, family.mutable, mask)
        assert index.assignment_key(labels) in seen
    assert family.stats["label_materializations"] == 0
    for witness in witnesses:
        assert witness["movement_lower_bound"] > 2
        assert all(family.movement_count(mask) > 2 for mask in witness["active_masks"])


@pytest.mark.parametrize("strategy", ["tree", "leaf"])
def test_score_callback_uses_strict_keys_and_retains_uncertain_or_tied_subtrees(strategy):
    seed = (0,) * 8 + (1,) * 8
    family = ResidualFamily(seed, seed, (0, 1), (0, 1, 2, 3, 8, 9, 10, 11))
    certificates = FamilyCertificates(family, ForecastViews(np.full((16, 2), .01), 1.))
    active = tuple(mask for mask in family.nonempty_masks() if mask % 5)
    keys = {mask: (mask // 32, -family.movement_count(mask), -float(family.movement_count(mask)))
            for mask in family.nonempty_masks()}

    class Bounds:
        def upper_keys(self, ranges, movement_lower_bounds=None, cost_lower_bounds=None):
            return [max(keys[mask] for mask in range(start, stop)) for start, stop in ranges]

    cutoff = (5, -2, -2.)
    retained, witnesses = certificates.prune(active, migration_limit=16, cutoff=cutoff,
        score_bounds=Bounds(), score_min_leaves=2, score_strategy=strategy)
    assert all(keys[mask] < cutoff for row in witnesses for mask in row["active_masks"])
    assert all(mask in retained for mask in active if keys[mask] >= cutoff)
    assert sorted(retained + tuple(mask for row in witnesses for mask in row["active_masks"])) == list(active)

    class Tied:
        def upper_keys(self, ranges, movement_lower_bounds=None, cost_lower_bounds=None):
            return [cutoff] * len(ranges)

    same, none = certificates.prune(active, migration_limit=16, cutoff=cutoff,
        score_bounds=Tied(), score_min_leaves=2, score_strategy=strategy)
    assert same == active and none == ()

    class Unknown:
        def upper_keys(self, ranges, movement_lower_bounds=None, cost_lower_bounds=None):
            return [None] * len(ranges)

    same, none = certificates.prune(active, migration_limit=16, cutoff=cutoff,
        score_bounds=Unknown(), score_min_leaves=2, score_strategy=strategy)
    assert same == active and none == ()


def test_invalid_family_ranges_and_mask_sequences_are_rejected():
    seed = (0, 0, 0, 1, 1, 1)
    with pytest.raises(ValueError):
        ResidualFamily(seed, seed, (0, 1), (1, 0))
    family = ResidualFamily(seed, seed, (0, 1), (0, 1, 3, 4))
    for start, stop in ((1, 5), (0, 3), (-1, 1), (0, 32)):
        with pytest.raises(ValueError):
            family.forced_groups(start, stop)
    certificates = FamilyCertificates(family, ForecastViews(np.full((6, 1), .01), 1.))
    with pytest.raises(ValueError):
        certificates.prune((2, 1), migration_limit=1)
