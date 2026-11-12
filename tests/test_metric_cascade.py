"""Independent native-loss oracles for the endpoint refinement cascade."""
import numpy as np
import pytest

from cara.data import host_loads
from cara.family_certificates import ResidualFamily
from cara.family_score_bounds import FamilyScoreBounds
from cara.metric_cascade import IndicatorBounds
from cara.refinement import PROFILE_NAMES, _relative_gains, migration_profile, panel_profile


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("horizon", [1, 3, 8])
@pytest.mark.parametrize("scale", [2.**-100, 1., 2.**200])
def test_indicator_bounds_cover_every_canonical_descendant(symmetric, horizon, scale):
    rng = np.random.default_rng(202611 + horizon)
    seed = (0, 1, 2, 0, 1, 2, 0, 1, 2)
    panel = rng.uniform(0., 1.2, (5, len(seed), horizon)) * scale
    panel[rng.random(panel.shape) < .25] = 0.
    panel[:, 2, :] = .125 * scale
    mutable = (0, 1, 3, 4, 6, 7) if symmetric else (0, 1, 3, 4)
    family = ResidualFamily(seed, seed, (0, 1), mutable)
    reference = panel_profile(panel, seed, scale)
    full = FamilyScoreBounds(panel, seed, (0, 1), mutable, scale, reference,
                             seed_loads=host_loads(panel, seed))
    coarse = IndicatorBounds(full)
    ranges = [(start, start + span) for span in (1 << bit for bit in range(family.bits + 1))
              for start in range(0, family.mask_count, span)]
    profiles = {mask: panel_profile(panel, family.assignment(mask), scale)
                for mask in family.nonempty_masks()}
    movements = {mask: migration_profile(seed, family.assignment(mask))["count"] for mask in profiles}
    lower_movement = [min((movements[mask] for mask in profiles if start <= mask < stop), default=0)
                      for start, stop in ranges]
    lowers = coarse.lower_profiles(ranges)
    uppers = coarse.upper_keys(ranges, lower_movement, lower_movement)
    assert full.available
    assert all(lower is not None for lower in lowers)
    for (start, stop), lower, upper in zip(ranges, lowers, uppers):
        assert np.all(lower[:, 2] == -1.) and np.all(lower[:, 5] == -1.)
        for mask, profile in profiles.items():
            if start <= mask < stop:
                assert np.all(lower <= profile)
                gains = _relative_gains(reference, profile, PROFILE_NAMES)
                actual = tuple(sorted(int(np.floor(value * (1 << 40))) for value in gains))
                actual += (-movements[mask], -float(movements[mask]))
                assert actual <= upper


def test_idle_source_is_not_reinterpreted_as_positive_native_loss():
    seed = (0, 1, 2, 0, 1, 2)
    panel = np.zeros((3, 6, 8))
    reference = panel_profile(panel, seed, 1.)
    full = FamilyScoreBounds(panel, seed, (0, 1), (0, 1, 3, 4), 1., reference)
    coarse = IndicatorBounds(full)
    assert not full.available
    assert coarse.upper_keys([(0, full.mask_count)]) == (None,)


def test_subnormal_nonidle_source_keeps_unknown_metric_sentinels():
    seed = (0, 1, 2, 0, 1, 2)
    panel = np.full((2, 6, 3), np.nextafter(0., 1.))
    reference = panel_profile(panel, seed, 1.)
    full = FamilyScoreBounds(panel, seed, (0, 1), (0, 1, 3, 4), 1., reference)
    coarse = IndicatorBounds(full)
    lower = coarse.lower_profiles([(0, full.mask_count)])[0]
    assert lower is not None
    family = ResidualFamily(seed, seed, (0, 1), (0, 1, 3, 4))
    for mask in family.nonempty_masks():
        assert np.all(lower <= panel_profile(panel, family.assignment(mask), 1.))


def test_tolerant_capacity_boundary_never_creates_a_false_indicator():
    seed = (0, 1, 2, 0, 1, 2)
    threshold = 1. + 1e-12
    row = np.asarray([np.nextafter(threshold, -np.inf), threshold,
                      np.nextafter(threshold, np.inf), 1.])
    panel = np.zeros((1, 6, 4))
    panel[0, 0] = row
    panel[0, 1] = row[::-1]
    panel[0, 2] = .125
    reference = panel_profile(panel, seed, 1.)
    full = FamilyScoreBounds(panel, seed, (0, 1), (0, 1, 3, 4), 1., reference)
    coarse = IndicatorBounds(full)
    family = ResidualFamily(seed, seed, (0, 1), (0, 1, 3, 4))
    for mask in family.nonempty_masks():
        lower = coarse.lower_profiles([(mask, mask + 1)])[0]
        assert np.all(lower <= panel_profile(panel, family.assignment(mask), 1.))


def test_lru_eviction_preserves_current_request_and_does_not_pin_batches():
    seed = tuple(host for _ in range(6) for host in range(3))
    panel = np.random.default_rng(721).uniform(.01, .5, (3, len(seed), 1))
    mutable = tuple(vm for vm in range(15) if seed[vm] in (0, 1))
    reference = panel_profile(panel, seed, 1.)
    full = FamilyScoreBounds(panel, seed, (0, 1), mutable, 1., reference)
    coarse = IndicatorBounds(full)
    ranges = [(mask, mask + 1) for mask in range(full.mask_count)]
    result = coarse.lower_profiles(ranges)
    assert len(result) == full.mask_count and len(coarse._cache) <= 512
    assert coarse._cache_cells <= 1 << 20
    assert all(value is None or value.base is None for value in coarse._cache.values())
    first = coarse.lower_profiles(ranges[:1])[0]
    assert np.array_equal(first, result[0])


def test_overflowing_projection_is_not_used_as_a_finite_endpoint_certificate():
    seed = (0, 1, 2, 0, 1, 2)
    panel = np.full((2, 6, 8), 1e300)
    full = FamilyScoreBounds(panel, seed, (0, 1), (0, 1, 3, 4),
                             1e-300, np.zeros((2, 6)))
    coarse = IndicatorBounds(full)
    assert coarse.upper_keys([(0, full.mask_count)]) == (None,)


def test_finite_seed_does_not_hide_overflowing_candidate_behind_saturated_lower_sum():
    panel = np.full((1, 4, 1), 7e307)
    seed = (0, 1, 0, 1)
    reference = np.asarray([[1., 1., 1., .4, .4, 0.]])
    full = FamilyScoreBounds(panel, seed, (0, 1), (0, 1), 1e308, reference)
    assert full.available
    family = ResidualFamily(seed, seed, (0, 1), (0, 1))
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises((ValueError, OverflowError, ArithmeticError)):
        panel_profile(panel, family.assignment(0), 1e308)
    coarse = IndicatorBounds(full)
    assert not coarse.available
    assert coarse.upper_keys([(mask, mask + 1) for mask in range(full.mask_count)]) == (None,) * full.mask_count
