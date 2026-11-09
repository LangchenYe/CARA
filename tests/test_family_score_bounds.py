"""Exhaustive leaf oracles for optimistic native residual-family keys."""
import numpy as np
import pytest

from cara.data import canonical_assignment, host_loads
from cara.family_certificates import ResidualFamily
from cara.family_score_bounds import FamilyScoreBounds, FamilyScoreContext
from cara.query_engine import SharedQueryReplay
from cara.refinement import PROFILE_NAMES, _relative_gains, migration_profile, panel_profile


def _ranges(size):
    return tuple((start, start + span) for span in (1 << k for k in range(size.bit_length()))
                 for start in range(0, size, span))


def _key(reference, profile, movement=0, names=PROFILE_NAMES):
    gains = _relative_gains(reference, profile, names)
    return tuple(sorted(int(np.floor(value * (1 << 40))) for value in gains)) + (-movement, -float(movement))


def _check(panel, seed, pair, mutable, capacity, names=PROFILE_NAMES, reference=None):
    seed = canonical_assignment(seed)
    family = ResidualFamily(seed, seed, pair, mutable)
    reference = panel_profile(panel, seed, capacity) if reference is None else reference
    bounds = FamilyScoreBounds(panel, seed, pair, mutable, capacity, reference,
                               objective_names=names, seed_loads=host_loads(panel, seed))
    ranges = _ranges(family.mask_count)
    lower = bounds.lower_profiles(ranges)
    valid = family.nonempty_masks()
    profiles = {mask: panel_profile(panel, family.assignment(mask), capacity) for mask in valid}
    moves = {mask: migration_profile(seed, family.assignment(mask))["count"] for mask in valid}
    movement_lowers = [min((moves[mask] for mask in valid if start <= mask < stop), default=0)
                       for start, stop in ranges]
    uppers = bounds.upper_keys(ranges, movement_lowers, movement_lowers)
    checked = 0
    for (start, stop), loss_lower, upper in zip(ranges, lower, uppers):
        if loss_lower is None:
            assert upper is None
            continue
        checked += 1
        for mask in valid:
            if start <= mask < stop:
                actual = profiles[mask]
                assert np.all(loss_lower <= actual), (start, stop, mask, loss_lower - actual)
                actual_key = _key(reference, actual, moves[mask], names)
                assert upper >= actual_key, (start, stop, mask, upper, actual_key)
    return bounds, checked, uppers


@pytest.mark.parametrize("scenes,horizon", [(1, 1), (1, 8), (3, 3), (4, 8), (7, 9)])
@pytest.mark.parametrize("symmetric", [False, True])
def test_exhaustive_random_profiles_and_complete_keys(scenes, horizon, symmetric):
    rng = np.random.default_rng(431 + scenes + horizon)
    seed = (0, 1, 2, 0, 1, 2, 0, 1, 2)
    panel = rng.uniform(0, 1.5, (scenes, len(seed), horizon))
    panel[rng.random(panel.shape) < .25] = 0.
    panel[:, 2, :] = .125  # Every scene is nonidle.
    mutable = (0, 1, 3, 4, 6, 7) if symmetric else (0, 1, 3, 4)
    bounds, checked, _ = _check(panel, seed, (0, 1), mutable, 1.)
    assert bounds.available
    assert checked > 0


@pytest.mark.parametrize("horizon", [1, 3, 8])
def test_two_host_empty_common_projection(horizon):
    panel = np.random.default_rng(12).uniform(.1, 2., (3, 8, horizon))
    bounds, checked, _ = _check(panel, (0, 1) * 4, (1, 0), tuple(range(8)), 2.)
    assert bounds.available and checked


def test_common_j_plateau_retains_exact_quantized_zero():
    panel = np.full((5, 6, 8), .1)
    panel[:, 2, :] = 4.
    seed = (0, 1, 2, 0, 1, 2)
    reference = panel_profile(panel, seed, 1.)
    bounds = FamilyScoreBounds(panel, seed, (0, 1), (0, 1, 3, 4), 1., reference)
    lower = bounds.lower_profiles(((0, bounds.mask_count),))[0]
    assert np.array_equal(lower[:, 0], np.ones(5))
    gains = _relative_gains(reference, lower, PROFILE_NAMES)
    assert gains[0] == 0.
    assert int(np.floor(gains[0] * (1 << 40))) == 0


def test_zero_reference_j_plateau_also_stays_exact():
    panel = np.full((3, 6, 8), .01)
    seed = (0, 1, 2, 0, 1, 2)
    reference = panel_profile(panel, seed, 10.)
    bounds = FamilyScoreBounds(panel, seed, (0, 1), (0, 1, 3, 4), 10., reference)
    lower = bounds.lower_profiles(((0, bounds.mask_count),))[0]
    assert lower is not None
    assert _relative_gains(reference, lower, PROFILE_NAMES)[0] == 0.


def test_capacity_boundary_and_native_tolerance_are_distinct():
    panel = np.asarray([[[np.nextafter(1., -np.inf), 1., np.nextafter(1., np.inf), 1. + 2e-12]],
                        [[.25, .5, .75, .125]], [[.125, .25, .5, .75]],
                        [[0., 1e-14, 1e-12, 3e-12]], [[.25, 0., .5, 0.]],
                        [[.5, .25, .125, .75]]]).transpose(1, 0, 2)
    _, checked, _ = _check(panel, (0, 1, 2, 0, 1, 2), (0, 1), (0, 1, 3, 4), 1.)
    assert checked


@pytest.mark.parametrize("scale", [2.**-600, 2.**-100, 1., 2.**300])
def test_wide_finite_exponents(scale):
    rng = np.random.default_rng(93)
    panel = rng.uniform(.01, 2., (3, 8, 8)) * scale
    _, checked, _ = _check(panel, (0, 1, 2, 0, 1, 2, 0, 1), (0, 1), (0, 1, 3, 4), scale)
    assert checked


def test_signed_row_cancellation_objective_order_and_strided_reference():
    rng = np.random.default_rng(109)
    panel = rng.uniform(.05, .8, (17, 8, 8))
    seed = (0, 1, 2, 0, 1, 2, 0, 1)
    ordinary = panel_profile(panel, seed, 1.)
    larger = np.repeat(ordinary, 2, axis=0)
    reference = larger[::2]  # Preserve the actual reference's reduction layout.
    reference[1::2, 2:] += np.spacing(reference[1::2, 2:])
    names = ("PAR", "raw", "J", "VS", "V", "VR")
    _, checked, _ = _check(panel, seed, (0, 1), (0, 1, 3, 4), 1., names, reference)
    assert checked


@pytest.mark.parametrize("only_one_idle", [False, True])
def test_idle_sentinels_disable_instead_of_using_zero(only_one_idle):
    panel = np.zeros((3, 6, 8))
    if only_one_idle:
        panel[1:] = .5
    seed = (0, 1, 2, 0, 1, 2)
    reference = panel_profile(panel, seed, 1.)
    assert reference[0, 2] == reference[0, 5] == -1.
    bounds = FamilyScoreBounds(panel, seed, (0, 1), (0, 1, 3, 4), 1., reference)
    assert not bounds.available
    assert "idle" in bounds.disabled_reason
    assert bounds.upper_key(0, bounds.mask_count) is None


def test_subnormal_bounds_never_create_positive_idle_lower_loss():
    tiny = np.nextafter(0., 1.)
    panel = np.full((2, 6, 3), tiny)
    seed = (0, 1, 2, 0, 1, 2)
    bounds, _, _ = _check(panel, seed, (0, 1), (0, 1, 3, 4), 1.)
    # A conservative inability to prove positive stored PAR denominator is a
    # supported exact-refinement fallback, not a fabricated PAR=0 profile.
    assert all(value is None or np.isfinite(value).all()
               for value in bounds.lower_profiles(_ranges(bounds.mask_count)))


def test_overflowing_native_projection_requires_refinement():
    panel = np.full((2, 6, 8), 1e300)
    seed = (0, 1, 2, 0, 1, 2)
    bounds = FamilyScoreBounds(panel, seed, (0, 1), (0, 1, 3, 4), 1e-300,
                               np.zeros((2, 6)))
    assert bounds.upper_key(0, bounds.mask_count) is None


def test_negative_nonfinite_and_invalid_ranges_are_not_certified():
    seed = (0, 1, 2, 0, 1, 2)
    panel = np.ones((2, 6, 8))
    reference = panel_profile(panel, seed, 1.)
    for value in (-1., np.inf, np.nan):
        invalid = panel.copy()
        invalid[0, 0, 0] = value
        bounds = FamilyScoreBounds(invalid, seed, (0, 1), (0, 1, 3, 4), 1., reference)
        assert bounds.upper_key(0, bounds.mask_count) is None
    bounds = FamilyScoreBounds(panel, seed, (0, 1), (0, 1, 3, 4), 1., reference)
    for interval in ((1, 3), (0, 3), (0.5, 1.5), (-1, 1), (0, 17)):
        with pytest.raises(ValueError):
            bounds.upper_key(*interval)


def test_strict_key_rule_never_discards_equal_key():
    panel = np.full((2, 6, 8), .1)
    seed = (0, 1, 2, 0, 1, 2)
    reference = panel_profile(panel, seed, 1.)
    bounds = FamilyScoreBounds(panel, seed, (0, 1), (0, 1, 3, 4), 1., reference)
    upper = bounds.upper_key(0, bounds.mask_count)
    assert upper is not None
    assert not (upper < upper)
    with pytest.raises(ValueError):
        bounds.upper_keys(((0, 1),), (0, 1))
    assert bounds.upper_key(0, 1, cost_lower_bound=np.inf) is None


@pytest.mark.parametrize("order", ["C", "F"])
def test_reader_context_preserves_bounds_and_original_reference_scale(order):
    rng = np.random.default_rng(401)
    panel = rng.uniform(.05, .8, (121, 9, 8))
    seed = (0, 1, 2) * 3
    reference = np.array(panel_profile(panel, seed, 1.), order=order)
    names = ("PAR", "raw", "J", "VS", "V", "VR")
    reader = SharedQueryReplay(panel, seed, 1.)
    context = FamilyScoreContext.from_reader(reader, reference, objective_names=names)
    for pair, mutable in (((0, 1), (0, 1, 3, 4)), ((1, 2), (1, 2, 4, 5))):
        loads = reader.loads(seed)
        ordinary = FamilyScoreBounds(panel, seed, pair, mutable, 1., reference,
                                     objective_names=names, seed_loads=loads)
        reused = context.for_family(seed, pair, mutable, seed_loads=loads)
        ranges = _ranges(reused.mask_count)
        assert ordinary.upper_keys(ranges) == reused.upper_keys(ranges)
        for direct, cached in zip(ordinary.lower_profiles(ranges), reused.lower_profiles(ranges)):
            assert direct is not None and cached is not None
            assert direct.tobytes() == cached.tobytes()


def test_context_snapshots_reference_and_accepts_fork_registry():
    panel = np.random.default_rng(124).uniform(.1, .8, (3, 9, 8))
    seed = (0, 1, 2) * 3
    reader = SharedQueryReplay(panel, seed, 1.)
    reference = panel_profile(panel, seed, 1.)
    expected_reference = reference.copy()
    fork = reader.fork((0, 1, 2, 1, 0, 2, 0, 1, 2))
    context = FamilyScoreContext.from_reader(fork, reference)
    owner_context = FamilyScoreContext.from_registry(reader._registry, reference)
    reference[:] = 123.
    assert np.array_equal(context.reference, expected_reference)
    for candidate_seed in (seed, fork.seed):
        pair, mutable = (0, 1), (0, 1, 3, 4)
        one = context.for_family(candidate_seed, pair, mutable)
        two = owner_context.for_family(candidate_seed, pair, mutable)
        assert one.upper_keys(_ranges(one.mask_count)) == two.upper_keys(_ranges(two.mask_count))


def test_context_does_not_bypass_reader_source_contract_or_idle_guard():
    panel = np.zeros((3, 6, 8))
    seed = (0, 1, 2) * 2
    reader = SharedQueryReplay(panel, seed, 1.)
    reference = panel_profile(panel, seed, 1.)
    with pytest.raises(TypeError):
        FamilyScoreContext.from_reader(object(), reference)
    with pytest.raises(ValueError):
        FamilyScoreContext.from_reader(reader, np.zeros((2, 6)))
    context = FamilyScoreContext.from_reader(reader, reference)
    bounds = context.for_family(seed, (0, 1), (0, 1, 3, 4))
    assert not bounds.available
    assert "idle" in bounds.disabled_reason
    assert bounds.upper_key(0, bounds.mask_count) is None
    foreign = reader.fork(seed)
    foreign.panel = panel.copy()
    with pytest.raises(ValueError):
        FamilyScoreContext.from_reader(foreign, reference)


def test_context_nonfinite_reference_still_falls_back():
    panel = np.ones((3, 6, 8))
    seed = (0, 1, 2) * 2
    reader = SharedQueryReplay(panel, seed, 1.)
    reference = panel_profile(panel, seed, 1.)
    reference[0, 2] = np.inf
    context = FamilyScoreContext.from_reader(reader, reference)
    bounds = context.for_family(seed, (0, 1), (0, 1, 3, 4))
    assert bounds.upper_key(0, bounds.mask_count) is None


def test_finite_reference_with_overflowing_mean_requires_refinement():
    panel = np.ones((3, 6, 8))
    seed = (0, 1, 2) * 2
    reference = panel_profile(panel, seed, 1.)
    reference[:, 3] = np.finfo(float).max
    assert np.isfinite(reference).all()
    ordinary = FamilyScoreBounds(panel, seed, (0, 1), (0, 1, 3, 4), 1., reference)
    reader = SharedQueryReplay(panel, seed, 1.)
    context = FamilyScoreContext.from_reader(reader, reference)
    reused = context.for_family(seed, (0, 1), (0, 1, 3, 4))
    for bounds in (ordinary, reused):
        assert not bounds.available
        assert "reference scale" in bounds.disabled_reason
        assert bounds.upper_key(0, bounds.mask_count) is None
