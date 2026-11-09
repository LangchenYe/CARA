"""Bitwise checks against the unchanged native scalar relative-gain oracle."""
import numpy as np
import pytest

from cara.gain_circuit import GainsContext
from cara.refinement import PROFILE_NAMES, _relative_gains


def _assert_same_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.float64
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("scenes", [1, 2, 3, 7, 8, 9, 63, 64, 65, 120, 121, 122,
                                     255, 256, 257, 513, 2049])
def test_batch_matches_scalar_across_reduction_lengths_and_layouts(scenes):
    rng = np.random.default_rng(872316 + scenes)
    for kind in ("ordinary", "large_range", "cancellation", "subnormal"):
        reference = rng.uniform(-1, 4, (scenes, 6))
        candidates = rng.uniform(-1, 4, (3, scenes, 6))
        if kind == "large_range":
            reference = np.ldexp(reference, rng.integers(-900, 900, reference.shape))
            candidates = np.ldexp(candidates, rng.integers(-900, 900, candidates.shape))
        elif kind == "cancellation":
            reference = np.maximum(reference, 0.)
            candidates = np.broadcast_to(reference, (3, scenes, 6)).copy()
            candidates = np.nextafter(candidates,
                np.where(rng.random(candidates.shape) < .5, -np.inf, np.inf))
        elif kind == "subnormal":
            reference *= np.nextafter(0., 1.) * 128
            candidates *= np.nextafter(0., 1.) * 128
        # Neither input is contiguous; the scalar oracle's reference mean
        # must nevertheless keep its original layout before selecting axes.
        reference_storage = np.empty((scenes * 2, 12))
        reference_storage[::2, ::2] = reference
        reference = reference_storage[::2, ::2]
        candidate_storage = np.empty((3, scenes * 2, 12))
        candidate_storage[:, ::2, ::2] = candidates
        candidates = candidate_storage[:, ::2, ::2]
        for names in (PROFILE_NAMES, ("VR", "raw", "VS"), ("PAR", "J", "VS", "V"), ("J",)):
            context = GainsContext(reference, names)
            with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
                expected = np.stack([_relative_gains(reference, row, names) for row in candidates])
            _assert_same_bits(context.batch(candidates), expected)
            for row, want in zip(candidates, expected):
                _assert_same_bits(context.scalar(row), want)


@pytest.mark.parametrize("order", ["C", "F"])
def test_reference_layout_is_used_before_packing_and_context_snapshots(order):
    rng = np.random.default_rng(167)
    reference = np.array(np.ldexp(rng.uniform(-1, 1, (121, 6)),
                                 rng.integers(-30, 30, (121, 6))), order=order)
    candidates = rng.uniform(-1, 1, (5, 121, 6))
    names = ("PAR", "raw", "J", "VS", "V", "VR")
    expected = np.stack([_relative_gains(reference, row, names) for row in candidates])
    context = GainsContext(reference, names)
    reference[:] = 123.
    _assert_same_bits(context.batch(candidates), expected)


def test_idle_sentinels_zero_coordinates_and_duplicate_objectives():
    reference = np.zeros((121, 6))
    reference[:, (2, 5)] = -1.
    candidates = np.stack((reference, np.nextafter(reference, np.inf),
                            np.nextafter(reference, -np.inf)))
    names = ("PAR", "J", "VR", "J")
    context = GainsContext(reference, names)
    expected = np.stack([_relative_gains(reference, row, names) for row in candidates])
    _assert_same_bits(context.batch(candidates), expected)
    assert np.array_equal(context.batch(candidates)[0], np.zeros(4))


def test_nonfinite_results_are_not_clipped_to_finite_gains():
    reference = np.ones((9, 6))
    candidates = np.zeros((3, 9, 6))
    candidates[0, 0, 2] = np.inf
    candidates[1, 0, 3] = np.nan
    candidates[2, :, 5] = -np.finfo(float).max
    context = GainsContext(reference)
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        expected = np.stack([_relative_gains(reference, row, PROFILE_NAMES) for row in candidates])
    actual = context.batch(candidates)
    _assert_same_bits(actual, expected)
    assert np.isneginf(actual[0, 2])
    assert np.isnan(actual[1, 3])
    assert np.isposinf(actual[2, 5])


def test_shapes_empty_batch_and_invalid_objectives():
    context = GainsContext(np.zeros((3, 6)), ("J",))
    assert context.batch(np.empty((0, 3, 6))).shape == (0, 1)
    for reference in (np.empty((0, 6)), np.zeros(6), np.zeros((3, 5))):
        with pytest.raises(ValueError):
            GainsContext(reference)
    for names in ((), ("unknown",)):
        with pytest.raises(ValueError):
            GainsContext(np.zeros((3, 6)), names)
    for profile in (np.zeros((4, 6)), np.zeros((3, 5)), np.zeros((1, 3, 6))):
        with pytest.raises(ValueError):
            context.scalar(profile)
    with pytest.raises(ValueError):
        context.batch(np.zeros((3, 6)))
