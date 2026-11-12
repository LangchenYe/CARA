"""Real-rational oracles for factorized certificate sums."""
from fractions import Fraction
import numpy as np
import pytest

from cara.family_score_bounds import _real_lower, _real_upper
from cara.subset_certificates import MutableSumIndex


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("force_index", [False, True])
@pytest.mark.parametrize("scale", [2.**-100, 1., 2.**200])
def test_every_dyadic_sum_encloses_its_exact_mutable_sum(symmetric, force_index, scale):
    values = np.random.default_rng(6311).uniform(.01, 1., (6, 2, 3)) * scale
    index = MutableSumIndex(values, symmetric=symmetric,
                            minimum_ranges=1 if force_index else 1 << 20)
    ranges = [(start, start + span) for span in (1 << k for k in range(index.bits + 1))
              for start in range(0, index.mask_count, span)]
    mandatory, possible = index.sum_ranges(ranges)
    for kind, raw in [(0, mandatory), (1, possible)]:
        lower, upper = _real_lower(raw, len(values)), _real_upper(raw, len(values))
        for row, (start, stop) in enumerate(ranges):
            free = (stop - start).bit_length() - 1
            groups = [([0] if symmetric else []), []]
            for bit in range(index.bits):
                j = bit + int(symmetric)
                if bit < free:
                    if kind:
                        groups[0].append(j)
                        groups[1].append(j)
                else:
                    groups[(start >> bit) & 1].append(j)
            for side, residents in enumerate(groups):
                for b in range(values.shape[1]):
                    for t in range(values.shape[2]):
                        exact = sum((Fraction(float(values[j, b, t])) for j in residents), Fraction())
                        assert Fraction(float(lower[row, side, b, t])) <= exact
                        assert exact <= Fraction(float(upper[row, side, b, t]))
    assert (index.tables is not None) == force_index


@pytest.mark.parametrize("force_index", [False, True])
def test_leaf_mandatory_and_possible_share_the_same_array(force_index):
    values = np.full((8, 2, 3), .1)
    index = MutableSumIndex(values, minimum_ranges=1 if force_index else 1 << 20)
    mandatory, possible = index.sum_ranges([(mask, mask + 1) for mask in range(32)])
    assert mandatory is possible
    assert index.sum_ranges([(0, 1)], possible=False).shape == (1, 2, 2, 3)


def test_wide_relation_is_factorized_and_memory_guarded():
    values = np.random.default_rng(2).uniform(0, 1, (16, 2, 3))
    index = MutableSumIndex(values, minimum_ranges=1)
    index.sum_ranges([(0, 1 << 16)])
    assert index.stats["mutable_sum_half_rows"] == 512
    assert index.stats["mutable_sum_bytes"] == 512 * 2 * 3 * 8
    guarded = MutableSumIndex(values, minimum_ranges=1, max_bytes=0)
    guarded.sum_ranges([(0, 1 << 16)])
    assert guarded.tables is None


def test_zero_and_subnormal_inputs_remain_enclosed():
    values = np.full((8, 1, 3), np.nextafter(0., 1.))
    values[::2] = 0.
    index = MutableSumIndex(values, minimum_ranges=1)
    mandatory, possible = index.sum_ranges([(0, 256), (16, 32), (17, 18)])
    assert np.isfinite(mandatory).all() and np.isfinite(possible).all()
    assert np.all(mandatory >= 0) and np.all(possible >= 0)


def test_default_construction_gate_counts_work_and_never_target_values():
    values = np.full((16, 1, 2), .1)
    index = MutableSumIndex(values)
    assert index.minimum_ranges == 256
    request = [(mask, mask + 1) for mask in range(32)]
    for _ in range(7):
        index.sum_ranges(request)
        assert index.tables is None
    index.sum_ranges(request)
    assert index.tables is not None


def test_native_lookup_reports_overflow_without_wrapping_to_a_finite_sum():
    values = np.full((4, 1, 1), np.finfo(float).max / 2)
    index = MutableSumIndex(values, minimum_ranges=1)
    mandatory, possible = index.sum_ranges([(0, 16)])
    assert np.all(mandatory == 0.)
    assert np.isinf(possible).all()


def test_symmetry_fixed_53_resident_family_retains_selector_fallback():
    from cara.family_score_bounds import FamilyScoreContext
    from cara.query_engine import SharedQueryReplay

    panel = np.full((1, 53, 1), .001)
    seed, mutable = tuple(i % 2 for i in range(53)), tuple(range(53))
    reader = SharedQueryReplay(panel, seed, 1.)
    reference = reader.profile(seed)
    compiled = FamilyScoreContext.from_reader(reader, reference, compiled_subsets=True)
    ordinary = FamilyScoreContext.from_reader(reader, reference)
    bound = compiled.for_family(seed, (0, 1), mutable)
    expected = ordinary.for_family(seed, (0, 1), mutable)
    assert bound.available and bound.symmetric and bound.bits == 52
    assert bound.mutable_sum_index is None
    assert np.array_equal(bound.lower_profiles([(1, 2)])[0],
                          expected.lower_profiles([(1, 2)])[0])
    assert bound.upper_key(1, 2) == expected.upper_key(1, 2)
