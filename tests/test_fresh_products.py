"""Exact arithmetic must remain unbounded when values come from saved arrays."""
from fractions import Fraction
import numpy as np
from cara.admission import GRID, M, _product_tree, _wealth_parts, certify_fresh


def test_numpy_integer_inputs_do_not_overflow_exact_products():
    assert _product_tree([np.int64(2**40)] * 3) == 2**120
    counts = np.array([M // 2, -(M // 3), M // 4], dtype=np.int64)
    numerator, denominator = _wealth_parts(counts, np.int64(M + 1), np.int64(255))
    expected = Fraction(1)
    for count in counts:
        expected *= 1 + Fraction(255, GRID) * Fraction(int(count), M + 1)
    assert Fraction(numerator, denominator) == expected


def test_fixed_and_stopped_kernel_admit_the_same_strict_improvement():
    reference = np.tile([1., .9, .9, .9], (64, 1))
    candidate = np.tile([0., .1, .1, .1], (64, 1))
    fixed = certify_fresh(reference, candidate, [255] * 4, batch=64)
    stopped = certify_fresh(reference, candidate, [255] * 4, batch=1)
    assert fixed['admitted'] and stopped['admitted']
    assert fixed['rows_read'] == 64 and stopped['rows_read'] == 7
