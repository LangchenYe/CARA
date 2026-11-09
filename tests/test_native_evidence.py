"""Native execution keeps the scalar filter's interval bytes and decisions."""
from fractions import Fraction

import numpy as np

from cara.action_query import _GainIntervalMixture
from cara.native_evidence import NativeGainIntervalMixture


def test_native_rates_have_the_identical_directed_fold_and_component_growth():
    rng = np.random.default_rng(66571)
    data = [(float(x), float(x)) for x in rng.uniform(-1., 1., 289)]
    for step in (1, 19, 32, 73):
        scalar, native = _GainIntervalMixture(), NativeGainIntervalMixture()
        for start in range(0, len(data), step):
            values = data[start:start + step]
            scalar.update(values);native.update(values)
            assert scalar.high == native.high and scalar.small == native.small
            assert scalar.bounds() == native.bounds()
    for value in (-1., 1.):
        scalar, native = _GainIntervalMixture(), NativeGainIntervalMixture()
        values = [(value, value)] * 1100
        scalar.update(values);native.update(values)
        assert scalar.high == native.high and scalar.small == native.small
        assert scalar.bounds() == native.bounds()


def test_native_filter_keeps_exact_equality_refinement():
    from cara.certification import _DyadicMixture
    truth = [Fraction(1, 3)] * 7
    exact = _DyadicMixture();exact.update(truth)
    numerator, denominator = exact.parts()
    native = NativeGainIntervalMixture();native.update([(-.5, .5)] * len(truth))
    assert native.crossed(Fraction(denominator, numerator), lambda: truth)
    assert native.exact_fallbacks == 1
