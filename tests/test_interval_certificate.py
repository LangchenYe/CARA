from fractions import Fraction

import numpy as np

from cara.certification import _DyadicMixture, certify_streaming_mixture
from cara.interval_certificate import _IntervalDyadicMixture, certify_streaming_interval


def test_interval_encloses_exact_partial_mixture_and_growing_components():
    rng = np.random.default_rng(917)
    exact, interval = _DyadicMixture(2), _IntervalDyadicMixture(2)
    for count in (1, 2, 5, 11, 32, 87):
        values = [Fraction(float(value)) for value in rng.uniform(-1, 1, count)]
        exact.update(values)
        interval.update(values)
        numerator, denominator = exact.parts()
        lo, hi = interval.bounds()
        assert Fraction(lo) <= Fraction(numerator, denominator) <= Fraction(hi)
        assert set(interval.small) == set(exact.small)
        assert interval.rows == exact.rows
        assert interval.crossed(Fraction(1, 20)) == exact.crossed(Fraction(1, 20))


def test_exact_boundary_reconstructs_integer_reference():
    values = [Fraction(1, 3), Fraction(-1, 7), Fraction(2, 5)]
    exact, interval = _DyadicMixture(), _IntervalDyadicMixture()
    exact.update(values)
    interval.update(values)
    numerator, denominator = exact.parts()
    alpha = Fraction(denominator, numerator)
    assert 0 < alpha < 1
    assert exact.crossed(alpha) is True
    assert interval.crossed(alpha) is True
    assert interval.exact_fallbacks == 1
    assert interval.crossed(alpha - Fraction(1, 10**100)) is False
    assert interval.exact_fallbacks == 2


def test_products_beyond_binary_float_range_and_large_rational_denominators():
    for value, count in ((Fraction(-1), 2048), (Fraction(1), 2048),
                         (Fraction(10**120 - 1, 10**120 + 1), 13)):
        exact, interval = _DyadicMixture(), _IntervalDyadicMixture()
        values = [value] * count
        exact.update(values)
        interval.update(values)
        numerator, denominator = exact.parts()
        lo, hi = interval.bounds()
        assert Fraction(lo) <= Fraction(numerator, denominator) <= Fraction(hi)
        assert interval.crossed(Fraction(1, 20)) == exact.crossed(Fraction(1, 20))


def test_streaming_interval_is_semantically_identical_on_varied_paired_tables():
    rng = np.random.default_rng(5124)
    for shift in (-.03, 0., .015, .15):
        reference = rng.uniform(.2, .8, (513, 4))
        candidate = np.clip(reference - shift + rng.uniform(-.06, .06, reference.shape), 0., 1.)
        for batch in (1, 32, 513):
            calls = [[], []]
            def reader(which):
                def acquire(indices):
                    calls[which].extend(indices.tolist())
                    return reference[indices], candidate[indices]
                return acquire
            options = dict(gain_bounds=(1., .4, .7, 1.), noninferiority_margins=(.02, .03, .04, .05),
                           anchor=2, min_anchor_gain=.005, batch=batch, metric_names=("J", "V", "R", "u"))
            exact = certify_streaming_mixture(reader(0), len(reference), 4, **options)
            interval = certify_streaming_interval(reader(1), len(reference), 4, **options)
            assert interval == exact
            assert calls[0] == calls[1]
