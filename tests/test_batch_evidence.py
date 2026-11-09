"""Batched filter enclosures, component growth, and exact boundary fallback."""
from fractions import Fraction

import numpy as np

from cara.batch_evidence import BatchGainIntervalMixture
from cara.certification import _DyadicMixture


def test_matrix_products_enclose_rational_mixtures_across_request_partitions():
    rng = np.random.default_rng(31841)
    truth = [Fraction(int(v), 1001) for v in rng.integers(-1001, 1002, 289)]
    intervals = [(np.nextafter(float(v), -np.inf), np.nextafter(float(v), np.inf)) for v in truth]
    intervals = [(max(-1., l), min(1., u)) for l, u in intervals]
    for step in (1, 19, 32, 73):
        process, exact = BatchGainIntervalMixture(), _DyadicMixture()
        for start in range(0, len(truth), step):
            process.update(intervals[start:start + step]); exact.update(truth[start:start + step])
            low, high = process.bounds(); numerator, denominator = exact.parts()
            assert Fraction(low) <= Fraction(numerator, denominator) <= Fraction(high)


def test_batch_filter_preserves_extreme_products_and_exact_equality_decisions():
    for value in (-1., 1.):
        process, exact = BatchGainIntervalMixture(), _DyadicMixture()
        truth = [Fraction(value)] * 1100
        process.update([(value, value)] * len(truth)); exact.update(truth)
        low, high = process.bounds(); numerator, denominator = exact.parts()
        assert Fraction(low) <= Fraction(numerator, denominator)
        assert np.isinf(high) or Fraction(numerator, denominator) <= Fraction(high)
        assert process.crossed(Fraction(1, 20), lambda: truth) == (value > 0)
    process, exact = BatchGainIntervalMixture(), _DyadicMixture()
    truth = [Fraction(1, 3)] * 7
    exact.update(truth); numerator, denominator = exact.parts()
    process.update([(-.5, .5)] * len(truth))
    assert process.crossed(Fraction(denominator, numerator), lambda: truth)
    assert process.exact_fallbacks == 1


def test_batch_query_certificate_keeps_original_locks_stops_and_actions():
    from cara.action_query import ActionSceneReader, certify_action_queries
    from cara.migration_envelope import stored_action_gain_bounds
    from cara.refinement import migration_profile
    from cara.synthetic import LAW, make_environment
    environment = make_environment(205001)
    old = tuple(np.arange(LAW["vms"]) % 8); new = list(old)
    new[0], new[1] = new[1], new[0]
    aligned = migration_profile(old, new)["aligned_assignment"]
    options = dict(gain_bounds=stored_action_gain_bounds(old, aligned, LAW["capacity"],
        horizon=LAW["horizon"])["gain_bounds"], noninferiority_margins=(.01, .01, 0., .001))
    results = []
    for engine in ("scalar", "batch"):
        reader = ActionSceneReader(environment, "complementary", 15, old, new,
            LAW["capacity"], horizon=LAW["horizon"], execution="global")
        result = certify_action_queries(reader, 512, evidence_engine=engine, **options)
        results.append({k: v for k, v in result.items() if k not in
            ("query_stats", "reader_stats", "exact_filter_fallbacks")})
    assert results[0] == results[1]
