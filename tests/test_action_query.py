"""Local gain enclosures and canonical-mixture decision equivalence."""
from fractions import Fraction

import numpy as np

from cara.action_query import (ActionQueryPlan, ActionSceneReader, _GainIntervalMixture,
    _outward, _sum_error, canonical_fresh_losses, certify_action_queries)
from cara.certification import _DyadicMixture, certify_fresh_mixture
from cara.migration_envelope import stored_action_gain_bounds
from cara.refinement import migration_profile
from cara.synthetic import LAW, make_environment


def test_gain_enclosure_at_fixed_batch_shapes_and_numeric_boundaries():
    rng = np.random.default_rng(1317)
    for b, t, h in ((1, 1, 3), (1, 8, 4), (32, 1, 5), (32, 8, 8)):
        n = h * 4
        old = tuple(np.arange(n) % h)
        new = list(old)
        new[0], new[1] = new[1], new[0]
        panel = rng.uniform(0., 1., (b, n, t))
        panel[:, :3, 0] = (1., 2. ** -53, np.nextafter(0., 1.))
        plan = ActionQueryPlan(old, new, 1., t)
        lower, upper = plan.evaluate(panel)
        reference = canonical_fresh_losses(panel, old, 1.)
        action = canonical_fresh_losses(panel, new, 1.)
        for row in range(b):
            for e in range(4):
                exact = Fraction(float(reference[row, e])) - Fraction(float(action[row, e]))
                assert Fraction(float(lower[row, e])) <= exact <= Fraction(float(upper[row, e]))
        assert np.array_equal(lower[:, 0], reference[:, 0] - action[:, 0])
        assert np.array_equal(upper[:, 0], lower[:, 0])
        glo, ghi = plan.evaluate_global(panel)
        for row in range(b):
            for e in range(4):
                exact = Fraction(float(reference[row, e])) - Fraction(float(action[row, e]))
                assert Fraction(float(glo[row, e])) <= exact <= Fraction(float(ghi[row, e]))


def test_common_host_witness_reads_only_unresolved_scene_rows():
    old = (0, 0, 1, 1, 2, 2, 3, 3)
    new = (0, 1, 0, 1, 2, 2, 3, 3)
    panel = np.zeros((32, 8, 3))
    patterns = ((.6, .6, .2, .2), (.6, .6, .2, .2),
                (.6, .2, .6, .2), (.6, .6, .6, .6))
    for row in range(32):
        panel[row, :4, :] = np.asarray(patterns[row % 4])[:, None]
        if row % 4 == 1:
            panel[row, 4:6, :] = .6
        if row % 4 == 2:
            panel[row, 6:8, :] = .6
    plan = ActionQueryPlan(old, new, 1., 3)
    lower, upper = plan.evaluate(panel, active=(True, False, False, False))
    reference = canonical_fresh_losses(panel, old, 1.)
    action = canonical_fresh_losses(panel, new, 1.)
    assert np.array_equal(lower[:, 0], reference[:, 0] - action[:, 0])
    assert np.array_equal(lower, upper)
    assert 0 < plan.stats["common_operand_cells"] < 32 * 4 * 3
    before = plan.stats["common_operand_cells"]
    plan.evaluate(panel, active=(False, True, True, True))
    assert plan.stats["common_operand_cells"] == before


def test_binary_filter_resolves_exact_boundary_by_prefix_fallback():
    exact = _DyadicMixture()
    truth = [Fraction(1, 3)] * 7
    exact.update(truth)
    numerator, denominator = exact.parts()
    alpha = Fraction(denominator, numerator)
    assert 0 < alpha < 1
    process = _GainIntervalMixture()
    process.update([(-.5, .5)] * len(truth))
    assert process.crossed(alpha, lambda: truth)
    assert process.exact_fallbacks == 1


def test_binary_interval_products_cover_exact_mixtures():
    rng = np.random.default_rng(8021)
    for value in (-1., 1.):
        process, exact = _GainIntervalMixture(), _DyadicMixture()
        values = [Fraction(value)] * 160
        process.update([(value, value)] * len(values))
        exact.update(values)
        low, high = process.bounds()
        num, den = exact.parts()
        result = Fraction(num, den)
        assert Fraction(low) <= result
        assert np.isinf(high) or result <= Fraction(high)
    process, exact = _GainIntervalMixture(), _DyadicMixture()
    for _ in range(9):
        values = [Fraction(float(x)) for x in rng.uniform(-1., 1., 32)]
        process.update([(float(x), float(x)) for x in values])
        exact.update(values)
        low, high = process.bounds()
        num, den = exact.parts()
        assert Fraction(low) <= Fraction(num, den) <= Fraction(high)


def test_same_certification_fields_and_stopping_as_canonical_reader():
    from cara.scene_stream import sample_stream
    environment = make_environment(246001)
    old = tuple(np.arange(LAW["vms"]) % 8)
    new = list(old)
    new[0], new[1] = new[1], new[0]
    move = migration_profile(old, new)
    bounds = stored_action_gain_bounds(old, move["aligned_assignment"], LAW["capacity"],
                                      horizon=LAW["horizon"])["gain_bounds"]
    panel = sample_stream(environment, "complementary", 96, 15)
    reference = canonical_fresh_losses(panel, old, LAW["capacity"])
    action = canonical_fresh_losses(panel, new, LAW["capacity"])
    expected = certify_fresh_mixture(reference, action, gain_bounds=bounds,
        noninferiority_margins=(.01, .01, 0., .001), metric_names=("J", "V", "R", "raw"))
    for execution in ("local", "global", "adaptive"):
        reader = ActionSceneReader(environment, "complementary", 15, old, new, LAW["capacity"],
                                   horizon=LAW["horizon"], execution=execution)
        actual = certify_action_queries(reader, 96, gain_bounds=bounds,
            noninferiority_margins=(.01, .01, 0., .001), metric_names=("J", "V", "R", "raw"))
        assert all(actual[key] == value for key, value in expected.items())
        assert reader.rows == actual["rows"]


def test_empty_action_has_zero_gain_and_no_operand_reads():
    old = (0, 0, 1, 1)
    plan = ActionQueryPlan(old, old, 1., 2)
    lower, upper = plan.evaluate(np.ones((3, 4, 2)))
    assert not lower.any() and not upper.any()
    assert plan.stats["changed_operand_cells"] == plan.stats["common_operand_cells"] == 0


def test_common_raw_values_do_not_cancel_in_stored_global_reduction():
    old = tuple(np.arange(16) % 8)
    new = list(old)
    new[0] = 3  # A zero-demand departure changes the canonical host ordering.
    panel = np.zeros((1, 16, 1))
    panel[0, 1, 0] = 3.
    panel[0, (2, 4), 0] = 1. + 2. ** -52
    plan = ActionQueryPlan(old, new, 1., 1, demand_upper=3.)
    lower, upper = plan.evaluate(panel)
    reference = canonical_fresh_losses(panel, old, 1.)
    action = canonical_fresh_losses(panel, new, 1.)
    gain = Fraction(float(reference[0, 3])) - Fraction(float(action[0, 3]))
    assert gain != 0  # An exact-real cancellation shortcut would return zero.
    assert Fraction(float(lower[0, 3])) <= gain <= Fraction(float(upper[0, 3]))


def test_binary_product_overflow_and_underflow_are_outward():
    for value in (-1., 1.):
        process, exact = _GainIntervalMixture(), _DyadicMixture()
        values = [Fraction(value)] * 1100
        process.update([(value, value)] * len(values))
        exact.update(values)
        low, high = process.bounds()
        numerator, denominator = exact.parts()
        assert Fraction(low) <= Fraction(numerator, denominator)
        assert np.isinf(high) or Fraction(numerator, denominator) <= Fraction(high)
        def unnecessary_fallback():
            raise AssertionError("A widely separated crossing must not need replay")
        assert process.crossed(Fraction(1, 20), unnecessary_fallback) == (value > 0)


def test_subnormal_normalization_and_zero_scale_keep_exact_decisions():
    class PointReader:
        def __init__(self, gains):
            self.gains, self.rows = gains, 0
            self.plan = type("Plan", (), {"stats": {}})()
            self.stats = {}
        def gain_intervals(self, indices, active):
            assert np.array_equal(indices, np.arange(self.rows, self.rows + len(indices)))
            self.rows += len(indices)
            return self.gains[indices].copy(), self.gains[indices].copy()
        def exact_prefix(self, e):
            return [Fraction(float(v)) for v in self.gains[:self.rows, e]]
    tiny = np.nextafter(0., 1.)
    for gain, bounds in ((np.full((12, 4), tiny), (tiny,) * 4),
                         (np.tile((0., 1., 1., 1.), (12, 1)), (0., 1., 1., 1.))):
        expected = certify_fresh_mixture(gain, np.zeros_like(gain), gain_bounds=bounds,
            noninferiority_margins=(0.,) * 4, min_anchor_gain=0., batch=1)
        result = certify_action_queries(PointReader(gain), len(gain), gain_bounds=bounds,
            noninferiority_margins=(0.,) * 4, min_anchor_gain=0., batch=1)
        assert all(result[k] == v for k, v in expected.items())
        if bounds[0] == tiny:
            assert result["exact_filter_fallbacks"] > 0


def test_delayed_exact_prefix_replay_with_short_inspection_batches():
    from cara.interval_certificate import certify_streaming_interval
    from run_streaming_validation import PairedWorkloadReader
    environment = make_environment(246001)
    old = tuple(np.arange(LAW["vms"]) % 8)
    new = list(old)
    new[0], new[1] = new[1], new[0]
    move = migration_profile(old, new)
    bounds = stored_action_gain_bounds(old, move["aligned_assignment"], LAW["capacity"],
                                      horizon=LAW["horizon"])["gain_bounds"]
    options = dict(gain_bounds=bounds, noninferiority_margins=(.01, .01, 0., .001), batch=19)
    reference_reader = PairedWorkloadReader(environment, "complementary", 15, old, new)
    expected = certify_streaming_interval(reference_reader, 35, 4, **options)
    class DelayedWideReader(ActionSceneReader):
        def gain_intervals(self, indices, active):
            lower, upper = super().gain_intervals(indices, active)
            if self.rows > 19:
                for e, enabled in enumerate(active):
                    if enabled:
                        lower[:, e], upper[:, e] = -bounds[e], bounds[e]
            return lower, upper
    reader = DelayedWideReader(environment, "complementary", 15, old, new,
        LAW["capacity"], horizon=LAW["horizon"])
    result = certify_action_queries(reader, 35, **options)
    assert all(result[k] == v for k, v in expected.items())
    assert result["exact_filter_fallbacks"] > 0
    assert reader.stats["fallback_rows_generated"] == 35


def test_nonbinary_population_support_does_not_cut_off_a_legal_gain():
    bound = Fraction(1, 3)
    reference = np.nextafter(float(bound), np.inf)
    action = 3 * 2. ** -56
    gain = Fraction(float(reference)) - Fraction(action)
    assert Fraction(float(bound)) < gain < bound
    center = reference - action
    support_limit = _outward(bound, True)
    low = max(-support_limit, np.nextafter(center, -np.inf))
    high = min(support_limit, np.nextafter(center, np.inf))
    assert Fraction(low) <= gain <= Fraction(high)


def test_fraction_subnormal_scale_fallback_keeps_normalized_factors_positive(monkeypatch):
    import cara.action_query as module
    original = module._GainIntervalMixture
    class CheckedMixture(original):
        def update(self, values):
            assert all(-1. <= low <= high <= 1. for low, high in values)
            return super().update(values)
    monkeypatch.setattr(module, "_GainIntervalMixture", CheckedMixture)
    tiny = np.nextafter(0., 1.)
    class WideZeroReader:
        def __init__(self):
            self.rows, self.stats = 0, {}
            self.plan = type("Plan", (), {"stats": {}})()
        def gain_intervals(self, indices, active):
            self.rows += len(indices)
            return np.full((len(indices), 4), -tiny), np.full((len(indices), 4), tiny)
        def exact_prefix(self, e):
            return [Fraction()] * self.rows
    for exponent in (1075, 2200):
        bounds = (Fraction(1, 2**exponent),) * 4
        expected = certify_fresh_mixture(np.zeros((8, 4)), np.zeros((8, 4)),
            gain_bounds=bounds, noninferiority_margins=(0.,) * 4, min_anchor_gain=0., batch=1)
        actual = certify_action_queries(WideZeroReader(), 8, gain_bounds=bounds,
            noninferiority_margins=(0.,) * 4, min_anchor_gain=0., batch=1)
        assert all(actual[key] == value for key, value in expected.items())


def test_underflowing_dyadic_weights_request_exact_threshold_replay():
    process = _GainIntervalMixture(minimum_components=1074)
    process.update([(1., 1.)])
    assert np.isinf(process.bounds()[1])
    reference = _DyadicMixture(minimum_components=1074)
    reference.update([Fraction(1)])
    alpha = Fraction(1, 2)
    assert process.crossed(alpha, lambda: [Fraction(1)]) == reference.crossed(alpha)
    assert process.exact_fallbacks == 1


def test_sum_error_geometric_bound_declares_its_count_contract():
    import pytest
    assert _sum_error(1, Fraction(1)) == 0
    with pytest.raises(ValueError, match="counts"):
        _sum_error(2**53 + 1, Fraction(1))
