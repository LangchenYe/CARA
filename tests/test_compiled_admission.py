"""The stronger compiled complete-pair control preserves canonical results."""
import numpy as np

from cara.action_query import ActionQueryPlan
from cara.compiled_admission import CompiledGlobalPlan


def test_compiled_pair_keeps_canonical_targets_and_live_projection_results():
    rng = np.random.default_rng(98017)
    for b, t, h in ((1, 1, 3), (1, 8, 4), (32, 1, 5), (32, 8, 8)):
        old = tuple(np.repeat(np.arange(h), 4)); new = list(old)
        new[0], new[4] = new[4], new[0]
        panel = rng.uniform(0., 1., (b, len(old), t))
        panel[:, :3, 0] = (1., 2. ** -53, np.nextafter(0., 1.))
        original = ActionQueryPlan(old, new, 1., t, raw_error_bound=False)
        compiled = CompiledGlobalPlan(old, new, 1., t, raw_error_bound=False)
        for active in ((True,) * 4, (False, False, True, False), (False, True, False, True)):
            expected, actual = original.evaluate_global(panel, active), compiled.evaluate_global(panel, active)
            assert all(np.array_equal(x, y) for x, y in zip(expected, actual))


def test_sparse_full_control_encloses_the_canonical_pair():
    from fractions import Fraction
    from cara.action_query import canonical_fresh_losses
    from cara.compiled_admission import CSRGlobalPlan, NativeGlobalPlan
    rng = np.random.default_rng(33158)
    for b, t, h in ((1, 1, 3), (1, 8, 4), (32, 1, 5), (32, 8, 8)):
        old = tuple(np.repeat(np.arange(h), 4)); new = list(old)
        new[0], new[4] = new[4], new[0]
        panel = rng.uniform(0., 1., (b, len(old), t))
        panel[:, :3, 0] = (1., 2. ** -53, np.nextafter(0., 1.))
        losses = [canonical_fresh_losses(panel, a, 1.) for a in (old, new)]
        for cls in (CSRGlobalPlan, NativeGlobalPlan):
            plan = cls(old, new, 1., t, raw_error_bound=False)
            lower, upper = plan.evaluate_global(panel)
            for row in range(b):
                for e in range(4):
                    target = Fraction(float(losses[0][row,e])) - Fraction(float(losses[1][row,e]))
                    assert Fraction(float(lower[row,e])) <= target <= Fraction(float(upper[row,e]))


def test_point_loads_keep_aggregate_allowance_for_a_different_host_order():
    from cara.bounded_query import BoundedPairProjection, projection_raw_allowance
    old = np.asarray((1e16, 2., 2., 2., 2., 1e16)).reshape(6, 1, 1)
    reordered = np.sort(old[:, 0, 0]).reshape(6, 1, 1)
    def raw(a):return np.maximum(a - 1., 0.).sum(axis=(0, 2)) / 6
    assert raw(old)[0] != raw(reordered)[0]
    beta = projection_raw_allowance((1,) * 6, np.zeros(6), 1., 1, 1e16)
    projection = BoundedPairProjection(1., 1, np.zeros(6), np.zeros(6),
        raw_allowance=(0., beta))
    low, high = projection.evaluate(old, reordered, (False, False, False, True))
    # The canonical target on both sides has old's order, hence exact gain 0.
    assert low[0, 3] <= 0. <= high[0, 3]
