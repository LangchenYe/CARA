import numpy as np
import pytest

from cara.compiled_admission import CompiledGlobalPlan
from cara.native_canonical import NativeCanonicalPlan, _same_bits
from cara.data import canonical_assignment, host_loads
from cara.refinement import migration_profile


@pytest.mark.parametrize("batch,horizon", [(1, 1), (1, 8), (2, 1), (32, 2), (32, 8)])
def test_native_canonical_loads_losses_and_intervals(batch, horizon):
    rng = np.random.default_rng(7123 + batch + horizon)
    old = np.repeat(np.arange(4), (1, 3, 8, 17))
    old = tuple(canonical_assignment(old[rng.permutation(len(old))]))
    new = tuple(canonical_assignment(np.asarray(old)[rng.permutation(len(old))]))
    aligned = migration_profile(old, new)["aligned_assignment"]
    plan = NativeCanonicalPlan(old, new, 1., horizon, aligned_assignment=aligned)
    reference = CompiledGlobalPlan(old, new, 1., horizon, raw_error_bound=False)
    for law in ("uniform", "wide"):
        panel = rng.random((batch, len(old), horizon))
        if law == "wide":
            panel = np.ldexp(panel, rng.integers(-100, 1, panel.shape))
        assert plan.audit(panel)["all_load_and_loss_bits_equal"]
        for active in ((True,)*4, (False, False, True, False), (False, False, False, True)):
            actual, expected = plan.evaluate_global(panel, active), reference.evaluate_global(panel, active)
            assert all(_same_bits(left, right) for left, right in zip(actual, expected))


def test_scalar_pairwise_counterexample_uses_original_path():
    panel = np.full((1, 17, 1), 2.**-54)
    panel[:, 0] = 1.
    plan = NativeCanonicalPlan((0,)*17, (0,)*17, np.nextafter(1., np.inf), 1, tolerance=0.)
    assert host_loads(panel, (0,)*17)[0, 0, 0] > 1.
    assert plan.audit(panel)["scalar_fallback"]
    plan.evaluate_global(panel)
    assert plan.stats["scalar_canonical_fallback_batches"] == 1


def test_common_memberships_are_shared_and_output_is_canonical_contiguous():
    old, new = (0, 0, 1, 1, 2, 2), (0, 1, 0, 1, 2, 2)
    panel = np.random.default_rng(921).random((32, 6, 8))
    plan = NativeCanonicalPlan(old, new, 1., 8)
    reference, candidate = plan.canonical_loads(panel)
    assert reference.flags.c_contiguous and candidate.flags.c_contiguous
    assert _same_bits(reference[2], candidate[2])
    assert plan.audit(panel)["all_load_and_loss_bits_equal"]
    plan.evaluate_global(panel)
    assert plan.stats["aggregate_input_cells"] == (6 + 4) * 32 * 8
    assert plan.stats["source_panel_reads"] == panel.size
    assert plan.stats["source_workspace_values"] == 0
