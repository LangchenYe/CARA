import numpy as np
import pytest

from cara.compiled_admission import CompiledGlobalPlan
from cara.fused_canonical import FusedCanonicalPlan
from cara.data import canonical_assignment
from cara.native_canonical import _same_bits
from cara.refinement import migration_profile


@pytest.mark.parametrize("batch,horizon", [(1, 1), (1, 8), (2, 8), (32, 8), (32, 2)])
def test_fused_canonical_bytes_masks_and_shape_fallback(batch, horizon):
    rng = np.random.default_rng(32361 + batch + horizon)
    old = np.repeat(np.arange(5), (1, 2, 3, 8, 17))
    old = canonical_assignment(old[rng.permutation(len(old))])
    new = canonical_assignment(np.asarray(old)[rng.permutation(len(old))])
    aligned = migration_profile(old, new)["aligned_assignment"]
    plan = FusedCanonicalPlan(old, new, 1., horizon, aligned_assignment=aligned)
    target = CompiledGlobalPlan(old, new, 1., horizon, raw_error_bound=False)
    for law in ("uniform", "wide", "rounding"):
        panel = rng.random((batch, len(old), horizon))
        if law == "wide":
            panel = np.ldexp(panel, rng.integers(-100, 1, panel.shape))
        if law == "rounding":
            panel[:] = 2.**-54
            panel[:, ::3, :] = 1.
        audit = plan.audit(panel)
        assert audit["all_load_and_loss_bits_equal"]
        assert audit["canonical_shape_fallback"] == (horizon != 8 or batch == 1)
        if horizon != 8 or batch == 1:
            assert plan.scratch is None
        for mask in ((True,) * 4, (False, False, True, False), (False, False, False, True)):
            actual, expected = plan.evaluate_global(panel, mask), target.evaluate_global(panel, mask)
            assert all(_same_bits(left, right) for left, right in zip(actual, expected))


def test_fused_canonical_common_hosts_and_scratch_ledger():
    old, new = (0, 0, 1, 1, 2, 2), (0, 1, 0, 1, 2, 2)
    plan = FusedCanonicalPlan(old, new, 1., 8)
    panel = np.random.default_rng(9241).random((32, 6, 8))
    assert plan.audit(panel)["all_load_and_loss_bits_equal"]
    plan.evaluate_global(panel)
    assert plan.stats["source_panel_reads"] == panel.size
    assert plan.stats["aggregate_input_cells"] == (6 + 4) * 32 * 8
    assert plan.stats["peak_load_workspace_values"] == 2 * 3 * 8
    assert plan.stats["materialized_load_table_values"] == 0
