import numpy as np
import pytest

from cara.compiled_admission import CompiledGlobalPlan
from cara.data import canonical_assignment
from cara.native_canonical import _same_bits
from cara.prefix_canonical import PrefixCanonicalPlan
from cara.refinement import migration_profile


def symbolic_flags(old, new):
    states_old, states_new = {}, {}
    flags = []
    for i, (a, b) in enumerate(zip(old, new)):
        left, right = states_old.get(a, ()), states_new.get(b, ())
        flags.append(int(left == right))
        states_old[a], states_new[b] = left + (i,), right + (i,)
    return flags


def test_shared_prefixes_fork_and_do_not_rejoin_on_numeric_coincidence():
    old = (0, 0, 0, 1, 1, 1)
    new = (0, 0, 1, 1, 1, 0)
    plan = PrefixCanonicalPlan(old, new, 1., 8, aligned_assignment=new)
    assert plan.share_flags.tolist() == [1, 1, 0, 0, 0, 0]
    assert plan.saved_prefix_terms == 2
    panel = np.ones((3, 6, 8))
    assert plan.audit(panel)["all_load_and_loss_bits_equal"]


def test_nonprefix_resident_intersection_cannot_be_shared():
    old, new = (0, 0, 1, 1), (0, 1, 1, 0)
    plan = PrefixCanonicalPlan(old, new, 1., 8, aligned_assignment=new)
    # VM2 is stationary in both roots but their earlier resident sequences
    # differ; only the very first common variable is a shared prefix.
    assert plan.share_flags.tolist() == [1, 0, 0, 0]
    assert plan.saved_prefix_terms == 1


@pytest.mark.parametrize("batch,horizon", [(1, 1), (1, 8), (2, 8), (32, 8), (32, 2)])
def test_flags_and_target_bytes_with_uneven_noncontiguous_residents(batch, horizon):
    rng = np.random.default_rng(63911 + batch + horizon)
    old = np.repeat(np.arange(5), (1, 2, 3, 8, 17))
    old = canonical_assignment(old[rng.permutation(len(old))])
    new = canonical_assignment(np.asarray(old)[rng.permutation(len(old))])
    aligned = migration_profile(old, new)["aligned_assignment"]
    plan = PrefixCanonicalPlan(old, new, 1., horizon, aligned_assignment=aligned)
    assert plan.share_flags.tolist() == symbolic_flags(plan.native.old, plan.native.new)
    reference = CompiledGlobalPlan(old, new, 1., horizon, raw_error_bound=False)
    for law in ("uniform", "wide", "rounding"):
        panel = rng.random((batch, len(old), horizon))
        if law == "wide":
            panel = np.ldexp(panel, rng.integers(-100, 1, panel.shape))
        elif law == "rounding":
            panel[:] = 2.**-54
            panel[:, ::3] = 1.
        assert plan.audit(panel)["all_load_and_loss_bits_equal"]
        for active in ((True,) * 4, (False, False, True, False), (False, False, False, True)):
            actual, expected = plan.evaluate_global(panel, active), reference.evaluate_global(panel, active)
            assert all(_same_bits(a, b) for a, b in zip(actual, expected))


def test_common_whole_host_is_not_double_counted_as_extra_prefix_saving():
    old, new = (0, 0, 1, 1, 2, 2), (0, 1, 0, 1, 2, 2)
    plan = PrefixCanonicalPlan(old, new, 1., 8, aligned_assignment=new)
    assert plan.actions[4:].tolist() == [2, 2]
    assert plan.saved_prefix_terms == 1
    panel = np.random.default_rng(6112).random((32, 6, 8))
    assert plan.audit(panel)["all_load_and_loss_bits_equal"]
    plan.evaluate_global(panel)
    assert plan.stats["aggregate_input_cells"] == (6 + 4 - 1) * 32 * 8
    assert plan.stats["source_panel_reads"] == panel.size
    assert plan.stats["prefix_copy_cells"] == 32 * 8
    assert plan.stats["common_copy_cells"] == 32 * 8
    assert plan.stats["peak_load_workspace_values"] == 2 * 3 * 8
