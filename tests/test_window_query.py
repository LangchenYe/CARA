"""Absolute-hour reuse preserves window identities and canonical targets."""
from fractions import Fraction

import numpy as np
import pytest

from cara.action_query import canonical_fresh_losses
from cara.data import canonical_assignment
from cara.window_query import (DirectWindowPrefix, KeyedProfilePlan,
    WindowPrefixExact, WindowPrefixIndexed, materialize_windows)


def action(n=48, hosts=6):
    old = canonical_assignment(np.arange(n) % hosts)
    matched = np.asarray(old).copy()
    # Moving the first identity exercises the candidate's canonical-host
    # permutation as well as shared prefix and untouched-host actions.
    matched[0], matched[1] = matched[1], matched[0]
    return old, canonical_assignment(matched), tuple(map(int, matched))


@pytest.mark.parametrize("capacity", [1., 4., .125])
def test_window_exact_matches_every_canonical_load_and_loss_bit(capacity):
    rng = np.random.default_rng(862103)
    matrix = np.ascontiguousarray(rng.lognormal(-3., 2., (48, 31)))
    matrix[:, 0] = rng.choice([0., .1, np.nextafter(.1, 0.),
        np.nextafter(.1, 1.), 2. ** -53], 48)
    old, new, aligned = action()
    starts = np.arange(matrix.shape[1] - 7, dtype=np.int64)
    query = WindowPrefixExact(old, new, capacity, 8,
        aligned_assignment=aligned, demand_upper=float(matrix.max()))
    report = query.audit_windows(matrix, starts)
    assert report["all_load_bits_equal"] and report["all_boolean_loss_bits_equal"]
    assert report["all_raw_loss_bits_equal"]
    assert query.stats["source_key_input_cells"] == matrix.size
    assert query.stats["window_input_cells"] == len(starts) * len(old) * 8


def test_indexed_raw_encloses_canonical_targets_with_repeated_out_of_order_windows():
    rng = np.random.default_rng(10917)
    matrix = np.ascontiguousarray(rng.choice([0., 2. ** -53, .4, .9, 8., 64.], (48, 73)))
    old, new, aligned = action()
    starts = np.asarray([60, 0, 12, 12, 3, 49, 0, 64], dtype=np.int64)
    query = WindowPrefixIndexed(old, new, 4., 8,
        aligned_assignment=aligned, demand_upper=64.)
    report = query.audit_windows(matrix, starts)
    assert report["all_load_bits_equal"] and report["all_raw_loss_enclosures_valid"]
    assert query.raw_valid2.tolist() == [1, 1]
    assert np.all(query.raw_allowance2 > 0)
    low, high = query.evaluate_windows(matrix, starts)
    assert low.shape == high.shape == (len(starts), 4)
    assert low[2].tobytes() == low[3].tobytes()
    assert high[1].tobytes() == high[6].tobytes()


def test_duration_only_plans_share_the_exact_same_consumer_without_raw_work():
    rng = np.random.default_rng(10919)
    matrix = np.ascontiguousarray(rng.uniform(0., 1., (48, 37)))
    old, new, aligned = action()
    starts = np.arange(30, dtype=np.int64)
    result = []
    for cls in (WindowPrefixExact, WindowPrefixIndexed):
        query = cls(old, new, 4., 8, aligned_assignment=aligned)
        result.append(query.evaluate_windows(matrix, starts, (False, False, True, False)))
        assert query.stats["raw_cell_evaluations"] == 0
        assert query.stats["raw_index_prefix_cells"] == 0
        assert query.stats["window_raw_budget_compilations"] == 0
    assert all(a.tobytes() == b.tobytes() for a, b in zip(*result))


def test_single_scene_keeps_reference_shape_contract_and_source_validation():
    rng = np.random.default_rng(10921)
    matrix = np.ascontiguousarray(rng.uniform(0., 1., (48, 27)))
    old, new, aligned = action()
    query = WindowPrefixIndexed(old, new, 4., 8, aligned_assignment=aligned)
    starts = np.asarray([6], dtype=np.int64)
    panel = materialize_windows(matrix, starts)
    low, high = query.evaluate_windows(matrix, starts)
    expected = [canonical_fresh_losses(panel, a, 4.) for a in (old, new)]
    for e in range(4):
        target = Fraction(float(expected[0][0, e])) - Fraction(float(expected[1][0, e]))
        assert Fraction(float(low[0, e])) <= target <= Fraction(float(high[0, e]))
    assert query.stats["window_shape_fallback_batches"] == 1
    with pytest.raises(ValueError, match="inside"):
        query.evaluate_windows(matrix, np.asarray([20, 21]))
    with pytest.raises(ValueError, match="integer"):
        query.evaluate_windows(matrix, np.asarray([0., 1.]))
    bad = matrix.copy(); bad[0, 0] = 2.
    with pytest.raises(ValueError, match="upper bound"):
        query.evaluate_windows(bad, np.asarray([0, 1]))


def test_unsafe_index_allowance_delegates_each_side_to_exact(monkeypatch):
    import cara.window_budget as budget
    def unsafe(*args):
        raise OverflowError("unsupported prefix range")
    monkeypatch.setattr(budget, "window_raw_allowance_upper", unsafe)
    matrix = np.ascontiguousarray(np.random.default_rng(10923).uniform(0., 1., (48, 27)))
    old, new, aligned = action()
    starts = np.arange(20, dtype=np.int64)
    query = WindowPrefixIndexed(old, new, 4., 8, aligned_assignment=aligned)
    report = query.audit_windows(matrix, starts)
    assert report["all_raw_loss_bits_equal"]
    assert query.raw_valid2.tolist() == [0, 0]


def test_matched_ablations_keep_canonical_bytes_and_charge_the_executed_work(monkeypatch):
    import cara.prefix_canonical as prefix
    from cara.window_query import prepare_window_queries
    prepare_window_queries()
    matrix = np.ascontiguousarray(np.random.default_rng(10925).uniform(0., 1., (48, 37)))
    old, new, aligned = action()
    starts = np.asarray([0, 7, 3, 7, 15, 29], dtype=np.int64)
    full = WindowPrefixExact(old, new, 4., 8, aligned_assignment=aligned)
    expected = full.evaluate_windows(matrix, starts)
    full_additions = full.stats["aggregate_input_cells"]
    no_dag = WindowPrefixExact(old, new, 4., 8, aligned_assignment=aligned, reuse_raw_dag=False)
    actual = no_dag.evaluate_windows(matrix, starts)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(expected, actual))
    assert no_dag.audit_windows(matrix, starts)["all_raw_loss_bits_equal"]
    # The audit is another execution; per-call counts are checked on the full
    # plan and the first execution before querying the audited control again.
    assert full.stats["exact_raw_dag_additions"] == 2 * full.hosts * (3 * matrix.shape[1] - 11)
    assert no_dag.stats["exact_raw_dag_additions"] == 0
    assert no_dag.stats["raw_window_tree_additions"] == 2 * 2 * no_dag.hosts * len(starts) * 7
    assert no_dag.stats["aggregate_input_cells"] == 2 * full_additions
    def forbidden_compile(*args):
        raise AssertionError("NoPrefix must not compile unused prefix states")
    monkeypatch.setattr(prefix, "_COMPILE", forbidden_compile)
    no_prefix = WindowPrefixExact(old, new, 4., 8, aligned_assignment=aligned, share_prefix=False)
    actual = no_prefix.evaluate_windows(matrix, starts)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(expected, actual))
    assert no_prefix.saved_prefix_terms == 0
    assert not hasattr(no_prefix, "share_flags")
    assert no_prefix.stats["prefix_copy_cells"] == 0
    assert no_prefix.stats["aggregate_input_cells"] == (len(old) + no_prefix.support_residents) * matrix.shape[1]
    assert no_prefix.stats["aggregate_input_cells"] >= full_additions
    assert no_prefix.audit_windows(matrix, starts)["all_raw_loss_bits_equal"]


@pytest.mark.parametrize("options", [{"share_prefix": False}, {"reuse_raw_dag": False}])
def test_window_ablation_single_scene_keeps_original_shape_fallback(options):
    matrix = np.ascontiguousarray(np.random.default_rng(10927).uniform(0., 1., (48, 27)))
    old, new, aligned = action()
    starts = np.asarray([5], dtype=np.int64)
    first = WindowPrefixExact(old, new, 4., 8, aligned_assignment=aligned)
    ablation = WindowPrefixExact(old, new, 4., 8, aligned_assignment=aligned, **options)
    expected = first.evaluate_windows(matrix, starts)
    actual = ablation.evaluate_windows(matrix, starts)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(expected, actual))
    assert ablation.stats["window_shape_fallback_batches"] == 1


def test_direct_source_reference_never_packs_input_and_uses_only_one_scene_scratch(monkeypatch):
    import cara.window_query as module
    matrix = np.ascontiguousarray(np.random.default_rng(10929).uniform(0., 1., (48, 41)))
    old, new, aligned = action()
    starts = np.asarray([0, 13, 3, 13, 29, 33], dtype=np.int64)
    baseline = DirectWindowPrefix(old, new, 4., 8, aligned_assignment=aligned)
    assert baseline.audit_windows(matrix, starts)["all_raw_loss_bits_equal"]
    exact = WindowPrefixExact(old, new, 4., 8, aligned_assignment=aligned)
    expected = exact.evaluate_windows(matrix, starts)
    def no_panel(*args):
        raise AssertionError("The source reference must not materialize a B,N,T panel")
    monkeypatch.setattr(module, "materialize_windows", no_panel)
    observed = baseline.evaluate_windows(matrix, starts)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(expected, observed))
    assert baseline.window_scratch.shape == (2, baseline.hosts, 8)
    assert baseline.stats["source_panel_reads"] == 2 * len(old) * len(starts) * 8
    assert baseline.stats["exact_raw_dag_additions"] == 0


@pytest.mark.parametrize("options", [{}, {"reuse_hours": False},
    {"share_prefix": False}, {"reuse_raw_dag": False}])
def test_historical_native_profiles_reproduce_all_original_stored_metrics_and_gate(options):
    from cara.certification import exact_pareto_gate
    from cara.refinement import panel_profile
    rng = np.random.default_rng(10931)
    old, new, aligned = action()
    matrix = np.ascontiguousarray(rng.uniform(0., .4, (48, 37)))
    matrix[:, :10] = 0.
    matrix[0, 0:4] = [1., np.nextafter(1., 2.), 1. + 1e-12, 1. + 2e-12]
    starts = np.asarray([20, 0, 10, 10, 2, 29, 5], dtype=np.int64)
    panel = materialize_windows(matrix, starts)
    reference = tuple(panel_profile(panel, a, 1.) for a in (old, new))
    query = WindowPrefixExact(old, new, 1., 8, aligned_assignment=aligned,
                             demand_upper=float(matrix.max()), **options)
    observed = query.historical_profiles(matrix, starts)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(reference, observed))
    assert exact_pareto_gate(*reference) == exact_pareto_gate(*observed)
    # No-action and all-idle scenes retain all original sentinel values.
    idle = np.zeros_like(matrix)
    idle_query = WindowPrefixExact(old, old, 1., 8, aligned_assignment=old, **options)
    profiles = idle_query.historical_profiles(idle, starts)
    assert all(a.tobytes() == panel_profile(materialize_windows(idle, starts), old, 1.).tobytes()
               for a in profiles)
    assert not exact_pareto_gate(*profiles)["admitted"]


@pytest.mark.parametrize("reuse_hours", [True, False])
def test_candidate_only_historical_backend_avoids_pair_metadata_and_keeps_single_load_table(monkeypatch, reuse_hours):
    import cara.prefix_canonical as prefix
    from cara.refinement import panel_profile
    matrix = np.ascontiguousarray(np.random.default_rng(10933).uniform(0., 1., (48, 37)))
    _, candidate, _ = action()
    starts = np.asarray([0, 7, 3, 29], dtype=np.int64)
    expected = panel_profile(materialize_windows(matrix, starts), candidate, 4.)
    def no_pair(*args, **kwargs):
        raise AssertionError("Candidate-only profiling must not construct a paired prefix plan")
    monkeypatch.setattr(prefix.PrefixCanonicalPlan, "__init__", no_pair)
    query = KeyedProfilePlan(candidate, 4., reuse_hours=reuse_hours)
    observed = query.historical_profile(matrix, starts)
    assert observed.tobytes() == expected.tobytes()
    assert not hasattr(query, "candidate")
    assert query.native.new.size == query.actions.size == query.candidate_order.size == 0
    assert query.window_scratch.shape == (1, query.hosts, 37 if reuse_hours else 8)
    assert query.stats["aggregate_input_cells"] == len(candidate) * (37 if reuse_hours else len(starts) * 8)
    assert query.stats["prefix_copy_cells"] == 0
    one = np.asarray([3], dtype=np.int64)
    assert query.historical_profile(matrix, one).tobytes() == panel_profile(
        materialize_windows(matrix, one), candidate, 4.).tobytes()


def test_indexed_historical_accessor_explicitly_delegates_to_exact_metrics():
    from cara.refinement import panel_profile
    matrix = np.ascontiguousarray(np.random.default_rng(10935).uniform(0., 1., (48, 37)))
    old, new, aligned = action()
    starts = np.arange(30, dtype=np.int64)
    query = WindowPrefixIndexed(old, new, 4., 8, aligned_assignment=aligned)
    observed = query.historical_profiles(matrix, starts)
    panel = materialize_windows(matrix, starts)
    assert all(value.tobytes() == panel_profile(panel, a, 4.).tobytes()
               for value, a in zip(observed, (old, new)))
    assert query.stats["historical_exact_delegations"] == 1
    assert query.stats["window_raw_budget_compilations"] == 0
