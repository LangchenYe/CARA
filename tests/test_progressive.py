"""Top-P and exact-profile oracles for progressive family scoring."""
import numpy as np
import pytest

import benchmark_family_execution as execution
import benchmark_family_queries as queries
from cara.family_certificates import FamilyCertificates, ResidualFamily
from cara.query_engine import ForecastViews
from cara.refinement import search_refined


def certificate():
    incumbent = (0,) * 8 + (1,) * 8
    family = ResidualFamily(incumbent, incumbent, (0, 1), (0, 1, 2, 3, 8, 9, 10, 11))
    return FamilyCertificates(family, ForecastViews(np.zeros((16, 1)), 1.))


def test_exact_optimistic_order_stops_after_first_batch_and_preserves_top_p():
    cert = certificate()
    cert.leaf_upper_keys = {mask: (100 - mask,) for mask in range(70)}
    pool = {-i - 1: (-100 - i,) for i in range(4)}
    processed, witnesses = [], []
    def cutoff():
        return min(pool.values()) if len(pool) == 4 else None
    for batch in cert.score_batches(range(70), cutoff, batch_size=32, observer=witnesses.append):
        processed.extend(batch)
        pool.update({mask: (100 - mask,) for mask in batch})
        pool = dict(sorted(pool.items(), key=lambda item: (-item[1][0], item[0]))[:4])
    assert tuple(pool) == (0, 1, 2, 3)
    assert processed == list(range(32))
    assert cert.stats['progressive_batches'] == 1
    assert cert.stats['progressive_excluded_masks'] == cert.stats['score_excluded_masks'] == 38
    assert tuple(witnesses[0]['active_masks']) == tuple(range(32, 70))
    assert witnesses[0]['optimistic_key'] < witnesses[0]['cutoff']


def test_equal_upper_keys_are_profiled_for_a_later_assignment_tie_winner():
    cert = certificate()
    cert.leaf_upper_keys = {mask: (1,) for mask in range(40)}
    processed = [mask for batch in cert.score_batches(range(40), lambda: (1,), batch_size=7)
                 for mask in batch]
    assert processed == list(range(40))
    # Placement order can disagree with mask order; equal-key leaves stay live.
    winner = min(processed, key=lambda mask: (-mask,))
    assert winner == 39
    assert cert.stats['progressive_excluded_masks'] == 0


def test_unknown_upper_keys_precede_finite_keys_and_an_unfilled_pool_cannot_stop():
    cert = certificate()
    cert.leaf_upper_keys = {0: (10,), 1: None, 2: (9,), 3: None, 4: (-10,)}
    batches = list(cert.score_batches(range(5), lambda: None, batch_size=2))
    assert batches == [(1, 3), (0, 2), (4,)]
    assert cert.stats['progressive_excluded_masks'] == 0
    with pytest.raises(ValueError):
        list(cert.score_batches(range(5), lambda: None, batch_size=0))


def test_cached_leaf_bounds_are_reused_and_cleared_before_a_new_prune():
    cert = certificate()
    masks = cert.family.nonempty_masks()[:10]
    class Bounds:
        calls = 0
        def upper_keys(self, ranges, **kwargs):
            self.calls += 1
            return [None if start == 0 else (start, 0) for start, _ in ranges]
    bounds = Bounds()
    surviving, _ = cert.prune(masks, migration_limit=16, cutoff=(-100, 0),
                             score_bounds=bounds, certify_migration=False,
                             score_min_leaves=1, score_strategy='leaf')
    assert cert.leaf_upper_keys == {mask: None if mask == 0 else (mask, 0) for mask in masks}
    list(cert.score_batches(surviving, lambda: None, batch_size=3))
    assert bounds.calls == 1
    cert.prune(masks, migration_limit=16, certify_score=False, certify_migration=False)
    assert cert.leaf_upper_keys == {}


@pytest.mark.parametrize('hosts,batch_values', [(4, 1024), (128, 512)])
def test_progressive_queries_match_eager_with_independent_profiles_and_exclusion_oracles(hosts, batch_values):
    rng = np.random.default_rng(419)
    incumbent = tuple(np.repeat(np.arange(hosts), 4))
    forecast = np.full((len(incumbent), 3), .08)
    panel = rng.uniform(.015, .2, (5, len(incumbent), 3))
    config = {'top_p': 4, 'migration_fraction': .5, 'archive_mode': 'balanced',
              'score_mode': 'gain', 'family_score_min_hosts': 0,
              'family_score_min_leaves': 1, 'family_score_strategy': 'leaf',
              'family_forecast_strategy': 'leaf', 'family_progressive_batch': 3,
              'query_batch_values': batch_values}
    expected = None
    progressive_batches = 0
    for plan in ('eager', 'compact', 'no_progressive', 'certified'):
        counts = {}
        def witness(value):
            execution.verify_witness(value, incumbent, forecast, panel, .51, counts)
        with execution.profile_oracle(counts):
            snapshots, stats = queries.run_plan(panel, forecast, incumbent, .51, config, plan,
                                                families=3, width=6, certificate_observer=witness)
        actual = execution.encoded(snapshots)
        expected = actual if expected is None else expected
        assert actual == expected
        for reason in ('forecast', 'migration', 'score'):
            excluded = sum(row['counters'].get(reason + '_excluded_masks', 0) for row in stats['families'])
            assert counts.get(reason + '_leaves', 0) == excluded
        batches = sum(row['counters'].get('progressive_batches', 0) for row in stats['families'])
        if plan == 'certified':
            progressive_batches = batches
        else:
            assert batches == 0
        for row in stats['families']:
            assert row['profile_cpu_seconds'] >= 0 and row['retention_cpu_seconds'] >= 0
    if hosts >= 128:
        assert progressive_batches > 0


def test_low_host_frozen_gate_does_not_enable_progressive_scoring():
    rng = np.random.default_rng(420)
    incumbent = tuple(np.repeat(np.arange(4), 4))
    forecast = np.full((16, 3), .08)
    panel = rng.uniform(.015, .2, (5, 16, 3))
    config = {'top_p': 4, 'migration_fraction': .5, 'family_score_min_hosts': 128,
              'family_score_min_leaves': 1, 'family_score_strategy': 'leaf'}
    _, stats = queries.run_plan(panel, forecast, incumbent, .51, config, 'certified', families=3, width=6)
    assert all(row['counters']['progressive_batches'] == 0 for row in stats['families'])


def test_controller_preserves_every_complete_family_and_final_physical_action():
    rng = np.random.default_rng(421)
    panel = rng.uniform(.02, .2, (7, 24, 3))
    incumbent = tuple(np.repeat(np.arange(6), 4))
    forecast = np.full((24, 3), .08)
    config = {'cpu_limit': 1., 'initial_broad': 0, 'mutable_limit': 6, 'top_p': 4,
              'migration_fraction': .5, 'diagnostic': False, 'archive_mode': 'balanced',
              'seed_schedule': 'coverage', 'score_mode': 'gain', 'shared_queries': True,
              'batch_queries': True, 'family_score_min_hosts': 0, 'family_score_min_leaves': 1,
              'family_score_strategy': 'leaf', 'family_progressive_batch': 3,
              'replay_arms': ['exact'] * 8}
    results, histories = {}, {}
    for mode, progressive in [('eager', False), ('certified', False), ('certified', True)]:
        snapshots = []
        settings = {**config, 'family_execution': mode, 'family_progressive': progressive,
                    'family_observer': lambda records, details: snapshots.append(execution.encoded(records))}
        result = search_refined(incumbent, forecast, panel, panel, .51, settings, np.random.default_rng(18))
        key = (mode, progressive)
        results[key] = {field: result.get(field) for field in ('candidates', 'selected', 'movement', 'selection_reason')}
        histories[key] = snapshots
    assert results['eager', False] == results['certified', False] == results['certified', True]
    assert histories['eager', False] == histories['certified', False] == histories['certified', True]
