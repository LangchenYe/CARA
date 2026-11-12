"""Semantic and resource checks for shared what-if query execution."""
import numpy as np
import pytest

from cara.data import canonical_assignment, host_loads
from cara.query_engine import ForecastViews, SharedQueryReplay
from cara.refinement import panel_profile, search_refined
from cara.repair import feasible


def test_shared_batches_preserve_full_replay_at_thresholds():
    rng = np.random.default_rng(7318)
    for b, n, t, h in ((1, 12, 1, 3), (7, 68, 8, 11), (24, 150, 24, 17), (3, 64, 65, 8)):
        base = canonical_assignment(np.arange(n) % h)
        panel = rng.lognormal(-3, 3, (b, n, t))
        panel[0, :, 0] = rng.choice([0., .1, np.nextafter(.1, 0.),
                                     np.nextafter(.1, 1.), 1., 2. ** -53,
                                     np.nextafter(0., 1.)], n)
        assignments = []
        for _ in range(32):
            candidate = np.asarray(base).copy()
            x, y = rng.choice(n, 2, replace=False)
            candidate[x], candidate[y] = candidate[y], candidate[x]
            assignments.append(canonical_assignment(candidate))
        # A deliberately small registry forces eviction during the same batch.
        engine = SharedQueryReplay(panel, base, 1., max_views=5)
        outputs = engine.profiles(assignments)
        batch_loads = engine.loads_many(assignments)
        for assignment, actual in zip(assignments, outputs):
            expected = panel_profile(panel, assignment, 1.)
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert actual.tobytes() == expected.tobytes()
            assert engine.profile(assignment).tobytes() == expected.tobytes()
            assert engine.loads(assignment).tobytes() == host_loads(panel, assignment).tobytes()
        for assignment, actual in zip(assignments, batch_loads):
            assert actual.tobytes() == host_loads(panel, assignment).tobytes()


def test_registry_reuses_membership_across_seeds_and_obeys_storage_limits():
    panel = np.full((5, 12, 3), .1)
    first = canonical_assignment(np.arange(12) % 3)
    second = list(first)
    second[0], second[1] = second[1], second[0]
    engine = SharedQueryReplay(panel, first, 1., max_views=4, max_bytes=2000)
    engine.profile(first)
    fork = engine.fork(second)
    assert fork.views is engine.views
    assert fork._registry is engine
    fork.profile(second)
    assert fork.hits >= 1
    for assignment in (first, second, first):
        assert fork.profile(assignment).tobytes() == panel_profile(panel, assignment, 1.).tobytes()
        assert len(engine.views) <= 4
        assert engine.cached_view_bytes <= 2000
    # If even one view exceeds the byte allowance, execution remains correct
    # while nothing survives between requests.
    ephemeral = SharedQueryReplay(panel, first, 1., max_bytes=1)
    assert ephemeral.cached_view_bytes == 0
    assert ephemeral.profile(second).tobytes() == panel_profile(panel, second, 1.).tobytes()
    assert ephemeral.cached_view_bytes == 0
    with pytest.raises(ValueError, match="same immutable panel"):
        SharedQueryReplay(panel.copy(), first, 1., registry=engine)
    limited = SharedQueryReplay.from_config(panel, first, 1., {"query_cache_views": 2, "query_cache_bytes": 700})
    limited.profile(second)
    assert len(limited.views) <= 2 and limited.cached_view_bytes <= 700


def test_batch_keeps_query_order_empty_cases_and_inactive_host_semantics():
    panel = np.zeros((2, 6, 3))
    panel[1, :3] = [.5, 1., 1.5]
    first = (0, 0, 0, 1, 1, 1)
    second = (0, 0, 1, 0, 1, 1)
    engine = SharedQueryReplay(panel, first, 1.)
    assert engine.profiles([]).shape == (0, 2, 6)
    for actual, assignment in zip(engine.profiles([second, first, second]), [second, first, second]):
        assert actual.tobytes() == panel_profile(panel, assignment, 1.).tobytes()
    with pytest.raises(ValueError, match="equal host count"):
        engine.profiles([first, (0,) * 6])


def test_forecast_views_preserve_capacity_predicate_at_boundaries_and_eviction():
    rng = np.random.default_rng(9013)
    for times in (1, 8, 24):
        forecast = rng.uniform(0, .25, (45, times))
        forecast[:, 0] = rng.choice([0., .1, np.nextafter(.1, 0.), np.nextafter(.1, 1.)], 45)
        seed = canonical_assignment(np.arange(45) % 7)
        candidates = [seed]
        for _ in range(64):
            assignment = np.asarray(seed).copy()
            a, b = rng.choice(len(seed), 2, replace=False)
            assignment[a], assignment[b] = assignment[b], assignment[a]
            candidates.append(canonical_assignment(assignment))
        candidates += [(0,) * 45, (0,) * 44]
        views = ForecastViews(forecast, .6, seed, max_views=5)
        expected = [len(a) == 45 and feasible(a, forecast, .6, 7) for a in candidates]
        assert views.feasible_many(candidates, 7).tolist() == expected
        assert [views.feasible(a, 7) for a in candidates] == expected
        assert len(views.views) <= 5 and views.evictions > 0


def test_complete_fixed_search_path_keeps_retained_keys_and_selected_action():
    from execution_signatures import result_signature
    rng = np.random.default_rng(91731)
    forecast = np.full((24, 8), .18)
    from cara.repair import ffd
    incumbent = ffd(forecast, 1.)
    build = rng.lognormal(-.5 * .35 ** 2, .35, (12, 24, 8)) * .19
    selection = rng.lognormal(-.5 * .35 ** 2, .35, (10, 24, 8)) * .19
    config = {"cpu_limit": 1., "top_p": 8, "mutable_limit": 4, "initial_broad": 0,
              "migration_fraction": .5, "diagnostic": False, "frontier_seed": True,
              "frontier_fraction": .75, "score_mode": "gain", "seed_schedule": "coverage",
              "replay_arms": ["frontier:64", "exact", "exact", "exact"], "reuse_views": False}
    reference = search_refined(incumbent, forecast, build, selection, 1., config, np.random.default_rng(123456))
    shared = search_refined(incumbent, forecast, build, selection, 1.,
        {**config, "shared_queries": True, "batch_queries": True}, np.random.default_rng(123456))
    assert result_signature(reference) == result_signature(shared)


def test_feedback_allocator_replays_recorded_costs_across_execution_plans():
    from benchmark_query_engine import _run_path
    from execution_signatures import result_signature
    from cara.repair import ffd
    rng = np.random.default_rng(4017)
    forecast = np.full((24, 8), .18)
    incumbent = ffd(forecast, 1.)
    build = rng.lognormal(-.5 * .35 ** 2, .35, (12, 24, 8)) * .19
    selection = rng.lognormal(-.5 * .35 ** 2, .35, (10, 24, 8)) * .19
    config = {"cpu_limit": 1., "top_p": 8, "mutable_limit": 4, "initial_broad": 0,
              "migration_fraction": .5, "diagnostic": False, "frontier_seed": True,
              "frontier_fraction": .75, "score_mode": "gain", "seed_schedule": "coverage",
              "feedback_queries": True, "shared_queries": True, "batch_queries": True,
              "replay_arms": ["frontier:64", "exact", "exact", "exact", "exact"]}
    source = search_refined(incumbent, forecast, build, selection, 1., config, np.random.default_rng(7311))
    case = {"config": config, "history": source["stats"]["history"], "incumbent": incumbent,
            "forecast": forecast, "build": build, "selection": selection, "capacity": 1., "seed": 7311}
    for plan in ("full", "load", "project", "shared_scalar", "shared_batch"):
        actual = _run_path(case, plan)
        assert result_signature(actual) == result_signature(source)
        assert actual["stats"]["query_feedback"] == source["stats"]["query_feedback"]
