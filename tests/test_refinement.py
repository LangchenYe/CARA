"""Independent checks of physical movement, native profiles and exact families."""
from itertools import permutations, product
import json

import numpy as np
import pytest

from cara.data import canonical_assignment, native_metrics
from cara.refinement import (PROFILE_NAMES, diagnostic_weights, migration_profile,
                             panel_profile, search_refined, search_swap)
from cara.repair import feasible


def workload():
    panel = np.tile(np.array([[.7, .1], [.7, .1], [.1, .7], [.1, .7]]), (6, 1, 1))
    forecast = np.full((4, 2), .4)
    return panel, forecast, (0, 0, 1, 1)


def test_migration_matching_matches_brute_force_and_uses_one_mapping():
    inc = (0, 0, 1, 1, 2, 2)
    rng = np.random.default_rng(42)
    weights = np.array([1., 7., 2., 11., .5, 3.])
    for _ in range(50):
        candidate = tuple(rng.permutation(inc))
        actual = migration_profile(inc, candidate, weights)
        candidates = []
        for mapping in permutations(range(3)):
            aligned = np.array([mapping[h] for h in candidate])
            moved = aligned != inc
            candidates.append((int(moved.sum()), float(weights[moved].sum())))
        assert (actual["count"], actual["cost"]) == min(candidates)
        moved = np.asarray(actual["aligned_assignment"]) != inc
        assert actual["count"] == moved.sum()
        assert actual["cost"] == weights[moved].sum()
    assert migration_profile(inc, (8, 8, 4, 4, 17, 17))["count"] == 0
    with pytest.raises(ValueError):
        migration_profile(inc, inc, np.ones(5))


def test_profiles_reproduce_native_metrics_including_empty_active_hosts():
    panel = np.array([[[2., 4., 0.], [1., 1., 1.], [0., 0., 0.]],
                      [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                      [[1. + 5e-13, .5, .2], [.1, .2, .3], [0., 0., 0.]]])
    profile = panel_profile(panel, (0, 1, 2), 1.)
    for b, target in enumerate(panel):
        expected = native_metrics(target, (0, 1, 2), 1.)
        for name in ("VR", "raw", "VS", "PAR"):
            assert profile[b, PROFILE_NAMES.index(name)] == expected[name]
    # Native strict-capacity VR can differ from the CARA tolerance-based J.
    assert profile[2, 0] == 0 and profile[2, 2] > 0


def test_diagnostic_concentrates_on_actual_harmful_rows():
    reference = np.tile([1., .5, .4, .2, .3, 1.2], (4, 1))
    candidate = reference.copy()
    candidate[:, 2] -= .1
    candidate[:, 4] -= .1
    candidate[2, 3] += .2
    name, weights = diagnostic_weights(reference, candidate)
    assert name == "raw"
    assert weights[2] > .75
    assert np.all(weights > 0)
    np.testing.assert_allclose(weights.sum(), 1.)


@pytest.mark.parametrize("diagnostic,preview", [(False, False), (True, False), (True, True)])
def test_completed_family_equals_independent_migration_constrained_oracle(diagnostic, preview):
    panel, forecast, inc = workload()
    config = {"cpu_limit": .1, "initial_broad": 0, "mutable_limit": 4, "top_p": 16,
              "migration_fraction": .5, "diagnostic": diagnostic,
              "sketch_preview": preview, "replay_arms": ["exact"]}
    result = search_refined(inc, forecast, panel, panel.copy(), 1., config, np.random.default_rng(3))
    oracle = {canonical_assignment(a) for a in product(range(2), repeat=4)
              if len(set(a)) == 2 and canonical_assignment(a) != inc
              and feasible(a, forecast, 1., 2) and migration_profile(inc, a)["count"] <= 2}
    assert {r["assignment"] for r in result["candidates"]} == oracle
    assert result["stats"]["blocks"] == 1
    assert result["selected"] is not None
    json.dumps(result)
    selected = panel_profile(panel, result["selected"], 1.)
    reference = panel_profile(panel, inc, 1.)
    assert np.all(selected.mean(axis=0) <= reference.mean(axis=0) + 1e-12)
    replay = search_refined(inc, forecast, panel, panel.copy(), 1., config, np.random.default_rng(3))
    assert result["candidates"] == replay["candidates"]
    assert result["selected"] == replay["selected"]


def test_zero_migration_budget_falls_back_and_global_budget_is_enforced():
    panel, forecast, inc = workload()
    config = {"cpu_limit": .1, "initial_broad": 0, "mutable_limit": 4,
              "max_migrations": 0, "replay_arms": ["exact"]}
    result = search_refined(inc, forecast, panel, panel, 1., config, np.random.default_rng(5))
    assert result["selected"] is None
    assert result["candidates"] == []
    assert result["stats"]["migration_rejections"] > 0


def test_full_and_reused_execution_preserve_the_recorded_prefix_and_selector():
    panel, forecast, inc = workload()
    config = {"cpu_limit": .1, "initial_broad": 8, "broad_batch": 4, "mutable_limit": 4,
              "migration_fraction": .5, "sketch_preview": True, "top_p": 16,
              "replay_arms": ["exact", "broad"]}
    cached = search_refined(inc, forecast, panel, panel.copy(), 1., config, np.random.default_rng(18))
    full = search_refined(inc, forecast, panel, panel.copy(), 1., {**config, "reuse_views": False}, np.random.default_rng(18))
    assert cached["candidates"] == full["candidates"]
    assert cached["selected"] == full["selected"]
    assert cached["bets"] == full["bets"]
    assert cached["stats"]["evaluations"] == full["stats"]["evaluations"]
    assert full["stats"]["cache_hits"] == 0


def test_swap_baseline_improves_raw_with_physical_budget_and_heldout_guard():
    panel, forecast, inc = workload()
    result = search_swap(inc, forecast, panel, panel.copy(), 1.,
                         {"max_migrations": 2, "max_attempts": 100, "cpu_limit": .1},
                         np.random.default_rng(20))
    assert result["selected"] is not None
    assert result["movement"]["count"] == 2
    assert feasible(result["selected"], forecast, 1., 2)
    assert panel_profile(panel, result["selected"], 1.)[:, 3].mean() < panel_profile(panel, inc, 1.)[:, 3].mean()
    assert all(r["movement"]["count"] <= 2 for r in result["candidates"])


@pytest.mark.parametrize("score_mode", ["gain", "raw", "evidence"])
def test_frontier_fixed_attempts_replay_identically_across_all_view_plans(score_mode):
    panel, forecast, inc = workload()
    config = {"cpu_limit": .1, "frontier_seed": True, "frontier_fraction": .5,
              "mutable_limit": 4, "max_migrations": 2, "top_p": 16,
              "score_mode": score_mode, "replay_arms": ["frontier:8", "exact"]}
    results = [search_refined(inc, forecast, panel, panel.copy(), 1.,
               {**config, "reuse_views": reuse, "projected_views": project}, np.random.default_rng(16))
               for reuse, project in ((False, False), (True, False), (True, True), (True, "auto"))]
    assert all([row["arm"] for row in result["stats"]["history"]] == config["replay_arms"] for result in results)
    for result in results[1:]:
        assert result["candidates"] == results[0]["candidates"]
        assert result["selected"] == results[0]["selected"]
        assert result["bets"] == results[0]["bets"]
        assert result["stats"]["evaluations"] == results[0]["stats"]["evaluations"]


def test_auto_view_routing_and_bitwise_profiles_at_threshold():
    from cara.refinement import _FullReplay, _profile_replay, _replay
    from cara.repair import ReplayCache
    from cara.workload_views import ProjectionReplay

    rng = np.random.default_rng(4261)
    incumbent = (0, 0, 1, 1, 2, 2, 3, 3)
    candidate = (0, 1, 0, 1, 2, 2, 3, 3)
    for rows, times in ((24, 8), (31, 8), (32, 8), (33, 8)):
        panel = rng.uniform(0., .8, (rows, len(incumbent), times))
        automatic = _replay(panel, incumbent, 1., project="auto")
        assert type(automatic) is (ReplayCache if rows * times < 256 else ProjectionReplay)
        full = _replay(panel, incumbent, 1., reuse=False, project="auto")
        assert type(full) is _FullReplay
        for assignment in (incumbent, candidate):
            actual, _ = _profile_replay(automatic, assignment, 1.)
            expected, _ = _profile_replay(full, assignment, 1.)
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert actual.tobytes() == expected.tobytes()


def test_exact_residual_can_escape_a_raw_single_swap_local_optimum():
    # All values are dyadic. Every feasible one-swap neighbor has raw >= the
    # incumbent, but a four-VM recomposition improves raw, VS and PAR together.
    target = np.array([[6, 1, 7, 7], [4, 1, 4, 7], [2, 4, 3, 2], [3, 4, 2, 5],
                       [6, 6, 5, 2], [1, 2, 7, 5], [7, 1, 6, 6], [3, 3, 4, 6]]) / 16.
    panel = np.tile(target, (6, 1, 1))
    forecast = np.full((8, 4), .25)
    inc = (0, 0, 0, 0, 1, 1, 1, 1)
    reference = panel_profile(panel, inc, 1.)[:, 3].mean()
    for vm in range(4):
        for other in range(4, 8):
            candidate = list(inc)
            candidate[vm], candidate[other] = 1, 0
            assert panel_profile(panel, candidate, 1.)[:, 3].mean() >= reference
    config = {"cpu_limit": .1, "initial_broad": 0, "mutable_limit": 8,
              "max_migrations": 4, "top_p": 16, "score_mode": "raw", "replay_arms": ["exact"]}
    exact = search_refined(inc, forecast, panel, panel, 1., config, np.random.default_rng(12))
    assert exact["selected"] is not None
    assert panel_profile(panel, exact["selected"], 1.)[:, 3].mean() < reference
