"""Small exact runner checks; no real traces or held-out outcomes are loaded."""

import numpy as np

import benchmark_family_execution as execution
import benchmark_family_queries as queries
from cara.data import canonical_assignment


def toy():
    rng = np.random.default_rng(79)
    incumbent = canonical_assignment(np.arange(16) % 4)
    forecast = np.full((16, 3), .1)
    panel = rng.uniform(.015, .2, (5, 16, 3))
    config = {"cpu_limit": 1., "mode": "exact", "frontier_seed": False, "initial_broad": 0,
              "top_p": 2, "mutable_limit": 4, "migration_fraction": .125,
              "shared_queries": True, "batch_queries": True, "diagnostic": False,
              "archive_mode": "balanced", "score_mode": "gain", "seed_schedule": "coverage",
              "family_score_min_leaves": 1, "family_score_strategy": "leaf"}
    return incumbent, forecast, panel, config


def test_complete_query_consumers_and_all_exclusion_oracles():
    incumbent, forecast, panel, config = toy()
    expected = None
    excluded = 0
    oracle_cache = {}
    for plan in execution.PLANS:
        counts = {}

        def witness(value):
            execution.verify_witness(value, incumbent, forecast, panel, .51, counts, oracle_cache)

        with execution.profile_oracle(counts, oracle_cache):
            snapshots, stats = queries.run_plan(panel, forecast, incumbent, .51, config, plan,
                                                families=3, width=4, certificate_observer=witness)
        actual = execution.encoded(snapshots)
        expected = actual if expected is None else expected
        assert actual == expected
        assert counts["materialized_profiles"] > 0
        assert len(snapshots) == 3
        if plan == "no_matching_reuse":
            assert all(row["counters"]["migration_signature_hits"] == 0 for row in stats["families"])
        elif plan == "compact":
            assert sum(row["counters"]["migration_signature_hits"] for row in stats["families"]) > 0
        for reason in ("forecast", "migration", "score"):
            expected_count = sum(row["counters"].get(reason + "_excluded_masks", 0) for row in stats["families"])
            assert counts.get(reason + "_leaves", 0) == expected_count
            excluded += expected_count
    assert excluded > 0


def test_complete_search_replay_checks_intermediate_physical_pools():
    incumbent, forecast, panel, config = toy()
    config = {**config, "replay_arms": ["exact", "exact"]}
    case = {"dataset": "toy", "origin": 216, "seed": 2, "incumbent": incumbent,
            "forecast": forecast, "build": panel, "selection": panel[:3], "audit": panel[-3:],
            "capacity": .51, "config": config}
    result, admission, _ = execution.execute(case, "eager")
    frozen = {"history": result["stats"]["history"], "signature": execution.result_signature(result, admission)}
    verified = execution.compare_case(case, frozen, list(execution.PLANS), 0, 0)
    assert verified["complete_families"] == 2
    assert verified["all_family_pools_exact"]
    assert verified["all_exclusion_leaves_checked"]


def test_reference_profiles_cache_by_source_capacity_and_complete_assignment(monkeypatch):
    incumbent, forecast, panel, config = toy()
    original = execution.CanonicalProfileOracle.profile
    calls = []
    def checked(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(execution.CanonicalProfileOracle, "profile", checked)
    cache = {}
    first = execution.oracle_profile(panel, incumbent, .51, cache)
    second = execution.oracle_profile(panel, list(incumbent), .51, cache)
    assert first is second and not first.flags.writeable and len(calls) == 1
    other_source = execution.oracle_profile(panel.copy(), incumbent, .51, cache)
    assert other_source.tobytes() == first.tobytes() and len(calls) == 2
    execution.oracle_profile(panel, incumbent, .52, cache)
    assert len(calls) == 3


def test_reference_movement_cache_preserves_independent_exact_objective(monkeypatch):
    incumbent, forecast, panel, config = toy()
    candidate = list(incumbent)
    candidate[0], candidate[1] = candidate[1], candidate[0]
    original = execution.migration_profile
    expected = original(incumbent, candidate)
    calls = []
    def checked(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(execution, "migration_profile", checked)
    cache = {}
    first = execution.oracle_movement(incumbent, candidate, cache)
    assert first == {"count": expected["count"], "cost": expected["cost"]}
    assert execution.oracle_movement(incumbent, tuple(candidate), cache) is first
    assert len(calls) == 1
