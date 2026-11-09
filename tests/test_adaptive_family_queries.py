"""Persistent adaptive query trajectories against independent eager outputs."""
import numpy as np
import pytest

import benchmark_family_execution as execution
import benchmark_family_queries as queries
from benchmark_hourly_queries import family_pairs
from cara.data import canonical_assignment, host_memberships


def source(hosts=16):
    rng = np.random.default_rng(791117)
    incumbent = canonical_assignment(np.arange(hosts * 8) % hosts)
    forecast = np.full((len(incumbent), 3), .04)
    panel = rng.uniform(.015, .12, size=(5, len(incumbent), 3))
    config = {"top_p": 4, "migration_fraction": .2, "archive_mode": "balanced",
              "score_mode": "gain", "family_score_min_hosts": 0,
              "family_score_min_leaves": 1, "family_score_strategy": "leaf",
              "family_forecast_strategy": "leaf", "family_progressive_batch": 3,
              "query_batch_values": 1024}
    return incumbent, forecast, panel, config


def check_trajectory(incumbent, initial, snapshots, stats, width, families):
    seed = initial
    schedule = family_pairs(len(set(incumbent)), families)
    assert len(snapshots) == len(stats["families"]) == len(schedule)
    for pair, snapshot, row in zip(schedule, snapshots, stats["families"]):
        assert row["seed_assignment"] == seed
        assert row["pair"] == pair
        groups = host_memberships(seed)
        taken = min(width // 2, len(groups[pair[0]]), len(groups[pair[1]]))
        mutable = tuple(sorted(groups[pair[0]][:taken] + groups[pair[1]][:taken]))
        assert row["mutable"] == mutable
        if snapshot:
            seed = snapshot[0]["assignment"]


def test_six_plans_preserve_sixteen_adaptive_rich_pools_and_seed_paths():
    incumbent, forecast, panel, config = source()
    expected = expected_seeds = None
    cache = {}
    for plan in ("eager", "components", "sparse", "replay", "locked", "compiled"):
        counters = {}
        def witness(value):
            execution.verify_witness(value, incumbent, forecast, panel, .51, counters, cache)
        with execution.profile_oracle(counters, cache):
            snapshots, stats = queries.run_plan(panel, forecast, incumbent, .51, config, plan,
                adaptive=True, families=16, width=4, certificate_observer=witness)
        check_trajectory(incumbent, incumbent, snapshots, stats, 4, 16)
        assert len(snapshots) == 16
        assert len({row["seed_assignment"] for row in stats["families"]}) > 1
        assert any(row["seed_assignment"] != incumbent for row in stats["families"])
        actual = execution.encoded(snapshots)
        seeds = execution.encoded([(row["seed_assignment"], row["pair"], row["mutable"])
                                  for row in stats["families"]])
        if expected is None:
            expected, expected_seeds = actual, seeds
        assert actual == expected
        assert seeds == expected_seeds
        for reason in ("forecast", "migration", "score"):
            excluded = sum(row["counters"].get(reason + "_excluded_masks", 0)
                           for row in stats["families"])
            assert counters.get(reason + "_leaves", 0) == excluded
        assert counters["materialized_profiles"] > 0


def test_fixed_default_and_explicit_incumbent_seed_return_same_pools():
    incumbent, forecast, panel, config = source(4)
    common = (panel, forecast, incumbent, .51, config, "compiled")
    default, default_stats = queries.run_plan(*common, families=3, width=4)
    explicit, explicit_stats = queries.run_plan(*common, seed=incumbent, adaptive=False,
                                               families=3, width=4)
    assert execution.encoded(default) == execution.encoded(explicit)
    assert not default_stats["adaptive"]
    assert all("seed_assignment" not in row for row in default_stats["families"])
    assert all(row["seed_assignment"] == incumbent for row in explicit_stats["families"])


def test_empty_pool_keeps_previous_seed_without_resetting_session():
    incumbent, forecast, panel, config = source(4)
    config["max_migrations"] = 0
    snapshots, stats = queries.run_plan(panel, forecast, incumbent, .51, config,
                                      "compiled", adaptive=True, families=4, width=4)
    assert all(not snapshot for snapshot in snapshots)
    assert all(row["seed_assignment"] == incumbent for row in stats["families"])
    check_trajectory(incumbent, incumbent, snapshots, stats, 4, 4)


def test_adaptation_from_a_mixed_seed_preserves_original_reference_and_movement():
    incumbent, forecast, panel, config = source(6)
    seed = list(incumbent)
    for host, residents in enumerate(host_memberships(incumbent)):
        seed[residents[-1]] = (host + 1) % 6
    seed = canonical_assignment(seed)
    outputs = []
    for plan in ("eager", "compiled"):
        snapshots, stats = queries.run_plan(panel, forecast, incumbent, .51, config, plan,
            seed=seed, adaptive=True, families=4, width=4)
        check_trajectory(incumbent, seed, snapshots, stats, 4, 4)
        for snapshot in snapshots:
            for record in snapshot:
                assert record["movement"] == execution.migration_profile(incumbent, record["assignment"])
        outputs.append(execution.encoded(snapshots))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("seed", [(0, 1), (0,) * 32])
def test_invalid_seed_shape_or_host_count_is_rejected(seed):
    incumbent, forecast, panel, config = source(4)
    with pytest.raises(ValueError, match="complete incumbent"):
        queries.run_plan(panel, forecast, incumbent, .51, config, "compiled", seed=seed)

