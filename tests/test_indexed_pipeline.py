"""Whole-family semantics with all four compiled query modules enabled."""
import json
from pathlib import Path

import numpy as np

from benchmark_family_execution import encoded, profile_oracle, verify_witness
from benchmark_family_generality import make_mixed_seed
from benchmark_family_queries import run_plan
from cara.data import canonical_assignment


def inputs():
    rng = np.random.default_rng(20261119)
    hosts, residents, scenes, horizon = 16, 8, 5, 3
    incumbent = canonical_assignment(np.arange(hosts * residents) % hosts)
    # Common factors and VM-specific bursts give nonconstant native metrics.
    common = rng.uniform(.01, .06, (scenes, 1, horizon))
    phases = rng.uniform(.01, .13, (1, len(incumbent), 1))
    panel = common + phases + rng.exponential(.025, (scenes, len(incumbent), horizon))
    forecast = np.full((len(incumbent), horizon), .1)
    capacity = 1.
    seed, _ = make_mixed_seed(incumbent, forecast, capacity, "full", residents=1)
    base = Path(__file__).resolve().parents[1]
    config = json.loads((base / "configs/release/family_frozen.json").read_text())
    config.update(top_p=4, family_progressive_batch=4, family_score_min_hosts=0,
                  family_score_min_leaves=8, family_unique_physical_min_hosts=0,
                  query_cache_views=3, query_cache_bytes=1 << 20)
    return panel, forecast, incumbent, capacity, seed, config


def compare_complete_queries(*, adaptive, families):
    panel, forecast, incumbent, capacity, seed, config = inputs()
    oracle_cache, expected, expected_path = {}, None, None
    indexed_stats = None
    for plan in ("eager", "family_only", "indexed"):
        counters, path = {}, []

        def witness(value):
            verify_witness(value, incumbent, forecast, panel, capacity, counters, oracle_cache)

        def observe(records, metadata):
            path.append((metadata["seed_assignment"], metadata["pair"], metadata["mutable"]))

        with profile_oracle(counters, oracle_cache):
            snapshots, stats = run_plan(panel, forecast, incumbent, capacity, config, plan,
                seed=seed, adaptive=adaptive, families=families, width=8,
                observer=observe, certificate_observer=witness)
        actual = encoded(snapshots)
        trace = encoded(path)
        if expected is None:
            expected, expected_path = actual, trace
        else:
            assert actual == expected
            assert trace == expected_path
        assert len(snapshots) == families
        assert counters["materialized_profiles"] > 0
        for reason in ("forecast", "migration", "score"):
            excluded = sum(row["counters"].get(reason + "_excluded_masks", 0)
                           for row in stats["families"])
            assert counters.get(reason + "_leaves", 0) == excluded
        for pool in snapshots:
            for record in pool:
                assert "aligned_assignment" in record["movement"]
                assert "moved_vms" in record["movement"]
        if plan == "indexed":
            indexed_stats = stats
    work = [row["counters"] for row in indexed_stats["families"]]
    # These checks ensure all four new modules actually ran in the regression.
    assert sum(row.get("matching_seed_compilations", 0) for row in work) > 0
    assert indexed_stats["certificate_views"]["family_queries"] > 0
    assert sum(row.get("indicator_bound_nodes", 0) for row in work) > 0
    assert sum(row.get("mutable_sum_builds", 0) for row in work) > 0
    return indexed_stats


def test_all_compiled_modules_preserve_fixed_mixed_seed_families():
    stats = compare_complete_queries(adaptive=False, families=8)
    assert sum(row["counters"].get("matching_seed_reuses", 0) for row in stats["families"]) == 7
    assert stats["certificate_views"]["seed_updates"] == 0


def test_all_compiled_modules_preserve_adaptive_seeds_and_rich_records():
    stats = compare_complete_queries(adaptive=True, families=16)
    seeds = {tuple(row["seed_assignment"]) for row in stats["families"]}
    assert len(seeds) > 1
    assert stats["certificate_views"]["seed_updates"] > 0
