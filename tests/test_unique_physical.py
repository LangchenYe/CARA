"""Exact rich physical actions from strict family certificates."""
from copy import deepcopy

import numpy as np
import pytest

from cara.data import canonical_assignment, host_memberships
from cara.family_certificates import ResidualFamily
from cara.refinement import migration_profile, search_refined


def family(incumbent, seed, pair, mutable, **options):
    return ResidualFamily(incumbent, seed, pair, mutable, matching_backend="compiled",
                          unique_physical_min_hosts=0, **options)


def rich_signature(record):
    return tuple((name, value.hex() if isinstance(value, float) else value)
                 for name, value in sorted(record.items()))


def check_members(model):
    fast = 0
    for mask in model.nonempty_masks():
        assignment = model.assignment(mask)
        expected = migration_profile(model.incumbent, assignment)
        actual = model.unique_physical_profile(mask, assignment)
        if actual is not None:
            assert rich_signature(actual) == rich_signature(expected)
            assert actual["count"] == model.movement_count(mask)
            fast += 1
    if model._unique_physical is not None:
        assert len(model._unique_physical.permutations) <= 2
        assert all(not value.flags.writeable for value in model._unique_physical.permutations.values())
    return fast


def test_random_complete_families_preserve_all_rich_fields():
    rng = np.random.default_rng(88216)
    hits = 0
    for hosts in (2, 3, 5, 8):
        for residents in (3, 8):
            incumbent = canonical_assignment(np.arange(hosts * residents) % hosts)
            ring = list(incumbent)
            for host, group in enumerate(host_memberships(incumbent)):
                ring[group[-1]] = (host + 1) % hosts
            for seed in (incumbent, canonical_assignment(ring), canonical_assignment(rng.permutation(incumbent))):
                pair = (hosts - 1, 0)
                groups = host_memberships(seed)
                mutable = tuple(sorted(groups[pair[0]][:2] + groups[pair[1]][:2]))
                hits += check_members(family(incumbent, seed, pair, mutable))
    assert hits > 100


def test_same_signature_never_caches_vm_fields_or_canonical_order():
    incumbent = (0, 1, 0, 1, 0, 1, 0, 1)
    model = family(incumbent, incumbent, (0, 1), (0, 1, 2, 3))
    outputs = []
    for mask in (3, 6):
        assignment = model.assignment(mask)
        record = model.unique_physical_profile(mask, assignment)
        assert record == migration_profile(incumbent, assignment)
        outputs.append(record)
    assert model._left_signature(3) == model._left_signature(6)
    assert outputs[0]["moved_vms"] != outputs[1]["moved_vms"]
    assert model.stats["physical_unique_signature_hits"] == 1


def test_two_orientations_ties_and_seed_state_remain_separate():
    incumbent = (0, 1) * 4
    model = family(incumbent, incumbent, (1, 0), tuple(range(8)))
    original = model._factor._incumbent_plan.base.copy()
    assert check_members(model) > 0
    assert len(model._unique_physical.permutations) == 2
    assert model.stats["physical_unique_ties"] > 0
    assert np.array_equal(model._factor._incumbent_plan.base, original)
    assert any(choice is None for choice in model._unique_physical.choices.values())


def test_weak_family_lock_and_large_global_core_use_original_policy():
    for residents, moved in ((3, 1), (8, 4)):
        incumbent = canonical_assignment(np.arange(3 * residents) % 3)
        seed = list(incumbent)
        for host, group in enumerate(host_memberships(incumbent)):
            for vm in group[-moved:]:
                seed[vm] = (host + 1) % 3
        model = family(incumbent, seed, (0, 1), (0, 1))
        assert check_members(model) == 0
        assert not model._unique_physical.eligible


@pytest.mark.parametrize("backend", ("factorized", "dense", "sparse", "replay", "locked"))
def test_count_controls_do_not_enable_physical_optimization(backend):
    incumbent = (0, 1, 2) * 8
    model = ResidualFamily(incumbent, incumbent, (0, 1), (0, 1, 3, 4),
                           matching_backend=backend, use_unique_physical=True,
                           unique_physical_min_hosts=0)
    assert not model.use_unique_physical
    assert model.unique_physical_profile(0, model.assignment(0)) is None
    assert model._unique_physical is None


def test_host_gate_explicit_ablation_and_explicit_weights():
    incumbent = (0, 1, 2) * 8
    arguments = incumbent, incumbent, (0, 1), (0, 1, 3, 4)
    for options in ({}, {"unique_physical_min_hosts": 0, "use_unique_physical": False}):
        model = ResidualFamily(*arguments, matching_backend="compiled", **options)
        assert model.unique_physical_profile(0, model.assignment(0)) is None
        assert model._unique_physical is None
    model = family(*arguments)
    assert model.unique_physical_profile(0, model.assignment(0), np.ones(len(incumbent))) is None
    assert model._unique_physical is None
    with pytest.raises(ValueError, match="nonnegative integer"):
        ResidualFamily(*arguments, matching_backend="compiled", unique_physical_min_hosts=.5)


def test_adaptive_consumers_keep_rich_pools_and_single_factor_controls():
    from benchmark_family_execution import encoded, PLANS
    from benchmark_family_queries import run_plan
    rng = np.random.default_rng(791117)
    incumbent = canonical_assignment(np.arange(8 * 8) % 8)
    forecast = np.full((len(incumbent), 3), .04)
    panel = rng.uniform(.015, .12, size=(5, len(incumbent), 3))
    config = {"top_p": 4, "migration_fraction": .2, "archive_mode": "balanced",
              "score_mode": "gain", "family_score_min_hosts": 0,
              "family_score_min_leaves": 1, "family_score_strategy": "leaf",
              "family_forecast_strategy": "leaf", "family_progressive_batch": 3,
              "family_matching_backend": "compiled", "family_unique_physical_min_hosts": 0,
              "query_batch_values": 1024}
    reference, results = None, {}
    for plan in ("eager", "components", "sparse", "static", "replay", "locked",
                 "compiled_no_unique_maps", "compiled", "no_late_matching"):
        snapshots, stats = run_plan(panel, forecast, incumbent, .51, config, plan,
                                   adaptive=True, families=8, width=4)
        value = encoded(snapshots)
        reference = value if reference is None else reference
        assert value == reference
        results[plan] = {key: sum(row["counters"].get(key, 0) for row in stats["families"])
                         for key in ("physical_matching_calls", "physical_unique_profiles")}
    assert PLANS["static"]["family_unique_physical"] is False
    assert results["compiled"]["physical_unique_profiles"] > 0
    assert results["no_late_matching"]["physical_unique_profiles"] >= results["compiled"]["physical_unique_profiles"]
    for plan in ("components", "sparse", "static", "replay", "locked", "compiled_no_unique_maps"):
        assert results[plan]["physical_unique_profiles"] == 0
    assert sum(results["compiled"].values()) == sum(results["compiled_no_unique_maps"].values())


@pytest.mark.parametrize("weighted", (False, True))
def test_controller_selection_and_every_family_pool_match(weighted):
    rng = np.random.default_rng(29)
    panel = rng.uniform(.1, .5, (8, 24, 4))
    forecast = np.full((24, 4), .1)
    incumbent = tuple(np.tile(np.arange(4), 6))
    config = {"cpu_limit": .1, "initial_broad": 0, "mutable_limit": 4,
              "top_p": 4, "migration_fraction": .5, "diagnostic": False,
              "archive_mode": "balanced", "seed_schedule": "coverage",
              "score_mode": "gain", "shared_queries": True, "batch_queries": True,
              "family_execution": "certified", "family_matching_backend": "compiled",
              "family_unique_physical_min_hosts": 0, "family_certificate_score": False,
              "replay_arms": ["exact", "broad", "exact", "exact", "broad", "exact"],
              "broad_batch": 4}
    if weighted:
        config["migration_weights"] = np.linspace(.25, 2., 24)
    previous = None
    for enabled, defer in ((False, True), (True, True), (True, False)):
        snapshots = []
        settings = {**config, "family_unique_physical": enabled, "family_defer_physical": defer,
                    "family_observer": lambda records, details: snapshots.append(deepcopy(records))}
        result = search_refined(incumbent, forecast, panel, panel.copy(), 1., settings, np.random.default_rng(7))
        observed = (snapshots, *(result[key] for key in ("candidates", "selected", "movement", "selection_reason")))
        previous = observed if previous is None else previous
        assert observed == previous
        if weighted:
            assert result["stats"]["family_work"].get("physical_unique_profiles", 0) == 0
