"""Independent canonical oracle checks; no external data or timings required."""
import gc
import weakref

import numpy as np
import pytest

import benchmark_family_execution as execution
from benchmark_family_queries import run_plan
from benchmark_reference_oracle import CanonicalProfileOracle
from cara.data import host_loads, host_memberships
from cara.query_engine import SharedQueryReplay
from cara.refinement import panel_profile


def identical(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("layout", ("c", "fortran", "strided"))
@pytest.mark.parametrize("values", ("zero", "subnormal", "threshold", "rounding"))
def test_reference_profiles_and_cached_loads_preserve_bits(layout, values):
    rng = np.random.default_rng(20261107)
    incumbent = tuple(np.arange(40) % 5)
    changed = list(incumbent)
    changed[-2], changed[-1] = changed[-1], changed[-2]
    for scenes, horizon in ((1, 1), (7, 17)):
        shape, capacity = (scenes, 40, horizon), 1.
        if values == "zero":
            panel = np.zeros(shape)
        elif values == "subnormal":
            unit = np.nextafter(0., 1.)
            panel, capacity = rng.integers(0, 17, size=shape).astype(float) * unit, 7 * unit
        elif values == "threshold":
            panel = rng.choice([np.nextafter(1. + 1e-12, 0.), 1. + 1e-12,
                                np.nextafter(1. + 1e-12, 2.)], size=shape) / 8
        else:
            panel = np.ldexp(rng.random(shape), rng.integers(-100, 53, size=shape))
            capacity = float(2**49)
        if layout == "fortran":
            panel = np.asfortranarray(panel)
        elif layout == "strided":
            panel = panel[:, :, ::-1]
        oracle = CanonicalProfileOracle(panel, capacity)
        for assignment in (incumbent, tuple(changed)):
            expected = panel_profile(panel, assignment, capacity)
            identical(oracle.profile(assignment), expected)
            identical(oracle.profile(assignment), expected)
            loads = host_loads(panel, assignment)
            for host, group in enumerate(host_memberships(assignment)):
                identical(oracle.groups[group][0], loads[host])
        assert len(oracle.groups) == 7


def test_reference_cache_scopes_and_original_no_cache_path(monkeypatch):
    panel = np.random.default_rng(91).random((3, 40, 8))
    assignment, capacity, cache = tuple(np.arange(40) % 5), 4., {}
    first = execution.oracle_profile(panel, assignment, capacity, cache)
    identical(first, panel_profile(panel, assignment, capacity))
    assert not first.flags.writeable
    assert execution.oracle_profile(panel, assignment, capacity, cache) is first
    copied_panel, other_capacity = panel.copy(), float(np.nextafter(capacity, np.inf))
    identical(execution.oracle_profile(copied_panel, assignment, capacity, cache), first)
    identical(execution.oracle_profile(panel, assignment, other_capacity, cache),
              panel_profile(panel, assignment, other_capacity))
    assert sum(isinstance(value, CanonicalProfileOracle) for value in cache.values()) == 3

    def forbidden(*args, **kwargs):
        raise AssertionError("The uncached reference path must retain panel_profile")

    monkeypatch.setattr(execution, "CanonicalProfileOracle", forbidden)
    identical(execution.oracle_profile(panel, assignment, capacity), first)


def test_reference_scalar_group_hooks_and_cache_release():
    panel = np.random.default_rng(12).random((3, 40, 8))
    assignment, capacity, cache, counts = tuple(np.arange(40) % 5), 4., {}, {}
    reader = SharedQueryReplay(panel, assignment, capacity)
    original_group, original_scalar = SharedQueryReplay.profiles_groups, SharedQueryReplay.profile
    expected = panel_profile(panel, assignment, capacity)
    with execution.profile_oracle(counts, cache):
        identical(reader.profile(assignment), expected)
        actual = reader.profiles_groups([host_memberships(assignment)] * 2)
        identical(actual[0], expected)
        identical(actual[1], expected)
    assert SharedQueryReplay.profiles_groups is original_group
    assert SharedQueryReplay.profile is original_scalar
    assert counts["materialized_profiles"] == 3
    references = [weakref.ref(value) for value in cache.values()
                  if isinstance(value, CanonicalProfileOracle)]
    assert references
    cache.clear()
    gc.collect()
    assert all(reference() is None for reference in references)


def test_reference_complete_pools_excluded_leaves_and_untimed_scope(monkeypatch):
    rng = np.random.default_rng(79)
    incumbent, forecast = tuple(np.arange(16) % 4), np.full((16, 3), .1)
    panel, capacity = rng.uniform(.015, .2, (5, 16, 3)), .51
    config = {"top_p": 2, "migration_fraction": .125, "archive_mode": "balanced",
              "score_mode": "gain", "family_score_min_leaves": 1, "family_score_strategy": "leaf"}
    expected, cache, excluded_total = None, {}, 0
    for plan in ("eager", "components", "static", "replay", "locked", "compiled"):
        counts = {}

        def witness(value):
            execution.verify_witness(value, incumbent, forecast, panel, capacity, counts, cache)

        with execution.profile_oracle(counts, cache):
            snapshots, statistics = run_plan(panel, forecast, incumbent, capacity, config, plan,
                                             families=3, width=4, certificate_observer=witness)
        signature = execution.encoded(snapshots)
        expected = signature if expected is None else expected
        assert signature == expected and counts["materialized_profiles"] > 0
        for reason in ("forecast", "migration", "score"):
            excluded = sum(row["counters"].get(reason + "_excluded_masks", 0)
                           for row in statistics["families"])
            assert counts.get(reason + "_leaves", 0) == excluded
            excluded_total += excluded
    assert excluded_total > 0
    references = [weakref.ref(value) for value in cache.values()
                  if isinstance(value, CanonicalProfileOracle)]
    cache.clear()
    gc.collect()
    assert not cache and references and all(reference() is None for reference in references)

    def forbidden(*args, **kwargs):
        raise AssertionError("Verification entered the observer-free execution phase")

    monkeypatch.setattr(execution, "oracle_profile", forbidden)
    snapshots, _ = run_plan(panel, forecast, incumbent, capacity, config, "compiled", families=3, width=4)
    assert execution.encoded(snapshots) == expected
