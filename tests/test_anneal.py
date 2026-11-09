"""Annealing uses the common feasible action space and held-out selector."""
import numpy as np
import pytest

from cara.baseline_anneal import search_anneal
from cara.refinement import migration_profile, panel_profile
from cara.repair import feasible


def example():
    panel = np.tile(np.array([[.7, .1], [.7, .1], [.1, .7], [.1, .7]]), (8, 1, 1))
    return (0, 0, 1, 1), np.full((4, 2), .4), panel


def test_annealing_repairs_complementarity_and_preserves_global_constraints():
    incumbent, forecast, panel = example()
    result = search_anneal(incumbent, forecast, panel, panel.copy(), 1.,
        {"cpu_limit": .1, "max_attempts": 200, "max_migrations": 2}, np.random.default_rng(7))
    assert result["selected"] is not None
    for candidate in result["candidates"]:
        assert feasible(candidate["assignment"], forecast, 1., 2)
        assert migration_profile(incumbent, candidate["assignment"])["count"] <= 2
        assert np.array_equal(candidate["build_means"], panel_profile(panel, candidate["assignment"], 1.).mean(axis=0))
    assert result["movement"]["count"] == 2
    assert result["stats"]["attempts"] == 200


def test_zero_migration_budget_keeps_incumbent():
    incumbent, forecast, panel = example()
    result = search_anneal(incumbent, forecast, panel, panel, 1.,
        {"max_attempts": 50, "max_migrations": 0}, np.random.default_rng(9))
    assert result["selected"] is None and result["candidates"] == []
    assert result["stats"]["migration_rejections"] > 0


def test_fixed_attempt_annealing_replays_across_canonical_view_plans():
    incumbent, forecast, panel = example()
    results = [search_anneal(incumbent, forecast, panel, panel.copy(), 1.,
        {"max_attempts": 100, "max_migrations": 2, "reuse_views": reuse,
         "projected_views": projected}, np.random.default_rng(22))
        for reuse, projected in ((False, False), (True, False), (True, True))]
    assert all(result["candidates"] == results[0]["candidates"] for result in results[1:])
    assert all(result["selected"] == results[0]["selected"] for result in results[1:])
    with pytest.raises(ValueError, match="temperature"):
        search_anneal(incumbent, forecast, panel, panel, 1., {"anneal_temperature_fraction": 0}, np.random.default_rng(1))

