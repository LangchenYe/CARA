"""Selection dispersion is an empirical preference, not a confidence bound."""
import numpy as np
import pytest

import cara.refinement as refinement


def selection_example(monkeypatch):
    incumbent = (0, 0, 1, 1)
    stable, variable = (0, 1, 0, 1), (0, 1, 1, 0)
    reference = np.ones((4, 6))
    profiles = {incumbent: reference, stable: reference.copy(), variable: reference.copy()}
    profiles[stable][:, 3] -= .10
    profiles[variable][:, 3] -= np.array([0., .22, 0., .22])
    monkeypatch.setattr(refinement, "panel_profile", lambda panel, assignment, capacity: profiles[tuple(assignment)])
    records = [{"assignment": assignment, "movement": {"count": 2, "cost": 2.}}
               for assignment in (stable, variable)]
    config = {"objective_names": ["raw"], "selection_guard_names": ["raw"]}
    return incumbent, stable, variable, records, config


def test_positive_dispersion_prefers_stable_paired_gain_with_similar_mean(monkeypatch):
    incumbent, stable, variable, records, config = selection_example(monkeypatch)
    unpenalized, _ = refinement._select(records, incumbent, None, 1., config)
    penalized, details = refinement._select(records, incumbent, None, 1.,
                                           {**config, "selection_dispersion": 1.})
    assert unpenalized["assignment"] == variable
    assert penalized["assignment"] == stable
    assert details["selection_guard_passed"] == 2


def test_zero_penalty_keeps_original_selector_and_negative_is_rejected(monkeypatch):
    incumbent, _, variable, records, config = selection_example(monkeypatch)
    original = refinement._select(records, incumbent, None, 1., config)
    zero = refinement._select(records, incumbent, None, 1., {**config, "selection_dispersion": 0.})
    assert original == zero and zero[0]["assignment"] == variable
    with pytest.raises(ValueError, match="nonnegative"):
        refinement._select(records, incumbent, None, 1., {**config, "selection_dispersion": -.1})


def test_real_cpu_frontier_replays_with_observed_cooling_and_family_costs():
    from benchmark_query_engine import _run_path
    from benchmark_revised_execution import result_signature

    rng = np.random.default_rng(7719)
    incumbent = tuple(np.repeat(np.arange(4), 3))
    forecast = np.full((12, 8), .25)
    build = rng.uniform(.1, .55, (16, 12, 8))
    selection = rng.uniform(.1, .55, (16, 12, 8))
    config = {"cpu_limit": .12, "top_p": 8, "mutable_limit": 4,
        "migration_fraction": .5, "initial_broad": 0, "diagnostic": False,
        "frontier_seed": True, "frontier_engine": "anneal", "frontier_objective": "joint",
        "frontier_fraction": .35, "seed_schedule": "coverage", "archive_mode": "coverage",
        "feedback_queries": True, "shared_queries": True, "batch_queries": True,
        "selection_dispersion": .25, "anneal_temperature_fraction": .01}
    # The source is an actual time-budget execution, without a prescribed path
    # or attempt-based cooling. Its observed controller inputs are then frozen.
    source = refinement.search_refined(incumbent, forecast, build, selection, 1.,
                                       config, np.random.default_rng(77))
    history = source["stats"]["history"]
    assert len(history[0]["cooling_progress"]) == int(history[0]["arm"].split(":")[1]) > 0
    assert source["stats"]["blocks"] > 0
    case = {"config": config, "incumbent": incumbent, "forecast": forecast,
            "build": build, "selection": selection, "capacity": 1., "seed": 77,
            "history": history}
    for plan in ("full", "shared_batch"):
        replay = _run_path(case, plan)
        assert result_signature(replay) == result_signature(source)
