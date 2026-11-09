"""Budget exhaustion must not start an additional search/solver operation."""
from types import SimpleNamespace

import numpy as np
import pytest

from cara import baseline_solver, refinement


def workload():
    return ((0, 0, 1, 1), np.full((4, 2), .4),
            np.tile(np.array([[.7, .1], [.7, .1], [.1, .7], [.1, .7]]), (4, 1, 1)))


def frozen_clock(after):
    first = True
    def clock():
        nonlocal first
        if first:
            first = False
            return 0.
        return after
    return clock


@pytest.mark.parametrize("frontier", [False, True])
def test_exhausted_preparation_starts_no_frontier_or_initial_batch(monkeypatch, frontier):
    inc, forecast, panel = workload()
    monkeypatch.setattr(refinement, "process_time", frozen_clock(2.))
    def forbidden(*args, **kwargs):
        raise AssertionError("An operation began after the allowance was exhausted")
    monkeypatch.setattr(refinement, "search_swap", forbidden)
    monkeypatch.setattr(refinement, "_destroy_repair", forbidden)
    result = refinement.search_refined(inc, forecast, panel, panel, 1.,
        {"cpu_limit": 1., "frontier_seed": frontier, "initial_broad": 64}, np.random.default_rng(1))
    assert result["selected"] is None
    assert result["stats"]["broad_trials"] == 0
    assert result["stats"]["initial_broad_batches"] == 0
    assert result["stats"]["history"] == []
    assert result["stats"]["proposal_overshoot_seconds"] == 1.


def test_frontier_receives_only_remaining_outer_allowance(monkeypatch):
    inc, forecast, panel = workload()
    monkeypatch.setattr(refinement, "process_time", frozen_clock(.8))
    observed = []
    def frontier(inc, forecast, build, selection, capacity, config, rng, **kwargs):
        observed.append(config["cpu_limit"])
        return {"frontier_assignment": inc, "stats": {
            "attempts": 0, "accepted_steps": 0, "evaluations": 0,
            "cache_hits": 0, "projection_hits": 0, "changed_values": 0, "full_values": 0}}
    monkeypatch.setattr(refinement, "search_swap", frontier)
    refinement.search_refined(inc, forecast, panel, panel, 1.,
        {"cpu_limit": 1., "frontier_seed": True, "frontier_only": True}, np.random.default_rng(2))
    assert observed == pytest.approx([.2])


def test_fixed_work_replay_retains_recorded_frontier_after_clock_exhaustion(monkeypatch):
    inc, forecast, panel = workload()
    monkeypatch.setattr(refinement, "process_time", frozen_clock(2.))
    observed = []
    def frontier(inc, forecast, build, selection, capacity, config, rng, **kwargs):
        observed.append(config["max_attempts"])
        return {"frontier_assignment": inc, "stats": {
            "attempts": 0, "accepted_steps": 0, "evaluations": 0,
            "cache_hits": 0, "projection_hits": 0, "changed_values": 0, "full_values": 0}}
    monkeypatch.setattr(refinement, "search_swap", frontier)
    result = refinement.search_refined(inc, forecast, panel, panel, 1.,
        {"cpu_limit": 1., "frontier_seed": True, "replay_arms": ["frontier:0"]},
        np.random.default_rng(3))
    assert observed == [0]
    assert result["stats"]["history"][0]["arm"] == "frontier:0"


def test_exhausted_mip_setup_never_launches_solver(monkeypatch):
    inc, forecast, panel = workload()
    monkeypatch.setattr(baseline_solver, "process_time", frozen_clock(2.))
    def forbidden(*args, **kwargs):
        raise AssertionError("MIP solver was launched after setup exhausted the allowance")
    monkeypatch.setattr(baseline_solver, "milp", forbidden)
    result = baseline_solver.scenario_mip(inc, forecast, panel, panel, 1., {"cpu_limit": 1.})
    assert result["selected"] is None
    assert result["stats"]["solver_started"] is False
    assert result["stats"]["solver_wall_allowance_seconds"] == 0.
    assert result["stats"]["model_construction_cpu_seconds"] == 2.


def test_mip_solver_receives_remaining_allowance_and_original_options(monkeypatch):
    inc, forecast, panel = workload()
    monkeypatch.setattr(baseline_solver, "process_time", frozen_clock(.8))
    observed = []
    def solver(*args, **kwargs):
        observed.append(kwargs["options"])
        return SimpleNamespace(x=None, status=1, message="Time limit", fun=None)
    monkeypatch.setattr(baseline_solver, "milp", solver)
    result = baseline_solver.scenario_mip(inc, forecast, panel, panel, 1., {"cpu_limit": 1.})
    assert observed[0]["time_limit"] == pytest.approx(.2)
    assert observed[0]["mip_rel_gap"] == .01
    assert observed[0]["threads"] == 1
    assert observed[0]["presolve"] is True
    assert result["stats"]["solver_started"] is True
