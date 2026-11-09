"""Protocol coverage, budget charging and target-access checks for the runner."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

import run_family_study as study


@pytest.fixture
def protocol():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/release/family_protocol.json").read_text())


def test_complete_views_share_cases_without_rerunning(protocol):
    cases = study.case_grid(protocol, "evaluation")
    assert len(cases) == len({study.case_key(case) for case in cases}) == 354
    assert sum("primary" in case["views"] for case in cases) == 300
    assert sum("longitudinal" in case["views"] for case in cases) == 78
    assert sum(len(case["views"]) == 2 for case in cases) == 24
    for dataset, metadata in protocol["decision_cohorts"].items():
        daily = [case["origin"] for case in cases if case["dataset"] == dataset
                 and case["seed_index"] == 0 and "longitudinal" in case["views"]]
        assert daily[0] == 264
        assert all(right - left == 24 for left, right in zip(daily, daily[1:]))
        assert daily[-1] + 24 <= metadata["minimum_hours"]
    assert max(case["origin"] + 24 for case in study.case_grid(protocol, "development")) <= 264
    with pytest.raises(ValueError, match="all three"):
        study.case_grid(protocol, "evaluation", [0])


class SmallData:
    capacity = 4.

    def window_panel(self, origin, older, newer, horizon, stride):
        return np.zeros(((older - newer - horizon) // stride + 1, 2, horizon))


def test_input_preparation_reduces_engine_allowance(monkeypatch, protocol):
    clock = iter([10., 10.3, 10.3, 11.4])
    monkeypatch.setattr(study, "process_time", lambda: next(clock))
    observed = {}

    def engine(incumbent, forecast, build, selection, capacity, config, rng):
        observed.update(config)
        assert build.shape == (121, 2, 8)
        assert selection.shape == (25, 2, 8)
        return {"selected": None, "stats": {"cpu_seconds": 1., "selection_cpu_seconds": .1}}

    monkeypatch.setattr(study, "engine_for", lambda method: engine)
    result = study.prepare_and_search(SmallData(), 216, (0, 1), np.zeros((2, 24)), "CARA", {}, 7, protocol, 2.)
    assert observed["cpu_limit"] == pytest.approx(1.7)
    assert result["preparation_cpu_seconds"] == pytest.approx(.3)
    assert result["search_call_cpu_seconds"] == pytest.approx(1.1)


def test_exhausted_preparation_does_not_start_search(monkeypatch, protocol):
    clock = iter([0., 2.5, 2.5, 2.5001])
    monkeypatch.setattr(study, "process_time", lambda: next(clock))

    def forbidden(method):
        raise AssertionError("An exhausted proposal allowance started a search")

    monkeypatch.setattr(study, "engine_for", forbidden)
    result = study.prepare_and_search(SmallData(), 216, (0, 1), np.zeros((2, 24)), "CARA", {}, 7, protocol, 2.)
    assert result["selected"] is None
    assert result["selection_reason"] == "preparation_exhausted"
    assert result["remaining_proposal_allowance_seconds"] == 0.


def test_silently_ignored_execution_mode_is_rejected(monkeypatch, protocol):
    def old_engine(*args):
        return {"selected": None, "stats": {"cpu_seconds": 0., "selection_cpu_seconds": 0.}}

    monkeypatch.setattr(study, "search_refined", old_engine)
    with pytest.raises(RuntimeError, match="execution mode"):
        study.prepare_and_search(SmallData(), 216, (0, 1), np.zeros((2, 24)), "CARA", {}, 7, protocol, 2.)


def test_physical_unit_cost_summary_must_match_action():
    incumbent, selected = (0, 0, 1, 1), (0, 1, 0, 1)
    movement = {"aligned_assignment": selected, "moved_vms": [1, 2], "count": 2, "fraction": .5, "cost": 3.}
    with pytest.raises(AssertionError, match="movement summary"):
        study.validate_selected(incumbent, selected, np.full((4, 24), .1), 1., {"migration_fraction": 1.}, movement)


def test_targets_are_sliced_after_all_method_decisions(monkeypatch, protocol, tmp_path):
    protocol = deepcopy(protocol)
    protocol["decision_cohorts"]["azure2019"]["vms"] = 2
    completed, target_accesses = [], []

    class Data:
        n_vms, min_length, capacity = 2, 355, 4.

        def forecast(self, origin, horizon):
            return np.full((2, horizon), .1)

        def panel(self, origin, horizon=24):
            if origin == 288:
                target_accesses.append(tuple(completed))
                assert completed == ["CARA", "Eager-CARA"]
            return np.full((2, horizon), .2)

    def prepared(data, origin, incumbent, forecast, method, config, seed, protocol, budget):
        completed.append(method)
        return {"selected": None, "candidates": [], "stats": {"cpu_seconds": .2, "selection_cpu_seconds": .01,
                "evaluations": 0, "blocks": 0, "history": []}, "preparation_cpu_seconds": .01,
                "remaining_proposal_allowance_seconds": 1.99, "search_call_cpu_seconds": .21}

    monkeypatch.setattr(study, "DATA", {"azure2019": Data()})
    monkeypatch.setattr(study, "prepare_and_search", prepared)
    monkeypatch.setattr(study, "ffd", lambda forecast, capacity: (0, 1))
    case = {"dataset": "azure2019", "origin": 288, "seed_index": 0, "seed": 186288,
            "views": ["primary", "longitudinal"]}
    plan = {"methods": ["CARA", "Eager-CARA"], "protocol": protocol, "config": {},
            "proposal_allowance_seconds": 2., "data_root": "unused"}
    result = study.run_case((0, case, plan, "mocked-study", tmp_path))
    assert target_accesses == [("CARA", "Eager-CARA")]
    assert result["panel_rows"] == {"build": 121, "selection": 25, "audit": 25}
    assert all(set(policy["metrics"]) == set(study.NATIVE_METRICS) for policy in result["policies"].values())
