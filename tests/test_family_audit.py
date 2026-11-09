"""Current historical-audit coverage and exact-output contracts, without traces."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

import benchmark_family_audit as study
from run_family_study import case_grid


def test_current_complete_population_has_no_old_selected_count_requirement():
    protocol = json.loads((Path(__file__).resolve().parents[1] / "configs/release/family_protocol.json").read_text())
    cases = case_grid(protocol, "evaluation")
    plan = {"phase": "evaluation", "methods": ["CARA"], "protocol": protocol, "cases": cases}
    rows = [{**case, "policies": {"CARA": {"selected_assignment": None}}} for case in cases]
    assert study.validate_population(plan, list(reversed(rows))) == rows
    with pytest.raises(ValueError, match="354-case"):
        study.validate_population(plan, rows[:-1])
    altered = deepcopy(plan)
    altered["protocol"]["window_stride"] = 2
    with pytest.raises(ValueError, match="25-row"):
        study.validate_population(altered, rows)
    with pytest.raises(ValueError, match="exactly once"):
        study.validate_population(plan, [*rows[:-1], rows[0]])


def test_no_action_has_zero_cost_and_does_not_query(monkeypatch):
    def forbidden(*args):
        raise AssertionError("A no-action decision started an Audit query")
    monkeypatch.setattr(study.base, "source_for", forbidden)
    record = {"dataset": "toy", "origin": 288, "seed_index": 0, "seed": 1, "views": ["primary"],
              "policies": {"CARA": {"selected_assignment": None,
                  "admission": {"admitted": False, "rows_read": 0, "reason": "no_selected_action"}}}}
    row = study.run_case(record, {}, 0)
    assert row["original_admission"] == record["policies"]["CARA"]["admission"]
    assert all(all(value == 0. for value in costs.values()) for costs in row["medians"].values())
    selected = deepcopy(row)
    selected["status"] = "selected"
    selected["all_profile_bytes_and_current_gate_fields_equal"] = True
    for costs in selected["medians"].values():
        costs["total_cpu_seconds"] = 2.
    summary = study.summarize([row, selected])
    assert (summary["decisions"], summary["selected"], summary["no_action"]) == (2, 1, 1)
    for costs in summary["plans"].values():
        assert costs["mean_all_decisions"]["total_cpu_seconds"] == 1.
        assert costs["mean_selected"]["total_cpu_seconds"] == 2.


def test_profile_or_gate_bit_change_is_rejected():
    target = np.zeros((25, 6), dtype=np.float64)
    gate = {"admitted": False, "gains": [0.] * 6, "reason": "same"}
    study.check_result(target.copy(), deepcopy(gate), target, gate, "toy", "direct_original")
    changed = target.copy()
    changed[0, 0] = -0.
    with pytest.raises(AssertionError, match="candidate bytes"):
        study.check_result(changed, gate, target, gate, "toy", "direct_dyadic")
    changed_gate = deepcopy(gate)
    changed_gate["gains"][0] = -0.
    with pytest.raises(AssertionError, match="gate fields"):
        study.check_result(target, changed_gate, target, gate, "toy", "keyed_dyadic")


def test_every_plan_is_checked_then_receives_three_rotated_trials(monkeypatch):
    target = np.zeros((25, 6), dtype=np.float64)
    gate = {"admitted": False, "gains": [0.] * 6, "reason": "same"}
    record = {"dataset": "toy", "origin": 288, "seed_index": 0, "seed": 1, "views": ["primary"],
              "capacity": 1., "incumbent_assignment": (0, 0, 1, 1),
              "policies": {"CARA": {"selected_assignment": (0, 1, 0, 1), "admission": gate}}}
    monkeypatch.setattr(study.base, "source_for", lambda *_: (np.zeros((4, 32)), np.arange(25), 1.))
    monkeypatch.setattr(study.base, "materialize_windows", lambda *_: np.zeros((25, 4, 8)))
    monkeypatch.setattr(study.base, "panel_profile", lambda *_: target.copy())
    observed = []

    def fake_plan(*args):
        observed.append(args[-1])
        return target.copy(), deepcopy(gate), dict.fromkeys(study.FIELDS, 1.)

    monkeypatch.setattr(study.base, "run_plan", fake_plan)
    result = study.run_case(record, {}, 1)
    assert observed[:4] == list(study.PLANS)
    assert observed[4:] == [plan for order in result["orders"] for plan in order]
    assert result["orders"][0] == list(study.PLANS[1:] + study.PLANS[:1])
    assert all(len(trials) == 3 for trials in result["trials"].values())
    assert result["all_profile_bytes_and_current_gate_fields_equal"]
