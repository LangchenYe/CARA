"""Population and denominator contracts for the complete fixed-action study."""
from collections import Counter
import copy
import json
from pathlib import Path

import pytest

import run_frozen_fresh as study


SOURCE = Path(__file__).resolve().parents[1] / "results/icde2027_shared_streaming"


def test_original_population_is_complete_and_new_runner_does_not_require_lost_arrays():
    plan, records = study.load_contracts(SOURCE)
    assert len(records) == 96
    assert Counter(row["law"] for row in records) == dict.fromkeys(plan["scene_laws"], 32)
    assert Counter(row["status"] for row in records) == {"selected": 90, "no_candidate": 6}


def test_duplicate_or_missing_contract_cannot_be_a_complete_run(tmp_path):
    plan = json.loads((SOURCE / "study_plan.json").read_text())
    rows = json.loads((SOURCE / "CARA/records.json").read_text())
    (tmp_path / "CARA").mkdir()
    (tmp_path / "study_plan.json").write_text(json.dumps(plan))
    rows[-1] = copy.deepcopy(rows[0])
    (tmp_path / "CARA/records.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="complete original"):
        study.load_contracts(tmp_path)


def test_summary_retains_zero_cost_fallbacks_and_distinct_selected_denominator():
    rows = []
    for law in study.SCENE_LAWS:
        for selected in (True, False):
            rules = {arm: {"admitted": selected, "rows": 32 if selected else 0,
                "locked": [selected] * 4, "crossing_rows": [32 if selected else None] * 4}
                for arm in study.ARMS}
            rows.append({"law": law, "status": "selected" if selected else "no_candidate",
                "rules": rules, "medians": {arm: {"cpu_seconds": 2. if selected else 0.} for arm in study.ARMS},
                "evaluation": {"means": {"incumbent": {"J": .5}, "candidate": {"J": .4}},
                               "paired_diagnostic": {"estimated_all_targets_hold": False}}})
    summary = study.summarize(rows)
    assert summary["all_96_contracts_complete"] is False
    all_rows = summary["groups"]["all"]
    assert (all_rows["environments"], all_rows["selected"], all_rows["fallbacks"]) == (6, 3, 3)
    for result in all_rows["arms"].values():
        assert result["admitted_fraction_all"] == .5
        assert result["admitted_fraction_selected"] == 1.
        assert result["mean_rows_all"] == 16.
        assert result["mean_rows_selected"] == 32.
        assert result["mean_cpu_seconds_all"] == 1.
        assert result["admitted_evaluation_mean_target_disagreements"] == 3
        assert "false_admissions" not in result


def test_graph_ablation_keeps_action_independent_conservation():
    old = [0] * 24 + [1] * 24
    aligned = old.copy()
    aligned[:4] = [1] * 4
    aligned[24:28] = [0] * 4
    record = {"incumbent_assignment": old, "movement": {"aligned_assignment": aligned, "count": 8}}
    count = study.support(record, "count")
    global_bound = study.support(record, "global")
    # Movement/H is four, but the shared H=2 conservation envelope is one half.
    assert count[3] == global_bound[3]
    assert .5 < count[3] < .5000000001
