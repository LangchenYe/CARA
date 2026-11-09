"""Complete-arm provenance checks using tiny, self-contained study records."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

import analyze_family_study as analysis
from run_family_study import MAIN_METHODS, NATIVE_METRICS, case_grid, method_settings


def write_study(path, plan, records):
    path.mkdir(exist_ok=True)
    binding = sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    for row in records:
        row["study_binding"] = binding
    (path / "study_plan.json").write_text(json.dumps(plan))
    (path / "records.json").write_text(json.dumps(records))


@pytest.fixture
def studies(tmp_path):
    protocol = {"primary": {"streams": [0], "first_origin": 192, "last_origin": 193,
                             "stride": 1, "cases": 2},
                "decision_cohorts": {"toy": {"minimum_hours": 217, "longitudinal_origins": [192, 193]}},
                "evaluation_union": {"unique_cases": 2}, "longitudinal": {"cases": 2},
                "target_hours": 24, "expected_rows": {"build": 1, "selection": 1, "audit": 1}}
    config = {"cpu_limit": 2., "top_p": 2, "family_matching_backend": "compiled"}
    sources = {name: "a" * 64 for name in (*analysis.REPLACEMENT_SOURCE_FILES,
                                           "cara/data.py", "run_family_study.py")}

    def create(name, methods, replacement=False):
        settings = {**config, **(analysis.REPLACEMENT_ADDITIONS if replacement else {})}
        implementation = {**sources, **({name: "b" * 64 for name in analysis.REPLACEMENT_SOURCE_FILES}
                                        if replacement else {})}
        plan = {"study": "complete_window_family_execution", "phase": "evaluation", "protocol": protocol,
                "cases": case_grid(protocol, "evaluation"), "config": settings, "methods": list(methods),
                "method_settings": {method: method_settings(method, settings) for method in methods},
                "proposal_allowance_seconds": 2., "workers": 1 if list(methods) == ["Scenario-MIP"] else 6,
                "data_root": "unused", "inputs": {"toy": "d" * 64}, "implementation": implementation,
                "environment": {"numpy": "fixture", "threads": 1}, "rotation": "declared rotation",
                "timing": "declared CPU allowance"}
        rows = []
        for index, case in enumerate(plan["cases"]):
            policies = {}
            for method in ("FFD", *methods):
                policies[method] = {"marker": int(replacement and method != "FFD"),
                    "metrics": dict.fromkeys(NATIVE_METRICS, 0.),
                    "movement": {"count": 0, "fraction": 0., "cost": 0.},
                    "timing": dict.fromkeys(analysis.TIMINGS, 0.),
                    "search": dict.fromkeys(analysis.COUNTERS, 0), "admission": {"admitted": False}}
            shift = index % len(methods)
            rows.append({**case, "config": settings, "n_vms": 2, "capacity": 1.,
                         "incumbent_assignment": [0, 1], "panel_rows": protocol["expected_rows"],
                         "policies": policies, "method_order": list(methods[shift:]) + list(methods[:shift]),
                         "trace_cache_hit": False, "trace_loading_cpu_seconds": 0.,
                         "trace_loading_wall_seconds": 0., "common_forecast_incumbent_cpu_seconds": 0.,
                         "evaluation_cpu_seconds": 0.})
        path = tmp_path / name
        write_study(path, plan, rows)
        return path, plan, rows

    quality = create("quality", tuple(method for method in MAIN_METHODS if method != "Scenario-MIP"))
    solver = create("solver", ("Scenario-MIP",))
    replacement = create("replacement", analysis.REPLACEMENT_METHODS, True)
    return quality, solver, replacement


def test_complete_replacement_preserves_sources_and_unaffected_policies(studies):
    quality, solver, replacement = studies
    inputs = [quality[0], solver[0]]
    original_bytes = {(path, name): (path / name).read_bytes()
                      for path in (*inputs, replacement[0]) for name in ("study_plan.json", "records.json")}
    base_plan, base_rows = analysis.join_studies(inputs)
    prior = deepcopy((base_plan, base_rows))
    plan, rows = analysis.replace_complete_arms(base_plan, base_rows, replacement[0])
    assert (base_plan, base_rows) == prior
    assert plan["substudies"] == [quality[1], solver[1], replacement[1]]
    assert set(plan["replacement"]["implementation_changes"]) == analysis.REPLACEMENT_SOURCE_FILES
    assert plan["replacement"]["config_additions"] == analysis.REPLACEMENT_ADDITIONS
    for old, row in zip(base_rows, rows):
        assert row["study_components"][:2] == old["study_components"]
        assert row["study_components"][2]["binding"] == replacement[2][0]["study_binding"]
        for method, policy in row["policies"].items():
            replaced = method in analysis.REPLACEMENT_METHODS
            assert policy["marker"] == int(replaced)
            if not replaced:
                assert policy == old["policies"][method]
            assert row["policy_sources"][method] == (2 if replaced else 1 if method == "Scenario-MIP" else 0)
            assert plan["method_sources"][method]["study"] == row["policy_sources"][method]
    for (path, name), content in original_bytes.items():
        assert (path / name).read_bytes() == content
    with pytest.raises(ValueError):
        analysis.replace_complete_arms(plan, rows, replacement[0])


def test_default_join_still_rejects_implicit_overlapping_reruns(studies):
    quality, solver, replacement = studies
    plan, _ = analysis.join_studies([quality[0], solver[0]])
    assert set(plan["methods"]) == set(MAIN_METHODS) and "replacement" not in plan
    with pytest.raises(ValueError):
        analysis.join_studies([quality[0], solver[0], replacement[0]])


@pytest.mark.parametrize("violation", ("protocol", "cases", "inputs", "environment", "budget", "config", "effective_settings",
                                      "source", "source_set", "methods", "partial", "duplicate",
                                      "context", "incumbent", "binding", "rotation"))
def test_replacement_rejects_incomplete_or_incompatible_evidence(studies, violation):
    quality, solver, (path, original_plan, original_rows) = studies
    plan, records = deepcopy(original_plan), deepcopy(original_rows)
    if violation == "protocol":
        plan["protocol"]["description"] = "different protocol"
    elif violation == "cases":
        plan["cases"].reverse()
    elif violation == "inputs":
        plan["inputs"]["toy"] = "e" * 64
    elif violation == "environment":
        plan["environment"]["numpy"] = "different"
    elif violation == "budget":
        plan["proposal_allowance_seconds"] = 3.
    elif violation == "config":
        plan["config"]["top_p"] += 1
        for row in records:
            row["config"] = plan["config"]
    elif violation == "effective_settings":
        plan["method_settings"]["CARA"]["top_p"] += 1
    elif violation == "source":
        plan["implementation"]["cara/data.py"] = "f" * 64
    elif violation == "source_set":
        plan["implementation"].pop("run_family_study.py")
    elif violation == "methods":
        removed = plan["methods"].pop()
        for row in records:
            row["policies"].pop(removed)
    elif violation == "partial":
        records.pop()
    elif violation == "duplicate":
        records[1] = deepcopy(records[0])
    elif violation == "context":
        records[0]["capacity"] = 2.
    elif violation == "incumbent":
        records[0]["policies"]["FFD"]["metrics"]["raw"] = 1.
    elif violation == "rotation":
        records[0]["method_order"] = records[0]["method_order"][::-1]
    write_study(path, plan, records)
    if violation == "binding":
        records[0]["study_binding"] = "0" * 64
        (path / "records.json").write_text(json.dumps(records))
    base_plan, base_rows = analysis.join_studies([quality[0], solver[0]])
    with pytest.raises(ValueError):
        analysis.replace_complete_arms(base_plan, base_rows, path)
