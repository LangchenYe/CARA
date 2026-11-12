"""Matched historical-audit plans on every current frozen CARA decision.

This is an admission-query comparison. Search and Selection are never rerun;
the incumbent reference is common, outside all four candidate-query timers.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
from pathlib import Path
from time import process_time

import numpy as np

from analyze_family_study import load_complete
import benchmark_dyadic_audit as base
from run_family_study import case_key


ROOT = Path(__file__).resolve().parent
PLANS, FIELDS = base.PLANS, base.FIELDS
REPETITIONS = 3
COMMON_FIELDS = ("role_source_prep_cpu_seconds", "audit_panel_materialization_cpu_seconds",
                 "reference_profile_cpu_seconds", "canonical_target_cpu_seconds",
                 "plan_oracle_cpu_seconds")


def validate_population(plan, records):
    """Require the complete current union without fixing its selected count."""
    protocol = plan["protocol"]
    if (plan["phase"] != "evaluation" or "CARA" not in plan["methods"]
            or len(records) != 354 or len(plan["cases"]) != 354
            or protocol["historical_roles"]["audit"] != {"older": 32, "newer": 0}
            or protocol["window_hours"] != 8 or protocol["window_stride"] != 1
            or protocol["row_caps"] is not None or protocol["expected_rows"]["audit"] != 25):
        raise ValueError("The complete current 354-case, 25-row CARA Audit population is required")
    indexed = {case_key(row): row for row in records}
    if len(indexed) != 354 or set(indexed) != {case_key(case) for case in plan["cases"]}:
        raise ValueError("Every current frozen case must occur exactly once")
    return [indexed[case_key(case)] for case in plan["cases"]]


def check_result(candidate, gate, target, expected, identity, plan):
    if (candidate.shape != target.shape or candidate.dtype != target.dtype
            or candidate.tobytes() != target.tobytes() or not base.same_gate(gate, expected)):
        raise AssertionError((identity, plan, "Current candidate bytes or exact gate fields changed"))


def run_case(record, datasets, index):
    policy = record["policies"]["CARA"]
    identity = {key: record[key] for key in ("dataset", "origin", "seed_index", "seed")}
    result = {"identity": identity, "views": record["views"], "original_admission": policy["admission"]}
    selected = policy["selected_assignment"]
    if selected is None:
        return {**result, "status": "no_action", "trials": {plan: [] for plan in PLANS},
                "medians": {plan: dict.fromkeys(FIELDS, 0.) for plan in PLANS},
                **dict.fromkeys(COMMON_FIELDS, 0.)}

    began = process_time()
    matrix, starts, upper = base.source_for(record, datasets)
    role_cpu = process_time() - began
    began = process_time()
    panel = base.materialize_windows(matrix, starts)
    panel_cpu = process_time() - began
    if panel.shape != (25, len(selected), 8):
        raise AssertionError("A current action does not have all 25 prescribed Audit windows")
    began = process_time()
    reference = base.panel_profile(panel, record["incumbent_assignment"], record["capacity"])
    reference_cpu = process_time() - began
    began = process_time()
    target = base.panel_profile(panel, selected, record["capacity"])
    target_cpu = process_time() - began
    expected = policy["admission"]

    # Every plan is checked before its independently timed rotations. The full
    # population Fraction-versus-dyadic gate oracle also precedes this loop.
    began = process_time()
    for plan in PLANS:
        candidate, gate, _ = base.run_plan(matrix, starts, selected, record["capacity"], upper, reference, plan)
        check_result(candidate, gate, target, expected, identity, plan)
    oracle_cpu = process_time() - began
    trials, orders = {plan: [] for plan in PLANS}, []
    for repetition in range(REPETITIONS):
        shift = (index + repetition) % len(PLANS)
        order = PLANS[shift:] + PLANS[:shift]
        orders.append(list(order))
        for plan in order:
            candidate, gate, cost = base.run_plan(matrix, starts, selected, record["capacity"], upper, reference, plan)
            check_result(candidate, gate, target, expected, identity, plan)
            trials[plan].append(cost)
    medians = {plan: {field: float(np.median([cost[field] for cost in costs])) for field in FIELDS}
               for plan, costs in trials.items()}
    return {**result, "status": "selected", "N": len(selected), "H": len(set(selected)),
            "B": 25, "L": 32, "T": 8, "role_source_prep_cpu_seconds": role_cpu,
            "audit_panel_materialization_cpu_seconds": panel_cpu, "reference_profile_cpu_seconds": reference_cpu,
            "canonical_target_cpu_seconds": target_cpu, "plan_oracle_cpu_seconds": oracle_cpu,
            "all_profile_bytes_and_current_gate_fields_equal": True,
            "untimed_plans_checked": len(PLANS), "orders": orders, "trials": trials, "medians": medians}


def summarize(records):
    selected = sum(row["status"] == "selected" for row in records)
    plans = {}
    for plan in PLANS:
        totals = {field: sum(row["medians"][plan][field] for row in records) for field in FIELDS}
        plans[plan] = {"sum_of_case_medians": totals,
                       "mean_all_decisions": {key: value / len(records) if records else 0. for key, value in totals.items()},
                       "mean_selected": {key: value / selected if selected else 0. for key, value in totals.items()}}
    return {"decisions": len(records), "selected": selected, "no_action": len(records) - selected,
            "all_profile_bytes_and_current_gate_fields_equal": all(
                row["status"] == "no_action" or row["all_profile_bytes_and_current_gate_fields_equal"] for row in records),
            "common_cpu_seconds": {field: sum(row[field] for row in records) for field in COMMON_FIELDS},
            "plans": plans}


def run(source, output, data_root):
    source, output, data_root = Path(source), Path(output), Path(data_root)
    if source.resolve() == output.resolve():
        raise ValueError("The frozen study and audit output must be different directories")
    source_plan, records = load_complete(source)
    records = validate_population(source_plan, records)
    design = {"study": "complete_family_historical_audit", "plans": list(PLANS),
              "repetitions": REPETITIONS, "inputs": list(source_plan["protocol"]["decision_cohorts"]),
              "cases": source_plan["cases"],
              "population": "All 354 current frozen CARA decisions, including every no-action fallback; no selected-count condition or search rerun.",
              "audit": {"older": 32, "newer": 0, "horizon": 8, "stride": 1, "rows": 25},
              "factorial": "Direct/Keyed candidate profiles crossed with Fraction/dyadic exact gates, using the same stored native six-coordinate profiles.",
              "timing": "Cold candidate planning, source validation, canonical profile and one gate are inside each process-CPU timer. Gate cost is nested, not added a second time. Dataset loading, native preparation, role extraction, shared incumbent reference, independent canonical target and correctness passes are separately recorded outside every arm.",
              "rotation": "Three cyclic rotations per selected case, offset by the frozen case index; no-action cases retain zero costs in all-decision aggregates.",
              "correctness": "A full-population Fraction/dyadic check precedes timing. Every plan's untimed and timed candidate bytes and complete gate dictionary, including float bits, exact strings and key order, must equal the current frozen decision.",
              "scope": "Historical admission query execution only. These costs are not attributed to end-to-end search speedup."}
    path = output / "study_plan.json"
    if path.exists() and json.loads(path.read_text()) != design:
        raise ValueError("The frozen audit study settings changed; use a new output directory")
    base.save(path, design)
    datasets, loading = {}, {}
    for name in source_plan["protocol"]["decision_cohorts"]:
        began = process_time()
        datasets[name] = base.load_dataset(name, data_root)
        loading[name] = {"cpu_seconds": process_time() - began, "n_vms": datasets[name].n_vms}
    base.save(output / "loading.json", loading)
    base.save(output / "common_preparation.json", base.prepare_window_queries())
    began = process_time()
    population = base.population_check(records, datasets)
    population["cpu_seconds"] = process_time() - began
    base.save(output / "population_check.json", population)
    result = []
    for index, record in enumerate(records):
        result.append(run_case(record, datasets, index))
        base.save(output / "records.json", result)
        print(json.dumps({"completed": len(result), "identity": result[-1]["identity"],
                          "status": result[-1]["status"]}), flush=True)
    final_plan, final_records = load_complete(source)
    if final_plan != source_plan or final_records != records:
        raise AssertionError("Decision inputs changed during the audit experiment")
    summary = summarize(result)
    summary.update(complete=len(result) == 354, sources_unchanged_after_execution=True,
                   trace_loading_cpu_seconds=sum(row["cpu_seconds"] for row in loading.values()),
                   population_oracle_cpu_seconds=population["cpu_seconds"])
    base.save(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/family_budget_heuristics"))
    parser.add_argument("--output", type=Path, default=Path("results/release/family_audit"))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/cloudcons")
    args = parser.parse_args()
    run(args.source, args.output, args.data_root)
