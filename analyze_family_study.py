"""Join complete method studies and summarize both frozen evaluation views.

Default inputs must have the same cases, controller, data and implementation.
An explicit replacement accepts only a complete rerun of the four declared
physical-map arms, retaining every original source and binding. The serial
solver retains its distinct timing protocol. All comparisons include
incumbent fallbacks; paired differences are descriptive, without independence
or significance claims for overlapping targets or repeated search streams.
"""
import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from cara.data import canonical_assignment, load_dataset, native_metrics
from run_family_study import MAIN_METHODS, NATIVE_METRICS, case_grid, case_key, save


TIMINGS = ("preparation_cpu_seconds", "remaining_proposal_allowance_seconds", "proposal_cpu_seconds",
           "proposal_with_preparation_cpu_seconds", "proposal_budget_overshoot_cpu_seconds", "selection_cpu_seconds",
           "search_call_cpu_seconds", "post_search_movement_cpu_seconds", "audit_cpu_seconds",
           "total_decision_cpu_seconds", "total_decision_wall_seconds")
COUNTERS = ("evaluations", "blocks", "completed_operations")
COMPARATORS = ("Compact-CARA", "Eager-CARA", "Joint-SA")


def load_complete(directory):
    plan = json.loads((directory / "study_plan.json").read_text())
    records = json.loads((directory / "records.json").read_text())
    expected = case_grid(plan["protocol"], plan["phase"], sorted({c["seed_index"] for c in plan["cases"]}))
    if plan["study"] != "complete_window_family_execution" or plan["cases"] != expected:
        raise ValueError(f"Unexpected or incomplete scenario design: {directory}")
    planned = {case_key(case): case for case in expected}
    if len(records) != len(expected) or {case_key(row) for row in records} != set(planned):
        raise ValueError(f"Every planned scenario must appear exactly once: {directory}")
    binding = sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    for row in records:
        if (row.get("study_binding") != binding
                or any(row[field] != value for field, value in planned[case_key(row)].items())
                or set(row["policies"]) != {"FFD", *plan["methods"]}):
            raise ValueError("Record does not match its declared scenario and methods")
        if row["panel_rows"] != plan["protocol"]["expected_rows"]:
            raise ValueError("A record does not use all prescribed historical windows")
        for method, policy in row["policies"].items():
            numbers = [policy["metrics"][name] for name in NATIVE_METRICS]
            numbers += [policy["movement"][name] for name in ("count", "fraction", "cost")]
            if method != "FFD":
                timing = policy["timing"]
                numbers += [timing[name] for name in TIMINGS]
                numbers += [policy["search"][name] for name in COUNTERS]
                overshoot = max(0., timing["proposal_with_preparation_cpu_seconds"] - plan["proposal_allowance_seconds"])
                if not math.isclose(timing["proposal_budget_overshoot_cpu_seconds"], overshoot, rel_tol=0., abs_tol=1e-9):
                    raise ValueError("Inconsistent recorded proposal overshoot")
            if not all(math.isfinite(value) for value in numbers):
                raise ValueError("Nonfinite recorded metric, movement or timing")
    return plan, records


def join_studies(inputs):
    studies = [load_complete(path) for path in inputs]
    first = studies[0][0]
    methods, merged = [], {}
    for index, (plan, rows) in enumerate(studies):
        for field in ("phase", "protocol", "cases", "config", "inputs", "implementation", "environment", "proposal_allowance_seconds"):
            if plan[field] != first[field]:
                raise ValueError(f"Input studies disagree on {field}")
        if set(methods).intersection(plan["methods"]):
            raise ValueError("Every method must come from one complete study")
        if "Scenario-MIP" in plan["methods"] and (plan["methods"] != ["Scenario-MIP"] or plan["workers"] != 1):
            raise ValueError("Scenario-MIP must be a separate serial study")
        methods.extend(plan["methods"])
        for row in rows:
            identity = case_key(row)
            if identity not in merged:
                merged[identity] = {name: value for name, value in row.items() if name not in ("study_binding", "method_order")}
                merged[identity]["policies"] = dict(row["policies"])
                merged[identity]["study_components"] = []
            else:
                target = merged[identity]
                for field in ("seed", "views", "config", "n_vms", "capacity", "incumbent_assignment", "panel_rows"):
                    if row[field] != target[field]:
                        raise ValueError(f"Studies disagree on the common context: {field}")
                if row["policies"]["FFD"] != target["policies"]["FFD"]:
                    raise ValueError("Common incumbent metrics or movement differ")
                target["policies"].update({name: row["policies"][name] for name in plan["methods"]})
            merged[identity]["study_components"].append({
                "study": index, "binding": row["study_binding"], "method_order": row["method_order"],
                "workers": plan["workers"], "trace_cache_hit": row["trace_cache_hit"],
                "trace_loading_cpu_seconds": row["trace_loading_cpu_seconds"],
                "trace_loading_wall_seconds": row["trace_loading_wall_seconds"],
                "common_forecast_incumbent_cpu_seconds": row["common_forecast_incumbent_cpu_seconds"],
                "evaluation_cpu_seconds": row["evaluation_cpu_seconds"]})
    if first["phase"] == "evaluation" and set(methods) != set(MAIN_METHODS):
        raise ValueError("Final evaluation analysis requires all 13 prescribed methods")
    ordered = [merged[case_key(case)] for case in first["cases"]]
    plan = {"study": "complete_family_comparison", "phase": first["phase"], "cases": first["cases"],
            "methods": methods, "protocol": first["protocol"], "config": first["config"],
            "proposal_allowance_seconds": first["proposal_allowance_seconds"],
            "substudies": [item[0] for item in studies],
            "execution": "Heuristic method order rotates within cases. Scenario-MIP retains its separately serial wall-limited solve; it did not share the heuristic rotation."}
    return plan, ordered


REPLACEMENT_METHODS = ("CARA", "No-frontier", "Best-seed", "Raw-order")
REPLACEMENT_ADDITIONS = {"family_unique_physical": True, "family_unique_physical_min_hosts": 16}
REPLACEMENT_SOURCE_FILES = frozenset(("cara/family_certificates.py", "cara/refinement.py",
                                    "cara/compiled_matching.py"))


def replace_complete_arms(base_plan, base_records, directory):
    """Explicit complete-arm overlay; original studies and bindings are retained."""
    if base_plan.get("replacement") is not None or base_plan["phase"] != "evaluation":
        raise ValueError("One declared replacement is supported, on a complete evaluation only")
    replacement, records = load_complete(directory)
    if (len(replacement["methods"]) != len(REPLACEMENT_METHODS)
            or set(replacement["methods"]) != set(REPLACEMENT_METHODS)):
        raise ValueError("Replacement must contain all and only the four declared physical-map arms")
    original_studies = base_plan["substudies"]
    original = original_studies[0]
    for field in ("phase", "protocol", "cases", "inputs", "environment", "proposal_allowance_seconds",
                  "rotation", "timing"):
        if replacement[field] != original[field]:
            raise ValueError(f"Replacement disagrees on {field}")
    if (set(original["config"]).intersection(REPLACEMENT_ADDITIONS)
            or replacement["config"] != {**original["config"], **REPLACEMENT_ADDITIONS}):
        raise ValueError("Replacement may only add the two declared unique-physical settings")
    old_sources, new_sources = original["implementation"], replacement["implementation"]
    if set(old_sources) != set(new_sources):
        raise ValueError("Replacement changes the recorded implementation-file set")
    changed_sources = {name: {"original": old_sources[name], "replacement": new_sources[name]}
                       for name in old_sources if old_sources[name] != new_sources[name]}
    if not set(changed_sources).issubset(REPLACEMENT_SOURCE_FILES):
        raise ValueError("Replacement changes implementation outside the declared physical-map files")
    owners = {method: index for index, study in enumerate(original_studies) for method in study["methods"]}
    for method in REPLACEMENT_METHODS:
        if method not in owners:
            raise ValueError("A replacement arm is missing from the complete original comparison")
        expected = {**original_studies[owners[method]]["method_settings"][method], **REPLACEMENT_ADDITIONS}
        if replacement["method_settings"][method] != expected:
            raise ValueError(f"Replacement changes additional effective settings for {method}")
    index = len(original_studies)
    all_studies = [*original_studies, replacement]
    bindings = [sha256(json.dumps(study, sort_keys=True).encode()).hexdigest() for study in all_studies]
    policy_sources = {"FFD": 0, **owners, **dict.fromkeys(REPLACEMENT_METHODS, index)}
    by_case = {case_key(row): row for row in records}
    if len(base_records) != len(records) or {case_key(row) for row in base_records} != set(by_case):
        raise ValueError("Replacement and original comparison must have identical complete cases")
    result = []
    for position, original_row in enumerate(base_records):
        row = by_case[case_key(original_row)]
        for field in ("seed", "views", "n_vms", "capacity", "incumbent_assignment", "panel_rows"):
            if row[field] != original_row[field]:
                raise ValueError(f"Replacement changes the common case context: {field}")
        if original_row["config"] != original["config"] or row["config"] != replacement["config"]:
            raise ValueError("A record's configuration disagrees with its own source plan")
        if row["policies"]["FFD"] != original_row["policies"]["FFD"]:
            raise ValueError("Replacement changes the common incumbent result")
        shift = position % len(REPLACEMENT_METHODS)
        if row["method_order"] != replacement["methods"][shift:] + replacement["methods"][:shift]:
            raise ValueError("Replacement method order disagrees with its declared case rotation")
        component = {"study": index, "binding": row["study_binding"], "method_order": row["method_order"],
                     "workers": replacement["workers"], "role": "complete_arm_replacement"}
        for field in ("trace_cache_hit", "trace_loading_cpu_seconds", "trace_loading_wall_seconds",
                      "common_forecast_incumbent_cpu_seconds", "evaluation_cpu_seconds"):
            component[field] = row[field]
        result.append({**original_row,
                       "policies": {**original_row["policies"],
                                    **{method: row["policies"][method] for method in REPLACEMENT_METHODS}},
                       "study_components": [*original_row["study_components"], component],
                       "policy_sources": dict(policy_sources)})
    plan = {**base_plan, "substudies": all_studies,
            "method_sources": {method: {"study": owner, "binding": bindings[owner]}
                               for method, owner in policy_sources.items()},
            "method_settings": {method: all_studies[owner]["method_settings"][method]
                                for method, owner in policy_sources.items() if method != "FFD"},
            "replacement": {"methods": list(REPLACEMENT_METHODS), "source_directory": str(directory),
                            "source_study": index, "binding": bindings[index],
                            "config_additions": dict(REPLACEMENT_ADDITIONS),
                            "allowed_implementation_changes": sorted(REPLACEMENT_SOURCE_FILES),
                            "implementation_changes": changed_sources,
                            "selection": "Every declared arm uses its complete rerun; no case-level replacement."},
            "execution": base_plan["execution"] + " The four declared arms come from a separate complete rerun. "
                         "Other methods retain their original complete studies; each source's rotation, workers "
                         "and bindings remain recorded. No shared rotation across those runs is implied."}
    return plan, result


def numeric_leaves(value, prefix=""):
    result = {}
    for name, item in value.items():
        key = prefix + name
        if isinstance(item, dict):
            result.update(numeric_leaves(item, key + "."))
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
            result[key] = float(item)
    return result


def summarize_method(records, method):
    policies = [row["policies"][method] for row in records]
    summary = {"method": method, "cases": len(records),
               **{name: float(np.mean([p["metrics"][name] for p in policies])) for name in NATIVE_METRICS},
               "admitted": sum(bool(p["admission"]["admitted"]) for p in policies),
               "selected": sum(p.get("selected_assignment") is not None for p in policies),
               "moved_count": float(np.mean([p["movement"]["count"] for p in policies])),
               "moved_fraction": float(np.mean([p["movement"]["fraction"] for p in policies])),
               "moved_cost": float(np.mean([p["movement"]["cost"] for p in policies]))}
    summary["admitted_fraction"] = summary["admitted"] / len(policies)
    for metric in ("VR", "VS", "PAR", "raw"):
        differences = [p["metrics"][metric] - row["policies"]["FFD"]["metrics"][metric] for row, p in zip(records, policies)]
        summary[metric + "_above_FFD"] = sum(value > 1e-12 for value in differences)
    if method == "FFD":
        return summary
    for name in TIMINGS:
        values = [p["timing"][name] for p in policies]
        summary[name + "_mean"] = float(np.mean(values))
    overshoot = [p["timing"]["proposal_budget_overshoot_cpu_seconds"] for p in policies]
    summary.update({"proposal_budget_overshoot_cpu_seconds_" + name: float(value)
                    for name, value in zip(("median", "p95", "max"), np.quantile(overshoot, (.5, .95, 1.)))})
    charged_cpu = sum(p["timing"]["proposal_with_preparation_cpu_seconds"] for p in policies)
    for name in COUNTERS:
        total = sum(p["search"][name] for p in policies)
        summary[name + "_sum"] = int(total)
        summary[name + "_mean"] = total / len(policies)
        summary[name + "_per_proposal_cpu_second"] = total / charged_cpu if charged_cpu else None
    numeric = [numeric_leaves(p["search"]) for p in policies]
    summary["search_means"] = {name: float(np.mean([row[name] for row in numeric]))
                               for name in sorted(set.intersection(*(set(row) for row in numeric)))}
    if method == "Scenario-MIP":
        statuses = {}
        for policy in policies:
            status = str(policy["search"].get("solver_status"))
            statuses[status] = statuses.get(status, 0) + 1
        summary["solver_status_counts"] = statuses
    return summary


def paired(records, comparator):
    values = {}
    for row in records:
        left, right = row["policies"]["CARA"], row["policies"][comparator]
        for metric in NATIVE_METRICS:
            values.setdefault(metric, []).append(left["metrics"][metric] - right["metrics"][metric])
        for metric in ("count", "fraction", "cost"):
            values.setdefault("movement_" + metric, []).append(left["movement"][metric] - right["movement"][metric])
        values.setdefault("admitted", []).append(int(left["admission"]["admitted"]) - int(right["admission"]["admitted"]))
        for metric in TIMINGS:
            values.setdefault(metric, []).append(left["timing"][metric] - right["timing"][metric])
        for metric in COUNTERS:
            values.setdefault(metric, []).append(left["search"][metric] - right["search"][metric])
    summary = {"method": "CARA", "comparator": comparator, "cases": len(records),
               **{name + "_mean_delta": float(np.mean(items)) for name, items in values.items()}}
    for name in NATIVE_METRICS:
        summary[name + "_lower"] = sum(value < -1e-12 for value in values[name])
        summary[name + "_tied"] = sum(abs(value) <= 1e-12 for value in values[name])
        summary[name + "_higher"] = sum(value > 1e-12 for value in values[name])
    return summary


def write_csv(path, rows):
    fields = list(dict.fromkeys(name for row in rows for name in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in rows)


def export_case(record, data_root, output):
    policy = record["policies"]["CARA"]
    demand = load_dataset(record["dataset"], data_root).panel(record["origin"])
    incumbent = np.asarray(record["incumbent_assignment"], dtype=int)
    aligned = np.asarray(policy["movement"]["aligned_assignment"], dtype=int)
    terminal = policy["terminal_assignment"]
    if canonical_assignment(aligned) != canonical_assignment(terminal):
        raise ValueError("Recorded physical case alignment has a different partition")
    moved = np.flatnonzero(incumbent != aligned).tolist()
    if moved != policy["movement"]["moved_vms"] or len(moved) != policy["movement"]["count"]:
        raise ValueError("Recorded physical case movement is inconsistent")
    hosts = len(set(incumbent))
    memberships = {"FFD": [tuple(np.flatnonzero(incumbent == h).tolist()) for h in range(hosts)],
                   "CARA": [tuple(np.flatnonzero(aligned == h).tolist()) for h in range(hosts)]}
    loads = {name: np.stack([demand[list(group)].sum(axis=0) for group in groups]) for name, groups in memberships.items()}
    metrics = {"FFD": record["policies"]["FFD"]["metrics"], "CARA": policy["metrics"]}
    for name, placement in (("FFD", incumbent), ("CARA", terminal)):
        actual = native_metrics(demand, placement, record["capacity"])
        if any(value != metrics[name][key] for key, value in actual.items()):
            raise ValueError("Fixed case native metrics do not match its target data")
    case = {"dataset": record["dataset"], "origin": record["origin"], "seed_index": record["seed_index"],
            "N": len(incumbent), "hosts": hosts, "capacity": record["capacity"], "target_hours": demand.shape[1],
            "moved_count": len(moved), "moved_fraction": policy["movement"]["fraction"],
            "unchanged_memberships": len(set(memberships["FFD"]).intersection(memberships["CARA"])),
            "touched_physical_hosts": [h + 1 for h in range(hosts) if memberships["FFD"][h] != memberships["CARA"][h]],
            "peak_overloaded_hosts": {name: int((values > record["capacity"]).sum(axis=0).max()) for name, values in loads.items()},
            "overloaded_hosts_by_hour": {name: (values > record["capacity"]).sum(axis=0).tolist() for name, values in loads.items()},
            "metrics": metrics, "admission": policy["admission"], "aligned_assignment": aligned.tolist(),
            "selection_rule": "Explicit case argument; the analysis does not search outcomes for a replacement case."}
    save(output / "case_study.json", case)
    write_csv(output / "loads.csv", [{"host": h + 1, "hour": t, "incumbent": float(loads["FFD"][h, t]),
                                     "CARA": float(loads["CARA"][h, t]), "capacity": record["capacity"]}
                                    for h in range(hosts) for t in range(demand.shape[1])])


def run(args):
    plan, records = join_studies(args.inputs)
    if getattr(args, "replacement", None) is not None:
        plan, records = replace_complete_arms(plan, records, args.replacement)
    methods = ["FFD", *plan["methods"]]
    summaries, comparisons, outcomes = [], [], []
    views = sorted({view for row in records for view in row["views"]})
    for view in views:
        selected = [row for row in records if view in row["views"]]
        groups = [("all", "all", selected)]
        groups += [(cohort, "all", [row for row in selected if row["dataset"] == cohort]) for cohort in plan["protocol"]["decision_cohorts"]]
        groups += [("all", seed, [row for row in selected if row["seed_index"] == seed]) for seed in sorted({row["seed_index"] for row in selected})]
        groups += [(cohort, seed, [row for row in selected if row["dataset"] == cohort and row["seed_index"] == seed])
                   for cohort in plan["protocol"]["decision_cohorts"] for seed in sorted({row["seed_index"] for row in selected})]
        for cohort, stream, subset in groups:
            group = {"view": view, "cohort": cohort, "stream": stream}
            summaries.extend({**group, **summarize_method(subset, method)} for method in methods)
            if "CARA" in methods:
                comparisons.extend({**group, **paired(subset, method)} for method in COMPARATORS if method in methods)
        for row in selected:
            for method in methods:
                policy = row["policies"][method]
                outcomes.append({"view": view, "cohort": row["dataset"], "origin": row["origin"], "stream": row["seed_index"],
                                 "method": method, **policy["metrics"], "admitted": policy["admission"]["admitted"],
                                 "moved_count": policy["movement"]["count"], "moved_fraction": policy["movement"]["fraction"]})
    macro = []
    for view in views:
        for method in methods:
            rows = [row for row in summaries if row["view"] == view and row["method"] == method and row["cohort"] != "all" and row["stream"] == "all"]
            macro.append({"view": view, "method": method,
                          **{name: float(np.mean([row[name] for row in rows])) for name in (*NATIVE_METRICS, "admitted_fraction", "moved_count", "moved_fraction", "moved_cost")}})
    summary = {"complete": True, "unique_cases": len(records), "methods": plan["methods"],
               "view_cases": {view: sum(view in row["views"] for row in records) for view in views},
               "cohort_macro": macro, "group_summaries": summaries, "paired_differences": comparisons,
               "definitions": {"group_means": "Arithmetic case means including incumbent fallbacks. Cohort-macro results weight each cohort equally, including in the longer Borg daily sequences.",
                               "throughput": "Sum of recorded evaluations, blocks or operations divided by sum of actual proposal CPU including representation preparation. Solver nodes are not candidate evaluations.",
                               "paired": "CARA minus comparator on matching cohort/origin/stream cases. Lower/tied/higher counts use a 1e-12 display tolerance and do not encode desirability for interval width or coverage.",
                               "timing": "Recorded actual times, not nominal two-second values. Selection, physical-action validation and Audit contribute to total decision time; common trace loading and forecast/FFD preparation are recorded per original substudy.",
                               "overshoot": "Nonnegative charged proposal CPU above its allowance; median, p95 and maximum use NumPy linear quantiles.",
                               "scope": "Overlapping targets and search streams are not independent workload replicates. No inferential significance claim is computed. Separate solver CPU and wall observations retain their distinct budget meaning."}}
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / "study_plan.json", plan)
    save(args.output / "records.json", records)
    save(args.output / "summary.json", summary)
    write_csv(args.output / "method_summary.csv", summaries)
    write_csv(args.output / "cohort_macro.csv", macro)
    write_csv(args.output / "paired_differences.csv", comparisons)
    write_csv(args.output / "case_metrics.csv", outcomes)
    case = (args.case[0], int(args.case[1]), int(args.case[2]))
    selected_case = next((row for row in records if case_key(row) == case), None)
    if selected_case is not None and "CARA" in selected_case["policies"]:
        export_case(selected_case, args.data_root, args.output)
    print(json.dumps({"complete": True, "unique_cases": len(records), "view_cases": summary["view_cases"], "methods": plan["methods"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--replacement", type=Path, help="Complete rerun of the four declared unique-physical arms")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent / "data/raw/cloudcons")
    parser.add_argument("--case", nargs=3, default=["huawei2025", "297", "0"], metavar=("COHORT", "ORIGIN", "STREAM"))
    run(parser.parse_args())
