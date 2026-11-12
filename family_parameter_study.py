"""Prepare the declared CARA development grid or summarize every completed arm.

Preparation writes twelve ordinary run_family_study commands. It does not
launch a search. Summarization requires all 24 prescribed cases in every arm.
"""
import argparse
from itertools import product
import json
from pathlib import Path
import shlex
import sys

import numpy as np

from analyze_family_study import load_complete, summarize_method, write_csv
from run_family_study import NATIVE_METRICS, ROOT, case_grid, case_key, save, serial


def variants(spec):
    result = []
    for width, engine, fraction in product(spec["mutable_limit"], spec["frontier_engine"], spec["frontier_fraction"]):
        identity = f"w{width:02d}_{engine}_f{int(100 * fraction):02d}"
        result.append((identity, {"mutable_limit": width, "frontier_seed": True,
                                 "frontier_engine": engine, "frontier_objective": spec["frontier_objective"],
                                 "frontier_fraction": fraction}))
    if len(result) != spec["configurations"]:
        raise ValueError("The declared factor grid is incomplete")
    return result


def prepare(args):
    spec = json.loads(args.spec.read_text())
    protocol = json.loads(args.protocol.read_text())
    raw = json.loads(args.base_config.read_text())
    base = raw.get("config", raw)
    if (protocol["development"]["origins"] != spec["origins"]
            or list(protocol["decision_cohorts"]) != spec["cohorts"]
            or protocol["expected_rows"] != spec["historical_rows"]
            or protocol["proposal_allowance_seconds"] != spec["proposal_allowance_seconds"]):
        raise ValueError("Grid and complete-window protocol disagree")
    cases = case_grid(protocol, "development", spec["streams"])
    if len(cases) != spec["cases_per_configuration"]:
        raise ValueError("Grid must cover all declared cohort/origin/stream cases")
    configurations = [{"id": name, "settings": {**base, **settings, "cpu_limit": spec["proposal_allowance_seconds"]}}
                      for name, settings in variants(spec)]
    manifest = {"specification": spec, "protocol": protocol, "cases": cases,
                "base_config": base, "configurations": configurations, "workers": args.workers,
                "execution": "Ordinary complete-window decision runner, one configuration at a time. Configuration files and the ranking rule are written before outcomes; the commands are not launched by preparation."}
    path = args.output / "grid_plan.json"
    if path.exists() and json.loads(path.read_text()) != serial(manifest):
        raise ValueError("Use a new directory for a changed parameter grid")
    save(path, manifest)
    commands = []
    for item in configurations:
        config_path = args.output / "configs" / (item["id"] + ".json")
        save(config_path, item["settings"])
        command = [sys.executable, str(ROOT / "run_family_study.py"),
                   "--phase", "development", "--protocol", str(args.protocol.resolve()),
                   "--config", str(config_path.resolve()), "--methods", spec["method"],
                   "--seeds", *map(str, spec["streams"]), "--workers", str(args.workers),
                   "--data-root", str(args.data_root.resolve()),
                   "--output", str((args.output / "runs" / item["id"]).resolve())]
        commands.append(shlex.join(command))
    (args.output / "run_all.sh").write_text("#!/bin/sh\nset -eu\n" + "\n".join(commands) + "\n")
    print(json.dumps({"prepared_configurations": len(configurations), "cases_per_configuration": len(cases),
                      "method": spec["method"], "commands": str(args.output / "run_all.sh"), "launched": False}))


def summarize(args):
    manifest = json.loads((args.output / "grid_plan.json").read_text())
    spec, protocol = manifest["specification"], manifest["protocol"]
    method, metrics = spec["method"], spec["ranking"]["metrics"]
    expected = {case_key(case) for case in manifest["cases"]}
    ranking, detailed, budgets, reference_binding = [], [], [], None
    for item in manifest["configurations"]:
        plan, records = load_complete(args.output / "runs" / item["id"])
        if (plan["phase"] != "development" or plan["methods"] != [method]
                or plan["config"] != item["settings"] or plan["protocol"] != protocol
                or {case_key(row) for row in records} != expected):
            raise ValueError("A completed parameter arm differs from the frozen grid")
        binding = {field: plan[field] for field in ("inputs",)}
        if reference_binding is not None and binding != reference_binding:
            raise ValueError("Input datasets changed within the grid")
        reference_binding = binding
        proposal_cpu = np.asarray([row["policies"][method]["timing"]["proposal_with_preparation_cpu_seconds"] for row in records])
        overshoot = np.asarray([row["policies"][method]["timing"]["proposal_budget_overshoot_cpu_seconds"] for row in records])
        decision_cpu_values = np.asarray([row["policies"][method]["timing"]["total_decision_cpu_seconds"] for row in records])
        operational = {"configuration": item["id"], "cases": len(records),
                       **{key: item["settings"][key] for key in ("mutable_limit", "frontier_engine", "frontier_fraction")}}
        for label, values in (("charged_proposal_cpu_seconds", proposal_cpu), ("overshoot_cpu_seconds", overshoot),
                              ("complete_decision_cpu_seconds", decision_cpu_values)):
            operational[label + "_mean"] = float(np.mean(values))
            operational.update({label + "_" + name: float(value)
                                for name, value in zip(("p50", "p95", "max"), np.quantile(values, (.5, .95, 1.)))})
        worst = records[int(np.argmax(proposal_cpu))]
        operational["maximum_proposal_case"] = {key: worst[key] for key in ("dataset", "origin", "seed_index")}
        budgets.append(operational)
        gains, movements, decision_cpu = [], [], []
        per_metric = {name: [] for name in metrics}
        for cohort in spec["cohorts"]:
            rows = [row for row in records if row["dataset"] == cohort]
            candidate, reference = summarize_method(rows, method), summarize_method(rows, "FFD")
            normalized = {name: (reference[name] - candidate[name]) / max(abs(reference[name]), spec["ranking"]["normalization_floor"])
                          for name in metrics}
            gains.extend(normalized.values())
            for name, value in normalized.items():
                per_metric[name].append(value)
            movements.append(candidate["moved_fraction"])
            decision_cpu.append(candidate["total_decision_cpu_seconds_mean"])
            detailed.append({"configuration": item["id"], "cohort": cohort, **candidate,
                             **{"FFD_" + name: reference[name] for name in NATIVE_METRICS},
                             **{"normalized_" + name + "_gain": value for name, value in normalized.items()}})
        ranking.append({"configuration": item["id"], "score": float(np.mean(gains)),
                        "moved_fraction": float(np.mean(movements)),
                        "complete_decision_cpu_seconds": float(np.mean(decision_cpu)),
                        **{name + "_normalized_gain": float(np.mean(values)) for name, values in per_metric.items()},
                        "settings": item["settings"], "all_case_summary": summarize_method(records, method)})
    ranking.sort(key=lambda row: (-row["score"], row["moved_fraction"], row["complete_decision_cpu_seconds"], row["configuration"]))
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    output = args.output / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    save(output / "summary.json", {"complete": True, "ranking_rule": spec["ranking"],
                                   "configurations": len(ranking), "cases_per_configuration": len(expected),
                                   "ranking": ranking, "cohort_details": detailed})
    write_csv(output / "ranking.csv", ranking)
    write_csv(output / "cohort_details.csv", detailed)
    save(output / "budget_summary.json", budgets)
    write_csv(output / "budget_summary.csv", budgets)
    print(json.dumps({"complete_configurations": len(ranking), "selected_configuration": ranking[0]["configuration"],
                      "selected_score": ranking[0]["score"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "summarize"))
    parser.add_argument("--spec", type=Path, default=ROOT / "configs/development/family_parameter_grid.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/release/family_protocol.json")
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs/release/family_frozen.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/cloudcons")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.workers < 1:
        raise ValueError("A positive worker count is required")
    (prepare if options.action == "prepare" else summarize)(options)
