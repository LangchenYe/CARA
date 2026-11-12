"""Exact data acquisition as the retained repair budget varies."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from benchmark_acquisition_parameters import synthetic_matrix
from benchmark_family_execution import encoded, profile_oracle, verify_witness
from benchmark_family_queries import run_plan
from benchmark_hourly_queries import make_input
from run_family_study import ROOT, save


def measurements(stats):
    families = stats["families"]
    return {"cpu_seconds": stats["cpu_seconds"], "wall_seconds": stats["wall_seconds"],
            "materialized_values": stats["materialized_values"],
            "profile_cpu_seconds": sum(row["profile_cpu_seconds"] for row in families),
            **{name: sum(row["counters"].get(name, 0) for row in families)
               for name in ("profiled_masks", "workload_materialized_values", "host_group_queries",
                            "score_excluded_masks", "progressive_batches")}}


def run(args):
    if args.repetitions < 1:
        raise ValueError("At least one serial repetition is required")
    raw = json.loads(args.config.read_text())
    config = raw.get("config", raw)
    budgets, plans = (4, 8, 16, 32, 64), ("compiled", "no_score")
    seed, hosts, windows, families = 251101, 512, 96, 5
    matrix = synthetic_matrix(hosts, windows, seed)
    panel, forecast, incumbent, capacity, _ = make_input(matrix, "full", 8)
    save(args.output / "protocol.json", {"config": config, "retained_budgets": budgets,
        "plans": plans, "seed": seed, "hosts": hosts, "windows": windows,
        "families": families, "mutable_width": 8, "repetitions": args.repetitions,
        "purpose": "Independent synthetic workload; vary the query output budget without selecting controller parameters",
        "timing": "Serial rotated trials after untimed eager/profile/exclusion checks; no correctness observers in measured trials",
        "comparison": "Both plans share the family compiler and migration predicates; no_score acquires every eligible exact profile"})
    records = []
    for budget_index, budget in enumerate(budgets):
        settings = {**config, "top_p": budget}
        oracle_cache, expected, verification = {}, None, {}
        for plan in ("eager", *plans):
            counters = {}

            def witness(value):
                verify_witness(value, incumbent, forecast, panel, capacity, counters, oracle_cache)

            with profile_oracle(counters, oracle_cache):
                snapshots, stats = run_plan(panel, forecast, incumbent, capacity, settings, plan,
                                            families=families, certificate_observer=witness)
            actual = encoded(snapshots)
            if expected is not None and actual != expected:
                raise AssertionError("The acquisition plan changed an exact retained family result")
            expected = actual
            for reason in ("forecast", "migration", "score"):
                excluded = sum(row["counters"].get(reason + "_excluded_masks", 0) for row in stats["families"])
                if counters.get(reason + "_leaves", 0) != excluded:
                    raise AssertionError("An excluded leaf was not checked by the canonical oracle")
            verification[plan] = counters
        oracle_cache.clear()
        trials = []
        for repetition in range(args.repetitions):
            shift = (budget_index + repetition) % len(plans)
            order = plans[shift:] + plans[:shift]
            measured = {}
            for plan in order:
                gc.collect()
                snapshots, stats = run_plan(panel, forecast, incumbent, capacity, settings, plan, families=families)
                if encoded(snapshots) != expected:
                    raise AssertionError("A timed acquisition changed the exact retained result")
                measured[plan] = measurements(stats)
            trials.append({"order": order, "plans": measured})
        medians = {plan: {metric: float(np.median([trial["plans"][plan][metric] for trial in trials]))
                          for metric in trials[0]["plans"][plan]} for plan in plans}
        record = {"top_p": budget, "all_family_pools_exact": True, "all_profiles_and_exclusions_checked": True,
                  "verification": verification, "medians": medians, "trials": trials}
        records.append(record)
        save(args.output / "records.json", records)
        print(json.dumps({"top_p": budget, "medians": medians}), flush=True)
    save(args.output / "summary.json", {"complete": True, "retained_budgets": budgets,
        "all_family_pools_exact": True, "all_profiles_and_exclusions_checked": True,
        "rows": [{"top_p": row["top_p"], "medians": row["medians"]} for row in records]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/release/family_frozen.json")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
