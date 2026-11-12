"""Select the acquisition batch limit on separate synthetic query workloads."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from benchmark_family_execution import encoded
from benchmark_family_queries import run_plan
from benchmark_hourly_queries import make_input
from run_family_study import ROOT, save


def synthetic_matrix(hosts, windows, seed):
    rng = np.random.default_rng(seed)
    residents, hours = 8 * hosts, windows + 7
    time = np.arange(hours)[None, :]
    phase = rng.integers(0, 4, residents)[:, None] * (np.pi / 2)
    baseline = rng.uniform(.08, .12, (residents, 1))
    amplitude = rng.uniform(.12, .30, (residents, 1))
    return baseline + amplitude * (1 + np.sin(2 * np.pi * time / 24 + phase)) / 2 + rng.exponential(.015, (residents, hours))


def run(args):
    if args.repetitions < 1:
        raise ValueError("At least one repetition is required")
    raw = json.loads(args.config.read_text())
    # Retain the family-local executor on which the batch limit was selected.
    # The compiled-seed plans inherit this limit rather than retune it on traces.
    config = {**raw.get("config", raw), "family_bootstrap_bounds": True,
              "family_seed_compilation": False, "family_bound_views": False,
              "family_metric_cascade": False, "family_subset_views": False,
              "family_structural_bound_gate": False, "family_compiled_bounds": False,
              "family_staged_acquisition": False}
    batches = (8, 16, 32, 64)
    contexts = [(hosts, windows, 241001 + index)
                for index, (hosts, windows) in enumerate((
                    (128, 32), (128, 96), (256, 32), (256, 96), (512, 32), (512, 96)))]
    records = []
    save(args.output / "protocol.json", {"config": config, "batch_limits": batches,
        "contexts": contexts, "repetitions": args.repetitions, "families": 5,
        "purpose": "Development-only query workloads, generated independently of every trace benchmark",
        "selection": "Minimize the sum of per-context median process CPU; break ties by smaller batch limit",
        "timing": "Serial rotated trials after untimed warmup; all complete-family outputs must agree"})
    for context_index, (hosts, windows, seed) in enumerate(contexts):
        matrix = synthetic_matrix(hosts, windows, seed)
        panel, forecast, incumbent, capacity, _ = make_input(matrix, "full", 8)
        expected = None
        for batch in batches:
            snapshots, _ = run_plan(panel, forecast, incumbent, capacity,
                {**config, "family_progressive_batch": batch}, "compiled", families=5)
            actual = encoded(snapshots)
            if expected is not None and actual != expected:
                raise AssertionError("Acquisition granularity changed the exact retained result")
            expected = actual
        trials = []
        for repetition in range(args.repetitions):
            shift = (context_index + repetition) % len(batches)
            order = batches[shift:] + batches[:shift]
            measured = {}
            for batch in order:
                gc.collect()
                snapshots, stats = run_plan(panel, forecast, incumbent, capacity,
                    {**config, "family_progressive_batch": batch}, "compiled", families=5)
                if encoded(snapshots) != expected:
                    raise AssertionError("Timed acquisition changed the exact retained result")
                measured[str(batch)] = {"cpu_seconds": stats["cpu_seconds"],
                    "profile_cpu_seconds": sum(row["profile_cpu_seconds"] for row in stats["families"]),
                    "profiled_masks": sum(row["counters"]["profiled_masks"] for row in stats["families"]),
                    "progressive_batches": sum(row["counters"]["progressive_batches"] for row in stats["families"])}
            trials.append({"order": order, "plans": measured})
        medians = {str(batch): {metric: float(np.median([
            trial["plans"][str(batch)][metric] for trial in trials]))
            for metric in ("cpu_seconds", "profile_cpu_seconds", "profiled_masks", "progressive_batches")}
            for batch in batches}
        record = {"hosts": hosts, "windows": windows, "seed": seed,
                  "all_family_pools_equal": True, "medians": medians, "trials": trials}
        records.append(record)
        save(args.output / "records.json", records)
        print(json.dumps({"hosts": hosts, "windows": windows, "medians": medians}), flush=True)
    totals = {str(batch): sum(row["medians"][str(batch)]["cpu_seconds"] for row in records) for batch in batches}
    chosen = min(batches, key=lambda batch: (totals[str(batch)], batch))
    summary = {"complete": True, "contexts": len(records), "all_family_pools_equal": True,
               "total_median_cpu_seconds": totals, "selected_batch_limit": chosen}
    save(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/release/family_frozen.json")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
