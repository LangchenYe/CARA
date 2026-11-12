"""Calibrate score-bound acquisition from independent resident structures.

The indexed measurement arm forces family_score_min_hosts=0 (always acquire
bounds). Candidate metadata policies are selected offline from those physical
measurements, rather than recursively applying an already selected cost gate.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from benchmark_family_execution import encoded, profile_oracle, verify_witness
from benchmark_family_queries import run_plan
from cara.data import canonical_assignment
from run_family_study import ROOT, save


def synthetic_input(hosts, pair_residents, scenes, seed, horizon=8):
    """Only the edited pair changes occupancy; other hosts retain eight VMs."""
    rng = np.random.default_rng(seed)
    pair = (0, hosts // 2 + 1)
    occupancy = np.full(hosts, 8, dtype=int)
    occupancy[list(pair)] = pair_residents // 2
    incumbent = canonical_assignment(np.repeat(np.arange(hosts), occupancy))
    scale = 1. / occupancy[np.asarray(incumbent)]
    time = np.arange(scenes * horizon).reshape(scenes, 1, horizon)
    phase = rng.uniform(0., 2 * np.pi, (1, len(incumbent), 1))
    amplitude = rng.uniform(.35, .75, (1, len(incumbent), 1))
    signal = .72 + amplitude * (1 + np.sin(2 * np.pi * time / 24 + phase)) / 2
    panel = (signal + rng.exponential(.06, signal.shape)) * scale[None, :, None]
    forecast = np.repeat((.75 * scale)[:, None], horizon, axis=1)
    return np.ascontiguousarray(panel), forecast, incumbent, 1., pair, occupancy


def run(args):
    if args.repetitions < 1:
        raise ValueError("At least one serial rotation is required")
    raw = json.loads(args.config.read_text())
    config = {**raw.get("config", raw), "family_seed_compilation": True,
              "family_bound_views": True, "family_metric_cascade": True,
              "family_subset_views": True, "family_score_min_hosts": 0,
              "family_score_min_leaves": 32, "family_compiled_bounds": False,
              "family_staged_acquisition": False}
    contexts = [(hosts, residents, scenes, 726041 + index)
                for index, (hosts, residents, scenes) in enumerate(
                    (h, r, b) for h in (8, 32, 64, 128)
                    for r in (16, 64, 128) for b in (8, 64))]
    plans = ("indexed", "no_score")
    thresholds = (128, 160, 192, 224, 256, 384, 512, 768)
    save(args.output / "protocol.json", {"contexts": contexts, "config": config,
        "horizon": 8, "mutable_width": 8, "families": 1,
        "repetitions": args.repetitions, "plans": plans,
        "thresholds": thresholds, "seed_scope": "Independent synthetic demands only; no trace data or targets read",
        "comparison": "Both plans share seed matching; indexed acquires score certificates and no_score evaluates every eligible profile",
        "measurement_gate": "family_score_min_hosts=0 forces always-acquire in the indexed measurement arm; metadata policy selection is offline",
        "policy_evaluation": "Sum per-context median CPU of the selected physical plan. Selection uses only H,T,r_pair and candidate availability; no extra workload data are consulted.",
        "timing": "Serial rotated trials after independent profile, exclusion and complete-result checks"})
    records = []
    for context_index, (hosts, residents, scenes, seed) in enumerate(contexts):
        panel, forecast, incumbent, capacity, pair, occupancy = synthetic_input(hosts, residents, scenes, seed)
        oracle_cache, expected, probes = {}, None, {}
        for plan in plans:
            counters = {}

            def witness(value):
                verify_witness(value, incumbent, forecast, panel, capacity, counters, oracle_cache)

            with profile_oracle(counters, oracle_cache):
                snapshots, statistics = run_plan(panel, forecast, incumbent, capacity, config,
                    "recomputed_bounds" if plan == "indexed" else plan,
                    families=1, width=8, certificate_observer=witness)
            actual = encoded(snapshots)
            if expected is not None and actual != expected:
                raise AssertionError("Score acquisition changed the exact retained result")
            expected = actual
            for reason in ("forecast", "migration", "score"):
                excluded = sum(row["counters"].get(reason + "_excluded_masks", 0)
                               for row in statistics["families"])
                if counters.get(reason + "_leaves", 0) != excluded:
                    raise AssertionError("An excluded active leaf was not checked")
            probes[plan] = counters
        oracle_cache.clear()
        trials = []
        for repetition in range(args.repetitions):
            shift = (context_index + repetition) % len(plans)
            order = plans[shift:] + plans[:shift]
            measured = {}
            for plan in order:
                gc.collect()
                snapshots, statistics = run_plan(panel, forecast, incumbent, capacity, config,
                                                  "recomputed_bounds" if plan == "indexed" else plan,
                                                  families=1, width=8)
                if encoded(snapshots) != expected:
                    raise AssertionError("A measured plan changed the exact result")
                family = statistics["families"][0]
                measured[plan] = {"cpu_seconds": statistics["cpu_seconds"],
                    "profiled_masks": family["counters"]["profiled_masks"],
                    "forecast_passed_masks": family["counters"]["forecast_passed_masks"],
                    "profile_cpu_seconds": family["profile_cpu_seconds"],
                    "predicate_cpu_seconds": family["predicate_cpu_seconds"],
                    "materialized_values": statistics["materialized_values"]}
            trials.append({"order": order, "plans": measured})
        medians = {plan: {key: float(np.median([trial["plans"][plan][key] for trial in trials]))
                          for key in trials[0]["plans"][plan]} for plan in plans}
        record = {"hosts": hosts, "pair_residents": residents, "scenes": scenes,
            "horizon": 8, "width": 8, "seed": seed, "workloads": len(incumbent),
            "pair": pair, "occupancy": occupancy.tolist(), "structural_work": hosts + 8 * residents,
            "all_exact": True, "verification": probes, "medians": medians, "trials": trials}
        records.append(record)
        save(args.output / "records.json", records)
        print(json.dumps({"H": hosts, "R": residents, "B": scenes, "medians": medians}), flush=True)
    def total(choice):
        return sum(row["medians"][choice(row)]["cpu_seconds"] for row in records)
    totals = {"always": total(lambda row: "indexed"), "never": total(lambda row: "no_score"),
              "host128": total(lambda row: "indexed" if row["hosts"] >= 128 else "no_score")}
    structural = {str(threshold): total(lambda row: "indexed" if row["structural_work"] >= threshold
                                       else "no_score") for threshold in thresholds}
    chosen = min(thresholds, key=lambda threshold: (structural[str(threshold)], -threshold))
    save(args.output / "summary.json", {"complete": True, "contexts": len(records), "all_exact": True,
        "policy_cpu_seconds": totals, "structural_policy_cpu_seconds": structural,
        "selected_structural_threshold": chosen})
    print(json.dumps({"totals": totals, "structural": structural, "selected": chosen}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/release/family_frozen.json")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
