"""Budgeted repair on every hourly-source VM with a shorter history protocol.

This additional study uses all nonoverlapping daily targets available after
96 observed hours. It does not replace CloudCons' long-history decision
protocol. Each origin's complete search/selection/audit decisions are frozen
for every method, budget and stream before its target is evaluated.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
from pathlib import Path
from time import process_time

import numpy as np

from benchmark_forecast_metrics import aggregate as forecast_aggregate, forecast_totals, metrics as forecast_metrics
from benchmark_hourly_queries import SOURCES, load_hourly
from benchmark_native_intervals import predictive_metrics
from cara.data import Dataset, native_metrics
from cara.repair import ffd
from run_family_study import decide, save, serial


ROOT = Path(__file__).resolve().parent
METHODS = ("Swap", "Joint-SA", "Family-only", "CARA")
UPGRADE_FLAGS = ("family_seed_compilation", "family_bound_views", "family_metric_cascade", "family_subset_views",
                 "family_structural_bound_gate", "family_compiled_bounds", "family_staged_acquisition")
METRICS = ("hosts", "Util", "VR", "VS", "PAR", "raw", "PICP", "MPIW", "WinklerScore")
PROTOCOL = {"historical_roles": {"build": {"older": 96, "newer": 48},
    "selection": {"older": 48, "newer": 24}, "audit": {"older": 24, "newer": 0}},
    "window_hours": 8, "window_stride": 1, "target_hours": 24,
    "expected_rows": {"build": 41, "selection": 17, "audit": 17}}
DATA = {}


def source_data(name, data_root, smoke=False):
    key = (name, str(data_root), bool(smoke))
    if key in DATA:
        return DATA[key], True
    started = process_time()
    if smoke:
        rng = np.random.default_rng(261110)
        phases = rng.uniform(0, 2 * np.pi, (64, 1))
        time = np.arange(120)[None, :]
        matrix = .03 + rng.uniform(.06, .19, (64, 1)) * (1 + np.sin(2 * np.pi * time / 24 + phases)) / 2
        matrix += rng.exponential(.01, matrix.shape)
        metadata = {"source": "synthetic-smoke", "series": 64, "common_hours": 120,
                    "interpolated_values_in_common_interval": 0}
        item_ids, missing = tuple(map(str, range(len(matrix)))), {}
    else:
        matrix, metadata = load_hourly(data_root, name)
        import pyarrow as pa
        import pyarrow.ipc as ipc
        fields = ["item_id"] + (["target"] if metadata["interpolated_values_in_common_interval"] else [])
        with pa.memory_map(str(Path(data_root) / name / "data-00000-of-00001.arrow"), "r") as handle:
            rows = ipc.open_stream(handle).read_all().select(fields).to_pylist()
        rows.sort(key=lambda row: str(row["item_id"]))
        item_ids = tuple(str(row["item_id"]) for row in rows)
        missing = {}
        if "target" in fields:
            for index, row in enumerate(rows):
                raw = np.asarray(row["target"], dtype=np.float64)
                raw = raw if raw.ndim == 1 else raw[0]
                raw = raw[:matrix.shape[1]]
                if np.isnan(raw).any():
                    missing[index] = raw.copy()
    matrix.setflags(write=False)
    value = (matrix, item_ids, missing, {**metadata, "source_loading_cpu_seconds": process_time() - started})
    DATA[key] = value
    return value, False


def historical_data(name, matrix, item_ids, missing, origin, capacity):
    """Expose no target cells, including through the Dataset slicing API."""
    values = []
    for index, row in enumerate(matrix):
        history = row[:origin]
        if index in missing:
            raw = missing[index][:origin]
            known = np.flatnonzero(~np.isnan(raw))
            if not len(known):
                raise ValueError("A complete VM has no observed pre-origin history")
            history = np.interp(np.arange(origin), known, raw[known])
        history.setflags(write=False)
        values.append(history)
    return Dataset(name, tuple(values), capacity, item_ids)


def execution_config(method, base, flags):
    config = {**base, **dict.fromkeys(UPGRADE_FLAGS, False), **dict.fromkeys(flags, method == "CARA")}
    # Recording complete proposal paths is unnecessary for this independent
    # outcome study and is disabled identically for every policy.
    config.update(record_query_paths=False, record_seed_assignments=False)
    return config


def run_origin(job):
    source, origin, index, plan, output = job
    path = Path(output) / "records" / f"{source}_{origin}.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("study") != plan or existing.get("source") != source or existing.get("origin") != origin:
            raise ValueError("A cached origin belongs to a different declared study")
        return existing
    (matrix, item_ids, missing, metadata), cache_hit = source_data(source, plan["data_root"], plan["smoke"])
    if origin < 96 or origin + 24 > matrix.shape[1]:
        raise ValueError("The full source does not cover the specified history and target")
    started = process_time()
    preliminary = historical_data(source, matrix, item_ids, missing, origin, 1.)
    forecast = preliminary.forecast(origin, horizon=24)
    forecast.setflags(write=False)
    capacity = float(1.05 * max(float(forecast[:, 0].max()), 8 * float(forecast[:, 0].mean())))
    history = Dataset(source, preliminary.series, capacity, item_ids)
    if history.min_length != origin:
        raise AssertionError("Search data must end strictly before the target")
    incumbent = ffd(forecast, capacity)
    stationary = {"count": 0, "fraction": 0., "cost": 0., "aligned_assignment": incumbent, "moved_vms": []}
    common_cpu = process_time() - started
    pending = []
    for budget_index, budget in enumerate(plan["budgets"]):
        for stream in plan["streams"]:
            seed = 296000 + 100000 * stream + 10000 * plan["sources"].index(source) + origin
            methods = list(plan["methods"])
            shift = (index + stream + budget_index) % len(methods)
            order = methods[shift:] + methods[:shift]
            record = {"source": source, "origin": origin, "stream": stream, "seed": seed,
                "budget_cpu_seconds": budget, "method_order": order, "policies": {}}
            for method in order:
                gc.collect()
                config = execution_config(method, plan["config"], plan["upgrade_flags"])
                engine_method = "CARA" if method == "Family-only" else method
                record["policies"][method] = decide(history, origin, incumbent, forecast,
                    engine_method, config, seed, PROTOCOL, budget)
                record["policies"][method]["upgrade_flags"] = {name: config[name] for name in UPGRADE_FLAGS}
            pending.append(record)
    # All budgets, streams and methods at this origin are frozen. No search
    # receives the complete source matrix or a Dataset containing target cells.
    evaluation_started = process_time()
    truth = np.ascontiguousarray(matrix[:, origin:origin + 24])
    scales = history.panel(origin - 24, 24).std(axis=1, ddof=1)
    past = history.panel(0, origin)
    totals = forecast_totals(past, truth, forecast, scales)
    metric_cache = {}
    for record in pending:
        record["policies"]["FFD"] = {"terminal_assignment": incumbent, "movement": stationary,
            "selected_assignment": None, "admission": {"admitted": False, "reason": "packing_reference"}}
        for policy in record["policies"].values():
            placement = tuple(policy["terminal_assignment"])
            if placement not in metric_cache:
                metric_cache[placement] = {**native_metrics(truth, placement, capacity),
                    **predictive_metrics(truth, placement, forecast, scales)}
            policy["metrics"] = metric_cache[placement]
            policy["normalized_intervals"] = {"nMPIW": policy["metrics"]["MPIW"] / capacity,
                "nWinklerScore": policy["metrics"]["WinklerScore"] / capacity}
    result = {"study": plan, "source": source, "origin": origin, "N": len(matrix),
        "H": len(set(incumbent)), "capacity": capacity, "metadata": metadata,
        "trace_cache_hit": cache_hit, "incumbent_assignment": incumbent,
        "current_trace_loading_cpu_seconds": 0. if cache_hit else metadata["source_loading_cpu_seconds"],
        "common_forecast_ffd_cpu_seconds": common_cpu, "search_data_end_exclusive": history.min_length,
        "target_opened_after_all_decisions": True, "target_hours": 24,
        "forecast": {"totals": totals, "metrics": forecast_metrics(totals)},
        "evaluation_cpu_seconds": process_time() - evaluation_started,
        "contexts": pending}
    save(path, result)
    return result


def mean(values):
    return float(np.mean(values)) if values else None


def policy_summary(rows, method):
    policies = [row["policies"][method] for row in rows]
    output = {name: mean([p["metrics"][name] for p in policies]) for name in METRICS}
    output.update({name: mean([p["normalized_intervals"][name] for p in policies]) for name in ("nMPIW", "nWinklerScore")})
    output.update(contexts=len(rows), admitted_fraction=mean([p["admission"]["admitted"] for p in policies]),
        selected_fraction=mean([p.get("selected_assignment") is not None for p in policies]),
        moved_fraction=mean([p["movement"]["fraction"] for p in policies]),
        moved_count=mean([p["movement"]["count"] for p in policies]))
    if method != "FFD":
        for name in ("total_decision_cpu_seconds", "total_decision_wall_seconds", "proposal_with_preparation_cpu_seconds",
                     "selection_cpu_seconds", "audit_cpu_seconds", "preparation_cpu_seconds"):
            output[name] = mean([p["timing"][name] for p in policies])
        overrun = [p["timing"]["proposal_budget_overshoot_cpu_seconds"] for p in policies]
        output.update(overrun_median=float(np.median(overrun)), overrun_p95=float(np.percentile(overrun, 95)),
            overrun_max=float(np.max(overrun)), complete_families=mean([p["search"].get("blocks", 0) for p in policies]),
            evaluated_candidates=mean([p["search"].get("evaluations", 0) for p in policies]))
    return output


def summarize(records, plan):
    contexts = [row for origin in records for row in origin["contexts"]]
    summaries = []
    for budget in plan["budgets"]:
        rows = [row for row in contexts if row["budget_cpu_seconds"] == budget]
        for method in ("FFD", *plan["methods"]):
            cohorts = {source: policy_summary([row for row in rows if row["source"] == source], method)
                       for source in plan["sources"] if any(row["source"] == source for row in rows)}
            numeric = [name for name in next(iter(cohorts.values())) if name not in ("contexts", "MPIW", "WinklerScore")]
            summaries.append({"budget_cpu_seconds": budget, "method": method, "cohorts": cohorts,
                "equal_source": {name: mean([value[name] for value in cohorts.values()]) for name in numeric},
                "raw_interval_units": "Native MPIW and Winkler scores are reported per source; nMPIW/nWinkler divide by its decision capacity"})
    forecasts = {source: forecast_aggregate([row["forecast"] for row in records if row["source"] == source])
                 for source in plan["sources"] if any(row["source"] == source for row in records)}
    return {"complete": len(records) == len(plan["cases"]), "origin_contexts": len(records),
        "budget_stream_contexts": len(contexts), "nonoverlapping_cohort_hours": 24 * len(records),
        "all_target_access_guards": all(row["target_opened_after_all_decisions"] for row in records),
        "all_source_vms": not plan["smoke"], "policies": summaries, "shared_forecasts": forecasts,
        "forecast_equal_source": {name: mean([value[name] for value in forecasts.values() if value[name] is not None]) for name in ("MASE", "CRPS")}}


def run(args):
    if (args.workers < 1 or not args.budgets or any(x <= 0 or not np.isfinite(x) for x in args.budgets)
            or len(set(args.budgets)) != len(args.budgets) or not args.streams or any(x < 0 for x in args.streams)
            or len(set(args.streams)) != len(args.streams)):
        raise ValueError("Positive budgets/workers and distinct nonnegative streams are required")
    config = json.loads(args.config.read_text())
    config = config.get("config", config)
    sources = ["synthetic-smoke"] if args.smoke else args.sources
    cases = []
    metadata = {}
    for source in sources:
        (_, _, _, meta), _ = source_data(source, args.data_root, args.smoke)
        metadata[source] = {key: value for key, value in meta.items()
                            if not key.endswith(("cpu_seconds", "wall_seconds"))}
        cases.extend({"source": source, "origin": origin} for origin in range(96, meta["common_hours"] - 24 + 1, 24))
    plan = {"name": "Complete-source short-history repair utility", "sources": sources,
        "data_root": str(args.data_root.resolve()), "config": config, "methods": args.methods,
        "budgets": args.budgets, "streams": args.streams, "upgrade_flags": args.upgrade_flags,
        "cases": cases, "source_metadata": metadata, "protocol": PROTOCOL, "smoke": args.smoke,
        "workers": args.workers,
        "capacity": "1.05 * max(maximum VM trailing-24h mean, 8 * average VM trailing-24h mean)",
        "history": "Every eight-hour window in disjoint outer Build/Selection/Audit intervals: 41/17/17 rows",
        "forecast": "Same trailing-24h mean and ddof=1 scale for every policy; MASE/CRPS from the benchmark diagnostic functions",
        "target_access": "History-only Dataset ends at origin; every budget/stream/method at the origin freezes before target slicing",
        "imputation": "Current source reader's evaluation interpolation; historical missing values are recomputed solely within the pre-origin prefix",
        "budget_scope": "Per-policy historical representation, setup and proposals use process CPU; complete last operations may overrun and are recorded. Selection/Audit and common trace/FFD preparation are separately recorded",
        "scope": "Additional all-hourly-source short-history study, not a replacement of CloudCons long-history evaluation or observed deployment replay"}
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != serial(plan):
        raise ValueError("Use a new output directory for a changed declared study")
    save(path, plan)
    if args.plan_only:
        print(json.dumps({"origins": len(cases), "contexts": len(cases) * len(args.budgets) * len(args.streams),
                          "policies": args.methods, "source_days": cases}), flush=True)
        return
    jobs = [(case["source"], case["origin"], index, plan, str(args.output)) for index, case in enumerate(cases)]
    records = []
    if args.workers == 1:
        iterator = map(run_origin, jobs)
        for record in iterator:
            records.append(record)
            save(args.output / "summary.json", summarize(records, plan))
            print(json.dumps({"source": record["source"], "origin": record["origin"], "complete": len(records), "total": len(cases)}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_origin, job) for job in jobs]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                save(args.output / "summary.json", summarize(records, plan))
                print(json.dumps({"source": record["source"], "origin": record["origin"], "complete": len(records), "total": len(cases)}), flush=True)
    records.sort(key=lambda row: (sources.index(row["source"]), row["origin"]))
    save(args.output / "summary.json", summarize(records, plan))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/release/family_frozen.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/cloudcons_hourly")
    parser.add_argument("--sources", choices=SOURCES, nargs="+", default=list(SOURCES))
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--budgets", type=float, nargs="+", default=[2., 10.])
    parser.add_argument("--streams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--upgrade-flags", choices=UPGRADE_FLAGS, nargs="+", default=list(UPGRADE_FLAGS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
