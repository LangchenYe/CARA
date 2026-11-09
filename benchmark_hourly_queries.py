"""Shared-query execution on all series of CloudCons hourly forecasting data.

This is a query-processing workload in the sources' original CPU units. The
deterministic partition and derived threshold are not production placements or
the official consolidation benchmark and imply no deployment-quality result.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import gc
import json
from math import ceil
import multiprocessing as mp
from pathlib import Path
import resource
import sys
from time import process_time
import traceback
import tracemalloc

import numpy as np

from cara.data import canonical_assignment, host_loads, host_memberships
from cara.query_engine import ForecastViews, SharedQueryReplay
from cara.query_plan import compiled_profiles
from cara.refinement import panel_profile, profile_from_loads
from cara.repair import ReplayCache, feasible
from cara.workload_views import ProjectionReplay


SOURCES = ("azure2019", "borg2019_d_trace", "borg2019_e_trace", "huawei2025")
PLANS = ("full", "load", "project", "shared_scalar", "shared_batch")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_hourly(root, source):
    """Align every CPU series on its shared absolute interval; interpolate NaNs."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    started = process_time()
    path = Path(root) / source / "data-00000-of-00001.arrow"
    with pa.memory_map(str(path), "r") as stream:
        rows = ipc.open_stream(stream).read_all().select(["target", "start", "freq", "item_id"]).to_pylist()
    if not rows or len({row["start"] for row in rows}) != 1:
        raise ValueError("all source series must share one absolute start")
    if any(str(row["freq"]).lower() not in ("h", "1h") for row in rows):
        raise ValueError("hourly source data are required")
    rows.sort(key=lambda row: str(row["item_id"]))
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("series identities must be unique")
    values, all_nan = [], 0
    for row in rows:
        target = np.asarray(row["target"], dtype=np.float64)
        cpu = target if target.ndim == 1 else target[0]
        if cpu.ndim != 1 or not len(cpu) or np.isinf(cpu).any():
            raise ValueError("each CPU channel must be a finite-length real series")
        values.append(cpu)
        all_nan += int(np.isnan(cpu).sum())
    length = min(map(len, values))
    matrix = np.stack([value[:length] for value in values])
    interpolated = int(np.isnan(matrix).sum())
    for row in matrix:
        missing = np.isnan(row)
        if missing.any():
            if missing.all():
                raise ValueError("an all-missing CPU series cannot be interpolated")
            positions = np.arange(length)
            row[missing] = np.interp(positions[missing], positions[~missing], row[~missing])
    if np.any(matrix < 0) or not np.isfinite(matrix).all():
        raise ValueError("nonnegative finite CPU demand is required")
    metadata = {"source": source, "series": len(values), "common_hours": length,
        "absolute_start": rows[0]["start"].isoformat(), "frequency": "hourly",
        "cpu_channel": 0, "nan_values_in_all_cpu_series": all_nan,
        "interpolated_values_in_common_interval": interpolated,
        "interpolation": "Within each common-interval CPU series, deterministic linear interpolation; endpoint gaps use the nearest observed endpoint.",
        "series_order": "ascending item_id, every series included",
        "raw_units_preserved": True, "arrow_loading_cpu_seconds": process_time() - started}
    return matrix, metadata


def make_input(matrix, scenes, horizon):
    if matrix.shape[1] < max(24, horizon) or horizon < 1:
        raise ValueError("the common interval must cover the forecast and query horizon")
    available = matrix.shape[1] - horizon + 1
    count = available if scenes == "full" else min(int(scenes), available)
    if count < 1:
        raise ValueError("a positive scene count is required")
    starts = np.arange(available) if count == available else np.linspace(0, available - 1, count, dtype=int)
    windows = np.lib.stride_tricks.sliding_window_view(matrix, horizon, axis=1)
    panel = np.ascontiguousarray(windows[:, starts, :].transpose(1, 0, 2))
    hosts = ceil(len(matrix) / 8)
    incumbent = canonical_assignment(np.arange(len(matrix)) % hosts)
    forecast = np.repeat(matrix[:, :24].mean(axis=1, keepdims=True), horizon, axis=1)
    capacity = float(1.05 * host_loads(forecast, incumbent).max())
    if capacity <= 0 or not np.isfinite(capacity):
        raise ValueError("the deterministic workload requires a positive finite threshold")
    return panel, forecast, incumbent, capacity, starts


def family_pairs(hosts, count):
    if hosts < 2 or count < 1:
        raise ValueError("at least two hosts and one family are required")
    pairs = []
    for index in range(count):
        left = index * hosts // count
        right = (left + hosts // 2 + 1) % hosts
        if right == left:
            right = (left + 1) % hosts
        pair = tuple(sorted((left, right)))
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def family_queries(incumbent, pair, width=8):
    """Complete two-host residual family, retaining every fixed resident."""
    groups = host_memberships(incumbent)
    left, right = groups[pair[0]], groups[pair[1]]
    taken = min(width // 2, len(left), len(right))
    mutable = tuple(sorted(left[:taken] + right[:taken]))
    output, seen = [], {incumbent}
    labels = np.asarray(incumbent)
    for mask in range(1 << len(mutable)):
        assignment = labels.copy()
        for bit, vm in enumerate(mutable):
            assignment[vm] = pair[(mask >> bit) & 1]
        action = canonical_assignment(assignment)
        if action not in seen:
            seen.add(action)
            output.append(action)
    return output


def run_plan(matrix, scene_count, plan, *, horizon=8, families=8, query_batch=16,
             max_views=4096, max_bytes=64 << 20, batch_values=1 << 20,
             profile_cells=1 << 20, trace_allocations=False):
    started = process_time()
    panel, forecast, incumbent, capacity, starts = make_input(matrix, scene_count, horizon)
    input_cpu = process_time() - started
    hosts = len(set(incumbent))
    effective_batch = min(query_batch, max(1, profile_cells // (hosts * len(panel))))
    if trace_allocations:
        tracemalloc.start()
    tick = process_time()
    cache = (None if plan == "full" else ReplayCache(panel, incumbent, capacity) if plan == "load"
             else ProjectionReplay(panel, incumbent, capacity) if plan == "project"
             else SharedQueryReplay(panel, incumbent, capacity, max_views=max_views,
                                    max_bytes=max_bytes, batch_values=batch_values))
    forecast_cache = (ForecastViews(forecast, capacity, incumbent, max_views=max_views,
                                   batch_values=batch_values) if plan.startswith("shared") else None)
    setup_cpu = process_time() - tick
    preparation_cpu = forecast_cpu = query_cpu = 0.
    profiles, predicates, compiled_plans = [], [], []
    submitted = 0
    for pair in family_pairs(hosts, families):
        tick = process_time()
        queries = family_queries(incumbent, pair)
        preparation_cpu += process_time() - tick
        query_step = max(1, len(queries)) if plan == "shared_compiled" else effective_batch
        for start in range(0, len(queries), query_step):
            batch = queries[start:start + query_step]
            submitted += len(batch)
            tick = process_time()
            passing = (forecast_cache.feasible_many(batch, hosts) if plan in ("shared_batch", "shared_compiled") else
                       np.array([forecast_cache.feasible(a, hosts) for a in batch]) if forecast_cache is not None else
                       np.array([feasible(a, forecast, capacity, hosts) for a in batch]))
            forecast_cpu += process_time() - tick
            predicates.append(passing)
            eligible = [assignment for assignment, valid in zip(batch, passing) if valid]
            tick = process_time()
            if plan == "shared_compiled":
                plan_stats = {}
                result = compiled_profiles(cache, eligible, profile_cells, diagnostics=plan_stats)
                compiled_plans.append(plan_stats)
            elif plan == "shared_batch":
                result = cache.profiles(eligible)
            elif plan in ("project", "shared_scalar"):
                result = np.stack([cache.profile(a) for a in eligible]) if eligible else np.empty((0, len(panel), 6))
            elif plan == "load":
                result = np.stack([profile_from_loads(cache.loads(a), capacity) for a in eligible]) if eligible else np.empty((0, len(panel), 6))
            else:
                result = np.stack([panel_profile(panel, a, capacity) for a in eligible]) if eligible else np.empty((0, len(panel), 6))
            query_cpu += process_time() - tick
            profiles.append(result)
    output = np.concatenate(profiles)
    predicates = np.concatenate(predicates)
    elapsed = process_time() - started
    cached_views = (len(cache.views) if hasattr(cache, "views") else len(cache.groups) if cache is not None else 0)
    cached_bytes = (cache.cached_view_bytes if hasattr(cache, "cached_view_bytes") else
                    sum(value.nbytes for value in cache.groups.values()) +
                    (sum(column.nbytes for columns in cache.projections.values() for column in columns)
                     if hasattr(cache, "projections") else 0) if cache is not None else 0)
    materialized = (panel.size * len(output) if cache is None else
                    panel.size + cache.changed_values if plan in ("load", "project") else
                    cache.materialized_values)
    traced_peak = tracemalloc.get_traced_memory()[1] if trace_allocations else None
    if trace_allocations:
        tracemalloc.stop()
    stats = {"plan": plan, "N": panel.shape[1], "H": hosts, "B": len(panel), "T": horizon,
        "window_starts": starts.tolist(), "families": len(family_pairs(hosts, families)),
        "submitted_queries": submitted, "eligible_queries": len(output), "derived_capacity": capacity,
        "cpu_seconds": elapsed, "input_panel_cpu_seconds": input_cpu,
        "session_setup_cpu_seconds": setup_cpu, "query_preparation_cpu_seconds": preparation_cpu,
        "forecast_predicate_cpu_seconds": forecast_cpu, "workload_profile_cpu_seconds": query_cpu,
        "input_panel_bytes": panel.nbytes, "retained_views": cached_views,
        "effective_query_batch": effective_batch, "projected_cell_budget": profile_cells,
        "peak_traced_incremental_bytes": traced_peak,
        "traced_allocation_scope": "Incremental Python/NumPy allocations after the raw CPU matrix, input panel and forecast already exist; not process RSS. Enabled only for correctness/resource probes.",
        "retained_view_array_bytes": int(cached_bytes),
        "peak_retained_view_array_bytes": int(cache._registry.peak_view_bytes) if hasattr(cache, "_registry") else int(cached_bytes),
        "peak_retained_views": (int(cache._registry.peak_view_bytes // cache.bytes_per_view)
                                if hasattr(cache, "_registry") else cached_views),
        "forecast_retained_views": len(forecast_cache.views) if forecast_cache is not None else 0,
        "workload_materialized_values": int(materialized),
        "forecast_materialized_values": int(forecast_cache.forecast_values if forecast_cache is not None
                                             else forecast.size * submitted)}
    if compiled_plans:
        stats["compiled_plans"] = compiled_plans
        stats["maximum_compiled_accounted_array_bytes"] = max(
            value["maximum_accounted_array_bytes"] for value in compiled_plans)
        stats["effective_query_batch"] = max(
            value.get("effective_query_batch", 0) for value in compiled_plans)
    return output, predicates, stats


def _rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def _child(connection, matrix, scene_count, plan, options):
    try:
        gc.collect()
        before = _rss_bytes()
        profiles, predicates, stats = run_plan(matrix, scene_count, plan, **options)
        stats.update({"peak_process_rss_bytes": _rss_bytes(), "baseline_process_rss_bytes": before,
                      "rss_scope": "Isolated process peak, including Python runtime, raw CPU matrix, input panel, views, temporaries and outputs; library startup is excluded from CPU timing."})
        connection.send((profiles, predicates, stats, None))
    except Exception:
        connection.send((None, None, None, traceback.format_exc()))
    finally:
        connection.close()


def isolated_run(matrix, scenes, plan, options):
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(sender, matrix, scenes, plan, options))
    process.start()
    sender.close()
    try:
        profiles, predicates, stats, error = receiver.recv()
    finally:
        receiver.close()
        process.join()
    if error is not None:
        raise RuntimeError(error)
    if process.exitcode:
        raise RuntimeError(f"isolated execution exited with {process.exitcode}")
    return profiles, predicates, stats


def equal_outputs(reference, result):
    for left, right in zip(reference[:2], result[:2]):
        if left.shape != right.shape or left.dtype != right.dtype or left.tobytes() != right.tobytes():
            raise AssertionError("canonical profile bytes or forecast decisions differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent / "data/raw/cloudcons_hourly")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--plans", nargs="+", choices=(*PLANS, "shared_compiled"), default=list(PLANS))
    parser.add_argument("--reference-plan", choices=(*PLANS, "shared_compiled"), default="full")
    parser.add_argument("--scenes", nargs="+", default=["full"], help="full or positive counts, e.g. full 32 128")
    parser.add_argument("--families", type=int, default=8)
    parser.add_argument("--query-batch", type=int, default=16)
    parser.add_argument("--max-views", type=int, default=4096)
    parser.add_argument("--max-bytes", type=int, default=64 << 20)
    parser.add_argument("--batch-values", type=int, default=1 << 20)
    parser.add_argument("--profile-cells", type=int, default=1 << 20,
                        help="cap candidate*host*scene gather cells by chunking queries")
    parser.add_argument("--repetitions", type=int, default=0, help="0 checks correctness only; positive runs rotated serial timing")
    parser.add_argument("--allocation-probe", action="store_true",
                        help="Optional allocation tracing in separate correctness/resource probes; timing rotations stay untraced")
    parser.add_argument("--smoke", action="store_true", help="all selected series, 16 windows, one family; functional check only")
    args = parser.parse_args()
    if args.repetitions < 0 or min(args.families, args.query_batch, args.max_views, args.max_bytes, args.batch_values, args.profile_cells) < 1:
        raise ValueError("positive workload limits and nonnegative repetitions required")
    scenes = ["16"] if args.smoke else args.scenes
    options = dict(families=1 if args.smoke else args.families, query_batch=args.query_batch,
                   max_views=args.max_views, max_bytes=args.max_bytes, batch_values=args.batch_values,
                   profile_cells=args.profile_cells)
    selected_plans = tuple(args.plans)
    plan = {"sources": args.sources, "scenes": scenes, "plans": selected_plans,
        "reference_plan": args.reference_plan, "repetitions": args.repetitions,
        "options": options, "smoke": args.smoke, "allocation_probe": args.allocation_probe,
        "scope": "All hourly forecasting CPU series in source units; query-processing execution only, not the official consolidation-quality grid or a production capacity claim.",
        "partition": "ascending identity round-robin, H=ceil(N/8); threshold=1.05 times maximum host sum of first-24-hour means; complete width-eight two-host residual families preserve fixed residents and enumerate every mutable mask before common forecast filtering.",
        "input": "Common absolute interval, all eight-hour sliding windows by default. Arrow decoding is separately recorded; timed execution includes dense input-panel and forecast construction, session setup, candidate preparation, forecast predicates and complete profiles.",
        "timing": "Fresh isolated process per plan; one library thread; rotated serial plan order. Correctness probes are separate from timing rotations. Peak RSS includes runtime and all live temporaries."}
    plan["materialized_values_scope"] = "Demand operands gathered for host-load aggregation, not physical DRAM bytes; excludes panel validation, projections and final reductions."
    save(args.output / "study_plan.json", plan)
    records = []
    for source in args.sources:
        matrix, metadata = load_hourly(args.data_root, source)
        for count in scenes:
            probe_options = {**options, "trace_allocations": args.allocation_probe}
            reference = isolated_run(matrix, count, args.reference_plan, probe_options)
            probes = {args.reference_plan: reference[2]}
            for name in selected_plans:
                if name == args.reference_plan:
                    continue
                result = isolated_run(matrix, count, name, probe_options)
                equal_outputs(reference, result)
                probes[name] = result[2]
            timings = {name: [] for name in selected_plans}
            peaks = {name: [] for name in selected_plans}
            for rotation in range(args.repetitions):
                order = selected_plans[rotation % len(selected_plans):] + selected_plans[:rotation % len(selected_plans)]
                for name in order:
                    result = isolated_run(matrix, count, name, options)
                    equal_outputs(reference, result)
                    timings[name].append(result[2]["cpu_seconds"])
                    peaks[name].append(result[2]["peak_process_rss_bytes"])
            base = probes[args.reference_plan]
            record = {**metadata, **{key: base[key] for key in ("N", "H", "B", "T", "derived_capacity", "submitted_queries", "eligible_queries", "families")},
                "scene_setting": count, "all_profile_bytes_and_forecast_decisions_equal": True,
                "allocation_probe": args.allocation_probe,
                "correctness_probes": probes, "timing_cpu_seconds": timings,
                "median_cpu_seconds": {name: float(np.median(values)) for name, values in timings.items() if values},
                "timed_peak_process_rss_bytes": {name: max(values) for name, values in peaks.items() if values}}
            records.append(record)
            save(args.output / "records.json", records)
            print(json.dumps({"source": source, "N": base["N"], "B": base["B"], "exact": True,
                              "interpolated": metadata["interpolated_values_in_common_interval"],
                              "median_cpu_seconds": record["median_cpu_seconds"]}), flush=True)
        del matrix
        gc.collect()


if __name__ == "__main__":
    main()
