"""Identical complete-family queries under scalar and shared batch execution."""
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import sys
from time import process_time
import tracemalloc
import weakref

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from cara import data as data_module, refinement, repair, query_engine, workload_views
from cara.data import CAPACITIES, canonical_assignment, load_dataset
from cara.query_engine import ForecastViews, SharedQueryReplay
from cara.refinement import migration_profile, panel_profile, profile_from_loads
from cara.repair import ReplayCache, _mutable, _pair_order, ffd
from cara.workload_views import ProjectionReplay

PLANS = ("full", "load", "project", "shared_scalar", "shared_batch")


def _record_stream(path):
    """Read large aggregate arrays one scenario at a time."""
    path = Path(path)
    if path.is_dir():
        aggregate = path / "records.json"
        if aggregate.exists():
            yield from _record_stream(aggregate)
        else:
            directory = path / "records" if (path / "records").is_dir() else path
            for file in sorted(directory.glob("*.json")):
                record = json.loads(file.read_text())
                if isinstance(record, dict) and "dataset" in record:
                    yield record
        return
    decoder = json.JSONDecoder()
    with path.open() as stream:
        buffer = stream.read(1 << 20).lstrip()
        if not buffer.startswith("["):
            content = json.loads(buffer + stream.read())
            yield from content.get("records", [content])
            return
        buffer = buffer[1:]
        while True:
            buffer = buffer.lstrip(" \t\r\n,")
            if buffer.startswith("]"):
                return
            try:
                record, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                addition = stream.read(1 << 20)
                if not addition:
                    raise ValueError("Incomplete scenario result array")
                buffer += addition
                continue
            yield record
            buffer = buffer[end:]


def _path_records(args):
    from benchmark_revised_execution import chosen_records
    selected = []
    for record in _record_stream(args.records):
        if (record.get("seed_index", 0) != args.seed_index
                or args.policy not in record.get("policies", {})
                or (args.config_id is not None and record.get("config_id") != args.config_id)):
            continue
        # Other policies can be much larger than the selected search trace.
        compact = {k: record[k] for k in ("dataset", "origin", "seed", "seed_index", "config",
                   "config_id", "implementation", "incumbent_assignment", "panel_rows") if k in record}
        compact["policies"] = {args.policy: record["policies"][args.policy]}
        selected.append(compact)
    return chosen_records(selected, args.policy, args.per_cohort, args.config_id, args.seed_index)


def _run_path(case, plan):
    config = {**case["config"], "reuse_views": plan != "full", "projected_views": plan == "project",
              "shared_queries": plan.startswith("shared"), "batch_queries": plan == "shared_batch",
              "replay_arms": [row["arm"] for row in case["history"]],
              "replay_feedback_costs": [row.get("feedback_cpu", row.get("cpu", 0.)) for row in case["history"]]}
    if case["history"] and "cooling_progress" in case["history"][0]:
        config["replay_frontier_progress"] = case["history"][0]["cooling_progress"]
    return refinement.search_refined(case["incumbent"], case["forecast"], case["build"],
        case["selection"], case["capacity"], config, np.random.default_rng(case["seed"]))


@contextmanager
def _probe(expected=None, resource_probe=True):
    """Separate correctness/resource probe; never supplies speedup timing."""
    patches, objects, frames = [], [], []
    state = {"profiles": [], "candidates": [], "profile_position": 0, "candidate_position": 0,
             "cpu_seconds": {}, "peak_retained_array_bytes": 0 if resource_probe else None,
             "peak_membership_metadata_upper_bytes": 0 if resource_probe else None,
             "query_profile_calls": 0, "matching_calls": 0, "counters": {}}
    counter_names = ("hits", "changed_values", "full_values", "materialized_values", "group_queries",
                     "unique_groups", "forecast_values", "group_evaluations")

    def counters(cache, category, before=None):
        before = before or {}
        for name in counter_names:
            if hasattr(cache, name):
                key = category + "." + name
                state["counters"][key] = state["counters"].get(key, 0) + getattr(cache, name) - before.get(name, 0)

    def patch(module, name, value):
        patches.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    def invoke(category, function, *args, **kwargs):
        if not resource_probe:
            return function(*args, **kwargs)
        frame = [0.]
        frames.append(frame)
        start = process_time()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = process_time() - start
            frames.pop()
            state["cpu_seconds"][category] = state["cpu_seconds"].get(category, 0.) + elapsed - frame[0]
            if frames:
                frames[-1][0] += elapsed

    def emit(name, item):
        key = "profile_position" if name == "profiles" else "candidate_position"
        position = state[key]
        if expected is None:
            state[name].append(item)
        elif position >= len(expected[name]) or item != expected[name][position]:
            raise AssertionError(f"{name} differs at fixed-path event {position}")
        state[key] += 1

    def profile(value):
        value = np.ascontiguousarray(value)
        emit("profiles", (value.shape, value.dtype.str, value.tobytes()))
        state["query_profile_calls"] += 1

    def memory():
        if not resource_probe:
            return
        arrays = metadata = 0
        seen = set()
        for reference in objects:
            cache = reference()
            if cache is None:
                continue
            owner = getattr(cache, "_registry", cache)
            if id(owner) in seen:
                continue
            seen.add(id(owner))
            if hasattr(cache, "views"):
                mapping = cache.views
                if isinstance(cache, query_engine.SharedQueryReplay):
                    arrays += len(mapping) * cache.bytes_per_view
            else:
                mapping = cache.groups
                arrays += sum(array.nbytes for array in mapping.values())
                if hasattr(cache, "projections"):
                    arrays += sum(column.nbytes for columns in cache.projections.values() for column in columns)
            # Count referenced integers separately per key: this is a labelled
            # upper estimate, rather than pretending that shared objects are new.
            metadata += sys.getsizeof(mapping) + sum(sys.getsizeof(key) + len(key) * sys.getsizeof(0)
                                                     for key in mapping)
        state["peak_retained_array_bytes"] = max(state["peak_retained_array_bytes"], arrays)
        state["peak_membership_metadata_upper_bytes"] = max(state["peak_membership_metadata_upper_bytes"], metadata)

    def tracked(base, category, methods, projected=False):
        class Tracked(base):
            def __init__(self, *args, **kwargs):
                invoke(category, super().__init__, *args, **kwargs)
                if resource_probe:
                    objects.append(weakref.ref(self))
                    counters(self, category)
                    if category == "workload_query" and hasattr(self, "groups"):
                        key = category + ".eager_seed_demand_values"
                        state["counters"][key] = state["counters"].get(key, 0) + self.panel.size
                memory()
        for name in methods:
            original = getattr(base, name)
            def method(self, *args, _function=original, _name=name, **kwargs):
                before = {name: getattr(self, name) for name in counter_names if hasattr(self, name)} if resource_probe else None
                value = invoke(category, _function, self, *args, **kwargs)
                if resource_probe:
                    counters(self, category, before)
                if projected and _name == "profile":
                    profile(value)
                elif projected and _name == "profiles":
                    for row in value:
                        profile(row)
                memory()
                return value
            setattr(Tracked, name, method)
        return Tracked

    original_profile = refinement.profile_from_loads
    def profiled(*args, **kwargs):
        value = invoke("workload_query", original_profile, *args, **kwargs)
        profile(value)
        return value
    original_matching = refinement.migration_profile
    def matching(incumbent, candidate, *args, **kwargs):
        emit("candidates", canonical_assignment(candidate))
        state["matching_calls"] += 1
        return invoke("physical_matching", original_matching, incumbent, candidate, *args, **kwargs)
    patch(refinement, "profile_from_loads", profiled)
    patch(refinement, "migration_profile", matching)
    for module in (refinement, repair, data_module):
        original = module.host_loads
        def loads(*args, _original=original, **kwargs):
            return invoke("workload_query", _original, *args, **kwargs)
        patch(module, "host_loads", loads)
    for name, category in (("feasible", "forecast_predicate"), ("_ordered_records", "retention"),
                           ("_relative_gains", "retention"), ("_coverage_archive", "retention"),
                           ("_select", "selection")):
        original = getattr(refinement, name)
        def timed(*args, _function=original, _category=category, **kwargs):
            return invoke(_category, _function, *args, **kwargs)
        patch(refinement, name, timed)
    patch(refinement, "ReplayCache", tracked(refinement.ReplayCache, "workload_query", ("loads",)))
    patch(workload_views, "ProjectionReplay", tracked(workload_views.ProjectionReplay, "workload_query", ("loads", "profile"), True))
    patch(query_engine, "SharedQueryReplay", tracked(query_engine.SharedQueryReplay, "workload_query",
          ("loads", "loads_many", "profile", "profiles"), True))
    patch(query_engine, "ForecastViews", tracked(query_engine.ForecastViews, "forecast_predicate",
          ("feasible", "feasible_many", "groups_feasible", "groups_feasible_many")))
    try:
        yield state
        if expected is not None and any(state[key] != len(expected[name]) for name, key in
                (("profiles", "profile_position"), ("candidates", "candidate_position"))):
            raise AssertionError("Fixed-path probe produced fewer query or candidate events")
    finally:
        for module, name, original in reversed(patches):
            setattr(module, name, original)


def _compare_path(case, plans, repetitions, memory_probe=False, resource_probe=True):
    from benchmark_revised_execution import result_signature, serial, canonical_json, stable_history
    with _probe(resource_probe=resource_probe) as expected:
        reference = _run_path(case, "full")
    signature = result_signature(reference)
    if "source_selected" in case and serial(reference["selected"]) != serial(case["source_selected"]):
        raise AssertionError("Recorded selected action is not reproduced")
    if "source_candidates" in case and canonical_json(reference["candidates"]) != canonical_json(case["source_candidates"]):
        raise AssertionError("Recorded retained candidate keys are not reproduced")
    if stable_history(reference["stats"]["history"]) != stable_history(case["history"]):
        raise AssertionError("Recorded completed-family path is not reproduced")
    probes = {"full": {k: v for k, v in expected.items() if k not in ("profiles", "candidates")}}
    for plan in plans:
        if plan == "full":
            continue
        with _probe(expected, resource_probe=resource_probe) as observed:
            result = _run_path(case, plan)
        if result_signature(result) != signature:
            raise AssertionError(f"{plan} changes retained candidates, selection or completed-family history")
        probes[plan] = {k: v for k, v in observed.items() if k not in ("profiles", "candidates")}
    del expected
    gc.collect()
    timings = {plan: [] for plan in plans}
    for repetition in range(repetitions):
        offset = repetition % len(plans)
        for plan in plans[offset:] + plans[:offset]:
            start = process_time()
            result = _run_path(case, plan)
            elapsed = process_time() - start
            if result_signature(result) != signature:
                raise AssertionError("Uninstrumented timing run changed the fixed-path result")
            timings[plan].append(elapsed)
    median = {plan: float(np.median(values)) for plan, values in timings.items() if values}
    memory = {}
    if memory_probe:
        if tracemalloc.is_tracing():
            raise RuntimeError("Standalone memory probe requires no active allocation tracer")
        for plan in plans:
            gc.collect()
            tracemalloc.start()
            try:
                result = _run_path(case, plan)
                current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            if result_signature(result) != signature:
                raise AssertionError("Allocation tracing changed the fixed-path result")
            memory[plan] = {"current_traced_allocated_bytes": current, "peak_traced_allocated_bytes": peak}
    return {"cohort": case["dataset"], "origin": case["origin"], "seed_index": case.get("seed_index"),
            "N": len(case["incumbent"]), "H": len(set(case["incumbent"])), "B": len(case["build"]),
            "T": case["build"].shape[2], "all_canonical_profile_bytes_and_candidate_identities_equal": True,
            "retained_keys_selection_and_completed_path_equal": True,
            "profile_events": probes["full"]["profile_position"], "candidate_events": probes["full"]["candidate_position"],
            "resource_probe_enabled": resource_probe,
            "allocation_probe": memory,
            "timing_cpu_seconds": timings, "median_cpu_seconds": median,
            "instrumented_probes": probes, "speedup": {plan: median[plan] / median["shared_batch"]
                for plan in plans if plan != "shared_batch" and plan in median and "shared_batch" in median}}


def search_paths(args):
    from benchmark_revised_execution import case_from_record, save
    records = _path_records(args)
    plan = {"policy": args.policy, "seed_index": args.seed_index, "plans": args.plans,
            "repetitions": args.repetitions, "scenarios": [{"cohort": r["dataset"], "origin": r["origin"]} for r in records],
            "query_cache_views_override": args.query_cache_views,
            "query_cache_bytes_override": args.query_cache_bytes,
            "allocation_probe": args.memory_probe,
            "allocation_probe_scope": "Separate tracemalloc run; incremental traced Python/NumPy allocations during search, "
                                      "including batch-local temporaries. Input panels already exist. This is not process RSS.",
            "scope": "Serial fixed completed-operation replay including search, predicates, physical matching, retention, "
                     "view materialization and Selection. Workload loading and historical/fresh Admission are excluded.",
            "verification": "Separate instrumented byte comparison of every profile, candidate matching identity, retained key, "
                            "selected action and complete family path. Instrumented costs are diagnostic and are not speedup timings.",
            "timing": "One library thread, rotated plan order, uninstrumented process CPU. Zero repetitions means verification only."}
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / "study_plan.json", plan)
    if args.plan_only:
        print(json.dumps({"planned_scenarios": len(records)}), flush=True)
        return
    completed, cache = [], {}
    for record in records:
        case = case_from_record(record, args.policy, cache)
        if args.query_cache_views is not None:
            case["config"] = {**case["config"], "query_cache_views": args.query_cache_views}
        if args.query_cache_bytes is not None:
            case["config"] = {**case["config"], "query_cache_bytes": args.query_cache_bytes}
        result = _compare_path(case, args.plans, args.repetitions, args.memory_probe)
        completed.append(result)
        save(args.output / "records" / f'{case["dataset"]}_{case["origin"]}.json', result)
        save(args.output / "records.json", completed)
        if args.repetitions:
            costs = {name: float(np.mean([r["median_cpu_seconds"][name] for r in completed])) for name in args.plans}
            summary = {"scenarios": len(completed), "mean_cpu_seconds": costs,
                       "speedup": {name: costs[name] / costs["shared_batch"] for name in args.plans if name != "shared_batch"},
                       "all_exact_checks_passed": True}
        else:
            summary = {"scenarios": len(completed), "all_exact_checks_passed": True}
        save(args.output / "summary.json", summary)
        print(json.dumps({"cohort": result["cohort"], "origin": result["origin"],
                          "exact": True, "speedup": result["speedup"]}), flush=True)


def candidates(forecast, incumbent, pair, mutable, capacity, migration_limit, forecast_views=None):
    """Enumerate, filter and physically match one declared residual family."""
    labels = np.asarray(incumbent, dtype=int)
    mutable_set = set(mutable)
    background = [tuple(i for i in np.flatnonzero(labels == host) if i not in mutable_set) for host in pair]
    symmetric, seen, output = not any(background), {incumbent}, []
    pending = []
    for mask in range(1 << (len(mutable) - int(symmetric))):
        residents = [list(background[0]), list(background[1])]
        for j, vm in enumerate(mutable):
            side = 0 if symmetric and j == 0 else (mask >> (j - int(symmetric))) & 1
            residents[side].append(vm)
        groups = tuple(tuple(sorted(group)) for group in residents)
        if any(not group for group in groups):
            continue
        pending.append((mask, groups))
    passing = (forecast_views.groups_feasible_many([groups for _, groups in pending]) if forecast_views is not None
               else [all(np.all(forecast[list(group)].sum(axis=0) <= capacity + 1e-12)
                         for group in groups) for _, groups in pending])
    for (mask, groups), valid in zip(pending, passing):
        if not valid:
            continue
        proposal = labels.copy()
        for host, residents in zip(pair, groups):
            proposal[list(residents)] = host
        assignment = canonical_assignment(proposal)
        if assignment in seen:
            continue
        seen.add(assignment)
        movement = migration_profile(incumbent, assignment)
        if movement["count"] <= migration_limit:
            output.append((assignment, movement["count"]))
    return output


def execute(panel, forecast, incumbent, capacity, families, plan, *, width=8, max_views=2048):
    start = process_time()
    cache = (None if plan == "full" else ReplayCache(panel, incumbent, capacity) if plan == "load"
             else ProjectionReplay(panel, incumbent, capacity) if plan == "project"
             else SharedQueryReplay(panel, incumbent, capacity, max_views=max_views))
    forecast_views = ForecastViews(forecast, capacity, incumbent) if plan.startswith("shared") else None
    setup = process_time() - start
    filter_cpu = query_cpu = scoring_cpu = 0.
    outputs, counts, signatures = [], [], []
    reference = panel_profile(panel, incumbent, capacity).mean(axis=0)
    scale = np.maximum(np.abs(reference), 1e-8)
    for pair, mutable in families:
        tick = process_time()
        feasible = candidates(forecast, incumbent, pair, mutable, capacity, int(.2 * len(incumbent)), forecast_views)
        filter_cpu += process_time() - tick
        tick = process_time()
        if plan == "shared_batch":
            profiles = cache.profiles([assignment for assignment, _ in feasible])
        elif plan in ("project", "shared_scalar"):
            profiles = [cache.profile(assignment) for assignment, _ in feasible]
        elif plan == "load":
            profiles = [profile_from_loads(cache.loads(assignment), capacity) for assignment, _ in feasible]
        else:
            profiles = [panel_profile(panel, assignment, capacity) for assignment, _ in feasible]
        query_cpu += process_time() - tick
        tick = process_time()
        for (assignment, moved), profile in zip(feasible, profiles):
            gain = (reference - profile.mean(axis=0)) / scale
            key = tuple(sorted(int(np.floor(value * (1 << 40))) for value in gain)) + (-moved,)
            signatures.append((assignment, key))
            outputs.append(profile)
        scoring_cpu += process_time() - tick
        counts.append(len(feasible))
    total = process_time() - start
    result = {"total_cpu_seconds": total, "setup_cpu_seconds": setup,
              "filter_and_match_cpu_seconds": filter_cpu, "query_cpu_seconds": query_cpu,
              "scoring_cpu_seconds": scoring_cpu, "candidates_per_family": counts,
              "candidate_count": sum(counts)}
    if cache is not None:
        result.update({"membership_hits": cache.hits, "demand_values": cache.changed_values})
    if isinstance(cache, SharedQueryReplay):
        result.update({"cached_view_bytes": cache.cached_view_bytes,
                       "peak_view_bytes": cache.peak_view_bytes, "evictions": cache.evictions,
                       "unique_groups": cache.unique_groups, "group_queries": cache.group_queries})
        result.update({"forecast_values": forecast_views.forecast_values,
                       "forecast_group_evaluations": forecast_views.group_evaluations,
                       "forecast_hits": forecast_views.hits})
    return result, signatures, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", nargs="+", default=list(CAPACITIES), choices=list(CAPACITIES))
    parser.add_argument("--origins", nargs="+", type=int, default=[216, 228, 240])
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--max-views", type=int, default=2048)
    parser.add_argument("--query-cache-views", type=int, help="Override retained-view count in fixed-path cache-size controls")
    parser.add_argument("--query-cache-bytes", type=int, help="Override retained numeric bytes in fixed-path cache-size controls")
    parser.add_argument("--plans", nargs="+", choices=PLANS, default=list(PLANS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path, help="Recorded adaptive search paths; omit for fixed incumbent families")
    parser.add_argument("--policy", default="CARA")
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--config-id")
    parser.add_argument("--per-cohort", type=int, default=0, help="0 covers every recorded origin; otherwise evenly spaced")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--memory-probe", action="store_true", help="Separate Python/NumPy allocation probe, never speedup timing")
    args = parser.parse_args()
    if "full" not in args.plans:
        parser.error("The full replay correctness reference is required")
    if args.records:
        search_paths(args)
        return
    records = []
    for name in args.cohorts:
        data = load_dataset(name)
        for origin in args.origins:
            panel = data.window_panel(origin, 192, 64, max_rows=24)
            forecast = data.forecast(origin)
            incumbent = ffd(forecast, data.capacity)
            pairs = _pair_order(panel, incumbent, data.capacity, np.random.default_rng(519 + origin))[:args.pairs]
            families = [(pair, _mutable(panel, incumbent, pair, args.width, data.capacity)) for pair in pairs]
            trials = {plan: [] for plan in args.plans}
            for repetition in range(args.repetitions):
                results = {}
                order = args.plans[repetition % len(args.plans):] + args.plans[:repetition % len(args.plans)]
                for plan in order:
                    result, signatures, outputs = execute(panel, forecast, incumbent, data.capacity,
                        families, plan, width=args.width, max_views=args.max_views)
                    trials[plan].append(result)
                    results[plan] = signatures, outputs
                expected_signatures, expected_profiles = results["full"]
                for plan in args.plans:
                    actual_signatures, actual_profiles = results[plan]
                    if actual_signatures != expected_signatures or any(
                            a.tobytes() != e.tobytes() for a, e in zip(actual_profiles, expected_profiles)):
                        raise AssertionError(f"{name}/{origin}: {plan} changes candidates, profiles or keys")
            median = {plan: {key: float(np.median([row[key] for row in rows]))
                             for key in ("total_cpu_seconds", "setup_cpu_seconds", "filter_and_match_cpu_seconds",
                                         "query_cpu_seconds", "scoring_cpu_seconds")}
                      for plan, rows in trials.items()}
            record = {"cohort": name, "origin": origin, "N": data.n_vms, "H": len(set(incumbent)),
                      "B": panel.shape[0], "T": panel.shape[2], "families": len(families),
                      "bitwise_profiles_and_keys": True, "trials": trials, "median": median}
            if "shared_batch" in median:
                record["speedup"] = {plan: median[plan]["total_cpu_seconds"] / median["shared_batch"]["total_cpu_seconds"]
                                     for plan in args.plans if plan != "shared_batch"}
                record["query_speedup"] = {plan: median[plan]["query_cpu_seconds"] / median["shared_batch"]["query_cpu_seconds"]
                                           for plan in args.plans if plan != "shared_batch"}
            records.append(record)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"scope": "Fixed complete residual families on Build; no search or evaluation feedback. "
                "Total includes view construction, all family feasibility and physical-matching checks, canonical queries and score keys. "
                "Family choice and workload loading are common untimed inputs. Rotated one-thread process-CPU timing.",
                "plans": args.plans, "records": records}, indent=2) + "\n")
            print(json.dumps({"cohort": name, "origin": origin, "candidates": trials["full"][0]["candidate_count"],
                              "speedup": record.get("speedup"), "query_speedup": record.get("query_speedup")}), flush=True)


if __name__ == "__main__":
    main()
