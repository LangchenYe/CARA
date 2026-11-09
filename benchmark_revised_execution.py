"""Controlled execution experiments for revised CARA, without search retuning.

Examples (run from the artifact's code directory):
  python benchmark_revised_execution.py smoke --output results/execution_smoke
  python benchmark_revised_execution.py replay --records FROZEN/records.json \
      --policy CARA --per-cohort 3 --output results/execution --plan-only
  python benchmark_revised_execution.py replay --records FROZEN/records.json \
      --policy CARA --per-cohort 0 --output results/execution
  python benchmark_revised_execution.py kernel --output results/kernel_scale

Replay uses the recorded seed, config, and broad/exact arm sequence.  Timed
runs have no tracing; separate instrumented runs compare every emitted profile
byte and candidate identity across FULL, LOAD, PROJECT and AUTO. Kernel scaling measures one repeated aggregation
operator only, not large-instance placement quality.  No admission or future
outcome is used by this experiment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
from time import process_time, perf_counter

# Apply before NumPy/SciPy imports. Experiments run serially and use one BLAS
# thread. These settings and the installed numerical libraries are recorded.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from cara import refinement, workload_views
from cara.data import (CAPACITIES, canonical_assignment, host_loads,
                       load_dataset, trace_panels)
from cara.repair import ReplayCache, ffd
from cara.workload_views import ProjectionReplay

PLANS = ("full", "load_reuse", "project_reuse", "auto")
COMPARISONS = (("full", "load_reuse"), ("full", "project_reuse"),
               ("load_reuse", "project_reuse"), ("full", "auto"),
               ("load_reuse", "auto"), ("project_reuse", "auto"))


def environment():
    files = [Path(__file__), Path(refinement.__file__),
             Path(refinement.__file__).with_name("repair.py"),
             Path(refinement.__file__).with_name("data.py"),
             Path(refinement.__file__).with_name("workload_views.py")]
    return {"python": sys.version, "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {name: version(name) for name in ("numpy", "scipy", "pyarrow")},
            "threads": {name: os.environ[name] for name in
                       ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "source_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in files}}


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def canonical_json(value):
    return json.dumps(serial(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_reuse_switch(plans=PLANS):
    if "reuse_views" not in inspect.getsource(refinement.search_refined):
        raise RuntimeError("search_refined does not yet implement reuse_views=False; "
                           "refusing a misleading cached-versus-cached comparison")
    if "projected_views" not in inspect.getsource(refinement):
        raise RuntimeError("search_refined does not yet implement projected_views=True; "
                           "refusing a misleading three-plan comparison")
    if "auto" in plans and '"auto"' not in inspect.getsource(refinement._replay) and "'auto'" not in inspect.getsource(refinement._replay):
        raise RuntimeError("The frozen search does not implement AUTO routing")


def stable_history(history):
    return [{k: serial(v) for k, v in row.items()
             if "cpu" not in k.lower() and "seconds" not in k.lower()}
            for row in history]


def result_signature(result):
    stats = result["stats"]
    stable_counts = {k: stats.get(k) for k in ("evaluations", "blocks", "broad_trials",
                    "migration_rejections", "forecast_rejections", "selection_candidates",
                    "selection_guard_passed", "migration_limit")}
    return canonical_json({"selected": result.get("selected"), "bets": result.get("bets"),
           "candidates": result.get("candidates"), "movement": result.get("movement"),
           "history": stable_history(stats["history"]), "counts": stable_counts,
           "selection_reason": result.get("selection_reason")})


@contextmanager
def trace_execution(expected=None):
    """Instrumented correctness/decomposition probe; never used for speedup.

    Compare raw bytes, not tolerances or hashes. The reported digest is only a
    compact provenance record. Timing categories may nest and are not additive.
    """
    originals = {name: getattr(refinement, name) for name in
                 ("profile_from_loads", "migration_profile", "ReplayCache", "host_loads")}
    if hasattr(refinement, "ProjectionReplay"):
        originals["ProjectionReplay"] = refinement.ProjectionReplay
    original_projection = workload_views.ProjectionReplay
    state = {"events": [], "position": 0, "profile_calls": 0,
             "candidate_identity_calls": 0, "profile_cpu_seconds": 0.,
             "cache_setup_cpu_seconds": 0., "cache_load_cpu_seconds": 0.,
             "cache_profile_cpu_seconds": 0.,
             "direct_host_load_cpu_seconds": 0., "cache_instances": 0,
             "maximum_single_cache_bytes": 0, "digest": hashlib.sha256()}

    def event(kind, array):
        array = np.ascontiguousarray(array)
        item = (kind, array.shape, array.dtype.str, array.tobytes())
        position = state["position"]
        if expected is None:
            state["events"].append(item)
        elif position >= len(expected) or item != expected[position]:
            raise AssertionError(f"Bitwise execution mismatch at event {position}: {kind}")
        state["digest"].update(kind.encode() + repr((array.shape, array.dtype.str)).encode())
        state["digest"].update(item[3])
        state["position"] += 1

    def profiled(loads, *args, **kwargs):
        tick = process_time()
        value = originals["profile_from_loads"](loads, *args, **kwargs)
        state["profile_cpu_seconds"] += process_time() - tick
        state["profile_calls"] += 1
        event("profile", value)
        return value

    def moved(incumbent, candidate, *args, **kwargs):
        event("candidate", np.asarray(canonical_assignment(candidate), dtype=np.int64))
        state["candidate_identity_calls"] += 1
        return originals["migration_profile"](incumbent, candidate, *args, **kwargs)

    def direct_loads(*args, **kwargs):
        tick = process_time()
        value = originals["host_loads"](*args, **kwargs)
        state["direct_host_load_cpu_seconds"] += process_time() - tick
        return value

    class TracedCache(originals["ReplayCache"]):
        def __init__(self, *args, **kwargs):
            tick = process_time()
            super().__init__(*args, **kwargs)
            state["cache_setup_cpu_seconds"] += process_time() - tick
            state["cache_instances"] += 1
            stored = sum(array.nbytes for array in self.groups.values())
            state["maximum_single_cache_bytes"] = max(state["maximum_single_cache_bytes"], stored)

        def loads(self, *args, **kwargs):
            tick = process_time()
            value = super().loads(*args, **kwargs)
            state["cache_load_cpu_seconds"] += process_time() - tick
            return value

    class TracedProjection(originals.get("ProjectionReplay", ProjectionReplay)):
        def __init__(self, *args, **kwargs):
            tick = process_time()
            super().__init__(*args, **kwargs)
            state["cache_setup_cpu_seconds"] += process_time() - tick
            state["cache_instances"] += 1
            stored = sum(array.nbytes for array in self.groups.values())
            stored += sum(array.nbytes for columns in self.projections.values() for array in columns)
            state["maximum_single_cache_bytes"] = max(state["maximum_single_cache_bytes"], stored)

        def loads(self, *args, **kwargs):
            tick = process_time()
            value = super().loads(*args, **kwargs)
            state["cache_load_cpu_seconds"] += process_time() - tick
            return value

        def profile(self, *args, **kwargs):
            tick = process_time()
            value = super().profile(*args, **kwargs)
            state["cache_profile_cpu_seconds"] += process_time() - tick
            state["profile_calls"] += 1
            event("profile", value)
            return value

    refinement.profile_from_loads = profiled
    refinement.migration_profile = moved
    refinement.ReplayCache = TracedCache
    refinement.host_loads = direct_loads
    if "ProjectionReplay" in originals:
        refinement.ProjectionReplay = TracedProjection
    # _replay may import this class inside the factory, so instrument the
    # defining module as well as a possible module-level imported alias.
    workload_views.ProjectionReplay = TracedProjection
    try:
        yield state
        if expected is not None and state["position"] != len(expected):
            raise AssertionError("Execution produced fewer profile/candidate events than full replay")
    finally:
        for name, value in originals.items():
            setattr(refinement, name, value)
        workload_views.ProjectionReplay = original_projection


def run_search(case, plan):
    config = {**case["config"], "reuse_views": plan != "full",
              "projected_views": "auto" if plan == "auto" else plan == "project_reuse",
              "replay_arms": [row["arm"] for row in case["history"]]}
    return refinement.search_refined(case["incumbent"], case["forecast"], case["build"],
                case["selection"], case["capacity"], config, np.random.default_rng(case["seed"]))


def probe(case, plan, expected=None):
    with trace_execution(expected) as trace:
        result = run_search(case, plan)
    summary = {k: v for k, v in trace.items() if k not in ("events", "digest")}
    summary["event_sha256"] = trace["digest"].hexdigest()
    summary["planning_cpu_seconds"] = result["stats"].get("planning_cpu_seconds")
    summary["search_counters"] = {k: result["stats"].get(k) for k in
                                  ("cache_hits", "changed_values", "full_values", "evaluations", "blocks")}
    summary["timing_note"] = ("Separate instrumented probe. Cache setup, cache loads, direct loads "
                              "and profiles can be nested in planner time; do not sum categories. "
                              "Cache-profile time includes changed-group aggregation and projection reduction.")
    return result, trace["events"], summary


def compare_case(case, repetitions=4, plans=PLANS):
    require_reuse_switch(plans)
    full, events, full_probe = probe(case, "full")
    probes = {"full": full_probe}
    signature = result_signature(full)
    for plan in plans[1:]:
        reused, _, probes[plan] = probe(case, plan, events)
        if result_signature(reused) != signature:
            raise AssertionError(f"{plan}: selected action, retained keys, bets, history or candidate records differ")
    if "source_selected" in case and serial(full["selected"]) != serial(case["source_selected"]):
        raise AssertionError("Recorded winner was not reproduced: check seed/config/record version")
    if "source_candidates" in case and canonical_json(full["candidates"]) != canonical_json(case["source_candidates"]):
        raise AssertionError("Recorded retained candidate family/keys were not reproduced")
    if stable_history(full["stats"]["history"]) != stable_history(case["history"]):
        raise AssertionError("Recorded residual families/diagnostics were not reproduced")
    del events
    timings = {plan: [] for plan in plans}
    # Each plan occupies each timing position once per len(plans) repetitions.
    for repetition in range(repetitions):
        offset = repetition % len(plans)
        for plan in plans[offset:] + plans[:offset]:
            wall, cpu = perf_counter(), process_time()
            result = run_search(case, plan)
            process_elapsed, wall_elapsed = process_time() - cpu, perf_counter() - wall
            if result_signature(result) != signature:
                raise AssertionError("Fixed-arm timing run changed the deterministic result")
            stats = result["stats"]
            proposal = float(stats["cpu_seconds"])
            selection = float(stats["selection_cpu_seconds"])
            timings[plan].append({
                "proposal_cpu_seconds": proposal, "selection_cpu_seconds": selection,
                "proposal_selection_cpu_seconds": proposal + selection,
                "outer_process_cpu_seconds": process_elapsed, "wall_seconds": wall_elapsed,
                "planning_cpu_seconds": stats.get("planning_cpu_seconds")})
    medians = {plan: {k: float(np.median([r[k] for r in rows])) for k in rows[0]
                      if rows[0][k] is not None} for plan, rows in timings.items()}
    return {"dataset": case["dataset"], "origin": case["origin"], "seed": case["seed"],
            "seed_index": case.get("seed_index"), "source_implementation": case.get("source_implementation"),
            "config": case["config"], "plans": plans, "shape": {"N": len(case["incumbent"]),
            "H": len(set(case["incumbent"])), "B": len(case["build"]), "T": case["build"].shape[2]},
            "history_arms": [r["arm"] for r in case["history"]],
            "exact_identity": {"selected_retained_keys_bets_history": True,
                               "all_profile_and_candidate_bytes": True,
                               "events": full_probe["position"],
                               "profile_calls": full_probe["profile_calls"],
                               "event_sha256": full_probe["event_sha256"]},
            "selected": full["selected"], "retained_candidates": len(full["candidates"]),
            "timings": timings, "medians": medians,
            "speedups": {label: medians[left]["proposal_selection_cpu_seconds"] /
                                   medians[right]["proposal_selection_cpu_seconds"]
                         for label, left, right in (
                             ("full_over_load", "full", "load_reuse"),
                             ("full_over_project", "full", "project_reuse"),
                             ("load_over_project", "load_reuse", "project_reuse"))
                         if left in medians and right in medians},
            "instrumented_probes": probes}


def load_records(path):
    path = Path(path)
    if path.is_dir():
        paths = [path / "records.json"] if (path / "records.json").exists() else sorted(
            (path / "records" if (path / "records").is_dir() else path).glob("*.json"))
    else:
        paths = [path]
    records, sources = [], []
    for file in paths:
        raw = file.read_bytes()
        data = json.loads(raw)
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        entries = data if isinstance(data, list) else [data]
        matched = [r for r in entries if isinstance(r, dict) and "dataset" in r and "origin" in r]
        if matched:
            records.extend(matched)
            sources.append({"path": str(file.resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
    if not records:
        raise ValueError("No scenario records with dataset and origin were found")
    return records, sources


def chosen_records(records, policy, per_cohort, config_id=None, seed_index=0):
    eligible = [r for r in records if policy in r.get("policies", {})
                and (config_id is None or r.get("config_id") == config_id)
                and r.get("seed_index", 0) == seed_index]
    chosen = []
    for name in sorted({r["dataset"] for r in eligible}):
        subset = sorted((r for r in eligible if r["dataset"] == name), key=lambda r: r["origin"])
        if len({r["origin"] for r in subset}) != len(subset):
            raise ValueError("Multiple configs/repeats per scenario; select one frozen --config-id")
        indices = np.arange(len(subset)) if per_cohort == 0 or per_cohort >= len(subset) else np.unique(
            np.linspace(0, len(subset) - 1, per_cohort, dtype=int))
        chosen.extend(subset[int(i)] for i in indices)
    if not chosen:
        raise ValueError("No records match the requested policy/config")
    return chosen


def case_from_record(record, policy, data_cache, seed_base=None, panel_protocol="revision"):
    value = record["policies"][policy]
    source = value.get("proposal", value)
    stats = source.get("search", source.get("stats"))
    if not isinstance(stats, dict) or not isinstance(stats.get("history"), list):
        raise ValueError("Frozen policy record must contain search.history or stats.history")
    config = value.get("config", record.get("config"))
    if not isinstance(config, dict):
        raise ValueError("Frozen record must contain the exact search configuration")
    config = dict(config)
    # Older runner records saved the shared config, with these documented
    # per-policy overrides applied only at execution time.
    if policy == "Broad":
        config.update(mode="broad", frontier_seed=False)
    elif policy == "Broad-raw":
        config.update(mode="broad", frontier_seed=False, score_mode="raw")
    elif policy == "No-diagnostic":
        config.update(diagnostic=False, sketch_preview=False)
    elif policy == "With-diagnostic":
        config["diagnostic"] = True
    elif policy == "No-frontier":
        config["frontier_seed"] = False
    elif policy == "No-residual":
        config["frontier_only"] = True
    elif policy == "Best-seed":
        config["seed_schedule"] = "best"
    elif policy == "Gain-order":
        config["score_mode"] = "gain"
    elif policy == "Raw-order":
        config["score_mode"] = "raw"
    elif policy == "Preview":
        config["sketch_preview"] = True
    elif policy == "Evidence-order":
        config["score_mode"] = "evidence"
    elif policy == "No-migration-limit":
        config["migration_fraction"] = 1.
    elif policy in ("Swap", "Anneal", "Scenario-MIP", "FFD", "BFD"):
        raise ValueError("This controlled replay targets search_refined, not the selected baseline's engine")
    seed = source.get("search_seed", source.get("seed", record.get("search_seed", record.get("seed"))))
    if seed is None and seed_base is not None:
        seed = seed_base + 10000 * list(CAPACITIES).index(record["dataset"]) + int(record["origin"])
    if seed is None:
        raise ValueError("Missing recorded search seed; provide an explicit --seed-base only for a known protocol")
    name, origin = record["dataset"], int(record["origin"])
    if name not in data_cache:
        data_cache[name] = load_dataset(name)
    dataset = data_cache[name]
    build_rows = int(record.get("panel_rows", {}).get("build", config.get("build_rows", 24)))
    if panel_protocol == "revision":
        panels = {"build": dataset.window_panel(origin, 192, 64, max_rows=build_rows),
                  "selection": dataset.window_panel(origin, 64, 32),
                  "audit": dataset.window_panel(origin, 32, 0)}
    else:
        panels = trace_panels(dataset, origin, build_rows=build_rows)
    for role, count in record.get("panel_rows", {}).items():
        if role in panels and len(panels[role]) != count:
            raise ValueError(f"Recorded {role} panel has {count} rows; selected protocol reconstructs {len(panels[role])}")
    forecast = dataset.forecast(origin)
    forecast_mode = config.get("forecast_mode", "mean")
    if forecast_mode == "seasonal_blend":
        forecast = .5 * forecast + .5 * dataset.panel(origin - 24, 24)
    elif forecast_mode != "mean":
        raise ValueError("Unknown recorded common forecast mode")
    incumbent = record.get("incumbent_assignment")
    if incumbent is None:
        incumbent = ffd(forecast, dataset.capacity)
    case = {"dataset": name, "origin": origin, "seed": int(seed), "config": config,
            "seed_index": record.get("seed_index"), "source_implementation": record.get("implementation"),
            "history": stats["history"], "incumbent": incumbent, "forecast": forecast,
            "build": panels["build"], "selection": panels["selection"], "capacity": dataset.capacity}
    if "selected_assignment" in source:
        case["source_selected"] = source["selected_assignment"]
    elif "selected" in source:
        case["source_selected"] = source["selected"]
    if "candidates" in source:
        case["source_candidates"] = source["candidates"]
    return case


def aggregate(records):
    plans = tuple(plan for plan in PLANS if plan in records[0]["medians"])
    if any(set(r["medians"]) != set(plans) for r in records):
        raise ValueError("Cannot aggregate different execution plan sets")
    def one(rows):
        cpu = {plan: np.asarray([r["medians"][plan]["proposal_selection_cpu_seconds"] for r in rows]) for plan in plans}
        comparisons = {f"{left}_over_{right}": {
            "ratio_of_mean_costs": float(cpu[left].mean() / cpu[right].mean()),
            "median_paired_speedup": float(np.median(cpu[left] / cpu[right]))}
            for left, right in COMPARISONS if left in plans and right in plans}
        return {"scenarios": len(rows), "mean_cpu_seconds": {plan: float(values.mean()) for plan, values in cpu.items()},
                "comparisons": comparisons,
                "all_exact_checks_passed": all(r["exact_identity"]["all_profile_and_candidate_bytes"] for r in rows)}
    return {"overall": one(records), "cohorts": {name: one([r for r in records if r["dataset"] == name])
             for name in sorted({r["dataset"] for r in records})},
            "interpretation": "Controlled fixed-family execution comparison, not search quality; "
                              "means across overlapping trace origins are descriptive."}


def replay(args):
    records, sources = load_records(args.records)
    chosen = chosen_records(records, args.policy, args.per_cohort, args.config_id, args.seed_index)
    plan = {"source_records": sources, "policy": args.policy, "repetitions": args.repetitions,
            "plans": args.plans,
            "environment": environment(), "seed_index": args.seed_index, "panel_protocol": args.panel_protocol,
            "sampling": "Every scenario if per_cohort=0; otherwise equally spaced sorted origins, independent of outcomes",
            "scenarios": [{"dataset": r["dataset"], "origin": r["origin"]} for r in chosen],
            "seed_base_override": args.seed_base,
            "timing": "Serial one-thread, rotated plan order; complete recorded arm path; no admission or I/O",
            "auto_rule": "LOAD if B*T<256, PROJECT otherwise; frozen from prior kernel/development costs, not a learned model",
            "verification": "Separate instrumented raw-byte comparison of every profile and candidate identity",
            "decomposition": "Separate probes; inclusive categories may overlap; excluded from speedup timing"}
    save(args.output / "study_plan.json", plan)
    if args.plan_only:
        print(canonical_json({"planned_scenarios": len(chosen), "output": str(args.output)}))
        return
    results, data_cache = [], {}
    for record in chosen:
        case = case_from_record(record, args.policy, data_cache, args.seed_base, args.panel_protocol)
        result = compare_case(case, args.repetitions, args.plans)
        results.append(result)
        save(args.output / "records" / f'{case["dataset"]}_{case["origin"]}.json', result)
        save(args.output / "records.json", results)
        save(args.output / "summary.json", aggregate(results))
        print(canonical_json({"dataset": case["dataset"], "origin": case["origin"],
                              "speedups": result["speedups"], "exact": True}), flush=True)


def kernel_plan(panel, incumbent, candidates, capacity, plan):
    start = process_time()
    if plan == "auto":
        plan = "load_reuse" if panel.shape[0] * panel.shape[2] < 256 else "project_reuse"
    cache = (ProjectionReplay(panel, incumbent, capacity) if plan == "project_reuse" else
             ReplayCache(panel, incumbent, capacity) if plan == "load_reuse" else None)
    setup = process_time() - start
    load_cpu, profile_cpu, outputs = 0., 0., []
    for assignment in candidates:
        tick = process_time()
        if plan == "project_reuse":
            profile = cache.profile(assignment)
            profile_cpu += process_time() - tick
        else:
            loads = cache.loads(assignment) if cache else host_loads(panel, assignment)
            load_cpu += process_time() - tick
            tick = process_time()
            profile = refinement.profile_from_loads(loads, capacity)
            profile_cpu += process_time() - tick
        outputs.append(profile)
    total = process_time() - start
    stored = 0 if cache is None else sum(a.nbytes for a in cache.groups.values())
    if plan == "project_reuse":
        stored += sum(a.nbytes for columns in cache.projections.values() for a in columns)
    return {"total_cpu_seconds": total, "cache_setup_cpu_seconds": setup,
            "candidate_load_cpu_seconds": load_cpu, "profile_cpu_seconds": profile_cpu,
            "cached_view_bytes": stored,
            "candidate_demand_values": panel.size * len(candidates) if cache is None else cache.changed_values}, outputs


def kernel(args):
    # One-factor-at-a-time changes around (512,64,32,8). These are query
    # dimensions, not added evidence about placement search at these scales.
    shapes = [(512, 64, 32, 8), (128, 64, 32, 8), (2048, 64, 32, 8),
              (512, 16, 32, 8), (512, 128, 32, 8), (512, 64, 8, 8),
              (512, 64, 128, 8), (512, 64, 32, 2), (512, 64, 32, 32)]
    if args.shapes:
        shapes = [tuple(map(int, case.split(","))) for case in args.shapes.split(";")]
    results = []
    for index, shape in enumerate(shapes):
        if len(shape) != 4:
            raise ValueError("Each kernel shape must be N,H,B,T")
        n, h, b, t = shape
        if min(shape) <= 0 or h < 2 or n < 2 * h or n % h:
            raise ValueError("Kernel requires N divisible by H, H>=2, and at least two VMs per host")
        rng = np.random.default_rng(args.seed + index)
        incumbent = np.repeat(np.arange(h), n // h)
        panel = rng.lognormal(-.5 * .3**2, .3, size=(b, n, t)) * (.9 * h / n)
        candidates = []
        for _ in range(args.candidates):
            assignment = incumbent.copy()
            left, right = rng.choice(h, 2, replace=False)
            x = int(rng.choice(np.flatnonzero(incumbent == left)))
            y = int(rng.choice(np.flatnonzero(incumbent == right)))
            assignment[x], assignment[y] = right, left
            candidates.append(canonical_assignment(assignment))
        trials = {plan: [] for plan in args.plans}
        for repetition in range(args.repetitions):
            outputs = {}
            offset = repetition % len(args.plans)
            for plan in args.plans[offset:] + args.plans[:offset]:
                costs, outputs[plan] = kernel_plan(panel, incumbent, candidates, 1., plan)
                trials[plan].append(costs)
            for plan in args.plans[1:]:
                for a, z in zip(outputs["full"], outputs[plan]):
                    if a.shape != z.shape or a.dtype != z.dtype or a.tobytes() != z.tobytes():
                        raise AssertionError(f"Kernel six-coordinate profiles differ bitwise for {plan}")
        medians = {plan: {key: float(np.median([r[key] for r in rows])) for key in rows[0]}
                   for plan, rows in trials.items()}
        result = {"N": n, "H": h, "B": b, "T": t, "candidates": len(candidates),
                  "distinct_candidates": len(set(candidates)), "panel_bytes": panel.nbytes,
                  "seed": args.seed + index, "trials": trials, "medians": medians,
                  "bitwise_profiles": True,
                  "speedups_including_setup": {
                      f"{left}_over_{right}": medians[left]["total_cpu_seconds"] / medians[right]["total_cpu_seconds"]
                      for left, right in COMPARISONS if left in args.plans and right in args.plans}}
        results.append(result)
        save(args.output / "kernel_results.json", {
            "environment": environment(),
            "plans": args.plans,
            "auto_rule": "LOAD if B*T<256, PROJECT otherwise; fixed threshold",
            "scope": "Synthetic aggregation kernel only. Independent generated demands, fixed two-host swaps, "
                     "no search, forecasting, feasibility, admission or placement-quality inference. "
                     "Unchanged-group reuse stresses computation; it is not evidence of large-scale search quality.",
            "timing": "Fixture generation excluded; cached seed construction included in total; "
                      "all plans retain identical profiles; single numerical-library thread; rotated order. "
                      "PROJECT profile time includes aggregation of changed groups and canonical projection reduction.",
            "records": results})
        print(canonical_json({"shape": shape, "exact": True,
                              "speedups": result["speedups_including_setup"]}), flush=True)


def smoke(args):
    require_reuse_switch(args.plans)
    rng = np.random.default_rng(91731)
    forecast = np.full((24, 8), .18)
    incumbent = ffd(forecast, 1.)
    build = rng.lognormal(-.5 * .35**2, .35, (12, 24, 8)) * .19
    selection = rng.lognormal(-.5 * .35**2, .35, (10, 24, 8)) * .19
    base = {"cpu_limit": 1., "top_p": 4, "mutable_limit": 4,
              "initial_broad": 16, "broad_batch": 8, "migration_fraction": .5,
              "diagnostic": True, "sketch_preview": False,
              "replay_arms": ["broad", "exact", "broad", "exact"]}
    configurations = {"broad_exact": base,
                      "frontier_exact": {**base, "frontier_seed": True, "score_mode": "raw",
                                         "replay_arms": ["frontier:24", "exact", "exact"]},
                      "preview": {**base, "sketch_preview": True, "preview_pairs": 2,
                                  "preview_rows": 4},
                      "gain_auto_short": {**base, "frontier_seed": True, "score_mode": "gain",
                                          "diagnostic": False, "projected_views": "auto",
                                          "replay_arms": ["frontier:24", "exact", "exact"]},
                      "gain_auto_long": {**base, "frontier_seed": True, "score_mode": "gain",
                                         "diagnostic": False, "projected_views": "auto",
                                         "replay_arms": ["frontier:24", "exact", "exact"]}}
    results = []
    for index, (name, config) in enumerate(configurations.items()):
        case_build = np.tile(build, (3, 1, 1))[:32] if name == "gain_auto_long" else build
        case_selection = np.tile(selection, (4, 1, 1))[:32] if name == "gain_auto_long" else selection
        original = refinement.search_refined(incumbent, forecast, case_build, case_selection, 1., config,
                                             np.random.default_rng(123456))
        case = {"dataset": f"synthetic_smoke_{name}", "origin": index, "seed": 123456,
                "config": config, "history": original["stats"]["history"],
                "incumbent": incumbent, "forecast": forecast, "build": case_build,
                "selection": case_selection, "capacity": 1., "source_selected": original["selected"],
                "source_candidates": original["candidates"]}
        result = compare_case(case, args.repetitions, args.plans)
        results.append(result)
        print(canonical_json({"smoke": name, "status": "passed", "events": result["exact_identity"]["events"],
                              "profile_calls": result["exact_identity"]["profile_calls"]}))
    save(args.output / "smoke.json", {"scope": "Correctness smoke test, not paper evidence",
                                     "environment": environment(), "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", type=Path, required=True)
    common.add_argument("--repetitions", type=int, default=4)
    common.add_argument("--plans", nargs="+", choices=PLANS, default=list(PLANS),
                        help="FULL is mandatory and first; omit AUTO to reproduce the older three-plan study")
    replay_parser = sub.add_parser("replay", parents=[common])
    replay_parser.add_argument("--records", type=Path, required=True)
    replay_parser.add_argument("--policy", default="CARA")
    replay_parser.add_argument("--per-cohort", type=int, default=3, help="0 runs all matching scenarios")
    replay_parser.add_argument("--config-id")
    replay_parser.add_argument("--seed-base", type=int)
    replay_parser.add_argument("--seed-index", type=int, default=0,
                               help="Recorded proposal replicate to use; default 0")
    replay_parser.add_argument("--panel-protocol", choices=("revision", "legacy_adaptive"), default="revision",
                               help="Explicit chronological panel protocol; recorded panel sizes are checked")
    replay_parser.add_argument("--plan-only", action="store_true")
    replay_parser.set_defaults(run=replay)
    kernel_parser = sub.add_parser("kernel", parents=[common])
    kernel_parser.add_argument("--shapes", help="N,H,B,T cases separated by semicolons")
    kernel_parser.add_argument("--candidates", type=int, default=32)
    kernel_parser.add_argument("--seed", type=int, default=44017)
    kernel_parser.set_defaults(run=kernel)
    smoke_parser = sub.add_parser("smoke", parents=[common])
    smoke_parser.set_defaults(run=smoke)
    args = parser.parse_args()
    args.plans = tuple(args.plans)
    if not args.plans or args.plans[0] != "full" or len(set(args.plans)) != len(args.plans):
        parser.error("Plans must be unique and begin with full")
    if args.repetitions < 1 or getattr(args, "per_cohort", 0) < 0 or getattr(args, "candidates", 1) < 1:
        parser.error("Repetitions/candidates must be positive and per-cohort nonnegative")
    args.run(args)


if __name__ == "__main__":
    main()
