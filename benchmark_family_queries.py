"""Complete residual-family consumers on every hourly CPU series.

The deterministic partition and threshold define a query workload in the raw
source units. They do not enlarge the official placement-quality cohorts.
Every arm uses the same bounded canonical reader and exact balanced archive.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import gc
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter, process_time
import tracemalloc

import numpy as np

from benchmark_family_execution import PLANS, encoded, independent_assignment, matching_control, profile_oracle, verify_witness
from benchmark_hourly_queries import SOURCES, family_pairs, load_hourly, make_input
from cara.data import canonical_assignment, host_memberships
from cara.family_certificates import FamilyCertificates, MembershipIndex, ResidualFamily
from cara.family_score_bounds import FamilyScoreContext
from cara.gain_circuit import GainsContext
from cara.query_engine import ForecastViews, SharedQueryReplay
from cara.refinement import PROFILE_NAMES, _ordered_records, migration_profile
from run_family_study import CERTIFICATE_FLAGS, ROOT, save, serial


def requests(incumbent, count, width):
    groups = host_memberships(incumbent)
    result = []
    for pair in family_pairs(len(groups), count):
        left, right = groups[pair[0]], groups[pair[1]]
        taken = min(width // 2, len(left), len(right))
        mutable = tuple(sorted(left[:taken] + right[:taken]))
        result.append((pair, mutable))
    return result


def run_plan(panel, forecast, incumbent, capacity, config, plan, *, seed=None, adaptive=False, families=8, width=8,
             observer=None, certificate_observer=None, trace_allocations=False):
    """Execute a production plan or the temporary single-instance control."""
    with matching_control(plan == "static"):
        return _run_plan(panel, forecast, incumbent, capacity, config, plan,
                         seed=seed, adaptive=adaptive, families=families, width=width,
                         observer=observer, certificate_observer=certificate_observer,
                         trace_allocations=trace_allocations)


def _run_plan(panel, forecast, incumbent, capacity, config, plan, *, seed=None, adaptive=False, families=8, width=8,
              observer=None, certificate_observer=None, trace_allocations=False):
    """Return the exact pool after every complete family in one persistent session.

    Adaptive execution derives the next seed from the current best pool record.
    Pair labels follow the fixed schedule; mutable residents are recomputed from
    that seed. The original incumbent, reference, seen set, and pool never reset.
    """
    record_seeds = bool(adaptive or seed is not None)
    seed = canonical_assignment(incumbent if seed is None else seed)
    settings = {**config, **dict.fromkeys(CERTIFICATE_FLAGS, True), **PLANS[plan]}
    if (settings.get("archive_mode", "balanced") != "balanced" or settings.get("score_mode", "gain") != "gain"
            or settings.get("migration_weights") is not None):
        raise ValueError("This workload compares the unit-movement balanced gain consumer")
    objectives = tuple(settings.get("objective_names", PROFILE_NAMES))
    hosts, n, top = len(set(incumbent)), len(incumbent), int(settings.get("top_p", 16))
    limit = min(int(settings.get("max_migrations", np.floor(n * settings.get("migration_fraction", .2)))),
                settings.get("migration_cost_limit", np.inf))
    if top < 1 or width < 2 or width % 2 or families < 1:
        raise ValueError("Positive archive/family counts and an even residual width are required")
    if len(seed) != n or len(set(seed)) != hosts:
        raise ValueError("The family seed must retain the complete incumbent workload and host count")
    if trace_allocations:
        tracemalloc.start()
    started, wall_started = process_time(), perf_counter()
    cache = SharedQueryReplay(panel, incumbent, capacity,
                              max_views=settings.get("query_cache_views", 2048),
                              max_bytes=settings.get("query_cache_bytes", 32 << 20),
                              batch_values=settings.get("query_batch_values", 1 << 20))
    forecasts = ForecastViews.from_config(forecast, capacity, incumbent, settings)
    reference = cache.profile(incumbent)
    gain_context = GainsContext(reference, objectives)
    family_requests = ([(pair, None) for pair in family_pairs(hosts, families)] if adaptive
                       else requests(seed, families, width))
    seed_groups = None if adaptive else host_memberships(seed)
    index = MembershipIndex()
    eager = settings["family_execution"] == "eager"
    certified = settings["family_execution"] == "certified"
    seen = {incumbent} if eager else {index.assignment_key(incumbent)}
    pool, snapshots, rows = {}, [], []
    score_context = None
    setup_cpu = process_time() - started
    for pair, mutable in family_requests:
        family_started = process_time()
        if adaptive:
            seed_groups = host_memberships(seed)
            left, right = seed_groups[pair[0]], seed_groups[pair[1]]
            taken = min(width // 2, len(left), len(right))
            mutable = tuple(sorted(left[:taken] + right[:taken]))
        counters, assignments, movements, pending_mask = {}, {}, {}, {}
        before_values, before_queries = cache.materialized_values, cache.group_queries
        if eager:
            backgrounds = tuple(tuple(vm for vm in seed_groups[host] if vm not in mutable) for host in pair)
            symmetric = not any(backgrounds)
            mask_count = 1 << (len(mutable) - int(symmetric))
            group_sets = []
            for mask in range(mask_count):
                sides = [list(group) for group in backgrounds]
                for bit, vm in enumerate(mutable):
                    side = 0 if symmetric and bit == 0 else (mask >> (bit - int(symmetric))) & 1
                    sides[side].append(vm)
                group_sets.append(tuple(tuple(sorted(side)) for side in sides))
            family = certificates = None
        else:
            family = ResidualFamily(incumbent, seed, pair, mutable,
                                    reuse_migration_signatures=settings.get("family_migration_reuse", True),
                                    factor_matching=settings.get("family_factor_matching", True),
                                    matching_backend=settings.get("family_matching_backend"),
                                    use_unique_physical=settings.get("family_unique_physical", True),
                                    unique_physical_min_hosts=settings.get("family_unique_physical_min_hosts", 16))
            mask_count = family.mask_count
            certificates = FamilyCertificates(family, forecasts)
        preparation_cpu = process_time() - family_started

        def witness(value):
            if certificate_observer is not None:
                certificate_observer({**value, "family": family, "reference_profile": reference,
                                      "objective_names": objectives})

        tick = process_time()
        if eager:
            # Match the eager controller: changed-group predicates precede
            # complete-label expansion, so the baseline keeps this optimization.
            passing = forecasts.groups_feasible_many(group_sets)
            passed = tuple(mask for mask, fits in enumerate(passing) if fits)
            assignments = {mask: independent_assignment(seed, pair, mutable, mask) for mask in passed}
            active = tuple(mask for mask in passed if assignments[mask] not in seen)
        else:
            passed = certificates.forecast_masks(certify=certified and settings["family_certificate_forecast"],
                                                 observer=witness if certificate_observer is not None else None,
                                                 strategy=settings.get("family_forecast_strategy", "tree"))
            active = tuple(mask for mask in passed if family.identity(mask, index) not in seen)
        forecast_cpu = process_time() - tick
        tick = process_time()
        cutoff = min(record["key"] for record in pool.values()) if len(pool) == top else None
        if eager:
            surviving = active
        else:
            score = (certified and settings["family_certificate_score"] and cutoff is not None
                     and hosts >= int(settings.get("family_score_min_hosts", 0)))
            minimum = int(settings.get("family_score_min_leaves", 8))
            bounds = None
            if score and len(active) >= minimum:
                if score_context is None:
                    score_context = FamilyScoreContext.from_reader(cache, reference, objective_names=objectives)
                bounds = score_context.for_family(seed, pair, mutable, seed_loads=cache.loads(seed))
            surviving, excluded = certificates.prune(active, migration_limit=limit, cutoff=cutoff,
                score_bounds=bounds, certify_migration=certified and settings["family_certificate_migration"],
                certify_score=score, score_min_leaves=minimum,
                score_strategy=settings.get("family_score_strategy", "tree"),
                observer=witness if certificate_observer is not None else None)
            for exclusion in excluded:
                for mask in exclusion["active_masks"]:
                    seen.add(family.identity(mask, index))
        actions = []
        physical_calls = 0
        for mask in surviving:
            seen.add(assignments[mask] if eager else family.identity(mask, index))
            if eager:
                movement = migration_profile(incumbent, assignments[mask])
                physical_calls += 1
            else:
                count = family.movement_count(mask)
                movement = {"count": count, "cost": float(count)}
            if movement["count"] <= limit:
                actions.append(mask)
                movements[mask] = movement
        predicate_cpu = process_time() - tick
        tick = process_time()
        progressive = (not eager and score and settings.get("family_score_strategy", "tree") == "leaf"
                       and settings.get("family_progressive", True) and bool(certificates.leaf_upper_keys))
        batches = (certificates.score_batches(actions,
                   lambda: min(record["key"] for record in pool.values()) if len(pool) == top else None,
                   batch_size=settings.get("family_progressive_batch", 32),
                   observer=witness if certificate_observer is not None else None) if progressive else (actions,))
        profile_cpu, profiled = 0., 0
        for batch in batches:
            profile_started = process_time()
            profiles = (cache.profiles([assignments[mask] for mask in batch]) if eager else
                        cache.profiles_groups([family.memberships(mask) for mask in batch]))
            profile_cpu += process_time() - profile_started
            profiled += len(batch)
            batch_gains = gain_context.batch(profiles)
            for mask, profile, gains in zip(batch, profiles, batch_gains):
                movement = movements[mask]
                key = tuple(sorted(int(np.floor(value * (1 << 40))) for value in gains))
                key += (-movement["count"], -movement["cost"])
                if not eager and len(pool) == top and key < min(row["key"] for row in pool.values()):
                    continue
                assignment = assignments[mask] if eager else family.assignment(mask)
                if not eager and not settings.get("family_defer_physical", True):
                    physical = family.unique_physical_profile(mask, assignment) if family.use_unique_physical else None
                    if physical is None:
                        physical = migration_profile(incumbent, assignment)
                        physical_calls += 1
                    if (movement["count"], movement["cost"]) != (physical["count"], physical["cost"]):
                        raise AssertionError("Compact movement does not match the original physical action")
                    movement = physical
                pool[assignment] = {"assignment": assignment, "key": key, "movement": movement,
                                    "build_means": profile.mean(axis=0).tolist(), "relative_gains": gains.tolist()}
                if len(pool) > top:
                    retained = _ordered_records(pool.values())[:top]
                    pool = {row["assignment"]: row for row in retained}
                if (not eager and family.use_unique_physical and assignment in pool
                        and "aligned_assignment" not in movement):
                    pending_mask[assignment] = mask
                    if len(pending_mask) > top:
                        pending_mask = {a: m for a, m in pending_mask.items() if a in pool}
        for assignment, record in pool.items():
            if "aligned_assignment" in record["movement"]:
                continue
            physical = (family.unique_physical_profile(pending_mask[assignment], assignment)
                        if family is not None and family.use_unique_physical else None)
            if physical is None:
                physical = migration_profile(incumbent, assignment)
                physical_calls += 1
            if physical["count"] != record["movement"]["count"]:
                raise AssertionError("Factored count differs from physical matching")
            record["movement"] = physical
        records = _ordered_records(pool.values())
        snapshots.append(records)
        retention_cpu = process_time() - tick - profile_cpu
        family_cpu = process_time() - family_started
        if family is not None:
            counters.update(family.stats)
            counters.update(certificates.stats)
        counters.update(declared_masks=mask_count, forecast_passed_masks=len(passed),
                        duplicate_masks=len(passed) - len(active), profiled_masks=profiled,
                        physical_matching_calls=physical_calls,
                        workload_materialized_values=cache.materialized_values - before_values,
                        host_group_queries=cache.group_queries - before_queries)
        rows.append({"pair": pair, "mutable": mutable, "counters": counters,
                     **({"seed_assignment": seed} if record_seeds else {}),
                     "family_cpu_seconds": family_cpu, "preparation_cpu_seconds": preparation_cpu,
                     "forecast_cpu_seconds": forecast_cpu, "predicate_cpu_seconds": predicate_cpu,
                     "profile_cpu_seconds": profile_cpu, "retention_cpu_seconds": retention_cpu})
        if observer is not None:
            observer(records, {"pair": pair, "mutable": mutable, "seed_assignment": seed})
        if adaptive and records:
            seed = records[0]["assignment"]
    cpu, wall = process_time() - started, perf_counter() - wall_started
    peak = tracemalloc.get_traced_memory()[1] if trace_allocations else None
    if trace_allocations:
        tracemalloc.stop()
    statistics = {"cpu_seconds": cpu, "wall_seconds": wall, "session_setup_cpu_seconds": setup_cpu,
                  "adaptive": bool(adaptive),
                  "families": rows, "materialized_values": cache.materialized_values,
                  "forecast_materialized_values": forecasts.forecast_values,
                  "peak_retained_view_array_bytes": cache._registry.peak_view_bytes,
                  "retained_view_array_bytes": cache.cached_view_bytes,
                  "peak_traced_incremental_bytes": peak}
    return snapshots, statistics


def compare_source(matrix, metadata, args, config):
    started = process_time()
    panel, forecast, incumbent, capacity, starts = make_input(matrix, "full", 8)
    preparation_cpu = process_time() - started
    arguments = (panel, forecast, incumbent, capacity, config)
    expected, probes, oracle_cache = None, {}, {}
    # The eager oracle remains a separate untimed reference when omitted from trials.
    for plan in dict.fromkeys(("eager", *args.plans)):
        print(json.dumps({"source": metadata["source"], "verification_plan": plan}), flush=True)
        counters = {}

        def witness(value):
            verify_witness(value, incumbent, forecast, panel, capacity, counters, oracle_cache)

        with profile_oracle(counters, oracle_cache):
            snapshots, stats = run_plan(*arguments, plan, certificate_observer=witness)
        signature = encoded(snapshots)
        if expected is None:
            expected = signature
        elif signature != expected:
            raise AssertionError(f"{metadata['source']}: {plan} changed an exact family pool or physical tie")
        for reason in ("forecast", "migration", "score"):
            excluded = sum(row["counters"].get(reason + "_excluded_masks", 0) for row in stats["families"])
            if counters.get(reason + "_leaves", 0) != excluded:
                raise AssertionError("The witness observer missed excluded active leaves")
        probes[plan] = {**counters, "family_snapshots": len(snapshots),
                        "family_counters": [row["counters"] for row in stats["families"]]}
    oracle_cache.clear()  # Independent reference answers are outside all measurements.
    trials = []
    for repetition in range(args.repetitions):
        shift = (SOURCES.index(metadata["source"]) + repetition) % len(args.plans)
        order = args.plans[shift:] + args.plans[:shift]
        measurements = {}
        for plan in order:
            print(json.dumps({"source": metadata["source"], "trial": repetition + 1, "timed_plan": plan}), flush=True)
            gc.collect()
            snapshots, stats = run_plan(*arguments, plan)
            if encoded(snapshots) != expected:
                raise AssertionError("An uninstrumented query trial changed a retained pool")
            measurements[plan] = stats
        trials.append({"order": order, "plans": measurements})
    resources = {}
    if args.memory:
        for plan in args.plans:
            print(json.dumps({"source": metadata["source"], "memory_plan": plan}), flush=True)
            gc.collect()
            snapshots, stats = run_plan(*arguments, plan, trace_allocations=True)
            if encoded(snapshots) != expected:
                raise AssertionError("A resource probe changed a retained pool")
            resources[plan] = {name: stats[name] for name in ("peak_traced_incremental_bytes",
                                "peak_retained_view_array_bytes", "retained_view_array_bytes")}
    medians = {plan: {name: float(np.median([trial["plans"][plan][name] for trial in trials]))
                      for name in ("cpu_seconds", "wall_seconds", "session_setup_cpu_seconds")}
               for plan in args.plans} if trials else {}
    return {**metadata, "N": len(incumbent), "H": len(set(incumbent)), "B": len(panel), "T": 8,
            "capacity": capacity, "window_starts": starts.tolist(), "input_panel_bytes": panel.nbytes,
            "input_preparation_cpu_seconds": preparation_cpu, "complete_families": len(snapshots),
            "all_family_pools_exact": True, "all_materialized_profiles_exact": True,
            "all_exclusion_leaves_checked": True, "verification": probes, "trials": trials,
            "resources": resources, "median_seconds": medians}


def run(args):
    raw = json.loads(args.config.read_text())
    config = raw.get("config", raw)
    if (args.repetitions < 0 or len(set(args.sources)) != len(args.sources)
            or len(set(args.plans)) != len(args.plans)):
        raise ValueError("Distinct sources/plans and nonnegative repetitions required")
    files = [Path(__file__).resolve(), ROOT / "benchmark_family_execution.py", ROOT / "benchmark_static_matching.py",
             ROOT / "benchmark_reference_oracle.py", ROOT / "benchmark_hourly_queries.py",
             ROOT / "run_family_study.py", *sorted((ROOT / "cara").glob("*.py")), *sorted((ROOT / "cara/kernels").glob("*.c"))]
    plan = {"study": "complete_hourly_family_consumers", "sources": args.sources, "plans": args.plans,
            "settings": {name: PLANS[name] for name in args.plans}, "config": config,
            "families": 8, "mutable_width": 8, "horizon": 8, "windows": "every valid common-interval window",
            "repetitions": args.repetitions, "memory_probe": args.memory,
            "inputs": {name: sha256((args.data_root / name / "data-00000-of-00001.arrow").read_bytes()).hexdigest() for name in args.sources},
            "implementation": {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest() for path in files},
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "packages": {name: version(name) for name in ("numpy", "scipy", "pyarrow")}},
            "timing": "Serial rotated process CPU includes reader/forecast setup, family preparation, exact predicates and bounds, matching, canonical profiles, scoring, retained pools and original physical tie mapping. Raw Arrow loading, immutable panel construction, correctness oracles, garbage collection and serialization are outside timing.",
            "resource_scope": "Separate optional tracemalloc probes start after the raw matrix, full input panel and forecast exist. Report incremental traced Python/NumPy peak and retained view-array bytes; these values are not process RSS.",
            "correctness": "Untimed eager reference and every comparison verify exact scalar/group profile bytes, every ordered family pool including original rich physical movement, and every active excluded leaf. Independent canonical answers are memoized only during verification and discarded before timed and memory runs, which reproduce the full pool sequence.",
            "scope": "All original CPU series in each hourly input, with deterministic within-series NaN interpolation from the existing loader. Deterministic partition and threshold define query workloads, not official placement-quality benchmarks."}
    path = args.output / "study_plan.json"
    if path.exists() and json.loads(path.read_text()) != serial(plan):
        raise ValueError("Use a new directory for a changed complete-family query study")
    save(path, plan)
    if args.plan_only:
        print(json.dumps({"sources": args.sources, "plans": args.plans, "families_per_source": 8}))
        return
    results = []
    for name in args.sources:
        matrix, metadata = load_hourly(args.data_root, name)
        result = compare_source(matrix, metadata, args, config)
        results.append(result)
        save(args.output / "records.json", results)
        print(json.dumps({"source": name, "exact": True, "median_seconds": result["median_seconds"]}), flush=True)
    totals = {name: sum(row["median_seconds"][name]["cpu_seconds"] for row in results) for name in args.plans} if args.repetitions else {}
    save(args.output / "summary.json", {"complete": True, "sources": len(results), "all_exact": True,
                                       "total_median_cpu_seconds": totals,
                                       "speedup_over_compact": {name: totals["compact"] / value for name, value in totals.items()} if "compact" in totals else {}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/release/main_frozen.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/cloudcons_hourly")
    parser.add_argument("--sources", choices=SOURCES, nargs="+", default=list(SOURCES))
    parser.add_argument("--plans", choices=PLANS, nargs="+", default=list(PLANS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    run(parser.parse_args())
