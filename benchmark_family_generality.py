"""Complete-hourly repair queries beyond incumbent-rooted families.

Controlled cycle and dispersed/concentrated seeds preserve the source forecast,
capacity, complete panel and 20% movement budget. Adaptive traces instead carry
the retained best action into the next family. All plans retain the original
incumbent as their movement and scoring reference. This is an execution study;
it does not change the official consolidation-quality cohorts or targets.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
import gc
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import process_time

import numpy as np

from benchmark_family_execution import PLANS, encoded, profile_oracle, verify_witness
from benchmark_family_queries import requests, run_plan
from benchmark_hourly_queries import SOURCES, load_hourly, make_input
from cara.data import canonical_assignment, host_memberships, host_loads
from cara.family_certificates import ResidualFamily
from cara.refinement import migration_profile
from cara.repair import feasible
from run_family_study import ROOT, save, serial


def make_mixed_seed(incumbent, forecast, capacity, span, residents=1):
    groups = host_memberships(incumbent)
    hosts = len(groups)
    span = hosts if span == "full" else min(hosts, int(span))
    if residents < 1 or span < 2 or any(len(group) <= residents for group in groups):
        raise ValueError("Cycles need at least two hosts and leave a fixed resident on each host")
    if not feasible(incumbent, forecast, capacity, hosts):
        raise ValueError("The original incumbent must be forecast feasible")
    donors = {host: tuple(group[-residents:]) for host, group in enumerate(groups)}
    donor_loads = {host: forecast[list(vms)].sum(axis=0) for host, vms in donors.items()}
    old_loads = host_loads(forecast, incumbent)
    order = sorted(range(hosts), key=lambda host: (float(donor_loads[host].mean()), host))
    labels = np.asarray(incumbent, dtype=np.int64).copy()
    cycles, cuts = [], []
    for start in range(0, hosts, span):
        block = order[start:start + span]
        first = 0
        for position in range(len(block) - 1):
            destination, source = block[position], block[position + 1]
            changed = old_loads[destination] - donor_loads[destination] + donor_loads[source]
            if np.any(changed > capacity + 1e-12):
                cycles.append(tuple(block[first:position + 1]))
                cuts.append((destination, source))
                first = position + 1
        cycles.append(tuple(block[first:]))
    for cycle in cycles:
        if len(cycle) < 2:
            continue
        for position, destination in enumerate(cycle):
            source = cycle[(position + 1) % len(cycle)]
            labels[list(donors[source])] = destination
    seed = canonical_assignment(labels)
    if not feasible(seed, forecast, capacity, hosts):
        raise AssertionError("Cycle construction failed its complete canonical forecast check")
    movement = migration_profile(incumbent, seed)
    metadata = {"requested_span": span, "residents_per_host": residents,
                "cycle_lengths": [len(cycle) for cycle in cycles],
                "forecast_safety_cuts": cuts, "singleton_hosts": sum(len(cycle) == 1 for cycle in cycles),
                "seed_migrations": movement["count"], "seed_migration_fraction": movement["count"] / len(seed),
                "within_released_migration_budget": movement["count"] <= int(np.floor(.2 * len(seed))),
                "forecast_feasible": True,
                "actual_seed_component_max": max(map(len, cycles)),
                "seed_assignment": seed}
    return seed, metadata


def make_concentrated_seed(incumbent, forecast, capacity, width=8):
    """A connected, concentrated mixture that remains inside the 20% budget.

    One resident per host first moves around a whole-cluster cycle.  A central
    donor-load stratum of at most 20% of hosts exchanges three other residents.
    Some movement allowance is reserved for the subsequent residual queries.
    """
    seed, first = make_mixed_seed(incumbent, forecast, capacity, "full", 1)
    groups = host_memberships(incumbent)
    hosts, limit = len(groups), int(np.floor(.2 * len(incumbent)))
    remaining = limit - first["seed_migrations"]
    reserve = min(width, max(0, remaining // 2))
    target = min(hosts // 5, max(0, (remaining - reserve) // 3))
    if target < 2:
        target = 0
    donors = {host: tuple(group[-4:-1]) for host, group in enumerate(groups)}
    if any(len(group) < 3 or any(seed[vm] != host for vm in group) for host, group in donors.items()):
        raise ValueError("Concentrated mixing needs three further residents on each original host")
    donor_loads = {host: forecast[list(vms)].sum(axis=0) for host, vms in donors.items()}
    order = sorted(range(hosts), key=lambda host: (float(donor_loads[host].mean()), host))
    start = (hosts - target) // 2
    focus = order[start:start + target]
    current_loads = host_loads(forecast, seed)
    cycles, cuts, first_index = [], [], 0
    for position in range(len(focus) - 1):
        destination, source = focus[position], focus[position + 1]
        changed = current_loads[destination] - donor_loads[destination] + donor_loads[source]
        if np.any(changed > capacity + 1e-12):
            cycles.append(tuple(focus[first_index:position + 1]))
            cuts.append((destination, source))
            first_index = position + 1
    if focus:
        cycles.append(tuple(focus[first_index:]))
    labels = np.asarray(seed, dtype=np.int64).copy()
    for cycle in cycles:
        if len(cycle) > 1:
            for position, destination in enumerate(cycle):
                source = cycle[(position + 1) % len(cycle)]
                labels[list(donors[source])] = destination
    seed = canonical_assignment(labels)
    if not feasible(seed, forecast, capacity, hosts):
        raise AssertionError("Concentrated cycle seed is not canonically forecast feasible")
    physical = migration_profile(incumbent, seed)
    if physical["count"] > limit:
        raise AssertionError("Concentrated seed exceeds the released migration budget")
    metadata = {**first, "seed_kind": "concentrated", "focused_hosts": focus,
                "focused_cycle_lengths": list(map(len, cycles)), "focused_safety_cuts": cuts,
                "focused_residents_per_host": 3, "reserved_residual_moves": reserve,
                "seed_migrations": physical["count"], "seed_migration_fraction": physical["count"] / len(seed),
                "remaining_migration_allowance": limit - physical["count"],
                "within_released_migration_budget": True, "seed_assignment": seed}
    return seed, metadata


def make_dispersed_seed(incumbent, forecast, capacity, width=8):
    """Disperse the concentrated construction's additional migration count.

    Three times as many focus hosts each exchange one other resident. The
    original capacity, global cycle and residual allowance stay unchanged.
    This creates many degree-three vertices while retaining six of eight
    incumbent residents on most changed hosts.
    """
    seed, first = make_mixed_seed(incumbent, forecast, capacity, "full", 1)
    groups = host_memberships(incumbent)
    hosts, limit = len(groups), int(np.floor(.2 * len(incumbent)))
    remaining = limit - first["seed_migrations"]
    reserve = min(width, max(0, remaining // 2))
    concentrated_hosts = min(hosts // 5, max(0, (remaining - reserve) // 3))
    if concentrated_hosts < 2:
        concentrated_hosts = 0
    target = 3 * concentrated_hosts
    donors = {host: (group[-2],) for host, group in enumerate(groups)}
    if any(seed[group[0]] != host for host, group in donors.items()):
        raise ValueError("The dispersed donor must differ from the first-cycle donor")
    donor_loads = {host: forecast[list(vms)].sum(axis=0) for host, vms in donors.items()}
    order = sorted(range(hosts), key=lambda host: (float(donor_loads[host].mean()), host))
    start = (hosts - target) // 2
    focus = order[start:start + target]
    current_loads = host_loads(forecast, seed)
    cycles, cuts, first_index = [], [], 0
    for position in range(len(focus) - 1):
        destination, source = focus[position], focus[position + 1]
        changed = current_loads[destination] - donor_loads[destination] + donor_loads[source]
        if np.any(changed > capacity + 1e-12):
            cycles.append(tuple(focus[first_index:position + 1]))
            cuts.append((destination, source))
            first_index = position + 1
    if focus:
        cycles.append(tuple(focus[first_index:]))
    labels = np.asarray(seed, dtype=np.int64).copy()
    for cycle in cycles:
        if len(cycle) > 1:
            for position, destination in enumerate(cycle):
                source = cycle[(position + 1) % len(cycle)]
                labels[list(donors[source])] = destination
    seed = canonical_assignment(labels)
    if not feasible(seed, forecast, capacity, hosts):
        raise AssertionError("Dispersed cycle seed is not canonically forecast feasible")
    physical = migration_profile(incumbent, seed)
    if physical["count"] > limit:
        raise AssertionError("Dispersed seed exceeds the released migration budget")
    metadata = {**first, "seed_kind": "dispersed", "focused_hosts": focus,
                "focused_cycle_lengths": list(map(len, cycles)), "focused_safety_cuts": cuts,
                "focused_residents_per_host": 1, "matched_concentrated_focus_hosts": concentrated_hosts,
                "planned_extra_migrations": target, "reserved_residual_moves": reserve,
                "seed_migrations": physical["count"], "seed_migration_fraction": physical["count"] / len(seed),
                "remaining_migration_allowance": limit - physical["count"],
                "within_released_migration_budget": True, "seed_assignment": seed}
    return seed, metadata


def movement_core_bound(incumbent, seed, pair, physical=None):
    """A seed-optimum row-aligned bound, independent of the exact compiler.

    For an aligned unchanged column i, retained count d_i, outgoing a_i and
    incoming b_i, d_i >= a_i+b_i is a sufficient fixed-edge certificate.
    Every nonedited row failing it has n_i+1 <= 2 a_i+b_i; summing gives <=3m.
    """
    physical = migration_profile(incumbent, seed) if physical is None else physical
    aligned = tuple(physical["aligned_assignment"])
    old_sizes = Counter(incumbent)
    new_sizes = Counter(aligned)
    retained = Counter(old for old, new in zip(incumbent, aligned) if old == new)
    seed_to_physical = {}
    for original_label, physical_label in zip(seed, aligned):
        previous = seed_to_physical.setdefault(original_label, physical_label)
        if previous != physical_label:
            raise AssertionError("Physical alignment must map each complete group to one host")
    edited = {seed_to_physical[column] for column in pair}
    failures = []
    for old in old_sizes:
        d = retained[old]
        outgoing, incoming = old_sizes[old] - d, new_sizes[old] - d
        if old not in edited and d < outgoing + incoming:
            failures.append((old, old_sizes[old], outgoing, incoming, d))
    lhs = sum(n + 1 for _, n, _, _, _ in failures)
    rhs = sum(2*a + b for _, _, a, b, _ in failures)
    m = physical["count"]
    if not lhs <= rhs <= 3*m:
        raise AssertionError("Migration-parameter core bound failed")
    bound = min(len(old_sizes), 2 + (3*m) // (min(old_sizes.values()) + 1))
    return {"seed_migrations": m, "sufficient_unedited_failures": len(failures),
            "nonedited_size_sum": lhs, "nonedited_flow_sum": rhs,
            "three_m": 3*m, "bound_kernel_rows": bound,
            "minimum_incumbent_group_size": min(old_sizes.values())}


def trace_identity(statistics, initial_seed):
    return tuple((tuple(row.get("seed_assignment", initial_seed)), tuple(row["pair"]), tuple(row["mutable"]))
                 for row in statistics["families"])


def compact_statistics(statistics):
    result = dict(statistics)
    result["families"] = [{key: value for key, value in row.items() if key != "seed_assignment"}
                          for row in statistics["families"]]
    for row in result["families"]:
        row["counters"] = dict(row["counters"])
        counts = row["counters"]
        counts["physical_record_constructions"] = (counts.get("physical_matching_calls", 0)
                                                     + counts.get("physical_unique_profiles", 0))
    return result


def support_statistics(hosts, edges):
    """Cycle rank and branching of the original bipartite positive support."""
    parents = list(range(2 * hosts))
    rows, columns = [0] * hosts, [0] * hosts

    def root(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    count = 0
    for (row, column), value in edges.items():
        if value <= 0:
            continue
        count += 1
        rows[row] += 1
        columns[column] += 1
        first, second = root(row), root(hosts + column)
        if first != second:
            parents[second] = first
    components = len({root(node) for node in range(2 * hosts)})
    return {"edges": count, "vertices": 2 * hosts, "components": components,
            "feedback_rank": count - 2 * hosts + components,
            "rows_degree_at_least_three": sum(degree >= 3 for degree in rows),
            "columns_degree_at_least_three": sum(degree >= 3 for degree in columns),
            "row_degree_histogram": dict(Counter(rows)), "column_degree_histogram": dict(Counter(columns))}


def structural_records(incumbent, trace):
    """One seed table and all family cores, separate from execution timing."""
    indices, seeds, physicals, families = {}, [], {}, []
    for seed, pair, mutable in trace:
        if seed not in indices:
            indices[seed] = len(seeds)
            physicals[seed] = migration_profile(incumbent, seed)
            seeds.append({"assignment": seed, "migrations": physicals[seed]["count"],
                          "fraction": physicals[seed]["count"] / len(seed),
                          "support": support_statistics(len(set(seed)), Counter(zip(incumbent, seed)))})
        old = ResidualFamily(incumbent, seed, pair, mutable, matching_backend="factorized")
        current = ResidualFamily(incumbent, seed, pair, mutable, matching_backend="compiled")
        bound = movement_core_bound(incumbent, seed, pair, physicals[seed])
        actual_core = current.stats["matching_core_rows"]
        if actual_core > bound["bound_kernel_rows"]:
            raise AssertionError("The measured square core exceeds its migration-parameter bound")
        entries = Counter(zip(incumbent, seed))
        upper = entries.copy()
        for vm in mutable:
            upper[incumbent[vm], seed[vm]] -= 1
        if old.symmetric:
            upper[incumbent[mutable[0]], pair[0]] += 1
        for vm in old.free_vms:
            for column in pair:
                upper[incumbent[vm], column] += 1
        for edge, value in entries.items():
            upper[edge] = max(upper[edge], value)
        degrees = {}
        for name, graph in (("seed", entries), ("family", upper)):
            row_degrees, column_degrees = Counter(), Counter()
            for (row, column), weight in graph.items():
                if weight > 0:
                    row_degrees[row] += 1
                    column_degrees[column] += 1
            degrees[name] = row_degrees, column_degrees
        kernel = getattr(current._factor, "kernel", None)
        locks = kernel.fixed_edges if kernel is not None else ()
        locked_degrees = {}
        for name, (row_degrees, column_degrees) in degrees.items():
            pairs = Counter((row_degrees[row], column_degrees[column]) for row, column, *_ in locks)
            both = sum(count for (row_degree, column_degree), count in pairs.items()
                       if row_degree >= 3 and column_degree >= 3)
            either = sum(count for (row_degree, column_degree), count in pairs.items()
                         if row_degree >= 3 or column_degree >= 3)
            locked_degrees[name] = {
                "edges_with_both_endpoints_degree_at_least_three": both,
                "edges_with_any_endpoint_degree_at_least_three": either,
                "fraction_with_both_endpoints_degree_at_least_three": both / len(locks) if locks else 0.,
                "endpoint_degree_pairs": [{"row_degree": row_degree, "column_degree": column_degree, "edges": count}
                                          for (row_degree, column_degree), count in sorted(pairs.items())]}
        families.append({"seed_index": indices[seed], "pair": pair, "mutable": mutable,
                         "component_rows_before_peeling": old.stats["matching_active_rows"],
                         "component_columns_before_peeling": old.stats["matching_active_columns"],
                         "square_core_rows_after_peeling": actual_core,
                         "active_rows_after_peeling": current.stats["matching_active_rows"],
                         "active_columns_after_peeling": current.stats["matching_active_columns"],
                         "locked_edges": current.stats.get("matching_locked_edges", 0),
                         "trivial_fixed_edges": current.stats.get("matching_trivial_fixed_edges", 0),
                         "lock_rounds": current.stats.get("matching_lock_rounds", 0),
                         "locked_endpoint_degrees": locked_degrees,
                         "family_support": support_statistics(len(set(seed)), upper),
                         "migration_core_bound": bound})
    return {"seeds": seeds, "families": families}


def compare_setting(panel, forecast, incumbent, capacity, config, metadata, seed, specification, args, rotation):
    adaptive = specification["kind"] == "adaptive"
    families = args.adaptive_families if adaptive else args.families
    options = {"seed": seed, "adaptive": adaptive, "families": families, "width": args.width}
    arguments = (panel, forecast, incumbent, capacity, config)
    expected = expected_trace = None
    probes, oracle_cache = {}, {}
    for plan in dict.fromkeys(("eager", *args.plans)):
        print(json.dumps({"setting": specification, "verification_plan": plan}), flush=True)
        counters = {}

        def witness(value):
            verify_witness(value, incumbent, forecast, panel, capacity, counters, oracle_cache)

        with profile_oracle(counters, oracle_cache):
            snapshots, statistics = run_plan(*arguments, plan, certificate_observer=witness, **options)
        signature = encoded(snapshots)
        trace = trace_identity(statistics, seed)
        if expected is None:
            expected, expected_trace = signature, trace
        elif signature != expected or trace != expected_trace:
            raise AssertionError(f"{specification}: {plan} changed a rich family result or subsequent seed")
        for reason in ("forecast", "migration", "score"):
            excluded = sum(row["counters"].get(reason + "_excluded_masks", 0) for row in statistics["families"])
            if counters.get(reason + "_leaves", 0) != excluded:
                raise AssertionError("An excluded active leaf lacks an independent canonical witness")
        probes[plan] = {**counters, "family_snapshots": len(snapshots),
                        "family_counters": [row["counters"] for row in statistics["families"]]}
    oracle_cache.clear()
    structure = structural_records(incumbent, expected_trace)
    trials = []
    for repetition in range(args.repetitions):
        offset = (rotation + repetition) % len(args.plans)
        order = args.plans[offset:] + args.plans[:offset]
        measurements = {}
        for plan in order:
            print(json.dumps({"setting": specification, "trial": repetition + 1, "timed_plan": plan}), flush=True)
            gc.collect()
            snapshots, statistics = run_plan(*arguments, plan, **options)
            if encoded(snapshots) != expected or trace_identity(statistics, seed) != expected_trace:
                raise AssertionError("A timed trial changed the exact result or adaptive operation trajectory")
            measurements[plan] = compact_statistics(statistics)
        trials.append({"order": order, "plans": measurements})
    resources = {}
    if args.memory:
        for plan in args.plans:
            gc.collect()
            snapshots, statistics = run_plan(*arguments, plan, trace_allocations=True, **options)
            if encoded(snapshots) != expected or trace_identity(statistics, seed) != expected_trace:
                raise AssertionError("A memory probe changed the exact output")
            resources[plan] = {key: statistics[key] for key in ("peak_traced_incremental_bytes",
                                "peak_retained_view_array_bytes", "retained_view_array_bytes")}
    median = {plan: {name: float(np.median([trial["plans"][plan][name] for trial in trials]))
                      for name in ("cpu_seconds", "wall_seconds", "session_setup_cpu_seconds")}
               for plan in args.plans} if trials else {}
    return {**metadata, "setting": specification, "N": len(incumbent), "H": len(set(incumbent)),
            "B": len(panel), "T": panel.shape[2], "capacity": capacity, "complete_families": len(expected_trace),
            "all_profiles_exact": True, "all_family_pools_exact": True, "all_exclusion_leaves_checked": True,
            "all_adaptive_seeds_exact": True, "structure": structure, "verification": probes,
            "trials": trials, "resources": resources, "median_seconds": median, "complete": True}


def run(args):
    raw = json.loads(args.config.read_text())
    config = raw.get("config", raw)
    if (args.repetitions < 0 or args.width < 2 or args.width % 2 or args.families < 1 or args.adaptive_families < 1
            or len(set(args.plans)) != len(args.plans) or len(set(args.sources)) != len(args.sources)):
        raise ValueError("Use distinct sources/plans and valid counts")
    files = [Path(__file__).resolve(), ROOT / "benchmark_family_queries.py", ROOT / "benchmark_family_execution.py",
             ROOT / "benchmark_hourly_queries.py", ROOT / "benchmark_static_matching.py",
             ROOT / "benchmark_reference_oracle.py", ROOT / "run_family_study.py", *sorted((ROOT / "cara").glob("*.py")),
             *sorted((ROOT / "cara/kernels").glob("*.c"))]
    plan = {"study": "complete_hourly_nonincumbent_and_adaptive_queries", "sources": args.sources,
            "kinds": args.kinds, "spans": args.spans, "span_all_sources": args.span_all_sources,
            "residents_per_host": args.residents, "plans": args.plans, "config": config,
            "families": args.families, "adaptive_families": args.adaptive_families, "width": args.width,
            "horizon": 8, "windows": "every valid common-interval eight-hour window",
            "repetitions": args.repetitions, "memory_probe": args.memory,
            "implementation": {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest() for path in files},
            "inputs": {source: sha256((args.data_root / source / "data-00000-of-00001.arrow").read_bytes()).hexdigest()
                       for source in args.sources},
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "packages": {name: version(name) for name in ("numpy", "scipy", "pyarrow")}},
            "construction": "Cycle donors are the last resident(s) of each original host. Sort all donor bundles by mean forecast; partition by requested span; rotate each block; split only at forecast-infeasible increments. Concentrated seeds add a three-resident cycle in a central donor-load stratum of at most 20% of hosts, reserving residual migration room. Dispersed seeds use three times as many hosts with one different resident each, matching the planned extra movement while retaining the same capacity and reserve rule. Every primary seed passes the original forecast and movement constraints. All safety cuts and actual migration counts are retained; no measured runtime or future outcome chooses a seed.",
            "scope": "Unit-cost what-if query workloads on all source series. One-resident cycles, dispersed and concentrated mixtures respect the released 20% movement bound. Uniform two/three-resident stress cases explicitly use the full-N movement limit and are not deployment-quality evidence. Adaptive traces start at the incumbent and reuse each preceding best retained record as the next seed.",
            "timing": "Serial rotated process CPU includes shared readers, predicates, matching compilation/replay, complete canonical profiles, scoring, certified or full physical-map recovery, rich retention and adaptive seed selection. Source loading, immutable panels, seed construction, independent verification, structural analysis, garbage collection and serialization are outside timing.",
            "correctness": "Every materialized profile and excluded active leaf is canonically checked in untimed runs using an independent NumPy resident-profile oracle. Every rich ordered family pool and every next-seed/pair/mutable triple equals the eager reference. Verification caches are cleared before time/memory trials.",
            "physical_counters": "physical_matching_calls counts original full-matcher invocations; physical_unique_profiles counts certified unique-map recoveries. Their sum is physical_record_constructions and records the logical number of physical records constructed."}
    plan_path = args.output / "study_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != serial(plan):
        raise ValueError("Use a new output directory after changing the frozen study")
    save(plan_path, plan)
    if args.plan_only:
        print(json.dumps({"sources": args.sources, "kinds": args.kinds, "plans": args.plans}))
        return
    results = []
    for source in args.sources:
        matrix, metadata = load_hourly(args.data_root, source)
        panel, forecast, incumbent, capacity, starts = make_input(matrix, "full", 8)
        metadata = {**metadata, "window_starts": starts.tolist(), "input_panel_bytes": panel.nbytes}
        hosts = len(set(incumbent))
        settings = []
        if "cycles" in args.kinds:
            requested = args.spans if source == "azure2019" or args.span_all_sources else ["full"]
            spans = list(dict.fromkeys(hosts if span == "full" else min(hosts, int(span)) for span in requested))
            settings += [{"source": source, "kind": "cycles", "span": span, "residents": args.residents}
                         for span in spans]
        if "dispersed" in args.kinds:
            settings.append({"source": source, "kind": "dispersed"})
        if "concentrated" in args.kinds:
            settings.append({"source": source, "kind": "concentrated"})
        if "adaptive" in args.kinds:
            settings.append({"source": source, "kind": "adaptive"})
        for specification in settings:
            identity = "_".join(str(specification.get(key, "")) for key in ("source", "kind", "span", "residents")).rstrip("_")
            case_path = args.output / "cases" / (identity + ".json")
            if case_path.exists():
                result = json.loads(case_path.read_text())
                if result.get("setting") != specification or not result.get("complete"):
                    raise ValueError("A case checkpoint is incomplete or belongs to another setting")
            else:
                local_config = dict(config)
                if specification["kind"] == "cycles":
                    seed, seed_metadata = make_mixed_seed(incumbent, forecast, capacity,
                                                          specification["span"], args.residents)
                    if args.residents > 1:
                        local_config["max_migrations"] = len(incumbent)
                        local_config["migration_cost_limit"] = float(len(incumbent))
                    elif not seed_metadata["within_released_migration_budget"]:
                        raise AssertionError("Primary cycle seed exceeds the original migration budget")
                elif specification["kind"] == "concentrated":
                    seed, seed_metadata = make_concentrated_seed(incumbent, forecast, capacity, args.width)
                elif specification["kind"] == "dispersed":
                    seed, seed_metadata = make_dispersed_seed(incumbent, forecast, capacity, args.width)
                else:
                    seed, seed_metadata = incumbent, {"seed_kind": "adaptive", "seed_migrations": 0,
                                                       "within_released_migration_budget": True}
                seed_metadata = {key: value for key, value in seed_metadata.items() if key != "seed_assignment"}
                result = compare_setting(panel, forecast, incumbent, capacity, local_config,
                                         {**metadata, "seed_construction": seed_metadata}, seed,
                                         specification, args, len(results))
                save(case_path, result)
            results.append(result)
            save(args.output / "records.json", results)
            print(json.dumps({"completed": len(results), "setting": specification,
                              "median_seconds": result["median_seconds"]}), flush=True)
    totals = {name: sum(row["median_seconds"][name]["cpu_seconds"] for row in results) for name in args.plans} if args.repetitions else {}
    save(args.output / "summary.json", {"complete": True, "settings": len(results),
                                         "all_exact": True, "all_core_bounds_hold": True,
                                         "sum_of_setting_median_cpu_seconds": totals})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/release/family_frozen.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/cloudcons_hourly")
    parser.add_argument("--sources", choices=SOURCES, nargs="+", default=list(SOURCES))
    parser.add_argument("--kinds", choices=("cycles", "dispersed", "concentrated", "adaptive"), nargs="+",
                        default=["cycles", "dispersed", "concentrated", "adaptive"])
    parser.add_argument("--spans", nargs="+", default=["8", "32", "128", "full"])
    parser.add_argument("--span-all-sources", action="store_true")
    parser.add_argument("--residents", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--plans", choices=PLANS, nargs="+",
                        default=["components", "sparse", "static", "replay", "locked", "compiled_no_unique_maps", "compiled"])
    parser.add_argument("--families", type=int, default=8)
    parser.add_argument("--adaptive-families", type=int, default=16)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
