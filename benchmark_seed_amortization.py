"""Cold and amortized execution of one fixed connected-seed query sequence."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import OrderedDict
from contextlib import contextmanager
import gc
import json
from pathlib import Path
from time import process_time

import numpy as np
from scipy.optimize import linear_sum_assignment

import benchmark_family_queries as queries
from benchmark_family_execution import PLANS, encoded
from benchmark_family_generality import make_mixed_seed
from benchmark_hourly_queries import SOURCES, load_hourly, make_input
from benchmark_trace_packing import IndependentChecks, _ProjectionOracle
from cara.data import canonical_assignment, host_memberships
from cara.refinement import PROFILE_NAMES
from run_family_study import ROOT, save, serial


def fixed_pairs(hosts, count):
    """Spread the first round across all hosts; extend by unused offsets."""
    if hosts < 2 or count < 1 or count > hosts * (hosts - 1) // 2:
        raise ValueError("The fixed sequence requires enough distinct host pairs")
    sample = min(hosts, count)
    lefts = tuple(index * hosts // sample for index in range(sample))
    seen, pairs = set(), []
    for step in range(hosts):
        offset = (hosts // 2 + 1 + step) % hosts
        if not offset:
            continue
        for left in lefts:
            pair = tuple(sorted((left, (left + offset) % hosts)))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
                if len(pairs) == count:
                    return tuple(pairs)
    raise ArithmeticError("The fixed host-pair schedule is incomplete")


@contextmanager
def fixed_requests(seed, pairs, width):
    groups = host_memberships(seed)
    schedule = []
    for pair in pairs:
        taken = min(width // 2, *(len(groups[host]) for host in pair))
        mutable = tuple(sorted(groups[pair[0]][:taken] + groups[pair[1]][:taken]))
        if len(mutable) != width:
            raise ValueError("Every amortization family must use the declared complete width")
        schedule.append((pair, mutable))
    original = queries.requests

    def prefix(actual_seed, count, actual_width):
        if actual_seed != seed or actual_width != width or not 0 < count <= len(schedule):
            raise ValueError("A query changed the fixed seed or prefix workload")
        return schedule[:count]

    queries.requests = prefix
    try:
        yield schedule
    finally:
        queries.requests = original


class _BoundedOracle(_ProjectionOracle):
    """Independent NumPy projections with a bounded verification-only cache."""
    def __init__(self, panel, capacity, max_bytes=64 << 20):
        super().__init__(panel, capacity)
        self.groups = OrderedDict()
        self.max_bytes, self.bytes = max_bytes, 0

    def _group(self, residents):
        cached = self.groups.get(residents)
        if cached is not None:
            self.groups.move_to_end(residents)
            return cached
        value = super()._group(residents)
        self.bytes += sum(column.nbytes for column in value[1])
        while self.bytes > self.max_bytes and len(self.groups) > 1:
            _, (_, columns) = self.groups.popitem(last=False)
            self.bytes -= sum(column.nbytes for column in columns)
        return value


class SequenceChecks(IndependentChecks):
    """One dense seed matrix, independently updated once per count class.

    The general trace checker stores a dense base for every family. Reusing a
    single fixed seed here avoids H^2 times F verification memory; only sparse
    mutable descriptions and objective values remain cached across families.
    """
    def __init__(self, panel, forecast, incumbent, capacity, seed, objectives):
        super().__init__(panel, forecast, incumbent, capacity, objectives)
        self.reader = _BoundedOracle(panel, capacity)
        self.reference = self.reader.profile(incumbent)
        self.seed = canonical_assignment(seed)
        hosts = len(set(seed))
        self.seed_matrix = np.zeros((hosts, hosts), dtype=np.int64)
        np.add.at(self.seed_matrix, (self.incumbent, self.seed), 1)

    def family(self, family):
        if family.seed != self.seed:
            raise AssertionError("Amortization verification changed its fixed seed")
        key = family.pair, family.mutable
        if key not in self.descriptors:
            number = len(self.descriptors)
            self.descriptors[key] = number
            origins = tuple(self.incumbent[vm] for vm in family.free_vms)
            self.families[number] = (family.pair, family.mutable, family.symmetric, origins, {})
        return self.descriptors[key]

    def count(self, family, mask):
        pair, mutable, symmetric, origins, cache = self.families[self.family(family)]
        signature = tuple(sum(origin == host and not ((mask >> bit) & 1)
                              for bit, origin in enumerate(origins)) for host in sorted(set(origins)))
        if signature not in cache:
            matrix = self.seed_matrix.copy()
            for vm in mutable:
                matrix[self.incumbent[vm], self.seed[vm]] -= 1
            if symmetric:
                matrix[self.incumbent[mutable[0]], pair[0]] += 1
            for bit, origin in enumerate(origins):
                matrix[origin, pair[(mask >> bit) & 1]] += 1
            rows, columns = linear_sum_assignment(-matrix)
            cache[signature] = len(self.incumbent) - int(matrix[rows, columns].sum())
            self.counters["independent_count_classes"] = self.counters.get("independent_count_classes", 0) + 1
        return cache[signature]


def signatures(snapshots):
    # Prefixes share these per-family strings rather than duplicating a large
    # F-by-P-by-N encoded transcript for every requested prefix.
    return tuple(encoded(records) for records in snapshots)


COUNTERS = ("declared_masks", "forecast_passed_masks", "profiled_masks", "score_nodes",
    "score_excluded_masks", "indicator_bound_nodes", "indicator_excluded_masks",
    "matching_seed_compilations", "matching_seed_reuses", "matching_seed_index_entries",
    "matching_family_entries_scanned", "matching_core_rows", "matching_matrix_cells",
    "migration_matching_calls", "physical_matching_calls", "host_group_queries",
    "workload_materialized_values", "mutable_sum_builds", "mutable_sum_queries",
    "uniform_radius_compilations", "full_uniform_fast_nodes", "full_uniform_fallback_nodes",
    "aggregate_uniform_fast_nodes", "aggregate_uniform_fallback_nodes", "aggregate_bound_nodes",
    "aggregate_excluded_masks", "staged_pilot_profiles", "staged_sample_fallbacks", "staged_shape_fallbacks")
PHASES = ("preparation_cpu_seconds", "forecast_cpu_seconds", "predicate_cpu_seconds",
          "profile_cpu_seconds", "retention_cpu_seconds")


def measure(stats):
    families = stats["families"]
    views = stats.get("certificate_views", {})
    return {"cpu_seconds": stats["cpu_seconds"], "wall_seconds": stats["wall_seconds"],
        "session_setup_cpu_seconds": stats["session_setup_cpu_seconds"],
        **{name: sum(row[name] for row in families) for name in PHASES},
        **{name: sum(row["counters"].get(name, 0) for row in families) for name in COUNTERS},
        "certificate_view_compilations": views.get("compilations", 0),
        "certificate_view_queries": views.get("family_queries", 0),
        "certificate_view_update_nodes": views.get("aggregate_nodes", 0),
        "certificate_view_bytes": views.get("stored_bytes", 0),
        "maximum_subset_view_bytes": max((row["counters"].get("mutable_sum_bytes", 0) for row in families), default=0),
        "peak_resident_view_bytes": stats["peak_retained_view_array_bytes"],
        "materialized_values": stats["materialized_values"]}


def compare(panel, forecast, incumbent, capacity, seed, config, prefixes, plans, repetitions, memory_prefixes,
            reference_plan="no_score", width=8, memory_plans=None):
    memory_plans = tuple(plans) if memory_plans is None else tuple(memory_plans)
    if len(set(memory_plans)) != len(memory_plans) or any(plan not in plans for plan in memory_plans):
        raise ValueError("Memory plans must be a distinct subset of CPU plans")
    pairs = fixed_pairs(len(set(seed)), max(prefixes))
    checks = SequenceChecks(panel, forecast, incumbent, capacity, seed, config.get("objective_names", PROFILE_NAMES))
    expected, verification = None, {}
    with fixed_requests(seed, pairs, width) as requests:
        for plan in dict.fromkeys((reference_plan, *plans)):
            print(json.dumps({"verification_plan": plan, "families": max(prefixes)}), flush=True)
            before = dict(checks.counters)
            with checks.observe():
                snapshots, stats = queries.run_plan(panel, forecast, incumbent, capacity, config, plan,
                    seed=seed, families=max(prefixes), width=width,
                    observer=checks.pool, certificate_observer=checks.witness)
            actual = signatures(snapshots)
            if expected is None:
                expected = actual
            elif actual != expected:
                raise AssertionError("A full fixed sequence changed a ranked rich result")
            delta = {key: value - before.get(key, 0) for key, value in checks.counters.items()}
            for reason in ("forecast", "migration", "score"):
                count = sum(row["counters"].get(reason + "_excluded_masks", 0) for row in stats["families"])
                if delta.get(reason + "_leaves", 0) != count:
                    raise AssertionError("An excluded active leaf was not independently checked")
            verification[plan] = delta
        del checks, snapshots
        gc.collect()
        records = []
        for prefix_index, count in enumerate(prefixes):
            trials = []
            for repetition in range(repetitions):
                shift = (prefix_index + repetition) % len(plans)
                order = plans[shift:] + plans[:shift]
                measured = {}
                for plan in order:
                    print(json.dumps({"families": count, "trial": repetition + 1, "timed_plan": plan}), flush=True)
                    gc.collect()
                    snapshots, stats = queries.run_plan(panel, forecast, incumbent, capacity, config, plan,
                        seed=seed, families=count, width=width)
                    if signatures(snapshots) != expected[:count]:
                        raise AssertionError("A measured prefix changed the full-sequence answer")
                    measured[plan] = measure(stats)
                trials.append({"order": order, "plans": measured})
            resources = {}
            if count in memory_prefixes:
                for plan in memory_plans:
                    print(json.dumps({"families": count, "memory_plan": plan}), flush=True)
                    gc.collect()
                    snapshots, stats = queries.run_plan(panel, forecast, incumbent, capacity, config, plan,
                        seed=seed, families=count, width=width, trace_allocations=True)
                    if signatures(snapshots) != expected[:count]:
                        raise AssertionError("A memory probe changed a prefix answer")
                    resources[plan] = {"peak_traced_incremental_bytes": stats["peak_traced_incremental_bytes"],
                                       **{key: value for key, value in measure(stats).items() if key.endswith("_bytes")}}
            medians = {plan: {key: float(np.median([trial["plans"][plan][key] for trial in trials]))
                              for key in trials[0]["plans"][plan]} for plan in plans} if trials else {}
            records.append({"families": count, "all_prefix_answers_exact": True,
                            "trials": trials, "medians": medians, "resources": resources})
    return {"all_family_pools_exact": True, "all_materialized_profiles_exact": True,
            "all_exclusion_leaves_checked": True, "independent_count_classes": True,
            "retained_physical_oracles": True, "verification": verification,
            "fixed_requests": requests, "prefixes": records}


def run(args):
    if args.repetitions < 0 or args.width < 2 or args.width % 2:
        raise ValueError("Nonnegative repetitions and an even mutable width are required")
    if len(set(args.plans)) != len(args.plans) or any(plan not in PLANS for plan in args.plans):
        raise ValueError("Distinct registered physical plans are required")
    memory_plans = getattr(args, "memory_plans", None)
    if memory_plans is None:
        memory_plans = [plan for plan in ("family_only", "indexed") if plan in args.plans] or list(args.plans)
    if len(set(memory_plans)) != len(memory_plans) or any(plan not in args.plans for plan in memory_plans):
        raise ValueError("Memory plans must be a distinct subset of CPU plans")
    prefixes = tuple(sorted(set(min(value, 16) if args.smoke else value for value in args.prefixes)))
    if not prefixes or min(prefixes) < 1:
        raise ValueError("Positive family prefix lengths are required")
    raw = json.loads(args.config.read_text())
    config = raw.get("config", raw)
    protocol = {"sources": args.sources, "prefixes": prefixes, "width": args.width,
        "plans": args.plans, "config": config, "repetitions": args.repetitions,
        "memory_prefixes": args.memory_prefixes, "memory_plans": memory_plans,
        "reference_plan": args.reference_plan,
        "seed": "Feasible full-span one-resident-per-host cycle; original incumbent remains the movement reference",
        "query_schedule": "One fixed sequence of max(prefixes) distinct uniformly spread host pairs; all smaller runs use exact prefixes",
        "correctness": "One full sequence per plan checked against an exhaustive-profile reference, independent NumPy profiles, dense assignment once per count class, every excluded leaf, and every retained rich record; each measured prefix is checked against that sequence",
        "timing": "Fresh query session per prefix and serial rotated trial. Seed-index and bound-view construction are charged inside each session. All source loading, immutable panel construction, seed construction, correctness observers, garbage collection and output comparison are outside measurement",
        "memory_scope": "Separate tracemalloc probes after source/panel/forecast/seed exist: incremental Python and NumPy allocations, not process RSS or untracked solver allocations",
        "scope": "Complete-source what-if query execution under controlled connected overlap; not an observed production placement",
        "smoke": args.smoke}
    save(args.output / "protocol.json", protocol)
    records = []
    for source in args.sources[:1] if args.smoke else args.sources:
        if args.smoke:
            rng = np.random.default_rng(20261201)
            matrix = rng.uniform(.03, .22, (128, 48))
            metadata = {"source": "synthetic-smoke", "series": 128, "common_hours": 48}
        else:
            matrix, metadata = load_hourly(args.data_root, source)
        tick = process_time()
        panel, forecast, incumbent, capacity, starts = make_input(matrix, "full", 8)
        seed, seed_metadata = make_mixed_seed(incumbent, forecast, capacity, "full", 1)
        seed_metadata.pop("seed_assignment", None)
        preparation = process_time() - tick
        result = compare(panel, forecast, incumbent, capacity, seed, config,
            prefixes, args.plans, args.repetitions, set(args.memory_prefixes), args.reference_plan, args.width,
            memory_plans=memory_plans)
        records.append({**metadata, "N": len(incumbent), "H": len(set(incumbent)), "B": len(panel),
            "T": 8, "window_starts": starts.tolist(), "capacity": capacity,
            "source_preparation_cpu_seconds": preparation, "input_panel_bytes": panel.nbytes,
            "seed_metadata": seed_metadata, "seed_support_connected": seed_metadata["actual_seed_component_max"] == len(set(seed)),
            **result})
        save(args.output / "records.json", records)
        print(json.dumps({"source": source, "prefixes": prefixes, "all_exact": True,
                          "medians": {str(row["families"]): row["medians"] for row in result["prefixes"]}}), flush=True)
    save(args.output / "summary.json", {"complete": True, "sources": len(records),
        "all_exact": True, "prefixes": prefixes, "smoke": args.smoke,
        "rows": [{"source": row["source"], "seed_support_connected": row["seed_support_connected"],
                  "prefixes": [{"families": value["families"], "medians": value["medians"],
                                "resources": value["resources"]} for value in row["prefixes"]]} for row in records]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/release/family_frozen.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/cloudcons_hourly")
    parser.add_argument("--sources", choices=SOURCES, nargs="+", default=["azure2019"])
    parser.add_argument("--prefixes", type=int, nargs="+", default=[1, 4, 16, 64, 128])
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--plans", choices=PLANS, nargs="+", default=["family_only", "family_cost_gate", "indexed", "no_seed_index",
        "no_bound_views", "no_metric_cascade", "no_subset_views", "sparse"])
    parser.add_argument("--reference-plan", choices=["compact", "no_score"], default="no_score")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--memory-prefixes", type=int, nargs="*", default=[1, 128])
    parser.add_argument("--memory-plans", choices=PLANS, nargs="+",
                        help="CPU-plan subset for memory probes; defaults to family_only/indexed when present")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
