"""Complete-source repair queries from trace-derived feasible packings.

These are reproducible what-if workloads, not observed production placements.
The source table remains complete. A fixed first-day forecast constructs an
FFD incumbent; a completed prefix query sequence supplies a repaired seed.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
from time import process_time

import numpy as np
from scipy.optimize import linear_sum_assignment

import benchmark_family_queries as queries
from benchmark_family_execution import PLANS, encoded
from benchmark_hourly_queries import SOURCES, family_pairs, load_hourly
from benchmark_reference_oracle import CanonicalProfileOracle
from cara.data import canonical_assignment, host_memberships
from cara.family_certificates import ResidualFamily
from cara.family_score_bounds import FamilyScoreBounds
from cara.query_engine import SharedQueryReplay
from cara.refinement import PROFILE_NAMES, _relative_gains, migration_profile
from cara.repair import feasible, ffd
from run_family_study import ROOT, save, serial


class _ProjectionOracle(CanonicalProfileOracle):
    """Independent NumPy folds without retaining every intermediate load array."""
    def _group(self, residents):
        _, columns = super()._group(residents)
        self.groups[residents] = (None, columns)
        return self.groups[residents]


def make_packing(matrix):
    """One rule for every source, using only its first 24 observed hours."""
    forecast = np.repeat(matrix[:, :24].mean(axis=1, keepdims=True), 8, axis=1)
    means = forecast[:, 0]
    capacity = float(1.05 * max(float(means.max()), 8 * float(means.mean())))
    incumbent = ffd(forecast, capacity)
    if len(set(incumbent)) < 2 or not feasible(incumbent, forecast, capacity, len(set(incumbent))):
        raise ValueError("The complete-source packing needs two canonically feasible hosts")
    panel = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(matrix, 8, axis=1).transpose(1, 0, 2))
    return panel, forecast, incumbent, capacity


def packing_requests(seed, count, width):
    """Occupancy-stratified pairs and a total, potentially unbalanced budget."""
    groups = host_memberships(seed)
    hosts = sorted(range(len(groups)), key=lambda h: (len(groups[h]), h))
    n = len(hosts)
    ranks = [(0, n - 1), (n // 4, 3 * n // 4), (max(0, n // 2 - 1), n // 2), (max(0, n - 2), n - 1)]
    ranks.extend(family_pairs(n, max(count, 4)))
    result = []
    seen = set()
    for a, b in ranks:
        pair = tuple(sorted((hosts[a], hosts[b])))
        if a == b or pair in seen:
            continue
        seen.add(pair)
        sizes = tuple(len(groups[h]) for h in pair)
        total = min(width, sum(sizes))
        left = min(sizes[0], total // 2)
        right = min(sizes[1], total - left)
        left += total - left - right
        mutable = tuple(sorted(groups[pair[0]][:left] + groups[pair[1]][:right]))
        result.append((pair, mutable))
        if len(result) == count:
            break
    return result


@contextmanager
def request_schedule():
    old = queries.requests
    queries.requests = packing_requests
    try:
        yield
    finally:
        queries.requests = old


def occupancy(assignment):
    values = np.asarray([len(group) for group in host_memberships(assignment)])
    return {"min": int(values.min()), "median": float(np.median(values)), "max": int(values.max()),
            "quantiles": np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist(),
            "histogram": {str(x): int(np.count_nonzero(values == x)) for x in np.unique(values)}}


class IndependentChecks:
    """Canonical profiles and dense count-class objectives outside all timers.

    The compact plan enumerates every eligible profile first. Subsequent score
    exclusions are checked against those exact leaf keys. Dense matching is
    performed once per origin-count class, rather than once per VM mask.
    """

    def __init__(self, panel, forecast, incumbent, capacity, objectives):
        self.panel, self.forecast = panel, forecast
        self.incumbent, self.capacity = incumbent, capacity
        self.objectives = tuple(objectives)
        self.reader = _ProjectionOracle(panel, capacity)
        self.reference = self.reader.profile(incumbent)
        self.descriptors, self.families, self.pending, self.leaf_keys = {}, {}, {}, {}
        self.retained = {}
        self.counters = {}

    def family(self, family):
        key = (family.seed, family.pair, family.mutable)
        if key not in self.descriptors:
            number = len(self.descriptors)
            self.descriptors[key] = number
            base = np.zeros((family.hosts, family.hosts), dtype=np.int64)
            np.add.at(base, (self.incumbent, family.seed), 1)
            for vm in family.mutable:
                base[self.incumbent[vm], family.seed[vm]] -= 1
            if family.symmetric:
                base[self.incumbent[family.mutable[0]], family.pair[0]] += 1
            free_origins = tuple(self.incumbent[vm] for vm in family.free_vms)
            self.families[number] = (base, free_origins, {})
        return self.descriptors[key]

    def count(self, family, mask):
        number = self.family(family)
        base, origins, cache = self.families[number]
        signature = tuple(sum(origin == h and not ((mask >> bit) & 1)
                              for bit, origin in enumerate(origins)) for h in sorted(set(origins)))
        if signature not in cache:
            matrix = base.copy()
            for bit, origin in enumerate(origins):
                matrix[origin, family.pair[(mask >> bit) & 1]] += 1
            rows, columns = linear_sum_assignment(-matrix)
            cache[signature] = len(self.incumbent) - int(matrix[rows, columns].sum())
            self.counters["independent_count_classes"] = self.counters.get("independent_count_classes", 0) + 1
        return cache[signature]

    def profile(self, groups):
        # Same explicit NumPy oracle as CanonicalProfileOracle, without expanding
        # an N-label assignment for each enumerated mask.
        columns = [self.reader._group(tuple(group))[1] for group in groups]
        active, overload, frequency, excess, severity, affected, peaks = (
            np.stack(values, axis=0) for values in zip(*columns))
        scenes, active_count = len(self.panel), active.sum(axis=0)
        vs = np.divide(severity.sum(axis=0), affected.sum(axis=0), out=np.zeros(scenes), where=affected.any(axis=0))
        vr = np.divide(frequency.sum(axis=0), active_count, out=np.full(scenes, -1.), where=active_count > 0)
        raw = np.divide(excess.sum(axis=0), active_count, out=np.zeros(scenes), where=active_count > 0)
        mean_peak = np.divide(peaks.sum(axis=0), active_count, out=np.zeros(scenes), where=active_count > 0)
        par = np.divide(peaks.max(axis=0), mean_peak, out=np.full(scenes, -1.), where=mean_peak > 0)
        return np.column_stack((overload.any(axis=0), overload.mean(axis=0), vr, raw, vs, par))

    def key(self, profile, movement):
        gains = _relative_gains(self.reference, profile, self.objectives)
        return tuple(sorted(int(np.floor(value * (1 << 40))) for value in gains)) + (-movement, -float(movement))

    @contextmanager
    def observe(self):
        old_groups, old_count = ResidualFamily.memberships, ResidualFamily.movement_count
        old_profiles, old_scalar = SharedQueryReplay.profiles_groups, SharedQueryReplay.profile

        def groups(family, mask):
            result = old_groups(family, mask)
            self.pending[id(result)] = (family, int(mask))
            return result

        def count(family, mask):
            value = old_count(family, mask)
            if value != self.count(family, mask):
                raise AssertionError("A count-class objective differs from independent dense matching")
            self.counters["movement_count_calls"] = self.counters.get("movement_count_calls", 0) + 1
            return value

        def scalar(reader, assignment):
            result = old_scalar(reader, assignment)
            expected = self.reader.profile(assignment)
            if result.tobytes() != expected.tobytes():
                raise AssertionError("A scalar profile differs from independent canonical replay")
            return result

        def profiles(reader, memberships):
            result = old_profiles(reader, memberships)
            for groups, actual in zip(memberships, result):
                expected = self.profile(groups)
                if actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
                    raise AssertionError("A materialized profile differs from independent canonical replay")
                self.counters["materialized_profiles"] = self.counters.get("materialized_profiles", 0) + 1
                pending = self.pending.pop(id(groups), None)
                if pending is not None:
                    family, mask = pending
                    key = self.key(expected, self.count(family, mask))
                    leaf = (self.family(family), mask)
                    if leaf in self.leaf_keys and self.leaf_keys[leaf] != key:
                        raise AssertionError("A leaf changed its canonical key")
                    self.leaf_keys[leaf] = key
            return result

        ResidualFamily.memberships, ResidualFamily.movement_count = groups, count
        SharedQueryReplay.profiles_groups, SharedQueryReplay.profile = profiles, scalar
        try:
            yield
        finally:
            ResidualFamily.memberships, ResidualFamily.movement_count = old_groups, old_count
            SharedQueryReplay.profiles_groups, SharedQueryReplay.profile = old_profiles, old_scalar
            self.pending.clear()

    def witness(self, witness):
        family, reason = witness["family"], witness["reason"]
        masks = tuple(witness["active_masks"])
        for mask in masks:
            if not witness["start"] <= mask < witness["stop"]:
                raise AssertionError("Invalid exclusion interval")
            if reason == "forecast":
                changed = family.groups(mask)
                fits = all(bool(np.all(self.forecast[list(group)].sum(axis=0) <= self.capacity + 1e-12)) for group in changed)
                if fits:
                    raise AssertionError("A forecast exclusion removed a canonical feasible leaf")
            else:
                count = self.count(family, mask)
                if witness["movement_lower_bound"] > count:
                    raise AssertionError("A movement bound exceeds an exact count-class objective")
                if reason == "migration":
                    if count <= witness["migration_limit"]:
                        raise AssertionError("A movement exclusion removed an eligible leaf")
                elif reason == "score":
                    leaf = (self.family(family), mask)
                    if leaf not in self.leaf_keys:
                        # Such a leaf may have been forecast-ineligible in a
                        # different original-order mask only if the benchmark
                        # has changed its logical workload, which is an error.
                        raise AssertionError("The exhaustive-profile control did not score an excluded leaf")
                    if not self.leaf_keys[leaf] <= tuple(witness["optimistic_key"]) < tuple(witness["cutoff"]):
                        raise AssertionError("A score exclusion disagrees with its exhaustive canonical key")
                else:
                    raise AssertionError("Unknown exclusion reason")
            field = reason + "_leaves"
            self.counters[field] = self.counters.get(field, 0) + 1

    def pool(self, records, _context):
        for record in records:
            assignment = tuple(record["assignment"])
            if assignment not in self.retained:
                profile = self.reader.profile(assignment)
                physical = migration_profile(self.incumbent, assignment)
                self.retained[assignment] = (profile, physical)
            profile, physical = self.retained[assignment]
            if (not feasible(assignment, self.forecast, self.capacity, len(set(self.incumbent)))
                    or encoded(record["movement"]) != encoded(physical)
                    or tuple(record["key"]) != self.key(profile, physical["count"])
                    or np.asarray(record["build_means"]).tobytes() != profile.mean(axis=0).tobytes()
                    or np.asarray(record["relative_gains"]).tobytes() != _relative_gains(self.reference, profile, self.objectives).tobytes()):
                raise AssertionError("A retained rich record differs from its independent oracle")


@contextmanager
def phase_probe(result):
    originals = []
    def add(cls, name, label):
        original = getattr(cls, name)
        def wrapped(*args, **kwargs):
            tick = process_time()
            if name == "upper_keys":
                ranges = args[1] if len(args) > 1 else kwargs["ranges"]
                result["bound_ranges"] = result.get("bound_ranges", 0) + len(ranges)
                result["bound_batches"] = result.get("bound_batches", 0) + 1
            try:
                return original(*args, **kwargs)
            finally:
                result[label] = result.get(label, 0.) + process_time() - tick
        originals.append((cls, name, original))
        setattr(cls, name, wrapped)
    add(ResidualFamily, "identity", "identity_cpu_seconds")
    add(FamilyScoreBounds, "__init__", "bound_preparation_cpu_seconds")
    add(FamilyScoreBounds, "upper_keys", "bound_evaluation_cpu_seconds")
    try:
        yield
    finally:
        for cls, name, original in reversed(originals):
            setattr(cls, name, original)


def measures(stats):
    families = stats["families"]
    return {"cpu_seconds": stats["cpu_seconds"], "wall_seconds": stats["wall_seconds"],
        **{name: sum(row[name] for row in families) for name in ("preparation_cpu_seconds", "forecast_cpu_seconds", "predicate_cpu_seconds", "profile_cpu_seconds", "retention_cpu_seconds")},
        **{name: sum(row["counters"].get(name, 0) for row in families) for name in
            ("declared_masks", "forecast_passed_masks", "profiled_masks", "workload_materialized_values", "host_group_queries", "migration_matching_calls", "matching_matrix_cells", "matching_core_rows", "matching_dual_replays", "score_nodes", "score_excluded_masks",
             "uniform_radius_compilations", "full_uniform_fast_nodes", "full_uniform_fallback_nodes",
             "aggregate_uniform_fast_nodes", "aggregate_uniform_fallback_nodes", "aggregate_bound_nodes",
             "aggregate_excluded_masks", "staged_pilot_profiles", "staged_sample_fallbacks", "staged_shape_fallbacks")},
        "maximum_core_rows": max((row["counters"].get("matching_core_rows", 0) for row in families), default=0),
        "effective_widths": [len(row["mutable"]) for row in families],
        "mutable_sides": [[sum(row["seed_assignment"][vm] == host for vm in row["mutable"]) for host in row["pair"]] for row in families]}


def compare(panel, forecast, incumbent, capacity, seed, width, families, config, plans, repetitions):
    checks = IndependentChecks(panel, forecast, incumbent, capacity, config.get("objective_names", PROFILE_NAMES))
    expected, verification = None, {}
    with request_schedule():
        for plan in dict.fromkeys(("compact", *plans)):
            before = dict(checks.counters)
            with checks.observe():
                snapshots, stats = queries.run_plan(panel, forecast, incumbent, capacity, config, plan,
                    seed=seed, families=families, width=width, observer=checks.pool,
                    certificate_observer=checks.witness)
            actual = encoded(snapshots)
            if expected is not None and actual != expected:
                raise AssertionError("A plan changed a complete ordered rich family result")
            expected = actual
            delta = {key: value - before.get(key, 0) for key, value in checks.counters.items()}
            for reason in ("forecast", "migration", "score"):
                excluded = sum(row["counters"].get(reason + "_excluded_masks", 0) for row in stats["families"])
                if delta.get(reason + "_leaves", 0) != excluded:
                    raise AssertionError("A claimed exclusion did not receive an independent leaf check")
            verification[plan] = delta
        del checks
        gc.collect()
        probes = {}
        for plan in plans:
            diagnostic = {}
            with phase_probe(diagnostic):
                snapshots, stats = queries.run_plan(panel, forecast, incumbent, capacity, config, plan,
                    seed=seed, families=families, width=width)
            if encoded(snapshots) != expected:
                raise AssertionError("Diagnostic instrumentation changed the result")
            probes[plan] = {**diagnostic, "counters": [row["counters"] for row in stats["families"]]}
        trials = []
        for repetition in range(repetitions):
            shift = repetition % len(plans)
            order = plans[shift:] + plans[:shift]
            measured = {}
            for plan in order:
                gc.collect()
                snapshots, stats = queries.run_plan(panel, forecast, incumbent, capacity, config, plan,
                    seed=seed, families=families, width=width)
                if encoded(snapshots) != expected:
                    raise AssertionError("A measured query changed the exact result")
                measured[plan] = measures(stats)
            trials.append({"order": order, "plans": measured})
    numeric = [key for key, value in measures(stats).items() if isinstance(value, (int, float))]
    medians = {plan: {key: float(np.median([trial["plans"][plan][key] for trial in trials])) for key in numeric}
               for plan in plans} if trials else {}
    return {"all_family_pools_exact": True, "all_materialized_profiles_exact": True,
        "all_exclusion_leaves_checked": True, "independent_count_class_oracles": True,
        "retained_physical_oracles": True, "verification": verification, "diagnostics": probes,
        "trials": trials, "medians": medians,
        "effective_widths": measures(stats)["effective_widths"], "mutable_sides": measures(stats)["mutable_sides"]}


def run(args):
    if args.repetitions < 0 or args.families < 1 or args.width16_families < 1 or any(w < 2 or w % 2 for w in args.widths):
        raise ValueError("Nonnegative repetitions, positive families, and even widths are required")
    if any(plan not in PLANS for plan in args.plans):
        raise ValueError("A requested plan is not registered in benchmark_family_execution.PLANS")
    raw = json.loads(args.config.read_text())
    config = raw.get("config", raw)
    specification = {"sources": args.sources, "widths": args.widths, "kinds": args.kinds,
        "plans": args.plans, "repetitions": args.repetitions, "families": args.families,
        "width16_families": args.width16_families, "seed_prefix_windows": 32, "seed_prefix_families": 8,
        "capacity": "1.05 * max(maximum first-day VM mean, 8 * average first-day VM mean)",
        "packing": "Forecast-feasible first fit decreasing, all VMs retained, no fixed occupancy",
        "pair_policy": "Seed occupancy strata: small-large, lower-upper quartiles, central, large-large; no target scores used",
        "mutable_policy": "Total width budget; unused half-side quota goes to the other host. Every pair with at least width residents uses the full width",
        "seed": "Packed incumbent or final best record from eight completed adaptive width-eight prefix queries; all fallbacks retained",
        "windows": "Every complete eight-hour window in every source's common interval",
        "scope": "Trace-derived what-if workloads; neither observed physical placements nor placement-quality generalization",
        "correctness": "Exhaustive-profile Compact first; independent NumPy canonical profiles, dense movement once per count class, every excluded leaf and every retained rich physical record",
        "diagnostics": "Separate instrumented untimed pass reports identity and bound CPU; formal timing has no observers",
        "config": config, "smoke": args.smoke}
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != serial(specification):
        raise ValueError("Use a new output directory for a changed packing-query protocol")
    save(path, specification)
    records_path = args.output / "records.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    completed = {(row["source"], row["kind"], row["width"]) for row in records}
    for source in args.sources:
        if args.smoke:
            rng = np.random.default_rng(251110)
            matrix = rng.uniform(.03, .22, (64, 48))
            metadata = {"source": "synthetic-smoke", "series": 64, "common_hours": 48}
        else:
            matrix, metadata = load_hourly(args.data_root, source)
        started = process_time()
        panel, forecast, incumbent, capacity = make_packing(matrix)
        preparation = process_time() - started
        with request_schedule():
            prefix, prefix_stats = queries.run_plan(panel[:min(32, len(panel))], forecast, incumbent, capacity,
                config, "compiled", adaptive=True, families=8, width=8)
        repaired = prefix[-1][0]["assignment"] if prefix[-1] else incumbent
        if (not feasible(repaired, forecast, capacity, len(set(incumbent)))
                or migration_profile(incumbent, repaired)["count"] > int(np.floor(len(incumbent) * config.get("migration_fraction", .2)))):
            raise AssertionError("The prefix-derived seed violates the released constraints")
        for kind in args.kinds:
            seed = incumbent if kind == "packed" else repaired
            for width in args.widths:
                if (metadata["source"], kind, width) in completed:
                    continue
                families = args.width16_families if width >= 16 else args.families
                print(json.dumps({"source": metadata["source"], "kind": kind, "width": width, "phase": "verification"}), flush=True)
                result = compare(panel, forecast, incumbent, capacity, seed, width, families,
                                 config, tuple(args.plans), args.repetitions)
                record = {**metadata, "kind": kind, "width": width, "families": families,
                    "N": len(incumbent), "H": len(set(incumbent)), "B": len(panel), "T": 8,
                    "capacity": capacity, "input_preparation_cpu_seconds": preparation,
                    "incumbent_assignment": incumbent, "seed_assignment": seed,
                    "incumbent_occupancy": occupancy(incumbent), "seed_occupancy": occupancy(seed),
                    "seed_migrations": migration_profile(incumbent, seed)["count"],
                    "migration_core_bound": min(len(set(incumbent)), 2 + 3 * migration_profile(incumbent, seed)["count"] // (occupancy(incumbent)["min"] + 1)),
                    "seed_prefix_cpu_seconds": prefix_stats["cpu_seconds"], **result}
                records.append(record)
                save(args.output / "records.json", records)
                print(json.dumps({"source": metadata["source"], "kind": kind, "width": width, "medians": result["medians"]}), flush=True)
        if args.smoke:
            break
    totals = {plan: sum(row["medians"][plan]["cpu_seconds"] for row in records) for plan in args.plans} if args.repetitions else {}
    expected_cases = (1 if args.smoke else len(args.sources)) * len(args.kinds) * len(args.widths)
    save(args.output / "summary.json", {"complete": len(records) == expected_cases, "contexts": len(records), "all_exact": True,
        "all_series_and_windows": not args.smoke, "total_median_cpu_seconds": totals})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/release/family_frozen.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/cloudcons_hourly")
    parser.add_argument("--sources", choices=SOURCES, nargs="+", default=list(SOURCES))
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--kinds", choices=["packed", "repaired"], nargs="+", default=["packed", "repaired"])
    parser.add_argument("--plans", nargs="+", default=["family_only", "family_cost_gate", "indexed",
        "no_seed_index", "no_bound_views", "no_metric_cascade", "no_subset_views", "sparse", "components",
        "static", "replay", "no_compiled_bounds", "full_uniform", "recomputed_bounds"])
    parser.add_argument("--families", type=int, default=4)
    parser.add_argument("--width16-families", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
