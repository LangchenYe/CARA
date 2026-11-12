"""Fixed-work complete-search comparison of residual-family execution plans.

An eager search freezes its completed operation path. Untimed runs verify every
materialized canonical profile, every family-level retained pool, the selected
physical action and all excluded leaves. Serial rotated trials then execute the
same path without observers. No target outcomes are read by this benchmark.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from cara.certification import exact_pareto_gate
from cara.data import canonical_assignment, load_dataset
from cara.query_engine import SharedQueryReplay
from cara.refinement import PROFILE_NAMES, _relative_gains, migration_profile, panel_profile, search_refined
from cara.repair import feasible, ffd
from run_family_study import CERTIFICATE_FLAGS, ROOT, panels_for, save, serial
from benchmark_static_matching import matching_control
from benchmark_reference_oracle import CanonicalProfileOracle


PLANS = {
    "components": {"family_execution": "certified", "family_matching_backend": "factorized"},
    "static": {"family_execution": "certified", "family_matching_backend": "compiled", "family_unique_physical": False,
               "family_seed_compilation": False},
    "replay": {"family_execution": "certified", "family_matching_backend": "replay"},
    "locked": {"family_execution": "certified", "family_matching_backend": "locked"},
    "compiled": {"family_execution": "certified", "family_matching_backend": "compiled"},
    "compiled_no_unique_maps": {"family_execution": "certified", "family_matching_backend": "compiled", "family_unique_physical": False},
    "sparse": {"family_execution": "certified", "family_matching_backend": "sparse"},
    "no_progressive": {"family_execution": "certified", "family_progressive": False},
    "no_bootstrap": {"family_execution": "certified", "family_bootstrap_bounds": False},
    "eager": {"family_execution": "eager"},
    "compact": {"family_execution": "compact", "family_factor_matching": False,
                "family_matching_backend": "dense", "family_defer_physical": False},
    "certified": {"family_execution": "certified"},
    "no_factorization": {"family_execution": "certified", "family_factor_matching": False,
                         "family_matching_backend": "dense"},
    "no_late_matching": {"family_execution": "certified", "family_defer_physical": False},
    "no_forecast": {"family_execution": "certified", "family_certificate_forecast": False},
    "no_migration": {"family_execution": "certified", "family_certificate_migration": False},
    "no_score": {"family_execution": "certified", "family_certificate_score": False},
    "no_matching_reuse": {"family_execution": "certified", "family_migration_reuse": False},
    "family_only": {"family_execution": "certified", "family_matching_backend": "compiled",
                    "family_seed_compilation": False, "family_bound_views": False, "family_metric_cascade": False, "family_subset_views": False, "family_structural_bound_gate": False},
    "family_cost_gate": {"family_execution": "certified", "family_matching_backend": "compiled",
                         "family_seed_compilation": False, "family_bound_views": False,
                         "family_metric_cascade": False, "family_subset_views": False,
                         "family_structural_bound_gate": True},
    "indexed": {"family_execution": "certified", "family_matching_backend": "compiled",
                "family_seed_compilation": True, "family_bound_views": True, "family_metric_cascade": True, "family_subset_views": True , "family_structural_bound_gate": True},
    "no_seed_index": {"family_execution": "certified", "family_matching_backend": "compiled",
                      "family_seed_compilation": False, "family_bound_views": True, "family_metric_cascade": True, "family_subset_views": True , "family_structural_bound_gate": True},
    "no_bound_views": {"family_execution": "certified", "family_matching_backend": "compiled",
                       "family_seed_compilation": True, "family_bound_views": False, "family_metric_cascade": True, "family_subset_views": True , "family_structural_bound_gate": True},
    "no_metric_cascade": {"family_execution": "certified", "family_matching_backend": "compiled",
                          "family_seed_compilation": True, "family_bound_views": True, "family_metric_cascade": False, "family_subset_views": True , "family_structural_bound_gate": True},
    "no_subset_views": {"family_execution": "certified", "family_matching_backend": "compiled",
                        "family_seed_compilation": True, "family_bound_views": True,
                        "family_metric_cascade": True, "family_subset_views": False , "family_structural_bound_gate": True},
}

for _plan in ("family_only", "family_cost_gate"):
    PLANS[_plan].update(family_compiled_bounds=False, family_staged_acquisition=False)
for _plan in ("indexed", "no_seed_index", "no_bound_views", "no_subset_views"):
    PLANS[_plan].update(family_compiled_bounds=True, family_staged_acquisition=True)
PLANS["no_metric_cascade"].update(family_compiled_bounds=True, family_staged_acquisition=False)
PLANS["recomputed_bounds"] = {**PLANS["indexed"], "family_compiled_bounds": False,
                              "family_staged_acquisition": False}
PLANS["full_uniform"] = {**PLANS["indexed"], "family_staged_acquisition": False}
PLANS["no_compiled_bounds"] = {**PLANS["indexed"], "family_compiled_bounds": False}


def encoded(value):
    return json.dumps(serial(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def independent_assignment(seed, pair, mutable, mask):
    """Original complete-label enumeration, independent of ResidualFamily."""
    labels = np.asarray(seed, dtype=int).copy()
    mutable = tuple(mutable)
    fixed = [vm for vm in np.flatnonzero(np.isin(labels, pair)) if vm not in mutable]
    symmetric = not fixed
    for bit, vm in enumerate(mutable):
        side = 0 if symmetric and bit == 0 else (mask >> (bit - int(symmetric))) & 1
        labels[vm] = pair[side]
    return canonical_assignment(labels)


def oracle_profile(panel, assignment, capacity, cache=None):
    """Independent canonical answer, shared only within an untimed immutable case."""
    if cache is None:
        return panel_profile(panel, assignment, capacity)
    cache.setdefault(("source", id(panel)), panel)  # Keep source identities alive until the case ends.
    key = ("profile", id(panel), float(capacity), np.asarray(assignment, dtype=np.int64).tobytes())
    if key not in cache:
        reader_key = ("canonical_oracle", id(panel), float(capacity))
        if reader_key not in cache:
            cache[reader_key] = CanonicalProfileOracle(panel, capacity)
        expected = cache[reader_key].profile(assignment)
        expected.flags.writeable = False
        cache[key] = expected
    return cache[key]


def oracle_movement(incumbent, assignment, cache=None):
    if cache is None:
        return migration_profile(incumbent, assignment)
    key = ("movement", tuple(incumbent), np.asarray(assignment, dtype=np.int64).tobytes())
    if key not in cache:
        full = migration_profile(incumbent, assignment)
        cache[key] = {"count": full["count"], "cost": full["cost"]}
    return cache[key]


def verify_witness(witness, incumbent, forecast, build, capacity, counters, oracle_cache=None):
    family = witness["family"]
    reason = witness["reason"]
    masks = tuple(witness["active_masks"])
    if not masks or any(not witness["start"] <= mask < witness["stop"] for mask in masks):
        raise AssertionError("Exclusion witness has invalid active leaves")
    counters["witnesses"] = counters.get("witnesses", 0) + 1
    counters[reason + "_leaves"] = counters.get(reason + "_leaves", 0) + len(masks)
    for mask in masks:
        assignment = independent_assignment(family.seed, family.pair, family.mutable, mask)
        if len(set(assignment)) != len(set(incumbent)):
            raise AssertionError("A witness included an empty-host leaf")
        if reason == "forecast":
            if feasible(assignment, forecast, capacity, len(set(incumbent))):
                raise AssertionError("Forecast certificate excluded a feasible canonical leaf")
            continue
        movement = oracle_movement(incumbent, assignment, oracle_cache)
        if witness["movement_lower_bound"] > movement["count"]:
            raise AssertionError("Migration lower bound exceeds an exact leaf count")
        if reason == "migration":
            if movement["count"] <= witness["migration_limit"]:
                raise AssertionError("Migration certificate excluded an admissible leaf")
        elif reason == "score":
            profile = oracle_profile(build, assignment, capacity, oracle_cache)
            gains = _relative_gains(witness["reference_profile"], profile, tuple(witness["objective_names"]))
            key = tuple(sorted(int(np.floor(value * (1 << 40))) for value in gains))
            key += (-movement["count"], -movement["cost"])
            if not key <= tuple(witness["optimistic_key"]) < tuple(witness["cutoff"]):
                raise AssertionError("Score certificate failed its canonical leaf oracle")
        else:
            raise AssertionError(f"Unknown certificate reason: {reason}")


@contextmanager
def profile_oracle(counters, oracle_cache=None):
    """Check scalar and group APIs, without double-counting delegated batches."""
    original = SharedQueryReplay.profiles_groups
    original_scalar = SharedQueryReplay.profile

    def checked_scalar(reader, assignment):
        output = original_scalar(reader, assignment)
        expected = oracle_profile(reader.panel, assignment, reader.capacity, oracle_cache)
        if output.shape != expected.shape or output.dtype != expected.dtype or output.tobytes() != expected.tobytes():
            raise AssertionError("Materialized scalar profile differs from canonical binary64 evaluation")
        counters["materialized_profiles"] = counters.get("materialized_profiles", 0) + 1
        counters["scalar_profiles"] = counters.get("scalar_profiles", 0) + 1
        return output

    def checked(reader, memberships):
        output = original(reader, memberships)
        if output.shape != (len(memberships), len(reader.panel), len(PROFILE_NAMES)):
            raise AssertionError("The group reader did not return every complete profile")
        for groups, actual in zip(memberships, output):
            labels = np.full(reader.panel.shape[1], -1, dtype=int)
            previous = -1
            for host, group in enumerate(groups):
                if not group or tuple(group) != tuple(sorted(set(group))) or group[0] <= previous:
                    raise AssertionError("Compact memberships are not in canonical order")
                previous = group[0]
                if np.any(labels[list(group)] != -1):
                    raise AssertionError("Compact memberships repeat a VM")
                labels[list(group)] = host
            if np.any(labels < 0):
                raise AssertionError("Compact memberships omit a VM")
            expected = oracle_profile(reader.panel, tuple(labels), reader.capacity, oracle_cache)
            if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
                raise AssertionError("Materialized profile differs from canonical binary64 evaluation")
            counters["materialized_profiles"] = counters.get("materialized_profiles", 0) + 1
        return output

    SharedQueryReplay.profiles_groups = checked
    SharedQueryReplay.profile = checked_scalar
    try:
        yield
    finally:
        SharedQueryReplay.profiles_groups = original
        SharedQueryReplay.profile = original_scalar


def history_signature(history):
    fields = ("arm", "pair", "mutable", "seed_assignment", "top_improved", "bottleneck",
              "frontier_assignment", "accepted_steps", "cooling_progress", "feedback_cpu")
    return [{key: row[key] for key in fields if key in row} for row in history]


def result_signature(result, admission):
    return encoded({"selected": result.get("selected"), "movement": result.get("movement"),
                    "candidates": result.get("candidates"), "bets": result.get("bets"),
                    "selection_reason": result.get("selection_reason"),
                    "history": history_signature(result["stats"]["history"]), "admission": admission})


def execute(case, plan, history=None, family_observer=None, certificate_observer=None):
    config = {**case["config"], **dict.fromkeys(CERTIFICATE_FLAGS, True), **PLANS[plan]}
    if history is not None:
        config["replay_arms"] = [row["arm"] for row in history]
        config["replay_feedback_costs"] = [row.get("feedback_cpu", row["cpu"]) for row in history]
        if history and "cooling_progress" in history[0]:
            config["replay_frontier_progress"] = history[0]["cooling_progress"]
    if family_observer is not None:
        config["family_observer"] = family_observer
    if certificate_observer is not None:
        config["certificate_observer"] = certificate_observer
    started, wall_started = process_time(), perf_counter()
    with matching_control(plan == "static"):
        result = search_refined(case["incumbent"], case["forecast"], case["build"], case["selection"],
                                case["capacity"], config, np.random.default_rng(case["seed"]))
    if result["stats"].get("family_execution") != PLANS[plan]["family_execution"]:
        raise RuntimeError("The requested family execution mode was not reported")
    audit_started = process_time()
    admission = {"admitted": False, "reason": "no_selected_action"} if result["selected"] is None else exact_pareto_gate(
        panel_profile(case["audit"], case["incumbent"], case["capacity"]),
        panel_profile(case["audit"], result["selected"], case["capacity"]), metric_names=PROFILE_NAMES)
    audit_cpu = process_time() - audit_started
    timing = {"cpu_seconds": process_time() - started, "wall_seconds": perf_counter() - wall_started,
              "proposal_cpu_seconds": result["stats"]["cpu_seconds"],
              "selection_cpu_seconds": result["stats"]["selection_cpu_seconds"], "audit_cpu_seconds": audit_cpu}
    return result, admission, timing


def compare_case(case, frozen, plans, repetitions, rotation):
    history = frozen["history"]
    snapshots, probes, oracle_cache = [], {}, {}

    def capture(records, details):
        identity = {key: details[key] for key in ("seed_assignment", "pair", "mutable")}
        snapshots.append(encoded({"records": records, "family": identity}))

    counters = {}
    with profile_oracle(counters, oracle_cache):
        eager, admission, _ = execute(case, "eager", history, capture)
    signature = result_signature(eager, admission)
    if signature != frozen["signature"]:
        raise AssertionError("The frozen complete-window eager path was not reproduced")
    if len(snapshots) != eager["stats"]["blocks"]:
        raise AssertionError("The eager observer missed a completed residual family")
    probes["eager"] = {**counters, "family_snapshots": len(snapshots), "exclusions_checked": 0}
    for plan in plans:
        if plan == "eager":
            continue
        position, counters = 0, {}

        def compare(records, details):
            nonlocal position
            identity = {key: details[key] for key in ("seed_assignment", "pair", "mutable")}
            actual = encoded({"records": records, "family": identity})
            if position >= len(snapshots) or actual != snapshots[position]:
                raise AssertionError(f"{plan}: ordered pool or family continuation differs at {position}")
            position += 1

        def certificate(witness):
            verify_witness(witness, case["incumbent"], case["forecast"], case["build"], case["capacity"], counters, oracle_cache)

        with profile_oracle(counters, oracle_cache):
            result, admission, _ = execute(case, plan, history, compare, certificate)
        if position != len(snapshots) or result_signature(result, admission) != signature:
            raise AssertionError(f"{plan}: final physical action, archive, Selection or admission differs")
        for reason in ("forecast", "migration", "score"):
            if counters.get(reason + "_leaves", 0) != result["stats"].get("family_work", {}).get(reason + "_excluded_masks", 0):
                raise AssertionError("The witness observer missed excluded logical leaves")
        probes[plan] = {**counters, "family_snapshots": position,
                        "family_work": result["stats"].get("family_work", {})}
    oracle_cache.clear()  # Reference memoization never enters timed or memory trials.
    trials = []
    for repetition in range(repetitions):
        offset = (rotation + repetition) % len(plans)
        order = plans[offset:] + plans[:offset]
        measurements = {}
        for plan in order:
            gc.collect()
            result, admission, timing = execute(case, plan, history)
            if result_signature(result, admission) != signature:
                raise AssertionError("An uninstrumented fixed-work trial changed the exact result")
            measurements[plan] = {**timing, "family_work": result["stats"].get("family_work", {})}
        trials.append({"order": order, "plans": measurements})
    medians = {plan: {name: float(np.median([trial["plans"][plan][name] for trial in trials]))
                      for name in ("cpu_seconds", "wall_seconds", "proposal_cpu_seconds", "selection_cpu_seconds", "audit_cpu_seconds")}
               for plan in plans} if trials else {}
    return {"dataset": case["dataset"], "origin": case["origin"], "seed": case["seed"],
            "N": len(case["incumbent"]), "H": len(set(case["incumbent"])), "B": len(case["build"]),
            "complete_families": len(snapshots), "all_family_pools_exact": True,
            "all_materialized_profiles_exact": True, "all_exclusion_leaves_checked": True,
            "selected": serial(eager["selected"]), "movement": serial(eager.get("movement")),
            "admission": admission, "verification": probes, "trials": trials, "median_seconds": medians}


def run(args):
    protocol = json.loads(args.protocol.read_text())
    config = json.loads(args.config.read_text())
    config = {**config.get("config", config), "cpu_limit": args.cpu_limit}
    if (not np.isfinite(args.cpu_limit) or args.cpu_limit <= 0 or args.repetitions < 0
            or len(args.plans) != len(set(args.plans))):
        raise ValueError("Positive reference allowance, distinct plans and nonnegative repetitions required")
    origins = protocol["development"]["origins"] if args.phase == "development" else [288, 300, 312]
    plan = {"study": "fixed_complete_family_execution", "phase": args.phase, "origins": origins,
            "cohorts": list(protocol["decision_cohorts"]), "protocol": protocol, "config": config,
            "plans": args.plans, "settings": {name: PLANS[name] for name in args.plans}, "repetitions": args.repetitions,
            
            "inputs": list(protocol["decision_cohorts"]),
            
            "timing": "Serial process CPU includes complete search setup, candidate generation, matching, profiles, pool retention, Selection and canonical Audit. Immutable input panels, Arrow loading, forecast/FFD construction, freezing, verification, garbage collection and serialization are outside timing.",
            "correctness": "Exact materialized profile bytes, ordered pool and family continuation after every completed residual family, original rich physical movement, selected action, bets and historical admission. Every excluded active leaf is independently enumerated and canonically checked. Independent canonical answers are memoized across untimed plans within each immutable case and discarded before timing. Physical evaluation counters may differ by certified exclusion.",
            "path": "Complete eager paths over all 121 Build windows; replay fixes operation arms, frontier cooling progress and feedback costs."}
    path = args.output / "study_plan.json"
    if path.exists() and json.loads(path.read_text()) != serial(plan):
        raise ValueError("Use a new directory for a different fixed-work study")
    save(path, plan)
    if args.plan_only:
        print(json.dumps({"planned_cases": len(origins) * len(plan["cohorts"]), "plans": args.plans}))
        return
    frozen_path = args.output / "frozen_paths.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else []
    frozen_by_case = {(row["dataset"], row["origin"]): row for row in frozen}
    results = []
    for index, (name, metadata) in enumerate(protocol["decision_cohorts"].items()):
        data = load_dataset(name, args.data_root)
        for origin in origins:
            forecast = data.forecast(origin)
            case = {"dataset": name, "origin": origin, "seed": 186000 + 10000 * index + origin,
                    "config": config, "forecast": forecast, "incumbent": ffd(forecast, data.capacity),
                    "capacity": data.capacity, **panels_for(data, origin, protocol, ("build", "selection", "audit"))}
            identity = (name, origin)
            if identity not in frozen_by_case:
                result, admission, _ = execute(case, "eager")
                item = {"dataset": name, "origin": origin, "seed": case["seed"],
                        "history": result["stats"]["history"], "signature": result_signature(result, admission)}
                frozen.append(item)
                frozen_by_case[identity] = item
                save(frozen_path, frozen)
            result = compare_case(case, frozen_by_case[identity], args.plans, args.repetitions, len(results))
            results.append(result)
            save(args.output / "records.json", results)
            print(json.dumps({"dataset": name, "origin": origin, "complete_families": result["complete_families"],
                              "exact": True, "median_seconds": result["median_seconds"]}), flush=True)
    totals = {name: sum(row["median_seconds"][name]["cpu_seconds"] for row in results) for name in args.plans} if args.repetitions else {}
    save(args.output / "summary.json", {"complete": True, "cases": len(results), "all_exact": True,
                                       "total_median_cpu_seconds": totals,
                                       "speedup_over_eager": {name: totals["eager"] / value for name, value in totals.items()}
                                       if "eager" in totals else {}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "evaluation"), required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/release/family_protocol.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/release/family_frozen.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/cloudcons")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plans", choices=PLANS, nargs="+", default=list(PLANS))
    parser.add_argument("--cpu-limit", type=float, default=2.)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--plan-only", action="store_true")
    run(parser.parse_args())
