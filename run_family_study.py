"""Complete-window CARA comparisons on frozen development and evaluation grids.

Evaluation executes each unique cohort/origin/search-stream once. Its records
serve both the original 300-case view and the nonoverlapping daily view. Run
Scenario-MIP separately with one worker; combine complete studies in analysis.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from benchmark_native_intervals import predictive_metrics
from cara.baseline_anneal import search_anneal
from cara.baseline_solver import scenario_mip
from cara.certification import exact_pareto_gate
from cara.data import CAPACITIES, canonical_assignment, load_dataset, native_metrics
from cara.refinement import PROFILE_NAMES, migration_profile, panel_profile, search_refined, search_swap
from cara.repair import feasible, ffd


ROOT = Path(__file__).resolve().parent
BASE_METHODS = ("CARA", "Component-CARA", "Compact-CARA", "Eager-CARA", "Joint-SA", "Anneal", "Risk-SA", "Broad", "Swap",
                "Scenario-MIP", "No-residual", "No-frontier", "Best-seed", "Raw-order")
MAIN_METHODS = (BASE_METHODS[0], "Family-CARA", *BASE_METHODS[1:])
CERTIFICATE_ABLATIONS = ("No-forecast-certificate", "No-migration-certificate", "No-score-certificate")
HEURISTICS = tuple(name for name in MAIN_METHODS if name != "Scenario-MIP")
CERTIFICATE_FLAGS = ("family_certificate_forecast", "family_certificate_migration", "family_certificate_score")
NATIVE_METRICS = ("hosts", "Util", "VR", "VS", "PAR", "raw", "PICP", "MPIW", "WinklerScore")
DATA = {}


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(item) for item in value]
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def case_key(case):
    return case["dataset"], case["origin"], case["seed_index"]


def case_grid(protocol, phase, streams=None):
    streams = list(streams if streams is not None else protocol["development"]["default_streams"]
                   if phase == "development" else protocol["primary"]["streams"])
    if len(streams) != len(set(streams)) or any(seed < 0 for seed in streams):
        raise ValueError("Search streams must be distinct nonnegative integers")
    if phase == "evaluation" and streams != protocol["primary"]["streams"]:
        raise ValueError("Evaluation requires all three prescribed search streams")
    primary = set(range(protocol["primary"]["first_origin"], protocol["primary"]["last_origin"] + 1,
                        protocol["primary"]["stride"]))
    cases = []
    for stream in streams:
        for index, (dataset, metadata) in enumerate(protocol["decision_cohorts"].items()):
            daily = set(metadata["longitudinal_origins"])
            origins = protocol["development"]["origins"] if phase == "development" else sorted(primary | daily)
            for origin in origins:
                if origin < 192 or origin + protocol["target_hours"] > metadata["minimum_hours"]:
                    raise ValueError("A planned target is not complete for its full cohort")
                views = ["development"] if phase == "development" else (
                    (["primary"] if origin in primary else []) + (["longitudinal"] if origin in daily else []))
                cases.append({"dataset": dataset, "origin": origin, "seed_index": stream,
                              "seed": 186000 + 100000 * stream + 10000 * index + origin, "views": views})
    if phase == "evaluation":
        if (len(cases) != protocol["evaluation_union"]["unique_cases"]
                or sum("primary" in row["views"] for row in cases) != protocol["primary"]["cases"]
                or sum("longitudinal" in row["views"] for row in cases) != protocol["longitudinal"]["cases"]):
            raise ValueError("The evaluation union does not cover both complete views")
    return cases


def method_settings(method, config):
    if method not in (*MAIN_METHODS, *CERTIFICATE_ABLATIONS):
        raise ValueError(f"Unknown comparison method: {method}")
    settings = {**config, "family_execution": "eager", **dict.fromkeys(CERTIFICATE_FLAGS, True)}
    if method == "Compact-CARA":
        settings["family_execution"] = "compact"
        settings["family_factor_matching"] = False
        settings["family_matching_backend"] = "dense"
        settings["family_defer_physical"] = False
    elif method in ("CARA", "Family-CARA", "Component-CARA", "No-residual", "No-frontier", "Best-seed", "Raw-order", *CERTIFICATE_ABLATIONS):
        settings["family_execution"] = "certified"
    if method == "Component-CARA":
        settings["family_matching_backend"] = "factorized"
    if method in CERTIFICATE_ABLATIONS:
        settings[CERTIFICATE_FLAGS[CERTIFICATE_ABLATIONS.index(method)]] = False
    if method == "Broad":
        settings.update(mode="broad", frontier_seed=False)
    elif method == "No-residual":
        settings["frontier_only"] = True
    elif method == "No-frontier":
        settings["frontier_seed"] = False
    elif method == "Best-seed":
        settings["seed_schedule"] = "best"
    elif method == "Raw-order":
        settings["score_mode"] = "raw"
    elif method == "Risk-SA":
        settings["anneal_objective"] = "risk"
    elif method == "Joint-SA":
        settings["anneal_objective"] = "joint"
    if method == "Family-CARA":
        settings.update(family_seed_compilation=False, family_bound_views=False,
                        family_metric_cascade=False, family_subset_views=False,
                        family_structural_bound_gate=False, family_compiled_bounds=False,
                        family_staged_acquisition=False)
    return settings


def engine_for(method):
    return scenario_mip if method == "Scenario-MIP" else search_swap if method == "Swap" else (
        search_anneal if method in ("Anneal", "Risk-SA", "Joint-SA") else search_refined)


def panels_for(data, origin, protocol, roles=("build", "selection")):
    panels = {role: data.window_panel(origin, **protocol["historical_roles"][role],
                                     horizon=protocol["window_hours"], stride=protocol["window_stride"])
              for role in roles}
    if any(len(value) != protocol["expected_rows"][role] for role, value in panels.items()):
        raise ValueError("The complete historical window table was not constructed")
    return panels


def prepare_and_search(data, origin, incumbent, forecast, method, config, seed, protocol, budget):
    preparation_started = process_time()
    settings = method_settings(method, config)
    panels = panels_for(data, origin, protocol)
    rng = np.random.default_rng(seed)
    preparation_cpu = process_time() - preparation_started
    remaining = budget - preparation_cpu
    settings["cpu_limit"] = max(0., remaining)
    call_started = process_time()
    if remaining <= 0:
        result = {"selected": None, "movement": None, "candidates": [], "selection_reason": "preparation_exhausted",
                  "stats": {"cpu_seconds": 0., "selection_cpu_seconds": 0., "evaluations": 0, "blocks": 0,
                            "history": [], "family_execution": settings["family_execution"], "allowance_seconds": 0.}}
    else:
        result = engine_for(method)(incumbent, forecast, panels["build"], panels["selection"],
                                    data.capacity, settings, rng)
        if (engine_for(method) is search_refined
                and result["stats"].get("family_execution") != settings["family_execution"]):
            raise RuntimeError("The search did not report the requested family execution mode")
    call_cpu = process_time() - call_started
    result["preparation_cpu_seconds"] = preparation_cpu
    result["remaining_proposal_allowance_seconds"] = max(0., remaining)
    result["search_call_cpu_seconds"] = call_cpu
    return result


def validate_selected(incumbent, selected, forecast, capacity, config, movement=None):
    if not feasible(selected, forecast, capacity, len(set(incumbent))):
        raise AssertionError("Selected placement violates common forecast feasibility")
    movement = migration_profile(incumbent, selected, config.get("migration_weights")) if movement is None else movement
    aligned = np.asarray(movement["aligned_assignment"], dtype=int)
    moved = np.flatnonzero(np.asarray(incumbent) != aligned).tolist()
    if (canonical_assignment(aligned) != canonical_assignment(selected) or moved != list(movement["moved_vms"])
            or len(moved) != movement["count"] or movement["fraction"] != len(moved) / len(incumbent)
            or config.get("migration_weights") is None and movement["cost"] != float(len(moved))):
        raise AssertionError("Selected physical placement disagrees with its movement summary")
    limit = int(config.get("max_migrations", np.floor(len(incumbent) * config.get("migration_fraction", .2))))
    if movement["count"] > limit or movement["cost"] > config.get("migration_cost_limit", np.inf):
        raise AssertionError("Selected placement exceeds the common movement constraint")
    return movement


def decide(data, origin, incumbent, forecast, method, config, seed, protocol, budget):
    """One charged proposal, held-out Selection, physical action and Audit."""
    started, wall_started = process_time(), perf_counter()
    proposal = prepare_and_search(data, origin, incumbent, forecast, method, config, seed, protocol, budget)
    selected = proposal["selected"]
    selected = None if selected is None else canonical_assignment(selected)
    materialization_started = process_time()
    selected_movement = None if selected is None else validate_selected(
        incumbent, selected, forecast, data.capacity, config, proposal.get("movement"))
    materialization_cpu = process_time() - materialization_started
    audit_started = process_time()
    if selected is None:
        admission = {"admitted": False, "rows_read": 0, "reason": "no_selected_action"}
    else:
        audit = panels_for(data, origin, protocol, ("audit",))["audit"]
        admission = exact_pareto_gate(panel_profile(audit, incumbent, data.capacity),
                                     panel_profile(audit, selected, data.capacity), metric_names=PROFILE_NAMES)
    audit_cpu = process_time() - audit_started
    terminal = selected if admission["admitted"] else incumbent
    stationary = {"count": 0, "fraction": 0., "cost": 0., "aligned_assignment": incumbent, "moved_vms": []}
    movement = selected_movement if admission["admitted"] else stationary
    decision_cpu, decision_wall = process_time() - started, perf_counter() - wall_started
    stats = dict(proposal["stats"])
    stats["completed_operations"] = len(stats.pop("history", []))
    prep, proposal_cpu = proposal["preparation_cpu_seconds"], stats["cpu_seconds"]
    return {"selected_assignment": selected, "terminal_assignment": terminal,
            "movement": movement, "selected_movement": selected_movement, "admission": admission,
            "selection_reason": proposal.get("selection_reason"), "search": stats,
            "retained_candidates": len(proposal.get("candidates", [])),
            "timing": {"preparation_cpu_seconds": prep,
                       "remaining_proposal_allowance_seconds": proposal["remaining_proposal_allowance_seconds"],
                       "proposal_cpu_seconds": proposal_cpu, "proposal_with_preparation_cpu_seconds": prep + proposal_cpu,
                       "proposal_budget_overshoot_cpu_seconds": max(0., prep + proposal_cpu - budget),
                       "selection_cpu_seconds": stats["selection_cpu_seconds"],
                       "search_call_cpu_seconds": proposal["search_call_cpu_seconds"],
                       "post_search_movement_cpu_seconds": materialization_cpu, "audit_cpu_seconds": audit_cpu,
                       "total_decision_cpu_seconds": decision_cpu, "total_decision_wall_seconds": decision_wall}}


def run_case(job):
    index, case, plan, _context, output = job
    path = Path(output) / "records" / f'{case["dataset"]}_{case["origin"]}_{case["seed_index"]}.json'
    if path.exists():
        cached = json.loads(path.read_text())
        if (cached.get("config") != plan["config"] or any(cached[name] != value for name, value in case.items())
                or set(cached["policies"]) != {"FFD", *plan["methods"]}):
            raise ValueError(f"Cached case belongs to a different complete study: {path}")
        return cached
    dataset = case["dataset"]
    loading_started, loading_wall_started = process_time(), perf_counter()
    cache_hit = dataset in DATA
    if dataset not in DATA:
        DATA[dataset] = load_dataset(dataset, plan["data_root"])
    data = DATA[dataset]
    loading_cpu, loading_wall = process_time() - loading_started, perf_counter() - loading_wall_started
    metadata = plan["protocol"]["decision_cohorts"][dataset]
    if data.n_vms != metadata["vms"] or data.min_length != metadata["minimum_hours"] or data.capacity != metadata["capacity"]:
        raise ValueError("Input cohort differs from the frozen protocol")
    common_started = process_time()
    origin, config, protocol = case["origin"], plan["config"], plan["protocol"]
    forecast = data.forecast(origin, horizon=protocol["target_hours"])
    forecast.setflags(write=False)
    incumbent = ffd(forecast, data.capacity)
    stationary = {"count": 0, "fraction": 0., "cost": 0., "aligned_assignment": incumbent, "moved_vms": []}
    common_cpu = process_time() - common_started
    methods = plan["methods"]
    shift = index % len(methods)
    order = methods[shift:] + methods[:shift]
    record = {**case, "n_vms": data.n_vms, "capacity": data.capacity,
              "config": config, "incumbent_assignment": incumbent, "method_order": order,
              "trace_loading_cpu_seconds": loading_cpu, "trace_loading_wall_seconds": loading_wall,
              "trace_cache_hit": cache_hit,
              "common_forecast_incumbent_cpu_seconds": common_cpu, "panel_rows": protocol["expected_rows"],
              "policies": {}}
    for method in order:
        gc.collect()
        record["policies"][method] = decide(data, origin, incumbent, forecast, method, config, case["seed"], protocol,
                                            plan["proposal_allowance_seconds"])
    # Every method's selected and released placements are fixed before targets are sliced.
    evaluation_started = process_time()
    demand = data.panel(origin, protocol["target_hours"])
    scales = data.panel(origin - 24, 24).std(axis=1, ddof=1)
    record["policies"]["FFD"] = {"terminal_assignment": incumbent, "movement": stationary,
                                   "admission": {"admitted": False, "reason": "packing_reference"}}
    metric_cache = {}
    for policy in record["policies"].values():
        placement = tuple(policy["terminal_assignment"])
        if placement not in metric_cache:
            metric_cache[placement] = {**native_metrics(demand, placement, data.capacity),
                                       **predictive_metrics(demand, placement, forecast, scales)}
        policy["metrics"] = metric_cache[placement]
    record["evaluation_cpu_seconds"] = process_time() - evaluation_started
    save(path, record)
    return record


def make_plan(args):
    protocol = json.loads(args.protocol.read_text())
    raw = json.loads(args.config.read_text())
    config = raw.get("config", raw)
    if config.get("forecast_mode", "mean") != "mean":
        raise ValueError("This protocol requires the common trailing-mean forecast")
    methods = list(args.methods)
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("Methods must be nonempty and distinct")
    if "Scenario-MIP" in methods and (methods != ["Scenario-MIP"] or args.workers != 1):
        raise ValueError("Run Scenario-MIP separately with one worker")
    if args.phase == "evaluation" and set(methods).intersection(CERTIFICATE_ABLATIONS):
        raise ValueError("Certificate-specific ablations are initially restricted to the development grid")
    budget = protocol["proposal_allowance_seconds"] if args.cpu_limit is None else args.cpu_limit
    if not np.isfinite(budget) or budget <= 0 or args.workers < 1:
        raise ValueError("Positive proposal CPU and worker count required")
    if args.phase == "evaluation" and budget != protocol["proposal_allowance_seconds"]:
        raise ValueError("Evaluation requires the frozen proposal allowance")
    config = {**config, "cpu_limit": float(budget)}
    return {"study": "complete_window_family_execution", "phase": args.phase, "protocol": protocol,
            "cases": case_grid(protocol, args.phase, args.seeds), "config": config,
            "methods": methods, "method_settings": {method: method_settings(method, config) for method in methods},
            "proposal_allowance_seconds": float(budget), "workers": args.workers,
            "data_root": str(args.data_root),
            "inputs": list(protocol["decision_cohorts"]),
            
            
            "rotation": "Each case rotates the declared method order by its index; the separate solver study is not in that joint rotation.",
            "timing": "Actual process CPU. Preparation and proposal share the allowance; Selection, physical-action validation and Audit are included in complete decision cost. Common data loading, forecast/FFD construction, outcome evaluation and serialization are excluded. Solver CPU and wall observations retain their separate meaning."}


def run(args):
    plan = make_plan(args)
    plan_path = args.output / "study_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != serial(plan):
        raise ValueError("Use a new output directory for a changed study plan")
    save(plan_path, plan)
    if args.plan_only:
        print(json.dumps({"planned_cases": len(plan["cases"]), "methods": plan["methods"], "output": str(args.output)}))
        return
    jobs = [(index, case, plan, None, str(args.output)) for index, case in enumerate(plan["cases"])]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 12 == 0 or len(records) == len(jobs):
                print(json.dumps({"completed": len(records), "planned": len(jobs)}), flush=True)
    positions = {case_key(case): index for index, case in enumerate(plan["cases"])}
    records.sort(key=lambda row: positions[case_key(row)])
    save(args.output / "records.json", records)
    save(args.output / "completion.json", {"complete": True, "cases": len(records), "methods": plan["methods"],
                                           "view_cases": {view: sum(view in row["views"] for row in records)
                                                          for view in sorted({view for row in records for view in row["views"]})}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "evaluation"), required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/release/family_protocol.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/release/family_frozen.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/cloudcons")
    parser.add_argument("--methods", nargs="+", choices=(*MAIN_METHODS, *CERTIFICATE_ABLATIONS), default=list(HEURISTICS))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--cpu-limit", type=float)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    run(parser.parse_args())
