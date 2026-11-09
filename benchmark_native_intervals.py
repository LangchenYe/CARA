"""Complete native prediction-interval diagnostics on frozen terminal policies.

The forecast is common to every placement: its median is the existing packing
forecast and its scale is estimated from the preceding 24 hours only. The three
metrics and host-level Monte Carlo protocol follow CloudCons. This evaluates
placement-dependent uncertainty diagnostics; it does not reproduce CloudCons'
trained forecasting-model leaderboard or modify any placement decision.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtri

from benchmark_query_engine import _record_stream
from cara.data import CAPACITIES, canonical_assignment, host_loads, host_memberships, load_dataset

NAMES = ("PICP", "MPIW", "WinklerScore")


def predictive_metrics(demand, assignment, medians, scales, *, simulations=100,
                       confidence=.95, seed=42):
    """CloudCons' active-host average, strict interval formula and fixed RNG.

    Every host draws independent VM quantile levels. One VM's level is shared
    across its complete trajectory, as in the upstream quantile-forecast path.
    Canonical ascending VM order specifies resident summation reproducibly.
    """
    demand, medians, scales = map(np.asarray, (demand, medians, scales))
    if (demand.ndim != 2 or medians.shape != demand.shape
            or scales.shape != (len(demand),) or not 0 < confidence < 1
            or int(simulations) != simulations or simulations < 1
            or not all(np.isfinite(x).all() for x in (demand, medians, scales))
            or any(np.any(x < 0) for x in (demand, medians, scales))):
        raise ValueError("Finite nonnegative demand/median/scales and a valid simulation design required")
    groups = host_memberships(assignment)
    truth = host_loads(demand, assignment)
    values = []
    alpha = 1 - confidence
    active_index = 0
    for residents, actual in zip(groups, truth):
        if not np.any(actual != 0):
            continue
        rng = np.random.default_rng(seed + active_index)
        active_index += 1
        ids = np.asarray(residents)
        levels = rng.uniform(0., 1., (int(simulations), len(ids)))
        shifts = np.zeros_like(levels)
        # A zero-scale point distribution remains defined at the RNG's
        # possible zero endpoint; positive scales map that endpoint to zero
        # after the declared nonnegative clipping.
        np.multiply(ndtri(levels), scales[ids][None, :], out=shifts,
                    where=scales[ids][None, :] > 0)
        simulated = np.maximum(medians[ids][None, :, :] + shifts[:, :, None], 0.).sum(axis=1)
        lower, upper = np.percentile(simulated, [alpha * 50, (1 - alpha / 2) * 100], axis=0)
        width = upper - lower
        coverage = np.mean((actual >= lower) & (actual <= upper))
        score = width + (2 / alpha) * np.maximum(0., lower - actual)
        score += (2 / alpha) * np.maximum(0., actual - upper)
        values.append((coverage, np.mean(width), np.mean(score)))
    means = np.mean(values, axis=0) if values else np.zeros(3)
    return {name: float(value) for name, value in zip(NAMES, means)}


def summarize(records, methods):
    output = {}
    for cohort in ("all", *CAPACITIES):
        selected = [row for row in records if cohort == "all" or row["dataset"] == cohort]
        output[cohort] = {method: {
            "decisions": len(selected),
            **{name: float(np.mean([row["policies"][method][name] for row in selected]))
               for name in NAMES}}
            for method in methods}
    return output


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(source, output):
    source, output = Path(source), Path(output)
    plan = json.loads((source / "study_plan.json").read_text())
    if (plan["phase"] != "official" or plan["cohorts"] != list(CAPACITIES)
            or plan["origins"] != list(range(288, 313)) or plan["seeds"] != [0, 1, 2]
            or len(plan["configurations"]) != 1 or len(plan["methods"]) != 13
            or len(set(plan["methods"])) != 13):
        raise ValueError("The complete original main-study comparison is required")
    config_id, config = plan["configurations"][0]
    if config.get("forecast_mode", "mean") != "mean":
        raise ValueError("This diagnostic is prespecified for the main trailing-mean forecast")
    methods = ["FFD", "BFD", *plan["methods"]]
    expected = {(cohort, origin, seed) for cohort in plan["cohorts"]
                for origin in plan["origins"] for seed in plan["seeds"]}
    design = {"source": str(source), "cohorts": plan["cohorts"], "origins": plan["origins"],
        "search_streams": plan["seeds"], "methods": methods, "simulations_per_host": 100,
        "confidence_level": .95, "host_seed": "42 + active-host index in canonical order",
        "forecast": "Q_i,t(u)=max(0,mu_i,t+sigma_i*Phi^{-1}(u)); mu is the unchanged trailing24-hour mean packing forecast, sigma is the preceding24-hour sample standard deviation (ddof=1). Independent VM quantile levels are shared over each VM's complete forecast trajectory.",
        "metrics": {"PICP": "Active-host average of mean_t 1[lower_t<=truth_t<=upper_t]",
                    "MPIW": "Active-host average of mean_t(upper_t-lower_t)",
                    "WinklerScore": "Active-host average of mean_t(width_t+2/alpha*(lower_t-truth_t)_+ +2/alpha*(truth_t-upper_t)_+), alpha=0.05"},
        "upstream_protocol": "CloudCons simulation/assignment_simpy.py MonteCarloSimulation, quantile-forecast branch; 100 draws, default NumPy linear percentiles at 2.5 and97.5, 95% central interval.",
        "scope": "Complete frozen terminal placements; shared history-only probabilistic forecast. No proposal search, tuning or admission change. These metrics diagnose placement-dependent predictive intervals and do not reproduce trained-model forecasting leaderboards.",
        "source_study_plan": plan}
    save(output / "study_plan.json", design)
    datasets, contexts, cache, records, identities = {}, {}, {}, [], set()
    for row in _record_stream(source):
        identity = row["dataset"], row["origin"], row["seed_index"]
        if identity not in expected or identity in identities:
            raise AssertionError("Unplanned or duplicated source scenario")
        if (row["config_id"] != config_id or row["config"] != config
                or set(row["policies"]) != set(methods)):
            raise AssertionError("Source comparison differs from its complete frozen plan")
        identities.add(identity)
        cohort, origin, _ = identity
        if cohort not in datasets:
            datasets[cohort] = load_dataset(cohort)
        if (cohort, origin) not in contexts:
            data = datasets[cohort]
            history = data.panel(origin - 24, 24)
            medians = data.forecast(origin)
            scales = history.std(axis=1, ddof=1)
            contexts[cohort, origin] = data.panel(origin), medians, scales
        demand, medians, scales = contexts[cohort, origin]
        values = {}
        for method in methods:
            assignment = canonical_assignment(row["policies"][method]["terminal_assignment"])
            key = cohort, origin, assignment
            if key not in cache:
                cache[key] = predictive_metrics(demand, assignment, medians, scales)
            values[method] = cache[key]
        records.append({"dataset": cohort, "origin": origin,
                        "seed_index": row["seed_index"], "policies": values})
    if identities != expected:
        raise AssertionError("Source is missing designated main-study scenarios")
    records.sort(key=lambda row: (row["dataset"], row["origin"], row["seed_index"]))
    summary = {"scenarios": len(records), "policy_outcomes": len(records) * len(methods),
               "unique_placements_evaluated": len(cache), "complete_design": True,
               "metrics": summarize(records, methods)}
    save(output / "records.json", records)
    save(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "metrics"}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/icde2027_shared_main"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
