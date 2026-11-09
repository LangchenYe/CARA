"""Metropolis move/swap annealing under the shared placement constraints.

This is a standard optimization baseline, not a reproduction of a production
cluster manager. Temperature is a fraction of incumbent Build raw excess and
decreases geometrically to one thousandth of its initial value.
"""
from time import process_time
import numpy as np

from .admission import fit_bets
from .data import burdens, canonical_assignment
from .repair import feasible
from .refinement import (PROFILE_NAMES, _constraints, _coordinates, _profile_replay,
                         _relative_gains, _replay, _select, _ordered_records, _coverage_archive, migration_profile,
                         panel_profile)


def search_anneal(inc, forecast, build, selection, capacity, config, rng, *, on_candidate=None,
                  select_action=True, query_context=None):
    started = process_time()
    inc = canonical_assignment(inc)
    hosts = len(set(inc))
    budget = float(config.get("cpu_limit", 2.))
    temperature_fraction = float(config.get("anneal_temperature_fraction", .05))
    top = int(config.get("top_p", 16))
    maximum = config.get("max_attempts")
    cooling = config.get("replay_cooling")
    if cooling is not None:
        if maximum is None or len(cooling) != maximum or any(not np.isfinite(v) or not 0 <= v <= 1 for v in cooling):
            raise ValueError("Recorded cooling must cover every declared attempt")
    if (budget <= 0 or temperature_fraction <= 0 or not np.isfinite(temperature_fraction)
            or top < 1 or (maximum is not None and (int(maximum) != maximum or maximum < 1))):
        raise ValueError("Invalid annealing time, temperature, pool or attempt limit")
    if not feasible(inc, forecast, capacity, hosts):
        raise ValueError("The incumbent must satisfy the shared forecast constraint")
    limit, weights, cost_limit = _constraints(inc, config)
    objectives = tuple(config.get("objective_names", PROFILE_NAMES))
    _coordinates(objectives)
    reference = panel_profile(build, inc, capacity)
    objective = config.get("anneal_objective", "raw")
    reference_means = reference.mean(axis=0)
    def energy(profile):
        means = profile.mean(axis=0)
        if objective == "raw":
            return float(means[3])
        if objective == "risk":
            return float(means[1] + .1 * means[3] / max(reference_means[3], 1e-8))
        if objective == "joint":
            indices = [2, 3, 4, 5]
            return float(np.mean(means[indices] / np.maximum(reference_means[indices], 1e-8)))
        raise ValueError("Unknown annealing objective")
    current, loss = inc, energy(reference)
    initial_temperature = max(loss, 1e-8) * temperature_fraction
    cache = _replay(build, current, capacity, config.get("reuse_views", True),
                    config.get("projected_views", False))
    forecast_views = None
    shared_cache = None
    if config.get("shared_queries", False):
        from .query_engine import SharedQueryReplay, ForecastViews
        shared_cache, forecast_views = (query_context if query_context is not None and query_context[0] is not None else
            (SharedQueryReplay.from_config(build, inc, capacity, config),
             ForecastViews.from_config(forecast, capacity, inc, config)))
        cache = shared_cache.fork(inc)
    pool, memo = {}, {inc: (reference, migration_profile(inc, inc, weights))}
    stats = {"evaluations": 0, "attempts": 0, "accepted_steps": 0, "uphill_steps": 0,
             "migration_rejections": 0, "forecast_rejections": 0, "blocks": 0,
             "migration_limit": limit, "history": [], "cache_hits": 0,
             "projection_hits": 0, "changed_values": 0, "full_values": 0,
             "initial_temperature": initial_temperature,
             "temperature_fraction": temperature_fraction}
    if config.get("record_cooling", False):
        stats["cooling_progress"] = []

    def account_cache():
        for key, attr in (("cache_hits", "hits"), ("projection_hits", "projection_hits"),
                          ("changed_values", "changed_values"), ("full_values", "full_values")):
            stats[key] += getattr(cache, attr, 0)

    while hosts > 1 and (stats["attempts"] < maximum if maximum is not None
                         else process_time() - started < budget):
        progress = (cooling[stats["attempts"]] if cooling is not None else
                    stats["attempts"] / maximum if maximum is not None else
                    min(1., (process_time() - started) / budget))
        if "cooling_progress" in stats:
            stats["cooling_progress"].append(progress)
        temperature = initial_temperature * .001 ** progress
        stats["attempts"] += 1
        proposal = np.asarray(current).copy()
        vm = int(rng.integers(len(inc)))
        if rng.random() < .5:
            target = int(rng.integers(hosts - 1))
            proposal[vm] = target + (target >= proposal[vm])
        else:
            other = int(rng.integers(len(inc)))
            proposal[vm], proposal[other] = proposal[other], proposal[vm]
        candidate = canonical_assignment(proposal)
        if candidate == current or len(set(candidate)) != hosts:
            continue
        if candidate not in memo:
            movement = migration_profile(inc, candidate, weights)
            if movement["count"] > limit or movement["cost"] > cost_limit:
                stats["migration_rejections"] += 1
                continue
            if not (forecast_views.feasible(candidate, hosts) if forecast_views is not None else
                    feasible(candidate, forecast, capacity, hosts)):
                stats["forecast_rejections"] += 1
                continue
            profile, loads = _profile_replay(cache, candidate, capacity)
            memo[candidate] = (profile, movement)
            stats["evaluations"] += 1
            if on_candidate is not None:
                on_candidate(candidate, cache, profile, loads, movement)
            if select_action or on_candidate is None:
                gains = _relative_gains(reference, profile, objectives)
                key = tuple(sorted(int(np.floor(value * (1 << 40))) for value in gains))
                key += (-movement["count"], -movement["cost"])
                pool[candidate] = {"assignment": candidate, "movement": movement, "key": key,
                                   "build_means": profile.mean(axis=0).tolist(),
                                   "relative_gains": gains.tolist()}
                retained = (_coverage_archive(list(pool.values()), top) if config.get("archive_mode") == "coverage"
                            else _ordered_records(pool.values())[:top])
                pool = {record["assignment"]: record for record in retained}
        profile, _ = memo[candidate]
        candidate_loss = energy(profile)
        delta = candidate_loss - loss
        if delta <= 0 or rng.random() < np.exp(-delta / temperature):
            stats["accepted_steps"] += 1
            stats["uphill_steps"] += int(delta > 0)
            account_cache()
            current, loss = candidate, candidate_loss
            cache = (shared_cache.fork(current) if shared_cache is not None else
                     _replay(build, current, capacity, config.get("reuse_views", True),
                             config.get("projected_views", False)))
    account_cache()
    proposal_cpu = process_time() - started
    records = _ordered_records(pool.values())
    selected, selection_stats = (_select(records, inc, selection, capacity, config) if select_action else
                                (None, {"selection_candidates": 0, "selection_guard_passed": 0}))
    bets = None if selected is None else fit_bets(burdens(build, inc, capacity),
                                               burdens(build, selected["assignment"], capacity))
    stats.update(selection_stats)
    stats.update({"cpu_seconds": proposal_cpu, "allowance_seconds": budget,
                  "selection_cpu_seconds": process_time() - started - proposal_cpu})
    return {"selected": None if selected is None else selected["assignment"], "bets": bets,
            "candidates": records, "movement": None if selected is None else selected["movement"],
            "frontier_assignment": current, "stats": stats}
