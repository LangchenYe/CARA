"""Exact family enumeration and deterministic search-replay helpers."""
import numpy as np
from cara import refinement
from cara.data import canonical_assignment
from cara.refinement import migration_profile


def _run_path(case, plan):
    config = {**case["config"], "reuse_views": plan != "full", "projected_views": plan == "project",
              "shared_queries": plan.startswith("shared"), "batch_queries": plan == "shared_batch",
              "replay_arms": [row["arm"] for row in case["history"]],
              "replay_feedback_costs": [row.get("feedback_cpu", row.get("cpu", 0.)) for row in case["history"]]}
    if case["history"] and "cooling_progress" in case["history"][0]:
        config["replay_frontier_progress"] = case["history"][0]["cooling_progress"]
    return refinement.search_refined(case["incumbent"], case["forecast"], case["build"],
        case["selection"], case["capacity"], config, np.random.default_rng(case["seed"]))


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
