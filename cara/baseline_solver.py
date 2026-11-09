"""Scenario mixed-integer placement with a global migration constraint.

This is an independent sample-average optimization baseline, not a reproduction
of a production cluster manager. It minimizes uncapped excess on Build rows.
Every returned placement is checked using the same canonical workload kernel.
"""
from time import perf_counter, process_time
import warnings
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from .data import canonical_assignment
from .repair import feasible
from .refinement import panel_profile, PROFILE_NAMES


def scenario_mip(inc, forecast, build, selection, capacity, config, rng=None):
    if config.get('migration_weights') is not None or np.isfinite(config.get('migration_cost_limit', np.inf)):
        raise ValueError('Scenario-MIP supports a moved-VM count limit only; weighted movement costs are not implemented')
    started, wall_started = process_time(), perf_counter()
    inc = canonical_assignment(inc)
    n, h = len(inc), len(set(inc))
    limit = int(config.get('max_migrations', np.floor(n * config.get('migration_fraction', .2))))
    budget = float(config.get('cpu_limit', 2.))
    rows = int(config.get('mip_rows', 4))
    if limit < 0 or budget <= 0 or rows < 1:
        raise ValueError('Invalid migration, time, or scenario budget')
    idx = np.unique(np.linspace(0, len(build) - 1, min(rows, len(build)), dtype=int))
    demand = np.asarray(build, dtype=float)[idx].transpose(0, 2, 1).reshape(-1, n) / capacity
    scenarios = len(demand)
    nx, nq = n * h, scenarios * h
    variables = nx + nq
    objective = np.r_[np.zeros(nx), np.full(nq, 1. / max(1, nq))]
    integrality = np.r_[np.ones(nx, dtype=np.uint8), np.zeros(nq, dtype=np.uint8)]
    lower = np.zeros(variables)
    upper = np.r_[np.ones(nx), np.full(nq, np.inf)]
    rr, cc, vv, lbs, ubs = [], [], [], [], []

    def constraint(columns, values, lo=-np.inf, hi=np.inf):
        row = len(lbs)
        columns = np.asarray(columns, dtype=np.int64)
        rr.extend([row] * len(columns)); cc.extend(columns.tolist())
        vv.extend(np.asarray(values, dtype=float).tolist())
        lbs.append(lo); ubs.append(hi)

    for i in range(n):
        constraint(np.arange(i * h, (i + 1) * h), np.ones(h), 1., 1.)
    for host in range(h):
        columns = np.arange(n) * h + host
        constraint(columns, np.ones(n), 1., np.inf)
        # Identical forecast columns need only one copy of the constraint.
        for values in np.unique(np.asarray(forecast).T, axis=0):
            constraint(columns, values / capacity, hi=1. + 1e-12 / capacity)
    constraint(np.arange(n) * h + np.asarray(inc), np.ones(n), max(0, n - limit), np.inf)
    for row, values in enumerate(demand):
        for host in range(h):
            columns = np.r_[np.arange(n) * h + host, nx + row * h + host]
            constraint(columns, np.r_[values, -1.], hi=1.)
    matrix = coo_array((np.asarray(vv), (np.asarray(rr, dtype=np.int32), np.asarray(cc, dtype=np.int32))), shape=(len(lbs), variables)).tocsc()
    construction_cpu = process_time() - started
    construction_wall = perf_counter() - wall_started
    remaining = max(0., budget - construction_cpu)
    # HiGHS uses wall time; one thread makes it comparable with process-CPU
    # budgets. Setup and actual CPU/wall cost are reported, not discarded.
    result = None
    solver_started, solver_wall_started = process_time(), perf_counter()
    if remaining > 0:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unrecognized options detected')
            result = milp(objective, integrality=integrality, bounds=Bounds(lower, upper),
                          constraints=LinearConstraint(matrix, np.asarray(lbs), np.asarray(ubs)),
                          options={'time_limit': remaining, 'mip_rel_gap': .01, 'threads': 1, 'presolve': True})
    solver_cpu = process_time() - solver_started
    solver_wall = perf_counter() - solver_wall_started
    assignment = None
    if result is not None and result.x is not None:
        values = result.x[:nx].reshape(n, h)
        labels = values.argmax(axis=1)
        if (np.all(values[np.arange(n), labels] >= 1 - 1e-5)
                and np.count_nonzero(labels != np.asarray(inc)) <= limit
                and feasible(labels, forecast, capacity, h)):
            normalized = canonical_assignment(labels)
            if normalized != inc:
                assignment = normalized
    proposal_cpu = process_time() - started
    selection_started = process_time()
    selection_passed = False
    if assignment is not None:
        reference_profile = panel_profile(selection, inc, capacity)
        candidate_profile = panel_profile(selection, assignment, capacity)
        names = tuple(config.get('selection_guard_names', PROFILE_NAMES))
        columns = [PROFILE_NAMES.index(name) for name in names]
        gain = (reference_profile - candidate_profile).mean(axis=0)[columns]
        tolerance = float(config.get('selection_tolerance', 1e-12))
        selection_passed = bool(np.all(gain >= -tolerance) and np.any(gain > tolerance))
        if not selection_passed:
            assignment = None
    stats = {'cpu_seconds': proposal_cpu, 'wall_seconds': perf_counter() - wall_started,
             'selection_cpu_seconds': process_time() - selection_started, 'selection_guard_passed': int(selection_passed),
             'solver_status': None if result is None else int(result.status),
             'solver_message': 'Model construction exhausted the proposal allowance' if result is None else str(result.message),
             'mip_gap': None if getattr(result, 'mip_gap', None) is None or not np.isfinite(result.mip_gap) else float(result.mip_gap),
             'mip_objective': None if result is None or result.fun is None else float(result.fun),
             'mip_nodes': int(getattr(result, 'mip_node_count', 0) or 0),
             'mip_rows': len(idx), 'variables': variables, 'constraints': len(lbs),
             'max_migrations': limit, 'evaluations': int(assignment is not None), 'blocks': 0,
             'model_construction_cpu_seconds': construction_cpu,
             'model_construction_wall_seconds': construction_wall,
             'solver_cpu_seconds': solver_cpu, 'solver_wall_seconds': solver_wall,
             'solver_wall_allowance_seconds': remaining, 'solver_started': result is not None,
             'allowance_seconds': budget, 'proposal_overshoot_seconds': max(0., proposal_cpu - budget)}
    return {'selected': assignment, 'bets': None, 'candidates': [], 'stats': stats}
