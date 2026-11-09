"""Canonical host projections for repeated placement queries.

Unchanged membership groups reuse both their load sums and time reductions.
Changed groups use the same NumPy summation shape as complete replay. Host
reductions retain canonical order; no subtraction updates are performed.
"""
import numpy as np
from .data import canonical_assignment, host_loads, host_memberships


def host_projection(loads, capacity, tolerance=1e-12):
    """Return host-by-scene sufficient statistics, preserving replay arithmetic."""
    loads = np.asarray(loads, dtype=float)
    active = np.any(loads != 0, axis=2)
    native_overload = loads > capacity
    tolerant_overload = loads > capacity + tolerance
    excess = np.maximum(loads / capacity - 1., 0.)
    count = native_overload.sum(axis=2)
    affected = count > 0
    severity = np.divide(excess.sum(axis=2), count,
                         out=np.zeros_like(count, dtype=float), where=affected)
    return (active, tolerant_overload.any(axis=2), affected,
            native_overload.mean(axis=2), excess.mean(axis=2), severity,
            loads.max(axis=2))


def profile_from_projection(projection):
    active, tolerant, affected, frequency, excess, severity, peaks = projection
    active_count = active.sum(axis=0)
    rows = active.shape[1]
    vr = np.divide(frequency.sum(axis=0), active_count,
                   out=np.full(rows, -1.), where=active_count > 0)
    raw = np.divide(excess.sum(axis=0), active_count,
                    out=np.zeros(rows), where=active_count > 0)
    vs = np.divide(severity.sum(axis=0), affected.sum(axis=0),
                   out=np.zeros(rows), where=affected.any(axis=0))
    mean_peak = np.divide(peaks.sum(axis=0), active_count,
                          out=np.zeros(rows), where=active_count > 0)
    par = np.divide(peaks.max(axis=0), mean_peak,
                    out=np.full(rows, -1.), where=mean_peak > 0)
    return np.column_stack((tolerant.any(axis=0), tolerant.mean(axis=0), vr, raw, vs, par))


class ProjectionReplay:
    """Seed-scoped workload views with canonical per-host time projections."""
    def __init__(self, panel, seed, capacity):
        self.panel = np.asarray(panel, dtype=np.float64)
        self.seed = canonical_assignment(seed)
        self.capacity = capacity
        loads = host_loads(self.panel, self.seed)
        projected = host_projection(loads, capacity)
        members = host_memberships(self.seed)
        self.groups = dict(zip(members, loads))
        self.projections = {group: tuple(column[h] for column in projected)
                            for h, group in enumerate(members)}
        self.hits = self.projection_hits = self.changed_values = self.full_values = 0
        self.materialized_values = self.panel.size

    def loads(self, assignment):
        arrays = []
        b, n, t = self.panel.shape
        self.full_values += b * n * t
        for members in host_memberships(assignment):
            if members in self.groups:
                self.hits += 1
                arrays.append(self.groups[members])
            else:
                arrays.append(self.panel[:, list(members), :].sum(axis=1))
                self.changed_values += b * len(members) * t
        return np.stack(arrays)

    def profile(self, assignment):
        groups = host_memberships(assignment)
        b, n, t = self.panel.shape
        self.full_values += b * n * t
        columns = [[] for _ in range(7)]
        for members in groups:
            if members in self.projections:
                self.hits += 1
                self.projection_hits += 1
                values = self.projections[members]
            else:
                loads = self.panel[:, list(members), :].sum(axis=1)
                values = tuple(column[0] for column in host_projection(loads[None, :, :], self.capacity))
                self.changed_values += b * len(members) * t
            for output, value in zip(columns, values):
                output.append(value)
        return profile_from_projection(tuple(np.stack(column) for column in columns))
