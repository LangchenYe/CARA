"""Independent NumPy reference cache for an immutable (B,N,T) panel.

Only identical ascending resident tuples share work. Each host uses the same
resident gather and sum as data.host_loads. Time reductions keep the original
contiguous (B,T) layout; host reductions assemble the full canonical (H,B)
layout used by refinement.profile_from_loads. No native or query-engine code
is imported. This is an untimed verification oracle, not an execution plan.
"""
import numpy as np


class CanonicalProfileOracle:
    def __init__(self, panel, capacity, tolerance=1e-12):
        self.panel = np.asarray(panel, dtype=np.float64).view()
        if self.panel.ndim != 3 or not self.panel.shape[1] or capacity <= 0:
            raise ValueError("Expected a nonempty (scenes, workloads, time) panel")
        self.panel.flags.writeable = False
        self.capacity, self.tolerance = float(capacity), float(tolerance)
        self.groups = {}

    def _group(self, residents):
        cached = self.groups.get(residents)
        if cached is not None:
            return cached
        # This expression is the original canonical resident reduction.
        loads = self.panel[:, list(residents), :].sum(axis=1)
        if not np.isfinite(loads).all() or np.any(loads < 0):
            raise ValueError("Expected finite nonnegative host loads")
        active = np.any(loads != 0, axis=1)
        native_overload = loads > self.capacity
        overload = loads > self.capacity + self.tolerance
        excess = np.maximum(loads / self.capacity - 1., 0.)
        violating_times = native_overload.sum(axis=1)
        affected = violating_times > 0
        severity = np.divide(excess.sum(axis=1), violating_times,
                             out=np.zeros_like(violating_times, dtype=float),
                             where=affected)
        columns = (active, overload.any(axis=1), native_overload.mean(axis=1),
                   excess.mean(axis=1), severity, affected, loads.max(axis=1))
        loads.flags.writeable = False
        for column in columns:
            column.flags.writeable = False
        cached = (loads, columns)
        self.groups[residents] = cached
        return cached

    def profile(self, assignment):
        # First-occurrence host order and ascending residents match the scalar
        # reference, including assignments with noncanonical external labels.
        groups = {}
        for vm, label in enumerate(assignment):
            groups.setdefault(int(label), []).append(vm)
        if sum(map(len, groups.values())) != self.panel.shape[1]:
            raise ValueError("Panel and complete assignment disagree")
        columns = [self._group(tuple(group))[1] for group in groups.values()]
        active, overload, frequency, excess, severity, affected, peaks = (
            np.stack(values, axis=0) for values in zip(*columns))
        scenes = self.panel.shape[0]
        active_count = active.sum(axis=0)
        vs = np.divide(severity.sum(axis=0), affected.sum(axis=0),
                       out=np.zeros(scenes), where=affected.any(axis=0))
        vr = np.divide(frequency.sum(axis=0), active_count,
                       out=np.full(scenes, -1.), where=active_count > 0)
        raw = np.divide(excess.sum(axis=0), active_count,
                        out=np.zeros(scenes), where=active_count > 0)
        mean_peak = np.divide(peaks.sum(axis=0), active_count,
                             out=np.zeros(scenes), where=active_count > 0)
        par = np.divide(peaks.max(axis=0), mean_peak,
                        out=np.full(scenes, -1.), where=mean_peak > 0)
        return np.column_stack((overload.any(axis=0), overload.mean(axis=0),
                                vr, raw, vs, par))
