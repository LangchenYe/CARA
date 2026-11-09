"""A compiled complete-pair control for certificate query execution.

Both membership indices and cardinality buckets are compiled before scene
acquisition. Boolean projections are shared by complete membership. The raw
projection retains the original canonical host/time reduction.
"""
from collections import defaultdict

import numpy as np

from .action_query import ActionQueryPlan


class CompiledGlobalPlan(ActionQueryPlan):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        groups = tuple(dict.fromkeys(self.old + self.new))
        positions = {group: j for j, group in enumerate(groups)}
        self.old_index = np.asarray([positions[g] for g in self.old])
        self.new_index = np.asarray([positions[g] for g in self.new])
        buckets = defaultdict(list)
        for j, group in enumerate(groups):
            buckets[len(group)].append((j, group))
        self.compiled_groups = tuple((np.asarray([j for j, _ in bucket]),
                                     np.asarray([g for _, g in bucket], dtype=np.intp))
                                    for bucket in buckets.values())
        self.group_count = len(groups)

    def evaluate_global(self, panel, active=(True, True, True, True)):
        panel = np.asarray(panel, dtype=np.float64)
        if panel.ndim != 3 or panel.shape[1:] != (len(self.incumbent), self.horizon):
            raise ValueError("The fixed action and scene panel disagree")
        active = tuple(map(bool, active))
        if len(active) != 4:
            raise ValueError("Four active-coordinate flags are required")
        b, _, t = panel.shape
        self.stats["query_batches"] += 1; self.stats["global_batches"] += 1
        for e, enabled in enumerate(active):
            self.stats["active_coordinate_batches"][e] += int(enabled)
        loads = np.empty((self.group_count, b, t))
        for positions, residents in self.compiled_groups:
            loads[positions] = panel[:, residents, :].transpose(1, 0, 2, 3).sum(axis=2)
        self.stats["global_operand_cells"] += (len(self.incumbent) + self.support_residents) * b * t
        values = [np.zeros((b, 4)), np.zeros((b, 4))]
        if any(active[:3]):
            bits = loads > self.capacity + self.tolerance
            occurrence = bits.any(axis=2) if any(active[:2]) else None
            counts = bits.sum(axis=2) if active[2] else None
            for output, index in zip(values, (self.old_index, self.new_index)):
                if active[0]:output[:, 0] = occurrence[index].any(axis=0)
                if active[1]:output[:, 1] = occurrence[index].sum(axis=0) / self.hosts
                if active[2]:output[:, 2] = counts[index].sum(axis=0) / (self.hosts * t)
        if active[3]:
            excess = np.maximum(loads / self.capacity - 1., 0.)
            for output, index in zip(values, (self.old_index, self.new_index)):
                output[:, 3] = excess[index].sum(axis=(0, 2)) / (self.hosts * t)
        lo, hi = np.zeros((b, 4)), np.zeros((b, 4))
        for e, enabled in enumerate(active):
            if enabled:
                center = values[0][:, e] - values[1][:, e]
                lo[:, e], hi[:, e] = np.nextafter(center, -np.inf), np.nextafter(center, np.inf)
                zero = values[0][:, e] == values[1][:, e]
                lo[zero, e] = hi[zero, e] = 0.
                if e == 0:lo[:, e] = hi[:, e] = center
        return lo, hi


class CSRGlobalPlan(ActionQueryPlan):
    """Complete shared views with the same reassociated sparse kernel as edits."""
    def __init__(self, *args, raw_projection="aggregate", budget_backend="float", **kwargs):
        from scipy.sparse import csr_matrix
        from .action_query import _outward, _sum_error
        from .bounded_query import BoundedPairProjection, projection_raw_allowance
        from .error_budget import projection_raw_allowance_upper, sum_error_upper
        super().__init__(*args, **kwargs)
        groups = tuple(dict.fromkeys(self.old + self.new))
        positions = {group: j for j, group in enumerate(groups)}
        self.old_index = np.asarray([positions[g] for g in self.old])
        self.new_index = np.asarray([positions[g] for g in self.new])
        self.new_selector = (slice(int(self.new_index[0]), int(self.new_index[0]) + self.hosts)
            if np.array_equal(self.new_index, np.arange(self.new_index[0], self.new_index[0] + self.hosts))
            else self.new_index)
        lengths = np.asarray(list(map(len, groups)))
        columns = np.asarray([i for g in groups for i in g], dtype=np.intp)
        self.membership_matrix = csr_matrix((np.ones(len(columns)),
            (np.repeat(np.arange(len(groups)), lengths), columns)),
            shape=(len(groups), len(self.incumbent)))
        upper = self._raw_parameters[1]
        if budget_backend not in ("float", "exact"):
            raise ValueError("Arithmetic budget is float or exact")
        by_size = {r: (2 * sum_error_upper(int(r), upper)
                       if budget_backend == "float" else _outward(2 * _sum_error(int(r), upper), True))
                   for r in set(lengths)}
        radii = np.asarray([by_size[r] for r in lengths])
        if raw_projection not in ("aggregate", "bounds"):
            raise ValueError("Raw projection is aggregate or bounds")
        allowance = None
        if raw_projection == "aggregate":
            raw_bound = projection_raw_allowance_upper if budget_backend == "float" else projection_raw_allowance
            allowance = tuple(raw_bound(sizes, radii[index], self.capacity,
                self.horizon, upper) for sizes, index in
                ((tuple(map(len, self.old)), self.old_index),
                 (tuple(map(len, self.new)), self.new_index)))
        self.projection = BoundedPairProjection(self.capacity, self.horizon,
            radii[self.old_index], radii[self.new_index], tolerance=self.tolerance,
            raw_allowance=allowance)
        self.stats.update({"source_workspace_values": 0, "aggregate_input_cells": 0,
                           "source_panel_reads": 0, "ambiguous_host_scene_queries": 0})

    def evaluate_global(self, panel, active=(True, True, True, True)):
        panel = np.asarray(panel, dtype=np.float64)
        if panel.ndim != 3 or panel.shape[1:] != (len(self.incumbent), self.horizon):
            raise ValueError("The fixed action and scene panel disagree")
        active = tuple(map(bool, active))
        if len(active) != 4:
            raise ValueError("Four active-coordinate flags are required")
        b, n, t = panel.shape
        self.stats["query_batches"] += 1; self.stats["global_batches"] += 1
        for e, enabled in enumerate(active):
            self.stats["active_coordinate_batches"][e] += int(enabled)
        workspace = panel.transpose(1, 0, 2).reshape(n, -1)
        centers = (self.membership_matrix @ workspace).reshape(-1, b, t)
        self.stats["global_operand_cells"] += (n + self.support_residents) * b * t
        self.stats["aggregate_input_cells"] += (n + self.support_residents) * b * t
        self.stats["source_workspace_values"] += n * b * t
        self.stats["source_panel_reads"] += ((n + self.support_residents) if np.shares_memory(workspace, panel) else n) * b * t
        result = self.projection.evaluate(centers[:self.hosts], centers[self.new_selector], active)
        self.stats["ambiguous_host_scene_queries"] = self.projection.ambiguous_host_scenes
        return result


class NativeGlobalPlan(ActionQueryPlan):
    """Complete shared reconstruction using the same direct scan as edit plans."""
    def __init__(self, *args, aligned_assignment=None, budget_backend="float", **kwargs):
        from .action_query import _outward, _sum_error
        from .bounded_query import BoundedPairProjection, projection_raw_allowance
        from .error_budget import projection_raw_allowance_upper, sum_error_upper
        from .native_aggregate import NativeAggregatePlan
        super().__init__(*args, **kwargs)
        if budget_backend not in ("float", "exact"):
            raise ValueError("Arithmetic budget is float or exact")
        aligned, physical = self.physical_layout(aligned_assignment)
        old_sizes = tuple(map(len, self.old))
        new_sizes = np.empty(self.hosts, dtype=int)
        for p, group in zip(physical, self.new):new_sizes[p] = len(group)
        upper = self._raw_parameters[1]
        by_size = {r: (2 * sum_error_upper(int(r), upper)
                       if budget_backend == "float" else _outward(2 * _sum_error(int(r), upper), True))
                   for r in set(old_sizes) | set(new_sizes)}
        radii = tuple(np.asarray([by_size[r] for r in sizes]) for sizes in (old_sizes, new_sizes))
        raw_bound = projection_raw_allowance_upper if budget_backend == "float" else projection_raw_allowance
        allowance = tuple(raw_bound(sizes, rho, self.capacity, self.horizon, upper)
                          for sizes, rho in zip((old_sizes, new_sizes), radii))
        self.projection = BoundedPairProjection(self.capacity, self.horizon, *radii,
            tolerance=self.tolerance, raw_allowance=allowance)
        self.native = NativeAggregatePlan(self.incumbent, aligned)
        self.stats.update({"source_workspace_values": 0, "aggregate_input_cells": 0,
                           "source_panel_reads": 0, "ambiguous_host_scene_queries": 0})

    def evaluate_global(self, panel, active=(True, True, True, True)):
        panel = np.ascontiguousarray(panel, dtype=np.float64)
        if panel.ndim != 3 or panel.shape[1:] != (len(self.incumbent), self.horizon):
            raise ValueError("The fixed action and scene panel disagree")
        active = tuple(map(bool, active))
        if len(active) != 4:
            raise ValueError("Four active-coordinate flags are required")
        b, n, t = panel.shape
        self.stats["query_batches"] += 1; self.stats["global_batches"] += 1
        for e, enabled in enumerate(active):
            self.stats["active_coordinate_batches"][e] += int(enabled)
        reference, candidate = self.native.aggregate(panel, "full")
        self.stats["global_operand_cells"] += (n + self.support_residents) * b * t
        self.stats["aggregate_input_cells"] += (n + self.support_residents) * b * t
        self.stats["source_panel_reads"] += n * b * t
        result = self.projection.evaluate(reference, candidate, active)
        self.stats["ambiguous_host_scene_queries"] = self.projection.ambiguous_host_scenes
        return result
