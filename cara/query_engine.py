"""Shared, order-preserving execution of placement what-if queries.

The workload panel is immutable within a session. A host view is identified by
its complete ascending resident tuple, independently of candidate or host label.
Queries can arrive individually or as a complete residual batch. Only residents
of a single host are reduced together; unchanged inputs never use delta sums.
"""
from collections import OrderedDict, defaultdict

import numpy as np

from .data import canonical_assignment, host_memberships
from .workload_views import host_projection, profile_from_projection


def _group_loads(panel, groups):
    """Batch groups of equal cardinality with canonical resident reductions."""
    indices = np.asarray(groups, dtype=np.intp)
    # Each logical (B,r,T) slice has the same strides and resident reduction
    # axis as panel[:, list(group), :].sum(axis=1), including B=1 or T=1.
    work = panel[:, indices, :].transpose(1, 0, 2, 3)
    return work.sum(axis=2)


def _profiles(projected):
    """Reduce (query, canonical host, scene) rows without reassociation."""
    active, tolerant, affected, frequency, excess, severity, peaks = projected
    count = active.sum(axis=1)
    shape = count.shape
    vr = np.divide(frequency.sum(axis=1), count,
                   out=np.full(shape, -1.), where=count > 0)
    raw = np.divide(excess.sum(axis=1), count,
                    out=np.zeros(shape), where=count > 0)
    vs = np.divide(severity.sum(axis=1), affected.sum(axis=1),
                   out=np.zeros(shape), where=affected.any(axis=1))
    mean_peak = np.divide(peaks.sum(axis=1), count,
                          out=np.zeros(shape), where=count > 0)
    par = np.divide(peaks.max(axis=1), mean_peak,
                    out=np.full(shape, -1.), where=mean_peak > 0)
    return np.stack((tolerant.any(axis=1), tolerant.mean(axis=1), vr, raw, vs, par), axis=2)


class ForecastViews:
    """Membership-keyed canonical capacity predicates for one fixed forecast."""
    def __init__(self, forecast, capacity, seed=None, *, tolerance=1e-12,
                 max_views=4096, batch_values=1 << 20):
        self.forecast = np.asarray(forecast, dtype=np.float64)
        if (self.forecast.ndim != 2 or min(self.forecast.shape) < 1
                or not np.isfinite(self.forecast).all() or np.any(self.forecast < 0)):
            raise ValueError("Expected a finite nonnegative VM-by-time forecast")
        if (not np.isfinite(capacity) or capacity <= 0 or not np.isfinite(tolerance)
                or tolerance < 0 or any(int(x) != x or x < 1 for x in (max_views, batch_values))):
            raise ValueError("Invalid capacity, tolerance or capacity-view limits")
        self.capacity, self.tolerance = capacity, tolerance
        self.max_views, self.batch_values = int(max_views), int(batch_values)
        self.views = OrderedDict()
        self.hits = self.group_evaluations = self.forecast_values = self.evictions = 0
        if seed is not None:
            if len(seed) != self.forecast.shape[0]:
                raise ValueError("Complete assignment and forecast disagree")
            self._obtain(host_memberships(seed))

    @classmethod
    def from_config(cls, forecast, capacity, seed, config):
        return cls(forecast, capacity, seed, max_views=config.get("query_cache_views", 4096),
                   batch_values=config.get("query_batch_values", 1 << 20))

    def _obtain(self, groups):
        local, missing = {}, defaultdict(list)
        times = self.forecast.shape[1]
        for group in groups:
            if group in local:
                self.hits += 1
            elif group in self.views:
                self.hits += 1
                local[group] = self.views[group]
                self.views.move_to_end(group)
            else:
                local[group] = None
                missing[len(group)].append(group)
        for size, residents in missing.items():
            chunk_size = max(1, self.batch_values // max(1, size * times))
            for start in range(0, len(residents), chunk_size):
                chunk = residents[start:start + chunk_size]
                loads = _group_loads(self.forecast[None, :, :], chunk)[:, 0, :]
                passing = np.all(loads <= self.capacity + self.tolerance, axis=1)
                for group, value in zip(chunk, passing):
                    local[group] = bool(value)
                    self.views[group] = bool(value)
                    if len(self.views) > self.max_views:
                        self.views.popitem(last=False)
                        self.evictions += 1
                self.group_evaluations += len(chunk)
                self.forecast_values += len(chunk) * size * times
        return local

    def feasible(self, assignment, host_count):
        if host_count < 1 or len(assignment) != self.forecast.shape[0]:
            return False
        groups = host_memberships(assignment)
        if len(groups) != host_count:
            return False
        values = self._obtain(groups)
        return all(values[group] for group in groups)

    def feasible_many(self, assignments, host_count):
        """Predicate results in input order, preserving nonempty host counts."""
        if host_count < 1:
            return np.zeros(len(assignments), dtype=bool)
        memberships = [host_memberships(a) if len(a) == self.forecast.shape[0] else ()
                       for a in assignments]
        eligible = [groups for groups in memberships if len(groups) == host_count]
        values = self._obtain([group for groups in eligible for group in groups])
        return np.asarray([len(groups) == host_count and all(values[group] for group in groups)
                           for groups in memberships], dtype=bool)

    def groups_feasible(self, groups):
        """Check changed nonempty groups of a previously feasible placement."""
        if not groups or any(not group for group in groups):
            return False
        values = self._obtain(groups)
        return all(values[group] for group in groups)

    def groups_feasible_many(self, group_sets):
        """Batched changed-group predicates; fixed residents stay in each key."""
        eligible = [groups for groups in group_sets if groups and all(groups)]
        values = self._obtain([group for groups in eligible for group in groups])
        return np.asarray([bool(groups) and all(groups) and all(values[group] for group in groups)
                           for groups in group_sets], dtype=bool)


class SharedQueryReplay:
    """Bounded host-view registry shared by an adaptive what-if query session.

    ``profiles`` executes a batch in its supplied order; it neither selects nor
    filters candidates. At most ``max_views`` host groups remain between calls.
    Batch-local views survive eviction until that batch finishes. All storage
    and materialization are charged to the caller's execution allowance.
    """
    def __init__(self, panel, seed, capacity, *, max_views=2048, max_bytes=32 << 20,
                 batch_values=1 << 20, registry=None, materialize_seed=False):
        self.panel = np.asarray(panel, dtype=np.float64)
        if self.panel.ndim != 3 or min(self.panel.shape) < 1:
            raise ValueError("Expected a nonempty scene-by-VM-by-time panel")
        if registry is None and (not np.isfinite(self.panel).all() or np.any(self.panel < 0)):
            raise ValueError("Expected finite nonnegative scene-by-VM-by-time demands")
        if not np.isfinite(capacity) or capacity <= 0 or int(max_views) != max_views or max_views < 1:
            raise ValueError("Capacity and view limit must be positive")
        if any(int(value) != value or value < 1 for value in (batch_values, max_bytes)):
            raise ValueError("Batch and stored-byte limits must be positive")
        self.seed = canonical_assignment(seed)
        if len(self.seed) != self.panel.shape[1] or not self.seed:
            raise ValueError("Complete assignment and workload panel disagree")
        self.capacity, self.batch_values = capacity, int(batch_values)
        b, n, t = self.panel.shape
        self.bytes_per_view = b * t * 8 + b * (4 * 8 + 3)
        if registry is None:
            # Numerical arrays, membership tuples and mappings are all bounded
            # by max_views; max_bytes additionally bounds retained array data.
            self.max_views = min(int(max_views), max(0, int(max_bytes) // self.bytes_per_view))
            self.max_bytes = int(max_bytes)
            self.views = OrderedDict()
            self._registry = self
            self.evictions = self.peak_view_bytes = 0
        else:
            owner = registry._registry
            if self.panel is not owner.panel or capacity != owner.capacity:
                raise ValueError("Shared views require the same immutable panel and capacity")
            self._registry = owner
            self.views, self.max_views, self.max_bytes = owner.views, owner.max_views, owner.max_bytes
        self.hits = self.projection_hits = self.changed_values = self.full_values = 0
        self.materialized_values = self.group_queries = self.unique_groups = 0
        if materialize_seed:
            self._obtain(host_memberships(self.seed), materializing=True)

    def fork(self, seed):
        """Use a new seed while retaining the same bounded, demand-driven registry."""
        return type(self)(self.panel, seed, self.capacity, max_views=max(1, self.max_views),
                          max_bytes=self.max_bytes, batch_values=self.batch_values, registry=self)

    @classmethod
    def from_config(cls, panel, seed, capacity, config):
        return cls(panel, seed, capacity, max_views=config.get("query_cache_views", 2048),
                   max_bytes=config.get("query_cache_bytes", 32 << 20),
                   batch_values=config.get("query_batch_values", 1 << 20))

    def _obtain(self, groups, *, materializing=False):
        local, missing = {}, defaultdict(list)
        b, n, t = self.panel.shape
        self.group_queries += len(groups)
        for group in groups:
            if group in local:
                self.hits += 1
            elif group in self.views:
                self.hits += 1
                local[group] = self.views[group]
                self.views.move_to_end(group)
            else:
                local[group] = None
                missing[len(group)].append(group)
        for size, residents in missing.items():
            limit = max(1, self.batch_values // max(1, b * size * t))
            for start in range(0, len(residents), limit):
                chunk = residents[start:start + limit]
                loads = _group_loads(self.panel, chunk)
                projected = host_projection(loads, self.capacity)
                for j, group in enumerate(chunk):
                    # Copies detach views from a larger temporary batch buffer.
                    value = (loads[j].copy(), tuple(column[j].copy() for column in projected))
                    local[group] = value
                    self.views[group] = value
                    self.unique_groups += 1
                    if len(self.views) > self.max_views:
                        self.views.popitem(last=False)
                        self._registry.evictions += 1
                accessed = b * size * t * len(chunk)
                self.materialized_values += accessed
                if not materializing:
                    self.changed_values += accessed
        # All entries have identical B,T shapes; storage depends only on count.
        self._registry.peak_view_bytes = max(self._registry.peak_view_bytes,
                                             len(self.views) * self.bytes_per_view)
        return local

    def loads(self, assignment):
        groups = host_memberships(assignment)
        if sum(map(len, groups)) != self.panel.shape[1]:
            raise ValueError("Complete assignment and workload panel disagree")
        self.full_values += self.panel.size
        values = self._obtain(groups)
        return np.stack([values[group][0] for group in groups])

    def loads_many(self, assignments):
        """Return candidate-by-host-by-scene-by-time canonical load arrays."""
        memberships = [host_memberships(assignment) for assignment in assignments]
        b, n, t = self.panel.shape
        if not memberships:
            return np.empty((0, len(set(self.seed)), b, t), dtype=np.float64)
        hosts = len(memberships[0])
        if any(len(groups) != hosts or sum(map(len, groups)) != n for groups in memberships):
            raise ValueError("A batch requires complete assignments with equal host count")
        self.full_values += self.panel.size * len(memberships)
        values = self._obtain([group for groups in memberships for group in groups])
        return np.stack([np.stack([values[group][0] for group in groups]) for groups in memberships])

    def profile(self, assignment):
        groups = host_memberships(assignment)
        if sum(map(len, groups)) != self.panel.shape[1]:
            raise ValueError("Complete assignment and workload panel disagree")
        self.full_values += self.panel.size
        before = self.hits
        values = self._obtain(groups)
        self.projection_hits += self.hits - before
        columns = tuple(np.stack([values[group][1][e] for group in groups]) for e in range(7))
        return profile_from_projection(columns)

    def profiles(self, assignments):
        """Return (candidate, scene, metric) canonical profiles for a batch."""
        memberships = [host_memberships(assignment) for assignment in assignments]
        return self.profiles_groups(memberships)

    def profiles_groups(self, memberships):
        """Execute complete canonical resident groups without rebuilding labels.

        Each supplied partition uses the same ascending resident tuples and
        minimum-VM host order as ``host_memberships``. Family producers enforce
        this contract before calling; loads and floating reductions are unchanged.
        """
        b, n, t = self.panel.shape
        if not memberships:
            return np.empty((0, b, 6), dtype=np.float64)
        hosts = len(memberships[0])
        if len(memberships) * hosts * b > self.batch_values:
            from .query_plan import compiled_profiles
            return compiled_profiles(self, memberships=memberships, profile_cells=self.batch_values)
        if any(len(groups) != hosts or sum(map(len, groups)) != n for groups in memberships):
            raise ValueError("A batch requires complete assignments with equal host count")
        self.full_values += self.panel.size * len(memberships)
        before = self.hits
        values = self._obtain([group for groups in memberships for group in groups])
        self.projection_hits += self.hits - before
        unique = list(values)
        ids = {group: i for i, group in enumerate(unique)}
        lookup = np.asarray([[ids[group] for group in groups] for groups in memberships], dtype=np.intp)
        columns = tuple(np.stack([values[group][1][e] for group in unique])[lookup] for e in range(7))
        return _profiles(columns)

    @property
    def cached_view_bytes(self):
        return sum(loads.nbytes + sum(column.nbytes for column in projected)
                   for loads, projected in self.views.values())
