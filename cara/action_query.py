"""Certificate-directed queries for one frozen placement action.

Unchanged memberships cancel in real-valued gains, but not necessarily in the
stored global floating reductions. The local reader therefore encloses those
same canonical stored gains. Positive betting products propagate the enclosures;
only an unresolved threshold comparison requests exact canonical prefix replay.
"""
from collections import defaultdict
from fractions import Fraction
from math import inf, isfinite, nextafter
from time import process_time

import numpy as np

from .admission import _alpha
from .certification import _DyadicMixture, _declared_threshold, _names, _rational
from .data import burdens_from_loads, canonical_assignment, host_loads, host_memberships
from .query_engine import _group_loads
from .scene_stream import SceneStream


def _outward(value, upper):
    result = float(value)
    if (Fraction(result) < value) if upper else (Fraction(result) > value):
        result = nextafter(result, inf if upper else -inf)
    return result


def _sum_error(count, operand):
    if not 1 <= count <= 2**53:
        raise ValueError("Summation-tree counts must lie between one and 2**53")
    unit, tiny = Fraction(1, 2**53), Fraction(1, 2**1075)
    # (1+u)^m <= 1/(1-mu). The rational geometric bound avoids a
    # denominator with 53*m bits while staying outward of the tree bound.
    growth = 1 / (1 - max(0, count - 1) * unit)
    return (growth - 1) * count * operand + tiny * max(0, count - 1) * growth


def _raw_error(groups, hosts, horizon, capacity, upper):
    """Uniform error of this subset's raw sum divided by the full H*T.

    The maximum resident count bounds each load tree; the number of included
    host-time terms bounds the final sum tree. No observed maximum is used.
    """
    if not groups:
        return Fraction()
    unit, tiny = Fraction(1, 2**53), Fraction(1, 2**1075)
    residents = max(map(len, groups))
    terms, normalizer = len(groups) * horizon, hosts * horizon
    absolute = residents * upper / capacity
    load = _sum_error(residents, upper)
    ratio = (1 + unit) * load / capacity + unit * absolute + tiny
    excess = (1 + unit) * ratio + unit * (absolute + 1) + tiny
    total = terms * excess + _sum_error(terms, absolute + excess)
    return (1 + unit) * total / normalizer + unit * terms * absolute / normalizer + tiny


def canonical_fresh_losses(panel, assignment, capacity, tolerance=1e-12):
    loads = host_loads(panel, assignment)
    first = burdens_from_loads(loads, capacity, tolerance=tolerance)[:, :3]
    raw = np.maximum(loads / capacity - 1., 0.).sum(axis=(0, 2)) / (len(loads) * loads.shape[2])
    return np.column_stack((first, raw))


class ActionQueryPlan:
    """Read changed memberships, then only unresolved common-host witnesses."""
    def __init__(self, incumbent, candidate, capacity, horizon, *, demand_upper=1,
                 tolerance=1e-12, common_order=None, raw_error_bound=True):
        self.incumbent, self.candidate = map(canonical_assignment, (incumbent, candidate))
        self.old, self.new = map(host_memberships, (self.incumbent, self.candidate))
        if (not self.old or len(self.old) != len(self.new)
                or len(self.incumbent) != len(self.candidate)
                or not isfinite(capacity) or capacity <= 0
                or int(horizon) != horizon or horizon < 1
                or not isfinite(tolerance) or tolerance < 0):
            raise ValueError("A complete equal-host action and valid capacity/horizon are required")
        self.hosts, self.horizon = len(self.old), int(horizon)
        if len(self.incumbent) > 2**53 or self.hosts * self.horizon > 2**53:
            raise ValueError("Resident and host-time counts must not exceed 2**53")
        self.capacity, self.tolerance = float(capacity), float(tolerance)
        common = set(self.old) & set(self.new)
        self.changed_old = tuple(group for group in self.old if group not in common)
        self.changed_new = tuple(group for group in self.new if group not in common)
        self.changed = self.changed_old + self.changed_new
        self.support_residents = sum(map(len, self.changed_old))
        self.common = tuple(sorted(common, key=lambda group: (len(group), group)))
        if common_order is not None:
            requested = tuple(map(tuple, common_order))
            if len(requested) != len(common) or set(requested) != common:
                raise ValueError("The witness order must contain every common group exactly once")
            self.common = requested
        support = tuple(map(_rational, demand_upper)) if np.ndim(demand_upper) else (_rational(demand_upper),)
        if (np.ndim(demand_upper) and len(support) != len(self.incumbent)) or any(v < 0 for v in support):
            raise ValueError("A nonnegative population demand support is required")
        upper, cap = max(support), _rational(self.capacity)
        self._raw_parameters = cap, upper
        self.raw_radius = None
        if raw_error_bound:
            self._prepare_raw_radius()
        self.stats = {"query_batches": 0, "changed_operand_cells": 0,
            "common_operand_cells": 0, "common_host_scene_queries": 0,
            "common_hosts_opened": 0, "unresolved_occurrence_rows": 0,
            "global_operand_cells": 0, "local_batches": 0, "global_batches": 0,
            "active_coordinate_batches": [0] * 4}

    def physical_layout(self, aligned_assignment=None):
        """Validate and retain the selected action's fixed physical matching."""
        if aligned_assignment is None:
            from .refinement import migration_profile
            aligned_assignment = migration_profile(self.incumbent, self.candidate)["aligned_assignment"]
        aligned = np.asarray(aligned_assignment, dtype=np.intp)
        if (aligned.shape != (len(self.incumbent),)
                or set(aligned) != set(range(self.hosts))
                or canonical_assignment(aligned) != self.candidate):
            raise ValueError("The frozen matching must preserve the complete candidate partition")
        physical = np.asarray([aligned[group[0]] for group in self.new])
        locations = {group: h for h, group in enumerate(self.old)}
        if any(group in locations and physical[j] != locations[group]
               for j, group in enumerate(self.new)):
            raise ValueError("An unchanged membership must retain its incumbent physical host")
        return aligned, physical

    def _prepare_raw_radius(self):
        if self.raw_radius is None:
            cap, upper = self._raw_parameters
            radius = sum((_raw_error(groups, self.hosts, self.horizon, cap, upper)
                          for groups in (self.old, self.new, self.changed_old, self.changed_new)), Fraction())
            self.raw_radius = Fraction(nextafter(float(radius), inf))

    def evaluate_global(self, panel, active=(True, True, True, True)):
        """Strong all-host control: same active projections, no native extras."""
        panel = np.asarray(panel, dtype=np.float64)
        if panel.ndim != 3 or panel.shape[1:] != (len(self.incumbent), self.horizon):
            raise ValueError("The fixed action and scene panel disagree")
        active = tuple(map(bool, active))
        if len(active) != 4:
            raise ValueError("Four active-coordinate flags are required")
        b, _, t = panel.shape
        self.stats["query_batches"] += 1
        self.stats["global_batches"] += 1
        for e, enabled in enumerate(active):
            self.stats["active_coordinate_batches"][e] += int(enabled)
        groups, buckets, loads = tuple(dict.fromkeys(self.old + self.new)), defaultdict(list), {}
        for group in groups:
            buckets[len(group)].append(group)
        for bucket in buckets.values():
            loads.update(zip(bucket, _group_loads(panel, bucket)))
        self.stats["global_operand_cells"] += b * t * sum(map(len, groups))
        old = np.stack([loads[group] for group in self.old])
        new = np.stack([loads[group] for group in self.new])
        values = [np.zeros((b, 4)), np.zeros((b, 4))]
        for output, data in zip(values, (old, new)):
            if any(active[:3]):
                overload = data > self.capacity + self.tolerance
                if active[0]:
                    output[:, 0] = overload.any(axis=(0, 2))
                if active[1]:
                    output[:, 1] = overload.any(axis=2).sum(axis=0) / self.hosts
                if active[2]:
                    output[:, 2] = overload.sum(axis=(0, 2)) / (self.hosts * t)
            if active[3]:
                output[:, 3] = np.maximum(data / self.capacity - 1., 0.).sum(axis=(0, 2)) / (self.hosts * t)
        lower, upper = np.zeros((b, 4)), np.zeros((b, 4))
        for e, enabled in enumerate(active):
            if enabled:
                center = values[0][:, e] - values[1][:, e]
                lower[:, e], upper[:, e] = np.nextafter(center, -np.inf), np.nextafter(center, np.inf)
                exact_zero = values[0][:, e] == values[1][:, e]
                lower[exact_zero, e] = upper[exact_zero, e] = 0.
                if e == 0:
                    lower[:, e] = upper[:, e] = center
        return lower, upper

    def evaluate(self, panel, active=(True, True, True, True)):
        """Return outward bounds on active canonical gain coordinates.

        The panel is a trusted finite nonnegative population-supported batch.
        Common-host gathers preserve its full scene dimension and strides:
        unqueried scene rows are zero-filled, never trimmed from the reduction.
        """
        panel = np.asarray(panel, dtype=np.float64)
        if panel.ndim != 3 or panel.shape[1:] != (len(self.incumbent), self.horizon):
            raise ValueError("The fixed action and scene panel disagree")
        active = tuple(map(bool, active))
        if len(active) != 4:
            raise ValueError("Four active-coordinate flags are required")
        b, _, t = panel.shape
        lower, upper = np.zeros((b, 4)), np.zeros((b, 4))
        self.stats["query_batches"] += 1
        self.stats["local_batches"] += 1
        for e, enabled in enumerate(active):
            self.stats["active_coordinate_batches"][e] += int(enabled)
        if not self.changed or not any(active):
            return lower, upper
        buckets, loads = defaultdict(list), {}
        for group in self.changed:
            buckets[len(group)].append(group)
        for groups in buckets.values():
            values = _group_loads(panel, groups)
            loads.update(zip(groups, values))
        self.stats["changed_operand_cells"] += b * t * sum(map(len, self.changed))
        old = np.stack([loads[group] for group in self.changed_old])
        new = np.stack([loads[group] for group in self.changed_new])
        if any(active[:3]):
            oi, oa = old > self.capacity + self.tolerance, new > self.capacity + self.tolerance
            if active[0]:
                ji, ja = oi.any(axis=(0, 2)), oa.any(axis=(0, 2))
                unresolved = ji != ja
                self.stats["unresolved_occurrence_rows"] += int(unresolved.sum())
                gain = ji.astype(float) - ja.astype(float)
                for group in self.common:
                    rows = np.flatnonzero(unresolved)
                    if not len(rows):
                        break
                    # Canonical panel[:, group, :] has the (T,r*BT,1) logical
                    # layout obtained by this transpose, at the SAME B and T.
                    work = np.zeros((len(group), b, t))
                    work[:, rows, :] = panel[rows[:, None], np.asarray(group)[None, :], :].transpose(1, 0, 2)
                    common_load = work.transpose(1, 0, 2).sum(axis=1)
                    witness = (common_load > self.capacity + self.tolerance).any(axis=1)
                    resolved = unresolved & witness
                    gain[resolved] = 0.
                    unresolved[resolved] = False
                    self.stats["common_operand_cells"] += len(rows) * len(group) * t
                    self.stats["common_host_scene_queries"] += len(rows)
                    self.stats["common_hosts_opened"] += 1
                lower[:, 0] = upper[:, 0] = gain
            for e, denominator in ((1, self.hosts), (2, self.hosts * t)):
                if not active[e]:
                    continue
                difference_counts = (oi.any(axis=2).sum(axis=0) - oa.any(axis=2).sum(axis=0)
                    if e == 1 else oi.sum(axis=(0, 2)) - oa.sum(axis=(0, 2)))
                dyadic = denominator & (denominator - 1) == 0
                center = difference_counts / denominator
                if dyadic:
                    lower[:, e] = upper[:, e] = center
                else:
                    # Two original ratio roundings plus the local center's
                    # division are <=3u absolute; 4u and outward endpoints
                    # cover them without a per-scene rational calculation.
                    lower[:, e] = np.nextafter(center - 2. ** -51, -np.inf)
                    upper[:, e] = np.nextafter(center + 2. ** -51, np.inf)
                    lower[difference_counts == 0, e] = upper[difference_counts == 0, e] = 0.
        if active[3]:
            self._prepare_raw_radius()
            partial_i = np.maximum(old / self.capacity - 1., 0.).sum(axis=(0, 2)) / (self.hosts * t)
            partial_a = np.maximum(new / self.capacity - 1., 0.).sum(axis=(0, 2)) / (self.hosts * t)
            center = partial_i - partial_a
            # First enclose the subtraction of stored local values, then add
            # the predeclared full/local replay allowance outwards.
            radius = float(self.raw_radius)
            lower[:, 3] = np.nextafter(np.nextafter(center, -np.inf) - radius, -np.inf)
            upper[:, 3] = np.nextafter(np.nextafter(center, np.inf) + radius, np.inf)
        return lower, upper


class ActionSceneReader:
    """Same iid scene stream, with local queries and lazy exact-prefix fallback."""
    def __init__(self, environment, law, role, incumbent, candidate, capacity, *,
                 horizon, demand_upper=1, batch_size=32, common_order=None, execution="local",
                 stream_factory=None, plan_factory=None):
        if int(batch_size) != batch_size or batch_size < 1 or execution not in ("local", "global", "adaptive"):
            raise ValueError("A positive fixed replay batch size is required")
        self.execution = execution
        self.environment, self.law, self.role = environment, law, role
        self.stream_factory = stream_factory or (lambda: SceneStream(environment, law, role))
        self.stream = self.stream_factory()
        began = process_time()
        factory = ActionQueryPlan if plan_factory is None else plan_factory
        self.plan = factory(incumbent, candidate, capacity, horizon,
                                   demand_upper=demand_upper, common_order=common_order,
                                   raw_error_bound=execution == "local")
        self.planning_cpu_seconds = process_time() - began
        self.generation_cpu_seconds = self.replay_cpu_seconds = self.reader_cpu_seconds = 0.
        self.fallback_cpu_seconds = 0.
        self.batch_size = int(batch_size)
        self.exact_stream, self.exact_reference, self.exact_action = None, [], []
        self.stats = {"fallback_rows_generated": 0, "exact_prefix_requests": 0}

    @property
    def rows(self):
        return self.stream.rows

    def gain_intervals(self, indices, active):
        began = process_time()
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
                or not np.array_equal(indices, np.arange(self.rows, self.rows + len(indices)))):
            raise ValueError("Consecutive unread scene indices are required")
        lo, hi = [], []
        for start in range(0, len(indices), self.batch_size):
            count = min(self.batch_size, len(indices) - start)
            tick = process_time()
            panel = self.stream.next(count)
            self.generation_cpu_seconds += process_time() - tick
            if count < self.batch_size:
                panel = np.pad(panel, ((0, self.batch_size - count), (0, 0), (0, 0)))
            tick = process_time()
            mode = self.execution
            if mode == "adaptive":
                dense_raw = active[3] and 2 * self.plan.support_residents > len(self.plan.incumbent)
                mode = "local" if self.plan.common and not dense_raw else "global"
            lower, upper = (self.plan.evaluate if mode == "local" else self.plan.evaluate_global)(panel, active)
            self.replay_cpu_seconds += process_time() - tick
            lo.append(lower[:count]); hi.append(upper[:count])
        result = np.concatenate(lo), np.concatenate(hi)
        self.reader_cpu_seconds += process_time() - began
        return result

    def exact_prefix(self, coordinate):
        began = process_time()
        self.stats["exact_prefix_requests"] += 1
        if self.exact_stream is None:
            self.exact_stream = self.stream_factory()
        while self.exact_stream.rows < self.rows:
            count = min(self.batch_size, self.rows - self.exact_stream.rows)
            tick = process_time()
            panel = self.exact_stream.next(count)
            self.generation_cpu_seconds += process_time() - tick
            if count < self.batch_size:
                panel = np.pad(panel, ((0, self.batch_size - count), (0, 0), (0, 0)))
            tick = process_time()
            self.exact_reference.append(canonical_fresh_losses(panel, self.plan.incumbent, self.plan.capacity, self.plan.tolerance)[:count])
            self.exact_action.append(canonical_fresh_losses(panel, self.plan.candidate, self.plan.capacity, self.plan.tolerance)[:count])
            self.replay_cpu_seconds += process_time() - tick
            self.stats["fallback_rows_generated"] += count
        reference, action = np.concatenate(self.exact_reference), np.concatenate(self.exact_action)
        result = [Fraction(float(r)) - Fraction(float(a)) for r, a in zip(reference[:, coordinate], action[:, coordinate])]
        elapsed = process_time() - began
        self.fallback_cpu_seconds += elapsed
        self.reader_cpu_seconds += elapsed
        return result


class _GainIntervalMixture:
    """Binary64 outward products enclosing the unchanged positive mixture."""
    def __init__(self, minimum_components=8):
        self.minimum_components, self.rows = minimum_components, []
        self.high, self.small = (1., 1.), {}
        self.exact_fallbacks = 0

    @staticmethod
    def _extend(product, values, rate):
        lo, hi = product
        for lower, upper in values:
            factor_lo = nextafter(1. + nextafter(rate * lower, -inf), -inf)
            factor_hi = nextafter(1. + nextafter(rate * upper, inf), inf)
            lo = max(0., nextafter(lo * factor_lo, -inf))
            hi = nextafter(hi * factor_hi, inf)
        return lo, hi

    def update(self, values):
        self.rows.extend(values)
        self.high = self._extend(self.high, values, 255 / 256)
        for j in self.small:
            self.small[j] = self._extend(self.small[j], values, 2. ** -j)
        count = max(self.minimum_components, len(self.rows).bit_length())
        for j in range(1, count + 1):
            if j not in self.small:
                self.small[j] = self._extend((1., 1.), self.rows, 2. ** -j)

    def bounds(self):
        lo = max(0., nextafter(self.high[0] * .5, -inf))
        hi = nextafter(self.high[1] * .5, inf)
        for j, (lower, upper) in self.small.items():
            weight = 2. ** (-j - 1)
            if weight == 0.:
                # The exact dyadic weight is positive. Do not approximate its
                # scaled contribution by zero: force exact threshold replay.
                hi = inf
                continue
            lo = max(0., nextafter(lo + max(0., nextafter(lower * weight, -inf)), -inf))
            hi = nextafter(hi + nextafter(upper * weight, inf), inf)
        return lo, hi

    def crossed(self, alpha, exact_values):
        lower, upper = self.bounds()
        if Fraction(lower) * alpha.numerator >= alpha.denominator:
            return True
        if isfinite(upper) and Fraction(upper) * alpha.numerator < alpha.denominator:
            return False
        exact = _DyadicMixture(self.minimum_components)
        exact.update(exact_values())
        self.exact_fallbacks += 1
        return exact.crossed(alpha)


def certify_action_queries(reader, row_cap, *, gain_bounds, noninferiority_margins,
                           anchor=2, min_anchor_gain=.005, alpha=.05, batch=32,
                           metric_names=None, minimum_components=8, evidence_engine="scalar"):
    """Preserve canonical-mixture decisions using support-local gain queries."""
    if int(row_cap) != row_cap or row_cap < 1 or int(batch) != batch or batch < 1:
        raise ValueError("Positive integer row cap and inspection batch are required")
    if int(anchor) != anchor or not 0 <= anchor < 4:
        raise ValueError("A fixed anchor among four coordinates is required")
    if int(minimum_components) != minimum_components or minimum_components < 1:
        raise ValueError("A positive component count is required")
    names = _names(metric_names, 4)
    bounds = tuple(map(_rational, gain_bounds))
    margins = tuple(map(_declared_threshold, noninferiority_margins))
    eta, level = _declared_threshold(min_anchor_gain), _alpha(alpha)
    if len(bounds) != 4 or len(margins) != 4 or any(value < 0 for value in bounds + margins) or eta < 0:
        raise ValueError("Four nonnegative bounds and margins are required")
    targets = tuple(eta if e == anchor else -margins[e] for e in range(4))
    scales = tuple(bounds[e] + abs(targets[e]) for e in range(4))
    support_limits = tuple(_outward(bound, True) for bound in bounds)
    transforms = []
    for target, scale in zip(targets, scales):
        try:
            interval = (_outward(target, False), _outward(target, True),
                        _outward(scale, False), _outward(scale, True))
            transforms.append(interval if interval[2] > 0 and all(map(isfinite, interval)) else None)
        except OverflowError:
            transforms.append(None)
    if evidence_engine == "native":
        from .native_evidence import NativeGainIntervalMixture
        mixture_type = NativeGainIntervalMixture
    elif evidence_engine == "batch":
        from .batch_evidence import BatchGainIntervalMixture
        mixture_type = BatchGainIntervalMixture
    elif evidence_engine == "scalar":
        mixture_type = _GainIntervalMixture
    else:
        raise ValueError("A scalar, batch or native evidence engine is required")
    processes = [mixture_type(int(minimum_components)) for _ in range(4)]
    locked, crossing = [False] * 4, [None] * 4
    for start in range(0, int(row_cap), int(batch)):
        indices = np.arange(start, min(start + int(batch), int(row_cap)))
        lower, upper = reader.gain_intervals(indices, [not value for value in locked])
        if lower.shape != (len(indices), 4) or upper.shape != lower.shape:
            raise ValueError("The action reader returned an invalid gain enclosure")
        read = start + len(indices)
        for e in range(4):
            if locked[e]:
                continue
            lo = np.maximum(-support_limits[e], lower[:, e])
            hi = np.minimum(support_limits[e], upper[:, e])
            if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
                raise ValueError("Observed enclosure is disjoint from the finite population support")
            if transforms[e] is not None:
                qlo, qhi, slo, shi = transforms[e]
                nlo, nhi = np.nextafter(lo - qhi, -np.inf), np.nextafter(hi - qlo, np.inf)
                yl = np.maximum(-1., np.nextafter(nlo / np.where(nlo < 0, slo, shi), -np.inf))
                yu = np.minimum(1., np.nextafter(nhi / np.where(nhi > 0, slo, shi), np.inf))
                values = list(zip(map(float, yl), map(float, yu)))
            else:
                values = []
                for low, high in zip(lo, hi):
                    yl = (_rational(low) - targets[e]) / scales[e] if scales[e] else Fraction()
                    yu = (_rational(high) - targets[e]) / scales[e] if scales[e] else Fraction()
                    # Clip the rational enclosure before float conversion:
                    # an outward dyadic support can greatly exceed a tiny
                    # nonbinary scale, despite the true Y lying in [-1, 1].
                    values.append((_outward(max(Fraction(-1), yl), False),
                                   _outward(min(Fraction(1), yu), True)))
            processes[e].update(values)
            def exact_values(e=e):
                return [(gain - targets[e]) / scales[e] if scales[e] else Fraction()
                        for gain in reader.exact_prefix(e)]
            if processes[e].crossed(level, exact_values):
                locked[e], crossing[e] = True, read
        if all(locked):
            break
    admitted = all(locked)
    return {"admitted": admitted, "rows": read, "rows_read": read,
        "decision": "mixture" if admitted else "cap", "reason": "mixture" if admitted else "cap",
        "metric_names": list(names), "anchor": names[anchor],
        "noninferiority_margins": [float(v) for v in margins], "min_anchor_gain": float(eta),
        "gain_bounds": [float(v) for v in bounds], "exact_gain_bounds": list(map(str, bounds)),
        "exact_thresholds": list(map(str, targets)), "crossing_rows": crossing, "locked": locked,
        "mixture_components": [1 + len(p.small) for p in processes],
        "claim_scope": "the specified fresh-scene law under predeclared gain bounds",
        "query_stats": dict(reader.plan.stats), "reader_stats": dict(reader.stats),
        "exact_filter_fallbacks": sum(p.exact_fallbacks for p in processes)}
