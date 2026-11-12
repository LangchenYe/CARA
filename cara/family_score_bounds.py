"""Conservative native-profile bounds for complete two-host mask subtrees.

This module supplies an *optimistic key*, not an approximate candidate score.
Only a strict comparison with the worst key of an already full balanced/gain
pool can exclude a subtree.  Unsupported inputs return ``None`` and require
the ordinary exact reader.  Native idle sentinels, overflowing reductions and
an uncertain denominator are never silently replaced by a zero loss.

Common resident groups reuse canonical projections.  The two changing groups
use nonnegative sum enclosures and the conservation of their combined demand.
All bounds include arbitrary binary64 reduction trees; they do not assume a
subtraction update has the same bits as a new canonical resident reduction.
"""
from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
from functools import lru_cache
from math import inf, isfinite, nextafter

import numpy as np

from .data import canonical_assignment, host_loads, host_memberships
from .gain_circuit import GainsContext
from .workload_views import host_projection


PROFILE_NAMES = ("J", "V", "VR", "raw", "VS", "PAR")
_U = Fraction(1, 1 << 53)
_ETA = Fraction(1, 1 << 1074)
_QUANTUM = 1 << 40


def should_bound_profiles(config, hosts, horizon, edited_residents):
    """Choose an exact physical plan using only query shape, never outcomes."""
    minimum = int(config.get("family_score_min_hosts", 0))
    if minimum <= 0:
        return True
    if not config.get("family_structural_bound_gate", False):
        return hosts >= minimum
    work = hosts + horizon * edited_residents
    return work >= int(config.get("family_bound_work_min", 192))


def _outward(value, upward):
    result = float(value)
    if not isfinite(result):
        raise OverflowError("The bound exceeds the finite binary64 range")
    if (Fraction(result) < value if upward else Fraction(result) > value):
        result = nextafter(result, inf if upward else -inf)
    if not isfinite(result):
        raise OverflowError("The outward bound exceeds the finite binary64 range")
    return result


@lru_cache(maxsize=256)
def _reduction(count):
    """Outward (1-gamma, 1+gamma, delta) for a nonnegative sum."""
    if not 0 <= count <= 2**50:
        raise ValueError("A certified reduction requires at most 2**50 terms")
    if count < 2:
        return 1., 1., 0.
    steps = count - 1
    denominator = 1 - steps * _U
    gamma, delta = steps * _U / denominator, steps * _ETA / denominator
    if gamma >= 1:
        raise ValueError("A positive lower reduction factor is required")
    return (_outward(1 - gamma, False), _outward(1 + gamma, True),
            _outward(delta, True))


def _down(value):
    return np.maximum(np.nextafter(value, -np.inf), 0.)


def _add_lo(left, right):
    return _down(np.asarray(left) + np.asarray(right))


def _add_hi(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return np.where((left == 0) & (right == 0), 0.,
                    np.nextafter(left + right, np.inf))


def _mul_hi(left, right):
    left, right = np.asarray(left), np.asarray(right)
    # Positive underflow must produce eta, not an unsound zero upper bound.
    return np.where((left == 0) | (right == 0), 0.,
                    np.nextafter(left * right, np.inf))


def _div_hi(numerator, denominator):
    numerator = np.asarray(numerator)
    return np.where(numerator == 0, 0.,
                    np.nextafter(numerator / denominator, np.inf))


def _real_lower(stored, count):
    """Infer a real sum lower bound from a rounded nonnegative sum."""
    if count < 2:
        return np.asarray(stored)
    _, growth, delta = _reduction(count)
    return _down(_down(np.asarray(stored) - delta) / growth)


def _real_upper(stored, count):
    if count < 2:
        return np.asarray(stored)
    shrink, _, delta = _reduction(count)
    stored = np.asarray(stored)
    result = _div_hi(_add_hi(stored, delta), shrink)
    # A sum of nonnegative stored operands is zero iff every operand is zero.
    return np.where(stored == 0, 0., result)


def _stored_lower(real_lower, count):
    if count < 2:
        return np.asarray(real_lower)
    shrink, _, delta = _reduction(count)
    return _down(_down(np.asarray(real_lower) * shrink) - delta)


def _stored_upper(real_upper, count):
    if count < 2:
        return np.asarray(real_upper)
    _, growth, delta = _reduction(count)
    real_upper = np.asarray(real_upper)
    return np.where(real_upper == 0, 0.,
                    _add_hi(_mul_hi(real_upper, growth), delta))


def _sum_real_lower(values, axis, count):
    return _real_lower(np.sum(values, axis=axis), count)


def _sum_real_upper(values, axis, count):
    return _real_upper(np.sum(values, axis=axis), count)


class FamilyScoreContext:
    """Reusable score data for one validated immutable shared-query source.

    A SharedQueryReplay registry owner validates its finite, nonnegative panel
    when it is created.  Readers sharing that owner require the same panel
    object and capacity.  This context reuses precisely that contract; it does
    not expose a caller-supplied flag for bypassing source validation.  As with
    the reader's resident views, the source must remain immutable for the
    session.  Family-dependent seed loads and sentinel guards are still checked
    by every FamilyScoreBounds instance.

    The selected reference operands and original-layout scale are snapshotted
    once.  Standalone FamilyScoreBounds construction retains full validation.
    """

    def __init__(self, reader, reference_profile, *, objective_names=PROFILE_NAMES, compiled_views=False,
                 compiled_subsets=False, compiled_bounds=False):
        from .query_engine import SharedQueryReplay

        if not isinstance(reader, SharedQueryReplay):
            raise TypeError("Score context requires a validated SharedQueryReplay reader")
        owner = reader._registry
        if (not isinstance(owner, SharedQueryReplay) or owner._registry is not owner
                or reader.panel is not owner.panel or reader.capacity != owner.capacity):
            raise ValueError("Score context requires one immutable panel and capacity")
        self.panel, self.capacity = owner.panel, float(owner.capacity)
        self.objective_names = tuple(objective_names)
        reference = np.asarray(reference_profile, dtype=np.float64)
        if reference.shape != (self.panel.shape[0], 6):
            raise ValueError("Expected a source-aligned B-by-6 reference profile")
        # Construct before copying so reference.mean(axis=0) sees its original
        # strides, exactly as in refinement._relative_gains.
        self.gains = GainsContext(reference, self.objective_names)
        self.reference = np.array(reference, copy=True, order="K")
        self.reference.setflags(write=False)
        self.reference_finite = bool(np.isfinite(reference).all())
        self._registry = owner
        self.compiled_views = bool(compiled_views)
        self.compiled_subsets = bool(compiled_subsets)
        self.compiled_bounds = bool(compiled_bounds)
        self._bound_index = None
        self.view_fallbacks = 0

    @classmethod
    def from_reader(cls, reader, reference_profile, *, objective_names=PROFILE_NAMES, compiled_views=False,
                    compiled_subsets=False, compiled_bounds=False):
        return cls(reader, reference_profile, objective_names=objective_names, compiled_views=compiled_views,
                   compiled_subsets=compiled_subsets, compiled_bounds=compiled_bounds)

    @classmethod
    def from_registry(cls, registry, reference_profile, *, objective_names=PROFILE_NAMES, compiled_views=False,
                      compiled_subsets=False, compiled_bounds=False):
        """Accept the shared reader registry owner, using the same validation."""
        return cls(registry, reference_profile, objective_names=objective_names, compiled_views=compiled_views,
                   compiled_subsets=compiled_subsets, compiled_bounds=compiled_bounds)

    def for_family(self, seed, pair, mutable, *, seed_loads=None):
        view_index = None
        if self.compiled_views and seed_loads is None:
            from .bound_views import BoundViewIndex
            try:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                    if self._bound_index is None:
                        self._bound_index = BoundViewIndex(self._registry, seed)
                    else:
                        self._bound_index.update(seed)
                view_index = self._bound_index
            except ArithmeticError:
                self._bound_index = None
                self.view_fallbacks += 1
                seed_loads = self._registry.loads(seed)
        return FamilyScoreBounds(self.panel, seed, pair, mutable, self.capacity,
            self.reference, objective_names=self.objective_names,
            seed_loads=seed_loads, compiled_bounds=self.compiled_bounds, _context=self, _view_index=view_index)

    def view_diagnostics(self):
        index = self._bound_index
        return {"fallbacks": self.view_fallbacks, **({} if index is None else index.stats),
                "stored_bytes": 0 if index is None else index.stored_bytes}


class FamilyScoreBounds:
    """Batched optimistic balanced-gain keys for aligned dyadic mask ranges.

    ``seed_loads``, when supplied, must be the ordinary canonical H-by-B-by-T
    loads for this seed and panel.  A mask bit of zero routes its VM left.  If
    both fixed backgrounds are empty, mutable[0] is fixed left, exactly as in
    the original residual family.  Only complete nonempty-host descendants
    are covered.  Movement/cost lower bounds are supplied by the caller.

    This first implementation deliberately declines an entire family if any
    scene is globally idle, or if any required finite arithmetic enclosure is
    unavailable.  In particular VR/PAR=-1 is not treated as a nonnegative loss.
    """

    def __init__(self, panel, seed, pair, mutable, capacity, reference_profile,
                 *, objective_names=PROFILE_NAMES, seed_loads=None, compiled_bounds=False,
                 _context=None, _view_index=None):
        self.available = False
        self.disabled_reason = None
        self.bound_queries = 0
        self.panel = np.asarray(panel, dtype=np.float64)
        self.seed = canonical_assignment(seed)
        self.pair, self.mutable = tuple(map(int, pair)), tuple(map(int, mutable))
        self.capacity = float(capacity)
        self.reference = np.asarray(reference_profile, dtype=np.float64)
        self.objective_names = tuple(objective_names)
        self.compiled_bounds = bool(compiled_bounds)
        if _context is not None and (not isinstance(_context, FamilyScoreContext)
                or self.panel is not _context.panel or self.capacity != _context.capacity
                or self.reference is not _context.reference
                or self.objective_names != _context.objective_names
                or self.compiled_bounds != _context.compiled_bounds):
            raise ValueError("Family score context and immutable source disagree")
        if _view_index is not None:
            from .bound_views import BoundViewIndex
            if (not isinstance(_view_index, BoundViewIndex) or _context is None
                    or _view_index.reader is not _context._registry or _view_index.seed != self.seed):
                raise ValueError("Certificate views require the same validated source and seed")
        self._view_index = _view_index
        self.groups = host_memberships(self.seed) if _view_index is None else _view_index.groups
        self.hosts = len(self.groups)
        if (self.panel.ndim != 3 or self.panel.shape[1] != len(self.seed)
                or not self.panel.shape[0] or not self.panel.shape[2]
                or self.hosts < 2 or len(self.pair) != 2 or len(set(self.pair)) != 2
                or any(h < 0 or h >= self.hosts for h in self.pair)
                or not self.mutable or self.mutable != tuple(sorted(set(self.mutable)))
                or any(v < 0 or v >= len(self.seed) or self.seed[v] not in self.pair
                       for v in self.mutable)
                or self.reference.shape != (self.panel.shape[0], 6)
                or not self.objective_names
                or any(name not in PROFILE_NAMES for name in self.objective_names)):
            raise ValueError("Expected a complete native two-host family and B-by-6 reference")
        self.scenes, _, self.horizon = self.panel.shape
        mutable_set = set(self.mutable)
        self.backgrounds = tuple(tuple(v for v in self.groups[h] if v not in mutable_set)
                                 for h in self.pair)
        self.symmetric = not any(self.backgrounds)
        self.free_vms = self.mutable[int(self.symmetric):]
        self.bits, self.mask_count = len(self.free_vms), 1 << len(self.free_vms)
        self.resident_bound = sum(len(self.groups[h]) for h in self.pair)
        self._cache = OrderedDict()
        self.bound_batch_values = 1 << 20
        self.bound_cache_values = 1 << 20
        self.mutable_sum_index = None
        source_valid = (_context is not None or
                        (np.isfinite(self.panel).all() and not np.any(self.panel < 0)))
        reference_valid = (_context.reference_finite if _context is not None
                           else np.isfinite(self.reference).all())
        if (not isfinite(self.capacity) or self.capacity <= 0
                or not source_valid or not reference_valid
                or max(len(self.seed), self.horizon, self.scenes,
                       self.hosts * self.horizon) > 2**50
                or self.bits > 52):
            self.disabled_reason = "unsupported source, capacity, or reduction range"
            return
        self.gains_context = (_context.gains if _context is not None
                              else GainsContext(self.reference, self.objective_names))
        if not np.isfinite(self.gains_context.scale).all():
            self.disabled_reason = "nonfinite native reference scale requires exact refinement"
            return
        try:
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                self._prepare(seed_loads)
        except (OverflowError, ValueError):
            self.disabled_reason = "unavailable finite reduction enclosure"
            return
        if (self.available and _context is not None and _context.compiled_subsets
                and len(self.mutable) <= 52):
            from .subset_certificates import MutableSumIndex
            self.mutable_sum_index = MutableSumIndex(self.mutable_values, symmetric=self.symmetric)

    def _prepare(self, seed_loads):
        if self._view_index is None:
            loads = host_loads(self.panel, self.seed) if seed_loads is None else np.asarray(seed_loads, dtype=float)
            if loads.shape != (self.hosts, self.scenes, self.horizon):
                raise ValueError("seed_loads must be the canonical H-by-B-by-T seed loads")
            if not np.isfinite(loads).all() or np.any(loads < 0):
                self.disabled_reason = "nonfinite canonical seed loads"
                return
            if np.any(~np.any(loads != 0, axis=(0, 2))):
                self.disabled_reason = "native idle sentinel requires exact refinement"
                return
            common = [h for h in range(self.hosts) if h not in self.pair]
            projected = host_projection(loads[common], self.capacity)
            active, tolerant, affected, frequency, excess, severity, peaks = projected
            self.common_active = active.sum(axis=0)
            self.common_tolerant = tolerant.sum(axis=0)
            self.common_affected = affected.sum(axis=0)
            self.common_native_count = (loads[common] > self.capacity).sum(axis=(0, 2))
            count = len(common)
            self.common_frequency_lo = _sum_real_lower(frequency, 0, count)
            self.common_excess_lo = _sum_real_lower(excess, 0, count)
            self.common_severity_lo = _sum_real_lower(severity, 0, count)
            self.common_peak_hi = _sum_real_upper(peaks, 0, count)
            self.common_peak_max = peaks.max(axis=0, initial=0.)
            self.common_ratio_max = (loads[common] / self.capacity).max(initial=0.)
            pair_loads = loads[list(self.pair)]
        else:
            common = self._view_index.common(self.pair)
            if not common["all_scenes_active"]:
                self.disabled_reason = "native idle sentinel requires exact refinement"
                return
            for name in ("active", "tolerant", "affected", "native_count", "frequency_lo",
                         "excess_lo", "severity_lo", "peak_hi", "peak_max"):
                setattr(self, "common_" + name, common[name])
            self.common_ratio_max = (self.common_peak_max / self.capacity).max(initial=0.)
            pair_loads = common["loads"]
        self.mutable_values = np.ascontiguousarray(self.panel[:, self.mutable, :].transpose(1, 0, 2))
        self.mutable_flat = self.mutable_values.reshape(len(self.mutable), -1)
        self.mutable_peaks = self.mutable_values.max(axis=2)
        self.background_lo, self.background_hi = [], []
        seed_real_lo, seed_real_hi = [], []
        for side, h in enumerate(self.pair):
            lo = _real_lower(pair_loads[side], len(self.groups[h]))
            hi = _real_upper(pair_loads[side], len(self.groups[h]))
            seed_real_lo.append(lo)
            seed_real_hi.append(hi)
            residents = [j for j, vm in enumerate(self.mutable) if self.seed[vm] == h]
            removed = self.mutable_values[residents].sum(axis=0)
            removed_lo = _real_lower(removed, len(residents))
            removed_hi = _real_upper(removed, len(residents))
            if not self.backgrounds[side]:
                self.background_lo.append(np.zeros_like(lo))
                self.background_hi.append(np.zeros_like(hi))
            else:
                self.background_lo.append(_down(lo - removed_hi))
                bg_hi = np.maximum(np.nextafter(hi - removed_lo, np.inf), 0.)
                self.background_hi.append(np.where(pair_loads[side] == 0, 0., bg_hi))
        self.background_lo, self.background_hi = np.stack(self.background_lo), np.stack(self.background_hi)
        pair_real_lo = _add_lo(seed_real_lo[0], seed_real_lo[1])
        self.pair_load_upper = _stored_upper(_add_hi(seed_real_hi[0], seed_real_hi[1]), self.resident_bound)
        shrink, _, delta = _reduction(self.resident_bound)
        self.pair_load_sum_lo = _down(_down(pair_real_lo * shrink) - 2. * delta)
        self.pair_nonzero = np.any(pair_loads != 0, axis=(0, 2))
        self.tolerant_capacity = self.capacity + 1e-12
        # Required upper enclosures must remain finite, including raw divisions.
        required = (self.background_hi, self.common_peak_hi, self.common_excess_lo,
                    self.common_severity_lo, self.mutable_values)
        if not all(np.isfinite(value).all() for value in required):
            self.disabled_reason = "nonfinite projection or background enclosure"
            return
        self.available = True

    def _range(self, start, stop):
        if int(start) != start or int(stop) != stop:
            raise ValueError("A subtree has integer mask endpoints")
        start, stop = int(start), int(stop)
        span = stop - start
        if span <= 0 or span & (span - 1) or start < 0 or stop > self.mask_count or start % span:
            raise ValueError("A subtree must be an aligned dyadic mask interval")
        return start, stop, span.bit_length() - 1

    def _load_enclosures(self, ranges, free_counts=None, *, consumer="full"):
        q, w = len(ranges), len(self.mutable)
        if free_counts is None:
            free_counts = [self._range(start, stop)[2] for start, stop in ranges]
        if self.compiled_bounds:
            from .load_certificates import leaf_load_enclosures
            compiled = leaf_load_enclosures(self, ranges)
            suffix = "fast_nodes" if compiled is not None else "fallback_nodes"
            name = consumer + "_uniform_" + suffix
            setattr(self, name, getattr(self, name, 0) + q)
            if compiled is not None:
                return compiled[0], compiled[1], None
        if self.mutable_sum_index is None:
            mandatory = np.zeros((q, 2, w))
            possible = np.zeros_like(mandatory)
            for row, (start, stop) in enumerate(ranges):
                free = free_counts[row]
                offset = int(self.symmetric)
                if offset:
                    mandatory[row, 0, 0] = possible[row, 0, 0] = 1.
                for bit in range(self.bits):
                    j = bit + offset
                    if bit < free:
                        possible[row, :, j] = 1.
                    else:
                        side = (start >> bit) & 1
                        mandatory[row, side, j] = possible[row, side, j] = 1.
            selectors = np.concatenate((mandatory.reshape(-1, w), possible.reshape(-1, w)))
            sums = (selectors @ self.mutable_flat).reshape(2, q, 2, self.scenes, self.horizon)
        else:
            sums = self.mutable_sum_index.sum_ranges(ranges)
        mandatory_lo = _add_lo(_real_lower(sums[0], w), self.background_lo)
        mandatory_hi = _add_hi(_real_upper(sums[0], w), self.background_hi)
        possible_hi = _add_hi(_real_upper(sums[1], w), self.background_hi)
        lo = _stored_lower(mandatory_lo, self.resident_bound)
        hi = _stored_upper(possible_hi, self.resident_bound)
        return lo, hi, mandatory_hi

    def _lower_batch(self, ranges):
        q = len(ranges)
        free_counts = [self._range(start, stop)[2] for start, stop in ranges]
        lo, hi, mandatory_hi = self._load_enclosures(ranges, free_counts)

        native_lo, native_hi = lo > self.capacity, hi > self.capacity
        tolerant_lo = lo > self.tolerant_capacity
        pair_native = np.maximum(native_lo.sum(axis=1),
                                 self.pair_load_sum_lo > 2. * self.capacity)
        pair_tolerant_hosts = np.maximum(tolerant_lo.any(axis=3).sum(axis=1),
            (self.pair_load_sum_lo > 2. * self.tolerant_capacity).any(axis=1))
        tolerant_count = self.common_tolerant + pair_tolerant_hosts
        j_loss = tolerant_count > 0
        v_loss = tolerant_count / float(self.hosts)
        active_max = self.common_active + np.any(hi > 0, axis=3).sum(axis=1)
        pair_active_min = np.maximum(np.any(lo > 0, axis=3).sum(axis=1), self.pair_nonzero)
        active_min = self.common_active + pair_active_min
        affected_max = self.common_affected + native_hi.any(axis=3).sum(axis=1)

        total_native_count = self.common_native_count + pair_native.sum(axis=2)
        if self.horizon & (self.horizon - 1) == 0:
            # Dyadic frequencies and their H-fold sum are exact in this range.
            vr_loss = (total_native_count / float(self.horizon)) / active_max
        else:
            # Sum_i RN(n_i/T) >= (1-u) sum_i n_i/T - H*eta.
            frequency_real_lo = _down(_down(total_native_count / float(self.horizon))
                                      * float(1 - _U))
            frequency_real_lo = _down(frequency_real_lo - _outward(self.hosts * _ETA, True))
            vr_loss = _stored_lower(frequency_real_lo, self.hosts) / active_max

        individual_excess = np.maximum(lo / self.capacity - 1., 0.)
        time_real_lo = _sum_real_lower(individual_excess, 3, self.horizon)
        time_sum_lo = _stored_lower(time_real_lo, self.horizon)
        individual_mean_lo = time_sum_lo / float(self.horizon)
        pair_mean_lo = _add_lo(individual_mean_lo[:, 0], individual_mean_lo[:, 1])

        # Two-host conservation supplies excess even when neither mandatory
        # group alone already overloads.  See the companion derivation.
        combined_ratio_lo = _down(self.pair_load_sum_lo / self.capacity)
        coupled_excess = _down(_down(combined_ratio_lo * float(1 - 2 * _U)) - 2.)
        per_time_lo = np.maximum(_add_lo(individual_excess[:, 0], individual_excess[:, 1]),
                                 coupled_excess)
        pair_time_real_lo = _sum_real_lower(per_time_lo, 2, self.horizon)
        shrink_t, _, delta_t = _reduction(self.horizon)
        pair_time_stored_lo = _down(_down(pair_time_real_lo * shrink_t) - 2. * delta_t)
        coupled_mean_lo = _down(_down(pair_time_stored_lo / float(self.horizon))
                                * float(1 - _U))
        coupled_mean_lo = _down(coupled_mean_lo - float(2 * _ETA))
        pair_mean_lo = np.maximum(pair_mean_lo, coupled_mean_lo)
        raw_sum_lo = _stored_lower(_add_lo(self.common_excess_lo, pair_mean_lo), self.hosts)
        raw_loss = raw_sum_lo / active_max

        possible_times = native_hi.sum(axis=3)
        severity_lo = np.divide(time_sum_lo, possible_times, out=np.zeros_like(time_sum_lo),
                                where=possible_times > 0)
        pair_severity_lo = np.maximum(_add_lo(severity_lo[:, 0], severity_lo[:, 1]), pair_mean_lo)
        severity_sum_lo = _stored_lower(_add_lo(self.common_severity_lo, pair_severity_lo), self.hosts)
        vs_loss = np.divide(severity_sum_lo, affected_max, out=np.zeros_like(severity_sum_lo),
                            where=affected_max > 0)

        peak_lo, peak_hi = lo.max(axis=3), hi.max(axis=3)
        largest_peak_lo = np.maximum(self.common_peak_max, peak_lo.max(axis=1))
        largest_peak_lo = np.maximum(largest_peak_lo, _down(self.pair_load_sum_lo.max(axis=1) / 2.))
        pair_peak_hi = _add_hi(peak_hi[:, 0], peak_hi[:, 1])
        # Unresolved VM peaks are counted once across the two changing hosts.
        if mandatory_hi is not None:
            mandatory_peak_hi = _add_hi(mandatory_hi[:, 0].max(axis=2), mandatory_hi[:, 1].max(axis=2))
            free_peak_hi = np.zeros((q, self.scenes))
            for free in set(free_counts):
                start = int(self.symmetric)
                value = _sum_real_upper(self.mutable_peaks[start:start + free], 0, free)
                free_peak_hi[np.asarray(free_counts) == free] = value
            conserved_peak_hi = _add_hi(mandatory_peak_hi, free_peak_hi)
            _, growth, delta = _reduction(self.resident_bound)
            conserved_peak_hi = _add_hi(_mul_hi(conserved_peak_hi, growth), 2. * delta)
            pair_peak_hi = np.minimum(pair_peak_hi, conserved_peak_hi)
        peak_sum_hi = _stored_upper(_add_hi(self.common_peak_hi, pair_peak_hi), self.hosts)
        mean_peak_hi = peak_sum_hi / active_min
        # A globally nonzero scene can nevertheless have a stored mean peak
        # rounded to zero in the subnormal range, triggering native PAR=-1.
        # The host sum is at least its largest operand, so this lower bound
        # proves that the original mean-peak division has a positive result.
        mean_peak_lo = largest_peak_lo / active_max
        par_loss = largest_peak_lo / mean_peak_hi
        result = np.stack((j_loss, v_loss, vr_loss, raw_loss, vs_loss, par_loss), axis=2)
        safe = (np.isfinite(result).all(axis=(1, 2)) & np.isfinite(hi).all(axis=(1, 2, 3))
                & np.isfinite(mean_peak_hi).all(axis=1) & (mean_peak_hi > 0).all(axis=1)
                & (mean_peak_lo > 0).all(axis=1)
                & (active_min > 0).all(axis=1) & (active_max > 0).all(axis=1))
        # Also exclude finite-load cases whose native excess reductions could
        # overflow.  The factor four dominates every sum-growth allowance in
        # this module's bounded-count scope.
        ratio_max = np.maximum((hi / self.capacity).max(axis=(1, 2, 3)), self.common_ratio_max)
        safe &= ratio_max <= np.finfo(float).max / (4. * self.hosts * self.horizon)
        return tuple(np.array(result[i], copy=True, order="C") if safe[i] else None for i in range(q))

    def diagnostics(self):
        return {name: int(getattr(self, name, 0)) for name in (
            "uniform_radius_compilations", "full_uniform_fast_nodes", "full_uniform_fallback_nodes",
            "aggregate_uniform_fast_nodes", "aggregate_uniform_fallback_nodes")}

    def lower_profiles(self, ranges):
        """Return B-by-6 loss lower bounds, or None for an uncertified range."""
        ranges = tuple(self._range(start, stop)[:2] for start, stop in ranges)
        if not self.available:
            return (None,) * len(ranges)
        missing = tuple(dict.fromkeys(pair for pair in ranges if pair not in self._cache))
        local = {pair: self._cache[pair] for pair in ranges if pair in self._cache}
        if missing:
            self.bound_queries += len(missing)
            limit = max(1, self.bound_batch_values // (2 * self.scenes * self.horizon))
            cache_limit = max(1, self.bound_cache_values // (6 * self.scenes))
            for start in range(0, len(missing), limit):
                chunk = missing[start:start + limit]
                with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                    values = self._lower_batch(chunk)
                local.update(zip(chunk, values))
                self._cache.update(zip(chunk, values))
                while len(self._cache) > cache_limit:
                    self._cache.popitem(last=False)
        return tuple(local[pair] for pair in ranges)

    def upper_keys(self, ranges, movement_lower_bounds=None, cost_lower_bounds=None):
        """Use the native gain circuit, unchanged quantization, and sorting."""
        ranges = tuple(ranges)
        movement = (0,) * len(ranges) if movement_lower_bounds is None else tuple(movement_lower_bounds)
        costs = (0.,) * len(ranges) if cost_lower_bounds is None else tuple(cost_lower_bounds)
        if len(movement) != len(ranges) or len(costs) != len(ranges):
            raise ValueError("Each range needs one movement and cost lower bound")
        output = [None] * len(ranges)
        limit = max(1, self.bound_batch_values // (2 * self.scenes * self.horizon))
        for start in range(0, len(ranges), limit):
            stop = min(start + limit, len(ranges))
            lower = self.lower_profiles(ranges[start:stop])
            eligible = [i for i, (profile, moved, cost) in enumerate(zip(lower, movement[start:stop], costs[start:stop]))
                        if profile is not None and isfinite(moved) and isfinite(cost)
                        and moved >= 0 and cost >= 0]
            if eligible:
                profiles = np.stack([lower[i] for i in eligible])
                with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                    gains = self.gains_context.batch(profiles)
                    quantized = np.floor(gains * _QUANTUM)
                for i, row in zip(eligible, quantized):
                    if np.isfinite(row).all():
                        j = start + i
                        output[j] = tuple(sorted(map(int, row))) + (-movement[j], -costs[j])
        return tuple(output)

    def upper_key(self, start, stop, movement_lower_bound=0, cost_lower_bound=0):
        return self.upper_keys(((start, stop),), (movement_lower_bound,), (cost_lower_bound,))[0]
