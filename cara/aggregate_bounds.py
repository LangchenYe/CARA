"""Inexpensive native-profile envelopes before full bound acquisition."""
import numpy as np

from .family_score_bounds import (_ETA, _U, _add_hi, _add_lo, _down, _mul_hi,
    _outward, _reduction, _stored_lower, _stored_upper, _sum_real_lower, _sum_real_upper)
from .metric_cascade import IndicatorBounds


class AggregateBounds(IndicatorBounds):
    """Share load endpoints while avoiding candidate time-excess tensors."""

    def __init__(self, full_bounds):
        super().__init__(full_bounds)
        self._coupled_mean_lower = np.zeros(self.scenes)
        if not self.available:
            return
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            ratio = self.pair_load_sum_lo / self.capacity
            if not np.isfinite(ratio).all():
                return
            coupled = _down(_down(_down(ratio) * float(1 - 2 * _U)) - 2.)
            real_time = _sum_real_lower(coupled, 1, self.horizon)
            shrink, _, delta = _reduction(self.horizon)
            stored_time = _down(_down(real_time * shrink) - 2. * delta)
            mean = _down(_down(stored_time / float(self.horizon)) * float(1 - _U))
            mean = _down(mean - float(2 * _ETA))
        if all(np.isfinite(value).all() for value in (coupled, real_time, stored_time, mean)):
            self._coupled_mean_lower = mean

    def _lower_batch(self, ranges):
        q = len(ranges)
        free_counts = [self.full._range(start, stop)[2] for start, stop in ranges]
        lo, hi, mandatory_hi = self.full._load_enclosures(ranges, free_counts, consumer="aggregate")
        tolerant_lo = lo > self.tolerant_capacity
        tolerant_pair = np.maximum(tolerant_lo.any(axis=3).sum(axis=1),
            (self.pair_load_sum_lo > 2. * self.tolerant_capacity).any(axis=1))
        count = self.common_tolerant + tolerant_pair
        active_max = self.common_active + np.any(hi > 0, axis=3).sum(axis=1)
        active_min = self.common_active + np.maximum(np.any(lo > 0, axis=3).sum(axis=1), self.pair_nonzero)
        peak_lo, peak_hi = lo.max(axis=3), hi.max(axis=3)

        native_count = self.common_native_count + np.maximum(tolerant_lo.sum(axis=1),
            self.pair_load_sum_lo > 2. * self.capacity).sum(axis=2)
        if self.horizon & (self.horizon - 1) == 0:
            vr = (native_count / float(self.horizon)) / active_max
        else:
            frequency = _down(_down(native_count / float(self.horizon)) * float(1 - _U))
            frequency = _down(frequency - _outward(self.hosts * _ETA, True))
            vr = _stored_lower(frequency, self.hosts) / active_max

        ratio = peak_lo / self.capacity
        excess_peak = _down(np.maximum(_down(ratio) - 1., 0.))
        excess_peak = np.where(np.isfinite(ratio), excess_peak, 0.)
        mean_lower = excess_peak / float(self.horizon)
        possible_times = (hi > self.capacity).sum(axis=3)
        severity_lower = excess_peak / np.maximum(possible_times, 1)
        pair_mean = np.maximum(self._coupled_mean_lower, _add_lo(mean_lower[:, 0], mean_lower[:, 1]))
        pair_severity = np.maximum(self._coupled_mean_lower,
                                  _add_lo(severity_lower[:, 0], severity_lower[:, 1]))
        raw = _stored_lower(_add_lo(self.common_excess_lo, pair_mean), self.hosts) / active_max
        affected_max = self.common_affected + (peak_hi > self.capacity).sum(axis=1)
        severity_sum = _stored_lower(_add_lo(self.common_severity_lo, pair_severity), self.hosts)
        severity = np.divide(severity_sum, affected_max, out=np.zeros_like(severity_sum), where=affected_max > 0)

        largest_peak = np.maximum(self.common_peak_max, peak_lo.max(axis=1))
        largest_peak = np.maximum(largest_peak, _down(self.pair_load_sum_lo.max(axis=1) / 2.))
        pair_peak = _add_hi(peak_hi[:, 0], peak_hi[:, 1])
        if mandatory_hi is not None:
            mandatory_peak = _add_hi(mandatory_hi[:, 0].max(axis=2), mandatory_hi[:, 1].max(axis=2))
            free_peak = np.zeros((q, self.scenes))
            for free in set(free_counts):
                start = int(self.symmetric)
                free_peak[np.asarray(free_counts) == free] = _sum_real_upper(self.mutable_peaks[start:start + free], 0, free)
            _, growth, delta = _reduction(self.resident_bound)
            conserved = _add_hi(_mul_hi(_add_hi(mandatory_peak, free_peak), growth), 2. * delta)
            pair_peak = np.minimum(pair_peak, conserved)
        peak_sum = _stored_upper(_add_hi(self.common_peak_hi, pair_peak), self.hosts)
        mean_peak_hi = peak_sum / active_min
        mean_peak_lo = largest_peak / active_max
        par = largest_peak / mean_peak_hi
        finite_hi = np.isfinite(hi).all(axis=(1, 2, 3))
        par_valid = (finite_hi & np.isfinite(par).all(axis=1) & np.isfinite(mean_peak_hi).all(axis=1)
            & (mean_peak_hi > 0).all(axis=1) & (mean_peak_lo > 0).all(axis=1)
            & (active_min > 0).all(axis=1) & (active_max > 0).all(axis=1))
        ratio_max = np.maximum(peak_hi.max(axis=(1, 2)) / self.capacity, self.common_ratio_max)
        par_valid &= ratio_max <= np.finfo(float).max / (4. * self.hosts * self.horizon)
        result = np.zeros((q, self.scenes, 6))
        result[:, :, 0] = count > 0
        result[:, :, 1] = count / float(self.hosts)
        result[:, :, 2] = np.where(np.isfinite(vr) & (vr >= 0), vr, -1.)
        result[:, :, 3] = np.where(np.isfinite(raw) & (raw >= 0), raw, 0.)
        result[:, :, 4] = np.where(np.isfinite(severity) & (severity >= 0), severity, 0.)
        result[:, :, 5] = -1.
        result[par_valid, :, 5] = par[par_valid]
        valid = np.isfinite(lo).all(axis=(1, 2, 3)) & finite_hi
        return tuple(np.array(row, copy=True, order="C") if good else None for row, good in zip(result, valid))

    def lower_profiles(self, ranges):
        ranges = tuple(self.full._range(start, stop)[:2] for start, stop in ranges)
        if not self.available:
            return (None,) * len(ranges)
        local = {pair: self._cache[pair] for pair in ranges if pair in self._cache}
        missing = tuple(dict.fromkeys(pair for pair in ranges if pair not in self._cache))
        limit = max(1, (1 << 20) // (2 * self.scenes * self.horizon))
        for start in range(0, len(missing), limit):
            chunk = missing[start:start + limit]
            with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
                values = self._lower_batch(chunk)
            for pair, value in zip(chunk, values):
                local[pair] = value
                cells = 0 if value is None else value.size
                if cells <= self._cache_limit:
                    self._cache[pair] = value
                    self._cache_cells += cells
                while self._cache_cells > self._cache_limit or len(self._cache) > 512:
                    _, evicted = self._cache.popitem(last=False)
                    self._cache_cells -= 0 if evicted is None else evicted.size
            self.bound_queries += len(chunk)
        return tuple(local[pair] for pair in ranges)
