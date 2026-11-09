"""Lift load enclosures through the original monotone stored-loss circuit."""
import numpy as np


def projection_raw_allowance(sizes, radii, capacity, horizon, demand_upper):
    """Frozen Lipschitz plus two stored-projection roundoff allowances."""
    from fractions import Fraction
    from collections import Counter
    from .action_query import _outward, _raw_error, _sum_error
    sizes, radii = tuple(map(int, sizes)), np.asarray(radii).reshape(-1)
    if len(sizes) != len(radii) or not sizes:
        raise ValueError("One radius for each complete host membership is required")
    cap, upper = Fraction(float(capacity)), Fraction(demand_upper)
    errors = {r: _sum_error(r, upper) for r in set(sizes)}
    largest = {}
    for r, rho in zip(sizes, radii):
        largest[r] = max(largest.get(r, 0.), float(rho))
    max_load = max(r * upper + errors[r] + Fraction(rho) for r, rho in largest.items())
    projection_error = _raw_error(((0,),) * len(sizes), len(sizes), horizon, cap, max_load)
    radius_sum = sum((count * Fraction(rho) for rho, count in Counter(map(float, radii)).items()), Fraction())
    allowance = radius_sum / (len(sizes) * cap) + 2 * projection_error
    return _outward(allowance, True)


def projection_late_raw_allowance(sizes, radii, capacity, horizon, demand_upper):
    """Exact frozen budget for clipping before a final mean/capacity division.

    The canonical target retains its original per-cell capacity division.
    Overflow of the uniform unnormalized bound requires the early consumer.
    """
    from fractions import Fraction
    from collections import Counter
    from math import isfinite
    from .action_query import _outward, _raw_error, _sum_error
    sizes, radii = tuple(map(int, sizes)), tuple(map(float, radii))
    if (not sizes or len(sizes) != len(radii) or not isfinite(capacity) or capacity <= 0
            or int(horizon) != horizon or horizon < 1
            or any(not isfinite(r) or r < 0 for r in radii)):
        raise ValueError("Complete finite load radii and positive normalization required")
    cap, upper = Fraction(float(capacity)), Fraction(demand_upper)
    terms = len(sizes) * int(horizon)
    unit, tiny = Fraction(1, 2**53), Fraction(1, 2**1075)
    errors = {r: _sum_error(r, upper) for r in set(sizes)}
    max_load = max(r * upper + errors[r] + Fraction(rho) for r, rho in zip(sizes, radii))
    projection_error = _raw_error(((0,),) * len(sizes), len(sizes), horizon, cap, max_load)
    clipped_max = max(max_load - cap, Fraction())
    clipped_error = unit * clipped_max + tiny
    summation = _sum_error(terms, clipped_max + clipped_error)
    sum_max = terms * (clipped_max + clipped_error) + summation
    total = terms * clipped_error + summation
    mean_error = (1 + unit) * total / terms + unit * clipped_max + tiny
    for value in (sum_max, (clipped_max + mean_error) / cap):
        if not isfinite(_outward(value, True)):
            raise OverflowError("The late projection requires finite uniform intermediates")
    late_error = (1 + unit) * mean_error / cap + unit * clipped_max / cap + tiny
    radius_sum = sum((count * Fraction(rho) for rho, count in Counter(radii).items()), Fraction())
    return _outward(radius_sum / (len(sizes) * cap) + projection_error + late_error, True)


class BoundedPairProjection:
    def __init__(self, capacity, horizon, old_radius, new_radius, *, tolerance=1e-12,
                 raw_allowance=None):
        self.capacity, self.horizon = float(capacity), int(horizon)
        self.threshold = self.capacity + tolerance
        self.radii = tuple(np.asarray(r, dtype=np.float64).reshape(-1, 1, 1)
                           for r in (old_radius, new_radius))
        self.hosts = len(self.radii[0])
        if (len(self.radii[1]) != self.hosts or self.hosts < 1
                or not np.isfinite(self.threshold) or self.capacity <= 0 or self.horizon < 1
                or any(not np.isfinite(r).all() or np.any(r < 0) for r in self.radii)):
            raise ValueError("Finite nonnegative load radii and fixed positive normalization required")
        self.points = tuple(not np.any(r) for r in self.radii)
        self.guards = tuple((np.where(r == 0, self.threshold,
                                      np.nextafter(self.threshold + r, np.inf)),
                             np.where(r == 0, self.threshold,
                                      np.nextafter(self.threshold - r, -np.inf)))
                            for r in self.radii)
        self.ambiguous_host_scenes = 0
        self.raw_allowance = None if raw_allowance is None else tuple(map(float, raw_allowance))
        if self.raw_allowance is not None and (len(self.raw_allowance) != 2
                or any(not np.isfinite(v) or v < 0 for v in self.raw_allowance)):
            raise ValueError("Two finite outward stored-raw allowances required")

    def _count_profile(self, bits, active):
        values = np.zeros((bits.shape[1], 3))
        occurrence = bits.any(axis=2) if any(active[:2]) else None
        if active[0]:values[:, 0] = occurrence.any(axis=0)
        if active[1]:values[:, 1] = occurrence.sum(axis=0) / self.hosts
        if active[2]:values[:, 2] = bits.sum(axis=(0, 2)) / (self.hosts * self.horizon)
        return values

    def _loss_bounds(self, center, side, active):
        lo, hi = np.zeros((center.shape[1], 4)), np.zeros((center.shape[1], 4))
        if any(active[:3]):
            if self.points[side]:
                low_bits = high_bits = center > self.threshold
                identical = True
            else:
                guard_hi, guard_lo = self.guards[side]
                low_bits, high_bits = center > guard_hi, center > guard_lo
                identical = np.array_equal(low_bits, high_bits)
            lo[:, :3] = self._count_profile(low_bits, active)
            hi[:, :3] = lo[:, :3] if identical else self._count_profile(high_bits, active)
            if not identical:
                self.ambiguous_host_scenes += int(np.count_nonzero((low_bits != high_bits).any(axis=2)))
        if active[3]:
            if self.points[side] or self.raw_allowance is not None:
                value = np.maximum(center / self.capacity - 1., 0.).sum(axis=(0, 2)) / (self.hosts * self.horizon)
                # A supplied aggregate allowance also covers a different
                # host summation order, including point-valued load views.
                allowance = self.raw_allowance[side] if self.raw_allowance is not None else 0.
                if allowance == 0:
                    lo[:, 3] = hi[:, 3] = value
                else:
                    lo[:, 3] = np.maximum(0., np.nextafter(value - allowance, -np.inf))
                    hi[:, 3] = np.nextafter(value + allowance, np.inf)
            else:
                radius = self.radii[side]
                lower = np.maximum(0., np.nextafter(center - radius, -np.inf))
                upper = np.nextafter(center + radius, np.inf)
                lo[:, 3] = np.maximum(lower / self.capacity - 1., 0.).sum(axis=(0, 2)) / (self.hosts * self.horizon)
                hi[:, 3] = np.maximum(upper / self.capacity - 1., 0.).sum(axis=(0, 2)) / (self.hosts * self.horizon)
        return lo, hi

    def evaluate(self, old_center, new_center, active=(True, True, True, True)):
        active = tuple(map(bool, active))
        old_center, new_center = np.asarray(old_center), np.asarray(new_center)
        if (len(active) != 4 or old_center.ndim != 3 or old_center.shape != new_center.shape
                or old_center.shape[0] != self.hosts or old_center.shape[2] != self.horizon):
            raise ValueError("Complete canonical host/time arrays and four live flags required")
        old_lo, old_hi = self._loss_bounds(old_center, 0, active)
        new_lo, new_hi = self._loss_bounds(new_center, 1, active)
        lower = np.nextafter(old_lo - new_hi, -np.inf)
        upper = np.nextafter(old_hi - new_lo, np.inf)
        zero = (old_lo == old_hi) & (new_lo == new_hi) & (old_lo == new_lo)
        lower[zero] = upper[zero] = 0.
        lower[:, 0], upper[:, 0] = old_lo[:, 0] - new_hi[:, 0], old_hi[:, 0] - new_lo[:, 0]
        return lower, upper
