"""Low-cost endpoint certificates before acquiring full profile bounds."""
from collections import OrderedDict
import numpy as np

from .family_score_bounds import (FamilyScoreBounds, _real_lower, _add_lo, _stored_lower,
                                 _add_hi, _mul_hi, _div_hi, _stored_upper)


class IndicatorBounds:
    """Only J and V are refined; other losses keep universal lower bounds.

    VR and PAR use -1 so their idle sentinels remain covered. Raw excess and
    severity use zero. Each returned profile is consequently a lower enclosure
    without computing upper loads, violating-time denominators or host peaks.
    """

    def __init__(self, full_bounds):
        self.full = full_bounds
        self._cache = OrderedDict()
        self._cache_cells = 0
        self._cache_limit = 1 << 20
        self.bound_queries = 0
        self.available = bool(full_bounds.available and self._finite_profile_domain())

    def _finite_profile_domain(self):
        """Certify every changed load and native positive reduction finite."""
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            changed = self.full.pair_load_upper
            if not np.isfinite(changed).all():
                return False
            peaks = changed.max(axis=1)
            peak_sum = _stored_upper(_add_hi(self.common_peak_hi, _mul_hi(peaks, 2.)), self.hosts)
            ratio = _div_hi(np.maximum(peaks, self.common_peak_max), self.capacity)
            return bool(np.isfinite(peak_sum).all() and np.isfinite(ratio).all()
                        and np.all(ratio <= np.finfo(float).max / (4. * self.hosts * self.horizon)))

    def __getattr__(self, name):
        return getattr(self.full, name)

    def lower_profiles(self, ranges):
        ranges = tuple(self.full._range(start, stop)[:2] for start, stop in ranges)
        if not self.available:
            return (None,) * len(ranges)
        local = {}
        for pair in ranges:
            if pair in self._cache:
                local[pair] = self._cache[pair]
                self._cache.move_to_end(pair)
        missing = tuple(dict.fromkeys(pair for pair in ranges if pair not in self._cache))
        if missing:
            w, q = len(self.mutable), len(missing)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                index = getattr(self.full, "mutable_sum_index", None)
                if index is None:
                    mandatory = np.zeros((q, 2, w))
                    for row, (start, stop) in enumerate(missing):
                        _, _, free = self.full._range(start, stop)
                        offset = int(self.symmetric)
                        if offset:
                            mandatory[row, 0, 0] = 1.
                        for bit in range(free, self.bits):
                            mandatory[row, (start >> bit) & 1, bit + offset] = 1.
                    sums = (mandatory.reshape(-1, w) @ self.mutable_flat).reshape(
                        q, 2, self.scenes, self.horizon)
                else:
                    sums = index.sum_ranges(missing, possible=False)
                lo = _stored_lower(_add_lo(_real_lower(sums, w), self.background_lo), self.resident_bound)
                individually = (lo > self.tolerant_capacity).any(axis=3).sum(axis=1)
                conserved = (self.pair_load_sum_lo > 2. * self.tolerant_capacity).any(axis=1)
                count = self.common_tolerant + np.maximum(individually, conserved)
                values = np.zeros((q, self.scenes, 6))
                values[:, :, 0] = count > 0
                values[:, :, 1] = count / float(self.hosts)
                values[:, :, 2] = -1.
                values[:, :, 5] = -1.
                finite = np.isfinite(sums).all(axis=(1, 2, 3)) & np.isfinite(lo).all(axis=(1, 2, 3))
            for pair, value, valid in zip(missing, values, finite):
                # Own each cached row, so evicting other rows releases the
                # batch allocation instead of retaining it through a view.
                row = np.array(value, copy=True, order="C") if valid else None
                local[pair] = row
                cells = 0 if row is None else row.size
                if cells <= self._cache_limit:
                    self._cache[pair] = row
                    self._cache_cells += cells
                while self._cache_cells > self._cache_limit or len(self._cache) > 512:
                    _, evicted = self._cache.popitem(last=False)
                    self._cache_cells -= 0 if evicted is None else evicted.size
            self.bound_queries += len(missing)
        return tuple(local[pair] for pair in ranges)

    def upper_keys(self, ranges, movement_lower_bounds=None, cost_lower_bounds=None):
        ranges = tuple(ranges)
        movement = (0,) * len(ranges) if movement_lower_bounds is None else tuple(movement_lower_bounds)
        costs = (0.,) * len(ranges) if cost_lower_bounds is None else tuple(cost_lower_bounds)
        if len(movement) != len(ranges) or len(costs) != len(ranges):
            raise ValueError("Each range needs a movement and cost lower bound")
        block = max(1, (1 << 20) // (2 * self.scenes * self.horizon))
        output = []
        for start in range(0, len(ranges), block):
            stop = min(start + block, len(ranges))
            output.extend(FamilyScoreBounds.upper_keys(self, ranges[start:stop],
                movement[start:stop], costs[start:stop]))
        return tuple(output)


def screen_indicators(certificates, score_bounds, masks, cutoff, observer=None, *, movement=None):
    """Return survivors and one valid witness for the active strict losers."""
    masks = tuple(masks)
    if cutoff is None or not masks or not score_bounds.available:
        return masks, ()
    movement = ([certificates.family.movement_count(mask) for mask in masks]
                if movement is None else movement)
    coarse = IndicatorBounds(score_bounds)
    if not coarse.available:
        return masks, ()
    keys = coarse.upper_keys([(mask, mask + 1) for mask in masks], movement, movement)
    certificates.stats["indicator_bound_nodes"] = certificates.stats.get("indicator_bound_nodes", 0) + len(masks)
    passing, rejected, rejected_keys = [], [], []
    for mask, key in zip(masks, keys):
        if key is not None and tuple(key) < tuple(cutoff):
            rejected.append(mask)
            rejected_keys.append(tuple(key))
        else:
            passing.append(mask)
    if not rejected:
        return tuple(passing), ()
    witness = certificates._record("score", min(rejected), max(rejected) + 1,
        rejected, observer, movement_lower_bound=0, optimistic_key=max(rejected_keys),
        cutoff=tuple(cutoff), bound_scope="indicator endpoint cascade over listed active leaves")
    certificates.stats["indicator_excluded_masks"] = certificates.stats.get("indicator_excluded_masks", 0) + len(rejected)
    return tuple(passing), (witness,)
