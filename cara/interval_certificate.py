"""Directed-rounding filter for the unchanged rational mixture decision.

Intervals enclose the same materialized partial mixture as certification.py.
An inconclusive interval is resolved by that module's exact integer reference.
The reader, observation schedule, component schedule and returned semantics are
unchanged; this module only supplies an alternative physical execution.
"""
from decimal import Context, Decimal, MAX_EMAX, MIN_EMIN, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction

import numpy as np

from .admission import _alpha
from .certification import (
    _DyadicMixture as _ExactDyadicMixture,
    _declared_threshold, _losses, _names, _rational,
)


class _IntervalDyadicMixture:
    """Positive-product enclosure with exact fallback at ambiguous comparisons."""

    def __init__(self, minimum_components=8):
        self.minimum_components = minimum_components
        self.rows = []
        self.high = (Decimal(1), Decimal(1))
        self.small = {}
        self.lower = Context(prec=50, rounding=ROUND_FLOOR, Emin=MIN_EMIN, Emax=MAX_EMAX)
        self.upper = Context(prec=50, rounding=ROUND_CEILING, Emin=MIN_EMIN, Emax=MAX_EMAX)
        self.exact_fallbacks = 0

    def _extend(self, product, pairs, numerator, denominator):
        lo, hi = product
        low, up = self.lower, self.upper
        for z, d in pairs:
            # Decimal(int) is exact and ignores context precision. Rounding is
            # applied only to division and the positive product operations.
            a, b = Decimal(denominator * d + numerator * z), Decimal(denominator * d)
            if a <= 0:
                raise ValueError("interval mixture requires positive betting factors")
            lo = low.multiply(lo, low.divide(a, b))
            hi = up.multiply(hi, up.divide(a, b))
        return lo, hi

    def update(self, values):
        pairs = [(value.numerator, value.denominator) for value in values]
        self.rows.extend(pairs)
        self.high = self._extend(self.high, pairs, 255, 256)
        for j, product in self.small.items():
            self.small[j] = self._extend(product, pairs, 1, 1 << j)
        count = max(self.minimum_components, len(self.rows).bit_length())
        for j in range(1, count + 1):
            if j not in self.small:
                self.small[j] = self._extend((Decimal(1), Decimal(1)), self.rows, 1, 1 << j)

    def bounds(self):
        low, up = self.lower, self.upper
        lo = low.divide(self.high[0], Decimal(2))
        hi = up.divide(self.high[1], Decimal(2))
        for j, (a, b) in self.small.items():
            denominator = Decimal(1 << (j + 1))
            lo = low.add(lo, low.divide(a, denominator))
            hi = up.add(hi, up.divide(b, denominator))
        return lo, hi

    def crossed(self, alpha):
        lo, hi = self.bounds()
        numerator, denominator = Decimal(alpha.numerator), Decimal(alpha.denominator)
        if self.lower.multiply(lo, numerator) >= denominator:
            return True
        if self.upper.multiply(hi, numerator) < denominator:
            return False
        reference = _ExactDyadicMixture(self.minimum_components)
        reference.update([Fraction(z, d) for z, d in self.rows])
        self.exact_fallbacks += 1
        return reference.crossed(alpha)


def certify_streaming_interval(reader, row_cap, dimension, *, gain_bounds,
                              noninferiority_margins, anchor=2, min_anchor_gain=0.005,
                              alpha=0.05, batch=32, metric_names=None,
                              minimum_components=8):
    """The validated streaming-mixture API with a certified interval filter.

The output dictionary is identical to certify_streaming_mixture, including
the non-strict exact crossing rule and locks at the same inspected batches.
"""
    if not callable(reader):
        raise ValueError("reader must return paired loss batches")
    if int(row_cap) != row_cap or row_cap < 1 or int(dimension) != dimension or dimension < 1:
        raise ValueError("row_cap and dimension must be positive integers")
    n, dimension = int(row_cap), int(dimension)
    names = _names(metric_names, dimension)
    if int(batch) != batch or batch < 1 or int(minimum_components) != minimum_components or minimum_components < 1:
        raise ValueError("batch and minimum_components must be positive integers")
    if int(anchor) != anchor or not 0 <= anchor < dimension:
        raise ValueError("anchor must identify one fixed metric")
    anchor, batch = int(anchor), int(batch)
    bounds = tuple(map(_rational, gain_bounds))
    margins = tuple(map(_declared_threshold, noninferiority_margins))
    eta, level = _declared_threshold(min_anchor_gain), _alpha(alpha)
    if len(bounds) != dimension or len(margins) != dimension:
        raise ValueError("bounds and margins must identify every metric")
    if any(value < 0 for value in bounds + margins) or eta < 0:
        raise ValueError("gain bounds, margins, and minimum anchor gain must be nonnegative")
    thresholds = tuple(eta if e == anchor else -margins[e] for e in range(dimension))
    scales = tuple(bounds[e] + abs(thresholds[e]) for e in range(dimension))
    processes = [_IntervalDyadicMixture(int(minimum_components)) for _ in range(dimension)]
    locked, crossing_rows = [False] * dimension, [None] * dimension
    for start in range(0, n, batch):
        indices = np.arange(start, min(start + batch, n))
        reference, observed = reader(indices)
        reference = _losses(reference, "reference batch")
        observed = _losses(observed, "candidate batch")
        if reference.shape != (len(indices), dimension) or observed.shape != reference.shape:
            raise ValueError("paired reader returned the wrong batch shape")
        values = [[] for _ in range(dimension)]
        for offset in range(len(indices)):
            for e in range(dimension):
                gain = Fraction(float(reference[offset, e])) - Fraction(float(observed[offset, e]))
                if abs(gain) > bounds[e]:
                    raise ValueError(f"observed gain for {names[e]} exceeds its predeclared bound")
                if not locked[e]:
                    values[e].append((gain - thresholds[e]) / scales[e] if scales[e] else Fraction())
        read = start + len(indices)
        for e in range(dimension):
            if not locked[e]:
                processes[e].update(values[e])
                if processes[e].crossed(level):
                    locked[e], crossing_rows[e] = True, read
        if all(locked):
            break
    admitted = all(locked)
    return {"admitted": admitted, "rows": read, "rows_read": read,
            "decision": "mixture" if admitted else "cap", "reason": "mixture" if admitted else "cap",
            "metric_names": list(names), "anchor": names[anchor],
            "noninferiority_margins": [float(value) for value in margins],
            "min_anchor_gain": float(eta), "gain_bounds": [float(value) for value in bounds],
            "exact_gain_bounds": [str(value) for value in bounds],
            "exact_thresholds": [str(value) for value in thresholds],
            "crossing_rows": crossing_rows, "locked": locked,
            "mixture_components": [1 + len(process.small) for process in processes],
            "claim_scope": "the specified fresh-scene law under predeclared gain bounds"}
