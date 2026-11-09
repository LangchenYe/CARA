"""Certificates for one frozen action in the revised CARA.

The legacy ``admission`` module remains unchanged for reproducible ablations.
This module separates an exact finite-history Pareto decision from a fresh-law
claim. Fresh gain bounds and tolerances must be fixed before certification and
must follow from the data-generating law or a physical contract, never a sample
maximum. Bounds checked on observed rows do not establish a population bound.
"""

from fractions import Fraction
from math import isfinite, nextafter

import numpy as np

from .admission import _alpha, _product_tree


def _rational(value):
    """Preserve a supplied Fraction, decimal string, or stored binary64 value."""
    if isinstance(value, (float, np.floating)):
        if not isfinite(float(value)):
            raise ValueError("finite rational values are required")
        return Fraction(float(value))
    return Fraction(value)


def _declared_threshold(value):
    """Interpret displayed decimal tolerances exactly, not one binary64 ulp away.

    Losses and outward-rounded gain bounds intentionally retain their exact
    binary64 values; human-declared tolerances use decimal configuration meaning.
    """
    if isinstance(value, (float, np.floating)):
        if not isfinite(float(value)):
            raise ValueError("finite declared thresholds are required")
        return Fraction(str(float(value)))
    return _rational(value)


def _losses(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError(f"{name} must be a nonempty scene-by-metric matrix")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains nonfinite losses")
    return values


def _names(metric_names, dimension):
    names = tuple(metric_names) if metric_names is not None else tuple(f"loss_{j}" for j in range(dimension))
    if len(names) != dimension or len(set(names)) != dimension:
        raise ValueError("metric_names must identify every metric exactly once")
    return names


def exact_pareto_gate(reference, candidate, *, metric_names=None):
    """Exact finite-table Pareto comparison, including uncapped/native losses.

    All metrics are losses: an action passes iff every original stored-value
    mean is no greater than the incumbent's and at least one is smaller. No
    probability, distributional extrapolation, capping, or tolerance is used.
    Computing all candidate losses is a full census; it is not a lazy row saving.
    """
    reference, candidate = _losses(reference, "reference"), _losses(candidate, "candidate")
    if reference.shape != candidate.shape:
        raise ValueError("paired loss matrices must have the same shape")
    names = _names(metric_names, reference.shape[1])
    totals = [sum((Fraction(float(r)) - Fraction(float(c))
                   for r, c in zip(reference[:, e], candidate[:, e])), Fraction())
              for e in range(reference.shape[1])]
    n = len(reference)
    means = [total / n for total in totals]
    admitted = all(value >= 0 for value in totals) and any(value > 0 for value in totals)
    return {"admitted": admitted, "rows": n, "rows_read": n,
            "decision": "exact_pareto_census", "reason": "exact_pareto_census",
            "metric_names": list(names), "gains": [float(value) for value in means],
            "exact_gains": [str(value) for value in means],
            "regressed_metrics": [name for name, value in zip(names, totals) if value < 0],
            "strict_metrics": [name for name, value in zip(names, totals) if value > 0],
            "claim_scope": "the supplied finite table only"}


class _DyadicMixture:
    """Computable lower bound on a countable, unit-mass betting mixture.

    One component has lambda=255/256 and weight=1/2. Components j>=1 have
    lambda=2**(-j), weight=2**(-j-1). All components conceptually start at time
    zero; newly materialized ones replay the observed prefix. Omitting positive
    components is conservative, including when the materialization count grows.
    """

    def __init__(self, minimum_components=8):
        self.minimum_components = minimum_components
        self.rows = []
        self.denominator = 1
        self.high = 1
        self.small = {}

    def update(self, values):
        pairs = [(value.numerator, value.denominator) for value in values]
        self.rows.extend(pairs)
        self.denominator *= _product_tree(d for _, d in pairs)
        self.high *= _product_tree(256 * d + 255 * z for z, d in pairs)
        for j in self.small:
            scale = 1 << j
            self.small[j] *= _product_tree(scale * d + z for z, d in pairs)
        count = max(self.minimum_components, len(self.rows).bit_length())
        for j in range(1, count + 1):
            if j not in self.small:
                scale = 1 << j
                self.small[j] = _product_tree(scale * d + z for z, d in self.rows)

    def parts(self):
        n, count = len(self.rows), len(self.small)
        exponent = max(8 * n + 1, count * n + count + 1)
        numerator = self.high << (exponent - 8 * n - 1)
        for j, value in self.small.items():
            numerator += value << (exponent - j * n - j - 1)
        return numerator, self.denominator << exponent

    def crossed(self, alpha):
        numerator, denominator = self.parts()
        return numerator * alpha.numerator >= denominator * alpha.denominator


def certify_fresh_mixture(reference, candidate, *, gain_bounds,
                          noninferiority_margins, anchor=2, min_anchor_gain=0.005,
                          alpha=0.05, batch=32, metric_names=None,
                          minimum_components=8):
    """Array-compatible interface to the paired streaming mixture certificate.

    A candidate callback receives requested indices. For generation and replay
    of both placements on demand, use ``certify_streaming_mixture`` directly.
    Bounds, targets, and the selected action must be fixed before inspection.
    """
    reference = _losses(reference, "reference")
    if callable(candidate):
        fetch = candidate
    else:
        candidate = _losses(candidate, "candidate")
        if candidate.shape != reference.shape:
            raise ValueError("paired loss matrices must have the same shape")
        fetch = lambda indices: candidate[indices]
    return certify_streaming_mixture(
        lambda indices: (reference[indices], fetch(indices)),
        len(reference), reference.shape[1], gain_bounds=gain_bounds,
        noninferiority_margins=noninferiority_margins, anchor=anchor,
        min_anchor_gain=min_anchor_gain, alpha=alpha, batch=batch,
        metric_names=metric_names, minimum_components=minimum_components)


def certify_streaming_mixture(reader, row_cap, dimension, *, gain_bounds,
                          noninferiority_margins, anchor=2, min_anchor_gain=0.005,
                          alpha=0.05, batch=32, metric_names=None,
                          minimum_components=8):
    """Certify one frozen action while acquiring only inspected paired batches.

    ``reader(indices)`` returns reference and candidate loss matrices for exactly
    those scene indices. Scene generation and paired replay can therefore stop
    with the statistical decision. The reader must expose one predeclared iid
    stream; batches and adaptive stopping must not change the scene law.
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
    processes = [_DyadicMixture(int(minimum_components)) for _ in range(dimension)]
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


def raw_migration_gain_bound(moved_count, hosts, capacity, demand_upper=1):
    """Mathematical |paired raw gain| bound after matching physical host labels.

    Every moved VM has demand at most demand_upper at every time, on every
    possible scene. Fixed host count and a common capacity are required.
    """
    if int(moved_count) != moved_count or moved_count < 0 or int(hosts) != hosts or hosts < 1:
        raise ValueError("use nonnegative moved_count and positive host count")
    capacity, demand_upper = _rational(capacity), _rational(demand_upper)
    if capacity <= 0 or demand_upper < 0:
        raise ValueError("positive capacity and nonnegative demand bound required")
    return int(moved_count) * demand_upper / (int(hosts) * capacity)


def stored_raw_gain_bound(moved_count, hosts, capacity, *, vm_count, horizon,
                          demand_upper=1):
    """Physical migration bound plus a conservative binary64 replay error.

    Covers arbitrary nonnegative summation trees, division, subtraction of one,
    positive-part clipping, aggregate summation, and final mean division. Assumes
    correctly rounded binary64 operations, finite intermediate results, and the
    raw replay in data.burdens_from_loads. The proof is in theory_derivations.tex.
    Returns an outward-rounded float suitable for ``gain_bounds``.
    """
    if int(vm_count) != vm_count or vm_count < 1 or int(horizon) != horizon or horizon < 1:
        raise ValueError("positive VM count and horizon required")
    if moved_count > vm_count:
        raise ValueError("moved_count cannot exceed vm_count")
    physical = raw_migration_gain_bound(moved_count, hosts, capacity, demand_upper)
    n, k = int(vm_count), int(hosts) * int(horizon)
    if k > 2**53:
        raise ValueError("the raw normalization integer must be exactly representable")
    capacity, demand_upper = _rational(capacity), _rational(demand_upper)
    unit, tiny = Fraction(1, 2**53), Fraction(1, 2**1075)

    def sum_error(count, operand_bound):
        growth = (1 + unit) ** max(0, count - 1)
        return (growth - 1) * count * operand_bound + tiny * max(0, count - 1) * growth

    load_error = sum_error(n, demand_upper)
    ratio_bound = n * demand_upper / capacity
    ratio_error = (1 + unit) * load_error / capacity + unit * ratio_bound + tiny
    excess_error = (1 + unit) * ratio_error + unit * (ratio_bound + 1) + tiny
    total_error = k * excess_error + sum_error(k, ratio_bound + excess_error)
    raw_error = (1 + unit) * total_error / k + unit * ratio_bound + tiny
    largest = Fraction(float(np.finfo(np.float64).max))
    if max(n * demand_upper + load_error, ratio_bound + ratio_error + 1,
           k * ratio_bound + total_error) > largest:
        raise ValueError("the raw arithmetic contract cannot exclude intermediate overflow")
    bound = physical + 2 * raw_error
    result = nextafter(float(bound), float("inf"))
    if not isfinite(result):
        raise ValueError("the stored-raw arithmetic contract is not finite")
    return result
